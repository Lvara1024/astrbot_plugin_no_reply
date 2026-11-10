import asyncio
import unittest
from dataclasses import dataclass, field
from matcher import Matcher
from guard import EventGuard, Settings


@dataclass
class Chunk:
    text: str = ''
    media: bool = False
    metadata: dict = field(default_factory=dict)


def inspect(chunk):
    return chunk.text, chunk.media


class Event:
    def __init__(self):
        self.sent = []
        self.stopped = False

    def stop_event(self):
        self.stopped = True

    async def send(self, message):
        self.sent.append(message)

    async def send_streaming(self, source, use_fallback=True):
        # Model an adapter that emits each chunk immediately. This catches even
        # leaks that a QQ adapter would temporarily hide in its own buffer.
        async for chunk in source:
            await self.send(chunk)


async def source(*texts):
    for text in texts:
        yield Chunk(text)


def guarded(mode='exact', **kwargs):
    event = Event()
    guard = EventGuard(event, Settings(Matcher(['[NO_REPLY]', '不回复']), mode, **kwargs), inspect, lambda *_: None)
    return event, guard


class GuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_sentinel_never_reaches_adapter(self):
        event, guard = guarded()
        await event.send_streaming(source(' \n', '[NO', '_REP', 'LY]', ' \t'))
        self.assertEqual(event.sent, [])
        self.assertTrue(guard.blocked)
        self.assertTrue(event.stopped)
        await event.send(Chunk('later plugin appended text'))
        await event.send_streaming(source('another stream'))
        self.assertEqual(event.sent, [])

    async def test_normal_is_delivered_before_model_finishes(self):
        event, guard = guarded()
        async def generate():
            yield Chunk('你好')
            self.assertEqual([c.text for c in event.sent], ['你好'])
            yield Chunk('，世界')
        await event.send_streaming(generate())
        self.assertEqual(''.join(c.text for c in event.sent), '你好，世界')
        self.assertFalse(guard.blocked)

    async def test_exact_marker_mentioned_in_normal_text_is_preserved(self):
        event, guard = guarded()
        await event.send_streaming(source('[NO', '_REPLY]', ' 是一个标记。'))
        self.assertEqual(''.join(c.text for c in event.sent), '[NO_REPLY] 是一个标记。')
        self.assertFalse(guard.blocked)

    async def test_contains_marker_inside_window_discards_entire_reply(self):
        event, guard = guarded('contains')
        await event.send_streaming(source('很长的正常内容。', '末尾[NO', '_REPLY]', '后续内容'))
        self.assertEqual(event.sent, [])
        self.assertTrue(guard.blocked)

    async def test_contains_without_marker_waits_and_preserves_order(self):
        event, guard = guarded('contains')
        async def generate():
            yield Chunk('第一段。', metadata={'key': 1})
            self.assertFalse(event.sent)
            yield Chunk('第二段。')
            self.assertFalse(event.sent)
        await event.send_streaming(generate())
        self.assertEqual([c.text for c in event.sent], ['第一段。', '第二段。'])
        self.assertEqual(event.sent[0].metadata, {'key': 1})

    async def test_followup_option_off_allows_later_sends(self):
        event, guard = guarded(suppress_followups=False)
        await event.send_streaming(source('[NO_REPLY]'))
        self.assertFalse(event.sent)
        self.assertFalse(event.stopped)
        await event.send(Chunk('allowed later'))
        self.assertEqual(event.sent[0].text, 'allowed later')

    async def test_limits_release_normal_content(self):
        for kwargs, parts in [({'max_buffer_chars': 3}, ['four']), ({'max_buffer_chunks': 1}, ['', ''])]:
            event, guard = guarded('contains', **kwargs)
            await event.send_streaming(source(*parts))
            self.assertEqual(''.join(c.text for c in event.sent), ''.join(parts))
            self.assertTrue(guard.limit_reached)
            self.assertFalse(guard.blocked)

    async def test_contains_only_scans_character_window_regardless_of_chunking(self):
        text = 'x' * 32 + '[NO_REPLY]'
        for parts in [[text], list(text), [text[:32], text[32:]]]:
            event, guard = guarded('contains')
            await event.send_streaming(source(*parts))
            self.assertEqual(''.join(c.text for c in event.sent), text)
            self.assertFalse(guard.blocked)

    async def test_contains_marker_finishing_at_window_boundary(self):
        event, guard = guarded('contains')
        await event.send_streaming(source('x' * 22 + '[NO_REPLY]' + 'trailing text'))
        self.assertFalse(event.sent)
        self.assertTrue(guard.blocked)

    async def test_chunk_limit_is_not_rescanned(self):
        event, guard = guarded('contains', max_buffer_chunks=2)
        await event.send_streaming(source('', '', '[NO_REPLY]'))
        self.assertEqual(''.join(c.text for c in event.sent), '[NO_REPLY]')
        self.assertFalse(guard.blocked)

    async def test_silence_does_not_start_platform_placeholder(self):
        event = Event()
        started = []
        original = event.send_streaming
        async def placeholder_adapter(generator, **kwargs):
            started.append(True)
            await original(generator, **kwargs)
        event.send_streaming = placeholder_adapter
        EventGuard(event, Settings(Matcher(['[NO_REPLY]'])), inspect, lambda *_: None)
        await event.send_streaming(source('[NO_REPLY]'))
        self.assertFalse(started)

    async def test_webchat_gets_empty_completion(self):
        event = Event()
        completed = []
        event.get_platform_name = lambda: 'webchat'
        original = event.send_streaming
        async def adapter(generator):
            await original(generator)
            completed.append(True)
        event.send_streaming = adapter
        EventGuard(event, Settings(Matcher(['[NO_REPLY]'])), inspect, lambda *_: None)
        await event.send_streaming(source('[NO_REPLY]'))
        self.assertTrue(completed)
        self.assertFalse(event.sent)

    async def test_media_is_not_only_a_marker(self):
        event, guard = guarded()
        async def generate():
            yield Chunk('[NO_REPLY]')
            yield Chunk(media=True)
        await event.send_streaming(generate())
        self.assertEqual(len(event.sent), 2)
        self.assertFalse(guard.blocked)

    async def test_cancellation_closes_source_without_flushing_unchecked_text(self):
        event, guard = guarded('contains')
        closed = []
        async def generate():
            try:
                yield Chunk('pending')
                raise asyncio.CancelledError()
            finally:
                closed.append(True)
        with self.assertRaises(asyncio.CancelledError):
            await event.send_streaming(generate())
        self.assertTrue(closed)
        self.assertFalse(event.sent)

    async def test_source_exception_does_not_flush_unchecked_reply(self):
        event, guard = guarded('contains')
        async def generate():
            yield Chunk('pending')
            raise RuntimeError('upstream failed')
        with self.assertRaises(RuntimeError):
            await event.send_streaming(generate())
        self.assertFalse(event.sent)

    async def test_concurrent_events_do_not_share_state(self):
        one, first = guarded()
        two, second = guarded()
        await asyncio.gather(one.send_streaming(source('[NO_REPLY]')), two.send_streaming(source('正常')))
        self.assertFalse(one.sent)
        self.assertEqual(two.sent[0].text, '正常')

    async def test_unload_restores_instance_methods(self):
        event, guard = guarded()
        guard.restore()
        await event.send_streaming(source('[NO_REPLY]'))
        self.assertEqual(event.sent[0].text, '[NO_REPLY]')

    async def test_buffer_copies_reused_chunk(self):
        event, guard = guarded('contains')
        async def generate():
            chunk = Chunk('甲')
            yield chunk
            chunk.text = '乙'
            yield chunk
        await event.send_streaming(generate())
        self.assertEqual([c.text for c in event.sent], ['甲', '乙'])


if __name__ == '__main__':
    unittest.main()
