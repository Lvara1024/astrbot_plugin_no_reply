"""Run pinned official platform methods with recording network/UI boundaries."""
import ast
import asyncio
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from guard import EventGuard, Settings
from matcher import Matcher
from test_plugin import Chain, Plain, Event, plugin_module


def load_adapter(path, name, base, **namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    upstream = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    method = next(n for n in upstream.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'send_streaming')
    cls = ast.ClassDef(name=name, bases=[ast.Name(id='Base', ctx=ast.Load())],
                      keywords=[], body=[method], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace.update(Base=base, asyncio=asyncio, MessageChain=Chain, Plain=Plain)
    exec(compile(module, 'AstrBot-v4.29.0/' + name, 'exec'), namespace)
    return namespace[name]


async def source(parts):
    for part in parts:
        yield part if isinstance(part, Chain) else Chain([Plain(part)])
        await asyncio.sleep(0)


def protect(event, mode='exact'):
    return EventGuard(event, Settings(Matcher(['[NO_REPLY]']), mode),
                      plugin_module.inspect_chain, lambda *_: None)


class Metric:
    @staticmethod
    async def upload(**kwargs):
        pass


class LarkBase(Event):
    def __init__(self):
        super().__init__()
        self.platform_meta = SimpleNamespace(name='lark')
        self.message_obj = SimpleNamespace(message_id='message-1')
        self.operations = []
        self.cards = 0
        self.fail_create = False
        self.fail_send = False

    async def _create_streaming_card(self):
        self.cards += 1
        self.operations.append(('create', self.cards))
        return None if self.fail_create else 'card-' + str(self.cards)

    async def _send_card_message(self, card_id, **kwargs):
        self.operations.append(('send', card_id))
        return not self.fail_send

    async def _update_streaming_text(self, card_id, text, sequence):
        self.operations.append(('update', card_id, text, sequence))
        return True

    async def _close_streaming_mode(self, card_id, sequence):
        self.operations.append(('close', card_id, sequence))


@unittest.skipUnless(os.environ.get('ASTRBOT_429_LARK'), 'Set ASTRBOT_429_LARK for upstream Lark tests')
class LarkTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.Adapter = load_adapter(os.environ['ASTRBOT_429_LARK'], 'LarkMessageEvent', LarkBase,
                                    Metric=Metric, logger=logging.getLogger('LarkTest'))

    async def test_silence_creates_no_card_in_both_modes(self):
        for mode in ['exact', 'contains']:
            event = self.Adapter()
            guard = protect(event, mode)
            await event.send_streaming(source(['[NO', '_REPLY]']))
            await event.send(Chain([Plain('late append')]))
            self.assertTrue(guard.blocked)
            self.assertEqual(event.operations, [])
            self.assertEqual(event.sent, [])

    async def test_normal_stream_updates_and_closes_card(self):
        event = self.Adapter()
        protect(event)
        await event.send_streaming(source(['你好', '，世界']))
        updates = [op for op in event.operations if op[0] == 'update']
        self.assertEqual(updates[-1][2], '你好，世界')
        self.assertEqual(event.operations[-1][0], 'close')
        self.assertEqual(event.cards, 1)

    async def test_tool_break_preserves_card_boundaries(self):
        event = self.Adapter()
        protect(event)
        await event.send_streaming(source(['前段', Chain(type='break'), '后段']))
        updates = {op[1]: op[2] for op in event.operations if op[0] == 'update'}
        self.assertEqual(updates, {'card-1': '前段', 'card-2': '后段'})
        self.assertEqual(sum(op[0] == 'close' for op in event.operations), 2)

    async def test_contains_overflow_keeps_long_reply(self):
        event = self.Adapter()
        guard = protect(event, 'contains')
        parts = ['x' * 33, '[NO_REPLY]', 'tail']
        await event.send_streaming(source(parts))
        updates = [op for op in event.operations if op[0] == 'update']
        self.assertEqual(updates[-1][2], ''.join(parts))
        self.assertTrue(guard.limit_reached)
        self.assertFalse(guard.blocked)

    async def test_failed_card_uses_original_plain_fallback(self):
        for failure in ['fail_create', 'fail_send']:
            event = self.Adapter()
            setattr(event, failure, True)
            protect(event)
            with self.assertLogs('LarkTest', level='WARNING'):
                await event.send_streaming(source(['正常', '回复']))
            self.assertEqual(''.join(c.text for m in event.sent for c in m.chain), '正常回复')


class WebBase(Event):
    def __init__(self):
        super().__init__()
        self.message_obj = SimpleNamespace(message_id='web-1')
        self.session_id = 'session-1'

    def get_platform_name(self):
        return 'webchat'

    async def send_streaming(self, generator, use_fallback=False):
        pass


@unittest.skipUnless(os.environ.get('ASTRBOT_429_WEBCHAT'), 'Set ASTRBOT_429_WEBCHAT for upstream WebChat tests')
class WebChatTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.frames, self.messages = [], []

        async def put_back_queue(request_id, payload):
            self.frames.append(payload)
            return True

        async def send(**kwargs):
            text = ''.join(c.text for c in kwargs['message'].chain if isinstance(c, Plain))
            self.messages.append(text)
            return text

        self.Adapter = load_adapter(os.environ['ASTRBOT_429_WEBCHAT'], 'WebChatMessageEvent', WebBase,
                                    webchat_queue_mgr=SimpleNamespace(put_back_queue=put_back_queue))
        self.Adapter._send = staticmethod(send)

    async def test_silence_only_emits_empty_completion_frame(self):
        event = self.Adapter()
        protect(event)
        await event.send_streaming(source(['[NO', '_REPLY]']))
        self.assertEqual(self.messages, [])
        self.assertEqual(len(self.frames), 1)
        self.assertEqual(self.frames[0]['type'], 'complete')
        self.assertEqual(self.frames[0]['data'], '')

    async def test_normal_stream_and_completion_are_preserved(self):
        event = self.Adapter()
        protect(event)
        await event.send_streaming(source(['正常', '回复']))
        self.assertEqual(self.messages, ['正常', '回复'])
        self.assertEqual(self.frames[0]['data'], '正常回复')
