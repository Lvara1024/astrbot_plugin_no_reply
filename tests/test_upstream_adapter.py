"""Execute the official 4.29 send_streaming method with a recording transport.

Set ASTRBOT_429_ADAPTER to a downloaded official aiocqhttp_message_event.py.
Only the send_streaming method is executed; no QQ connection is made.
"""
import ast
import asyncio
from collections.abc import AsyncGenerator
import os
from pathlib import Path
import re
import unittest
from guard import EventGuard, Settings
from matcher import Matcher
from test_plugin import Chain, Plain, Event, plugin_module


class Base(Event):
    async def send_streaming(self, generator, use_fallback=False):
        pass

    async def process_buffer(self, buffer, pattern):
        while True:
            match = pattern.search(buffer)
            if not match:
                return buffer
            await self.send(Chain([Plain(match.group().strip())]))
            buffer = buffer[match.end():]


async def source(parts):
    for part in parts:
        yield Chain([Plain(part)])


@unittest.skipUnless(os.environ.get('ASTRBOT_429_ADAPTER'), 'Set ASTRBOT_429_ADAPTER for real upstream adapter tests')
class UpstreamAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tree = ast.parse(Path(os.environ['ASTRBOT_429_ADAPTER']).read_text(encoding='utf-8'))
        upstream = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AiocqhttpMessageEvent')
        method = next(n for n in upstream.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'send_streaming')
        cls = ast.ClassDef(name='AiocqhttpMessageEvent', bases=[ast.Name(id='Base', ctx=ast.Load())],
                          keywords=[], body=[method], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        namespace = dict(Base=Base, asyncio=asyncio, re=re, AsyncGenerator=AsyncGenerator, MessageChain=Chain, Plain=Plain)
        exec(compile(module, 'AstrBot-v4.29.0/send_streaming', 'exec'), namespace)
        self.Adapter = namespace['AiocqhttpMessageEvent']

    async def test_baseline_leak_and_guard_matrix(self):
        for fallback in [True, False]:
            baseline = self.Adapter()
            await baseline.send_streaming(source(['[NO', '_REPLY]']), fallback)
            self.assertTrue(baseline.sent)
            for mode, parts, blocked in [
                ('exact', ['[NO', '_REPLY]'], True),
                ('exact', ['你好。', '世界。'], False),
                ('exact', ['[NO_REPLY]', ' 是说明。'], False),
                ('contains', ['先说的话。', '[NO', '_REPLY]'], True),
                ('contains', ['你好。', '世界。'], False),
            ]:
                with self.subTest(fallback=fallback, mode=mode, parts=parts):
                    event = self.Adapter()
                    guard = EventGuard(event, Settings(Matcher(['[NO_REPLY]']), mode), plugin_module.inspect_chain, lambda *_: None)
                    await event.send_streaming(source(parts), fallback)
                    self.assertEqual(guard.blocked, blocked)
                    self.assertEqual(bool(event.sent), not blocked)
                    if not blocked:
                        self.assertEqual(''.join(c.text for chain in event.sent for c in chain.chain), ''.join(parts))
