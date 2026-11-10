import importlib.util
import json
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class Plain:
    def __init__(self, text):
        self.text = text


class At:
    pass


class Reply:
    pass


class Image:
    pass


class Chain:
    def __init__(self, chain=None, **kwargs):
        self.chain = chain or []
        self.type = kwargs.get('type')

    def message(self, text):
        self.chain.append(Plain(text))
        return self

    def squash_plain(self):
        text = ''.join(c.text for c in self.chain if isinstance(c, Plain))
        self.chain = [Plain(text)]


class Event:
    def __init__(self):
        self.extras, self.sent = {}, []
        self.stopped, self.result = False, None

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_result(self):
        return self.result

    def clear_result(self):
        self.result = None

    def stop_event(self):
        self.stopped = True

    async def send(self, chain):
        self.sent.append(chain)

    async def send_streaming(self, source, use_fallback=True):
        async for chain in source:
            await self.send(chain)


def load_plugin():
    modules = {name: types.ModuleType(name) for name in [
        'astrbot', 'astrbot.api', 'astrbot.api.event', 'astrbot.api.message_components',
        'astrbot.api.provider', 'astrbot.api.star']}
    modules['astrbot.api'].AstrBotConfig = dict
    modules['astrbot.api'].logger = logging.getLogger('NoReplyTest')
    modules['astrbot.api.event'].AstrMessageEvent = Event
    def decorator(**kwargs):
        return lambda func: func
    modules['astrbot.api.event'].filter = types.SimpleNamespace(
        on_llm_request=decorator, on_llm_response=decorator, on_decorating_result=decorator)
    comps = modules['astrbot.api.message_components']
    comps.At, comps.Plain, comps.Reply = At, Plain, Reply
    modules['astrbot.api.provider'].LLMResponse = types.SimpleNamespace
    modules['astrbot.api.provider'].ProviderRequest = types.SimpleNamespace
    class Star:
        def __init__(self, context):
            self.context = context
    modules['astrbot.api.star'].Star = Star
    modules['astrbot.api.star'].Context = object
    root = Path(__file__).resolve().parents[1]
    package_name = '_no_reply_test_package'
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    modules[package_name] = package
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(package_name + '.main', root / 'main.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


plugin_module = load_plugin()


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_prompt_and_duplicate_preparation(self):
        plugin = plugin_module.NoReplyPlugin(None, {})
        event = Event()
        req = types.SimpleNamespace(system_prompt='角色')
        await plugin.prepare(event, req)
        guard = event.get_extra(plugin_module.KEY)
        await plugin.prepare(event, req)
        self.assertIs(event.get_extra(plugin_module.KEY), guard)
        self.assertEqual(req.system_prompt.count(plugin_module.PROMPT_TAG), 1)
        self.assertIn('[NO_REPLY]', req.system_prompt)
        await plugin.terminate()
        self.assertFalse(guard.enabled)

    async def test_custom_multiple_markers_and_prompt_off(self):
        plugin = plugin_module.NoReplyPlugin(None, {'markers': ['不回复', '[SILENT]'], 'inject_prompt': False})
        event = Event()
        req = types.SimpleNamespace(system_prompt='原始')
        await plugin.prepare(event, req)
        self.assertEqual(req.system_prompt, '原始')
        resp = types.SimpleNamespace(completion_text=' [SILENT] ', result_chain=None)
        await plugin.check_response(event, resp)
        self.assertTrue(event.stopped)
        self.assertEqual(resp.completion_text, '')
        await event.send(Chain([Plain('追加')]))
        self.assertFalse(event.sent)

    async def test_nonstream_contains_and_late_append(self):
        plugin = plugin_module.NoReplyPlugin(None, {'match_mode': 'contains'})
        event = Event()
        await plugin.prepare(event, types.SimpleNamespace(system_prompt=''))
        event.result = Chain([Plain('正常'), Plain('[NO_REPLY]')])
        await plugin.final_check(event)
        self.assertTrue(event.stopped)
        self.assertIsNone(event.result)

    async def test_streaming_hook_does_not_decide_using_final_text_alone(self):
        plugin = plugin_module.NoReplyPlugin(None, {})
        event = Event()
        await plugin.prepare(event, types.SimpleNamespace(system_prompt=''))
        async def source():
            yield Chain([Plain('[NO_REPLY]')])
            yield Chain([Image()])
            await plugin.check_response(event, types.SimpleNamespace(completion_text='[NO_REPLY]', result_chain=None))
        await event.send_streaming(source())
        event.result = Chain([Plain('[NO_REPLY]')])
        await plugin.final_check(event)
        self.assertFalse(event.stopped)
        self.assertEqual(len(event.sent), 2)

    async def test_streaming_exact_with_hook_and_followup(self):
        plugin = plugin_module.NoReplyPlugin(None, {})
        event = Event()
        await plugin.prepare(event, types.SimpleNamespace(system_prompt=''))
        async def source():
            yield Chain([Plain('[NO')])
            yield Chain([Plain('_REPLY]')])
            await plugin.check_response(event, types.SimpleNamespace(completion_text='[NO_REPLY]', result_chain=None))
            # A lower-priority response hook attempts a direct send before EOF.
            await event.send(Chain([Plain('hook appended before generator EOF')]))
        await event.send_streaming(source())
        await event.send(Chain([Plain('later')]))
        self.assertTrue(event.stopped)
        self.assertFalse(event.sent)

    async def test_settings_snapshot_and_rebuild(self):
        config = {'markers': ['[NO_REPLY]']}
        plugin = plugin_module.NoReplyPlugin(None, config)
        first, second = Event(), Event()
        await plugin.prepare(first, types.SimpleNamespace(system_prompt=''))
        config['markers'] = ['NEW']
        await plugin.prepare(second, types.SimpleNamespace(system_prompt=''))
        self.assertEqual(first.get_extra(plugin_module.KEY).settings.matcher.words, ('[NO_REPLY]',))
        self.assertEqual(second.get_extra(plugin_module.KEY).settings.matcher.words, ('NEW',))

    def test_schema_defaults_match_runtime(self):
        schema = json.loads((Path(__file__).resolve().parents[1] / '_conf_schema.json').read_text(encoding='utf-8'))
        self.assertEqual(schema['markers']['default'], ['[NO_REPLY]'])
        self.assertEqual(schema['system_instruction']['default'], plugin_module.DEFAULT_INSTRUCTION)
        self.assertEqual(schema['max_buffer_chars']['default'], 32)
        self.assertEqual(schema['max_buffer_chunks']['default'], 32)

    async def test_late_response_hook_does_not_truncate_released_stream(self):
        plugin = plugin_module.NoReplyPlugin(None, {'match_mode': 'contains'})
        event = Event()
        await plugin.prepare(event, types.SimpleNamespace(system_prompt=''))
        full = 'x' * 33 + '[NO_REPLY]'
        resp = types.SimpleNamespace(completion_text=full, result_chain=None)
        async def source():
            yield Chain([Plain(full)])
            await plugin.check_response(event, resp)
            yield Chain([Plain('tail')])
        await event.send_streaming(source())
        self.assertEqual(''.join(c.text for chain in event.sent for c in chain.chain), full + 'tail')
        self.assertFalse(event.stopped)

    async def test_nonstream_only_scans_short_window_on_any_platform(self):
        plugin = plugin_module.NoReplyPlugin(None, {'match_mode': 'contains'})
        for platform in ['aiocqhttp', 'lark', 'telegram', 'discord', 'slack', 'wecom', 'dingtalk', 'custom_adapter']:
            event = Event()
            event.get_platform_name = lambda: platform
            await plugin.prepare(event, types.SimpleNamespace(system_prompt=''))
            resp = types.SimpleNamespace(completion_text='x' * 32 + '[NO_REPLY]', result_chain=None)
            await plugin.check_response(event, resp)
            self.assertFalse(event.stopped)
            resp.completion_text = '[NO_REPLY]'
            await plugin.check_response(event, resp)
            self.assertTrue(event.stopped)
