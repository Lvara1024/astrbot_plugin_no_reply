"""AstrBot entry point for no-reply markers."""
import json
import weakref

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain, Reply
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

from .guard import EventGuard, Settings
from .matcher import Matcher

KEY = "astrbot_plugin_no_reply.guard"
PROMPT_TAG = "[no_reply policy]"
DEFAULT_INSTRUCTION = (
    "当你判断当前消息无需回复（例如对话已经结束或没有需要回应的内容）时，"
    "仅输出 {primary}，不要添加解释、标点、引号或 Markdown。"
    "允许的沉默标记：{markers}。需要回复时正常回答，不要附加沉默标记。"
)


def inspect_chain(chain):
    if chain is None:
        return "", False
    parts = getattr(chain, "chain", chain if isinstance(chain, list) else []) or []
    return (
        "".join(part.text for part in parts if isinstance(part, Plain)),
        any(not isinstance(part, (Plain, At, Reply)) for part in parts),
    )


class NoReplyPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.guards = weakref.WeakSet()
        self._cache_key = None
        self._settings = None
        self._get_settings()

    def _get_settings(self):
        words = self.config.get("markers", ["[NO_REPLY]"])
        case = bool(self.config.get("ignore_case", False))
        mode = self.config.get("match_mode", "exact")
        if mode not in ("exact", "contains"):
            raise ValueError("match_mode 必须为 exact 或 contains")
        chars = int(self.config.get("max_buffer_chars", 262144))
        chunks = int(self.config.get("max_buffer_chunks", 8192))
        if chars < 1 or chunks < 1:
            raise ValueError("缓存上限必须大于 0")
        key = json.dumps([words, case, mode, chars, chunks, bool(self.config.get("suppress_followups", True))], ensure_ascii=False)
        if key != self._cache_key:
            self._settings = Settings(Matcher(words, case), mode,
                                      bool(self.config.get("suppress_followups", True)), chars, chunks)
            self._cache_key = key
        return self._settings

    def _blocked(self, guard, reason):
        # Do not stop the model in the middle of its generator. Drain it so its
        # normal finalization/hooks run; event.send is already guarded locally.
        logger.info("[NoReply] 已拦截当前回复，原因=%s，模式=%s", reason, guard.settings.mode)

    def _guard(self, event):
        guard = event.get_extra(KEY)
        if guard is None or not guard.enabled:
            guard = EventGuard(event, self._get_settings(), inspect_chain, self._blocked)
            event.set_extra(KEY, guard)
            self.guards.add(guard)
        return guard

    @filter.on_llm_request(priority=10000)
    async def prepare(self, event: AstrMessageEvent, req: ProviderRequest):
        guard = self._guard(event)
        if self.config.get("inject_prompt", True) and PROMPT_TAG not in (req.system_prompt or ""):
            instruction = self.config.get("system_instruction", DEFAULT_INSTRUCTION) or DEFAULT_INSTRUCTION
            # Literal replacement leaves unrelated braces in custom prompts intact.
            instruction = instruction.replace("{primary}", guard.settings.matcher.words[0]).replace(
                "{markers}", json.dumps(guard.settings.matcher.words, ensure_ascii=False))
            req.system_prompt = (req.system_prompt or "") + "\n\n" + PROMPT_TAG + "\n" + instruction

    @filter.on_llm_response(priority=10000)
    async def check_response(self, event: AstrMessageEvent, resp: LLMResponse):
        guard = event.get_extra(KEY)
        if guard is None:
            return
        # In streaming mode the outbound gate sees the actual visible chunks,
        # including media and intermediate text. The final LLMResponse alone
        # can differ, so it must not override that decision.
        if guard.stream_active:
            return
        text = resp.completion_text or ""
        chain_text, media = inspect_chain(getattr(resp, "result_chain", None))
        matched = guard.settings.matcher.matches(text or chain_text, guard.settings.mode, media)
        if matched:
            guard.block()
        if matched or (guard.blocked and guard.settings.suppress_followups):
            resp.completion_text = ""
            if hasattr(resp, "result_chain"):
                resp.result_chain = None
            if guard.settings.suppress_followups:
                event.stop_event()

    @filter.on_decorating_result(priority=-10000)
    async def final_check(self, event: AstrMessageEvent):
        guard = event.get_extra(KEY)
        if guard is None:
            return
        result = event.get_result()
        if guard.blocked:
            if guard.settings.suppress_followups:
                event.clear_result()
                event.stop_event()
            return
        if result is not None:
            if guard.stream_seen:
                return
            text, media = inspect_chain(result)
            if guard.settings.matcher.matches(text, guard.settings.mode, media):
                guard.block()
                event.clear_result()
                if guard.settings.suppress_followups:
                    event.stop_event()

    async def terminate(self):
        for guard in list(self.guards):
            guard.restore()
        self.guards.clear()
