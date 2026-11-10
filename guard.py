"""Per-event outbound guard. Adapter integration is supplied via callbacks."""
from dataclasses import dataclass
import copy


@dataclass(frozen=True)
class Settings:
    matcher: object
    mode: str = "exact"
    suppress_followups: bool = True
    max_buffer_chars: int = 262144
    max_buffer_chunks: int = 8192


class EventGuard:
    def __init__(self, event, settings, inspect_chain, on_block):
        self.event, self.settings = event, settings
        self.inspect_chain, self.on_block = inspect_chain, on_block
        self.original_send = event.send
        self.original_stream = event.send_streaming
        self.blocked = False
        self.reason = None
        self.enabled = True
        self.stream_active = False
        self.stream_seen = False
        # Only this event instance is patched, never the class/platform/global API.
        async def send(message, *args, **kwargs):
            if self.enabled and self.blocked and self.settings.suppress_followups:
                return None
            return await self.original_send(message, *args, **kwargs)

        async def stream(generator, *args, **kwargs):
            if not self.enabled:
                return await self.original_stream(generator, *args, **kwargs)
            filtered = self.filter_stream(generator)
            self.stream_seen = True
            self.stream_active = True
            try:
                return await self.original_stream(filtered, *args, **kwargs)
            finally:
                self.stream_active = False
                await filtered.aclose()
                if self.blocked and self.settings.suppress_followups:
                    self.event.stop_event()

        self.send_wrapper, self.stream_wrapper = send, stream
        event.send, event.send_streaming = send, stream

    def block(self, reason="marker"):
        if not self.blocked:
            self.blocked, self.reason = True, reason
            self.on_block(self, reason)

    def restore(self):
        self.enabled = False
        if self.event.send is self.send_wrapper:
            self.event.send = self.original_send
        if self.event.send_streaming is self.stream_wrapper:
            self.event.send_streaming = self.original_stream

    async def filter_stream(self, source):
        cfg = self.settings
        cursor = cfg.matcher.cursor(cfg.mode)
        pending, size = [], 0
        passthrough = False
        rejected = self.blocked and cfg.suppress_followups
        try:
            async for chunk in source:
                if chunk is None:
                    continue
                if rejected or (self.enabled and self.blocked and cfg.suppress_followups):
                    pending.clear()
                    continue
                if passthrough or not self.enabled:
                    if pending:
                        for item in pending:
                            yield item
                        pending.clear()
                    yield chunk
                    continue
                text, media = self.inspect_chain(chunk)
                cursor.feed(text)
                if cfg.mode == "contains" and cursor.matches:
                    rejected = True
                    pending.clear()
                    self.block()
                    continue
                if cfg.mode == "exact" and (media or not cursor.possible):
                    passthrough = True
                    for item in pending:
                        yield item
                    pending.clear()
                    yield chunk
                    continue
                size += len(text)
                if size > cfg.max_buffer_chars or len(pending) >= cfg.max_buffer_chunks:
                    rejected = True
                    pending.clear()
                    self.block("buffer_limit")
                    continue
                # Providers may reuse/mutate a chain object after yielding it.
                pending.append(copy.deepcopy(chunk))
            if rejected or (self.enabled and self.blocked and cfg.suppress_followups):
                return
            if self.enabled and not passthrough and cursor.matches:
                self.block()
                return
            for item in pending:
                yield item
        finally:
            pending.clear()
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()
