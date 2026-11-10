"""Incremental literal matching; no AstrBot or third-party dependencies."""
from collections import deque


class Matcher:
    def __init__(self, words, ignore_case=False):
        if not isinstance(words, (list, tuple)):
            raise ValueError("检测词必须是字符串列表")
        self.ignore_case = ignore_case
        self.words = tuple(dict.fromkeys(w.strip() for w in words if isinstance(w, str) and w.strip()))
        if not self.words:
            raise ValueError("至少配置一个非空检测词")
        if len(self.words) > 4096 or sum(map(len, self.words)) > 131072:
            raise ValueError("检测词最多 4096 个，总长度最多 131072 字符")
        self.edges, self.fail, self.terminal, self.output = [{}], [0], [False], [False]
        for word in self.words:
            node = 0
            for char in self.normalize(word):
                if char not in self.edges[node]:
                    self.edges[node][char] = len(self.edges)
                    self.edges.append({})
                    self.fail.append(0)
                    self.terminal.append(False)
                    self.output.append(False)
                node = self.edges[node][char]
            self.terminal[node] = self.output[node] = True
        queue = deque(self.edges[0].values())
        while queue:
            node = queue.popleft()
            for char, child in self.edges[node].items():
                queue.append(child)
                fallback = self.fail[node]
                while fallback and char not in self.edges[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.edges[fallback].get(char, 0)
                self.output[child] |= self.output[self.fail[child]]

    def normalize(self, text):
        return text.casefold() if self.ignore_case else text

    def cursor(self, mode):
        return Cursor(self, mode)

    def matches(self, text, mode, has_media=False):
        cursor = self.cursor(mode)
        cursor.feed(text)
        if mode == "exact" and has_media:
            return False
        return cursor.matches


class Cursor:
    def __init__(self, matcher, mode):
        if mode not in ("exact", "contains"):
            raise ValueError("match_mode 必须为 exact 或 contains")
        self.matcher, self.mode = matcher, mode
        self.node = 0
        self.started = False
        self.trailing = False
        self.hit = False
        self.possible = True

    def feed(self, text):
        m = self.matcher
        if self.mode == "contains":
            if self.hit:
                return
            for char in m.normalize(text):
                while self.node and char not in m.edges[self.node]:
                    self.node = m.fail[self.node]
                self.node = m.edges[self.node].get(char, 0)
                if m.output[self.node]:
                    self.hit = True
                    return
            return
        if not self.possible:
            return
        for char in m.normalize(text):
            if not self.started and char.isspace():
                continue
            self.started = True
            terminal = self.node is not None and m.terminal[self.node]
            self.trailing = char.isspace() and (self.trailing or terminal)
            self.node = m.edges[self.node].get(char) if self.node is not None else None
            self.possible = self.node is not None or self.trailing
            if not self.possible:
                return

    @property
    def matches(self):
        if self.mode == "contains":
            return self.hit
        return self.started and (self.trailing or (self.node is not None and self.matcher.terminal[self.node]))
