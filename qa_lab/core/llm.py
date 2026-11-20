"""LLM adapters. Ollama (local) is the verified default; OpenAI requires a valid key."""
from __future__ import annotations
import json, os, re, time, httpx
from typing import AsyncIterator, Protocol

class LLM(Protocol):
    name: str
    local: bool
    async def json(self, system: str, user: str, max_tokens: int = 300) -> dict: ...
    async def stream(self, system: str, user: str, max_tokens: int = 120) -> AsyncIterator[str]: ...

def parse_json(txt: str) -> dict:
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
    try: return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return {}

class OllamaLLM:
    local = True
    def __init__(self, model="qwen2.5:7b", url=None, keep_alive="30m", tag="ollama"):
        self.model = model; self.name = f"{tag}:{model}"; self.think = "low" if model.startswith("gpt-oss") else False
        self.url = url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.keep_alive = keep_alive; self.num_ctx = int(os.environ.get("QA_NUM_CTX", 4096))
        self.c = httpx.AsyncClient(timeout=900 if tag == "ollamax" else 180)      # first load of a big model from the external USB drive can take minutes
    def _body(self, system, user, max_tokens, stream, fmt=None):
        b = {"model": self.model, "stream": stream, "keep_alive": self.keep_alive,
             "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
             "options": {"temperature": 0.3, "num_predict": max_tokens, "num_ctx": self.num_ctx}}
        if fmt: b["format"] = fmt
        if self.think is not None: b["think"] = self.think
        return b
    async def json(self, system, user, max_tokens=300):
        r = await self.c.post(self.url + "/api/chat", json=self._body(system, user, max_tokens, False, "json"))
        if r.status_code == 400 and "think" in r.text.lower() and self.think is not None:       # model does not support the think switch
            self.think = None; r = await self.c.post(self.url + "/api/chat", json=self._body(system, user, max_tokens, False, "json"))
        r.raise_for_status()
        return parse_json(r.json()["message"]["content"])
    async def stream(self, system, user, max_tokens=120):
        async with self.c.stream("POST", self.url + "/api/chat", json=self._body(system, user, max_tokens, True)) as r:
            if r.status_code == 400 and self.think is not None:
                await r.aread(); self.think = None
                async for t in self.stream(system, user, max_tokens): yield t
                return
            async for line in r.aiter_lines():
                if not line: continue
                d = json.loads(line)
                t = d.get("message", {}).get("content", "")
                if t: yield t
                if d.get("done"): break
    async def warm(self):
        await self.json("Reply {}", "{}", 5)

class OpenAILLM:
    """OpenAI chat models. Newer (reasoning-family) models reject max_tokens and think before answering, so we use max_completion_tokens and the
    lowest reasoning effort the model accepts (probed once per model)."""
    local = False
    _probed: dict = {}
    def __init__(self, model="gpt-4o-mini"):
        from openai import AsyncOpenAI
        self.model = model; self.name = f"openai:{model}"; self.c = AsyncOpenAI()
    async def _create(self, messages, max_tokens, stream=False, json_mode=False):
        kw = {"model": self.model, "messages": messages, "stream": stream}
        if json_mode: kw["response_format"] = {"type": "json_object"}
        variants = self._probed.get(self.model) or [dict(max_completion_tokens=max(max_tokens * 4, 160), reasoning_effort="minimal"), dict(max_completion_tokens=max(max_tokens * 4, 160), reasoning_effort="low"),
                                                    dict(max_completion_tokens=max(max_tokens * 4, 160)), dict(max_tokens=max_tokens)]
        last = None
        for v in variants:
            try:
                r = await self.c.chat.completions.create(**kw, **v); self._probed[self.model] = [v]; return r
            except Exception as ex:
                last = ex
                if "unsupported" in str(ex).lower() or "invalid" in str(ex).lower() or "max_tokens" in str(ex) or "reasoning" in str(ex).lower() or "does not support" in str(ex).lower(): continue
                raise
        raise last
    async def json(self, system, user, max_tokens=300):
        r = await self._create([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens, json_mode=True)
        return parse_json(r.choices[0].message.content or "")
    async def stream(self, system, user, max_tokens=120):
        s = await self._create([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens, stream=True)
        async for ch in s:
            if ch.choices and ch.choices[0].delta.content: yield ch.choices[0].delta.content
    async def warm(self):
        await self.json("Reply with a JSON object.", "{}", 5)

class ScriptedLLM:
    """Deterministic LLM for unit tests: json() pops dicts / calls fn; stream() yields text."""
    local = True
    def __init__(self, json_fn=None, text="질문입니다."):
        self.name = "scripted"; self.json_fn = json_fn or (lambda s, u: {}); self.text = text; self.calls = []
    async def json(self, system, user, max_tokens=300):
        self.calls.append(("json", user)); return self.json_fn(system, user)
    async def stream(self, system, user, max_tokens=120):
        self.calls.append(("stream", user))
        for w in self.text.split(" "): yield w + " "

def make_llm(spec: str):
    kind, _, model = spec.partition(":")
    if kind == "ollama": return OllamaLLM(model or "qwen2.5:7b")
    if kind == "ollamax": return OllamaLLM(model, url=os.environ.get("OLLAMA_EXT_URL", "http://127.0.0.1:11435"), tag="ollamax")   # models stored on the external drive
    if kind == "openai": return OpenAILLM(model or "gpt-4o-mini")
    raise ValueError(spec)
