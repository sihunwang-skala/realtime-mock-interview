from __future__ import annotations
import asyncio, re, tempfile, os, numpy as np, soundfile as sf
class SupertonicTTS:
    """Local neural TTS (Supertonic 3, ONNX, Korean). Model lives in ai/qa-lab/models/tts/supertonic-3 (git-ignored; delete or move that folder freely,
    or point SUPERTONIC_DIR at it). One sentence chunk per call; ~46 ms PCM pieces are not needed - the runtime accepts any chunk size."""
    local = True
    def __init__(self, voice="F1", steps=6, speed=1.05):
        import os, pathlib
        self.voice = voice; self.steps = steps; self.speed = speed; self.name = f"supertonic:{voice}"
        d = os.environ.get("SUPERTONIC_DIR") or str(pathlib.Path(__file__).resolve().parents[2] / "models" / "tts" / "supertonic-3")
        from supertonic import TTS
        self.tts = TTS(model_dir=d, auto_download=False); self.style = self.tts.get_voice_style(voice); self.sr = int(getattr(self.tts, "sample_rate", 44100))
    def warm(self): self._synth("네.")
    def _synth(self, text: str):
        wav, _ = self.tts.synthesize(normalize_ko(text), voice_style=self.style, total_steps=self.steps, speed=self.speed, lang="ko")
        a = np.asarray(wav).reshape(-1).astype(np.float32)
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(self.sr, 16000); return resample_poly(a, 16000 // g, self.sr // g).astype(np.float32)
        except Exception:
            return np.interp(np.linspace(0, len(a) - 1, max(int(len(a) * 16000 / self.sr), 1)), np.arange(len(a)), a).astype(np.float32)
    async def stream(self, text: str):
        a = await asyncio.to_thread(self._synth, text)
        if len(a): yield a
    async def synth(self, text: str):
        return await asyncio.to_thread(self._synth, text)

class OpenAITTS:
    """Cloud TTS (natural, human-like Korean). The text of each spoken sentence is sent to OpenAI - opt-in only."""
    local = False
    def __init__(self, voice="marin", model="gpt-4o-mini-tts"):
        self.voice = voice; self.model = model; self.name = f"openai-tts:{model}:{voice}"; self._c = None
    def warm(self): pass
    async def stream(self, text: str):
        import httpx, os
        if self._c is None: self._c = httpx.AsyncClient(timeout=60)
        body = {"model": self.model, "voice": self.voice, "input": normalize_ko(text), "response_format": "pcm",
                "instructions": "차분하고 친절한 한국어 면접관처럼, 사람이 대화하듯 자연스럽게 또박또박 말한다. 기계적인 억양 없이 문장 끝을 부드럽게 올리거나 내린다."}
        async with self._c.stream("POST", "https://api.openai.com/v1/audio/speech", json=body, headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}) as r:
            if r.status_code != 200:
                await r.aread(); raise RuntimeError(f"openai tts {r.status_code}: {r.text[:120]}")
            carry = b""
            async for b in r.aiter_bytes(4800):             # 24 kHz s16le mono
                carry += b; n = len(carry) // 2 * 2
                if n == 0: continue
                a24 = np.frombuffer(carry[:n], np.int16).astype(np.float32) / 32768; carry = carry[n:]
                yield np.interp(np.linspace(0, len(a24) - 1, max(int(len(a24) * 16000 / 24000), 1)), np.arange(len(a24)), a24).astype(np.float32)
    async def synth(self, text: str):
        parts = [c async for c in self.stream(text)]; return np.concatenate(parts) if parts else np.zeros(0, np.float32)

class SayTTS:
    """Local macOS TTS (Yuna). No true streaming: one subprocess per chunk; ~0.8s floor measured."""
    local = True; name = "macos-say:Yuna"
    def __init__(self, voice="Yuna", rate=200): self.voice = voice; self.rate = rate
    async def stream(self, text: str):
        a = await self.synth(text)
        if len(a): yield a
    async def synth(self, text: str) -> np.ndarray:
        text = normalize_ko(text)
        fd, p = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        proc = await asyncio.create_subprocess_exec("say", "-v", self.voice, "-r", str(self.rate), "-o", p, "--file-format=WAVE", "--data-format=LEI16@16000", text)
        await proc.wait()
        try: a, _ = sf.read(p, dtype="float32")
        finally: os.unlink(p)
        return a

_UNITS = {"%": "퍼센트", "ms": "밀리초", "TPS": "티피에스", "API": "에이피아이", "AI": "에이아이", "RFP": "알에프피", "HSM": "에이치에스엠", "AES": "에이이에스"}
def normalize_ko(t: str) -> str:
    """Minimal Korean TTS normalisation: units/acronyms. Digits are left to the engine (measured separately)."""
    for k, v in _UNITS.items(): t = t.replace(k, v)
    from .ko_pron import to_spoken
    t = to_spoken(t)                                                          # English words -> Hangul reading (a Korean voice otherwise switches to an English-like voice mid-sentence)
    t = re.sub(r"['\"\u2018\u2019\u201c\u201d`]", "", t)                     # quote marks make some voices pause or mumble: speak the words plainly
    return re.sub(r"\s+", " ", t).strip()

def chunk_sentences(buf: str, first: bool) -> tuple[list[str], str]:
    """Split streamed text into speakable chunks. The first chunk is flushed early (comma / >=14 chars) to cut time-to-first-audio."""
    out = []
    while True:
        m = re.search(r"[.?!。]\s*|\n", buf) or (re.search(r"[,，]\s*|\s", buf[14:]) if first and len(buf) >= 14 else None)
        if not m: break
        end = m.end() + (14 if (first and not re.search(r"[.?!。]\s*|\n", buf)) else 0)
        out.append(buf[:end].strip()); buf = buf[end:]; first = False
    return [x for x in out if x], buf

# ---- pre-synthesised audio: sentences that are known in advance (prepared questions, follow-ups, interjection templates) are voiced in the
# background, so speaking them costs ~0 s instead of the synthesis time of the chosen voice (Supertonic F1: ~0.4 s for the first chunk, ~1.3 s for a full question).
AUDIO_CACHE: dict = {}          # (voice name, chunk text) -> float32 PCM at 16 kHz
def _ck(tts, text: str): return (getattr(tts, "name", type(tts).__name__), text.strip())
def cached_audio(tts, text: str): return AUDIO_CACHE.get(_ck(tts, text))
def speak_chunks(text: str) -> list[str]:
    """The exact chunks the runtime speaks for a known sentence (same word-by-word feeding as VoiceRuntime._speak_stream), so cached chunks always match."""
    buf, first, out = "", True, []
    for w in re.findall(r"\S+\s*", text):
        buf += w; ch, buf = chunk_sentences(buf, first)
        for c in ch: first = False; out.append(c)
    if buf.strip(): out.append(buf.strip())
    return out
async def prewarm_audio(tts, texts, busy=lambda: False) -> int:
    """Synthesise every not-yet-cached chunk of the given sentences, in order. Local voices only (cloud voices cost money per call). Pauses while a live turn runs."""
    if not getattr(tts, "local", False): return 0
    n = 0
    for t in texts:
        for c in speak_chunks(t):
            if _ck(tts, c) in AUDIO_CACHE: continue
            while busy(): await asyncio.sleep(0.05)
            try: a = await tts.synth(c)
            except Exception: continue
            if a is not None and len(a): AUDIO_CACHE[_ck(tts, c)] = a; n += 1
    return n

class AVSpeechTTS:
    """Local streaming TTS: macOS AVSpeechSynthesizer (Yuna) in a worker process; ~46 ms PCM chunks, 16 kHz float32.
    One reader thread owns the worker's stdout and routes frames to the utterance that asked for them (by id), so cancelling an utterance
    (repeat / skip / barge-in / interjection) can never leave the pipe half-read or let two readers steal each other's frames."""
    local = True; name = "macos-avspeech:Yuna"
    def __init__(self, rate=0.5):
        import subprocess, sys
        self.rate = rate; self._id = 0; self._lock = None; self._subs = {}; self._started = False; self._loop = None
        self.p = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "qa_lab.providers.av_worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        hdr = self.p.stdout.read(8)   # ready signal
        assert hdr, "av_worker failed to start"
    def warm(self):
        import json, struct                       # sync, before the reader thread exists
        self._id += 1; self.p.stdin.write((json.dumps({"id": self._id, "text": "네.", "rate": self.rate}) + "\n").encode()); self.p.stdin.flush()
        while True:
            _, n = struct.unpack("<II", self.p.stdout.read(8))
            if n == 0: return
            self.p.stdout.read(n)
    def _reader(self):
        import struct
        rd = self.p.stdout.read
        while True:
            h = rd(8)
            if not h or len(h) < 8: self._loop.call_soon_threadsafe(self._dispatch, -1, None); return
            i, n = struct.unpack("<II", h); data = rd(n) if n else b""
            self._loop.call_soon_threadsafe(self._dispatch, i, data)
    def _dispatch(self, i, data):
        if i == -1:
            for q in list(self._subs.values()): q.put_nowait(None)
            return
        q = self._subs.get(i)
        if q is not None: q.put_nowait(data)        # frames of a cancelled utterance are simply dropped
    async def stream(self, text: str):
        import asyncio, json, threading, numpy as np
        if self._lock is None: self._lock = asyncio.Lock()
        async with self._lock:
            if not self._started:
                self._loop = asyncio.get_running_loop(); threading.Thread(target=self._reader, daemon=True).start(); self._started = True
            self._id += 1; i = self._id; qq: asyncio.Queue = asyncio.Queue(); self._subs[i] = qq
            self.p.stdin.write((json.dumps({"id": i, "text": normalize_ko(text), "rate": self.rate}) + "\n").encode()); self.p.stdin.flush()
            try:
                while True:
                    data = await qq.get()
                    if not data: return              # None = worker died, b"" = end of utterance
                    yield np.frombuffer(data, np.float32).copy()
            finally:
                self._subs.pop(i, None)
    async def synth(self, text: str):
        import numpy as np
        parts = [c async for c in self.stream(text)]; return np.concatenate(parts) if parts else np.zeros(0, np.float32)
