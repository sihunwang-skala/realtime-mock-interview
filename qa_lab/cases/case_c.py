"""Case C: direct OpenAI Realtime speech-to-speech (WebSocket from the Python engine; WebRTC from browser is the production path).
STATUS: IMPLEMENTED_NOT_VERIFIED / BLOCKED — the only available OPENAI_API_KEY returned 401 (invalid). The protocol code below follows the
documented event names but has never run against the service. Remove nothing from this docstring until a valid key has exercised it."""
from __future__ import annotations
import base64, json, os, time, httpx

def probe() -> tuple[bool, str]:
    k = os.environ.get("OPENAI_API_KEY")
    if not k: return False, "OPENAI_API_KEY not set"
    try:
        r = httpx.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {k}"}, timeout=10)
    except Exception as e: return False, f"network error: {e}"
    return (r.status_code == 200, "ok" if r.status_code == 200 else f"HTTP {r.status_code}: key rejected by OpenAI")

class RealtimeSession:
    """One response per user turn. External planner stays in control: we send the *exact* question text as instructions
    (`say exactly`), so Realtime acts as a speech layer rather than as the question author — this is the Case C control trade-off."""
    URL = "wss://api.openai.com/v1/realtime?model={model}"
    def __init__(self, model="gpt-realtime", voice="marin"): self.model = model; self.voice = voice; self.ws = None
    async def connect(self):
        import websockets
        self.ws = await websockets.connect(self.URL.format(model=self.model), additional_headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        await self.ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime", "output_modalities": ["audio"], "audio": {
            "input": {"format": {"type": "audio/pcm", "rate": 24000}, "transcription": {"model": "gpt-4o-transcribe", "language": "ko"}, "turn_detection": {"type": "semantic_vad", "create_response": False}},
            "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": self.voice}}, "instructions": "당신은 한국어 평가위원 음성 계층이다. 지시받은 문장을 그대로 읽는다."}}))
    async def send_audio(self, pcm16_24k: bytes): await self.ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm16_24k).decode()}))
    async def speak(self, text: str):
        await self.ws.send(json.dumps({"type": "response.create", "response": {"instructions": f"다음 문장을 한 글자도 바꾸지 말고 그대로 말하라: {text}"}}))
    async def events(self):
        async for m in self.ws: yield json.loads(m)


async def bench(scenario_ids, model="gpt-realtime"):
    """Case C measurement on SYNTHETIC audio only: audio is streamed in real time, then (client-side turn control, 0.6 s after speech end) a response is requested.
    Measures actual speech end -> first audio delta. Question is MODEL-authored (we cannot inject our planner's question without an extra 'say exactly' step)."""
    import asyncio, numpy as np, soundfile as sf, websockets, time, json as _j
    from ..benchmark import dataset
    ok, why = probe()
    if not ok: return {"status": "BLOCKED", "reason": why}
    out = []; scn = {s["id"]: s for s in dataset.load()}
    for sid in scenario_ids:
        s = scn[sid]; a, _ = sf.read(s["wav"], dtype="float32")
        a24 = np.interp(np.linspace(0, len(a) - 1, int(len(a) * 24000 / 16000)), np.arange(len(a)), a)
        pcm = (np.clip(a24, -1, 1) * 32767).astype(np.int16).tobytes(); speech_end_samples = int(s["speech_end_s"] * 24000)
        try:
            async with websockets.connect("wss://api.openai.com/v1/realtime?model=" + model, additional_headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}) as ws:
                await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime", "model": model, "output_modalities": ["audio"],
                    "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": None}, "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "marin"}},
                    "instructions": "당신은 한국어로 말하는 까다로운 RFP 평가위원이다. 발표자의 답변을 듣고 꼬리질문 한 문장만 짧게 말한다."}}))
                t_start = time.monotonic(); step = 480 * 2
                for i in range(0, min(len(pcm), speech_end_samples * 2), step):
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i:i + step]).decode()}))
                    await asyncio.sleep(max(0, t_start + (i / 2 + 480) / 24000 - time.monotonic()))
                t_end = time.monotonic(); await asyncio.sleep(0.6)
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"})); await ws.send(json.dumps({"type": "response.create"}))
                first = None; text = ""; err = None
                async for m in ws:
                    ev = _j.loads(m); t = ev.get("type", "")
                    if t == "error": err = ev.get("error"); break
                    if t.endswith("audio.delta") and first is None: first = time.monotonic()
                    if t.endswith("audio_transcript.delta"): text += ev.get("delta", "")
                    if t == "response.done": break
                out.append({"id": sid, "e2e_s": round(first - t_end, 3) if first else None, "question": text.strip(), "error": err})
        except Exception as ex: out.append({"id": sid, "e2e_s": None, "error": repr(ex)[:200]})
        print("T6/C", out[-1], flush=True)
    xs = [r["e2e_s"] for r in out if r["e2e_s"] is not None]
    from ..benchmark.replay import stats
    return {"status": "IMPLEMENTED_AND_VERIFIED" if xs else "FAILED", "e2e": stats(xs), "rows": out}
