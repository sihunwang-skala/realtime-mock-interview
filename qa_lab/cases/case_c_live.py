"""Case C live: browser mic -> this server -> OpenAI Realtime (speech-to-speech) -> browser speaker. The API key stays on the server.
PRIVACY: the user's voice and the RFP text are sent to OpenAI. The caller must pass consent=1. Use only public/synthetic RFPs.
Limits: follow-up questions are authored by the model (we cannot run our planner on them); proactive interjection is NOT implemented for C
(only OpenAI's own barge-in: the user can cut the model off). Latency is measured server-side from (speech_stopped - silence) to the first audio delta."""
from __future__ import annotations
import asyncio, base64, json, os, time
import numpy as np
from ..data.rfp import RFP_TEXT, SEED_BANK

URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"; SILENCE_MS = 600
INSTR = ("당신은 까다롭지만 공정한 RFP 제안 평가위원이다. 한국어 구어체 존댓말로 말한다. 한 번에 질문 하나만, 한두 문장으로 짧게 한다. "
         "발표자의 답변이 RFP 요구사항과 다르거나 근거(수치)가 없으면 그 점을 짚어 꼬리질문을 한다. 발표자 대신 답하지 않는다.\n[RFP 요구사항]\n" + RFP_TEXT)

def up(pcm16_16k: bytes) -> bytes:
    x = np.frombuffer(pcm16_16k, np.int16).astype(np.float32)
    return np.interp(np.linspace(0, len(x) - 1, int(len(x) * 1.5)), np.arange(len(x)), x).astype(np.int16).tobytes()
def down(pcm16_24k: bytes) -> bytes:
    x = np.frombuffer(pcm16_24k, np.int16).astype(np.float32)
    return np.interp(np.linspace(0, len(x) - 1, int(len(x) / 1.5)), np.arange(len(x)), x).astype(np.int16).tobytes()

async def run(ws, sid: str, manual: bool = True):
    import websockets
    key = os.environ.get("OPENAI_API_KEY")
    if not key: await ws.send_json({"type": "ERROR", "errors": ["OPENAI_API_KEY not set"]}); return
    seq = [0]; queue = list(SEED_BANK); asked: list[str] = []; turns: list[dict] = []; st = {"speaking": False, "user_end": None, "first": True, "last_q": "", "user_text": "", "t_stop": None, "spoke": False}
    async def emit(t, **d):
        seq[0] += 1; await ws.send_json({"type": t, "session_id": sid, "seq": seq[0], "ts": time.time(), "data": d})
    async with websockets.connect(URL, additional_headers={"Authorization": f"Bearer {key}"}) as oa:
        await oa.send(json.dumps({"type": "session.update", "session": {"type": "realtime", "model": "gpt-realtime", "output_modalities": ["audio"], "instructions": INSTR,
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "transcription": {"model": "gpt-4o-transcribe", "language": "ko"},
                                "turn_detection": (None if manual else {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": SILENCE_MS, "create_response": True, "interrupt_response": True})},
                      "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "marin"}}}}))
        await ws.send_json({"type": "SESSION_READY", "session_id": sid, "seq": 0, "components": {"architecture": "OpenAI Realtime (speech-to-speech)", "llm": "gpt-realtime", "stt": "gpt-4o-transcribe (transcript only)", "tts": "gpt-realtime voice marin", "data_leaves_machine": True}})
        async def say_exact(text):
            asked.append(text); st["last_q"] = text
            await oa.send(json.dumps({"type": "response.create", "response": {"instructions": f"다음 문장을 한 글자도 바꾸지 말고 그대로 말하라: {text}"}}))
        async def from_client():
            while True:
                m = await ws.receive()
                if m.get("type") == "websocket.disconnect": return
                if m.get("bytes") is not None:
                    if manual and not st["spoke"] and float(np.sqrt((np.frombuffer(m["bytes"], np.int16).astype(np.float32) ** 2).mean())) > 600:
                        st["spoke"] = True; await emit("USER_SPEAKING")                 # no server VAD in manual mode: simple energy gate for the indicator
                    await oa.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(up(m["bytes"])).decode()})); continue
                d = json.loads(m["text"]); t = d.get("type"); cmd = d.get("cmd"); res = {"ok": True}
                if t == "ping": await ws.send_json({"type": "pong", "t0": d["t0"], "server_ts": time.monotonic()}); continue
                if t == "audio_started": 
                    if st["user_end"]: await emit("LATENCY", browser_play_after_user_end_s=round(d["server_clock_ts"] - st["user_end"], 3))
                    continue
                if t == "played": continue
                if cmd == "START": 
                    if queue: await say_exact(queue.pop(0))
                elif cmd == "REPEAT_QUESTION": await say_exact(st["last_q"]) if st["last_q"] else res.update(ok=False, error="no question yet")
                elif cmd == "SKIP_QUESTION":
                    if queue: await say_exact(queue.pop(0))
                    else: res.update(ok=False, error="no more seed questions")
                elif cmd == "FINISH_ANSWER":
                    if manual and not st["spoke"]: res.update(ok=False, error="아직 말씀하신 내용이 없습니다")
                    elif st["speaking"]: res.update(ok=False, error="AI가 말하는 중입니다")
                    else:
                        st["user_end"] = time.monotonic(); st["spoke"] = False; await emit("AGENT_THINKING")      # latency clock starts at the button press
                        await oa.send(json.dumps({"type": "input_audio_buffer.commit"})); await oa.send(json.dumps({"type": "response.create"}))
                elif cmd == "END_SESSION":
                    await emit("REPORT_PROCESSING"); await emit("SESSION_ENDED")
                    await emit("REPORT_READY", report={"asked": asked, "turns": turns, "interjections": [], "note": "Case C: questions authored by the model"}); await ws.send_json({"type": "CMD_RESULT", "cmd": cmd, "ok": True}); return
                else: res.update(ok=False, error=f"unknown command {cmd}")
                await ws.send_json({"type": "CMD_RESULT", "cmd": cmd, **res})
        async def from_openai():
            async for raw in oa:
                ev = json.loads(raw); t = ev.get("type", "")
                if t == "input_audio_buffer.speech_started":
                    if st["speaking"]: st["speaking"] = False; await emit("AGENT_SPEAKING", state="interrupted")
                    await emit("USER_SPEAKING")
                elif t == "input_audio_buffer.speech_stopped" and not manual: st["user_end"] = time.monotonic() - SILENCE_MS / 1000; await emit("AGENT_THINKING")
                elif t == "conversation.item.input_audio_transcription.completed": st["user_text"] = ev.get("transcript", ""); await emit("ANSWER_COMPLETED", text=st["user_text"])
                elif t.endswith("audio.delta") and ev.get("delta"):
                    if st["first"]:
                        st["first"] = False; st["speaking"] = True; await emit("AGENT_SPEAKING", state="started")
                        if st["user_end"]: await emit("LATENCY", server_first_audio_after_user_end_s=round(time.monotonic() - st["user_end"], 3))
                    await ws.send_bytes(down(base64.b64decode(ev["delta"])))
                elif t.endswith("audio_transcript.done"): st["last_q"] = ev.get("transcript", st["last_q"]); asked.append(st["last_q"]) if st["last_q"] not in asked else None
                elif t == "response.done":
                    st["speaking"] = False; st["first"] = True; await emit("QUESTION_STARTED", text=st["last_q"]); turns.append({"stt_text": st["user_text"], "question": st["last_q"], "action": "MODEL"}); st["user_text"] = ""
                elif t == "error": await emit("ERROR", errors=[json.dumps(ev.get("error"), ensure_ascii=False)[:300]])
        tasks = [asyncio.create_task(from_client()), asyncio.create_task(from_openai())]
        done, pend = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for p in pend: p.cancel()
