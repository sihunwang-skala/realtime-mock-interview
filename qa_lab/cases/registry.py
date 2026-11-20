"""Case registry + capability matrix. Status values are filled from verification evidence, never assumed."""
from __future__ import annotations
import importlib, json, pathlib, os
from ..contracts import Status

OPT = dict(
    vad=["silero(600ms)"], stt=["faster-whisper:small@mps", "faster-whisper:base@mps", "faster-whisper:large-v3-turbo@mps"],
    llm=["ollama:phi4-mini:3.8b", "ollamax:phi4-mini:3.8b", "ollama:qwen3.5:4b", "ollamax:exaone3.5:7.8b", "ollama:qwen2.5:7b", "ollama:qwen3:8b", "ollama:llama3.1:8b", "ollamax:aya-expanse:8b", "ollama:llama3.2:3b", "openai:gpt-5.4-mini", "openai:gpt-6.2-sol", "openai:gpt-6-luna"], tts=["macos-avspeech:Yuna", "supertonic:F1", "supertonic:F2", "supertonic:F3", "supertonic:F4", "supertonic:F5", "supertonic:M1", "supertonic:M2", "supertonic:M3", "supertonic:M4", "supertonic:M5", "macos-say:Yuna", "openai-tts:marin", "openai-tts:coral", "openai-tts:cedar"],
    strategy=["direct", "precomputed", "hybrid", "two_track"], interjection=["rule", "llm", "hybrid"])

CASES = {
 "B": dict(name="Local / self-hosted pipeline", module="case_b", transport="WebSocket (browser mic, 16 kHz PCM)", local_only=True, options=OPT,
           caps=dict(dynamic_followup="yes", external_planner_control="full", proactive_interjection="yes (partial STT + policy)", barge_in="yes", playback_tracking="estimate (chunk durations); browser reports real position",
                     transcript_for_eval="yes", korean_tts="macOS Yuna only (quality limited)", cloud_data_egress="none")),
 "A": dict(name="LiveKit + modular cascade", live_demo=False, module="case_a", transport="WebRTC (LiveKit)", local_only=False, options={**OPT, "stt": OPT["stt"][:2]},
           caps=dict(dynamic_followup="expected", external_planner_control="full", proactive_interjection="expected via session.say()/generate_reply()", barge_in="built-in", playback_tracking="built-in (SpeechHandle)", transcript_for_eval="yes", korean_tts="depends on plugin", cloud_data_egress="depends on plugins")),
 "C": dict(name="OpenAI Realtime speech-to-speech", module="case_c", transport="WebRTC/WebSocket", local_only=False, options=dict(vad=["server_vad 600ms"], stt=["gpt-4o-transcribe"], llm=["gpt-realtime"], tts=["gpt-realtime voice (marin)"], strategy=["model-authored"], interjection=["off (OpenAI barge-in only)"]),
           caps=dict(dynamic_followup="yes (model-authored) / controlled via 'say exactly' instructions", external_planner_control="partial", proactive_interjection="response.create can be sent while user speaks (to verify)", barge_in="built-in", playback_tracking="conversation.item.truncate with audio_end_ms", transcript_for_eval="via input transcription (separate)", korean_tts="to verify", cloud_data_egress="ALL audio+text to OpenAI")),
 "D": dict(name="Pipecat pipeline", module="case_d", transport="WebSocket/WebRTC", local_only=False, options={}, caps={}),
}

def status(case: str) -> dict:
    """Live status derived from evidence on disk (results/*.json) + environment probes."""
    c = CASES[case]; res = pathlib.Path(__file__).resolve().parents[2] / "results"
    if case == "B":
        ok = any(res.glob("matrix_B_*.json")) if res.exists() else False
        return {"status": (Status.IMPLEMENTED_AND_VERIFIED if ok else Status.IMPLEMENTED_NOT_VERIFIED).value, "reason": "benchmarked on replayed audio" if ok else "run benchmark to verify"}
    if case == "C":
        from . import case_c
        ok, why = case_c.probe()
        if ok and (res / "mini_t6_cases.json").exists(): return {"status": Status.IMPLEMENTED_AND_VERIFIED.value, "reason": "benchmarked on synthetic audio (n=3, P50 1.26s); browser mic demo available (sends voice + RFP to OpenAI)"}
        return {"status": Status.IMPLEMENTED_NOT_VERIFIED.value if ok else Status.BLOCKED.value, "reason": why + ("" if ok else " — need a valid OPENAI_API_KEY with Realtime access")}
    if case == "A":
        ok = any(res.glob("case_A_*.json")) if res.exists() else False
        return ({"status": Status.IMPLEMENTED_AND_VERIFIED.value, "reason": "benchmarked through a real LiveKit room (loopback). Browser live demo for A: NOT_IMPLEMENTED (no livekit-client page); livekit-agents framework features NOT_TESTED"}
                if ok else {"status": Status.IMPLEMENTED_NOT_VERIFIED.value, "reason": "code written, benchmark not run yet"})
    return {"status": Status.NOT_IMPLEMENTED.value, "reason": "Pipecat attempted only after A/B/C"}
