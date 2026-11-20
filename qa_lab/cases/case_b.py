"""Case B: local/self-hosted — Silero VAD + Faster-Whisper + Ollama + macOS say, in-process audio replay / WebSocket."""
from __future__ import annotations
from ..core.engine import QAEngine
from ..core.llm import make_llm
from ..data.rfp import REQS
from ..providers.stt import FasterWhisperSTT, OpenAISTT
from ..providers.tts import AVSpeechTTS, SayTTS, OpenAITTS, SupertonicTTS
from ..providers.vad import SileroVAD
from .runtime import VoiceRuntime

_STT = {}
def build(opts: dict):
    llm = make_llm(opts.get("llm", "ollama:llama3.2:3b")); deep = make_llm(opts["deep"]) if opts.get("deep") else None
    model = opts.get("stt", "small")
    if model not in _STT: _STT[model] = OpenAISTT(model) if model.startswith("gpt-") else FasterWhisperSTT(model); _STT[model].warm()
    stt = _STT[model]
    tkey = 'tts:' + opts.get('tts', 'av')
    if tkey not in _STT: _STT[tkey] = AVSpeechTTS() if opts.get('tts', 'av') in ('av', 'macos-avspeech:Yuna') else SupertonicTTS(opts['tts'].split(':', 1)[1]) if str(opts.get('tts')).startswith('supertonic') else OpenAITTS(opts['tts'].split(':', 1)[1]) if str(opts.get('tts')).startswith('openai-tts') else SayTTS(); getattr(_STT[tkey], 'warm', lambda: None)()
    tts = _STT[tkey]
    def make_engine(): return QAEngine(llm, REQS, deep=deep, strategy=opts.get("strategy", "hybrid"), interjection=opts.get("interjection", "hybrid"))
    def make_runtime(eng, on_event, on_audio):
        manual = opts.get("turn_mode") == "manual"
        c = 0 if manual else int(opts.get("confirm_ms") or 0)
        vl = SileroVAD(silence_ms=c, own_model=True) if c else None
        return VoiceRuntime(eng, SileroVAD(silence_ms=opts.get("silence_ms", 600), own_model=bool(c)), stt, tts, vad_late=vl, turn_mode="manual" if manual else "auto", on_event=on_event, on_audio=on_audio, enable_interjection=opts.get("interjection", "hybrid") != "off")
    return make_engine, make_runtime, {"llm": llm.name, "stt": stt.name, "tts": tts.name, "turn": "버튼으로 직접 끝냄" if opts.get("turn_mode") == "manual" else "자동 감지", "vad": f"silero {opts.get('silence_ms', 600)}ms" + (f" + confirm {opts.get('confirm_ms')}ms (speculative)" if opts.get("confirm_ms") else "")}
