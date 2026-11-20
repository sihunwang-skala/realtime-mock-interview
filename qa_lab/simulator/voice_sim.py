"""Demo B: Presenter LLM -> TTS -> real audio -> Evaluator STT/VAD/Agent -> Evaluator TTS -> audio -> Presenter STT -> Presenter LLM."""
from __future__ import annotations
import asyncio, time, numpy as np
from ..cases.registry import CASES
from ..contracts import Question
from ..core.llm import make_llm
from ..data.rfp import SEED_BANK, RFP_TEXT
from .presenter import Presenter
import importlib

async def run(body: dict, turns=3, behaviours=("normal", "vague", "contradict")):
    case = body.get("case", "B"); opts = {k: body[k] for k in ("llm", "stt", "tts", "strategy", "interjection") if body.get(k)}
    opts.setdefault("llm", "ollama:qwen2.5:7b"); opts.setdefault("strategy", "two_track")
    mk_eng, mk_rt, info = await asyncio.to_thread(lambda: importlib.import_module(f"qa_lab.cases.{CASES[case]['module']}").build(opts))
    eng = mk_eng()
    for t in SEED_BANK: eng.queue.add(Question(t, source="seed", priority=.6))
    heard: list[np.ndarray] = []
    rt = mk_rt(eng, lambda e: None, lambda pcm, m: heard.append(pcm))
    pres_llm = make_llm(body.get("presenter", "ollama:qwen2.5:7b")); pres = Presenter(pres_llm, RFP_TEXT)
    stt = rt.stt; tts = rt.tts; log = []
    q = eng.first_question(); await rt.speak_text(q.text); await _quiet(rt, heard)
    for t in range(turns):
        qtext = rt_text(heard, stt) if False else eng.cur.text
        heard_text = await asyncio.to_thread(stt.transcribe, np.concatenate(heard)) if heard else ""   # presenter hears evaluator audio via STT
        ans = await pres.answer(heard_text or qtext, behaviours[t % len(behaviours)])
        wav = await tts.synth(ans); wav = np.concatenate([np.zeros(8000, np.float32), wav, np.zeros(int(1.5 * 16000), np.float32)])
        heard.clear(); rt.responses.clear(); t0 = time.monotonic(); end_wall = t0 + (len(wav) - int(1.5 * 16000)) / 16000
        for i in range(0, len(wav) - 512, 512):
            await rt.feed(wav[i:i + 512]); await asyncio.sleep(max(0, t0 + (i + 512) / 16000 - time.monotonic()))
        await _quiet(rt, heard, wait_resp=True)
        r = rt.responses[-1] if rt.responses else {}
        mk = r.get("marks", {}); e2e = (mk["AUDIO_FIRST_PLAYED"] - end_wall) if mk.get("AUDIO_FIRST_PLAYED") else None
        log.append({"turn": t, "evaluator_heard_by_presenter": heard_text, "presenter_answer": ans, "evaluator_stt": r.get("stt_text"), "action": r.get("action"), "evaluator_question": r.get("question"), "e2e_s": round(e2e, 3) if e2e else None,
                    "audio_s": round(sum(len(h) for h in heard) / 16000, 2)})
    await rt.close(); return {"components": info, "turns": log}

def rt_text(*a): return ""
async def _quiet(rt, heard, wait_resp=False, timeout=25):
    dl = time.monotonic() + timeout; n = -1
    while time.monotonic() < dl:
        await asyncio.sleep(0.4)
        cur = sum(len(h) for h in heard)
        if cur == n and cur > 0 and (not wait_resp or rt.responses) and not rt.agent_speaking: return
        n = cur
