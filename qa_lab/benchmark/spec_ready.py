"""How soon (seconds after the person starts talking) is the tailored follow-up AND its voice ready? Compares when the first partial transcription runs.
Synthetic answer, real-time feeding, one change per condition, rest between conditions. Usage: python -m qa_lab.benchmark.spec_ready --conditions base,fp3"""
from __future__ import annotations
import argparse, asyncio, json, pathlib, statistics as st, time
from .latency_ab import PASSAGE, BASE_OPTS, _thermal_guard
CONDS = {"base": {}, "fp3": {"first_partial": 3.0}, "fp3m15": {"first_partial": 3.0, "eng_spec_min_chars": 15}, "m15": {"eng_spec_min_chars": 15}}
QUESTION = "평가 단계에서는 무엇을 어떻게 측정하나요?"
ANSWER = "골든 세트를 사용해서 정확도와 비용을 측정하고요 Evaluation Gate에서 기준에 미달하면 배포를 막도록 설정해 두었습니다 그리고 Retry 간격은 점점 늘립니다"

async def run(name, clip, np_, n):
    from ..cases import case_b
    from ..core.passage import apply_passage
    from ..providers.tts import AUDIO_CACHE, cached_audio, speak_chunks, prewarm_audio
    from ..core.engine import FU_LEAD, FU_LEAD_CANNED, FU_LEAD_CHAIN
    from ..contracts import Question, Action
    mk, mr, _ = await asyncio.to_thread(case_b.build, {**BASE_OPTS, "tts": "supertonic:F1"})
    eng = mk(); eng.keep_going = True; await apply_passage(eng, PASSAGE); eng.first_question()
    rt = mr(eng, lambda ev: None, lambda pcm, m: None)
    for k, v in CONDS[name].items(): setattr(eng if k.startswith("eng_") else rt, k[4:] if k.startswith("eng_") else k, v)
    calls = {"n": 0, "s": 0.0}; orig = rt.stt.transcribe
    def counted(a, _o=orig):
        t = time.perf_counter(); r = _o(a); calls["n"] += 1; calls["s"] += time.perf_counter() - t; return r
    rt.stt.transcribe = counted
    res = []
    for i in range(n):
        AUDIO_CACHE.clear(); await prewarm_audio(rt.tts, [*FU_LEAD.values(), FU_LEAD_CANNED, FU_LEAD_CHAIN])       # the bridges are voiced at session start in real use: not part of the timing
        eng.cur = Question(QUESTION, Action.DEEPEN, None, 1.0, "x", ""); eng.asked_texts = [QUESTION]; eng.used_sents = set(); eng.last_fu_type = None; eng.spec_fu = None; eng.spec_why = ""; eng.followups = 0
        rt.in_speech = False; rt.buf = []; rt.pending.clear(); rt.last_partial_n = 0; rt.partial_task = None; rt.interjected = False; rt.ij_cut = 0; rt.vad.reset() if hasattr(rt.vad, "reset") else None
        await rt.feed(np_.zeros(int(1.4 * 16000), np_.float32))                       # silence first: VAD starts clean
        t_ready = None; step = 1600; t0 = None; clip_s = len(clip) / 16000
        for j in range(0, len(clip), step):
            await rt.feed(clip[j:j + step]); await asyncio.sleep(0.1)
            if t0 is None and rt.in_speech: t0 = rt.speech_t0
            sp = getattr(eng, "spec_fu", None)
            if t_ready is None and t0 and sp and all(cached_audio(rt.tts, c) is not None for c in speak_chunks(sp.get("say") or sp["q"])): t_ready = time.monotonic() - t0
        for _ in range(60):                                                           # the clip is over; wait (max 6 s) for a follow-up that is still being prepared
            sp = getattr(eng, "spec_fu", None)
            if t_ready is None and t0 and sp and all(cached_audio(rt.tts, c) is not None for c in speak_chunks(sp.get("say") or sp["q"])): t_ready = time.monotonic() - t0
            if t_ready is not None: break
            await asyncio.sleep(0.1)
        res.append({"ready_s": None if t_ready is None else round(t_ready, 2), "clip_s": round(clip_s, 1), "q": (eng.spec_fu or {}).get("q", ""), "why": getattr(eng, "spec_why", "")})
        rt.in_speech = False; rt.buf = []; await asyncio.sleep(1.0)
    ok = [r["ready_s"] for r in res if r["ready_s"] is not None]
    return {"cond": name, "n": n, "ready": len(ok), "median": round(st.median(ok), 2) if ok else None, "max": max(ok) if ok else None, "whisper_calls": calls["n"], "whisper_s": round(calls["s"], 1), "trials": res}

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--conditions", default="base,fp3"); ap.add_argument("--rest", type=int, default=30); ap.add_argument("--pre", type=int, default=0); ap.add_argument("--trials", type=int, default=5); a = ap.parse_args()
    from ..providers.tts import AVSpeechTTS
    import numpy as np
    y = AVSpeechTTS(); y.warm(); clip = await y.synth(ANSWER)
    try: y.p.terminate()
    except Exception: pass
    out = []
    if a.pre: print(f"   … 시작 전 {a.pre}초 쉬는 중", flush=True); await asyncio.sleep(a.pre)
    for i, n in enumerate(a.conditions.split(",")):
        if i: print(f"   … {a.rest}초 쉬는 중", flush=True); await asyncio.sleep(a.rest)
        g = await _thermal_guard(); print(f"[{n}] 부하 {g['load']} · 대기 {g['waited_s']}초 · 답변 길이 {len(clip)/16000:.1f}초", flush=True)
        r = await run(n, clip, np, a.trials); out.append(r)
        print(f"   → 준비 완료 {r['ready']}/{r['n']} · 중앙값 {r['median']}초 · 최대 {r['max']}초 · Whisper {r['whisper_calls']}회/{r['whisper_s']}초", flush=True)
        for t in r["trials"]: print("     ", t["ready_s"], "|", t["q"][:48] or t["why"][:48], flush=True)
    p = pathlib.Path(__file__).resolve().parents[2] / "results" / f"spec_ready_{time.strftime('%Y%m%d-%H%M')}.json"; p.write_text(json.dumps(out, ensure_ascii=False, indent=1)); print("저장:", p)
if __name__ == "__main__": asyncio.run(main())
