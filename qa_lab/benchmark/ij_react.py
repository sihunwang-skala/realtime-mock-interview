"""How long after a contradicting number is spoken does the AI interject? Compares the transcription interval (4 s now vs 2 s before). Synthetic speech, real-time feeding."""
from __future__ import annotations
import argparse, asyncio, json, pathlib, statistics as st, time
from .latency_ab import BASE_OPTS, _thermal_guard
PASSAGE = "민원 접수 API의 평균 응답시간은 2초 이내이어야 한다. 동시 접속자 5000명을 처리할 수 있어야 하며 가용성은 99.9% 이상이어야 한다. 개인정보는 암호화하여 저장해야 한다."
PREFIX = "음 그러니까 저희는 시스템을 설계할 때 여러 가지를 고려하는데요 우선 서버를 나누고 캐시를 두고 그리고"
WRONG = "민원 접수 API 응답시간은 5초입니다"
CONDS = {"pe4": {"partial_every": 4.0}, "pe3": {"partial_every": 3.0}, "pe2": {"partial_every": 2.0}}

async def run(name, pre, wrong, np_, n):
    from ..cases import case_b
    from ..core.passage import apply_passage
    from ..contracts import Question, Action
    mk, mr, _ = await asyncio.to_thread(case_b.build, {**BASE_OPTS, "tts": "supertonic:F1"})
    eng = mk(); eng.keep_going = True; await apply_passage(eng, PASSAGE); eng.first_question()
    evs = []
    rt = mr(eng, lambda ev: evs.append((time.monotonic(), ev.to_dict())), lambda pcm, m: None)
    for k, v in CONDS[name].items(): setattr(rt, k, v)
    calls = {"n": 0, "s": 0.0}; orig = rt.stt.transcribe
    def counted(a, _o=orig):
        t = time.perf_counter(); r = _o(a); calls["n"] += 1; calls["s"] += time.perf_counter() - t; return r
    rt.stt.transcribe = counted
    clip = np_.concatenate([pre, wrong, np_.zeros(int(8 * 16000), np_.float32) + 1e-4]).astype(np_.float32)
    trig = (len(pre) + len(wrong)) / 16000; res = []
    for i in range(n):
        evs.clear(); q = "민원 접수 API의 응답시간 기준은 무엇인가요?"
        eng.cur = Question(q, Action.DEEPEN, None, 1.0, "x", ""); eng.asked_texts = [q]
        rt.in_speech = False; rt.buf = []; rt.pending.clear(); rt.last_partial_n = 0; rt.partial_task = None; rt.interjected = False; rt.ij_cut = 0; rt.ij_block_until = 0; rt.agent_speaking = False
        if hasattr(rt.vad, "reset"): rt.vad.reset()
        await rt.feed(np_.zeros(int(1.4 * 16000), np_.float32)); t0 = None; hit = None
        for j in range(0, len(clip), 1600):
            await rt.feed(clip[j:j + 1600]); await asyncio.sleep(0.1)
            if t0 is None and rt.in_speech: t0 = rt.speech_t0
            if hit is None and any(e[1].get("type") == "AGENT_INTERJECTED" for e in evs): hit = time.monotonic()
            if hit is not None: break
        res.append({"react_s": None if (hit is None or t0 is None) else round(hit - t0 - trig, 2), "kind": next((e[1].get("data", {}).get("kind") for e in evs if e[1].get("type") == "AGENT_INTERJECTED"), None)})
        if rt.agent_task and not rt.agent_task.done(): rt.agent_task.cancel()
        rt.in_speech = False; rt.buf = []; await asyncio.sleep(1.0)
    ok = [r["react_s"] for r in res if r["react_s"] is not None]
    return {"cond": name, "n": n, "hit": len(ok), "median": round(st.median(ok), 2) if ok else None, "max": max(ok) if ok else None, "whisper_calls": calls["n"], "whisper_s": round(calls["s"], 1), "trials": res, "trigger_s": round(trig, 1)}

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--conditions", default="pe4,pe2"); ap.add_argument("--rest", type=int, default=30); ap.add_argument("--pre", type=int, default=0); ap.add_argument("--trials", type=int, default=5); a = ap.parse_args()
    from ..providers.tts import AVSpeechTTS
    import numpy as np
    y = AVSpeechTTS(); y.warm(); pre = await y.synth(PREFIX); wrong = await y.synth(WRONG)
    try: y.p.terminate()
    except Exception: pass
    out = []
    if a.pre: print(f"   … 시작 전 {a.pre}초 쉬는 중", flush=True); await asyncio.sleep(a.pre)
    for i, n in enumerate(a.conditions.split(",")):
        if i: print(f"   … {a.rest}초 쉬는 중", flush=True); await asyncio.sleep(a.rest)
        g = await _thermal_guard(); print(f"[{n}] 부하 {g['load']} · 잘못된 숫자가 끝나는 시점 {(len(pre)+len(wrong))/16000:.1f}초", flush=True)
        r = await run(n, pre, wrong, np, a.trials); out.append(r)
        print(f"   → 끼어듦 {r['hit']}/{r['n']} · 숫자 말한 뒤 반응 중앙값 {r['median']}초 · 최대 {r['max']}초 · Whisper {r['whisper_calls']}회/{r['whisper_s']}초", flush=True)
        for t in r["trials"]: print("     ", t, flush=True)
    p = pathlib.Path(__file__).resolve().parents[2] / "results" / f"ij_react_{time.strftime('%Y%m%d-%H%M')}.json"; p.write_text(json.dumps(out, ensure_ascii=False, indent=1)); print("저장:", p)
if __name__ == "__main__": asyncio.run(main())
