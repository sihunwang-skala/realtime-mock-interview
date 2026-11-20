"""One-change-at-a-time latency / heat experiments on the LIVE flow (real VoiceRuntime + QAEngine + question bank), kept deliberately small and cool.

Design (see docs/LATENCY_AB.md)
- Same synthetic answers, same passage, same models for every condition; only ONE thing differs per condition.
- Answers are played in real time into the runtime; [답변 끝] is pressed at the true end; we read the runtime's own LATENCY event (button -> first audio).
- Heat proxies (no sudo needed): CPU-seconds of this process + Ollama, and Whisper seconds spent. Thermal guard before every condition.
- 60 s rest between conditions. 5 trials per condition (3 normal answers -> follow-up path, 2 "I don't know" -> next-area path).
- A candidate is ADOPTED only if it beats the baseline by the rule in RULES; otherwise it is rejected (nothing in the source is changed by this script).

Run:  .venv/bin/python -W ignore -m qa_lab.benchmark.latency_ab --conditions base,pe4_pw8,ctx2048,base --rest 60
"""
from __future__ import annotations
import argparse, asyncio, json, os, resource, statistics as st, subprocess, time, pathlib

PASSAGE = ("Tool Execution은 모델이 요청한 툴 호출을 실제 함수로 실행하고 그 결과를 에이전트에 돌려주는 단계이다. 병렬 실행에서는 Send로 작업을 여러 노드에 보내고 Reducer가 같은 키로 들어온 결과를 충돌 없이 합친다. "
           "툴 실행 결과는 JSON Schema와 Pydantic 모델로 검증해 형식이 맞지 않으면 오류 계약으로 처리한다. 에이전트의 실행 상태는 thread_id와 state 키로 유지하며 체크포인트로 복구할 수 있다. "
           "평가는 벤치마크와 골든 세트로 정확도와 비용, 지연시간을 함께 측정한다. 권한과 비즈니스 규칙은 분리해서 검증해야 한다. Evaluation Gate는 사전에 설정한 기준에 미달하면 배포를 차단한다. "
           "Retry에서 간격을 늘린다는 것은 실패한 Tool을 다시 호출하기 전까지 기다리는 시간을 늘린다는 뜻이다. Timeout, Retry, Circuit Breaker, 반복 호출 제한, Token 및 비용 예산을 적용한다.")
ANSWERS = [("normal", "에이전트는 모델이 요청한 도구 호출을 실제로 실행하고 그 결과를 다시 돌려주는 단계를 말합니다."),
           ("dontknow", "잘 모르겠습니다."),
           ("normal", "병렬로 실행할 때는 작업을 여러 노드에 나누어 보내고 결과를 같은 키로 합쳐서 충돌이 나지 않게 처리합니다."),
           ("dontknow", "모릅니다. 다음 질문 부탁드립니다."),
           ("normal", "평가는 벤치마크와 골든 세트를 사용해서 정확도와 비용 그리고 지연시간을 함께 측정합니다.")]
LLM = "ollama:phi4-mini:3.8b"
BASE_OPTS = {"llm": LLM, "stt": "small", "tts": "supertonic:M5", "strategy": "two_track", "interjection": "rule", "turn_mode": "manual"}

# name -> (kind, description, opts overrides, env overrides, function applied to (rt, eng) after creation)
CONDITIONS = {
    "base":      ("baseline", "현재 설정 그대로", {}, {}, None),
    "pe4_pw8":   ("heat", "부분 받아쓰기 2초→4초마다, 창 14초→8초", {}, {}, lambda rt, e: (setattr(rt, "partial_every", 4.0), setattr(rt, "partial_window", 8.0))),
    "pe4_pw14":  ("heat", "부분 받아쓰기 2초→4초마다 (창 14초는 그대로: 끼어들기 규칙이 14초 분량 글자수를 전제로 함)", {}, {}, lambda rt, e: setattr(rt, "partial_every", 4.0)),
    "pe3_pw10":  ("heat", "부분 받아쓰기 2초→3초마다, 창 14초→10초", {}, {}, lambda rt, e: (setattr(rt, "partial_every", 3.0), setattr(rt, "partial_window", 10.0))),
    "ctx2048":   ("latency", "LLM 문맥 길이 4096→2048", {}, {"QA_NUM_CTX": "2048"}, None),
    "interj_off": ("info", "끼어들기 끔(부분 받아쓰기 없음) — 열의 바닥값 확인용, 채택 대상 아님", {"interjection": "off"}, {}, None),
}
# adoption rules (relative to the most recent baseline)
RULES = {"latency": "중앙값이 8% 이상 AND 0.05초 이상 줄고, Whisper 시간이 15% 넘게 늘지 않음",
         "heat": "Whisper 시간이 15% 이상 줄고, 중앙값이 0.05초 넘게 늘지 않고, 최대값이 0.3초 넘게 늘지 않음 (CPU초는 Ollama 재시작·다른 세션 때문에 믿을 수 없어 참고만)"}


def _ollama_cpu_seconds() -> float:
    tot = 0.0
    try:
        out = subprocess.run(["ps", "-axo", "time=,command="], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "llama-server" in line:
                t = line.split()[0]; parts = t.replace("-", ":").split(":"); s = 0.0
                for p in parts: s = s * 60 + float(p)
                tot += s
    except Exception: pass
    return tot


def _cpu_now() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime + _ollama_cpu_seconds()


async def _thermal_guard(max_wait: int = 240):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        load = os.getloadavg()[0]
        therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
        limited = "CPU_Speed_Limit" in therm and "100" not in therm.split("CPU_Speed_Limit")[1][:12]
        if load < 4.0 and not limited: return {"load": round(load, 2), "waited_s": round(time.time() - t0)}
        await asyncio.sleep(10)
    return {"load": round(os.getloadavg()[0], 2), "waited_s": max_wait, "warn": "still warm"}


async def _clips():
    from ..providers.tts import AVSpeechTTS
    import numpy as np
    y = AVSpeechTTS(); y.warm(); out = []
    for kind, text in ANSWERS: out.append((kind, text, await y.synth(text)))
    try: y.p.terminate()
    except Exception: pass
    return out, np


async def run_condition(name: str, clips, np_, bank_ready: bool) -> dict:
    from ..cases import case_b
    from ..core.passage import apply_passage
    kind, desc, over, env, fn = CONDITIONS[name]
    old_env = {k: os.environ.get(k) for k in env}; os.environ.update(env)
    opts = {**BASE_OPTS, **over}
    try:
        make_engine, make_runtime, info = await asyncio.to_thread(case_b.build, opts)
        eng = make_engine(); eng.keep_going = True
        await apply_passage(eng, PASSAGE)
        events: list[dict] = []
        rt = make_runtime(eng, lambda ev: events.append({"t": time.monotonic(), **ev.to_dict()}), lambda pcm, m: None)
        if fn: fn(rt, eng)
        rt.start_audio_prewarm()
        # count Whisper work (heat proxy)
        stt_stat = {"calls": 0, "sec": 0.0}; orig = rt.stt.transcribe
        def counted(a, _o=orig):
            t = time.perf_counter(); r = _o(a); stt_stat["calls"] += 1; stt_stat["sec"] += time.perf_counter() - t; return r
        rt.stt.transcribe = counted
        q0 = eng.first_question(); await rt.speak_text(q0.text)
        for _ in range(200):                                       # wait until the first question is voiced (not measured)
            if any(e["type"] == "AGENT_SPEAKING" and (e.get("data") or {}).get("state") == "synth_done" for e in events): break
            await asyncio.sleep(0.1)
        await asyncio.sleep(1.5)
        # one warm-up answer (not counted) so model loads / caches do not pollute trial 1
        res, cpu0, t0 = [], _cpu_now(), time.time(); stt_stat.update(calls=0, sec=0.0)
        for i, (akind, text, wav) in enumerate(clips):
            for _ in range(100):                                   # previous AI speech must be finished
                if not rt.agent_speaking and (rt.agent_task is None or rt.agent_task.done()): break
                await asyncio.sleep(0.1)
            if i:                                                   # let the voice-activity detector fall back to idle (the real user is silent while the AI talks)
                for j in range(0, int(1.4 * 16000), 1024): await rt.feed(np_.zeros(1024, np_.float32)); await asyncio.sleep(1024 / 16000)
            n_before = len([e for e in events if e["type"] == "LATENCY"])
            sil = np_.zeros(int(0.4 * 16000), np_.float32); audio = np_.concatenate([sil, wav.astype(np_.float32)])
            for j in range(0, len(audio), 1024):
                await rt.feed(audio[j:j + 1024]); await asyncio.sleep(1024 / 16000)
            ok, why = await rt.finish_answer()
            for _ in range(150):                                   # wait for the first audio of the reply
                lat = [e for e in events if e["type"] == "LATENCY"]
                if len(lat) > n_before: break
                await asyncio.sleep(0.05)
            lat = [e for e in events if e["type"] == "LATENCY"][n_before:]
            d = (lat[0].get("data") if lat else None) or {}
            res.append({"kind": akind, "ok": ok, "why": why, "total_s": d.get("total_s"), "stt_s": d.get("stt_s"), "generate_s": d.get("generate_s"), "tts_s": d.get("tts_s")})
            for _ in range(150):
                if rt.agent_task is None or rt.agent_task.done(): break
                await asyncio.sleep(0.1)
            await asyncio.sleep(1.0)
        cpu = _cpu_now() - cpu0; wall = time.time() - t0
        tot = [r["total_s"] for r in res if r["total_s"] is not None]
        await rt.close()
        return {"name": name, "kind": kind, "desc": desc, "trials": res, "median": round(st.median(tot), 3) if tot else None, "mean": round(st.mean(tot), 3) if tot else None,
                "max": round(max(tot), 3) if tot else None, "cpu_s": round(cpu, 1), "wall_s": round(wall, 1), "stt_calls": stt_stat["calls"], "stt_s": round(stt_stat["sec"], 1)}
    finally:
        for k, v in old_env.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def decide(c: dict, base: dict) -> tuple[str, str]:
    if c["kind"] in ("baseline", "info") or base is None or c["median"] is None or base["median"] is None: return "참고", ""
    dm = c["median"] - base["median"]; dcpu = (c["stt_s"] - base["stt_s"]) / max(base["stt_s"], 1e-9); dmax = c["max"] - base["max"]
    if c["kind"] == "latency":
        ok = dm <= -0.05 and dm / base["median"] <= -0.08 and dcpu <= 0.15
        why = f"중앙값 {dm:+.2f}초({dm/base['median']*100:+.0f}%), Whisper시간 {dcpu*100:+.0f}%"
    else:
        ok = dcpu <= -0.15 and dm <= 0.05 and dmax <= 0.3
        why = f"Whisper시간 {dcpu*100:+.0f}%, 중앙값 {dm:+.2f}초, 최대 {dmax:+.2f}초"
    return ("채택" if ok else "기각"), why


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--conditions", default="base,pe4_pw8,ctx2048,base"); ap.add_argument("--rest", type=int, default=30); ap.add_argument("--pre", type=int, default=0); ap.add_argument("--normal-only", action="store_true"); ap.add_argument("--trials", type=int, default=5)
    a = ap.parse_args(); names = a.conditions.split(",")
    from ..core.passage import build_bank, BANK, bank_key
    print("① 질문 은행 만들기(1회, 측정 안 함) …", flush=True)
    await build_bank(PASSAGE, LLM)
    b = BANK.get(bank_key(PASSAGE, LLM)); print(f"   질문 {len(b['qs']) if b else 0}개 준비", flush=True)
    clips, np_ = await _clips()
    if a.normal_only: clips = [c for c in clips if c[0] == "normal"] * (a.trials // 3 + 1)      # only right answers: every trial needs a follow-up
    clips = clips[:a.trials]
    if a.pre: print(f"   … 시작 전 {a.pre}초 쉬는 중", flush=True); await asyncio.sleep(a.pre)
    results, base = [], None; ts = time.strftime("%Y%m%d-%H%M")
    for i, n in enumerate(names):
        if i: print(f"   … {a.rest}초 쉬는 중", flush=True); await asyncio.sleep(a.rest)
        g = await _thermal_guard(); print(f"② [{i+1}/{len(names)}] {n}  (부하 {g['load']}, 대기 {g['waited_s']}초) — {CONDITIONS[n][1]}", flush=True)
        r = await run_condition(n, clips, np_, True); r["guard"] = g
        if n == "base": base = r
        verdict, why = decide(r, base); r["verdict"], r["why"] = verdict, why
        results.append(r)
        print(f"   → 중앙값 {r['median']}초 · 평균 {r['mean']}초 · 최대 {r['max']}초 · CPU {r['cpu_s']}초 · Whisper {r['stt_calls']}회/{r['stt_s']}초 · 판정 {verdict} {why}", flush=True)
    out = pathlib.Path(__file__).resolve().parents[2] / "results" / f"latency_ab_{ts}.json"
    out.write_text(json.dumps({"rules": RULES, "results": results}, ensure_ascii=False, indent=1)); print("저장:", out)

if __name__ == "__main__": asyncio.run(main())
