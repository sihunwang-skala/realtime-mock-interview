"""How long does it take to prepare a tailored follow-up WHILE the person is still answering?
Measures, over N repeats: (1) writing the follow-up question from a partial transcript, (2) voicing it, (3) Whisper on a partial of 3/5/8 s.
Then prints the minimum answer length after which the follow-up is ready when the answer ends. Synthetic data only."""
from __future__ import annotations
import asyncio, inspect, json, pathlib, statistics as st, sys, time
from .latency_ab import PASSAGE, LLM, BASE_OPTS, ANSWERS, _clips

SHORT = ["Send로 보내고 Reducer로 합쳐서 충돌 없이 처리합니다 그렇게 하고 있습니다", "Pydantic으로 검증을 합니다 그런 식으로 확인합니다", "체크포인트를 씁니다 그래서 복구가 됩니다 이렇게 합니다", "골든 세트를 사용합니다 그걸로 측정해요 그렇게 합니다", "thread_id로 유지합니다 그렇게 합니다 상태를요"]
QS = ["Tool Execution 단계는 무엇을 하는 단계인가요?", "병렬 실행에서 Send와 Reducer는 어떤 역할을 하나요?", "툴 실행 결과는 어떻게 검증하나요?", "에이전트의 실행 상태는 어떻게 유지하고 복구하나요?", "평가 단계에서는 무엇을 어떻게 측정하나요?"]
PARTIALS = ["Tool Execution은 모델이 요청한 도구 호출을 실제 함수로 실행해서 그 결과를 에이전트에 돌려주는 단계라고 알고 있습니다 그리고 병렬로 돌릴 때는",
            "병렬 실행에서는 Send로 작업을 여러 노드에 나눠 보내고 Reducer가 같은 키로 들어온 결과를 합쳐서 충돌이 나지 않게 처리하는 걸로 이해하고 있습니다",
            "툴 실행 결과는 JSON Schema와 Pydantic 모델로 검증을 하고 형식이 맞지 않으면 오류 계약으로 처리해서 에이전트가 다시 판단하도록 합니다",
            "에이전트의 실행 상태는 thread_id와 state 키로 유지하고 중간에 실패하면 체크포인트로 복구할 수 있게 해 둡니다 그래서 처음부터 다시 하지 않습니다",
            "평가는 벤치마크와 골든 세트로 정확도와 비용 그리고 지연시간을 같이 측정하고 기준에 미달하면 배포를 막는 게이트를 둡니다"]

async def main(n=10):
    from ..cases import case_b
    from ..core.passage import apply_passage
    opts = {**BASE_OPTS, "tts": "supertonic:F1"}
    make_engine, make_runtime, info = await asyncio.to_thread(case_b.build, opts)
    eng = make_engine(); eng.keep_going = True; await apply_passage(eng, PASSAGE)
    rt = make_runtime(eng, lambda ev: None, lambda pcm, m: None)
    q0 = eng.first_question(); print("첫 질문:", q0.text, flush=True)
    gen, ok, tts = [], 0, []
    for i in range(n):
        from ..contracts import Question, Action
        p = (SHORT if i % 2 else PARTIALS)[(i // 2) % 5]; eng.spec_fu = None; eng.used_sents = set(); eng.last_fu_type = None; eng.cur = Question(QS[i % len(QS)], Action.DEEPEN, None, 1.0, 'x', ''); eng.asked_texts = [eng.cur.text]
        t = time.perf_counter(); await eng._spec(p, eng.cur); dt = time.perf_counter() - t
        fu = getattr(eng, "spec_fu", None); gen.append(dt); ok += bool(fu)
        if fu:
            t = time.perf_counter(); r = rt.tts.synth(fu["q"] + str(i % 2 == 0 and "" or "")) ; r = await r if inspect.isawaitable(r) else r; tts.append(time.perf_counter() - t)
        print(f"  {i+1:2d}. 질문 쓰기 {dt:.2f}s  {'성공 ['+fu['type']+']' if fu else '실패(버려짐)'}  음성 {tts[-1]:.2f}s  | {fu['q'] if fu else ''}", flush=True) if fu else print(f"  {i+1:2d}. 질문 쓰기 {dt:.2f}s 실패", flush=True)
    clips, np_ = await _clips(); w = {}
    base = np_.concatenate([c[2] for c in clips if c[0] == "normal"]); sr = 16000
    for sec in (3, 5, 8):
        seg = base[: int(sec * sr)] if len(base) >= sec * sr else np_.concatenate([base, base])[: int(sec * sr)]
        ts = []
        for _ in range(4): t = time.perf_counter(); rt.stt.transcribe(seg); ts.append(time.perf_counter() - t)
        w[sec] = st.median(ts[1:])
    med = lambda x: round(st.median(x), 2) if x else None
    out = {"n": n, "ok": ok, "gen_median": med(gen), "gen_max": round(max(gen), 2), "tts_median": med(tts), "tts_max": round(max(tts), 2) if tts else None, "whisper_s": {k: round(v, 2) for k, v in w.items()}}
    print(json.dumps(out, ensure_ascii=False)); 
    p = pathlib.Path(__file__).resolve().parents[2] / "results" / f"spec_timing_{time.strftime('%Y%m%d-%H%M')}.json"; p.write_text(json.dumps({**out, "gen": gen, "tts": tts}, ensure_ascii=False, indent=1)); print("저장:", p)
if __name__ == "__main__": asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
