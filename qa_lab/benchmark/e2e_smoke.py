"""End-to-end smoke test (2 turns): the question EVENT, the saved turn record and the follow-up marks must all exist. Run after every change to runtime.py / engine.py:  python -m qa_lab.benchmark.e2e_smoke
(Added after a bug where the question was spoken but the screen event and the turn record were never produced: latency-only tests did not notice.)"""
import asyncio, time
import numpy as np
from qa_lab.benchmark.latency_ab import BASE_OPTS, PASSAGE
async def main():
    from qa_lab.cases import case_b
    from qa_lab.core.passage import apply_passage
    from qa_lab.providers.tts import AVSpeechTTS
    mk, mr, _ = await asyncio.to_thread(case_b.build, {**BASE_OPTS, "tts": "supertonic:F1"})
    eng = mk(); eng.keep_going = True; await apply_passage(eng, PASSAGE)
    evs = []; rt = mr(eng, lambda ev: evs.append(ev.to_dict()), lambda pcm, m: None)
    y = AVSpeechTTS(); y.warm()
    answers = ["Tool Execution은 모델이 요청한 도구 호출을 실제 함수로 실행하고 그 결과를 에이전트에 돌려주는 단계입니다 병렬 실행에서는 Send로 작업을 나눕니다", "잘 모르겠습니다"]
    q0 = eng.first_question(); await rt.speak_text(q0.text); await asyncio.sleep(3)
    for k, a in enumerate(answers, 1):
        clip = await y.synth(a); evs.clear()
        await rt.feed(np.zeros(int(1.4 * 16000), np.float32))
        for j in range(0, len(clip), 1600): await rt.feed(clip[j:j + 1600]); await asyncio.sleep(0.1)
        ok, why = await rt.finish_answer(); t = rt.agent_task
        try: await asyncio.wait_for(t, 20)
        except Exception as e: print("  !! 턴 예외:", repr(e)[:120])
        qs = [e for e in evs if e.get("type") == "QUESTION_STARTED"]
        d = (qs[0].get("data") or {}) if qs else {}
        print(f"턴 {k}: finish={ok} | 질문 이벤트 {len(qs)}개 | 기록 {len(rt.responses)}개 | 꼬리질문={d.get('followup')} 유형={d.get('fu_type')} 깊이={d.get('fu_depth')}/{d.get('fu_max')} | 질문: {(d.get('text') or '')[:60]}")
        print("     기록 note:", (rt.responses[-1].get("fu_note") if rt.responses else None))
        await asyncio.sleep(4)
    try: y.p.terminate()
    except Exception: pass
asyncio.run(main())
