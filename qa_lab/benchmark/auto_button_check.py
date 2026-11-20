"""Automatic-detection mode: what [답변 끝] does right after the speech ended / after it was auto-confirmed / with no speech.  python -m qa_lab.benchmark.auto_button_check"""
import asyncio, time
import numpy as np
from qa_lab.benchmark.latency_ab import BASE_OPTS, PASSAGE
async def scenario(name, silence_s, presses):
    from qa_lab.cases import case_b
    from qa_lab.core.passage import apply_passage
    from qa_lab.providers.tts import AVSpeechTTS
    mk, mr, _ = await asyncio.to_thread(case_b.build, {**BASE_OPTS, "turn_mode": "auto", "confirm_ms": 1500, "silence_ms": 600, "tts": "supertonic:F1"})
    eng = mk(); eng.keep_going = True; await apply_passage(eng, PASSAGE)
    evs = []; rt = mr(eng, lambda ev: evs.append((time.monotonic(), ev.to_dict())), lambda pcm, m: None)
    y = AVSpeechTTS(); y.warm(); clip = await y.synth("아 그건 잘 모르겠습니다")
    q0 = eng.first_question(); await rt.speak_text(q0.text); await asyncio.sleep(3); evs.clear()
    if clip is not None and presses != "nospeech":
        await rt.feed(np.zeros(int(1.4*16000), np.float32))
        for j in range(0, len(clip), 1600): await rt.feed(clip[j:j+1600]); await asyncio.sleep(0.1)
        t_end = time.monotonic(); tail = np.zeros(1600, np.float32)
        for _ in range(int(silence_s * 10)): await rt.feed(tail); await asyncio.sleep(0.1)
    res = []
    for k in range(presses if isinstance(presses, int) else 1):
        t = time.monotonic(); r = await rt.force_end(); res.append((r, round(time.monotonic() - t, 3))); await asyncio.sleep(0.5)
    for _ in range(80):
        if any(e[1].get("type") == "QUESTION_STARTED" for e in evs): break
        await asyncio.sleep(0.1)
    got = next((e for e in evs if e[1].get("type") == "QUESTION_STARTED"), None)
    print(f"[{name}] 버튼 결과 {res} | 질문 이벤트: {'있음' if got else '없음'} | 기록 {len(rt.responses)}개")
    try: y.p.terminate()
    except Exception: pass
async def main():
    await scenario("말 끝난 직후 0.3초에 누름(준비 중)", 0.3, 1)
    await asyncio.sleep(20)
    await scenario("말 끝난 2.5초 뒤 누름(이미 자동 확정)", 2.5, 2)
    await asyncio.sleep(20)
    await scenario("아무 말 없이 누름", 0, "nospeech")
asyncio.run(main())
