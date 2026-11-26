import asyncio, numpy as np
from qa_lab.cases.runtime import VoiceRuntime
from qa_lab.contracts import Question
from qa_lab.core.engine import QAEngine
from qa_lab.core.llm import ScriptedLLM
from qa_lab.data.rfp import REQS

class FakeVAD:
    """start at frame 2, a pause (end) at frame 40, speech resumes at frame 60, then no more events."""
    def __init__(self): self.n = 0
    def feed(self, f): self.n += 1; return {2: "start", 40: "end", 60: "start"}.get(self.n)
class FakeSTT:
    def __init__(self): self.calls = []
    def transcribe(self, a): self.calls.append(len(a)); return ["첫 부분입니다.", "둘째 부분입니다."][len(self.calls) - 1] if len(self.calls) <= 2 else "더"
class FakeTTS:
    async def stream(self, text): yield np.zeros(800, np.float32)

def make():
    llm = ScriptedLLM(lambda s, u: {"sufficiency": .3}, text="구체적인 수치는 얼마인가요?")
    e = QAEngine(llm, REQS, strategy="two_track", interjection="off"); q = Question("질문입니다 알파벳", status="asked"); e.queue.items.append(q); e._asked(q)
    return VoiceRuntime(e, FakeVAD(), FakeSTT(), FakeTTS(), turn_mode="manual", enable_interjection=False)

def test_manual_turn_waits_for_button_and_merges_chunks():
    async def go():
        rt = make(); rng = np.random.default_rng(0)
        for _ in range(100): await rt.feed((rng.normal(0, .1, 512)).astype(np.float32))      # pause at frame 40 must NOT end the turn
        await asyncio.sleep(0.2)
        assert rt.responses == [] and rt.agent_task is None, "silence must not trigger a response in manual mode"
        ok, why = await rt.finish_answer(); assert ok and not why
        await asyncio.wait_for(rt.agent_task, 5)
        assert rt.responses[0]["stt_text"] == "첫 부분입니다. 둘째 부분입니다."                    # pause chunk + tail merged
        assert "BUTTON_PRESSED" in rt.responses[0]["marks"]
        ok2, why2 = await rt.finish_answer(); assert not ok2 and why2                           # nothing new was said
    asyncio.run(go())

def test_manual_finish_without_speech_is_rejected():
    rt = make(); ok, why = asyncio.run(rt.finish_answer()); assert not ok and "말씀" in why


def test_button_press_resets_the_speech_detector():
    """Pressing "answer done" before the detector heard enough silence must not leave it stuck in "speaking"."""
    from qa_lab.cases.runtime import VoiceRuntime
    class VAD:
        n = 0
        def reset(self): self.n += 1
    rt = VoiceRuntime.__new__(VoiceRuntime); rt.vad = VAD(); rt.vad_late = None
    rt._reset_vad()
    assert rt.vad.n == 1
