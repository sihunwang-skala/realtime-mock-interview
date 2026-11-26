"""Session time limit: after the limit no new question; the closing line is spoken once, after silence or after the last answer."""
import asyncio
from qa_lab.cases import runtime as R
from qa_lab.contracts import EventType

class Fake(R.VoiceRuntime):
    def __init__(self):                       # bypass heavy setup: only the attributes the time-limit code uses
        self.in_speech = False; self.agent_speaking = False; self.agent_task = None; self.events = []; self.spoken = []
    def emit(self, t, **d): self.events.append((t, d.get("state")))
    async def _speak_stream(self, gen, key):
        words = [w async for _, w in gen]; self.spoken.append("".join(words)); return None, "".join(words)

def test_closes_after_silence(monkeypatch):
    monkeypatch.setattr(R, "TIME_UP_SILENCE_S", 0.4)
    async def run():
        rt = Fake(); rt.arm_time_limit(0.2)
        await asyncio.sleep(1.5)
        return rt
    rt = asyncio.run(run())
    states = [s for t, s in rt.events if t == EventType.TIME_UP]
    assert states == ["limit_reached", "closing", "closed"]
    assert len(rt.spoken) == 1 and rt.spoken[0].startswith("네~ 여기까지 들었습니다")

def test_no_close_while_someone_speaks(monkeypatch):
    monkeypatch.setattr(R, "TIME_UP_SILENCE_S", 0.4)
    async def run():
        rt = Fake(); rt.arm_time_limit(0.1); rt.in_speech = True
        await asyncio.sleep(1.0)
        return rt
    rt = asyncio.run(run())
    assert rt.spoken == [] and rt.closing_done is False
