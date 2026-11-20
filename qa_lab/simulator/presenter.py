"""Presenter LLM: generates presentation segments and answers to live questions with controllable behaviours."""
from __future__ import annotations
from ..core.llm import LLM

BEHAV = {
    "normal": "구체적 수치와 근거를 들어 성실하게 답한다.",
    "vague": "추상적으로 답하고 수치나 근거는 언급하지 않는다.",
    "contradict": "RFP 요구보다 불리한 수치(예: 응답시간 5초, 구축 12개월)를 자연스럽게 말한다.",
    "ramble": "질문과 무관한 회사 소개와 과거 실적을 길게 늘어놓는다.",
    "repeat": "같은 문장을 표현만 바꿔 계속 반복한다.",
}
SYS = "당신은 RFP 입찰 제안 발표자다. 한국어 구어체로 말한다. 지시된 방식대로만 말하고 메타 설명은 하지 않는다."

class Presenter:
    def __init__(self, llm: LLM, rfp_text: str): self.llm = llm; self.rfp = rfp_text; self.said: list[str] = []
    async def _text(self, user, n=200):
        out = ""
        async for t in self.llm.stream(SYS, user, n): out += t
        return out.strip()
    async def segment(self, topic: str, behaviour="normal"):
        s = await self._text(f"[RFP 요구사항]\n{self.rfp}\n\n제안 발표의 한 구간을 3~4문장으로 말하라. 주제: {topic}\n방식: {BEHAV[behaviour]}\n지금까지 한 말: {' '.join(self.said)[-400:]}")
        self.said.append(s); return s
    async def answer(self, question: str, behaviour="normal", n=220):
        s = await self._text(f"[RFP 요구사항]\n{self.rfp}\n\n지금까지 발표 내용: {' '.join(self.said)[-600:]}\n\n평가위원 질문: {question}\n방식: {BEHAV[behaviour]}\n3~5문장으로 답하라.", n)
        self.said.append(s); return s
