"""Incremental answer analysis (fast model). Heuristic prefilter + LLM structured assessment."""
from __future__ import annotations
import re
from ..contracts import Assessment, Requirement
from .llm import LLM

SYS = """당신은 RFP 제안 평가위원의 보조 분석기다. 질문, RFP 요구사항, 이전 답변 요약, 지금까지의 답변을 보고 JSON만 출력한다.
키: sufficiency(0~1 질문에 대한 충족도), contradictions(문자열 배열: RFP 또는 이전 답변과 충돌하는 주장), off_topic(0~1),
repetitive(bool), unsupported_claims(문자열 배열: 근거나 수치 없이 단정한 주장), covered_reqs(충족한 요구사항 id 배열), summary(한 문장).
충돌은 명백할 때만 적는다. 한국어로."""

def repetition_ratio(text: str) -> float:
    toks = re.findall(r"[가-힣A-Za-z0-9]+", text)
    if len(toks) < 12: return 0.0
    grams = [" ".join(toks[i:i+3]) for i in range(len(toks) - 2)]
    return 1 - len(set(grams)) / len(grams)

from .quantities import numbers          # numbers with units ("1,200만원", "32억 5천만 원"), see core/quantities.py

class AnswerAnalyzer:
    def __init__(self, llm: LLM, reqs: list[Requirement]):
        self.llm = llm; self.reqs = reqs
    def heuristics(self, answer: str) -> dict:
        return {"repetition": round(repetition_ratio(answer), 2), "words": len(answer.split()), "numbers": sorted(numbers(answer))}
    async def analyze(self, question: str, answer: str, history: str = "") -> Assessment:
        reqs = "\n".join(f"{r.id}: {r.text}" for r in self.reqs)
        user = f"[요구사항]\n{reqs}\n\n[이전 답변 요약]\n{history or '없음'}\n\n[질문]\n{question}\n\n[답변]\n{answer}"
        d = await self.llm.json(SYS, user, 350)
        h = self.heuristics(answer)
        def f(k, default=0.0):
            try: return float(d.get(k, default))
            except Exception: return default
        return Assessment(sufficiency=max(0, min(1, f("sufficiency"))), contradictions=[str(x) for x in d.get("contradictions", []) or []],
            off_topic=max(0, min(1, f("off_topic"))), repetitive=bool(d.get("repetitive")) or h["repetition"] > 0.3,
            unsupported_claims=[str(x) for x in d.get("unsupported_claims", []) or []],
            covered_reqs=[str(x) for x in d.get("covered_reqs", []) or []], summary=str(d.get("summary", "")))
