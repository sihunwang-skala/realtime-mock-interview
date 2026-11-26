"""Agent proactive interjection policies. I1 rule, I2 LLM, I3 hybrid (rule trigger -> LLM confirm)."""
from __future__ import annotations
import re
from dataclasses import dataclass
from ..contracts import Requirement
from .answer_analyzer import repetition_ratio, numbers
from .llm import LLM
from .quantities import parse
from .question_queue import similarity

@dataclass
class Decision:
    interject: bool; reason: str = ""; kind: str = ""; policy: str = ""

def _unit(s): return parse(s)[0]
def _num(s): return parse(s)[1]
def _kw(t): return {(w[:2] if len(w) >= 3 else w) for w in re.findall(r"[가-힣A-Za-z]{2,}", t)}   # stem-ish: Korean particles differ (로그는/로그를)

def numeric_conflict(text: str, reqs: list[Requirement]) -> str | None:
    """Rule: answer states a number with the same unit and a shared keyword as an RFP requirement, but a different value."""
    for sent in re.split(r"[.?!\n]", text):
        for n in numbers(sent):
            for r in reqs:
                for rn in numbers(r.text):
                    if _unit(n) == _unit(rn) and _num(n) != _num(rn) and len(_kw(sent) & _kw(r.text)) >= 2:
                        lim = r.text
                        if "이내" in lim and _num(n) > _num(rn) or "이상" in lim and _num(n) < _num(rn) or "이내" not in lim and "이상" not in lim:
                            return f"RFP({r.id}) '{rn}' vs 답변 '{n}'"
    return None

class RulePolicy:
    name = "I1_rule"
    def __init__(self, reqs, max_secs=75, rep_th=0.3, drift_th=0.04, min_secs=3, off_chars=60):     # off_chars: 90 -> 60 (the 14 s transcription window holds only ~75-93 chars at normal speed, so 90 was out of reach). All thresholds here are starting values, not research-based: tune with the 10-pair false-positive / catch rates.
        self.off_chars = off_chars; self.reqs = reqs; self.max_secs = max_secs; self.rep_th = rep_th; self.drift_th = drift_th; self.min_secs = min_secs
    def triggers(self, partial: str, question: str, elapsed: float, prior_answers: list[str]) -> Decision:
        if elapsed < self.min_secs: return Decision(False, "too early", policy=self.name)
        c = numeric_conflict(partial, self.reqs)
        if c: return Decision(True, c, "contradiction_rfp", self.name)
        for p in prior_answers:
            a, b = numbers(partial), numbers(p)
            for x in a:
                for y in b:
                    if _unit(x) == _unit(y) and _num(x) != _num(y) and len(_kw(partial) & _kw(p)) >= 3:
                        return Decision(True, f"이전 답변 '{y}' vs 지금 '{x}'", "contradiction_prior", self.name)
        if repetition_ratio(partial) > self.rep_th: return Decision(True, "같은 설명 반복", "repetition", self.name)
        if len(partial) > self.off_chars and len(_kw(partial) & _kw(question)) / max(len(_kw(question)), 1) < 0.2 and elapsed > 8:
            return Decision(True, "질문과 무관한 설명 지속", "off_topic", self.name)
        if elapsed > self.max_secs: return Decision(True, f"{self.max_secs}s 초과", "overlong", self.name)
        return Decision(False, "", policy=self.name)
    async def decide(self, *a, **k): return self.triggers(*a, **k)

JSYS = """당신은 평가위원이 발표자의 답변 도중 끼어들어야 하는지 판단한다. JSON만: {"interject":bool,"kind":"contradiction_rfp|contradiction_prior|off_topic|repetition|unsupported|none","reason":"한 문장"}.
끼어들기는 명백한 모순/RFP 위반 주장/장시간 주제 이탈/반복일 때만 true. 말이 길어도 질문에 답하고 있으면 false. 판단이 애매하면 false."""

class LLMPolicy:
    name = "I2_llm"
    def __init__(self, llm: LLM, reqs, min_secs=3):
        self.llm = llm; self.reqs = reqs; self.min_secs = min_secs
    async def decide(self, partial, question, elapsed, prior_answers) -> Decision:
        if elapsed < self.min_secs: return Decision(False, "too early", policy=self.name)
        r = "\n".join(f"{x.id}: {x.text}" for x in self.reqs)
        d = await self.llm.json(JSYS, f"[RFP]\n{r}\n[이전 답변]\n{' / '.join(prior_answers)[-500:] or '없음'}\n[질문]\n{question}\n[진행 중인 답변 ({int(elapsed)}초)]\n{partial[-700:]}", 120)
        return Decision(bool(d.get("interject")), str(d.get("reason", "")), str(d.get("kind", "")), self.name)

class HybridPolicy:
    name = "I3_hybrid"
    def __init__(self, llm, reqs, **kw):
        self.rule = RulePolicy(reqs, **kw); self.llm = LLMPolicy(llm, reqs, min_secs=kw.get("min_secs", 3)); self.rule.name = "I3_hybrid"
    async def decide(self, partial, question, elapsed, prior_answers) -> Decision:
        t = self.rule.triggers(partial, question, elapsed, prior_answers)
        if not t.interject: return Decision(False, "no trigger", policy=self.name)
        if t.kind in ("contradiction_rfp", "contradiction_prior"):   # hard evidence -> no LLM needed
            t.policy = self.name; return t
        v = await self.llm.decide(partial, question, max(elapsed, self.llm.min_secs), prior_answers)
        v.policy = self.name; return v

class OffPolicy:
    name = "off"
    async def decide(self, *a, **k): return Decision(False, "disabled", policy="off")

def make_policy(name: str, llm, reqs):
    return {"off": lambda: OffPolicy(), "rule": lambda: RulePolicy(reqs), "llm": lambda: LLMPolicy(llm, reqs), "hybrid": lambda: HybridPolicy(llm, reqs)}[name]()
