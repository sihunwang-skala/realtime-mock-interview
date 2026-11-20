"""Demo A: Presenter LLM <-> Evaluator Agent, text only, live dependency (no canned dialog)."""
from __future__ import annotations
import asyncio, json, time, sys
from ..core.engine import QAEngine
from ..core.llm import make_llm
from ..data.rfp import REQS, RFP_TEXT
from .presenter import Presenter

TOPICS = [("시스템 아키텍처와 성능 목표", "normal"), ("보안과 개인정보 보호", "vague"), ("일정과 인력 계획", "contradict"), ("AI 민원 분류 모델", "normal")]
PLAN = ["normal", "contradict", "vague", "ramble", "normal", "repeat"]

async def partial_stream(text: str, wps=2.5, chunk=6):
    """Simulate live partial transcripts: yields (partial_text, elapsed_s) every `chunk` words."""
    w = text.split()
    for i in range(chunk, len(w) + chunk, chunk): yield " ".join(w[:i]), i / wps

async def run(fast="ollama:qwen2.5:7b", presenter="ollama:qwen2.5:7b", strategy="hybrid", interjection="hybrid", turns=5, out=None, seed_wait=0):
    f, p = make_llm(fast), make_llm(presenter)
    for m in (f, p):
        if hasattr(m, "warm"): await m.warm()
    eng = QAEngine(f, REQS, strategy=strategy, interjection=interjection); pres = Presenter(p, RFP_TEXT)
    log = {"config": dict(fast=fast, presenter=presenter, strategy=strategy, interjection=interjection), "phaseA": [], "phaseB": []}
    for i, (topic, beh) in enumerate(TOPICS):
        seg = await pres.segment(topic, beh); r = await eng.ingest_segment(seg, ts=i * 450.0)
        log["phaseA"].append({"topic": topic, "behaviour": beh, "segment": seg, **r})
    q = eng.first_question()
    for t in range(turns):
        if not q: break
        beh = PLAN[t % len(PLAN)]; qtext = eng.cur.text
        ans = await pres.answer(qtext, beh); interj = None; shown = ans
        async for part, el in partial_stream(ans):
            d = await eng.should_interject(part, max(el, 20))   # elapsed floor: sim answers are short text, treat as >=20s speech
            if d.interject: interj = {"at_words": len(part.split()), "kind": d.kind, "reason": d.reason}; shown = part; break
        a = await eng.analyze_answer(shown); t0 = time.monotonic(); first = None; nq = ""; act = None
        async for act, tok in eng.plan_and_stream(shown, a):
            if first is None: first = time.monotonic() - t0
            nq += tok
        log["phaseB"].append({"turn": t, "behaviour": beh, "question": qtext, "answer": ans, "interjection": interj, "assessment": a.__dict__,
                              "action": act.value if act else None, "next_question": nq.strip(), "first_token_s": round(first or 0, 3)})
        if act and act.value == "END": break
        q = eng.cur
    log["report"] = eng.report()
    if out: open(out, "w").write(json.dumps(log, ensure_ascii=False, indent=2))
    return log

if __name__ == "__main__":
    lg = asyncio.run(run(out=sys.argv[1] if len(sys.argv) > 1 else None))
    for x in lg["phaseA"]: print("A:", x["topic"], "pending", x["pending"], "resolved", x["resolved"])
    for x in lg["phaseB"]: print(f"B{x['turn']} [{x['behaviour']}] Q: {x['question']}\n   A: {x['answer'][:90]}...\n   interj={x['interjection']} -> {x['action']}: {x['next_question']} ({x['first_token_s']}s)")
