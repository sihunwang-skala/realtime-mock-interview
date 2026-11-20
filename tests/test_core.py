import asyncio, pytest
from qa_lab.contracts import Action, Question, Assessment
from qa_lab.core.question_queue import QuestionQueue, similarity
from qa_lab.core.coverage import Coverage
from qa_lab.core.planner import decide_action
from qa_lab.core.interjection import RulePolicy, HybridPolicy, numeric_conflict
from qa_lab.core.llm import ScriptedLLM, make_llm, parse_json
from qa_lab.core.engine import QAEngine
from qa_lab.data.rfp import REQS

def run(c): return asyncio.run(c)

def test_dedup_and_priority():
    q = QuestionQueue(); a = q.add(Question("응답시간 2초를 어떻게 보장하나요?", priority=.4))
    assert a and q.add(Question("응답시간 2초를 어떻게 보장합니까?", priority=.9)) is None
    assert q.pending()[0].priority == .9
    assert q.add(Question("보안 로그는 얼마나 보관하나요?")) is not None
    assert similarity("가나다라", "마바사아") == 0

def test_resolve_and_pop_and_bound():
    q = QuestionQueue(max_pending=3)
    q.add(Question("질문 하나입니다 알파", req_id="R1", priority=.5)); q.add(Question("완전히 다른 베타 감마", req_id="R2", priority=.9))
    assert q.resolve_req("R1") == 1
    assert q.pop().req_id == "R2" and q.pop() is None
    for i in range(6): q.add(Question(f"서로 다른 주제 {chr(0xAC00+i*37)}{chr(0xB000+i*91)}{chr(0xC000+i*13)} 질문", priority=i/10))
    assert len(q.pending()) <= 3

def test_coverage_monotonic():
    c = Coverage(REQS); c.update("R1", .8, "x", 1); c.update("R1", .2, "y", 2)
    assert c.score["R1"] == .8 and "R1" not in c.gaps() and "R2" in c.gaps()

def test_planner_transitions():
    A = Assessment
    assert decide_action(A(contradictions=["x"]), 0, True) == Action.CHALLENGE
    assert decide_action(A(sufficiency=.2), 0, True) == Action.CLARIFY
    assert decide_action(A(sufficiency=.6), 0, True) == Action.DEEPEN
    assert decide_action(A(sufficiency=.6), 2, True) == Action.SWITCH      # follow-up cap
    assert decide_action(A(sufficiency=.9), 0, False) == Action.END
    assert decide_action(A(sufficiency=.9), 0, True, elapsed_ratio=1.1) == Action.END
    assert decide_action(A(off_topic=.9), 0, True) == Action.CLARIFY
    assert decide_action(A(sufficiency=.9, unsupported_claims=["z"]), 0, True) == Action.VERIFY

def test_numeric_conflict_rule():
    assert numeric_conflict("민원 접수 API 응답시간은 평균 5초 정도 걸립니다", REQS)
    assert not numeric_conflict("민원 접수 API 응답시간은 평균 1.2초입니다", REQS)
    assert numeric_conflict("구축 기간은 12개월로 예상합니다", REQS)

def test_interjection_rule_policy():
    p = RulePolicy(REQS)
    assert not run(p.decide("5초 걸립니다 응답시간 민원 접수", "q", 2.5, [])).interject   # too early
    d = run(p.decide("민원 접수 API 응답시간은 평균 5초 정도입니다", "q", 20, []))
    assert d.interject and d.kind == "contradiction_rfp"
    d = run(p.decide("계속 말씀드리면 " + "같은 말을 반복합니다 " * 12, "q", 20, []))
    assert d.interject and d.kind == "repetition"
    assert not run(p.decide("캐시와 오토스케일링으로 부하를 분산합니다", "캐시 질문", 20, [])).interject
    d = run(p.decide("그리고 구축 기간은 10개월 걸릴 겁니다 구축 기간 계약", "q", 20, ["구축 기간은 6개월입니다 구축 기간 계약"]))
    assert d.interject

def test_hybrid_uses_llm_only_when_triggered():
    llm = ScriptedLLM(lambda s, u: {"interject": False, "kind": "none"})
    h = HybridPolicy(llm, REQS)
    assert not run(h.decide("정상적인 설명입니다", "q", 30, [])).interject and not llm.calls
    d = run(h.decide("민원 접수 API 응답시간은 평균 5초 정도입니다", "q", 30, []))
    assert d.interject and not llm.calls                                   # hard evidence skips LLM
    d = run(h.decide("같은 말 " * 40, "q", 30, []))
    assert llm.calls and not d.interject                                    # LLM vetoes soft trigger

def test_engine_phase_a_resolution_and_dedup():
    resp = iter([
        {"covered": [], "questions": [{"text": "응답시간 2초를 어떻게 보장합니까?", "req": "R1", "priority": .8}]},
        {"covered": [], "questions": [{"text": "응답시간 2초를 어떻게 보장하죠?", "req": "R1", "priority": .6}]},
        {"covered": [{"req": "R1", "score": .9, "quote": "캐시로 800ms"}], "questions": []}])
    e = QAEngine(ScriptedLLM(lambda s, u: next(resp)), REQS)
    r1 = run(e.ingest_segment("a")); r2 = run(e.ingest_segment("b"))
    assert r1["pending"] == 1 and r2["pending"] == 1            # duplicate suppressed
    r3 = run(e.ingest_segment("c"))
    assert r3["resolved"] == ["R1"] and r3["pending"] == 0     # later explanation resolves question

def test_engine_phase_b_followup_and_heard_tracking():
    llm = ScriptedLLM(lambda s, u: {"sufficiency": .3, "contradictions": [], "off_topic": 0, "covered_reqs": [], "summary": "s"}, text="구체적인 수치는 얼마인가요?")
    e = QAEngine(llm, REQS); e.queue.add(Question("첫 질문입니다 알파벳", priority=1)); q = e.first_question()
    async def go():
        a = await e.analyze_answer("잘 하겠습니다")
        out = [(act, t) async for act, t in e.plan_and_stream("잘 하겠습니다", a)]
        return a, out
    a, out = run(go())
    assert out[0][0] == Action.CLARIFY and "수치" in "".join(t for _, t in out)
    e.mark_played(5); assert len(e.heard_text()) == 5 and len(e.spoken_full) > 5   # unplayed part not "heard"

def test_parse_json_and_factory():
    assert parse_json('<think>x</think>{"a":1}') == {"a": 1} and parse_json("junk") == {}
    assert make_llm("ollama:qwen2.5:7b").model == "qwen2.5:7b"
    with pytest.raises(ValueError): make_llm("nope:x")

def test_duration_unit_conversion_and_stem_match():
    msg = "개인정보 접근 로그는 비용 문제로 3개월만 보관하려고 합니다"
    assert numeric_conflict(msg, REQS) and not numeric_conflict("개인정보 접근 로그는 2년 보관합니다", REQS)

def test_repeat_request_and_fast_assess():
    from qa_lab.core.planner import is_repeat_request
    assert is_repeat_request("죄송한데 질문을 다시 한번 말씀해 주시겠어요?") and not is_repeat_request("캐시를 사용해서 질문하신 응답시간을 다시 계산해 보면 1.2초이고 그 근거는 길게 설명됩니다 " * 2)
    e = QAEngine(ScriptedLLM(), REQS); e.queue.add(Question("응답시간 질문입니다", priority=1)); e.first_question()
    a = e.fast_assess("민원 접수 API 응답시간은 평균 5초입니다")
    assert a.contradictions and decide_action(a, 0, True) == Action.CHALLENGE
