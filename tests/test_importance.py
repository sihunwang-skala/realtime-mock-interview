"""Priority plan: which questions matter most is decided from the materials, with a visible reason."""
from qa_lab.core.importance import build_plan, planned, reference_text, split_materials, criteria, add_plan, josa, TIER_BASE

MATERIALS = """[문서: library_rfp.pdf]
평가 항목 배점
사업 이해도 20점
기술 구현 방안 40점
시범 운영 중에도 오프라인 대출은 반드시 유지해야 합니다.

[제안서: proposal.docx]
총사업비는 3억 2천만원입니다.
시범 이용자 인원은 120명입니다.
독서 완료율 목표는 35% 향상입니다.

[발표 전사: talk.srt]
총사업비는 3억 2천만원이고 인건비 산출 기준으로 계산했습니다.
시범 이용자 인원은 100명으로 모집합니다.
"""

def test_materials_and_criteria():
    mats = split_materials(MATERIALS)
    assert [m.type for m in mats] == ["RFP", "제안서", "발표 전사"]
    assert dict(criteria(mats)) == {"사업 이해도": 20, "기술 구현 방안": 40}

def test_transcript_is_not_part_of_the_reference_text():
    assert "100명" not in reference_text(MATERIALS) and "120명" in reference_text(MATERIALS)
    assert reference_text("머리말 없는 일반 글입니다.") == "머리말 없는 일반 글입니다."
    assert planned("총사업비는 3억원입니다.") == []            # plain pasted text keeps the old behaviour

def test_order_mismatch_then_missing_then_deeper():
    plan = build_plan(MATERIALS)
    assert plan[0].tier == 1 and "120명" in plan[0].question and "100명으로" in plan[0].question
    tiers = [x.tier for x in plan]
    assert tiers == sorted(tiers)
    budget = next(x for x in plan if x.category == "예산")
    assert budget.tier == 3 and "산출 근거" not in budget.question        # already explained: asked one step deeper, not repeated
    assert any(x.tier == 2 and "35%" in x.question for x in plan)

def test_same_money_written_differently_is_not_a_conflict():
    text = "[제안서: a]\n총사업비는 1,200만원입니다.\n\n[발표 전사: t]\n총사업비는 12,000,000원이고 산출 근거가 있습니다.\n"
    assert all(x.tier != 1 for x in build_plan(text))

def test_weight_orders_within_a_tier_but_never_crosses_it():
    plan = build_plan(MATERIALS)
    crit = [x for x in plan if x.tier == 5]
    assert [x.weight for x in crit] == sorted((x.weight for x in crit), reverse=True)
    for x in plan: assert TIER_BASE[x.tier] - 0.001 < x.priority < TIER_BASE[x.tier] + 0.0099
    assert [x.priority for x in plan] == sorted((x.priority for x in plan), reverse=True)

def test_planned_questions_are_all_asked_in_order_before_bank_questions():
    from qa_lab.core.engine import QAEngine
    from qa_lab.core.llm import ScriptedLLM
    from qa_lab.contracts import Question, Action
    from qa_lab.data.rfp import REQS
    eng = QAEngine(ScriptedLLM(), REQS, strategy="two_track", interjection="off")
    eng._bank = {"qs": [{"q": "재현성을 어떻게 확인하나요?"}, {"q": "서버 장애가 나면 복구 절차가 어떻게 됩니까?"}]}; eng._bank_i = 0; eng.sync_bank()
    eng.queue.add(Question("홍보 전략을 알려 주시겠어요?", Action.SWITCH, None, 0.99, "user", ""))
    plan = build_plan(MATERIALS); n = add_plan(eng, plan)
    order = []
    while (q := eng._pop_fresh()): eng._asked(q); order.append(q)
    assert n == len(plan)
    assert [q.source for q in order] == ["bank", "user"] + ["plan"] * n + ["bank"]        # opening, user, every planned one (none dropped as a "duplicate"), then the rest
    assert [q.text for q in order if q.source == "plan"] == [x.question for x in plan]

def test_josa():
    assert josa("120명", "으로/로") == "으로" and josa("35%", "은/는") == "는" and josa("1", "으로/로") == "로"

def test_point_explained_in_an_answer_is_not_asked_for_again():
    from qa_lab.core.engine import QAEngine
    from qa_lab.core.llm import ScriptedLLM
    from qa_lab.core.importance import revise_plan
    from qa_lab.data.rfp import REQS
    def fresh():
        e = QAEngine(ScriptedLLM(), REQS, strategy="two_track", interjection="off"); add_plan(e, build_plan(MATERIALS)); return e
    eng = fresh(); target = next(q for q in eng.queue.pending() if "35%" in q.text); before = target.text
    assert revise_plan(eng, "다른 이야기를 하겠습니다.") == 0 and target.text == before
    revise_plan(eng, "독서 완료율 목표 35% 향상은 지난 시범 분석 결과를 바탕으로 산출했습니다.")
    assert target.text != before and "측정" in target.text and target.plan.tier == 3 and target.status == "pending"      # core: go deeper
    eng = fresh(); crit = next(q for q in eng.queue.pending() if "기술 구현 방안" in q.text)
    revise_plan(eng, "기술 구현 방안은 기존 대출 시스템 분석을 바탕으로 정했습니다.")
    assert crit.status == "resolved"                                                                                    # general item: dropped
    eng = fresh(); mism = next(q for q in eng.queue.pending() if q.plan.tier == 1)
    revise_plan(eng, "시범 이용자 인원은 100명으로 분석해서 정했습니다.")
    assert mism.status == "pending" and mism.plan.tier == 1                                                             # a mismatch still needs confirmation
