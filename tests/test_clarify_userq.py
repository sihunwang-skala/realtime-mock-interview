import asyncio
from qa_lab.contracts import Action, Assessment, Question
from qa_lab.core import clarify
from qa_lab.core.engine import QAEngine
from qa_lab.core.llm import ScriptedLLM
from qa_lab.core.user_questions import add_converted, convert, quick_convert, USER_PRIORITY
from qa_lab.data.rfp import REQS

def run(c): return asyncio.run(c)

def test_classify():
    assert clarify.classify("다시 한 번 말씀해 주세요") == "repeat"
    assert clarify.classify("죄송한데 잘 못 들었어요") == "repeat"
    assert clarify.classify("너무 빨라서요 천천히 다시 말씀해 주세요") == "repeat"
    assert clarify.classify("조금 더 쉽게 말씀해 주시겠어요") == "easier"
    assert clarify.classify("질문이 이해가 안 돼요") == "easier"
    assert clarify.classify("더 쉽게 다시 말씀해 주세요") == "easier"           # both: easier wins
    long_answer = "캐시를 쓰면 쉽게 확장할 수 있고 응답시간도 줄어들며 운영 비용도 낮아진다고 생각합니다. 그래서 이 구조를 선택했습니다."
    assert clarify.classify(long_answer) is None                                 # a real answer that merely contains a trigger word
    assert clarify.classify("") is None

def _engine(text="쉬운 말로 바꾼 질문입니다 어떻게 보장하나요?"):
    llm = ScriptedLLM(lambda s, u: {}, text=text)
    e = QAEngine(llm, REQS); q = Question("응답시간 2초를 어떻게 보장하나요 레이턴시 목표", priority=1, status="asked"); e.queue.items.append(q); e._asked(q)
    return e, q

def _collect(e, text):
    async def go():
        out = ""
        async for act, tok in e.plan_and_stream(text, Assessment()):
            assert act == Action.REPEAT; out += tok
        return out
    return run(go())

def test_repeat_request_replays_same_question_and_keeps_state():
    e, q = _engine()
    out = _collect(e, "다시 한 번 말씀해 주세요")
    assert out.strip() == q.text and e.last_clarify == "repeat"
    assert e.cur is q and e.asked_texts == [q.text]                              # nothing new was "asked"

def test_easier_request_is_not_accommodated_same_question_no_llm_call():
    e, q = _engine()
    out = _collect(e, "더 쉽게 말씀해 주세요")
    assert out.strip() == q.text and e.last_clarify == "easier"                  # stance kept: same wording, no hint, no easier version
    assert e.cur is q and e.asked_texts == [q.text]
    assert e.fast.calls == []                                                    # no LLM call: fast and nothing to leak

def test_user_questions_converted_queued_first_and_deduped():
    llm = ScriptedLLM(lambda s, u: {}, text="장애가 나면 어떻게 복구하나요?")
    e = QAEngine(llm, REQS)
    e.queue.add(Question("보안 로그는 얼마나 보관하나요 보관 기간", priority=0.9, source="bank"))
    item = run(convert(llm, "장애 복구", "장애가 나면 자동 복구 절차를 실행한다."))               # LLM runs here, when the user adds it (before the practice)
    assert item["q"] == "장애가 나면 어떻게 복구하나요?" and item["raw"] == "장애 복구"
    calls = len(llm.calls)
    res = add_converted(e, [item, {"raw": "", "q": ""}])                                       # starting the session: no LLM call
    assert [r["status"] for r in res] == ["added"] and len(llm.calls) == calls
    top = e.queue.pending()[0]
    assert top.source == "user" and top.priority == USER_PRIORITY and top.user_text == "장애 복구"
    res2 = add_converted(e, [{"raw": "장애 복구 방법", "q": "장애가 나면 어떻게 복구하나요?", "ref": ""}])      # same question again: merged into the pending one, not asked twice
    assert res2[0]["status"] in ("merged", "duplicate") and len([x for x in e.queue.items if x.source == "user"]) == 1

def test_user_question_same_as_already_asked_is_dropped():
    e = QAEngine(ScriptedLLM(), REQS); q = Question("장애가 나면 어떻게 복구하나요?", status="asked"); e.queue.items.append(q); e._asked(q)
    res = add_converted(e, [{"raw": "장애 복구", "q": "장애가 나면 어떻게 복구하나요?", "ref": ""}])
    assert res[0]["status"] == "duplicate" and not e.queue.pending()

def test_quick_convert_needs_no_llm():
    assert quick_convert("응답시간은 어떻게 보장해?") == "응답시간은 어떻게 보장해?"
    assert quick_convert("장애 복구") == "장애 복구에 대해 제안서의 근거를 말씀해 주시겠어요?"
    assert quick_convert("   ") == ""
