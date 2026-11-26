import asyncio, json
from qa_lab.contracts import Question
from qa_lab.core import ledger as LD
from qa_lab.core.engine import QAEngine
from qa_lab.core.llm import ScriptedLLM
from qa_lab.data.rfp import REQS

class LowRng:                       # always the shortest gap (2 practices) so the scenarios are deterministic
    def randint(self, a, b): return a
    def uniform(self, a, b): return a

REF = "응답시간 목표는 평균 2초 이내이며 캐시와 비동기 처리를 사용한다."
Q = "응답시간 2초 목표를 어떻게 보장하나요?"
def res(ok, q=Q, ref=REF): return [{"q": q, "ref": ref, "a": "모범 답안", "ok": ok, "verdict": "맞음" if ok else "틀림"}]
def led(tmp_path, monkeypatch):
    monkeypatch.setattr(LD, "DIR", tmp_path); return LD.Ledger("t")
def item(l): return next(iter(l.items.values()))

def test_correct_then_one_more_time_then_never(tmp_path, monkeypatch):
    l = led(tmp_path, monkeypatch); r = LowRng()
    l.apply(1, res(True), r); it = item(l)
    assert it["streak"] == 1 and not it["retired"] and it["due"] == 3            # not the next practice
    assert l.due(2) == [] and len(l.due(3)) == 1                                 # asked again in a later practice ("연습 5" in the example)
    l.apply(3, res(True), r)
    assert item(l)["retired"] and l.due(4) == [] and l.due(99) == []             # two credits: never again

def test_wrong_comes_back_every_next_practice_until_correct(tmp_path, monkeypatch):
    l = led(tmp_path, monkeypatch); r = LowRng()
    l.apply(2, res(False), r); assert item(l)["due"] == 3 and len(l.due(3)) == 1
    l.apply(3, res(False), r); assert item(l)["due"] == 4 and len(l.due(4)) == 1
    l.apply(4, res(True), r);  assert item(l)["streak"] == 1 and l.due(5) == []  # then it needs one more correct, later

def test_credit_is_lost_on_a_later_wrong_answer(tmp_path, monkeypatch):
    l = led(tmp_path, monkeypatch); r = LowRng()
    l.apply(2, res(True), r)                                                     # correct in practice 2
    l.apply(5, res(False), r)                                                    # asked again in 5, wrong
    it = item(l); assert it["streak"] == 0 and it["due"] == 6 and not it["retired"]
    l.apply(6, res(True), r); assert item(l)["streak"] == 1 and not item(l)["retired"]    # needs two NEW correct answers
    l.apply(8, res(True), r); assert item(l)["retired"]

def test_credit_text_names_the_practices(tmp_path, monkeypatch):
    l = led(tmp_path, monkeypatch); r = LowRng()
    l.apply(2, res(True), r); l.apply(5, res(False), r); out = l.apply(6, res(True), r)
    txt = next(iter(out.values()))["text"]
    assert "1번째 맞음" in txt and "연습 2" in txt and "연습 5" in txt and "연습 6" in txt

def test_inject_review_priorities_and_retired_never_asked(tmp_path, monkeypatch):
    l = led(tmp_path, monkeypatch); r = LowRng()
    l.apply(1, res(False), r)                                                    # wrong -> due in 2
    l.apply(1, res(True, "보안 로그를 얼마나 보관하나요?", "보안 로그는 1년간 보관한다 암호화 저장."), r)
    l.apply(1, res(True, "다른 질문 하나입니다 알파?", "알파 베타 감마 델타 문장."), r); l.apply(3, res(True, "다른 질문 하나입니다 알파?", "알파 베타 감마 델타 문장."), r)   # retired
    e = QAEngine(ScriptedLLM(), REQS)
    e.queue.add(Question("다른 질문 하나입니다 알파?", priority=0.95, source="bank")); e.queue.items[-1].bank = {"ref": "알파 베타 감마 델타 문장."}
    e.queue.add(Question("전혀 새로운 질문입니다 오메가 시그마?", priority=0.9, source="bank"))
    due = LD.inject(e, l, 2, r)
    assert [d["q"] for d in due] == [Q]                                          # only the wrong one is due in practice 2
    top = e.queue.pending()[0]; assert top.source == "review" and top.priority == LD.WRONG_PRIO and top.bank["a"] == "모범 답안"   # instant model answer
    assert all("알파" not in x.text for x in e.queue.pending())                  # the retired question was dropped from the queue
    assert e._blocked(Question("다른 질문 하나입니다 알파?"))

def test_collect_counts_only_llm_graded_tracked_answers():
    rep = {"turns": [
        {"stt_text": "a1", "answered_q": "Q1", "ref_text": "R1", "verdict": "맞음", "verdict_pending": False, "model_answer": "m", "question": "Q2", "q_source": "bank"},
        {"stt_text": "다시 말씀해 주세요", "clarify": "repeat", "question": "Q2"},
        {"stt_text": "a2", "answered_q": "Q2", "ref_text": "R2", "verdict": "틀림", "verdict_pending": False, "question": "F1", "q_source": "followup"},
        {"stt_text": "a3", "answered_q": "F1", "ref_text": "R2", "verdict": "맞음", "verdict_pending": False, "question": "Q3", "q_source": "bank"},
        {"stt_text": "a4", "answered_q": "Q3", "ref_text": "R3", "verdict": "맞음", "verdict_pending": True},        # quick (keyword) verdict: does not count
    ]}
    got = LD.collect(rep)
    assert [(x["q"], x["ok"]) for x in got] == [("Q1", True), ("Q2", False)]       # follow-up answer, clarify turn and ungraded answer are not tracked

def test_save_is_atomic_and_reloads(tmp_path, monkeypatch):
    l = led(tmp_path, monkeypatch); l.apply(1, res(False), LowRng()); l.save()
    l2 = LD.Ledger("t"); assert l2.practice == 1 and len(l2.items) == 1 and not list(tmp_path.glob("*.tmp"))

def test_background_grading_credits_the_ledger_and_writes_credit_text(tmp_path, monkeypatch):
    from qa_lab.demo import server as S
    from qa_lab.core import llm as L
    monkeypatch.setattr(LD, "DIR", tmp_path / "led"); monkeypatch.setattr(S, "LIVE", tmp_path / "runs"); (tmp_path / "runs").mkdir()
    monkeypatch.setattr(L, "make_llm", lambda spec: ScriptedLLM(lambda s, u: {"verdict": "맞음" if "캐시" in u else "틀림", "why": "근거"}))
    turns = [{"stt_text": "캐시와 비동기로 보장합니다", "answered_q": Q, "ref_text": REF, "verdict": "틀림", "verdict_pending": True, "model_answer": "모범", "question": "다음 질문입니다 알파벳?", "q_source": "bank"},
             {"stt_text": "잘 모르겠어요", "answered_q": "다음 질문입니다 알파벳?", "ref_text": "알파벳 문장입니다.", "verdict": "틀림", "verdict_pending": True, "model_answer": "모범2"}]
    (tmp_path / "runs" / "s1.json").write_text(json.dumps({"id": "s1", "passage_text": "p", "report": {"turns": turns, "asked": [Q], "practice": 1}}, ensure_ascii=False))
    l = LD.Ledger("t"); calls_before = 0
    asyncio.run(S._grade_and_ledger("s1", {"llm": "x"}, l, 1))
    d = json.loads((tmp_path / "runs" / "s1.json").read_text()); t0, t1 = d["report"]["turns"]
    assert t0["verdict"] == "맞음" and "verdict_pending" not in t0 and "1번째 맞음" in t0["credit"]["text"]
    assert t1["verdict"] == "모름" and t1["credit"]["ok"] is False                 # "잘 모르겠어요": no LLM call needed, counted as wrong
    l2 = LD.Ledger("t"); assert l2.practice == 1 and len(l2.items) == 2

def test_generic_question_saved_earlier_is_not_asked_again(tmp_path, monkeypatch):
    from qa_lab.core import ledger as LD
    from qa_lab.core.question_queue import QuestionQueue
    monkeypatch.setattr(LD, "DIR", tmp_path)
    led = LD.Ledger("k")
    led.apply(1, [{"q": "이 글의 핵심 내용을 본인의 말로 설명해 주시겠어요?", "ref": "", "a": "", "ok": False, "verdict": "틀림"},
                  {"q": "목표 인원 120명의 근거는 무엇인가요?", "ref": "목표 인원은 120명이다.", "a": "", "ok": False, "verdict": "틀림"}])
    class Eng:
        queue = QuestionQueue()
        def ref_of(self, q): return getattr(q, "ref_text", "")
    eng = Eng(); LD.inject(eng, led, 2)
    assert [q.text for q in eng.queue.pending()] == ["목표 인원 120명의 근거는 무엇인가요?"]
