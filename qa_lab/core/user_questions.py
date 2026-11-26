"""Questions the user wants to be asked ("내가 추가하고 싶은 질문").

The user may type them roughly (a keyword, a half sentence, a full question). Each one is turned into a proper spoken question
(polite, one sentence) and put into the live queue ahead of the randomly generated bank questions.
Speed first: the LLM conversion happens when the user ADDS the question before starting (`convert`, the person is idle then);
starting a session and adding a question during a session never call the LLM (`add_converted`, `quick_convert`).
Never twice: a question that is (nearly) the same as one already asked is dropped, and one that matches a still-pending question
just promotes that question and marks it as the user's.
"""
from __future__ import annotations
from ..contracts import Action, Question
from .planner import clean_q, ko_ok, bad_question
from .interjection import _kw

USER_PRIORITY = 0.99          # right after the opening question (priority 1.0), before every bank question (<= 0.98) and generated follow-up
MAX_USER_Q = 10

USER_SYS = ("당신은 면접 질문자다. 사용자가 묻고 싶은 내용을 [메모]로 적었다. [참고 지문]에 맞춰 면접 질문 한 문장으로 바꾼다. "
            "존댓말 의문문 한 문장, 말로 읽기 쉽게 60자 안팎. 이미 질문 형태면 뜻을 바꾸지 말고 다듬기만 한다. "
            "지문에 없는 내용을 지어내지 않고, 정답이나 힌트를 질문에 넣지 않는다. 질문 한 문장만 출력한다.")

def quick_convert(raw: str) -> str:
    """No-LLM conversion (instant): a question stays as written, a keyword/phrase becomes "…에 대해 제안서의 근거를 말씀해 주시겠어요?"."""
    r = " ".join((raw or "").split())[:300].rstrip(".!").strip()
    if not r: return ""
    if r.endswith(("?", "？")): return r
    return r + "?" if r.endswith(("요", "까", "나요", "니까")) and len(r) >= 8 else f"{r}에 대해 제안서의 근거를 말씀해 주시겠어요?"

def ref_for(passage: str, raw: str) -> str:
    """The passage sentence that shares the most key terms with the note (keyword overlap, no model)."""
    from .passage import split_passage
    kw = _kw(raw); sents = split_passage(passage or "", 200)
    if not kw or not sents: return ""
    best = max(sents, key=lambda s: len(kw & _kw(s)))
    return best[:300] if kw & _kw(best) else ""

async def convert(llm, raw: str, passage: str = "") -> dict:
    """Free text -> {raw, q, ref}. LLM rewrite when it gives a clean question, else quick_convert."""
    raw = " ".join((raw or "").split())[:300]
    ref = ref_for(passage, raw); buf = ""
    try:
        async for tok in llm.stream(USER_SYS, f"[참고 지문]\n{ref[:600]}\n[메모]\n{raw}\n질문 한 문장만 출력한다.\n", 100):
            buf += tok
            if "?" in buf or "？" in buf: break
    except Exception: buf = ""
    q = clean_q(buf)
    if "?" in q: q = q[:q.index("?") + 1]
    if not (8 <= len(q) <= 120 and ko_ok(q)) or bad_question(q): q = quick_convert(raw)
    return {"raw": raw, "q": q, "ref": ref}

def add_converted(eng, items: list[dict]) -> list[dict]:
    """Queue already converted questions [{raw, q, ref}]. No LLM, no waiting. Returns one result per item: status added|merged|duplicate|empty."""
    out = []
    for i, it in enumerate([x for x in items if str(x.get("q", "")).strip()][:MAX_USER_Q]):
        q, raw, ref = it["q"], it.get("raw", ""), it.get("ref", "")
        if eng._seen(q): out.append({"raw": raw, "q": q, "status": "duplicate"}); continue          # already asked (or the same thing in other words)
        cand = Question(q, Action.SWITCH, None, USER_PRIORITY - 0.001 * i, "user", ref[:300]); cand.user_text = raw; cand.ref_text = ref[:300]
        dup = eng.queue._find_dup(cand)
        if dup is not None:
            if dup.status == "pending":                                                              # same as a waiting question: make it the user's, ask it early
                dup.priority = max(dup.priority, cand.priority); dup.source = "user"; dup.user_text = raw
                out.append({"raw": raw, "q": dup.text, "status": "merged", "priority": round(dup.priority, 3)})
            else: out.append({"raw": raw, "q": q, "status": "duplicate"})
            continue
        stored = eng.queue.add(cand)
        out.append({"raw": raw, "q": q, "status": "added" if stored else "duplicate", "priority": round(cand.priority, 3)})
    return out
