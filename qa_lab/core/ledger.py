"""Cumulative question ledger across practices (spaced review). Speed first: nothing here touches the live path except one small JSON read at
session start (<1 ms) and O(few) lookups; grading and the ledger update happen AFTER the session, in the background.

Per tracked question (identified by its source sentence, else its wording):
  * answered correctly ("맞음", judged by the LLM at report time)  -> credit +1. After 2 credits the question is retired (never asked again).
    Between credits it may come back after a random gap of 2-4 practices (never the very next practice, never right after).
  * answered wrongly / partly / "모름"                              -> credits reset to 0; it is asked again in the NEXT practice, until answered correctly.
  * so "correct in practice 2, wrong in practice 5" -> asked in practice 6, and it needs 2 new correct answers to retire.
Follow-ups, interjections and on-the-spot questions are not tracked (they are generated per answer).
"""
from __future__ import annotations
import asyncio, hashlib, json, pathlib, random
from ..contracts import Action, Question
from .question_queue import _norm, _grams

DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "ledger"      # local only, next to data/live_runs
CORRECT = "맞음"                  # "부분 맞음", "틀림", "모름" all count as not correct
RETIRE_AFTER = 2
GAP = (2, 4)                    # practices to wait after a correct answer before it may return
MAX_REVIEW = 8                  # at most this many review questions are put in the queue per practice (wrong ones first)
TRACKED = {"bank", "user", "seed", "passage", "review", "first"}
WRONG_PRIO = 0.985              # right after the opening question (1.0) and the user's own questions (0.99)
PENDING: dict = {"task": None}  # background grading + ledger update of the last session

def pkey(passage: str) -> str:
    p = (passage or "").strip()
    return hashlib.sha1(p.encode()).hexdigest()[:12] if p else "seed"

def qkey(q: str, ref: str = "") -> str:
    r = _norm(ref or "")
    return "r" + hashlib.sha1(r.encode()).hexdigest()[:10] if r else "t" + hashlib.sha1(_norm(q).encode()).hexdigest()[:10]

class Ledger:
    def __init__(self, key: str):
        self.key = key; self.path = DIR / f"{key}.json"; self.d = {"practice": 0, "items": {}}
        try:
            if self.path.exists(): self.d = json.loads(self.path.read_text())
        except Exception: pass
    @property
    def practice(self) -> int: return int(self.d.get("practice", 0))
    @property
    def items(self) -> dict: return self.d["items"]
    def save(self):
        DIR.mkdir(parents=True, exist_ok=True); tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, ensure_ascii=False, indent=1)); tmp.replace(self.path)       # atomic: a crash never leaves half a file
    def due(self, p: int) -> list[dict]:
        """Questions to review in practice `p`: wrong ones first (oldest first), then correct-once ones whose random gap has passed."""
        its = [x for x in self.items.values() if not x.get("retired") and x.get("due", 0) <= p]
        its.sort(key=lambda x: (x.get("streak", 0) > 0, x.get("due", 0)))
        return its[:MAX_REVIEW]
    def retired(self) -> list[dict]: return [x for x in self.items.values() if x.get("retired")]
    def _index(self):
        """Built once per session (retired never changes during a practice): O(1) key sets; bigram sets only for questions without a source sentence."""
        if getattr(self, "_idx", None) is None:
            keys, noref = set(), []
            for x in self.retired():
                keys.add(x["key"]); keys.add(qkey(x["q"], ""))
                if not x.get("ref"): noref.append(_grams(x["q"]))
            self._idx = (keys, noref)
        return self._idx
    def is_retired(self, q: str, ref: str = "") -> bool:
        """Measured: 0.014 ms per 13 checks with 148 retired (a similarity scan over all of them was 16 ms)."""
        keys, noref = self._index()
        if not keys: return False
        if qkey(q, ref) in keys or qkey(q, "") in keys: return True
        if noref:
            g = _grams(q)
            return any(len(g & h) / len(g | h) >= 0.5 for h in noref)
        return False
    def apply(self, p: int, results: list[dict], rng=random) -> dict:
        """results: [{q, ref, a, ok, verdict}] in the order asked. Returns {key: credit info} for the report."""
        out = {}
        for r in results:
            k = qkey(r["q"], r.get("ref", ""))
            it = self.items.setdefault(k, {"key": k, "q": r["q"], "ref": r.get("ref", ""), "a": r.get("a") or "", "streak": 0, "retired": False, "due": 0, "history": []})
            if r.get("a"): it["a"] = r["a"]
            it["history"].append({"p": p, "ok": bool(r["ok"]), "verdict": r.get("verdict", "")})
            if r["ok"]:
                it["streak"] += 1
                if it["streak"] >= RETIRE_AFTER: it["retired"] = True; it["due"] = None
                else: it["due"] = p + rng.randint(*GAP)
            else: it["streak"] = 0; it["retired"] = False; it["due"] = p + 1
            out[k] = {"practice": p, "ok": bool(r["ok"]), "streak": it["streak"], "retired": bool(it["retired"]), "next_due": it["due"], "history": list(it["history"]), "text": credit_text(it, p)}
        self.d["practice"] = max(self.practice, p)
        return out

def credit_text(it: dict, p: int) -> str:
    h = it["history"]; good = [x["p"] for x in h if x["ok"]]; bad = [x["p"] for x in h if not x["ok"]]
    hist = (" · 맞은 연습: " + ", ".join(f"연습 {x}" for x in good) if good else "") + (" · 틀린 연습: " + ", ".join(f"연습 {x}" for x in bad) if bad else "")
    last = h[-1]
    if last["ok"] and it.get("retired"): return f"🎓 {it['streak']}번째 맞음 (연습 {p}) — {RETIRE_AFTER}번 맞아서 졸업, 앞으로 안 물어봐요{hist}"
    if last["ok"]: return f"✔ {it['streak']}번째 맞음 (연습 {p}) — 연습 {it['due']} 이후 한 번 더 물어볼 수 있어요{hist}"
    return f"✖ 연습 {p}: 틀림 — 다음 연습에 다시 물어봐요 (맞은 횟수 0/{RETIRE_AFTER}){hist}"

def inject(eng, led: Ledger, p: int, rng=random) -> list[dict]:
    """Put the due review questions into the live queue (stored wording + stored model answer: no LLM call) and drop retired ones.
    Wrong ones go right after the opening/user questions; correct-once ones are mixed in at a random place, never next to each other."""
    from .passage import _general_q
    due = [x for x in led.due(p) if not _general_q(x["q"])]; n_wrong = 0       # generic questions saved by earlier practices never come back
    for it in due:
        wrong = it.get("streak", 0) == 0
        prio = WRONG_PRIO - 0.001 * n_wrong if wrong else rng.uniform(0.3, 0.9)
        n_wrong += wrong
        q = Question(it["q"], Action.SWITCH, None, prio, "review", it.get("ref", "")[:300]); q.ref_text = it.get("ref", "")[:300]
        q.bank = {"q": it["q"], "a": it.get("a") or None, "ref": it.get("ref", ""), "src": -1}              # instant model answer
        q.review = {"streak": it.get("streak", 0), "history": it.get("history", [])}
        eng.queue.add(q)
    eng.blocked_fn = lambda q: led.is_retired(q.text, eng.ref_of(q))                                         # retired questions are skipped when popped
    pend = eng.queue.pending(); keep = [x for x in pend if not led.is_retired(x.text, eng.ref_of(x))]
    if keep:
        for x in pend:
            if x not in keep: x.status = "discarded"
    return due

def collect(report: dict) -> list[dict]:
    """Tracked, LLM-graded answers of a finished report, in the order asked. Quick (keyword) verdicts never count."""
    src_by_q = {}; out = []
    for t in report.get("turns", []):
        if t.get("interjection") or t.get("clarify"): continue
        aq = t.get("answered_q") or ""
        if t.get("stt_text") and aq and t.get("verdict") and not t.get("verdict_pending"):
            src = src_by_q.get(aq, "first")
            if src in TRACKED: out.append({"q": aq, "ref": t.get("ref_text") or "", "a": t.get("model_answer") or "", "ok": t["verdict"] == CORRECT, "verdict": t["verdict"], "turn": t})
        if t.get("question"): src_by_q[t["question"]] = t.get("q_source") or "bank"
    return out

async def wait_pending(seconds: float = 8.0):
    """Called at session start: let the previous session's grading finish (briefly) so its results are in the ledger."""
    t = PENDING.get("task")
    if t is not None and not t.done():
        try: await asyncio.wait_for(asyncio.shield(t), seconds)
        except Exception: pass
