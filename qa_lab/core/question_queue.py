"""Question queue: dedup, priority, resolution by later evidence, asked-history."""
from __future__ import annotations
import re
from ..contracts import Question

def _norm(t: str) -> str: return re.sub(r"[^0-9a-z가-힣]", "", t.lower())
def _grams(t: str, n=2) -> set: s = _norm(t); return {s[i:i+n] for i in range(max(len(s)-n+1, 1))}
def similarity(a: str, b: str) -> float:
    A, B = _grams(a), _grams(b)
    return len(A & B) / len(A | B) if A and B else 0.0

class QuestionQueue:
    def __init__(self, dedup_threshold=0.55, max_pending=12):
        self.items: list[Question] = []; self.th = dedup_threshold; self.max_pending = max_pending
        self.log: list[dict] = []
    def _find_dup(self, q: Question):
        for x in self.items:
            if x.status in ("pending", "asked") and (similarity(x.text, q.text) >= self.th
               or (q.req_id and x.req_id == q.req_id and x.status == "pending" and q.action == x.action)):
                return x
    def add(self, q: Question) -> Question | None:
        """Returns the stored question, or None if it was a duplicate (priority of the original is raised)."""
        d = self._find_dup(q)
        if d:
            if d.status == "pending" and q.priority > d.priority: d.priority = q.priority
            self.log.append({"op": "dup", "text": q.text, "of": d.id}); return None
        self.items.append(q); self.log.append({"op": "add", "id": q.id, "text": q.text})
        pend = self.pending()
        for x in pend[self.max_pending:]: x.status = "discarded"   # keep the queue bounded
        return q
    def pending(self) -> list[Question]:
        return sorted((q for q in self.items if q.status == "pending"), key=lambda q: -q.priority)
    def pop(self) -> Question | None:
        p = self.pending()
        if not p: return None
        p[0].status = "asked"; return p[0]
    def peek(self): p = self.pending(); return p[0] if p else None
    def resolve_req(self, req_id: str, reason="answered later"):
        n = 0
        for q in self.items:
            if q.status == "pending" and q.req_id == req_id:
                q.status = "resolved"; n += 1; self.log.append({"op": "resolve", "id": q.id, "reason": reason})
        return n
    def reprioritize(self, req_id: str, delta: float):
        for q in self.pending():
            if q.req_id == req_id: q.priority = min(1.0, max(0.0, q.priority + delta))
    def asked(self) -> list[Question]: return [q for q in self.items if q.status == "asked"]
    def snapshot(self): return [{"id": q.id, "text": q.text, "status": q.status, "prio": round(q.priority, 2), "req": q.req_id} for q in self.items]
