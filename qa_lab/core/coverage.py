"""Requirement coverage tracking with evidence, revisable as later speech arrives."""
from __future__ import annotations
from ..contracts import Requirement

class Coverage:
    def __init__(self, reqs: list[Requirement]):
        self.reqs = {r.id: r for r in reqs}
        self.score = {r.id: 0.0 for r in reqs}
        self.evidence: dict[str, list[dict]] = {r.id: [] for r in reqs}
        self.verified: set[str] = set()
    def update(self, req_id: str, score: float, quote: str, ts: float):
        if req_id not in self.reqs: return
        score = max(0.0, min(1.0, score))
        self.evidence[req_id].append({"ts": ts, "score": score, "quote": quote})
        self.score[req_id] = max(self.score[req_id], score)   # later explanation can only raise coverage
    def gaps(self, th=0.6): return [i for i, s in self.score.items() if s < th]
    def mark_verified(self, req_id): self.verified.add(req_id)
    def report(self): return {i: {"req": self.reqs[i].text, "score": round(s, 2), "evidence": self.evidence[i][-2:]} for i, s in self.score.items()}
