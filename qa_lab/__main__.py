"""CLI: python -m qa_lab cases list | demo --case B | benchmark --case B | compare --latest | simulate | serve"""
from __future__ import annotations
import argparse, asyncio, json, pathlib, sys
from .cases.registry import CASES, status
RES = pathlib.Path(__file__).resolve().parents[1] / "results"

def main(argv=None):
    ap = argparse.ArgumentParser(prog="qa_lab"); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases"); c.add_argument("action", choices=["list", "show"]); c.add_argument("--case")
    for n in ("demo", "benchmark"):
        s = sub.add_parser(n); s.add_argument("--case", default="B"); s.add_argument("--all", action="store_true")
        s.add_argument("--llm", default="ollama:qwen2.5:7b"); s.add_argument("--stt", default="small"); s.add_argument("--strategy", default="two_track"); s.add_argument("--interjection", default="hybrid")
        s.add_argument("--only", nargs="*"); s.add_argument("--fast", action="store_true"); s.add_argument("--out")
    s = sub.add_parser("simulate"); s.add_argument("--fast", default="ollama:qwen2.5:7b"); s.add_argument("--presenter", default="ollama:qwen2.5:7b"); s.add_argument("--turns", type=int, default=5)
    s = sub.add_parser("compare"); s.add_argument("--latest", action="store_true")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    if a.cmd == "cases":
        if a.action == "list":
            print(f"{'ID':3} {'STATUS':26} {'NAME':36} REASON")
            for k, v in CASES.items(): st = status(k); print(f"{k:3} {st['status']:26} {v['name']:36} {st['reason']}")
        else: print(json.dumps({**CASES[a.case], **status(a.case)}, ensure_ascii=False, indent=1))
    elif a.cmd == "benchmark":
        from .benchmark.run import amain
        cases = list(CASES) if a.all else [a.case]
        for k in cases:
            st = status(k)
            if st["status"] in ("BLOCKED", "NOT_IMPLEMENTED"): print(f"Case {k}: {st['status']} — {st['reason']}"); continue
            asyncio.run(amain(k, dict(llm=a.llm, stt=a.stt, strategy=a.strategy, interjection=a.interjection), a.only, a.out or str(RES / f"bench_{k}_latest.json"), realtime=not a.fast))
    elif a.cmd == "demo":
        st = status(a.case)
        if st["status"] in ("BLOCKED", "NOT_IMPLEMENTED"): print(f"Case {a.case}: {st['status']} — {st['reason']}"); return 2
        from .demo.server import serve; serve(case=a.case, port=8765)
    elif a.cmd == "simulate":
        from .simulator.text_sim import run
        lg = asyncio.run(run(fast=a.fast, presenter=a.presenter, turns=a.turns, out=str(RES / "text_sim_latest.json")))
        for x in lg["phaseB"]: print(f"[{x['behaviour']}] Q: {x['question']}\n  A: {x['answer'][:100]}\n  -> {x['action']}: {x['next_question']}")
    elif a.cmd == "compare":
        from .benchmark.compare import compare; print(compare())
    elif a.cmd == "serve":
        from .demo.server import serve; serve(case=None, port=a.port)
main() if __name__ == "__main__" else None
