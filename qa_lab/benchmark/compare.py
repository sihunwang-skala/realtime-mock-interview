"""Build the comparison tables from result files. Only measured values are printed; missing -> NOT_TESTED / BLOCKED."""
from __future__ import annotations
import json, pathlib
from ..cases.registry import status
RES = pathlib.Path(__file__).resolve().parents[2] / "results"

def _f(x): return "NOT_TESTED" if x is None else x
def row(name, d):
    s = d["summary"]; e = s["e2e"]; i = s["interjection"]
    return f"| {name} | {d['components'].get('llm','')} | {_f(e.get('p50'))} | {_f(e.get('p90'))} | {_f(e.get('p95'))} | {_f(e.get('max'))} | {_f(round(e['under_2s']*100)) if 'under_2s' in e else 'NOT_TESTED'}% | {s['timeout_rate']} | {_f(s['mean_cer'])} | {_f(i['precision'])} / {_f(i['recall'])} | {_f(i['false_interjection_rate'])} |"

def compare() -> str:
    L = ["## Latency (actual user speech end -> first audio), 13 replayed scenarios, warm", "",
         "| Run | LLM | P50 s | P90 s | P95 s | Max s | <=2s | Timeout | mean CER | Interj P / R | FP rate |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(RES.glob("matrix_*.json")) + sorted(RES.glob("case_*_*.json")) + sorted(RES.glob("llm_*.json")):
        d = json.loads(f.read_text()); L.append(row(f.stem, d))
    L += ["", "## Case status", "", "| Case | Status | Reason |", "|---|---|---|"]
    from ..cases.registry import CASES
    for k in CASES: st = status(k); L.append(f"| {k} | {st['status']} | {st['reason']} |")
    return "\n".join(L)

if __name__ == "__main__": print(compare())
