"""Decide where each answer really ends (local LLM over the timestamped transcript) and store it in data/real_qa_bounds.json.
real_cut.py reads the bounds, so clips stop at the concluding sentence of the answer instead of at a fixed time cap."""
from __future__ import annotations
import asyncio, json, pathlib, sys
from ..core.llm import make_llm
ROOT = pathlib.Path(__file__).resolve().parents[2]; D = ROOT / "data/real_qa"; B = ROOT / "data/real_qa_bounds.json"
SYS = """질문과, 그 뒤에 이어지는 전사 구간([번호] 텍스트)이 주어진다. 구간은 질문에 답하는 말로 시작한다.
질문에 대한 답변이 끝나는 마지막 구간 번호를 고른다. 답변이 끝난 뒤에 이어지는 다른 사람의 말, 새 질문, 진행 멘트, 다른 주제는 포함하지 않는다.
결론을 말한 문장('결론만 말씀드리면', '그래서 ~하시면 됩니다' 등)에서 끝나는 경우가 많다. JSON만: {"end": 번호}"""

async def main():
    llm = make_llm("ollama:qwen2.5:7b"); await llm.warm(); bounds = json.loads(B.read_text()) if B.exists() else {}
    for a in sorted(D.glob("*_a.json")):
        m = json.loads(a.read_text()); q = D / a.name.replace("_a.json", "_q.json")
        if not q.exists(): continue
        qt = json.loads(q.read_text()); segs = json.loads((ROOT / "data/real" / f"{m['src']}.json").read_text())
        idx = [i for i, s in enumerate(segs) if s["start"] >= m["start"] - 0.1]
        if not idx: continue
        i0 = idx[0]; win = [(i, segs[i]) for i in range(i0, min(i0 + 45, len(segs))) if segs[i]["start"] - m["start"] <= 130]
        user = f"[질문]\\n{qt.get('ref') or qt['text']}\\n\\n[전사]\\n" + "\\n".join(f"[{i}] {s['text']}" for i, s in win)
        d = await llm.json(SYS, user, 60)
        try: e = int(d["end"]); ok = win[0][0] <= e <= win[-1][0]
        except Exception: ok = False
        if ok:
            bounds[m["id"]] = round(segs[e]["end"], 1); print(m["id"], "ends", bounds[m["id"]], f"(+{bounds[m['id']] - m['start']:.0f}s)", segs[e]["text"][:40], flush=True)
        else: print(m["id"], "no valid end; keeping cap")
    B.write_text(json.dumps(bounds, ensure_ascii=False, indent=1))
if __name__ == "__main__": asyncio.run(main())
