"""Find question->answer pairs in a timestamped transcript with a LOCAL LLM (recordings never leave the machine).
Segments are shown to the LLM with indices; it returns index ranges; we validate durations and cut with ffmpeg."""
from __future__ import annotations
import asyncio, json, pathlib, subprocess, sys
from ..core.llm import make_llm
ROOT = pathlib.Path(__file__).resolve().parents[2]; SRC = pathlib.Path("/Users/gplace/Documents/Sysaud")
SYS = """다음은 멘토링/Q&A 녹음의 전사 구간이다. 각 줄은 [번호] 텍스트. 화자 구분은 없다.
'질문하는 말'과 곧바로 이어지는 '그 질문에 대한 답변'이 한 쌍이다. 쌍을 모두 찾아 JSON으로만 답한다:
{"pairs":[{"q":[시작번호,끝번호],"a":[시작번호,끝번호],"topic":"한 줄"}]}
규칙: 질문 범위는 질문 문장들만, 답변 범위는 그 질문에 답하는 말만(다음 질문 전까지). 사회/진행 멘트, 인사, 잡담은 제외. 번호는 주어진 번호만 사용. 확실한 쌍만."""

async def run(name, win=70, step=55):
    segs = json.loads((ROOT / "data/real" / f"{name}.json").read_text()); llm = make_llm("ollama:qwen2.5:7b"); await llm.warm(); found = []
    for a in range(0, len(segs), step):
        chunk = segs[a:a + win]
        user = "\n".join(f"[{a+i}] {s['text']}" for i, s in enumerate(chunk))
        d = await llm.json(SYS, user, 700)
        for p in d.get("pairs", []) or []:
            try:
                q0, q1 = p["q"]; a0, a1 = p["a"]
                if not (a <= q0 <= q1 < a0 <= a1 < len(segs)): continue
                qd = segs[q1]["end"] - segs[q0]["start"]; ad = segs[a1]["end"] - segs[a0]["start"]
                if 2 <= qd <= 70 and 6 <= ad <= 120 and segs[a0]["start"] - segs[q1]["end"] < 8: found.append({"q": [q0, q1], "a": [a0, a1], "topic": p.get("topic", "")})
            except Exception: continue
    # de-duplicate overlapping windows
    found.sort(key=lambda p: p["q"][0]); out = []
    for p in found:
        if not out or p["q"][0] > out[-1]["a"][1] - 1: out.append(p)
    return segs, out

import re
QRX = re.compile(r"(\?|궁금|여쭤|질문|까요|나요|죠\b|인가요|건가요|세요|습니까|어떻게|어떤|왜)")
def keep(name, segs, pairs):
    """Drop pairs whose 'question' has no question wording, or that overlap the hand-verified clips."""
    from .real_cut import TIMES
    hand = [t for _, _, t in TIMES.get(name, [])]; out = []
    for p in pairs:
        qt = " ".join(s["text"] for s in segs[p["q"][0]:p["q"][1] + 1]); t0 = segs[p["q"][0]]["start"]
        if not QRX.search(qt) or any(abs(t0 - h) < 20 for h in hand): continue
        out.append(p)
    return out

def cut(name, segs, pairs, prefix, cap_a=75):
    d = ROOT / "data/real_qa"; d.mkdir(exist_ok=True); n = 0
    for k, p in enumerate(pairs, 1):
        for role, (i0, i1) in (("q", p["q"]), ("a", p["a"])):
            t0 = segs[i0]["start"]; dur = min(segs[i1]["end"] - t0 + 0.4, cap_a if role == "a" else 60); cid = f"{prefix}{k:02d}_{role}"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(max(t0 - 0.2, 0)), "-t", str(dur), "-i", str(SRC / f"{name}.m4a"), "-ar", "16000", "-ac", "1", str(d / f"{cid}.wav")], check=True)
            txt = " ".join(s["text"] for s in segs[i0:i1 + 1])
            (d / f"{cid}.json").write_text(json.dumps({"id": cid, "role": role.upper(), "src": name, "start": t0, "dur": round(dur, 1), "text": txt, "topic": p["topic"], "auto": True}, ensure_ascii=False, indent=1))
        n += 1
    return n

if __name__ == "__main__":
    name, prefix = sys.argv[1], sys.argv[2]
    segs, pairs = asyncio.run(run(name)); n0 = len(pairs); pairs = keep(name, segs, pairs); print(name, n0, "found,", len(pairs), "kept after filter")
    for k, p in enumerate(pairs, 1): print(f"{prefix}{k:02d} Q@{segs[p['q'][0]]['start']:.0f}-{segs[p['q'][1]]['end']:.0f} A@{segs[p['a'][0]]['start']:.0f}-{segs[p['a'][1]]['end']:.0f} {p['topic'][:50]}")
    print("cut", cut(name, segs, pairs, prefix))
