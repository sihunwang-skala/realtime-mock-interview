"""Rule-based Q/A pairing for recordings where questions come one after another ('... 궁금합니다.').
Sentence units (approx. times split by character position inside a segment) -> question = units up to a question-ending phrase,
answer = following units until the next question (cue-ended / capped). Pairs are cut with ffmpeg; local-only data."""
from __future__ import annotations
import json, pathlib, re, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]; SRC = pathlib.Path("/Users/gplace/Documents/Sysaud")
QEND = re.compile(r"(궁금합니다|궁금해요|궁금한데요|궁금하네요|여쭤봅니다|여쭤보고 싶습니다|여쭤보고 싶은데요|있을까요|될까요|인가요|건가요|할까요|있나요|하나요|되나요|어떤가요|어떻게 하시나요|맞나요)[?.!]?\s*$")
END = re.compile(r"[.?!]$")

def units_from_words(ws):
    """Exact sentence units from word-level timestamps (a word that ends with . ? ! closes the sentence)."""
    out = []; cur = []
    for seg in ws:
        for w in seg.get("words", []):
            cur.append(w)
            if re.search(r"[.?!]$", w["w"].strip()):
                out.append({"t0": cur[0]["s"], "t1": cur[-1]["e"], "text": "".join(x["w"] for x in cur).strip()}); cur = []
        if cur and seg is ws[-1]: pass
    if cur: out.append({"t0": cur[0]["s"], "t1": cur[-1]["e"], "text": "".join(x["w"] for x in cur).strip()})
    return out

def units(segs):
    out = []
    for s in segs:
        parts = [p for p in re.split(r"(?<=[.?!])\s+", s["text"].strip()) if p]
        tot = max(sum(len(p) for p in parts), 1); t = s["start"]
        for p in parts:
            d = (s["end"] - s["start"]) * len(p) / tot; out.append({"t0": t, "t1": t + d, "text": p}); t += d
    return out

def find_pairs(name, cap_a=75, min_a=6):
    wf = ROOT / "data/real" / f"{name}.words.json"
    segs = json.loads((ROOT / "data/real" / f"{name}.json").read_text()); U = units_from_words(json.loads(wf.read_text())) if wf.exists() else units(segs); qs = []
    for i, u in enumerate(U):
        if len(u["text"]) >= 8 and QEND.search(u["text"]):
            j = i
            while j > 0 and i - j < 3 and not END.search(U[j - 1]["text"]) and U[j]["t0"] - U[j - 1]["t1"] < 1.5: j -= 1
            qs.append((j, i))
    pairs = []
    for k, (qj, qi) in enumerate(qs):
        nxt_start = qs[k + 1][0] if k + 1 < len(qs) else len(U)
        a0 = qi + 1
        if a0 >= len(U): continue
        a1 = a0
        while a1 + 1 < nxt_start and U[a1 + 1]["t0"] - U[a0]["t0"] <= cap_a and U[a1 + 1]["t1"] - U[a0]["t0"] <= 130: a1 += 1   # a long unpunctuated sentence is kept if it STARTS inside the cap
        if U[a1]["t1"] - U[a0]["t0"] < min_a: continue
        pairs.append({"q": (U[qj]["t0"], U[qi]["t1"], " ".join(x["text"] for x in U[qj:qi + 1])), "a": (U[a0]["t0"], U[a1]["t1"], " ".join(x["text"] for x in U[a0:a1 + 1]))})
    return pairs

def excluded_times():
    f = ROOT / "data/real_qa_exclude_times.json"; return json.loads(f.read_text()) if f.exists() else []

def cut(name, pairs, prefix, skip_near=()):
    d = ROOT / "data/real_qa"; d.mkdir(exist_ok=True); n = 0
    for p in pairs:
        if any(abs(p["q"][0] - h) < 20 for h in skip_near): continue
        if any(e["src"] == name and abs(p["q"][0] - e["t"]) < 8 for e in excluded_times()): continue
        n += 1; base = f"{prefix}{n:02d}"
        for role, (t0, t1, text) in (("q", p["q"]), ("a", p["a"])):
            cid = f"{base}_{role}"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(max(t0 - 0.15, 0)), "-t", str(t1 - t0 + 0.45), "-i", str(SRC / f"{name}.m4a"), "-ar", "16000", "-ac", "1", str(d / f"{cid}.wav")], check=True)
            (d / f"{cid}.json").write_text(json.dumps({"id": cid, "role": role.upper(), "src": name, "start": round(t0, 1), "dur": round(t1 - t0 + 0.3, 1), "text": text, "auto": True, "rule": True}, ensure_ascii=False, indent=1))
    return n

if __name__ == "__main__":
    name, prefix = sys.argv[1], sys.argv[2]
    from .real_cut import TIMES
    pairs = find_pairs(name); hand = [t[2] for t in TIMES.get(name, [])]
    print(name, len(pairs), "rule pairs"); n = cut(name, pairs, prefix, skip_near=hand); print("cut", n)
    for p in pairs[:40]: print(f"Q@{p['q'][0]:6.0f} {p['q'][2][:44]} | A@{p['a'][0]:6.0f}-{p['a'][1]:6.0f} {p['a'][2][:40]}")
