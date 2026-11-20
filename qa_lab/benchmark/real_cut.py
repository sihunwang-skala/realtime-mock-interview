"""Cut real Q&A clips out of the mentoring recordings using anchor phrases (from the user-provided transcripts) matched to our timestamped STT.
Output (local only, git-ignored): data/real_qa/<id>.wav + <id>.json. Roles: Q = mentee question, A = mentor answer."""
from __future__ import annotations
import difflib, json, pathlib, subprocess, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]; SRC = pathlib.Path("/Users/gplace/Documents/Sysaud")
N = lambda s: re.sub(r"[^0-9a-z가-힣]", "", s.lower())

# (id, role, start_s) — start times verified against the timestamped transcript (anchors above located the first set; the rest were set from keyword hits)
TIMES = {"2026-11-17_140215": [("r1_q", "Q", 129.6), ("r1_a", "A", 156.6), ("r1_q2", "Q", 204.1), ("r1_a2", "A", 210.8), ("r2_q", "Q", 253.3), ("r2_a", "A", 284.2),
    ("r3_q", "Q", 1093.0), ("r3_a", "A", 1173.1), ("r3_q2", "Q", 1243.5), ("r3_a2", "A", 1298.4), ("r4_q", "Q", 1895.4), ("r4_a", "A", 1949.2),
    ("r5_q", "Q", 2133.9), ("r5_a", "A", 2198.7), ("r6_q", "Q", 2524.1), ("r6_a", "A", 2573.1), ("r7_q", "Q", 2746.3), ("r7_a", "A", 2791.5),
    ("r8_q", "Q", 3084.5), ("r8_a", "A", 3158.7),
    ("r2_q2", "Q", 470.0), ("r2_a2", "A", 496.6), ("r9_q", "Q", 1546.0), ("r10_q", "Q", 1631.8), ("r10_a", "A", 1667.6),
    ("r11_q", "Q", 1738.4), ("r11_a", "A", 1758.3), ("r12_q", "Q", 2870.6), ("r12_a", "A", 2898.3), ("r12_end", "-", 2960.0)],
    "2026-11-11_130119": [("m3_q", "Q", 736.3), ("m3_a", "A", 791.1), ("m1_q", "Q", 1836.5), ("m1_a", "A", 1860.5, 1887.0), ("m4_q", "Q", 978.2, 1001.4), ("m4_a", "A", 1001.4, 1047.2), ("m5_q", "Q", 1254.6, 1294.0), ("m5_a", "A", 1294.0, 1368.6), ("m5_end", "-", 1370.0), ("m2_q", "Q", 2563.5), ("m2_a", "A", 2574.0), ("m2_end", "-", 2650.0)]}

CUE = re.compile(r"(결론만|결론적으로|결론은|정리하면|하시면 됩니다|하시면 돼요|시키는 게 됩니다|게 됩니다|될 거고|어려운 부분이 있어|잘 모르겠|좋을 것 같습니다|하겠습니다\.|판단을 했었습니다)")
def cue_end(segs, t0, cap):
    """End of an answer = first segment after >=8 s that carries a concluding cue (e.g. '결론만 말씀드리면 ... 됩니다.'), else the cap."""
    last = None
    for x in segs:
        if x["start"] < t0 - 0.1: continue
        if x["end"] - t0 > cap: break
        last = x["end"]
        if x["end"] - t0 >= 8 and CUE.search(x["text"]):
            if re.search(r"(결론만|결론적으로|결론은|정리하면)", x["text"]):          # the concluding sentence continues in the next segment
                nxt = next((y for y in segs if y["start"] >= x["end"] - 0.05), None)
                if nxt and nxt["end"] - t0 <= cap: return nxt["end"]
            return x["end"]
    return last

def excluded():
    f = ROOT / 'data/real_qa_exclude.json'; return set(json.loads(f.read_text())) if f.exists() else set()

def run(name):
    """Clip boundaries snap to transcript segment boundaries: Q ends at its last sentence, A ends at the last full segment inside the cap
    (or at an explicit end time, e.g. when another speaker takes over)."""
    segs = json.loads((ROOT / "data/real" / f"{name}.json").read_text()); out = ROOT / "data/real_qa"; out.mkdir(exist_ok=True)
    plan = sorted(TIMES[name], key=lambda x: x[2])
    bf = ROOT / "data/real_qa_bounds.json"; bounds = json.loads(bf.read_text()) if bf.exists() else {}
    for k, e in enumerate(plan):
        cid, role, t = e[:3]
        if role == "-" or cid in excluded() or cid[:-2] in {x[:-2] for x in excluded()}: continue
        nxt = plan[k + 1][2] if k + 1 < len(plan) else t + 60
        hi = e[3] if len(e) > 3 else (min(nxt, cue_end(segs, t, 100) or nxt) if role == "A" else min(nxt, t + 90))
        ends = [x["end"] for x in segs if t < x["end"] <= hi + 0.3]; hi = max(ends) if ends else hi
        dur = hi - t
        if dur < 3: continue
        wav = out / f"{cid}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-t", str(dur), "-i", str(SRC / f"{name}.m4a"), "-ar", "16000", "-ac", "1", str(wav)], check=True)
        text = " ".join(x["text"] for x in segs if x["start"] >= t - 0.1 and x["end"] <= hi + 0.1)
        (out / f"{cid}.json").write_text(json.dumps({"id": cid, "role": role, "src": name, "start": t, "dur": round(dur, 1), "text": text}, ensure_ascii=False, indent=1))
        print(f"{cid:6} {role} {t:7.1f} +{dur:4.0f}s  {text[:60]}")
if __name__ == "__main__": run(sys.argv[1])
