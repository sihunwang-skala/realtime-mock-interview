"""Attach the USER-PROVIDED transcripts (txt, local) as reference text to each real clip, then measure STT CER on real human speech.
Caveat: the provided transcripts are themselves ASR-quality text (e.g. brand names spelled phonetically), so CER here = disagreement with that reference, not absolute WER."""
from __future__ import annotations
import difflib, json, pathlib, re, time
import soundfile as sf
from .replay import cer, stats

ROOT = pathlib.Path(__file__).resolve().parents[2]; D = ROOT / "data/real_qa"; SRC = pathlib.Path("/Users/gplace/Documents/Sysaud")
N = lambda s: re.sub(r"[^0-9a-z가-힣]", "", s.lower())

def find(txt_n: str, probe: str, lo=0, hi=None):
    best = (0, 0); hi = hi or len(txt_n); w = len(probe)
    for i in range(lo, max(hi - w, lo) + 1, 4):
        r = difflib.SequenceMatcher(None, txt_n[i:i + w], probe).quick_ratio()
        if r > best[0] + 1e-9: best = (r, i)
    # refine around the coarse best
    b2 = (0, best[1])
    for i in range(max(best[1] - 6, 0), best[1] + 7):
        r = difflib.SequenceMatcher(None, txt_n[i:i + w], probe).ratio()
        if r > b2[0]: b2 = (r, i)
    return b2

def attach():
    raw = {n: (SRC / f"{n}.txt").read_text() for n in ("2026-11-17_140215", "2026-11-11_130119")}
    # keep original text, but search in a normalised copy with an index map back to the original
    for clip in sorted(D.glob("*.json")):
        m = json.loads(clip.read_text()); t = raw[m["src"]]
        idx = [i for i, c in enumerate(t) if re.match(r"[0-9a-zA-Z가-힣]", c)]; tn = N(t)
        if m["src"] not in raw or m.get("no_audio"): continue
        mine = N(m["text"]); p = 24
        r0, s = find(tn, mine[:p]); r1, e = find(tn, mine[-p:], lo=s, hi=min(len(tn), s + int(len(mine) * 2.5) + 60)); e += p
        # tighten both ends so no fragment of the neighbouring sentence is included
        head, tail = mine[:10], mine[-10:]
        s = max(range(max(s - 12, 0), s + 13), key=lambda j: difflib.SequenceMatcher(None, tn[j:j + 10], head).ratio())
        e = max(range(max(e - 12, 10), min(e + 13, len(tn))), key=lambda j: difflib.SequenceMatcher(None, tn[j - 10:j], tail).ratio())
        a0, a1 = idx[s], idx[min(max(e - 1, s), len(idx) - 1)] + 1      # e is exclusive: last char is idx[e-1]
        isw = lambda ch: re.match(r"[0-9a-zA-Z가-힣]", ch) is not None
        while a0 > 0 and isw(t[a0 - 1]): a0 -= 1                      # never start mid-word ('희가' -> '저희가')
        while a1 < len(t) and isw(t[a1]): a1 += 1                     # never end mid-word ('부분인' -> '부분인지')
        ref = t[a0:a1].strip()
        fm = re.match(r"^(.{0,14}?[.?!])\s+", ref)                    # leading fragment of the previous sentence ('싶습니다. ...')
        if fm and N(fm.group(1)) not in mine[:24]: ref = ref[fm.end():]
        m.update(ref=ref, ref_match=[round(r0, 2), round(r1, 2)]); clip.write_text(json.dumps(m, ensure_ascii=False, indent=1))
        print(m["id"], m["role"], "ref match", m["ref_match"], "len mine/ref", len(mine), len(N(ref)))

def stt_cer(models=("small", "large-v3-turbo")):
    from ..providers.stt import FasterWhisperSTT
    out = {}
    for name in models:
        stt = FasterWhisperSTT(name); stt.warm(); rows = []
        for clip in sorted(D.glob("*_a.json")):
            m = json.loads(clip.read_text())
            if min(m["ref_match"]) < 0.55: continue
            a, _ = sf.read(D / f"{m['id']}.wav", dtype="float32"); t0 = time.time(); hyp = stt.transcribe(a); el = time.time() - t0
            rows.append({"id": m["id"], "cer_vs_provided_ref": round(cer(m["ref"], hyp), 3), "rtf": round(el / (len(a) / 16000), 3)})
        out[stt.name] = {"clips": len(rows), "mean_cer": round(sum(r["cer_vs_provided_ref"] for r in rows) / max(len(rows), 1), 3), "mean_rtf": round(sum(r["rtf"] for r in rows) / max(len(rows), 1), 3), "rows": rows}
        print(stt.name, out[stt.name]["mean_cer"], "rtf", out[stt.name]["mean_rtf"], flush=True)
    (ROOT / "results/real_stt_cer.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    import sys; attach(); stt_cer() if "--stt" in sys.argv else None
