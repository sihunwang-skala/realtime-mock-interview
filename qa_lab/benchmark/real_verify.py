"""Verify every real clip: (1) it contains speech, (2) its stored text agrees with a fresh transcription of the audio.
Pairs whose question or answer has no speech (or whose text clearly does not match the audio) are removed from the set (moved to data/real_qa_deleted)."""
from __future__ import annotations
import json, pathlib, re, sys
import numpy as np, soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps
from .replay import cer
from ..providers.stt import FasterWhisperSTT
ROOT = pathlib.Path(__file__).resolve().parents[2]; D = ROOT / "data/real_qa"; TRASH = ROOT / "data/real_qa_deleted"

def pair_ids(aid): return (aid[:-3] + "_q2") if aid.endswith("_a2") else aid[:-2] + "_q"

def delete_pair(aid, info, reason):
    TRASH.mkdir(exist_ok=True); qid = pair_ids(aid); times = []
    for i in (aid, qid):
        j = D / f"{i}.json"
        if j.exists() and not json.loads(j.read_text()).get("no_audio"): times.append({"src": json.loads(j.read_text())["src"], "t": json.loads(j.read_text())["start"]})
        for ext in (".wav", ".json"):
            for suffix in ("", "_tts"):
                f = D / f"{i}{suffix}{ext}"
                if f.exists(): f.rename(TRASH / f.name)
    ex = ROOT / "data/real_qa_exclude.json"; cur = set(json.loads(ex.read_text())) if ex.exists() else set(); cur |= {aid, qid}; ex.write_text(json.dumps(sorted(cur)))
    et = ROOT / "data/real_qa_exclude_times.json"; lst = json.loads(et.read_text()) if et.exists() else []; lst += times; et.write_text(json.dumps(lst))
    print("DELETED", aid, "-", reason)

def main():
    vad = load_silero_vad(); stt = FasterWhisperSTT("small"); stt.warm(); rows = {}
    for wav in sorted(D.glob("*.wav")):
        if wav.stem.endswith("_tts"): continue
        a, _ = sf.read(wav, dtype="float32"); ts = get_speech_timestamps(a, vad, sampling_rate=16000)
        ratio = sum(t["end"] - t["start"] for t in ts) / max(len(a), 1); meta = json.loads((D / f"{wav.stem}.json").read_text())
        ref = meta.get("ref") or meta["text"]; hyp = stt.transcribe(a) if ratio > 0.1 else ""
        rows[wav.stem] = {"speech_ratio": round(ratio, 2), "cer": round(cer(ref, hyp), 2) if hyp else None, "dur": round(len(a) / 16000, 1)}
    (ROOT / "results/real_qa_verify.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    deleted = []
    for aid in sorted(i for i in rows if i.endswith(("_a", "_a2"))):
        qid = pair_ids(aid); ra, rq = rows[aid], rows.get(qid)
        qmeta = json.loads((D / f"{qid}.json").read_text()) if (D / f"{qid}.json").exists() else None
        why = None
        if qmeta is None: why = "no question clip"
        elif qmeta.get("no_audio") or rq is None: why = "question has no recorded speech (slide text only)"
        elif rq["speech_ratio"] < 0.25: why = f"question has no speech (ratio {rq['speech_ratio']})"
        elif ra["speech_ratio"] < 0.25: why = f"answer has no speech (ratio {ra['speech_ratio']})"
        elif (ra["cer"] or 0) > 0.8 or (rq["cer"] or 0) > 0.8: why = f"text does not match audio (CER a={ra['cer']} q={rq['cer']})"
        if why: delete_pair(aid, rows, why); deleted.append(aid)
    border = [(i, r["cer"]) for i, r in rows.items() if r["cer"] is not None and 0.5 < r["cer"] <= 0.8 and i not in deleted]
    print("checked", len(rows), "clips;", len(deleted), "pairs deleted; borderline (CER 0.5-0.8):", border)
if __name__ == "__main__": main()
