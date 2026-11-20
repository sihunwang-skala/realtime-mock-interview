"""Questions that are missing from the recording (the user's own voice is not in system-audio captures) get a synthetic voice (macOS Yuna).
Detects them by measuring how much of each Q clip is speech; marks q JSON with q_audio_synth=true and writes <id>_tts.wav."""
from __future__ import annotations
import json, pathlib, subprocess
import numpy as np, soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps
ROOT = pathlib.Path(__file__).resolve().parents[2]; D = ROOT / "data/real_qa"

def speech_ratio(wav, model):
    a, sr = sf.read(wav, dtype="float32"); ts = get_speech_timestamps(a, model, sampling_rate=16000)
    return sum(t["end"] - t["start"] for t in ts) / max(len(a), 1)

def main():
    model = load_silero_vad(); n = 0
    for qf in sorted(D.glob("*_q*.json")):
        q = json.loads(qf.read_text()); wav = D / f"{q['id']}.wav"; text = q.get("ref") or q["text"]
        low = q.get("no_audio") or not wav.exists() or speech_ratio(wav, model) < 0.35
        q["q_audio_synth"] = bool(low)
        if low:
            subprocess.run(["say", "-v", "Yuna", "-r", "190", "-o", str(D / f"{q['id']}_tts.wav"), "--file-format=WAVE", "--data-format=LEI16@16000", text], check=True); n += 1
        qf.write_text(json.dumps(q, ensure_ascii=False, indent=1))
    print("synthesised question voice for", n, "of", len(list(D.glob('*_q*.json'))), "question clips")
if __name__ == "__main__": main()
