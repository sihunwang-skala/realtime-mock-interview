"""Long-running soak on REAL human speech (local files, never committed or uploaded). Real-time paced unless --speed > 1.
Feeds the full VoiceRuntime (VAD, partial STT, interjection policy, final STT, LLM, TTS); records RSS/CPU, event-loop lag, latency, errors."""
from __future__ import annotations
import argparse, asyncio, json, pathlib, subprocess, time, statistics
import numpy as np, psutil
from ..contracts import Question
from ..data.rfp import SEED_BANK
from . import replay

async def main(path, minutes, speed, opts, out):
    from ..cases import case_b
    make_engine, make_rt, info = case_b.build(opts)
    eng = make_engine()
    for t in SEED_BANK: eng.queue.add(Question(t, priority=.6))
    q = eng.first_question(); events = []; audio_bytes = [0]
    rt = make_rt(eng, lambda e: events.append(e.type.value), lambda pcm, m: audio_bytes.__setitem__(0, audio_bytes[0] + len(pcm)))
    secs = int(minutes * 60)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-t", str(secs), "-ar", "16000", "-ac", "1", "-f", "f32le", "-"], stdout=subprocess.PIPE)
    proc = psutil.Process(); series = []; lag = []; errors = []; t0 = time.monotonic(); fr = 512; n = 0; last = t0
    async def lag_probe():
        while True:
            a = time.monotonic(); await asyncio.sleep(0.1); lag.append(time.monotonic() - a - 0.1)
    lp = asyncio.create_task(lag_probe()); nresp = 0
    try:
        while True:
            buf = p.stdout.read(fr * 4)
            if len(buf) < fr * 4: break
            try: await rt.feed(np.frombuffer(buf, np.float32))
            except Exception as ex: errors.append(repr(ex))
            n += fr; target = t0 + n / 16000 / speed; w = target - time.monotonic()
            if w > 0: await asyncio.sleep(w)
            if time.monotonic() - last >= 30:
                last = time.monotonic(); series.append({"t": round(last - t0), "audio_s": round(n / 16000), "rss_mb": round(proc.memory_info().rss / 1e6), "cpu": psutil.cpu_percent(), "responses": len(rt.responses),
                                                        "interjections": len(rt.interjections), "lag_p99_ms": round(replay.pct(lag[-300:], 99) * 1000, 1) if lag else 0, "errors": len(errors)})
                print(series[-1], flush=True)
    finally: lp.cancel(); p.kill()
    finals = [r for r in rt.responses if not r.get("interjection")]
    lat = [r["marks"]["AUDIO_FIRST_PLAYED"] - r["marks"]["VAD_END_DETECTED"] for r in finals if "AUDIO_FIRST_PLAYED" in r["marks"] and "VAD_END_DETECTED" in r["marks"]]
    doc = {"file": pathlib.Path(path).name, "audio_minutes": round(n / 16000 / 60, 1), "wall_minutes": round((time.monotonic() - t0) / 60, 1), "speed": speed, "opts": opts, "components": info,
           "utterances_responded": len(finals), "interjections": len(rt.interjections), "vad_end_to_first_audio_s": replay.stats(lat), "errors": errors[:20], "n_errors": len(errors),
           "rss_mb_start_end_max": [series[0]["rss_mb"] if series else None, series[-1]["rss_mb"] if series else None, max((s["rss_mb"] for s in series), default=None)],
           "loop_lag_ms": {"p99": round(replay.pct(lag, 99) * 1000, 1), "max": round(max(lag) * 1000, 1)}, "series": series,
           "note": "Real recorded human meeting speech (multi-speaker), not Q&A; validates stability, VAD/STT behaviour, not question quality."}
    pathlib.Path(out).write_text(json.dumps(doc, ensure_ascii=False, indent=1)); print(json.dumps({k: v for k, v in doc.items() if k != "series"}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--minutes", type=float, default=60); ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--llm", default="ollama:qwen2.5:7b"); ap.add_argument("--stt", default="small"); ap.add_argument("--out", default="results/soak.json"); a = ap.parse_args()
    asyncio.run(main(a.path, a.minutes, a.speed, dict(llm=a.llm, stt=a.stt, strategy="two_track", interjection="hybrid"), a.out))
