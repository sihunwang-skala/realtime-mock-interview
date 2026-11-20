from __future__ import annotations
import asyncio, json, time, platform, pathlib, sys, argparse
from . import dataset
from .replay import run_scenario, summarize
from ..core.llm import make_llm

async def amain(case, opts, only=None, out=None, realtime=True):
    import importlib
    mod = importlib.import_module(f"qa_lab.cases.case_{case.lower()}")
    make_engine, make_runtime, info = mod.build(opts)
    llm = make_llm(opts.get("llm", "ollama:llama3.2:3b"))
    if hasattr(llm, "warm"): await llm.warm()
    scns = [s for s in dataset.load() if not only or s["id"] in only]
    res = []
    for s in scns:
        r = await run_scenario(make_runtime, s, realtime=realtime, make_engine=make_engine); res.append(r)
        print(s["id"], "e2e", r.get("e2e_s"), "cer", r.get("cer"), "interj", r["interjected"], "act", r.get("action"), "|", (r.get("question") or "")[:60], flush=True)
    doc = {"case": case, "opts": opts, "components": info, "hardware": {"machine": platform.machine(), "platform": platform.platform(), "cpu": "Apple M2 Pro 32GB"},
           "when": time.strftime("%Y-%m-%d %H:%M:%S"), "audio_path": "synthetic Yuna TTS wav -> real-time paced frames -> VoiceRuntime (VAD/STT/LLM/TTS). Playback = sink emission, no browser.",
           "summary": summarize(res), "results": res}
    if out: pathlib.Path(out).parent.mkdir(exist_ok=True, parents=True); pathlib.Path(out).write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(json.dumps(doc["summary"], ensure_ascii=False, indent=1)); return doc

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default="B"); ap.add_argument("--llm", default="ollama:llama3.2:3b"); ap.add_argument("--stt", default="small")
    ap.add_argument("--interjection", default="hybrid"); ap.add_argument("--strategy", default="hybrid"); ap.add_argument("--only", nargs="*"); ap.add_argument("--confirm-ms", type=int, default=0); ap.add_argument("--turn", default="auto"); ap.add_argument("--out"); ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    asyncio.run(amain(a.case, dict(llm=a.llm, stt=a.stt, interjection=a.interjection, strategy=a.strategy, confirm_ms=a.confirm_ms, turn_mode=a.turn), a.only, a.out, realtime=not a.fast))
