"""Case A benchmark: same scenarios, but audio travels through a real LiveKit room (client -> SFU -> agent -> SFU -> client)."""
from __future__ import annotations
import asyncio, json, pathlib, time, platform, argparse
from . import dataset
from .replay import run_scenario, summarize
from ..cases import case_a
from ..core.llm import make_llm

async def amain(opts, only=None, out=None):
    make_engine, make_rt, info = case_a.build(opts)
    llm = make_llm(opts.get("llm", "ollama:qwen2.5:7b"))
    if hasattr(llm, "warm"): await llm.warm()
    sess = case_a.LKSession(make_engine, make_rt); await sess.start(); res = []
    try:
        for s in [x for x in dataset.load() if not only or x["id"] in only]:
            await sess.quiesce()
            def mk(eng, on_event, on_audio): sess.new_turn(eng); return case_a.LKProxy(sess)
            r = await run_scenario(mk, s, realtime=True, make_engine=make_engine); res.append(r)
            print(s["id"], "e2e", r.get("e2e_s"), "cer", r.get("cer"), "interj", r["interjected"], "act", r.get("action"), flush=True)
    finally: await sess.stop()
    doc = {"case": "A", "opts": opts, "components": {**info}, "hardware": {"cpu": "Apple M2 Pro 32GB", "platform": platform.platform()}, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "audio_path": "wav -> LiveKit client track (WebRTC/Opus) -> livekit-server --dev -> agent track -> VoiceRuntime -> agent audio track -> client receipt (first audible frame). Loopback, no real network.",
           "summary": summarize(res), "results": res}
    if out: pathlib.Path(out).write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(json.dumps(doc["summary"], ensure_ascii=False)); return doc

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--llm", default="ollama:qwen2.5:7b"); ap.add_argument("--stt", default="small"); ap.add_argument("--strategy", default="two_track")
    ap.add_argument("--interjection", default="hybrid"); ap.add_argument("--only", nargs="*"); ap.add_argument("--out"); a = ap.parse_args()
    asyncio.run(amain(dict(llm=a.llm, stt=a.stt, strategy=a.strategy, interjection=a.interjection), a.only, a.out))
