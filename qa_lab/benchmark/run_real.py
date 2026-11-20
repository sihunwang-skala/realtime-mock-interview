"""Replay REAL human mentor answers (cut from the mentoring recordings; local only) through Case B in real time.
Per clip: question context = the mentee's real question; audio = the mentor's real answer. Records every VAD endpoint (premature cuts),
the final response latency, interjections, and the AI follow-up next to the human mentee's actual follow-up (when one exists)."""
from __future__ import annotations
import asyncio, json, pathlib, time, sys
import numpy as np, soundfile as sf
from ..contracts import Question
from ..data.rfp import SEED_BANK
from .replay import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]; D = ROOT / "data/real_qa"
FOLLOWUP = {"r1_a": "r1_q2", "r3_a": "r3_q2", "r2_a": "r2_q2"}      # human follow-up questions that actually followed the answer

async def run(clips, opts, out):
    from ..cases import case_b
    from ..core.llm import make_llm
    mk_eng, mk_rt, info = case_b.build(opts); llm = make_llm(opts["llm"]); await llm.warm()
    res = []
    for cid in clips:
        qid = cid.replace("_a", "_q"); q = json.loads((D / f"{qid}.json").read_text()); a = json.loads((D / f"{cid}.json").read_text())
        eng = mk_eng(); cq = Question(q["text"][:300], source="human", status="asked"); eng.queue.items.append(cq); eng._asked(cq)
        for t in SEED_BANK[:2]: eng.queue.add(Question(t, priority=.6))
        rt = mk_rt(eng, lambda e: None, lambda pcm, m: None)
        audio, _ = sf.read(D / f"{cid}.wav", dtype="float32"); audio = np.concatenate([audio, np.zeros(3 * 16000, np.float32)]); end_t = a["dur"]
        t0 = time.monotonic(); pressed = False
        for i in range(0, len(audio) - 512, 512):
            await rt.feed(audio[i:i + 512])
            if getattr(rt, "turn_mode", "auto") == "manual" and not pressed and (i + 512) / 16000 >= end_t: pressed = True; await rt.finish_answer()
            w = t0 + (i + 512) / 16000 - time.monotonic()
            if w > 0: await asyncio.sleep(w)
        await asyncio.sleep(3)
        finals = [r for r in rt.responses if not r.get("interjection")]
        turns = [{"vad_end_at_s": round(r["marks"]["VAD_END_DETECTED"] - t0, 1), "stt": r["stt_text"][:80], "action": r["action"], "question": r["question"],
                  "first_audio_after_vad_s": round(r["marks"]["AUDIO_FIRST_PLAYED"] - r["marks"]["VAD_END_DETECTED"], 2) if "AUDIO_FIRST_PLAYED" in r["marks"] else None} for r in finals]
        last = finals[-1] if finals else None
        e2e = round(last["marks"]["AUDIO_FIRST_PLAYED"] - (t0 + end_t), 2) if last and "AUDIO_FIRST_PLAYED" in last["marks"] else None
        hf = json.loads((D / f"{FOLLOWUP[cid]}.json").read_text())["text"]if cid in FOLLOWUP and (D / f"{FOLLOWUP[cid]}.json").exists() else None
        r = {"id": cid, "human_question": q["text"], "answer_dur_s": a["dur"], "endpoints_in_answer": len(finals), "turns": turns, "e2e_after_clip_end_s": e2e,
             "interjections": rt.interjections, "human_followup": hf, "ai_followup": last["question"] if last else None}
        res.append(r); print(cid, "endpoints", len(finals), "e2e", e2e, "interj", len(rt.interjections), flush=True); await rt.close()
    doc = {"components": info, "opts": opts, "note": "real human speech (mentoring recordings), real-time paced; clip end is an arbitrary cut when dur==75s", "results": res,
           "summary": {"clips": len(res), "avg_endpoints_per_answer": round(sum(r["endpoints_in_answer"] for r in res) / max(len(res), 1), 1), "e2e_after_clip_end": stats([r["e2e_after_clip_end_s"] for r in res if r["e2e_after_clip_end_s"] is not None]),
                       "interjection_clips": sum(bool(r["interjections"]) for r in res)}}
    pathlib.Path(out).write_text(json.dumps(doc, ensure_ascii=False, indent=1)); print(json.dumps(doc["summary"], ensure_ascii=False))

if __name__ == "__main__":
    clips = sys.argv[1].split(","); asyncio.run(run(clips, dict(llm="ollama:qwen2.5:7b", stt="small", strategy="two_track", interjection="hybrid"), str(ROOT / "results/real_qa_B.json")))
