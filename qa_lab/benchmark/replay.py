"""Replay pre-recorded audio through a case's real voice path in real time and collect latency/accuracy/interjection data."""
from __future__ import annotations
import asyncio, json, re, time, statistics, pathlib
import numpy as np, soundfile as sf
from ..contracts import Question
from ..core.engine import QAEngine
from ..data.rfp import REQS

def cer(ref: str, hyp: str) -> float:
    n = lambda s: re.sub(r"[\s.,?!]", "", s)
    a, b = n(ref), n(hyp)
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1] / max(len(a), 1)

def pct(xs, p):
    xs = sorted(xs)
    if not xs: return None
    k = (len(xs) - 1) * p / 100; f = int(k); c = min(f + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 3)

def stats(xs):
    if not xs: return {"n": 0}
    return {"n": len(xs), "mean": round(statistics.mean(xs), 3), "p50": pct(xs, 50), "p90": pct(xs, 90), "p95": pct(xs, 95), "max": round(max(xs), 3), "under_2s": round(sum(x <= 2.0 for x in xs) / len(xs), 3)}

SPARE = ["보안 설계에서 접근 로그는 어떻게 관리하실 건가요?", "AI 분류 정확도 90%를 어떻게 검증하실 계획인가요?"]

async def run_scenario(make_runtime, scn: dict, realtime=True, timeout_after=25.0, **kw):
    """make_runtime(engine, on_event, on_audio) -> runtime object with .feed(), .marks, .responses, .interjections, .wait()"""
    eng = kw["make_engine"]()
    q = Question(scn["q"], source="seed", status="asked"); eng.queue.items.append(q); eng._asked(q)
    for t in SPARE: eng.queue.add(Question(t, priority=.6))
    events, audio_chunks = [], []
    rt = make_runtime(eng, events.append, lambda pcm, m: audio_chunks.append((time.monotonic(), len(pcm))))
    audio, sr = sf.read(scn["wav"], dtype="float32"); assert sr == 16000
    fr = 512; t0 = time.monotonic(); actual_end = t0 + scn["speech_end_s"]; pressed = False
    for i in range(0, len(audio) - fr, fr):
        await rt.feed(audio[i:i + fr])
        if getattr(rt, "turn_mode", "auto") == "manual" and not pressed and (i + fr) / 16000 >= scn["speech_end_s"]:
            pressed = True; await rt.finish_answer()        # test assumption: the user presses 'answer done' exactly when the speech really ends
        if realtime:
            wait = t0 + (i + fr) / 16000 - time.monotonic()
            if wait > 0: await asyncio.sleep(wait)
        else: await asyncio.sleep(0)
    dl = time.monotonic() + timeout_after
    while time.monotonic() < dl:
        if rt.responses and not rt.agent_speaking and (rt.agent_task is None or rt.agent_task.done()): break
        await asyncio.sleep(0.05)
    timed_out = not rt.responses
    final = [r for r in rt.responses if not r.get("interjection")]; inter = [r for r in rt.responses if r.get("interjection")]
    r0 = final[0] if final else None
    out = {"id": scn["id"], "timed_out": timed_out, "interjected": bool(rt.interjections), "interjections": rt.interjections,
           "interjection_responses": inter, "expect": scn["expect"]}
    if r0:
        mk = r0["marks"]; g = lambda k: mk.get(k)
        out.update(stt_text=r0["stt_text"], cer=round(cer(scn["ref"], r0["stt_text"]), 3), action=r0["action"], question=r0["question"], assessment=r0["assessment"])
        if g("AUDIO_FIRST_PLAYED"):
            out["e2e_s"] = round(g("AUDIO_FIRST_PLAYED") - actual_end, 3)
            out["stages"] = {"speech_end_to_vad_end": round(g("VAD_END_DETECTED") - actual_end, 3), "stt_final": round(g("STT_FINALIZED") - g("VAD_END_DETECTED"), 3),
                "answer_analysis": round(g("ANSWER_ANALYSIS_COMPLETED") - g("STT_FINALIZED"), 3), "first_token": round(g("QUESTION_FIRST_TOKEN") - g("ANSWER_ANALYSIS_COMPLETED"), 3),
                "tts_first_chunk": round(g("TTS_FIRST_AUDIO_CHUNK") - g("QUESTION_FIRST_TOKEN"), 3)}
    if inter and inter[0]["marks"].get("AUDIO_FIRST_PLAYED"):
        out["interjection_audio_after_speech_start_s"] = round(inter[0]["marks"]["AUDIO_FIRST_PLAYED"] - (t0 + 0.5), 3)
    return out

def score_interjection(results):
    tp = sum(r["expect"]["interject"] and r["interjected"] for r in results); fp = sum((not r["expect"]["interject"]) and r["interjected"] for r in results)
    fn = sum(r["expect"]["interject"] and not r["interjected"] for r in results); tn = sum((not r["expect"]["interject"]) and not r["interjected"] for r in results)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "false_interjection_rate": round(fp / (fp + tn), 3) if fp + tn else None, "missed_rate": round(fn / (tp + fn), 3) if tp + fn else None}

def summarize(results):
    e2e = [r["e2e_s"] for r in results if "e2e_s" in r and not r["interjected"]]
    return {"e2e": stats(e2e), "timeout_rate": round(sum(r["timed_out"] for r in results) / len(results), 3), "mean_cer": round(statistics.mean(r["cer"] for r in results if "cer" in r), 3) if any("cer" in r for r in results) else None,
            "interjection": score_interjection(results)}
