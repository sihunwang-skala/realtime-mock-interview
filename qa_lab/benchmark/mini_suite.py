"""Mini suite: 7 tests, 3 samples each. Real recordings stay local (local models only); cloud is used for synthetic data only."""
from __future__ import annotations
import asyncio, json, pathlib, sys, time
import numpy as np, soundfile as sf
ROOT = pathlib.Path(__file__).resolve().parents[2]; D = ROOT / "data/real_qa"; R = ROOT / "results"
REAL = ["r1_a", "r6_a", "m1_a"]; SYN = ["s01_normal", "s08_contradict_rfp", "s11_ramble"]

def real(cid):
    a, _ = sf.read(D / f"{cid}.wav", dtype="float32"); m = json.loads((D / f"{cid}.json").read_text())
    q = json.loads((D / f"{cid.replace('_a','_q')}.json").read_text())
    return a, m.get("ref") or m["text"], q.get("ref") or q["text"], m["dur"]

def save(name, obj): (R / f"mini_{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1)); print(f"[{name}] saved", flush=True)

def t1():
    from .replay import cer
    from ..providers.stt import FasterWhisperSTT
    out = {}
    for model in ("base", "small", "large-v3-turbo"):
        stt = FasterWhisperSTT(model); stt.warm(); rows = []
        for cid in REAL:
            a, ref, _, dur = real(cid); t0 = time.time(); hyp = stt.transcribe(a); el = time.time() - t0
            rows.append({"clip": cid, "cer": round(cer(ref, hyp), 3), "rtf": round(el / (len(a) / 16000), 3), "sec": round(el, 1)})
        out[stt.name] = {"mean_cer": round(sum(r["cer"] for r in rows) / 3, 3), "mean_rtf": round(sum(r["rtf"] for r in rows) / 3, 3), "rows": rows}
        print("T1", stt.name, "CER", out[stt.name]["mean_cer"], "rtf", out[stt.name]["mean_rtf"], flush=True)
    save("t1_stt", out)

def t2():
    from ..providers.vad import SileroVAD
    from ..providers.stt import FasterWhisperSTT
    import re
    stt = FasterWhisperSTT("small"); stt.warm(); out = {"vad_only": {}, "semantic_hybrid": {}}
    FINAL = re.compile(r"([.?!]|다|요|죠|까|네요|니다|거든요|세요|어요|아요)\s*$")
    for ms in (400, 600, 1000, 1500, 2000):
        counts = {}
        for cid in REAL:
            a, *_ = real(cid); a = np.concatenate([a, np.zeros(3 * 16000, np.float32)]); v = SileroVAD(silence_ms=ms, own_model=True); n = 0
            for i in range(0, len(a) - 512, 512): n += v.feed(a[i:i + 512]) == "end"
            counts[cid] = n
        out["vad_only"][ms] = {"endpoints": counts, "mean": round(sum(counts.values()) / 3, 1)}; print("T2 vad", ms, counts, flush=True)
    # semantic hybrid: end at 600 ms only if the text so far looks complete, else wait until 1500 ms
    for cid in REAL:
        a, *_ = real(cid); a = np.concatenate([a, np.zeros(3 * 16000, np.float32)]); ve = SileroVAD(silence_ms=600, own_model=True); vl = SileroVAD(silence_ms=1500, own_model=True)
        buf = []; commits = 0; pend = False
        for i in range(0, len(a) - 512, 512):
            f = a[i:i + 512]; buf.append(f); re_, rl = ve.feed(f), vl.feed(f)
            if re_ == "end": pend = True
            if rl == "end": commits += 1; buf = []; pend = False
            elif re_ == "end":
                txt = stt.transcribe(np.concatenate(buf)[-int(30 * 16000):])
                if FINAL.search(txt): commits += 1; buf = []; pend = False
        out["semantic_hybrid"][cid] = commits
    out["semantic_hybrid"]["mean"] = round(sum(v for k, v in out["semantic_hybrid"].items()) / 3, 1); print("T2 semantic", out["semantic_hybrid"], flush=True)
    save("t2_endpoint", out)

JSYS = """평가위원의 꼬리질문을 채점한다. JSON만: {"relevance":1-5,"grounded":1-5,"depth":1-5,"natural":1-5,"note":"한 줄"}.
relevance=직전 답변 내용과 이어지는가, grounded=답변/요구사항의 구체적 내용에 근거하는가, depth=더 깊이 파고드는가, natural=자연스러운 한국어 구어체 질문 한 개인가."""

async def t3():
    """Follow-up quality on REAL pairs, local LLM only (generator and judge)."""
    from ..core.engine import QAEngine
    from ..core.llm import make_llm
    from ..contracts import Question
    from ..data.rfp import REQS
    llm = make_llm("ollama:qwen2.5:7b"); await llm.warm(); rows = []
    for cid in REAL:
        _, ans, q, _ = real(cid); e = QAEngine(llm, REQS, strategy="hybrid", interjection="rule"); cq = Question(q[:400], status="asked"); e.queue.items.append(cq); e._asked(cq)
        a = await e.analyze_answer(ans[:1500]); fu = ""
        async for act, tok in e.plan_and_stream(ans[:1500], a): fu += tok
        fu = fu.strip(); human = None
        hq = D / f"{cid.replace('_a','_q2')}.json"
        if hq.exists(): human = json.loads(hq.read_text()).get("ref") or json.loads(hq.read_text())["text"]
        async def judge(text): return await llm.json(JSYS, f"[질문]\n{q[:400]}\n[답변]\n{ans[:1200]}\n[꼬리질문]\n{text}", 150)
        rows.append({"clip": cid, "ai_followup": fu, "ai_scores": await judge(fu), "human_followup": human, "human_scores": await judge(human) if human else None})
        print("T3", cid, rows[-1]["ai_scores"], flush=True)
    save("t3_followup_REAL_local", rows)   # contains real text -> git-ignored by name pattern below

async def t4():
    from .run_real import run
    for tag, opts in (("baseline_small", dict(llm="ollama:qwen2.5:7b", stt="small", strategy="two_track", interjection="rule")),
                      ("spec1500_small", dict(llm="ollama:qwen2.5:7b", stt="small", strategy="two_track", interjection="rule", confirm_ms=1500)),
                      ("spec1500_turbo", dict(llm="ollama:qwen2.5:7b", stt="large-v3-turbo", strategy="two_track", interjection="rule", confirm_ms=1500))):
        print("T4 running", tag, flush=True); await run(REAL, opts, str(R / f"real_qa_mini_t4_{tag}.json"))

REC: dict = {}
def meta(**kw): REC.update(kw)

# ---------------- T5 interjection policies (synthetic, labelled) ----------------
async def t5():
    from .run import amain
    rows = {}
    for pol in ("rule", "llm", "hybrid"):
        t0 = time.time(); doc = await amain("B", dict(llm="ollama:qwen2.5:7b", stt="small", strategy="two_track", interjection=pol), SYN, None, realtime=True)
        rows[pol] = {"wall_sec": round(time.time() - t0, 1), "per_scenario": [{"id": r["id"], "expected": r["expect"]["interject"], "interjected": r["interjected"],
                     "kind": (r["interjections"][0]["kind"] if r["interjections"] else None), "t_after_speech_start_s": (round(r["interjections"][0]["t_after_speech_start"], 1) if r["interjections"] else None),
                     "interjection_audio_after_start_s": r.get("interjection_audio_after_speech_start_s")} for r in doc["results"]], "summary": doc["summary"]["interjection"]}
        print("T5", pol, rows[pol]["summary"], flush=True)
    save("t5_interjection", rows)

# ---------------- T6 cases A / B / C on the same synthetic audio ----------------
async def t6():
    from .run import amain
    from .run_livekit import amain as amain_lk
    SC = ["s01_normal", "s05_terms", "s06_numbers"]; out = {}
    base = dict(llm="ollama:qwen2.5:7b", stt="small", strategy="two_track", interjection="off")
    t0 = time.time(); b = await amain("B", base, SC, None, realtime=True); out["B"] = {"wall_sec": round(time.time() - t0, 1), "e2e": b["summary"]["e2e"], "rows": [(r["id"], r.get("e2e_s"), r.get("action"), r.get("question")) for r in b["results"]]}
    t0 = time.time(); a = await amain_lk(base, SC, None); out["A"] = {"wall_sec": round(time.time() - t0, 1), "e2e": a["summary"]["e2e"], "rows": [(r["id"], r.get("e2e_s"), r.get("action"), r.get("question")) for r in a["results"]]}
    from ..cases import case_c
    t0 = time.time(); out["C"] = await case_c.bench(SC); out["C"]["wall_sec"] = round(time.time() - t0, 1)
    for k in out: print("T6", k, out[k].get("e2e") or out[k].get("status"), flush=True)
    save("t6_cases", out)

# ---------------- T7 LLM comparison (synthetic only; cloud allowed) ----------------
async def t7():
    from ..core.engine import QAEngine
    from ..core.llm import make_llm
    from ..contracts import Question
    from ..data.rfp import REQS
    from . import dataset
    scn = {s["id"]: s for s in dataset.load()}; rows = []
    for spec in ("ollama:llama3.2:3b", "ollama:qwen3.5:4b", "ollama:qwen2.5:7b", "openai:gpt-4o-mini"):
        llm = make_llm(spec)
        if hasattr(llm, "warm"): await llm.warm()
        for sid in SYN:
            s = scn[sid]; e = QAEngine(llm, REQS, strategy="two_track", interjection="rule"); cq = Question(s["q"], status="asked"); e.queue.items.append(cq); e._asked(cq)
            a = e.fast_assess(s["text"]); t0 = time.time(); first = None; q = ""
            async for act, tok in e.plan_and_stream(s["text"], a):
                if first is None: first = time.time() - t0
                q += tok
            rows.append({"llm": spec, "scenario": sid, "action": act.value, "first_token_s": round(first or 0, 2), "total_s": round(time.time() - t0, 2), "question": q.strip()})
        print("T7 generated", spec, flush=True)
    judge = make_llm("openai:gpt-4o-mini")
    for r in rows:
        s = scn[r["scenario"]]; r["scores"] = await judge.json(JSYS, f"[질문]\n{s['q']}\n[답변]\n{s['text']}\n[꼬리질문]\n{r['question']}", 150)
    agg = {}
    for r in rows:
        a = agg.setdefault(r["llm"], {"ft": [], "sc": []}); a["ft"].append(r["first_token_s"]); sc = r["scores"]; a["sc"].append(sum(sc.get(k, 0) for k in ("relevance", "grounded", "depth", "natural")) / 4)
    summary = {k: {"mean_first_token_s": round(sum(v["ft"]) / 3, 2), "mean_quality_1to5": round(sum(v["sc"]) / 3, 2)} for k, v in agg.items()}
    print("T7", summary, flush=True); save("t7_llm", {"summary": summary, "rows": rows})


# ---------------- T8 / T9 / T10: newest OpenAI models, ladder from cheapest/newest upward (synthetic data only) ----------------
JUDGE = "gpt-6.2-sol"; JUDGE_CHEAP = "gpt-6-luna"
LLM_LADDER = ["gpt-6-luna", "gpt-6.1-terra", "gpt-6.2-astra", "gpt-6.2-sol"]
RT_LADDER = ["gpt-realtime-2.1-mini", "gpt-realtime-2", "gpt-realtime-2.1"]
STT_CLOUD = ["gpt-transcribe", "gpt-live-transcribe", "gpt-realtime-whisper"]

def _score(sc): return round(sum(sc.get(k, 0) for k in ("relevance", "grounded", "depth", "natural")) / 4, 2)

async def t8():
    from ..core.engine import QAEngine
    from ..core.llm import make_llm
    from ..contracts import Question
    from ..data.rfp import REQS
    from . import dataset
    scn = {s["id"]: s for s in dataset.load()}; rows = []; judge = make_llm("openai:" + JUDGE); await judge.warm()
    for m in LLM_LADDER:
        llm = make_llm("openai:" + m)
        try: await llm.warm()
        except Exception as ex: rows.append({"llm": m, "error": repr(ex)[:160]}); print("T8", m, "ERR", repr(ex)[:100], flush=True); continue
        for sid in SYN:
            s = scn[sid]; e = QAEngine(llm, REQS, strategy="two_track", interjection="rule"); cq = Question(s["q"], status="asked"); e.queue.items.append(cq); e._asked(cq)
            a = e.fast_assess(s["text"]); t0 = time.time(); first = None; q = ""
            async for act, tok in e.plan_and_stream(s["text"], a):
                if first is None: first = time.time() - t0
                q += tok
            sc = await judge.json(JSYS, f"[질문]\n{s['q']}\n[답변]\n{s['text']}\n[꼬리질문]\n{q.strip()}", 150)
            rows.append({"llm": m, "scenario": sid, "action": act.value, "first_token_s": round(first or 0, 2), "total_s": round(time.time() - t0, 2), "question": q.strip(), "scores": sc, "score": _score(sc)})
        print("T8 done", m, flush=True)
    summ = {}
    for r in rows:
        if "error" in r: summ[r["llm"]] = {"error": r["error"]}; continue
        d = summ.setdefault(r["llm"], {"ft": [], "sc": []}); d["ft"].append(r["first_token_s"]); d["sc"].append(r["score"])
    summary = {k: ({"error": v["error"]} if "error" in v else {"mean_first_token_s": round(sum(v["ft"]) / len(v["ft"]), 2), "mean_quality_1to5": round(sum(v["sc"]) / len(v["sc"]), 2)}) for k, v in summ.items()}
    print("T8", summary, flush=True); save("t8_openai_llm_ladder", {"judge": JUDGE, "summary": summary, "rows": rows})

async def t9():
    from ..cases import case_c
    from ..core.llm import make_llm
    from . import dataset
    scn = {s["id"]: s for s in dataset.load()}; judge = make_llm("openai:" + JUDGE); await judge.warm(); out = {}
    for m in RT_LADDER:
        t0 = time.time(); r = await case_c.bench(SYN, model=m); r["wall_sec"] = round(time.time() - t0, 1)
        for row in r.get("rows", []):
            if row.get("question"):
                s = scn[row["id"]]; row["scores"] = await judge.json(JSYS, f"[질문]\n{s['q']}\n[답변]\n{s['text']}\n[꼬리질문]\n{row['question']}", 150); row["score"] = _score(row["scores"])
        sc = [x["score"] for x in r.get("rows", []) if "score" in x]; r["mean_quality_1to5"] = round(sum(sc) / len(sc), 2) if sc else None; out[m] = r
        print("T9", m, r.get("status"), r.get("e2e", {}).get("p50"), "quality", r["mean_quality_1to5"], flush=True)
    save("t9_openai_realtime_ladder", {"judge": JUDGE, "models": out})

async def t10():
    import io
    from openai import AsyncOpenAI
    from .replay import cer
    from ..providers.stt import FasterWhisperSTT
    from . import dataset
    scn = {s["id"]: s for s in dataset.load()}; ids = ["s05_terms", "s06_numbers", "s12_noise"]; cl = AsyncOpenAI(); out = {}
    for m in STT_CLOUD:
        rows = []
        for sid in ids:
            s = scn[sid]; t0 = time.time()
            try:
                with open(s["wav"], "rb") as f: r = await cl.audio.transcriptions.create(model=m, file=f, language="ko")
                rows.append({"id": sid, "cer": round(cer(s["ref"], r.text), 3), "sec": round(time.time() - t0, 2), "text": r.text})
            except Exception as ex: rows.append({"id": sid, "error": repr(ex)[:140]})
        ok = [x for x in rows if "cer" in x]; out[m] = {"mean_cer": round(sum(x["cer"] for x in ok) / len(ok), 3) if ok else None, "mean_sec": round(sum(x["sec"] for x in ok) / len(ok), 2) if ok else None, "rows": rows}
        print("T10", m, out[m]["mean_cer"], out[m]["mean_sec"], flush=True)
    import soundfile as sf
    for name in ("small", "large-v3-turbo"):
        stt = FasterWhisperSTT(name); stt.warm(); rows = []
        for sid in ids:
            a, _ = sf.read(scn[sid]["wav"], dtype="float32"); t0 = time.time(); h = stt.transcribe(a); rows.append({"id": sid, "cer": round(cer(scn[sid]["ref"], h), 3), "sec": round(time.time() - t0, 2)})
        out["local:" + name] = {"mean_cer": round(sum(x["cer"] for x in rows) / 3, 3), "mean_sec": round(sum(x["sec"] for x in rows) / 3, 2), "rows": rows}; print("T10 local", name, out["local:" + name]["mean_cer"], flush=True)
    save("t10_cloud_stt", out)

# ---------------- T11: which cheap/local judge can be trusted?  T3b: re-judge REAL follow-ups with the best LOCAL judge ----------------
JUDGES = ["ollama:qwen3:8b", "ollama:qwen2.5:7b", "openai:gpt-6-luna"]

def _pearson(a, b):
    a, b = np.array(a, float), np.array(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")

async def t11():
    """Agreement of each candidate judge with the reference judge (gpt-5.5) on the SYNTHETIC follow-ups produced in T8 (21 questions)."""
    from ..core.llm import make_llm
    from . import dataset
    t8 = json.loads((R / "mini_t8_openai_llm_ladder.json").read_text()); scn = {x["id"]: x for x in dataset.load()}
    items = [r for r in t8["rows"] if "question" in r]; out = {}
    for spec in JUDGES:
        j = make_llm(spec)
        if hasattr(j, "warm"): await j.warm()
        mine = []; t0 = time.time()
        for r in items:
            s = scn[r["scenario"]]; sc = await j.json(JSYS, f"[질문]\n{s['q']}\n[답변]\n{s['text']}\n[꼬리질문]\n{r['question']}", 150); mine.append(_score(sc))
        ref = [r["score"] for r in items]
        out[spec] = {"n": len(items), "pearson_vs_ref": round(_pearson(mine, ref), 3), "mean_abs_diff": round(float(np.mean(np.abs(np.array(mine) - np.array(ref)))), 2),
                     "mean_score": round(float(np.mean(mine)), 2), "ref_mean_score": round(float(np.mean(ref)), 2), "sec_per_judgement": round((time.time() - t0) / len(items), 2)}
        print("T11", spec, out[spec], flush=True)
    save("t11_judge_agreement", out)

async def t3b():
    """Real follow-ups (private, local only): generator = qwen2.5:7b, judge = best LOCAL judge from T11 (a different model, to avoid self-judging)."""
    from ..core.llm import make_llm
    agree = json.loads((R / "mini_t11_judge_agreement.json").read_text())
    local = {k: v for k, v in agree.items() if k.startswith("ollama:") and k != "ollama:qwen2.5:7b"}
    best = max(local, key=lambda k: (local[k]["pearson_vs_ref"] if local[k]["pearson_vs_ref"] == local[k]["pearson_vs_ref"] else -1))
    judge = make_llm(best); await judge.warm(); rows = json.loads((R / "mini_t3_followup_REAL_local.json").read_text()); out = []
    for r in rows:
        _, ans, q, _ = real(r["clip"])
        async def jd(text): return await judge.json(JSYS, f"[질문]\n{q[:400]}\n[답변]\n{ans[:1200]}\n[꼬리질문]\n{text}", 150)
        r2 = {"clip": r["clip"], "judge": best, "ai_followup": r["ai_followup"], "ai_scores": await jd(r["ai_followup"]), "human_followup": r["human_followup"], "human_scores": await jd(r["human_followup"]) if r["human_followup"] else None}
        r2["ai_score"] = _score(r2["ai_scores"]); r2["human_score"] = _score(r2["human_scores"]) if r2["human_scores"] else None; out.append(r2); print("T3b", r["clip"], r2["ai_score"], r2["human_score"], flush=True)
    save("t3b_followup_REAL_local_rejudged", {"judge": best, "rows": out})

# ---------------- T12: follow-up QUALITY of every already-measured configuration (speed alone is misleading) ----------------
async def t12():
    """Judge the follow-up question of every scenario of every measured run (synthetic data -> cloud judge allowed).
    usable_rate = share of scenarios where the first audio came within 2 s AND quality >= 3.5/5."""
    import glob
    from ..core.llm import make_llm
    from . import dataset
    scn = {x["id"]: x for x in dataset.load()}; judge = make_llm("openai:" + JUDGE_CHEAP); await judge.warm(); out = {}
    KEEP = ["matrix_B_precomputed_hybrid", "matrix_B_two_track_hybrid", "matrix_B_direct_hybrid"]      # existing results are reused as-is; only these 3 are quality-checked
    for f in [str(R / f"{k}.json") for k in KEEP]:
        d = json.loads(pathlib.Path(f).read_text()); rows = []
        for r in [x for x in d["results"] if x["id"] in SYN]:
            q = r.get("question") or ""; s_ = scn[r["id"]]
            sc = await judge.json(JSYS, f"[질문]\n{s_['q']}\n[답변]\n{s_['text']}\n[꼬리질문]\n{q}", 150) if q else {}
            qual = _score(sc) if sc else None
            rows.append({"id": r["id"], "e2e_s": r.get("e2e_s"), "action": r.get("action"), "question": q, "quality": qual,
                         "usable": bool(r.get("e2e_s") is not None and 0 <= r["e2e_s"] <= 2.0 and qual is not None and qual >= 3.5)})
        qs = [x["quality"] for x in rows if x["quality"] is not None]
        out[pathlib.Path(f).stem] = {"mean_quality_1to5": round(sum(qs) / len(qs), 2) if qs else None, "usable_rate": round(sum(x["usable"] for x in rows) / len(rows), 2), "rows": rows}
        print("T12", pathlib.Path(f).stem, "quality", out[pathlib.Path(f).stem]["mean_quality_1to5"], "usable", out[pathlib.Path(f).stem]["usable_rate"], flush=True)
    save("t12_quality_matrix", {"judge": JUDGE_CHEAP, "runs": out})

# ---------------- T13: manual turn (user presses 'answer done'; in tests assumed pressed exactly at the real speech end) ----------------
async def t13():
    from .run import amain
    from .run_real import run
    SC = ["s01_normal", "s05_terms", "s06_numbers"]; out = {}
    man = dict(llm="ollama:qwen2.5:7b", stt="small", strategy="two_track", interjection="off", turn_mode="manual")
    t0 = time.time(); b = await amain("B", man, SC, None, realtime=True)
    out["synthetic_manual"] = {"wall_sec": round(time.time() - t0, 1), "e2e": b["summary"]["e2e"], "rows": [(r["id"], r.get("e2e_s"), r.get("stages"), r.get("question")) for r in b["results"]]}
    t0 = time.time(); await run(REAL, man, str(R / "real_qa_mini_t13_manual.json")); d = json.loads((R / "real_qa_mini_t13_manual.json").read_text())
    out["real_manual"] = {"wall_sec": round(time.time() - t0, 1), "summary": d["summary"], "rows": [{"id": r["id"], "endpoints": r["endpoints_in_answer"], "e2e": r["e2e_after_clip_end_s"]} for r in d["results"]]}
    print("T13 synthetic", out["synthetic_manual"]["e2e"], flush=True); print("T13 real", out["real_manual"]["summary"], flush=True)
    save("t13_manual_turn", out)

def timed(name, fn, *args):
    """Wall-clock seconds for a whole test (model load + all 3 samples), appended to results/mini_timing.json."""
    t0 = time.time(); r = fn(*args)
    if asyncio.iscoroutine(r): asyncio.run(r)
    f = R / "mini_timing.json"; d = json.loads(f.read_text()) if f.exists() else {}
    d[name] = {"wall_sec": round(time.time() - t0, 1), "when": time.strftime("%H:%M:%S")}; f.write_text(json.dumps(d, indent=1)); print(f"[{name}] took {d[name]['wall_sec']} s", flush=True)

if __name__ == "__main__":
    which = sys.argv[1]
    table = {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5, "t6": t6, "t7": t7, "t8": t8, "t9": t9, "t10": t10, "t11": t11, "t3b": t3b, "t12": t12, "t13": t13}
    timed(which, table[which])
