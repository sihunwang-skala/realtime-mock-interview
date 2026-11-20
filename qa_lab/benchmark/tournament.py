"""Tournament: vary ONE axis at a time (everything else fixed at the current best), keep the top 2, move on.
Ranking rules (documented in docs/TOURNAMENT_PLAN.md):
  STT           -> lowest CER, then fastest
  strategy/LLM  -> highest 'usable rate' (first audio <= 2.0 s AND follow-up quality >= 3.5/5), then quality, then speed
  interjection  -> F1 on labelled scenarios, then fewest false interjections
  TTS           -> fastest time to first audio
Everything runs in manual-turn mode (the user presses 'answer done'; in tests the press is assumed exactly at the real speech end).
Synthetic audio only (cloud judge / cloud STT allowed)."""
from __future__ import annotations
import asyncio, json, pathlib, sys, time
from .run import amain
from .mini_suite import JSYS, JUDGE_CHEAP, _score
from ..core.llm import make_llm
from . import dataset

R = pathlib.Path(__file__).resolve().parents[2] / "results"
E2E_SET = ["s01_normal", "s05_terms", "s06_numbers"]; IJ_SET = ["s01_normal", "s08_contradict_rfp", "s11_ramble"]
BASE = dict(llm="ollama:qwen2.5:7b", stt="small", tts="av", strategy="two_track", interjection="off")
_JUDGE = {}

async def judge_quality(results, scn):
    j = _JUDGE.get("j") or make_llm("openai:" + JUDGE_CHEAP); _JUDGE["j"] = j
    out = []
    for r in results:
        q = r.get("question") or ""
        if not q: out.append(None); continue
        s = scn[r["id"]]; sc = await j.json(JSYS, f"[질문]\n{s['q']}\n[답변]\n{s['text']}\n[꼬리질문]\n{q}", 150); out.append(_score(sc))
    return out

async def evaluate(cfg: dict, label: str, scenarios=None, with_quality=True):
    scenarios = scenarios or E2E_SET; scn = {s["id"]: s for s in dataset.load()}; t0 = time.time()
    opts = {**cfg, "turn_mode": "manual"}
    doc = await amain("B", opts, scenarios, None, realtime=True); res = doc["results"]
    quals = await judge_quality(res, scn) if with_quality else [None] * len(res)
    e2e = [r.get("e2e_s") for r in res]; ok = [x for x in e2e if x is not None]
    usable = [bool(e is not None and 0 <= e <= 2.0 and q is not None and q >= 3.5) for e, q in zip(e2e, quals)]
    cers = [r["cer"] for r in res if "cer" in r]; ij = [(r["expect"]["interject"], r["interjected"]) for r in res]
    tp = sum(a and b for a, b in ij); fp = sum((not a) and b for a, b in ij); fn = sum(a and (not b) for a, b in ij)
    f1 = round(2 * tp / (2 * tp + fp + fn), 3) if (2 * tp + fp + fn) else None
    m = {"label": label, "cfg": cfg, "wall_sec": round(time.time() - t0, 1), "mean_e2e": round(sum(ok) / len(ok), 2) if ok else None, "under_2s": round(sum(1 for x in ok if x <= 2.0) / len(res), 2),
         "mean_quality": round(sum(q for q in quals if q is not None) / max(1, sum(q is not None for q in quals)), 2) if with_quality else None,
         "usable_rate": round(sum(usable) / len(res), 2), "mean_cer": round(sum(cers) / len(cers), 3) if cers else None, "interj_f1": f1, "false_interj": fp,
         "rows": [{"id": r["id"], "e2e": r.get("e2e_s"), "quality": q, "question": r.get("question"), "interjected": r["interjected"]} for r, q in zip(res, quals)]}
    print(f"[tournament] {label}: e2e {m['mean_e2e']}s quality {m['mean_quality']} usable {m['usable_rate']} CER {m['mean_cer']} F1 {m['interj_f1']} ({m['wall_sec']}s)", flush=True)
    return m

KEYS = {"stt": lambda m: (m["mean_cer"] if m["mean_cer"] is not None else 9, m["mean_e2e"] or 99),
        "usable": lambda m: (-m["usable_rate"], -(m["mean_quality"] or 0), m["mean_e2e"] or 99),
        "ij": lambda m: (-(m["interj_f1"] or 0), m["false_interj"]), "tts": lambda m: (m["mean_e2e"] or 99,)}

async def axis(name, field, values, base, rule, scenarios=None, keep=2, with_quality=True):
    rows = []
    for v in values: rows.append(await evaluate({**base, field: v}, f"{name}={v}", scenarios, with_quality))
    rows.sort(key=KEYS[rule]); survivors = [r["cfg"][field] for r in rows[:keep]]
    print(f"[tournament] {name} survivors: {survivors}", flush=True)
    return {"axis": name, "field": field, "rule": rule, "standings": rows, "survivors": survivors}

async def stage1(base=BASE):
    out = {"base": base, "axes": []}; cfg = dict(base)
    a = await axis("STT", "stt", ["base", "small", "large-v3-turbo", "gpt-transcribe"], cfg, "stt"); out["axes"].append(a); cfg["stt"] = a["survivors"][0]
    a = await axis("질문 전략", "strategy", ["direct", "hybrid", "precomputed", "two_track"], cfg, "usable"); out["axes"].append(a); cfg["strategy"] = a["survivors"][0]
    a = await axis("끼어들기 정책", "interjection", ["rule", "llm", "hybrid", "off"], cfg, "ij", scenarios=IJ_SET, with_quality=False); out["axes"].append(a)
    cfg["interjection"] = a["survivors"][0]
    a = await axis("TTS", "tts", ["av", "say"], cfg, "tts"); out["axes"].append(a)
    out["best_single_config"] = cfg; out["survivors"] = {x["axis"]: x["survivors"] for x in out["axes"]}
    (R / "tournament_stage1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1)); print("[tournament] stage1 saved; best:", cfg, flush=True)

EXT = "http://127.0.0.1:11435"; INT = "http://127.0.0.1:11434"
REST_SEC = 30          # cool-down between local models (laptop heat); user asked to shorten to 30 s
# gemma4:26b, qwen3.6:27b, qwen3.6:35b and gpt-oss:20b are excluded (reasoning/too large for this 32 GB machine; gpt-oss also timed out loading): too large for this 32 GB machine (slow, hot).  The cloud model runs first (no local heat).
LLM_CANDIDATES = [("openai", "gpt-6-luna")] + [("ollama", m) for m in ["qwen2.5:7b", "qwen3:8b", "qwen3.5:4b", "llama3.1:8b", "llama3.2:3b"]] + \
                 [("ollamax", m) for m in ["phi4-mini:3.8b", "exaone3.5:7.8b", "exaone-deep:7.8b", "aya-expanse:8b", "qwen3.5:9b", "gemma4:12b", "ministral-3:14b"]]

async def unload_all():
    import httpx
    async with httpx.AsyncClient(timeout=30) as c:
        for url in (INT, EXT):
            try:
                ps = (await c.get(url + "/api/ps")).json().get("models", [])
                for m in ps: await c.post(url + "/api/generate", json={"model": m["name"], "keep_alive": 0})
            except Exception: pass

async def stage2():
    """LLM tournament under the best Stage-1 configuration (interjection off so only question quality/speed is compared)."""
    s1 = json.loads((R / "tournament_stage1.json").read_text()); cfg = {**s1["best_single_config"], "interjection": "off"}
    prev = json.loads((R / "tournament_stage2.json").read_text())["axes"][0]["standings"] if (R / "tournament_stage2.json").exists() else []
    rows = [r for r in prev if r.get("mean_e2e") is not None and not r.get("error")]          # resume: keep valid rows, re-measure the rest
    done = {r["cfg"]["llm"] for r in rows}
    from ..cases import case_b
    for kind, m in LLM_CANDIDATES:
        spec = f"{kind}:{m}"
        if spec in done: continue
        await unload_all()
        old = case_b._STT.pop("tts:av", None)                                                 # fresh TTS worker for every model (a stuck worker silently produced no audio)
        if old is not None:
            try: old.p.kill()
            except Exception: pass
        try: row = await asyncio.wait_for(evaluate({**cfg, "llm": spec}, f"LLM={spec}"), 900)
        except Exception as ex:
            row = {"label": f"LLM={spec}", "cfg": {**cfg, "llm": spec}, "mean_e2e": None, "under_2s": 0, "mean_quality": None, "usable_rate": 0.0, "mean_cer": None, "interj_f1": None, "false_interj": 0, "rows": [], "error": repr(ex)[:160], "wall_sec": None}
            print(f"[tournament] LLM={spec}: FAILED {repr(ex)[:120]}", flush=True)
        rows.append(row)
        if kind != "openai":
            await unload_all(); print(f"[tournament] cooling down {REST_SEC}s", flush=True); await asyncio.sleep(REST_SEC)
        (R / "tournament_stage2.json").write_text(json.dumps({"base": cfg, "axes": [{"axis": "LLM", "field": "llm", "rule": "usable", "standings": sorted(rows, key=KEYS["usable"]), "survivors": []}]}, ensure_ascii=False, indent=1))
    ok = sorted([r for r in rows if r.get("mean_e2e") is not None], key=KEYS["usable"]); local = [r for r in ok if not r["cfg"]["llm"].startswith("openai")]
    surv = [r["cfg"]["llm"] for r in local[:2]]; cloud = [r["cfg"]["llm"] for r in ok if r["cfg"]["llm"].startswith("openai")]
    out = {"base": cfg, "axes": [{"axis": "LLM", "field": "llm", "rule": "usable", "standings": ok + [r for r in rows if r.get("mean_e2e") is None], "survivors": surv, "cloud_lane": cloud}]}
    (R / "tournament_stage2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1)); print("[tournament] stage2 local survivors:", surv, "| cloud lane:", cloud, flush=True)
    await unload_all()


# ---------------- extra Stage-2 candidates, and the (reduced) finals ----------------
async def stage2_extra(specs):
    """Measure more LLMs under the Stage-2 conditions, merge into tournament_stage2.json and recompute the survivors."""
    d = json.loads((R / "tournament_stage2.json").read_text()); cfg = d["base"]; ax = d["axes"][0]; rows = [m for m in ax["standings"] if m["cfg"]["llm"] not in specs]
    for spec in specs:
        await unload_all()
        from ..cases import case_b
        old = case_b._STT.pop("tts:av", None)
        if old is not None:
            try: old.p.kill()
            except Exception: pass
        try: rows.append(await asyncio.wait_for(evaluate({**cfg, "llm": spec}, f"LLM={spec}"), 900))
        except Exception as ex: print(f"[tournament] LLM={spec}: FAILED {repr(ex)[:120]}", flush=True)
        await unload_all(); await asyncio.sleep(REST_SEC)
    ok = sorted([r for r in rows if r.get("mean_e2e") is not None], key=KEYS["usable"]); local = [r for r in ok if not r["cfg"]["llm"].startswith("openai")]
    ax["standings"] = ok + [r for r in rows if r.get("mean_e2e") is None]; ax["survivors"] = [r["cfg"]["llm"] for r in local[:2]]; ax["cloud_lane"] = [r["cfg"]["llm"] for r in ok if r["cfg"]["llm"].startswith("openai")]
    (R / "tournament_stage2.json").write_text(json.dumps(d, ensure_ascii=False, indent=1)); print("[tournament] stage2 local survivors:", ax["survivors"], "| cloud lane:", ax["cloud_lane"], flush=True)

FIN_RUN = ["s01_normal", "s05_terms", "s06_numbers", "s08_contradict_rfp", "s11_ramble"]

async def evaluate_final(cfg, label):
    """One pass over 5 synthetic scenarios: speed/quality/CER from the 3 non-interjection ones, interjection F1 from the 3 labelled ones."""
    scn = {x["id"]: x for x in dataset.load()}; t0 = time.time(); doc = await amain("B", {**cfg, "turn_mode": "manual"}, FIN_RUN, None, realtime=True)
    by = {r["id"]: r for r in doc["results"]}; E = [by[i] for i in E2E_SET if i in by]; F = [by[i] for i in IJ_SET if i in by]
    quals = await judge_quality(E, scn); e2e = [r.get("e2e_s") for r in E]; ok = [x for x in e2e if x is not None]
    usable = [bool(e is not None and 0 <= e <= 2.0 and q is not None and q >= 3.5) for e, q in zip(e2e, quals)]
    ij = [(r["expect"]["interject"], r["interjected"]) for r in F]; tp = sum(a and b for a, b in ij); fp = sum((not a) and b for a, b in ij); fn = sum(a and (not b) for a, b in ij)
    m = {"label": label, "cfg": cfg, "wall_sec": round(time.time() - t0, 1), "mean_e2e": round(sum(ok) / len(ok), 2) if ok else None, "under_2s": round(sum(1 for x in ok if x <= 2.0) / max(len(E), 1), 2),
         "mean_quality": round(sum(q for q in quals if q is not None) / max(1, sum(q is not None for q in quals)), 2), "usable_rate": round(sum(usable) / max(len(E), 1), 2),
         "mean_cer": round(sum(r["cer"] for r in E if "cer" in r) / max(1, sum("cer" in r for r in E)), 3), "interj_f1": round(2 * tp / (2 * tp + fp + fn), 3) if (2 * tp + fp + fn) else None, "false_interj": fp,
         "rows": [{"id": r["id"], "e2e": r.get("e2e_s"), "question": r.get("question"), "interjected": r["interjected"]} for r in doc["results"]]}
    print(f"[tournament] {label}: e2e {m['mean_e2e']}s quality {m['mean_quality']} usable {m['usable_rate']} CER {m['mean_cer']} F1 {m['interj_f1']} ({m['wall_sec']}s)", flush=True); return m

async def finals():
    """Reduced finals: STT(2) x interjection(2) x LLM(2 local + 1 cloud) with strategy=two_track, TTS=AVSpeech, plus two sanity runs for the pruned alternatives."""
    s1 = json.loads((R / "tournament_stage1.json").read_text()); s2 = json.loads((R / "tournament_stage2.json").read_text())
    stt2 = [x for x in s1["axes"] if x["axis"] == "STT"][0]["survivors"]; ij2 = [x for x in s1["axes"] if x["axis"] == "끼어들기 정책"][0]["survivors"]
    ax = s2["axes"][0]; llm_local = ax["survivors"]; llm_cloud = ax["cloud_lane"][:1]
    base = {"strategy": "two_track", "tts": "av"}; plan = []
    for llm in llm_local + llm_cloud:
        for stt in stt2:
            for ij in ij2: plan.append({**base, "llm": llm, "stt": stt, "interjection": ij})
    best = {**base, "llm": llm_local[0], "stt": stt2[0], "interjection": ij2[0]}
    plan += [{**best, "strategy": "precomputed"}, {**best, "tts": "say"}]       # sanity: the alternatives pruned from the full grid
    out_path = R / "tournament_finals.json"; rows = []
    for i, cfg in enumerate(plan, 1):
        label = f"결승 {cfg['llm'].split(':', 1)[1]} + {cfg['stt']} + 끼어들기 {cfg['interjection']}" + (f" + 전략 {cfg['strategy']}" if cfg["strategy"] != "two_track" else "") + (f" + TTS {cfg['tts']}" if cfg["tts"] != "av" else "")
        if i > 1 and plan[i - 2]["llm"] != cfg["llm"]:
            await unload_all(); print("[tournament] cooling down", REST_SEC, "s", flush=True); await asyncio.sleep(REST_SEC)
        try: rows.append(await asyncio.wait_for(evaluate_final(cfg, label), 900))
        except Exception as ex: print(f"[tournament] {label}: FAILED {repr(ex)[:100]}", flush=True)
        out_path.write_text(json.dumps({"base": base, "plan_size": len(plan), "standings": rows}, ensure_ascii=False, indent=1))
    await unload_all(); print("[tournament] finals done:", len(rows), "of", len(plan), flush=True)

# ---------------- cloud round: only OpenAI models whose input AND output price are <= $5 per 1M tokens ----------------
CLOUD_MODELS = ["gpt-6-luna", "gpt-6.1-terra", "gpt-5.6-luna", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5-mini", "gpt-5-nano", "gpt-4.1-mini", "gpt-4o-mini"]

CLOUD_MODELS_10 = ["gpt-6.2-sol", "gpt-6.1-sol", "gpt-6-sol", "gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o", "o3"]      # output price $5 < x <= $10 per 1M tokens

async def cloud_round(models=CLOUD_MODELS, out_name="tournament_cloud.json", axis_name="클라우드 LLM(입출력 $5 이하)"):
    """Same Stage-2 conditions (best Stage-1 config, interjection off) with each affordable OpenAI model as the question generator."""
    s1 = json.loads((R / "tournament_stage1.json").read_text()); cfg = {**s1["best_single_config"], "interjection": "off"}
    prev = json.loads((R / out_name).read_text())["axes"][0]["standings"] if (R / out_name).exists() else []
    rows = [r for r in prev if r["cfg"]["llm"] not in {f"openai:{m}" for m in models}]          # keep what was already measured; never re-measure
    from ..cases import case_b
    for m in models:
        old = case_b._STT.pop("tts:av", None)
        if old is not None:
            try: old.p.kill()
            except Exception: pass
        try: rows.append(await asyncio.wait_for(evaluate({**cfg, "llm": f"openai:{m}"}, f"LLM=openai:{m}"), 600))
        except Exception as ex: print(f"[tournament] LLM=openai:{m}: FAILED {repr(ex)[:120]}", flush=True)
        (R / out_name).write_text(json.dumps({"base": cfg, "axes": [{"axis": axis_name, "field": "llm", "rule": "usable", "standings": sorted(rows, key=KEYS["usable"]), "survivors": []}]}, ensure_ascii=False, indent=1))
    ok = sorted([r for r in rows if r.get("mean_e2e") is not None], key=KEYS["usable"])
    print("[tournament] cloud round top 2:", [r["cfg"]["llm"] for r in ok[:2]], flush=True)

# ---------------- focused trio: repeat each model 3x; cross-judging (a model never judges its own questions) ----------------
async def trio(models=("gpt-6.2-sol", "gpt-6-luna", "gpt-6.1-terra"), reps=3):
    import statistics
    s1 = json.loads((R / "tournament_stage1.json").read_text()); cfg = {**s1["best_single_config"], "interjection": "off"}; scn = {x["id"]: x for x in dataset.load()}
    judges = {m: make_llm("openai:" + m) for m in models}
    for j in judges.values(): await j.warm()
    from ..cases import case_b
    rows = []
    for m in models:
        samples = []; t0 = time.time(); per_rep = []
        for rep in range(reps):
            old = case_b._STT.pop("tts:av", None)
            if old is not None:
                try: old.p.kill()
                except Exception: pass
            doc = await amain("B", {**cfg, "llm": f"openai:{m}", "turn_mode": "manual"}, E2E_SET, None, realtime=True); rep_e2e = []
            for r in doc["results"]:
                q = r.get("question") or ""; s = scn[r["id"]]; sc = []
                for jm, j in judges.items():
                    if jm == m or not q: continue
                    sc.append(_score(await j.json(JSYS, f"[질문]\n{s['q']}\n[답변]\n{s['text']}\n[꼬리질문]\n{q}", 150)))
                qual = round(sum(sc) / len(sc), 2) if sc else None; e = r.get("e2e_s"); rep_e2e.append(e)
                samples.append({"id": r["id"], "rep": rep + 1, "e2e": e, "quality": qual, "question": q, "cer": r.get("cer")})
            per_rep.append(round(sum(x for x in rep_e2e if x is not None) / max(1, sum(x is not None for x in rep_e2e)), 2))
        e2e = [x["e2e"] for x in samples if x["e2e"] is not None]; ql = [x["quality"] for x in samples if x["quality"] is not None]
        usable = [bool(x["e2e"] is not None and x["e2e"] <= 2.0 and x["quality"] is not None and x["quality"] >= 3.5) for x in samples]
        row = {"label": f"LLM=openai:{m} (3회 반복)", "cfg": {**cfg, "llm": f"openai:{m}"}, "wall_sec": round(time.time() - t0, 1), "mean_e2e": round(statistics.mean(e2e), 2), "e2e_std": round(statistics.pstdev(e2e), 2),
               "e2e_min": round(min(e2e), 2), "e2e_max": round(max(e2e), 2), "per_rep_mean_e2e": per_rep, "under_2s": round(sum(x <= 2.0 for x in e2e) / len(samples), 2), "mean_quality": round(statistics.mean(ql), 2),
               "usable_rate": round(sum(usable) / len(samples), 2), "mean_cer": round(statistics.mean(x["cer"] for x in samples if x["cer"] is not None), 3), "interj_f1": None, "false_interj": 0, "rows": samples, "judges": [j for j in models if j != m]}
        rows.append(row); print(f"[tournament] {row['label']}: e2e {row['mean_e2e']}s (±{row['e2e_std']}, {row['e2e_min']}~{row['e2e_max']}) reps {per_rep} quality {row['mean_quality']} usable {row['usable_rate']} ({row['wall_sec']}s)", flush=True)
        (R / "tournament_trio.json").write_text(json.dumps({"base": cfg, "axes": [{"axis": "클라우드 3종 반복 비교", "field": "llm", "rule": "usable", "standings": sorted(rows, key=KEYS["usable"]), "survivors": []}]}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    asyncio.run(trio() if sys.argv[1] == "trio" else cloud_round(sys.argv[2:] or CLOUD_MODELS) if sys.argv[1] == "cloud" else cloud_round(CLOUD_MODELS_10, "tournament_cloud10.json", "클라우드 LLM(출력 $5~10)") if sys.argv[1] == "cloud10" else {"stage1": stage1, "stage2": stage2, "finals": finals}.get(sys.argv[1]) and {"stage1": stage1, "stage2": stage2, "finals": finals}[sys.argv[1]]() or stage2_extra(sys.argv[2:]))
