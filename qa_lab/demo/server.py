"""Demo server: Case selection API, browser voice WebSocket (Demo C), text sim (Demo A), voice sim (Demo B)."""
from __future__ import annotations
import asyncio, json, time, pathlib, importlib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from ..cases.registry import CASES, status, OPT
from ..contracts import Action, Cmd, EventType, Question
from ..data.rfp import SEED_BANK, REQS

STATIC = pathlib.Path(__file__).parent / "static"; RES = pathlib.Path(__file__).resolve().parents[2] / "results"
app = FastAPI(title="qa-lab demo"); STATE = {"active": None}; _BUILT: dict = {}

def validate(case: str, opts: dict) -> list[str]:
    """Return reasons a combination is unsupported (empty = ok)."""
    why = []
    st = status(case)
    if not CASES[case].get("live_demo", True): why.append(f"Case {case}: live browser demo not available (benchmark only)")
    if st["status"] in ("BLOCKED", "NOT_IMPLEMENTED"): why.append(f"Case {case} is {st['status']}: {st['reason']}")
    ALIAS = {"stt": lambda v: v if (":" in v or v.startswith("gpt-")) else f"faster-whisper:{v}", "tts": lambda v: {"av": "macos-avspeech:Yuna", "say": "macos-say:Yuna"}.get(v, v)}
    for k in ("llm", "stt", "tts", "strategy", "interjection"):
        v = opts.get(k); allowed = CASES[case]["options"].get(k, [])
        if not v or not allowed: continue
        v = ALIAS.get(k, lambda x: x)(v)
        if not any(a.split("@")[0] == v.split("@")[0] for a in allowed): why.append(f"{k}={opts.get(k)} is not supported by Case {case} (supported: {', '.join(allowed)})")
    return why

LIVE = pathlib.Path(__file__).resolve().parents[2] / "data" / "live_runs"      # real-voice runs: local only (git-ignored)
def save_run(sid, opts, report, passage_chars, passage_text=""):
    import datetime
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / f"{sid}.json").write_text(json.dumps({"id": sid, "saved_at": datetime.datetime.now().isoformat(timespec="seconds"), "config": opts, "passage_chars": passage_chars, "passage_text": passage_text, "report": report, "ratings": {}}, ensure_ascii=False, indent=1, default=str))
    return sid

@app.get("/api/live_runs")
def live_runs():
    out = []
    for f in sorted(LIVE.glob("s*.json"), reverse=True) if LIVE.exists() else []:
        try: d = json.loads(f.read_text())
        except Exception: continue
        m = d["report"].get("metrics", {}); rt = [v for v in d.get("ratings", {}).values() if isinstance(v, (int, float))]
        out.append({"id": d["id"], "saved_at": d["saved_at"], "config": d["config"], "passage_chars": d.get("passage_chars", 0), "wall_s": m.get("wall_s"), "n_turns": m.get("n_turns"),
                    "e2e_mean_s": m.get("e2e_mean_s"), "e2e_max_s": m.get("e2e_max_s"), "under_2s": m.get("under_2s"), "n_interjections": m.get("n_interjections"),
                    "ij": m.get("interjection_stats"), "cost": m.get("est_cost_usd"), "cloud": m.get("cloud"), "quality_mean": round(sum(rt) / len(rt), 2) if rt else None, "n_rated": len(rt)})
    return out

@app.get("/api/live_runs/{rid}")
def live_run(rid: str):
    f = LIVE / f"{pathlib.Path(rid).name}.json"
    return json.loads(f.read_text()) if f.exists() else JSONResponse({"error": "not found"}, 404)

def _grade_targets(rep: dict) -> list:
    """Turns that have an answer, with the question each answer belongs to (stored by the session, else rebuilt from the order of the conversation)."""
    cur = (rep.get("asked") or [""])[0]; out = []
    for i, t in enumerate(rep.get("turns", [])):
        if t.get("interjection"): continue
        if t.get("clarify"): continue                                    # "다시/더 쉽게 말씀해 주세요": not an answer, never graded; the question being answered stays the same
        if t.get("stt_text"): out.append((i, t.get("answered_q") or cur))
        if t.get("question"): cur = t["question"]
    return out

@app.post("/api/live_runs/{rid}/grade_plan")
def grade_plan(rid: str):
    """Which answers still need a verdict and a model answer (nothing is graded during the session: no LLM work while the person talks)."""
    f = LIVE / f"{pathlib.Path(rid).name}.json"
    if not f.exists(): return JSONResponse({"error": "not found"}, 404)
    d = json.loads(f.read_text()); rep = d.get("report", {})
    if not d.get("passage_text"): return {"todo": [], "reason": "no passage text saved for this run"}
    return {"todo": [i for i, _ in _grade_targets(rep) if not rep["turns"][i].get("model_answer") or rep["turns"][i].get("verdict_pending") or not rep["turns"][i].get("verdict")]}

@app.post("/api/live_runs/{rid}/grade")
async def grade_one(rid: str, body: dict):
    """Grade ONE answer (verdict + reason + model answer in a single LLM call) and store it in the run file, so opening the report again costs nothing."""
    f = LIVE / f"{pathlib.Path(rid).name}.json"
    if not f.exists(): return JSONResponse({"error": "not found"}, 404)
    d = json.loads(f.read_text()); rep = d.get("report", {}); idx = int(body.get("i", -1)); text = d.get("passage_text") or ""
    tg = dict(_grade_targets(rep))
    if idx not in tg or not text: return {"ok": False}
    from ..core.llm import make_llm
    from ..core.passage import split_passage
    from ..core.interjection import _kw
    from ..core.engine import QAEngine
    llm = make_llm(str((d.get("config") or {}).get("llm") or "ollama:phi4-mini:3.8b")); sents = split_passage(text, 40)
    t = rep["turns"][idx]; q = tg[idx]; kw = _kw(q)
    ref = (t.get("ref_text") or "").strip() or " ".join(x for x in sorted(sents, key=lambda z: -len(kw & _kw(z)))[:2])      # remembered source first; keyword search only for old runs
    ref = ref[:300]
    g = await QAEngine.grade_with(llm, ref, q, t["stt_text"])
    t["answered_q"] = q; t["verdict"], t["why"], t["ref"], t["model_answer"] = g["verdict"], g["why"], g["ref"], g["model_answer"]; t.pop("verdict_pending", None)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str)); return {"ok": True}

GRADE_LOCK = asyncio.Lock()
async def _grade_and_ledger(rid: str, opts: dict, led, practice: int):
    """Background, AFTER the report was sent: LLM-grade the answers (same grading as the report), then credit the ledger and write the credit text into the run file.
    One at a time, and never during a practice (it starts only after the session ended)."""
    from ..core import ledger as LD
    from ..core.llm import make_llm
    from ..core.engine import QAEngine
    f = LIVE / f"{pathlib.Path(rid).name}.json"
    try:
        async with GRADE_LOCK:
            d = json.loads(f.read_text()); rep = d.get("report", {}); llm = make_llm(str(opts.get("llm") or "ollama:phi4-mini:3.8b"))
            for idx, q in _grade_targets(rep):
                t = rep["turns"][idx]
                if t.get("verdict") and not t.get("verdict_pending") and t.get("model_answer"): continue
                ref = (t.get("ref_text") or "").strip()[:300]
                if not ref: continue
                if t.get("model_answer"):                          # the model answer was already prepared during the practice: only the verdict is missing (short call; "모름" needs no call at all)
                    g = await QAEngine.judge_with(llm, ref, q, t["stt_text"]); g["model_answer"] = t["model_answer"]
                else: g = await QAEngine.grade_with(llm, ref, q, t["stt_text"])
                t["answered_q"] = q; t["verdict"], t["why"], t["ref"], t["model_answer"] = g["verdict"], g["why"], g["ref"], g["model_answer"]; t.pop("verdict_pending", None)
            res = LD.collect(rep)
            credits = led.apply(practice, [{k: v for k, v in r.items() if k != "turn"} for r in res])
            for r in res: r["turn"]["credit"] = credits[LD.qkey(r["q"], r["ref"])]
            f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str)); led.save()
    except Exception as ex: print("[ledger] grading/update failed:", repr(ex)[:300], flush=True)

@app.post("/api/ledger/rebuild")
def ledger_rebuild(body: dict):
    """Build the ledger of a passage from earlier saved runs (oldest first; only LLM-graded answers count). Practice numbers follow the run order."""
    from ..core import ledger as LD
    key = LD.pkey(str(body.get("passage") or STATE.get("passage") or "")); led = LD.Ledger(key); led.d = {"practice": 0, "items": {}}; n = 0
    runs = sorted((json.loads(p.read_text()) for p in LIVE.glob("s*.json")), key=lambda d: d.get("saved_at", ""))
    for d in runs:
        if LD.pkey(d.get("passage_text") or "") != key: continue
        res = LD.collect(d.get("report", {}))
        if not res: continue
        n += 1; led.apply(n, [{k: v for k, v in r.items() if k != "turn"} for r in res])
    led.d["practice"] = n; led.save()
    return {"ok": True, "key": key, "practices": n, "questions": len(led.items), "retired": len(led.retired())}

def _summary_md(d: dict) -> str:
    """One readable report per run: settings, every number, every question and answer with timings, and the verdicts. Written next to the run file so it can be shared as is."""
    import statistics as st
    c = d.get("config") or {}; rep = d.get("report", {}); m = rep.get("metrics", {}) or {}; mt = m.get("turns", []) or []; turns = rep.get("turns", []) or []
    L = [f"# 실험 기록 {d.get('id','')}  ({d.get('saved_at','')})", "", "## 설정",
         "| 항목 | 값 |", "|---|---|",
         f"| 목소리 (TTS) | {c.get('tts','')} |", f"| 받아쓰기 (STT) | {c.get('stt','')} |", f"| 모델 (LLM) | {c.get('llm','')} |",
         f"| 질문 방식 | {c.get('strategy','')} |", f"| 끼어들기 방식 | {c.get('interjection','')} |", f"| 답변 끝내기 | {'버튼으로 직접' if c.get('turn_mode')=='manual' else '자동 감지'} |",
         f"| 지문 길이 | {d.get('passage_chars','')}자 |", "", "## 한눈에 보기",
         f"- 총 소요 {m.get('wall_s','-')}초, 턴 {m.get('n_turns','-')}개, 끼어들기 {m.get('n_interjections',0)}회",
         f"- 답변 끝 후 첫 소리: 평균 {m.get('e2e_mean_s','-')}초, 최대 {m.get('e2e_max_s','-')}초, 2초 이내 {m.get('under_2s','-')}"]
    rows = []; stt = []; kinds = {"SWITCH": "다음 질문", "DEEPEN": "꼬리질문", "CLARIFY": "꼬리질문", "VERIFY": "꼬리질문", "CHALLENGE": "꼬리질문(모순 확인)", "REPEAT": "다시 듣기"}
    for i, t in enumerate(turns):
        x = mt[i] if i < len(mt) else {}; sg = x.get("stages_s", {}) or {}
        kind = "끼어들기" if t.get("interjection") else kinds.get(str(t.get("action")), str(t.get("action") or ""))
        s_ = sg.get("받아쓰기 완료"); g_ = sg.get("질문 첫 글자"); v_ = sg.get("음성 첫 조각"); e_ = x.get("e2e_s"); a_ = t.get("at_end") or {}
        if s_ is not None: stt.append(s_)
        flag = "초과" if (e_ is not None and e_ > 2) else ""
        busy = ", ".join(k for k, v in (("보충 중", a_.get("bank_refill_running")), ("꼬리질문 작성 중", a_.get("followup_being_written"))) if v)
        rows.append(f"| {i+1} | {kind} | {x.get('answer_chars','')} | {s_ if s_ is not None else ''} | {g_ if g_ is not None else ''} | {v_ if v_ is not None else ''} | {e_ if e_ is not None else ''} | {flag} | {a_.get('load1','')} | {busy} | {a_.get('unasked_questions','')} |")
    if stt: L.append(f"- 받아쓰기: 평균 {round(st.mean(stt),2)}초, 중앙값 {round(st.median(stt),2)}초, 최대 {max(stt)}초 (조용할 때 기준값 약 0.5초, LLM이 같이 돌면 약 1.0초)")
    L += ["", "## 턴별 시간 (초)", "| # | 유형 | 답변 글자 | 받아쓰기 | 질문 첫 글자 | 음성 첫 조각 | 첫 소리까지 | 2초 | 부하(1분) | 답변 끝날 때 다른 작업 | 안 물어본 질문 |", "|---|---|---|---|---|---|---|---|---|---|---|"] + rows
    slow = [(i + 1, r) for i, r in enumerate(rows) if "| 초과 |" in r]
    L += ["", "## 2초를 넘긴 턴", *( [f"- {n}번: " + r.split("|")[2].strip() + " / 첫 소리까지 " + r.split("|")[7].strip() + "초 / 다른 작업: " + (r.split("|")[10].strip() or "없음") for n, r in slow] or ["- 없음"] )]
    L += ["", "## 질문과 답변 (사용자 목소리의 받아쓰기)", ""]
    cur = (rep.get("asked") or [""])[0]; L.append(f"**AI 첫 질문**: {cur}")
    ij_n = 0
    for i, t in enumerate(turns):
        if t.get("interjection"):
            ij = (rep.get("interjections") or [])[ij_n] if ij_n < len(rep.get("interjections") or []) else {}; ij_n += 1
            if ij.get("partial"): L += ["", f"**나 (끼어들기 전 말)**: {ij['partial']}"]
            sec = ij.get("t_after_speech_start"); L.append(f"- ⚡ AI 끼어들기 ({round(sec) if sec is not None else '?'}초째 · {ij.get('kind','')}{' · ' + ij['reason'] if ij.get('reason') else ''}): {t.get('question','')}"); continue
        if t.get("clarify"):                                             # asked to hear it again / more simply: not an answer, not graded
            L += ["", f"**나 (요청)**: {t.get('stt_text','')} → {'질문을 그대로 다시 들려줌' if t['clarify'] == 'repeat' else '쉽게 설명해 달라는 요청은 반영하지 않고 같은 질문을 그대로 다시 들려줌 (질문자 스탠스 유지)'}"]
            if t.get("question"): L += ["", f"**AI**: {t['question']}"]
            continue
        if t.get("stt_text"):
            L += ["", f"**나**: {t['stt_text']}"]
            if t.get("verdict"): L.append(f"- 평가: {t['verdict']}. {t.get('why','')}")
            if (t.get("credit") or {}).get("text"): L.append(f"- 누적 기록: {t['credit']['text']}")
            if t.get("model_answer"): L.append(f"- 실제 정답(모범 답안): {t['model_answer']}")
            elif t.get("verdict_pending") or not t.get("verdict"): L.append("- 평가와 정답: 아직 안 만들었어요 (내 실험 결과에서 상세를 열면 만들어져요)")
        if t.get("question"):
            SRC = {"user": "내가 추가한 질문", "bank": "랜덤 질문(지문 기반)", "passage": "첫 질문", "seed": "기본 질문", "skip": "건너뛰기로 만든 질문", "answer": "꼬리질문", "followup": "꼬리질문"}
            src_ = f" _({SRC.get(t.get('q_source'), t.get('q_source') or '출처 미상')} · 우선순위 점수 {t.get('q_priority', '')})_" if t.get("q_source") else ""
            L += ["", f"**AI**: {t['question']}{src_}"]
            if t.get("q_user_text"): L.append(f"- 내가 쓴 원문: {t['q_user_text']}")
    return "\n".join(L) + "\n"

@app.get("/api/live_runs/{rid}/summary")
def run_summary(rid: str):
    """Readable report text + a copy saved next to the run file (data/live_runs/<id>.summary.md) so it can be shared."""
    f = LIVE / f"{pathlib.Path(rid).name}.json"
    if not f.exists(): return JSONResponse({"error": "not found"}, 404)
    md = _summary_md(json.loads(f.read_text())); out = f.with_suffix(".summary.md"); out.write_text(md)
    return {"text": md, "path": str(out)}

@app.post("/api/live_runs/{rid}/rating")
def live_rating(rid: str, body: dict):
    f = LIVE / f"{pathlib.Path(rid).name}.json"
    if not f.exists(): return JSONResponse({"error": "not found"}, 404)
    d = json.loads(f.read_text()); d.setdefault("ratings", {})[str(body.get("turn"))] = body.get("score"); f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str)); return {"ok": True}

@app.delete("/api/live_runs/{rid}")
def live_delete(rid: str):
    f = LIVE / f"{pathlib.Path(rid).name}.json"
    if f.exists(): f.unlink()
    return {"ok": True}

@app.post("/api/passage")
async def set_passage(body: dict):
    from ..core.passage import BANK, bank_key, build_bank, MAX_CHARS
    STATE["passage"] = str(body.get("text", ""))[:MAX_CHARS]; llm = str(body.get("llm") or "ollama:phi4-mini:3.8b"); key = bank_key(STATE["passage"], llm)
    if STATE["passage"].strip() and key not in BANK:                            # start preparing questions + model answers right away, in the background
        if llm.startswith("ollamax:"): await asyncio.to_thread(ensure_ext_ollama)
        asyncio.create_task(build_bank(STATE["passage"], llm))
    return {"ok": True, "chars": len(STATE["passage"]), "key": key}

MAX_UPLOAD_TEXT = 150000

@app.post("/api/upload")
async def upload_material(file: UploadFile = File(...)):
    """Any document (PDF, PPTX, DOCX, XLSX, HTML, SRT/VTT, JSON, TXT ...): plain text out, format detected from the bytes. Nothing is stored."""
    from ..core.extract import extract
    raw = await file.read(); res = await asyncio.to_thread(extract, file.filename or "", raw)
    return {"ok": bool(res.text.strip()), "name": file.filename, "fmt": res.fmt, "doc_type": res.doc_type, "pages": res.pages,
            "chars": len(res.text), "text": res.text[:MAX_UPLOAD_TEXT], "warnings": res.warnings}

@app.post("/api/user_questions")
async def set_user_questions(body: dict):
    """Questions the user wants asked (rough wording is fine). Stored for the next session start, like the passage; sessions turn them into proper questions."""
    from ..core.user_questions import MAX_USER_Q, convert
    from ..core.llm import make_llm
    qs = [" ".join(str(x).split())[:300] for x in (body.get("questions") or []) if str(x).strip()][:MAX_USER_Q]
    llm = make_llm(str(body.get("llm") or "ollama:phi4-mini:3.8b"))
    STATE["user_questions"] = [await convert(llm, x, STATE.get("passage") or "") for x in qs]       # converted NOW (before the practice, nobody is waiting), so starting a session needs no LLM call
    return {"ok": True, "n": len(qs), "items": STATE["user_questions"]}

@app.get("/api/passage_status/{key}")
def passage_status(key: str):
    from ..core.passage import BANK
    b = BANK.get(key)
    return {"n": len(b["qs"]), "done": b["done"], "extending": bool(b.get("extending")), "exhausted": bool(b.get("exhausted")), "rounds": b.get("rounds", 1)} if b else {"n": 0, "done": False, "extending": False, "exhausted": False, "rounds": 0}

@app.get("/")
def index(): return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store, max-age=0"})

@app.get("/v2")
def index_v2(): return FileResponse(STATIC / "index_v2.html", headers={"Cache-Control": "no-store, max-age=0"})
SAMPLES = pathlib.Path(__file__).resolve().parents[2] / "samples" / "tts"      # voice comparison samples (git-ignored)

@app.get("/api/tts_samples")
def tts_samples():
    f = SAMPLES / "index.json"
    return json.loads(f.read_text()) if f.exists() else {"sentences": [], "items": []}

@app.get("/tts_samples/{name}")
def tts_sample_file(name: str):
    f = SAMPLES / pathlib.Path(name).name
    return FileResponse(f, headers={"Cache-Control": "no-store"}) if f.exists() else JSONResponse({"error": "not found"}, 404)

@app.get("/api/ui_version")
def ui_version(): return {"v": (STATIC / "index.html").stat().st_mtime_ns}

@app.get("/api/cases")
def api_cases():
    return [{"id": k, "name": v["name"], "transport": v["transport"], "local_only": v["local_only"], "options": v["options"], "caps": v["caps"], **status(k)} for k, v in CASES.items()]
@app.post("/api/validate")
def api_validate(body: dict): return {"errors": validate(body.get("case", "B"), body)}
@app.get("/api/key_status")
def api_key_status():
    """Only says whether a key is already stored (in .env or the environment). The key itself is never returned."""
    env = pathlib.Path(__file__).resolve().parents[2] / ".env"
    in_file = env.exists() and any(l.startswith("OPENAI_API_KEY=") and len(l.split("=", 1)[1].strip()) > 8 for l in env.read_text().splitlines())
    return {"set": bool(in_file or os.environ.get("OPENAI_API_KEY"))}

@app.post("/api/key")
def api_key(body: dict):
    """Store OPENAI_API_KEY in git-ignored ai/qa-lab/.env (+ process env) and probe it. The value is never returned or logged."""
    import os
    k = (body.get("key") or "").strip()
    if not k.startswith("sk-") or len(k) < 20: return JSONResponse({"ok": False, "error": "형식이 올바르지 않습니다"}, 400)
    env = pathlib.Path(__file__).resolve().parents[2] / ".env"
    lines = [l for l in (env.read_text().splitlines() if env.exists() else []) if not l.startswith("OPENAI_API_KEY=")]
    env.write_text("\n".join(lines + [f"OPENAI_API_KEY={k}"]) + "\n"); env.chmod(0o600); os.environ["OPENAI_API_KEY"] = k
    from ..cases import case_c
    ok, why = case_c.probe(); return {"ok": ok, "detail": why, "case_c": status("C")}
@app.get("/api/listen")
def api_listen():
    """Scenario audio (replayed user answer) + what the agent heard/said (Case B two_track+hybrid run) for listening in the UI."""
    root = pathlib.Path(__file__).resolve().parents[2]; f = RES / "matrix_B_two_track_hybrid.json"
    if not f.exists(): return []
    out = []
    for r in json.loads(f.read_text())["results"]:
        sid = r["id"]; meta = json.loads((root / "data/audio" / f"{sid}.json").read_text())
        out.append({"id": sid, "question": meta["q"], "user_text": meta["text"], "stt": r.get("stt_text"), "action": r.get("action"), "agent_question": r.get("question"), "e2e_s": r.get("e2e_s"),
                    "interjected": r["interjected"], "interjections": r.get("interjections", []), "user_audio": f"/audio/{sid}.wav", "q_audio": "/audio_out/q_api.wav" if "API" in meta["q"] else "/audio_out/q_sec.wav", "agent_audio": f"/audio_out/{sid}.wav" if (root / "data/audio_out" / f"{sid}.wav").exists() else None})
    return out
@app.get("/audio/{name}")
def audio_in(name: str): return FileResponse(pathlib.Path(__file__).resolve().parents[2] / "data/audio" / pathlib.Path(name).name, media_type="audio/wav")
@app.get("/audio_out/{name}")
def audio_out(name: str): return FileResponse(pathlib.Path(__file__).resolve().parents[2] / "data/audio_out" / pathlib.Path(name).name, media_type="audio/wav")
@app.get("/api/listen_real")
def api_listen_real():
    """LOCAL ONLY: all real Q&A pairs cut from the recordings (+ pipeline results when available). Nothing here is committed."""
    root = pathlib.Path(__file__).resolve().parents[2]; D = root / "data/real_qa"
    res = {r["id"]: r for r in json.loads((RES / "real_qa_B.json").read_text())["results"]} if (RES / "real_qa_B.json").exists() else {}
    FU = {"r1_a": "r1_q2", "r3_a": "r3_q2", "r2_a": "r2_q2"}; out = []
    def J(i):
        f = D / f"{i}.json"; return json.loads(f.read_text()) if f.exists() else None
    for af in sorted(D.glob("*_a.json")) + sorted(D.glob("r*_a2.json")):
        a = json.loads(af.read_text()); aid = a["id"]; qid = aid.replace("_a2", "_q2") if aid.endswith("_a2") else aid.replace("_a", "_q"); q = J(qid)
        if not q: continue
        fu = J(FU.get(aid)) if aid in FU else None
        r = res.get(aid, {})
        t0 = q.get("start", a["start"]) if not q.get("no_audio") else a["start"]
        out.append({"id": aid, "src": a["src"][:10], "t": round(min(t0, a["start"]), 1), "auto": bool(a.get("auto")), "topic": a.get("topic", ""), "answer_dur_s": a["dur"],
                    "question": q.get("ref") or q["text"], "answer": a.get("ref") or a["text"], "q_audio": f"/real_audio/{qid}.wav?v={int((D / f'{qid}.wav').stat().st_mtime)}" if (D / f"{qid}.wav").exists() and not q.get("q_audio_synth") else None, "q_synth": bool(q.get("q_audio_synth")), "a_audio": f"/real_audio/{aid}.wav?v={int((D / f'{aid}.wav').stat().st_mtime)}",
                    "human_followup": (fu.get("ref") or fu["text"]) if fu else None, "followup_audio": f"/real_audio/{FU[aid]}.wav" if fu and (D / f"{FU[aid]}.wav").exists() else None,
                    "endpoints_in_answer": r.get("endpoints_in_answer"), "turns": r.get("turns", []), "interjections": r.get("interjections", [])})
    out.sort(key=lambda x: (x["src"], x["t"]))      # chronological: recording date, then time inside the recording
    return out
@app.post("/api/real_delete")
def real_delete(body: dict):
    """Remove a reviewed-bad pair (both its Q and A clip) from the local set; recorded in data/real_qa_exclude.json so recuts don't bring it back."""
    root = pathlib.Path(__file__).resolve().parents[2]; D = root / "data/real_qa"; aid = pathlib.Path(body.get("id", "")).name
    if not aid.endswith(("_a", "_a2")) or not (D / f"{aid}.json").exists(): return JSONResponse({"ok": False, "error": "unknown pair"}, 404)
    qid = aid[:-3] + "_q2" if aid.endswith("_a2") else aid[:-2] + "_q"
    trash = root / "data/real_qa_deleted"; trash.mkdir(exist_ok=True)
    for i in (aid, qid):
        for ext in (".wav", ".json"):
            f = D / f"{i}{ext}"
            if f.exists(): f.rename(trash / f.name)
    ex = root / "data/real_qa_exclude.json"; cur = set(json.loads(ex.read_text())) if ex.exists() else set(); cur |= {aid, qid}; ex.write_text(json.dumps(sorted(cur)))
    return {"ok": True, "deleted": [aid, qid]}
@app.get("/real_audio/{name}")
def real_audio(name: str): return FileResponse(pathlib.Path(__file__).resolve().parents[2] / "data/real_qa" / pathlib.Path(name).name, media_type="audio/wav", headers={"Cache-Control": "no-store"})
@app.get("/api/ranking")
def api_ranking():
    """Composite ranking of tournament combinations (recomputed from the saved tournament results each time)."""
    from ..benchmark import rank
    rows, unr = rank.collect(True)
    return {"rows": rows, "unranked": unr, "weights": rank.WEIGHTS, "labels": rank.LABEL}
@app.get("/api/results")
def api_results():
    """All measurement results for the UI: synthetic matrix (+Case A, LLM runs), real-speech results (local only), mini suite T1-T7 with wall times."""
    def J(n):
        f = RES / n; return json.loads(f.read_text()) if f.exists() else None
    out = {"synthetic": [], "real_stt": J("real_stt_cer.json"), "real_endpointing": J("real_endpointing.json"), "mini": {}, "t4": [], "timing": J("mini_timing.json") or {}}
    for pat in ("matrix_*.json", "case_*_*.json", "llm_*.json"):
        for f in sorted(RES.glob(pat)):
            d = json.loads(f.read_text()); sm = d["summary"]
            out["synthetic"].append({"run": f.stem, "case": d.get("case"), "llm": d["components"].get("llm"), "stt": d["components"].get("stt"), "e2e": sm["e2e"], "cer": sm["mean_cer"], "interj": sm["interjection"], "timeout": sm["timeout_rate"]})
    for k, n in (("t1", "mini_t1_stt.json"), ("t2", "mini_t2_endpoint.json"), ("t5", "mini_t5_interjection.json"), ("t6", "mini_t6_cases.json"), ("t7", "mini_t7_llm.json"), ("t8", "mini_t8_openai_llm_ladder.json"), ("t9", "mini_t9_openai_realtime_ladder.json"), ("t10", "mini_t10_cloud_stt.json"), ("t11", "mini_t11_judge_agreement.json"), ("t12", "mini_t12_quality_matrix.json")):
        out["mini"][k] = J(n)
    for tag, lab in (("baseline_small", "기본: small, 침묵 0.6초"), ("spec1500_small", "미리 생성+1.5초 확인, small"), ("spec1500_turbo", "미리 생성+1.5초 확인, turbo")):
        d = J(f"real_qa_mini_t4_{tag}.json")
        if d: out["t4"].append({"label": lab, "summary": d["summary"]})
    out["fastpath"] = J("fast_path_latency.json")                      # 0.6 s target: where the wait went and what the pre-made questions / pre-voiced sentences save
    return out
@app.post("/api/simulate")
async def api_simulate(body: dict):
    from ..simulator.text_sim import run
    return await run(fast=body.get("llm", "ollama:qwen2.5:7b"), presenter=body.get("presenter", "ollama:qwen2.5:7b"), strategy=body.get("strategy", "hybrid"), interjection=body.get("interjection", "hybrid"), turns=int(body.get("turns", 4)))
@app.post("/api/voice_sim")
async def api_voice_sim(body: dict):
    from ..simulator.voice_sim import run
    return await run(body, turns=int(body.get("turns", 3)))

def ensure_ext_ollama() -> bool:
    """External-drive models (ollamax:) need a second Ollama server on :11435 that reads the SAMSUNG library. Start it if needed."""
    import httpx, os, subprocess
    url = os.environ.get("OLLAMA_EXT_URL", "http://127.0.0.1:11435")
    def up():
        try: httpx.get(url + "/api/tags", timeout=2); return True
        except Exception: return False
    lib = pathlib.Path("/Volumes/SAMSUNG/LocalLLM-Library/ollama-models")
    if not lib.exists(): return False                    # drive unplugged: the running server can't read the models
    if up(): return True
    subprocess.Popen(["ollama", "serve"], env={**os.environ, "OLLAMA_MODELS": str(lib), "OLLAMA_HOST": "127.0.0.1:11435"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        time.sleep(0.5)
        if up(): return True
    return False

def get_build(case, opts):
    key = (case, json.dumps(opts, sort_keys=True))
    if key not in _BUILT: _BUILT.clear(); _BUILT[key] = importlib.import_module(f"qa_lab.cases.{CASES[case]['module']}").build(opts)
    return _BUILT[key]

def opts_from_query(q) -> dict:
    opts = {k: q.get(k) for k in ("llm", "stt", "tts", "strategy", "interjection") if q.get(k)}; opts.setdefault("llm", "ollama:qwen2.5:7b"); opts.setdefault("strategy", "two_track"); opts.setdefault("interjection", "hybrid"); opts.setdefault("tts", "supertonic:F1")
    manual = q.get("turn", "manual") == "manual"
    opts["turn_mode"] = "manual" if manual else "auto"
    if not manual: opts.setdefault("confirm_ms", "1500")
    return opts

@app.get("/api/prewarm")
async def api_prewarm(request: Request):
    """Load the models (Whisper, voice, LLM client) for this configuration BEFORE the user presses Start, so Start itself has nothing to wait for."""
    q = request.query_params; case = q.get("case", "B")
    if case == "C" or case not in CASES: return {"ok": False}
    opts = opts_from_query(q)
    if validate(case, opts): return {"ok": False}
    t0 = time.monotonic(); await asyncio.to_thread(get_build, case, opts)
    print(f"[timing] prewarm models {time.monotonic()-t0:.1f}s ({opts.get('tts')})", flush=True)
    return {"ok": True, "s": round(time.monotonic() - t0, 1)}

@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await ws.accept()
    q = ws.query_params; case = q.get("case", "B")
    if case == "C":                                            # Case C: relay to OpenAI Realtime (needs explicit consent: voice + RFP text leave this machine)
        if q.get("consent") != "1": await ws.send_json({"type": "ERROR", "errors": ["Case C sends your voice and the RFP text to OpenAI. Tick the consent box to start."]}); await ws.close(); return
        if STATE["active"]: STATE["active"].cancel()
        STATE["active"] = asyncio.current_task()
        try:
            from ..cases.case_c_live import run as run_c
            await run_c(ws, f"c{int(time.time())}", manual=(q.get("turn", "manual") == "manual"))
        except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError): pass
        except Exception as ex:
            try: await ws.send_json({"type": "ERROR", "errors": [repr(ex)[:300]]})
            except Exception: pass
        finally:
            if STATE["active"] is asyncio.current_task(): STATE["active"] = None
        return
    opts = opts_from_query(q); manual = opts["turn_mode"] == "manual"   # real use: the user ends each answer with the "answer done" button
    errs = validate(case, opts)
    if not errs and str(opts.get("llm", "")).startswith("ollamax:") and not await asyncio.to_thread(ensure_ext_ollama): errs = ["외장하드(SAMSUNG) 모델을 쓰려면 외장하드가 연결돼 있어야 해요."]
    if errs: await ws.send_json({"type": "ERROR", "errors": errs}); await ws.close(); return
    if STATE["active"]: STATE["active"].cancel()               # case switch: tear down the previous session
    sid = f"s{int(time.time())}"; T_START = time.monotonic(); out: asyncio.Queue = asyncio.Queue(); rt = None
    me = asyncio.current_task(); STATE["active"] = me
    try:
        from ..core import ledger as LD0
        _w = asyncio.ensure_future(LD0.wait_pending())             # waits for the last session's grading IN PARALLEL with the model load below (costs nothing extra)
        _t0 = time.monotonic(); _cached = (case, json.dumps(opts, sort_keys=True)) in _BUILT
        make_engine, make_runtime, info = await asyncio.to_thread(get_build, case, opts)
        _t1 = time.monotonic()
        eng = make_engine(); eng.keep_going = True
        passage = (STATE.get("passage") or "").strip() if q.get("passage") == "1" else ""
        if passage:
            from ..core.passage import apply_passage
            await apply_passage(eng, passage)
            print(f"[timing] models {'cached' if _cached else 'LOADED'} {_t1-_t0:.1f}s | first question {time.monotonic()-_t1:.1f}s", flush=True)
        else:
            for t in SEED_BANK: eng.queue.add(Question(t, source="seed", priority=.6))
        UQ_RES = []
        if STATE.get("user_questions"):                            # questions the user typed before starting (already converted, no LLM call): queued right after the opening question
            from ..core.user_questions import add_converted
            UQ_RES = add_converted(eng, STATE["user_questions"])
        from ..core import ledger as LD                            # earlier practices: wrong answers come back, mastered questions never do (one small file read, no LLM)
        LED = None; LEDP = 0
        try:
            await _w                                               # the previous session's grading (it ran while the models loaded above), at most 8 s
            LED = LD.Ledger(LD.pkey(passage)); LEDP = LED.practice + 1
            REVIEW = LD.inject(eng, LED, LEDP)
            if REVIEW: UQ_RES.append({"raw": "", "q": f"복습 질문 {len(REVIEW)}개를 넣었어요 (연습 {LEDP})", "status": "review"})
        except Exception as ex_: print("[ledger] skipped:", repr(ex_)[:200], flush=True); LED = None
        serverplay = q.get("serverplay") == "1"                   # play the AI voice on this Mac directly (afplay) -> independent of the browser's audio output
        SP = {"buf": [], "proc": None}
        def sp_kill():
            p_ = SP["proc"]
            if p_ is not None and p_.poll() is None: p_.terminate()
            SP["proc"] = None
        def sp_play():
            import wave, tempfile, subprocess, threading, os
            if not SP["buf"]: return
            pcm = np.concatenate(SP["buf"]); SP["buf"] = []
            fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            with wave.open(path, "wb") as w_: w_.setnchannels(1); w_.setsampwidth(2); w_.setframerate(16000); w_.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
            sp_kill(); pr = subprocess.Popen(["afplay", path]); SP["proc"] = pr; loop_ = asyncio.get_running_loop()
            def waiter():
                pr.wait()
                try: os.unlink(path)
                except Exception: pass
                if SP["proc"] is pr: loop_.call_soon_threadsafe(out.put_nowait, ("j", {"type": "AGENT_SPEAKING", "session_id": sid, "seq": 0, "data": {"state": "server_played"}}))
            threading.Thread(target=waiter, daemon=True).start()
        def on_ev(ev):
            d_ = ev.to_dict()
            if serverplay:
                st_ = (d_.get("data") or {}).get("state")
                if d_["type"] == "AGENT_THINKING": sp_kill(); SP["buf"] = []
                elif d_["type"] == "AGENT_SPEAKING" and st_ == "synth_done": sp_play()
                elif d_["type"] == "AGENT_SPEAKING" and st_ == "interrupted": sp_kill()
            out.put_nowait(("j", {**d_, "session_id": sid}))
        def on_aud(pcm, m_):
            if serverplay: SP["buf"].append(np.array(pcm, copy=True))
            else: out.put_nowait(("b", (np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes()))
        rt = make_runtime(eng, on_ev, on_aud)
        rt.sid = sid
        TIME_LIMIT_MIN = int(q.get("time_limit") or 0)            # session length chosen on the page: 0 = no limit, else 10 / 20 / 30 minutes
        rt.start_audio_prewarm()                                   # voice the prepared questions / follow-ups / interjection templates ahead of time (local voices only)
        try:                                                       # Whisper vocabulary hint: the passage's own terms (or the default RFP terms when no passage)
            if hasattr(rt.stt, "prompt"):
                if not hasattr(rt.stt, "_default_prompt"): rt.stt._default_prompt = rt.stt.prompt
                from ..core.passage import stt_terms
                rt.stt.prompt = (stt_terms(passage) if passage else rt.stt._default_prompt) or None
        except Exception: pass
        async def sender():
            while True:
                k, v = await out.get()
                await (ws.send_json(v) if k == "j" else ws.send_bytes(v))
        st = asyncio.create_task(sender()); await ws.send_json({"type": "SESSION_READY", "session_id": sid, "components": info, "seq": 0})
        if UQ_RES: await ws.send_json({"type": "QUESTION_ADDED", "session_id": sid, "results": UQ_RES})
        started = False; PAIR_LABELS: list[str] = []; SAVED = [False]
        while True:
            m = await ws.receive()
            if m.get("bytes") is not None:
                if started: await rt.feed(np.frombuffer(m["bytes"], np.int16).astype(np.float32) / 32768)
                continue
            if m.get("type") == "websocket.disconnect": break
            d = json.loads(m["text"]); t = d.get("type")
            if t == "ping": await ws.send_json({"type": "pong", "t0": d["t0"], "server_ts": time.monotonic()}); continue
            if t == "ptt": rt.ptt = bool(d.get("on")); continue
            if t == "add_question":                                # a question added during the session: queued as the next one (same conversion/dedup as the ones typed before start)
                from ..core.user_questions import add_converted, quick_convert, ref_for     # no LLM mid-session: it would slow the next turn
                _raw = " ".join(str(d.get("text", "")).split())[:300]
                res_ = add_converted(eng, [{"raw": _raw, "q": quick_convert(_raw), "ref": ref_for(passage, _raw)}])
                await ws.send_json({"type": "QUESTION_ADDED", "session_id": sid, "results": res_}); continue
            if t == "pair_label": PAIR_LABELS.append(str(d.get("label", ""))[:20]); continue
            if t == "audio_started": rt.client_played(d["server_clock_ts"]); continue
            if t == "played": eng.mark_played(int(len(eng.spoken_full) * min(1, d["seconds"] / max(d["total"], 1e-6)))); continue
            cmd = d.get("cmd"); res = {"type": "CMD_RESULT", "cmd": cmd, "ok": True}
            try:
                if cmd == Cmd.START.value:
                    if started: res.update(ok=False, error="already started")
                    else:
                        started = True; q0 = eng.first_question(); await rt.speak_text(q0.text); res["question_id"] = q0.id
                        if TIME_LIMIT_MIN: rt.arm_time_limit(TIME_LIMIT_MIN * 60)     # 10/20/30 min: the clock starts with the first question
                elif not started and cmd: res.update(ok=False, error="session not started")
                elif cmd == Cmd.REPEAT_QUESTION.value: await rt.speak_text(eng.cur.text, Action.REPEAT)
                elif cmd == Cmd.SKIP_QUESTION.value:
                    nq = await eng.next_question_now() if getattr(eng, "keep_going", False) else eng.queue.pop()
                    if not nq: res.update(ok=False, error="no more questions")
                    else:
                        prev_ = eng.cur; pa_ = eng.get_answer(prev_) if getattr(eng, "keep_going", False) else {"text": "", "src": ""}
                        eng.stop_prefetch(); eng._asked(nq)
                        await rt.speak_text(nq.text, extra={"prev_q": prev_.text if prev_ else "", "prev_answer": pa_["text"], "prev_src": pa_["src"], "skipped": True})
                elif cmd == Cmd.START_ANSWER.value:
                    ok_, why_ = await rt.begin_answer()
                    if not ok_: res.update(ok=False, error=why_)
                elif cmd == Cmd.FINISH_ANSWER.value:
                    if getattr(rt, "turn_mode", "auto") == "manual":
                        ok_, why_ = await rt.finish_answer()
                        if not ok_: res.update(ok=False, error=why_)
                    else:
                        ok_, why_ = await rt.force_end()
                        if not ok_: res.update(ok=False, error=why_)
                elif cmd == Cmd.END_SESSION.value:
                    out.put_nowait(("j", {"type": "REPORT_PROCESSING", "session_id": sid})); await rt.close()
                    out.put_nowait(("j", {"type": "SESSION_ENDED", "session_id": sid}))
                    report = {"asked": eng.asked_texts, "turns": rt.responses, "interjections": rt.interjections, "coverage": eng.cov.report(), "metrics": __import__("qa_lab.demo.metrics", fromlist=["build"]).build(rt, eng, opts, time.monotonic() - T_START, PAIR_LABELS)}
                    report["practice"] = LEDP
                    report["run_id"] = save_run(sid, opts, report, len(passage), passage); SAVED[0] = True
                    if LED is not None: LD.PENDING["task"] = asyncio.create_task(_grade_and_ledger(sid, opts, LED, LEDP))      # after the report is out: grade, credit the ledger (never during the practice)
                    out.put_nowait(("j", {"type": "REPORT_READY", "session_id": sid, "report": report}))
                    out.put_nowait(("j", {"type": "CMD_RESULT", "cmd": cmd, "ok": True})); await asyncio.sleep(0.5); await ws.close(); break
                else: res.update(ok=False, error=f"unknown command {cmd}")
            except Exception as ex: res.update(ok=False, error=repr(ex))
            out.put_nowait(("j", res))
    except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError): pass
    finally:
        try:                                                   # tab closed / connection lost without [세션 종료]: still keep what was measured
            if rt and rt.responses and not SAVED[0]:
                rep = {"asked": eng.asked_texts, "turns": rt.responses, "interjections": rt.interjections, "coverage": eng.cov.report(), "partial_save": True,
                       "metrics": __import__("qa_lab.demo.metrics", fromlist=["build"]).build(rt, eng, opts, time.monotonic() - T_START, PAIR_LABELS)}
                rep["practice"] = LEDP
                save_run(sid, opts, rep, len(passage), passage)
                if LED is not None: LD.PENDING["task"] = asyncio.create_task(_grade_and_ledger(sid, opts, LED, LEDP))
        except Exception: pass
        try: sp_kill()
        except Exception: pass
        if rt: await rt.close()
        if STATE["active"] is me: STATE["active"] = None

def serve(case=None, port=8765):
    import uvicorn; print(f"qa-lab demo: http://localhost:{port}"); uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
