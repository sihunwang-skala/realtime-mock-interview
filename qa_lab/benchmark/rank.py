"""Composite ranking of tournament combinations. Metrics, normalisation and weights are listed in docs/RANKING.md and applied here.
A metric that was not measured for a run is dropped and the remaining weights are renormalised."""
from __future__ import annotations
import json, pathlib
R = pathlib.Path(__file__).resolve().parents[2] / "results"; D = pathlib.Path(__file__).resolve().parents[2] / "docs"
WEIGHTS = {"usable": .30, "quality": .20, "speed": .20, "cer": .10, "interj": .10, "cost": .05, "privacy": .05}
LABEL = {"usable": "쓸 만한 비율", "quality": "질문 품질", "speed": "응답 속도", "cer": "받아쓰기 정확도", "interj": "끼어들기 정확도", "cost": "비용", "privacy": "프라이버시"}
PRICE = {"gpt-6-luna": (0.10, 0.50), "gpt-6.1-terra": (1.00, 5.00), "gpt-6.2-sol": (2.00, 10.00), "gpt-6.2-astra": (8.00, 40.00), "gpt-5.6-luna": (0.20, 1.20), "gpt-5.4-mini": (0.75, 4.50),
         "gpt-5.4-nano": (0.20, 1.25), "gpt-5-mini": (0.25, 2.00), "gpt-5-nano": (0.05, 0.40), "gpt-4.1-mini": (0.40, 1.60), "gpt-4o-mini": (0.15, 0.60), "gpt-6.1-sol": (2.00, 10.00), "gpt-6-sol": (2.00, 10.00), "gpt-5.1": (1.25, 10.00), "gpt-5": (1.25, 10.00), "gpt-4.1": (2.00, 8.00), "gpt-4o": (2.50, 10.00), "o3": (2.00, 8.00)}     # USD per 1M tokens (official pricing page)
LLM_COST_PER_1000 = {m: round(0.7 * i + 0.15 * o, 3) for m, (i, o) in PRICE.items()}      # USD per 1,000 follow-ups (700 in + 150 out tokens each)
STT_COST_PER_1000 = {"gpt-transcribe": 2.25}                                                                 # 0.5 min answer x $0.0045/min x 1,000

def cloud_cost(cfg):
    llm = cfg.get("llm", ""); c = 0.0; uses_cloud = False
    if llm.startswith("openai:"): uses_cloud = True; c += LLM_COST_PER_1000.get(llm.split(":", 1)[1], 1.0)
    if str(cfg.get("stt", "")).startswith("gpt-"): uses_cloud = True; c += STT_COST_PER_1000.get(cfg["stt"], 2.0)
    return uses_cloud, c

def parts(m: dict) -> dict:
    p = {}; clamp = lambda x: max(0.0, min(1.0, x))
    if m.get("usable_rate") is not None and m.get("mean_quality") is not None: p["usable"] = m["usable_rate"]      # needs the quality judgement
    if m.get("mean_quality") is not None: p["quality"] = clamp((m["mean_quality"] - 1) / 4)
    if m.get("mean_e2e") is not None: p["speed"] = clamp(1 - (m["mean_e2e"] - 1.0) / 2.0)
    if m.get("mean_cer") is not None: p["cer"] = clamp(1 - m["mean_cer"] / 0.3)
    if m.get("interj_f1") is not None: p["interj"] = clamp(m["interj_f1"])
    cloud, cost = cloud_cost(m["cfg"]); p["cost"] = 1.0 if not cloud else clamp(1 - cost / 5.0); p["privacy"] = 0.0 if cloud else 1.0
    return p

def score(m: dict) -> tuple[float, dict]:
    p = parts(m); w = {k: WEIGHTS[k] for k in p}; tot = sum(w.values())
    return round(100 * sum(p[k] * w[k] for k in p) / tot, 1), {k: round(v, 2) for k, v in p.items()}

def collect(with_unranked=False):
    rows = []
    import datetime as _dt, glob
    def ts(x): return _dt.datetime.fromtimestamp(x).strftime("%m-%d %H:%M")
    windows = {}
    for f, stage, logs in (("tournament_stage1.json", "1단계", "log_tournament_stage1.txt"), ("tournament_stage2.json", "2단계", "log_tournament_stage2*.txt"), ("tournament_finals.json", "결승", "log_tournament_finals*.txt"), ("tournament_cloud.json", "클라우드 라운드", "log_tournament_cloud*.txt"), ("tournament_cloud10.json", "클라우드 라운드 $5~10", "log_tournament_cloud10*.txt"), ("tournament_trio.json", "클라우드 3종 반복", "log_tournament_trio*.txt")):
        fp = R / f
        if fp.exists():
            starts = [pathlib.Path(g).stat().st_birthtime for g in glob.glob(str(R / logs))] or [fp.stat().st_mtime]
            windows[stage] = (ts(min(starts)), ts(fp.stat().st_mtime))
    for f, stage in (("tournament_stage1.json", "1단계"), ("tournament_stage2.json", "2단계"), ("tournament_finals.json", "결승"), ("tournament_cloud.json", "클라우드 라운드"), ("tournament_cloud10.json", "클라우드 라운드 $5~10"), ("tournament_trio.json", "클라우드 3종 반복")):
        fp = R / f
        if not fp.exists(): continue
        d = json.loads(fp.read_text())
        for a in d.get("axes", []):
            for m in a["standings"]: rows.append({**m, "stage": stage, "axis": a["axis"]})
        for m in d.get("standings", []): rows.append({**m, "stage": stage, "axis": "-"})
    seen = {}
    for r in rows: seen[(r["stage"], r["label"], json.dumps(r["cfg"], sort_keys=True))] = r
    out = []; unranked = []
    for r in seen.values():
        if r.get("mean_e2e") is None:                                   # no valid latency measurement -> not ranked
            unranked.append({"stage": r["stage"], "label": r["label"], "cfg": r["cfg"], "error": r.get("error") or "응답 시간 측정값 없음(재측정 필요)"}); continue
        sc, pp = score(r); cloud, cost = cloud_cost(r["cfg"])
        out.append({"stage": r["stage"], "axis": r["axis"], "label": r["label"], "cfg": r["cfg"], "scenarios": [x["id"] for x in r.get("rows", [])], "n": len(r.get("rows", [])),
                    "wall_sec": r.get("wall_sec"), "window": windows.get(r["stage"]), "error": r.get("error"), "score": sc, "parts": pp, "mean_e2e": r.get("mean_e2e"), "mean_quality": r.get("mean_quality"),
                    "usable_rate": r.get("usable_rate"), "mean_cer": r.get("mean_cer"), "interj_f1": r.get("interj_f1"), "cloud": cloud, "cost_per_1000": round(cost, 2)})
    out.sort(key=lambda x: -x["score"])
    for i, r in enumerate(out, 1): r["rank"] = i
    return (out, unranked) if with_unranked else out

def write():
    rows, unr = collect(True); (R / "tournament_ranking.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    L = ["# 조합 랭킹과 점수 산출 지표", "", "점수 = 100 × Σ(지표 점수 × 가중치) ÷ Σ(측정된 지표의 가중치). 측정하지 않은 지표는 빼고 가중치를 다시 나눈다.", "",
         "| 지표 | 측정 방법 | 지표 점수(0~1) | 가중치 |", "|---|---|---|---|",
         "| 쓸 만한 비율 | 첫 소리 2.0초 이내 **이면서** 꼬리질문 품질 ≥ 3.5/5 인 시나리오 비율 | 그대로 | 30% |",
         "| 질문 품질 | 심사 모델(gpt-6-luna)이 관련성·근거·깊이·자연스러움을 1~5점 채점한 평균 | (평균−1)÷4 | 20% |",
         "| 응답 속도 | 버튼을 누른 뒤 AI 첫 소리까지 평균 시간 | 1초 이하=1, 3초 이상=0, 사이는 직선 | 20% |",
         "| 받아쓰기 정확도 | 글자 오류율(CER) | 1−CER÷0.3 (0 이하는 0) | 10% |",
         "| 끼어들기 정확도 | 정답 라벨 시나리오에서 F1 | 그대로 | 10% |",
         "| 비용 | 클라우드 사용 시 질문 1,000개당 추정 비용(USD) = 0.7×입력가 + 0.15×출력가(백만 토큰당 단가, 질문당 입력 700·출력 150토큰 가정) + 클라우드 받아쓰기 | 로컬=1, 클라우드=1−비용÷5 | 5% |",
         "| 프라이버시 | 음성·텍스트가 외부로 나가는가 | 로컬=1, 클라우드=0 | 5% |", "",
         "**주의:** ☁️ 표시는 클라우드를 쓰는 조합이다. 실제 RFP나 실제 녹음에는 쓸 수 없고 합성·공개 데이터 전용이다. 표본은 시나리오 3개이므로 점수 차이가 작으면 동률로 본다(±3점).", ""]
    scn = sorted({x for r in rows for x in r["scenarios"]})
    L += ["## 측정 방식 (어떻게 쟀나)", "",
          f"- **날짜·장비:** 2026-11-19, Apple M2 Pro 32GB(Metal GPU). 내 컴퓨터 Ollama + 외장하드 Ollama(별도 포트).",
          "- **입력:** 합성 한국어 음성(macOS Yuna) 시나리오. 같은 파일을 모든 조합에 똑같이 재생. 실제 녹음은 쓰지 않음(클라우드 조합이 있어서).",
          f"- **시나리오 3개(조합당):** 응답 속도·품질 비교 = `s01_normal`, `s05_terms`, `s06_numbers` / 끼어들기 비교 = `s01_normal`(끼어들면 안 됨), `s08_contradict_rfp`, `s11_ramble`(끼어들어야 함). 사용된 시나리오: {', '.join(scn)}",
          "- **답변 끝내는 방식:** 사용자가 **`답변 끝` 버튼**을 누르는 방식. 테스트에서는 **실제 발화가 끝나는 순간에 버튼이 눌렸다고 가정**했다. 응답 시간 = 버튼 → AI 첫 소리 송출(서버 기준, 브라우저 재생은 아님).",
          "- **반복 횟수:** 조합당 시나리오 3개를 **1회씩**(반복·평균 없음). 모델은 미리 불러 둔(웜) 상태에서 쟀고, 첫 로드 시간은 응답 시간에 넣지 않았다.",
          "- **고정 조건:** 질문 큐 2개 대기, 같은 질문 프롬프트, 한 번에 하나의 축만 변경하고 나머지는 현재 최선으로 고정(토너먼트).",
          "- **질문 품질 채점:** 심사 모델 `gpt-6-luna`가 관련성·근거·깊이·자연스러움을 1~5점으로 채점해 평균(합성 텍스트만 전송). 끼어들기 비교 실험은 품질을 채점하지 않음.",
          "- **한계:** 표본 3개, 합성 음성, 서버 기준 지연. 점수 차이 ±3점 이내는 동률로 해석. 1~2단계는 서로 다른 축을 기준 구성에서 하나씩 바꾼 비교라 서로 다른 단계의 점수를 직접 비교하면 안 된다.", ""]
    L += ["## 측정 범위", "", "| 단계 | 바꾼 축 | 후보 수 | 시험 횟수(조합×시나리오 3) | 측정 시간대 |", "|---|---|---|---|---|"]
    for stage in ("1단계", "2단계", "결승", "클라우드 라운드", "클라우드 라운드 $5~10", "클라우드 3종 반복"):
        sub = [r for r in rows if r["stage"] == stage]
        if sub: L.append(f"| {stage} | {', '.join(sorted({r['axis'] for r in sub}))} | {len(sub)} | {len(sub)}회 × 3 = {len(sub)*3}개 응답 | {sub[0]['window'][0]} ~ {sub[0]['window'][1]} (날짜 2026-11-19) |")
    L.append("")
    for stage in ("1단계", "2단계", "결승", "클라우드 라운드", "클라우드 라운드 $5~10", "클라우드 3종 반복"):
        sub = [r for r in rows if r["stage"] == stage]
        if not sub: continue
        L += [f"## {stage} 랭킹", "", "| 순위 | 조합 | 점수 | 쓸 만함 | 품질 | 속도(s) | CER | 끼어들기 F1 | 비용(질문 1,000개당) |", "|---|---|---|---|---|---|---|---|---|"]
        for r in sub: L.append(f"| {r['rank']} | {'☁️ ' if r['cloud'] else ''}{r['label']} | **{r['score']}** | {r['parts'].get('usable','-')} | {r['mean_quality'] or '-'} | {r['mean_e2e'] or '-'} | {r['mean_cer'] if r['mean_cer'] is not None else '-'} | {r['interj_f1'] if r['interj_f1'] is not None else '-'} | ${r['cost_per_1000']} |")
        L.append("")
        L += [f"### {stage} 조합 구성 상세", "", "| 순위 | LLM | 받아쓰기 | 질문 전략 | 끼어들기 | TTS | 답변 끝내기 | 시나리오 수 | 시험 전체 소요(초, 모델 불러오기 포함·점수 미반영) | 오류 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            c = r["cfg"]; L.append(f"| {r['rank']} | {c.get('llm')} | {c.get('stt')} | {c.get('strategy')} | {c.get('interjection')} | {c.get('tts')} | 버튼(발화 끝에 누름 가정) | {r['n']} | {r['wall_sec'] or '-'} | {r['error'] or ''} |")
        L.append("")
    if unr:
        L += ["## 순위에서 제외된 조합 (측정 불가·재측정 중)", "", "| 단계 | 조합 | 사유 |", "|---|---|---|"] + [f"| {u['stage']} | {u['label']} | {u['error']} |" for u in unr] + [""]
    (D / "RANKING.md").write_text("\n".join(L) + "\n"); print("ranking written:", len(rows), "combinations")
if __name__ == "__main__": write()
