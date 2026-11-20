"""Per-session measurements for a live run: timing per turn, stage breakdown, interjections, wall time, estimated cost."""
from __future__ import annotations
import statistics as st

STAGES = [("STT_FINALIZED", "받아쓰기 완료"), ("ANSWER_ANALYSIS_COMPLETED", "답변 분석 완료"), ("QUESTION_FIRST_TOKEN", "질문 첫 글자"),
          ("TTS_FIRST_AUDIO_CHUNK", "음성 첫 조각"), ("AUDIO_FIRST_PLAYED", "첫 소리(체감 응답)")]

def build(rt, eng, opts: dict, wall_s: float, labels: list[str]) -> dict:
    turns, k = [], 0
    for r in rt.responses:
        mk = r.get("marks") or {}; base = mk.get("BUTTON_PRESSED") or mk.get("VAD_END_DETECTED")
        d = {"n": len(turns) + 1, "kind": "끼어들기" if r.get("interjection") else "문답", "label": labels[k] if k < len(labels) and not r.get("interjection") else None}
        if not r.get("interjection"): k += 1
        if base is not None:
            d["stages_s"] = {name: round(mk[key] - base, 3) for key, name in STAGES if key in mk}
            d["e2e_s"] = round(mk["AUDIO_FIRST_PLAYED"] - base, 3) if "AUDIO_FIRST_PLAYED" in mk else None
        else: d["stages_s"] = {}; d["e2e_s"] = None
        d["interjected"] = bool(r.get("interjected_during")) if not r.get("interjection") else None
        d["question_chars"] = len(r.get("question") or ""); d["answer_chars"] = len(r.get("stt_text") or ""); d["action"] = r.get("action") or r.get("kind")
        turns.append(d)
    normal = [t for t in turns if t["kind"] == "문답" and t["label"] == "문답"]; induce = [t for t in turns if t["kind"] == "문답" and t["label"] == "끼어들기 유도"]
    ij = {"normal_pairs": len(normal), "false_interjections": sum(1 for t in normal if t["interjected"]), "induce_pairs": len(induce), "caught": sum(1 for t in induce if t["interjected"])}
    e2e = [t["e2e_s"] for t in turns if t["e2e_s"] is not None and t["kind"] == "문답"]
    llm = str(opts.get("llm", "")); cloud = llm.startswith("openai:") or llm.startswith("gpt-")
    cost = 0.0
    if cloud:
        from ..benchmark.rank import PRICE
        pi, po = PRICE.get(llm.split(":")[-1], (0, 0)); cost = len(rt.responses) * (700 * pi + 150 * po) / 1e6
    return {"wall_s": round(wall_s, 1), "turns": turns, "n_turns": len(turns), "n_interjections": len(rt.interjections),
            "interjection_times_s": [round(i["t_after_speech_start"], 2) for i in rt.interjections],
            "e2e_mean_s": round(st.mean(e2e), 2) if e2e else None, "e2e_max_s": round(max(e2e), 2) if e2e else None,
            "under_2s": f"{sum(1 for x in e2e if x <= 2)}/{len(e2e)}" if e2e else None,
            "llm": llm, "cloud": cloud, "est_cost_usd": round(cost, 5),
            "cost_note": "로컬 모델은 0원(전기·장비 제외). 클라우드는 질문당 입력 700·출력 150토큰 가정의 추정치이며 정확한 값은 OpenAI 대시보드에서 확인하세요.",
            "pair_labels": labels, "interjection_stats": ij}
