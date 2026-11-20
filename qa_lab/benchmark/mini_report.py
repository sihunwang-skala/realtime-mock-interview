"""Build the detailed mini-suite report from result files. Public doc: docs/MINI_SUITE.md (no recording text). Local doc: results/mini_report_REAL_local.md."""
from __future__ import annotations
import json, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parents[2]; R = ROOT / "results"
J = lambda n: json.loads((R / n).read_text())

def main():
    tm = J("mini_timing.json"); t1, t2, t3 = J("mini_t1_stt.json"), J("mini_t2_endpoint.json"), J("mini_t3_followup_REAL_local.json")
    t4 = {k: J(f"real_qa_mini_t4_{k}.json") for k in ("baseline_small", "spec1500_small", "spec1500_turbo")}; t5, t6, t7 = J("mini_t5_interjection.json"), J("mini_t6_cases.json"), J("mini_t7_llm.json")
    t4_wall = 405  # 3 configs x ~135 s; measured from result-file timestamps (the timer was added after T4 ran)
    L = []; A = L.append
    A("# 미니 테스트 스위트 (7개 테스트 × 3개 샘플) — 방식, 결과, 사례"); A("")
    A(f"실행일: {time.strftime('%Y-%m-%d')} · 장비: Apple M2 Pro 32GB (GPU=Metal/MPS) · 규격: `docs/TEST_RECORD_SPEC.md`"); A("")
    A("**데이터 취급:** 실제 녹음(사람 음성)은 로컬 모델로만 처리했고 OpenAI에는 보내지 않았다. 클라우드(Case C, gpt-4o-mini, 심사 모델)는 **합성 음성/텍스트에만** 사용했다. 이 문서에는 녹음에서 나온 텍스트를 넣지 않았다(ID와 수치만).")
    A(""); A("## 한눈에 보기"); A("")
    A("| # | 테스트 | 샘플 | 소요 시간 | 핵심 결과 |"); A("|---|---|---|---|---|")
    b = t1["faster-whisper:base@mps"]; s = t1["faster-whisper:small@mps"]; tu = t1["faster-whisper:large-v3-turbo@mps"]
    A(f"| T1 | 음성 인식 모델 비교 | 실제 답변 3 | {tm['t1']['wall_sec']}초 | CER base {b['mean_cer']} / small {s['mean_cer']} / turbo **{tu['mean_cer']}** |")
    A(f"| T2 | 끝점 감지 정책 | 실제 답변 3 | {tm['t2']['wall_sec']}초 | 답변당 끝점: 0.6초 {sum(t2['vad_only']['600']['endpoints'].values())/3:.1f}회 → 2.0초 {sum(t2['vad_only']['2000']['endpoints'].values())/3:.1f}회, 의미 기반 {t2['semantic_hybrid']['mean']}회 |")
    A(f"| T3 | 꼬리질문 품질 | 실제 쌍 3 | {tm['t3']['wall_sec']}초 | 로컬 심사 평균 약 3점대, 내부 라벨 누수·엉뚱한 RFP 혼입 발견 |")
    bs = t4['baseline_small']['summary']; sp = t4['spec1500_small']['summary']
    A(f"| T4 | 종단 지연(실제 음성) | 실제 답변 3 × 3구성 | 약 {t4_wall}초 | 평균 {bs['e2e_after_clip_end']['mean']}s → {sp['e2e_after_clip_end']['mean']}s (P50 {bs['e2e_after_clip_end']['p50']}→{sp['e2e_after_clip_end']['p50']}s), 끊김 {bs['avg_endpoints_per_answer']}→{sp['avg_endpoints_per_answer']}회 |")
    A(f"| T5 | 끼어들기 정책 | 합성 3 × 3정책 | {tm['t5']['wall_sec']}초 | 3정책 모두 정밀도·재현율 1.0 (표본 3) |")
    A(f"| T6 | Case A/B/C | 합성 3 × 3케이스 | {tm['t6']['wall_sec']}초 | P50: B {t6['B']['e2e']['p50']}s / A {t6['A']['e2e']['p50']}s / C **{t6['C']['e2e']['p50']}s** |")
    A(f"| T7 | LLM 비교 | 합성 3 × 4모델 | {tm['t7']['wall_sec']}초 | 첫 토큰 llama3.2:3b가 가장 빠름, 품질은 qwen2.5:7b가 최고 |"); A("")

    A("## T1. 음성 인식 모델 비교"); A("")
    A("- **목적:** 사람 목소리(회의 녹음)에서 모델별 정확도와 속도 차이.")
    A("- **방법:** 같은 답변 오디오 3개를 모델별로 통째로 받아쓰고, 제공 전사문을 기준으로 글자 오류율(CER = 편집거리 ÷ 기준 글자 수, 공백/문장부호 제거)을 계산. 속도는 처리시간 ÷ 오디오 길이.")
    A("- **샘플:** `r1_a`, `r6_a`, `m1_a` (실제 사람, 각 약 26~45초). **구성:** faster-whisper, Metal GPU, 빔 1, 한국어 고정, 도메인 용어 프롬프트.")
    A(""); A("| 모델 | 평균 CER | 평균 처리배율 | 클립별 CER |"); A("|---|---|---|---|")
    for k, v in t1.items(): A(f"| {k} | {v['mean_cer']} | {v['mean_rtf']} | " + ", ".join(f"{r['clip']} {r['cer']}" for r in v["rows"]) + " |")
    A("- **해석:** 이 3개(짧고 또렷한 답변)에서는 base≈small, turbo가 더 낫다. 앞서 전체 쌍으로 잰 값(small 0.32 vs turbo 0.13)보다 격차가 작은 건 표본이 쉬웠기 때문.")
    A("- **한계:** 제공 전사문 자체가 음성 인식 결과라 turbo에 유리할 수 있음. 절대 정확도 아님."); A("")

    A("## T2. 끝점 감지(답변이 끝났다는 판단) 정책"); A("")
    A("- **목적:** 사람이 길게 말할 때 AI가 답변 도중에 끊고 들어오는 횟수를 정책별로 비교(이상적으로는 한 답변 = 1번).")
    A("- **방법:** 같은 3개 답변 뒤에 3초 침묵을 붙여 Silero VAD에 흘리고, 침묵 길이 기준(0.4/0.6/1.0/1.5/2.0초)별로 '끝' 판정 횟수를 센다. 의미 기반 혼합 = 0.6초 침묵에서 지금까지의 텍스트가 문장 종결 표현이면 확정, 아니면 1.5초까지 대기.")
    A(""); A("| 정책 | r1_a | r6_a | m1_a | 평균 |"); A("|---|---|---|---|---|")
    for ms, v in t2["vad_only"].items(): e = v["endpoints"]; A(f"| 침묵 {int(ms)/1000}초 | {e['r1_a']} | {e['r6_a']} | {e['m1_a']} | {v['mean']} |")
    sh = t2["semantic_hybrid"]; A(f"| 의미 기반 혼합 | {sh['r1_a']} | {sh['r6_a']} | {sh['m1_a']} | {sh['mean']} |")
    A("- **해석:** 침묵 2.0초에서야 1번이 되지만 그만큼 응답이 늦어진다. 의미 기반 규칙은 0.6초 단독보다 나빴다(한국어 종결 어미가 문장 중간에도 자주 나와 '완결'로 잘못 통과).")
    A("- **한계:** 표본 3, '끝' 정답은 사람이 확인하지 않음(답변 1번 기준 가정)."); A("")

    A("## T3. 꼬리질문 품질 (실제 쌍, 로컬 전용)"); A("")
    A("- **목적:** 사람의 답변을 듣고 AI가 만든 꼬리질문이 맥락에 맞는지.")
    A("- **방법:** 실제 질문과 답변(제공 전사문)을 넣고 로컬 `qwen2.5:7b`가 꼬리질문 1개를 생성. 같은 모델이 relevance/grounded/depth/natural(1~5)로 채점. 사람 꼬리질문이 있는 쌍은 같은 기준으로 채점해 비교.")
    avg = lambda sc: round(sum(sc[k] for k in ("relevance", "grounded", "depth", "natural")) / 4, 2)
    A(""); A("| 클립 | AI 꼬리질문 평균 | 사람 꼬리질문 평균 |"); A("|---|---|---|")
    for r in t3: A(f"| {r['clip']} | {avg(r['ai_scores'])} | {avg(r['human_scores']) if r['human_scores'] else '-'} |")
    A("- **발견된 문제:** (1) `Action: DEEPEN` 같은 내부 라벨이 질문 앞에 섞여 출력됨 (2) 멘토링 대화인데 샘플 RFP(스마트시티) 내용을 끌어와 질문함 → 컨텍스트 분리 필요.")
    A("- **한계:** 같은 모델이 생성·채점해 편향 큼, 사람 꼬리질문이 있는 쌍은 1개뿐. 질문/답변 원문은 `results/mini_report_REAL_local.md`(로컬)에만 있음."); A("")

    A("## T4. 종단 지연 (실제 사람 음성, 실시간 재생)"); A("")
    A("- **목적:** 답변이 끝난 뒤 AI의 첫 소리까지 걸리는 시간과 답변 도중 끊김, 개선 전후 비교.")
    A("- **방법:** 같은 3개 답변을 실시간 속도로 음성 파이프라인(VAD→STT→질문 생성→TTS)에 흘림. 응답 지연 = 클립 끝 → AI 첫 소리 송출(서버 기준, 브라우저 재생 아님). 구성 3가지.")
    A(""); A("| 구성 | 평균 | P50 | 최대 | 2초 이내 | 답변당 끝점 판단 | 끼어들기 발생 |"); A("|---|---|---|---|---|---|---|")
    for k, lab in (("baseline_small", "기본: small, 침묵 0.6초"), ("spec1500_small", "미리 생성+1.5초 확인, small"), ("spec1500_turbo", "미리 생성+1.5초 확인, turbo")):
        d = t4[k]; e = d["summary"]["e2e_after_clip_end"]; A(f"| {lab} | {e['mean']}s | {e['p50']}s | {e['max']}s | {int(e['under_2s']*3)}/3 | {d['summary']['avg_endpoints_per_answer']} | {d['summary']['interjection_clips']}/3 클립 |")
    A("- **해석:** 미리 생성은 끊김을 절반 가까이 줄이고 평균 지연도 줄였다. turbo는 정확도 이득 대신 2초 이내 비율이 낮다.")
    A("- **발견된 문제:** 규칙 기반 끼어들기가 정상적인 사람 답변에서 '주제 이탈'로 오작동(3구성 × 3클립 = 9번 중 7번). 질문과의 단어 겹침 규칙이 한국어 구어에 너무 거칠다.")
    A("- **한계:** '실제 발화 끝'을 클립 끝으로 근사(클립 끝에 침묵이 붙은 경우 지연이 과소 측정됨, 예: 0.17초). 표본 3."); A("")

    A("## T5. 끼어들기 정책 (합성, 정답 라벨 있음)"); A("")
    A("- **목적:** AI가 답변 도중 끼어들어야 할 때만 끼어드는가.")
    A("- **방법:** 합성 음성 3개(정상 `s01`, RFP 모순 `s08`, 주제 이탈 `s11`)를 실시간 재생. 정답: s01=개입 없음, s08·s11=개입. 정책 3가지(규칙/LLM/혼합).")
    A(""); A("| 정책 | s01(없어야) | s08(모순) | s11(주제 이탈) | 정밀도/재현율 | 소요 |"); A("|---|---|---|---|---|---|")
    for pol, v in t5.items():
        c = {r["id"]: r for r in v["per_scenario"]}; f = lambda r: ("개입 " + str(r["kind"]) + f" @{r['t_after_speech_start_s']}s") if r["interjected"] else "개입 없음"
        A(f"| {pol} | {f(c['s01_normal'])} | {f(c['s08_contradict_rfp'])} | {f(c['s11_ramble'])} | {v['summary']['precision']}/{v['summary']['recall']} | {v['wall_sec']}초 |")
    A("- **해석:** 3정책 모두 맞았지만 표본 3개라 변별력이 없다. 앞선 13개 시나리오 전체에서는 LLM 단독이 재현율 1.0, 규칙은 오탐 0.11이었다.")
    A("- **한계:** 합성 음성, 표본 3. 실제 음성 오탐은 T4에서 별도 확인(규칙 정책이 오작동)."); A("")

    A("## T6. Case A / B / C (같은 합성 음성 3개)"); A("")
    A("- **목적:** 구조별 응답 지연 차이.")
    A("- **방법:** `s01_normal`, `s05_terms`, `s06_numbers`를 실시간 재생. 응답 지연 = 실제 발화 끝 → 첫 소리. 끼어들기 끔. B=로컬, A=같은 로컬 파이프라인을 LiveKit(WebRTC, 루프백)로 전송, C=OpenAI Realtime(클라우드, 서버 VAD 대신 0.6초 뒤 커밋해 B와 맞춤).")
    A(""); A("| Case | 평균 | P50 | 최대 | 2초 이내 | 질문 작성 주체 | 소요 |"); A("|---|---|---|---|---|---|---|")
    for k, who in (("B", "우리 엔진(통제 가능)"), ("A", "우리 엔진(통제 가능)"), ("C", "모델이 직접(통제 제한)")):
        e = t6[k]["e2e"]; A(f"| {k} | {e['mean']}s | {e['p50']}s | {e['max']}s | {int(e['under_2s']*3)}/3 | {who} | {t6[k]['wall_sec']}초 |")
    A(""); A("**사례 (합성이라 원문 공개):**"); A("")
    for k in ("B", "A", "C"):
        for r in (t6[k]["rows"] if k != "C" else [(x["id"], x["e2e_s"], None, x["question"]) for x in t6["C"]["rows"]]):
            A(f"- Case {k} · {r[0]} · {r[1]}s → {r[3]}")
    A("- **해석:** C가 가장 빠르다(P50 1.26s). 다만 질문을 모델이 직접 만들어 우리 planner·RFP 근거 통제를 못 하고, 모든 음성이 외부로 나간다. A는 B와 같은 처리에 전송 지연이 더해졌고, 이번 3개에서는 2초 이내가 0/3.")
    A("- **한계:** 표본 3, 합성 음성, 루프백(실제 네트워크 아님). C는 사용자 발화 끝을 우리가 지정(0.6초)했고 질문 품질은 미평가."); A("")

    A("## T7. LLM 비교 (합성, 클라우드 포함)"); A("")
    A("- **목적:** 꼬리질문 생성용 LLM별 첫 토큰 지연과 질문 품질.")
    A("- **방법:** 합성 시나리오 3개에서 같은 프롬프트/경로(투트랙 빠른 경로)로 모델 4개가 꼬리질문 생성. 첫 토큰 시간 측정. 품질은 gpt-4o-mini가 relevance/grounded/depth/natural(1~5) 채점(평균).")
    A(""); A("| 모델 | 평균 첫 토큰 | 평균 품질(1~5) |"); A("|---|---|---|")
    for k, v in t7["summary"].items(): A(f"| {k} | {v['mean_first_token_s']}s | {v['mean_quality_1to5']} |")
    A(""); A("**사례 (`s08_contradict_rfp`: 응답시간 5초, 구축 12개월 모순 답변):**"); A("")
    for r in t7["rows"]:
        if r["scenario"] == "s08_contradict_rfp": A(f"- {r['llm']} ({r['action']}, 첫 토큰 {r['first_token_s']}s): {r['question']}")
    A("- **해석:** 속도는 3B > 4B≈7B > gpt-4o-mini(네트워크), 품질은 7B > 4B > gpt-4o-mini > 3B. 로컬 7B가 속도·품질 균형이 좋다.")
    A("- **한계:** 표본 3(합성), 심사 모델이 한 종류(gpt-4o-mini)라 편향 가능. 심사는 gpt-4o-mini 자신의 출력도 채점."); A("")

    # ---- T8-T12 (newest OpenAI models, judge validation, quality of existing configs) ----
    t8, t9, t10 = J("mini_t8_openai_llm_ladder.json"), J("mini_t9_openai_realtime_ladder.json"), J("mini_t10_cloud_stt.json")
    t11, t12, t3b = J("mini_t11_judge_agreement.json"), J("mini_t12_quality_matrix.json"), J("mini_t3b_followup_REAL_local_rejudged.json")
    A("## T8. 최신 OpenAI 텍스트 모델 단계별 비교 (합성 3개)"); A("")
    A("- **목적:** 최신·가성비 모델을 낮은 단계부터 올려가며 꼬리질문 생성의 속도와 품질 비교. **방법:** 합성 시나리오 3개 × 모델 4개, 같은 프롬프트, 가장 낮은 추론 단계. 채점은 `" + t8["judge"] + "`(기준 심사 모델).")
    A(""); A("| 모델 | 평균 첫 토큰 | 평균 품질(1~5) |"); A("|---|---|---|")
    for k, v in t8["summary"].items(): A(f"| {k} | {v.get('mean_first_token_s')}s | {v.get('mean_quality_1to5')} |")
    A(f"- **해석:** 4개 모두 품질이 4.75~5.0으로 천장에 붙어 변별이 약하다. 속도는 `gpt-6.2-sol` 0.9s ≈ `gpt-6.1-terra` 0.97s < `gpt-6-luna` 1.19s < `gpt-6.2-astra` 1.66s. 가격은 API로 확인 불가라 비용 비교는 못 함. **소요 {tm['t8']['wall_sec']}초**."); A("")
    A("## T9. 실시간 음성 모델(Case C) 단계별 비교 (합성 3개)"); A("")
    A("- **방법:** 합성 음성 3개를 실시간으로 보내고 0.6초 뒤 응답 요청. 지연 = 발화 끝 → 첫 오디오. 질문 품질은 기준 심사 모델이 채점. (앞서 T6의 `gpt-realtime`은 P50 1.26초.)")
    A(""); A("| 모델 | P50 | 최대 | 질문 품질 | 소요 |"); A("|---|---|---|---|---|")
    for m, v in t9["models"].items(): A(f"| {m} | {v['e2e']['p50']}s | {v['e2e']['max']}s | {v['mean_quality_1to5']} | {v['wall_sec']}초 |")
    A("- **해석:** 더 최신·고성능일수록 품질은 오르지만(4.08→4.67) 지연도 늘어난다(1.69→2.22초). `gpt-realtime-2.1`은 2초를 넘는다. 어떤 신형도 기존 `gpt-realtime`(1.26초)보다 빠르지 않았다."); A("")
    A("## T10. 클라우드 음성 인식 (합성 3개: 전문용어, 숫자, 소음)"); A("")
    A("- **방법:** 같은 합성 음성 3개를 모델별로 받아쓰고 글자 오류율(CER) 비교. 실제 녹음은 보안상 클라우드로 보내지 않음.")
    A(""); A("| 모델 | 평균 CER | 평균 시간 | 비고 |"); A("|---|---|---|---|")
    for k, v in t10.items(): A(f"| {k} | {v['mean_cer']} | {v['mean_sec']}s | " + ("일반 인식 API 미지원(404), 실시간 전용 추정" if v["mean_cer"] is None else "") + " |")
    A("- **해석:** `gpt-transcribe`가 가장 정확(0.061). 로컬 turbo 0.122, small 0.157. 합성 음성이라 사람 목소리에서는 다를 수 있다."); A("")
    A("## T11. 어떤 심사 모델을 믿을 수 있나 (기준 심사 모델과의 일치도)"); A("")
    A("- **방법:** T8이 만든 합성 꼬리질문 12개를 후보 심사 모델이 채점하고, 기준 심사 모델(`" + t8["judge"] + "`)과 상관계수(Pearson), 평균 절대 차이를 비교.")
    A(""); A("| 심사 모델 | 상관 | 평균 절대 차이 | 평균 점수(기준 4.9) |"); A("|---|---|---|---|")
    for k, v in t11.items(): A(f"| {k} | {v['pearson_vs_ref']} | {v['mean_abs_diff']} | {v['mean_score']} |")
    A("- **해석:** 싼 클라우드 `gpt-6-luna`가 가장 일치(0.79). 로컬 모델은 상관이 낮고 점수가 훨씬 박하다(qwen2.5:7b 평균 2.46). **실제 녹음은 로컬 심사만 가능하므로 로컬 점수는 절대값이 아니라 대략적 순위로만 해석해야 한다.**"); A("")
    A("## T12. 이미 잰 구성의 질문 적절성 (속도만으로는 오해)"); A("")
    A("- **방법:** 말씀하신 3개 구성(precomputed, two_track, direct)의 합성 시나리오 3개 질문을 `" + t12["judge"] + "`가 채점. 쓸 만함 = 응답 2.0초 이내이고 품질 3.5 이상.")
    A(""); A("| 구성 | 평균 질문 품질 | 쓸 만한 비율 | 시나리오별(응답 s / 품질) |"); A("|---|---|---|---|")
    for k, v in t12["runs"].items(): A(f"| {k} | {v['mean_quality_1to5']} | {v['usable_rate']} | " + ", ".join(f"{r['id'][:4]} {r['e2e_s']}/{r['quality']}" for r in v["rows"]) + " |")
    A("- **해석:** 고정 질문(precomputed)은 속도가 가장 빠르지만 답변과 무관한 질문이 나와 품질이 가장 낮다(s01: 2.5). 세 구성 모두 이 3개에서는 '쓸 만한 비율'이 0이다(2초를 넘기거나 개입 시나리오는 질문이 비어 평가 불가).")
    A("- **이상 징후:** 기록된 소요 시간(3117초)은 호출 9회 대비 비정상적으로 길다. 이전 실행이 겹쳐 돌았던 것으로 보여 신뢰하지 않는다."); A("")
    A("## T3b. 실제 꼬리질문 재채점 (로컬 심사 `" + t3b["judge"] + "`, 실제 쌍 3개)"); A("")
    A("- 실제 답변에 대한 AI 꼬리질문 평균 점수: " + ", ".join(f"{r['clip']} {r['ai_score']}" for r in t3b["rows"]) + ". 사람 꼬리질문이 있는 쌍(r1_a): AI " + str(t3b["rows"][0]["ai_score"]) + " vs 사람 " + str(t3b["rows"][0]["human_score"]) + ".")
    A("- **한계:** 로컬 심사 모델은 기준 모델과 상관 0.54로 신뢰도가 낮다(T11). 원문은 `results/mini_report_REAL_local.md`(로컬)에 있다."); A("")
    A("## 재현"); A("```bash"); A("cd ai/qa-lab && for t in t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t12 t11 t3b; do .venv/bin/python -W ignore -m qa_lab.benchmark.mini_suite $t; done"); A(".venv/bin/python -W ignore -m qa_lab.benchmark.mini_report"); A("```")
    A("T1~T4는 로컬 녹음 조각(`data/real_qa/`)이 필요하다(저장소 미포함).")
    out = []
    for ln in L:
        if ln.startswith("- ") and out and out[-1].startswith("|"): out.append("")
        out.append(ln)
    (ROOT / "docs/MINI_SUITE.md").write_text("\n".join(out) + "\n")

    # local-only detail with REAL text
    M = ["# 실제 녹음 기반 상세 사례 (로컬 전용, 저장소 제외)", ""]
    M.append("## T3 꼬리질문: AI vs 사람 (원문)"); M.append("")
    for r in t3:
        M += [f"### {r['clip']}", f"- AI 꼬리질문: {r['ai_followup']}", f"- AI 점수: {r['ai_scores']}", f"- 사람 꼬리질문: {r['human_followup'] or '(없음)'}", f"- 사람 점수: {r['human_scores'] or '-'}", ""]
    M.append("## T4 끼어들기 오작동 사례"); M.append("")
    for k, d in t4.items():
        for r in d["results"]: M.append(f"- [{k}] {r['id']}: 끼어들기 {[(i['kind'], round(i['t_after_speech_start'],1), i['partial'][-60:]) for i in r['interjections']]}")
    M.append(""); M.append("## T3b 로컬 심사(재채점)"); [M.append(f"- {r['clip']}: AI {r['ai_score']} / 사람 {r['human_score']} — {r['ai_followup'][:120]}") for r in t3b["rows"]]
    (R / "mini_report_REAL_local.md").write_text("\n".join(M) + "\n"); print("wrote docs/MINI_SUITE.md and results/mini_report_REAL_local.md")
if __name__ == "__main__": main()
