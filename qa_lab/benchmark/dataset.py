"""Korean benchmark scenarios rendered to 16 kHz wav (macOS `say`, Yuna voice - the only Korean macOS voice that renders audio). Same files replayed to every case.
LIMITATION: synthetic TTS speech, not human speech -> STT accuracy is optimistic; documented in the report."""
from __future__ import annotations
import json, subprocess, tempfile, pathlib
import numpy as np, soundfile as sf

SR = 16000
Q_API = "민원 접수 API의 응답시간 목표를 어떻게 달성하실 계획인가요?"
Q_SEC = "개인정보 보호 방안을 설명해 주세요."
SCEN = [
 dict(id="s01_normal", q=Q_API, segs=[("민원 접수 API는 캐시와 비동기 큐를 사용해서 평균 응답시간을 일 점 이 초 수준으로 유지하겠습니다.", 0)], expect=dict(interject=False)),
 dict(id="s02_short", q=Q_SEC, segs=[("네, 암호화하겠습니다.", 0)], expect=dict(interject=False)),
 dict(id="s03_long", q=Q_API, segs=[("저희는 먼저 로드밸런서를 두 대 이중화하고, 오토스케일링으로 동시 접속자 오천 명까지 대응합니다. 또한 읽기 요청은 레디스 캐시로 처리하고 쓰기 요청은 메시지 큐로 비동기 처리합니다. 데이터베이스는 읽기 복제본을 세 대 두어 응답시간을 평균 일 점 이 초 이내로 유지합니다. 부하 테스트는 배포 전에 매번 수행합니다.", 0)], expect=dict(interject=False)),
 dict(id="s04_pause", q=Q_API, segs=[("응답시간은 캐시로 줄이겠습니다.", 1.3), ("그리고 부하 테스트로 평균 일 점 이 초를 검증하겠습니다.", 0)], expect=dict(interject=False, pause=True)),
 dict(id="s05_terms", q=Q_SEC, segs=[("개인정보는 에이이에스 이백오십육으로 암호화하고 접근 로그는 일 년 이상 보관합니다. 키 관리는 에이치에스엠을 사용합니다.", 0)], expect=dict(interject=False), ref="개인정보는 AES 256으로 암호화하고 접근 로그는 1년 이상 보관합니다. 키 관리는 HSM을 사용합니다."),
 dict(id="s06_numbers", q=Q_API, segs=[("목표 응답시간은 평균 일 점 이 초이고 동시 접속자는 오천 명, 가용성은 구십구 점 구 퍼센트입니다.", 0)], expect=dict(interject=False), ref="목표 응답시간은 평균 1.2초이고 동시 접속자는 5000명, 가용성은 99.9퍼센트입니다."),
 dict(id="s07_vague", q=Q_API, segs=[("저희 회사는 경험이 많아서 잘 해결하겠습니다. 걱정하지 않으셔도 됩니다.", 0)], expect=dict(interject=False, follow="CLARIFY|VERIFY|DEEPEN")),
 dict(id="s08_contradict_rfp", q=Q_API, segs=[("민원 접수 API 응답시간은 평균 오 초 정도로 예상하고 있고 구축 기간은 십이 개월이 필요합니다. 그리고 계속 말씀드리면 부하 상황에서는 그렇습니다.", 0)], expect=dict(interject=True, kind="contradiction_rfp"), ref="민원 접수 API 응답시간은 평균 5초 정도로 예상하고 있고 구축 기간은 12개월이 필요합니다. 그리고 계속 말씀드리면 부하 상황에서는 그렇습니다."),
 dict(id="s09_rfp_violation", q=Q_SEC, segs=[("개인정보 접근 로그는 비용 문제로 삼 개월만 보관하려고 합니다. 이 부분은 협의 가능하다고 생각합니다.", 0)], expect=dict(interject=True, kind="contradiction_rfp")),
 dict(id="s10_repeat_request", q=Q_SEC, segs=[("죄송한데 질문을 다시 한번 말씀해 주시겠어요?", 0)], expect=dict(interject=False, action="REPEAT")),
 dict(id="s11_ramble", q=Q_API, segs=[("저희 회사는 이천 년에 설립되어 지금까지 여러 프로젝트를 수행했습니다. 사무실은 판교에 있고 직원들은 매우 화목하며 매년 워크숍을 갑니다. 창립 기념일에는 전 직원이 모여 행사를 열고 사내 동호회도 활발합니다. 작년에는 봉사활동도 다녀왔고 올해는 체육대회를 계획하고 있습니다. 구내식당 음식도 좋다는 평가를 받고 있습니다.", 0)], expect=dict(interject=True, kind="off_topic")),
 dict(id="s12_noise", q=Q_API, segs=[("캐시와 비동기 큐로 응답시간을 일 점 이 초로 유지하겠습니다.", 0)], noise_snr_db=8, expect=dict(interject=False)),
 dict(id="s13_repetition", q=Q_API, segs=[("응답시간을 빠르게 하겠습니다. 응답시간을 빠르게 하겠습니다. 응답시간을 빠르게 하겠습니다. 응답시간을 빠르게 하겠습니다. 응답시간을 빠르게 하겠습니다. 응답시간을 빠르게 하겠습니다.", 0)], expect=dict(interject=True, kind="repetition")),
]

REFS = {"s01_normal": "민원 접수 API는 캐시와 비동기 큐를 사용해서 평균 응답시간을 1.2초 수준으로 유지하겠습니다.",
 "s03_long": "저희는 먼저 로드밸런서를 두 대 이중화하고, 오토스케일링으로 동시 접속자 5000명까지 대응합니다. 또한 읽기 요청은 레디스 캐시로 처리하고 쓰기 요청은 메시지 큐로 비동기 처리합니다. 데이터베이스는 읽기 복제본을 세 대 두어 응답시간을 평균 1.2초 이내로 유지합니다. 부하 테스트는 배포 전에 매번 수행합니다.",
 "s04_pause": "응답시간은 캐시로 줄이겠습니다. 그리고 부하 테스트로 평균 1.2초를 검증하겠습니다.",
 "s09_rfp_violation": "개인정보 접근 로그는 비용 문제로 3개월만 보관하려고 합니다. 이 부분은 협의 가능하다고 생각합니다.",
 "s11_ramble": "저희 회사는 2000년에 설립되어 지금까지 여러 프로젝트를 수행했습니다. 사무실은 판교에 있고 직원들은 매우 화목하며 매년 워크숍을 갑니다. 창립 기념일에는 전 직원이 모여 행사를 열고 사내 동호회도 활발합니다. 작년에는 봉사활동도 다녀왔고 올해는 체육대회를 계획하고 있습니다. 구내식당 음식도 좋다는 평가를 받고 있습니다.",
 "s12_noise": "캐시와 비동기 큐로 응답시간을 1.2초로 유지하겠습니다."}
for _s in SCEN:
    if _s["id"] in REFS: _s["ref"] = REFS[_s["id"]]

def _say(text: str, voice="Yuna", rate=170) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "x.wav"
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(p), "--file-format=WAVE", "--data-format=LEI16@16000", text], check=True)
        a, sr = sf.read(p, dtype="float32"); assert sr == SR; return a

def build(out="data/audio", force=False):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True); meta = []
    rng = np.random.default_rng(0)
    for s in SCEN:
        wav, meta_p = out / f"{s['id']}.wav", out / f"{s['id']}.json"
        parts = [np.zeros(int(.5 * SR), np.float32)]
        for text, pause in s["segs"]:
            parts.append(_say(text))
            if pause: parts.append(np.zeros(int(pause * SR), np.float32))
        speech_end = sum(len(p) for p in parts)
        audio = np.concatenate(parts)
        if s.get("noise_snr_db"):
            sig = np.sqrt(np.mean(audio[audio != 0] ** 2)); n = rng.normal(0, sig / (10 ** (s["noise_snr_db"] / 20)), len(audio)).astype(np.float32)
            audio = audio + n
        audio = np.concatenate([audio, np.zeros(int(3 * SR), np.float32)])   # trailing silence for endpointing
        sf.write(wav, audio, SR, subtype="PCM_16")
        m = {**{k: v for k, v in s.items() if k != "segs"}, "text": " ".join(t for t, _ in s["segs"]), "ref": s.get("ref") or " ".join(t for t, _ in s["segs"]), "speech_end_s": speech_end / SR, "dur_s": len(audio) / SR, "wav": str(wav)}
        meta_p.write_text(json.dumps(m, ensure_ascii=False, indent=1)); meta.append(m)
    return meta

def load(path="data/audio"):
    return [json.loads(p.read_text()) for p in sorted(pathlib.Path(path).glob("*.json"))]

if __name__ == "__main__":
    for m in build(): print(m["id"], round(m["dur_s"], 1), "speech_end", round(m["speech_end_s"], 1))
