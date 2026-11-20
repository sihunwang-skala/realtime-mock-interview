# realtime-mock-interview

[English](README.md) · **한국어**

발표(지문)를 듣고 AI가 **실시간으로 질문하는 음성 모의 면접**입니다. 모든 처리가 내 컴퓨터에서 도는 로컬 파이프라인(마이크 → VAD → Whisper → 판정 → LLM 질문 → 음성)이 기본입니다.

## 특징
- **일반 질문(랜덤):** 지문에서 미리 질문 은행을 만들어 두고, 남은 질문이 0~1개가 되면 3개를 다시 채웁니다. 이전 질문과 겹치지 않게, 지문의 앞·중간·뒤를 골고루 묻습니다.
- **꼬리질문:** 내 답변에서 새로 나온 단어를 지문에서 찾아 그 문장만 근거로 이어서 묻습니다. 더 자세히 · 검증 · 근거 · 순서 묻기 유형, 최대 3번 연속.
- **끼어들기:** 주제 이탈·반복·지문과 다른 숫자 등을 3초마다 확인해, 필요할 때 "말씀 중에 잠시만요"로 끼어듭니다.
- **버튼으로 진행:** AI 질문이 끝나면 마이크가 열리고, `말하기`를 누른 뒤 바로 말하고 `답변 끝`으로 마칩니다.
- **보고서:** 걸린 시간, 꼬리질문·끼어들기 횟수, 질문 종류(검정/파랑/빨강)별 기록, 지문 근거 기반 판정을 남깁니다.
- 실제 녹음·전사는 로컬에만 두고, 클라우드 모델은 합성 데이터에만 쓰도록 설계했습니다.

## 설치
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python openai httpx pydantic fastapi "uvicorn[standard]" websockets numpy pytest faster-whisper silero-vad torch soundfile
ollama serve &                      # 로컬 LLM (예: ollama pull phi4-mini)
```
환경 변수: `OPENAI_API_KEY`(선택, 클라우드 비교용 · 커밋 금지), `OLLAMA_URL`(기본 http://localhost:11434)

## 실행
```bash
.venv/bin/python -W ignore -m qa_lab serve --port 8765   # http://127.0.0.1:8765
.venv/bin/python -m pytest -q tests
```

## 문서
`docs/`에 모델 비교(`RANKING.md`, `MINI_SUITE.md`), 지연·발열 실험(`LATENCY_AB.md`), 보안(`SECURITY.md`)이 있습니다.

> 실험용 프로젝트이며 표본이 작아 수치는 참고용입니다.
