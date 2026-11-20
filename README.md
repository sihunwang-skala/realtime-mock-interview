# realtime-mock-interview

**English** · [한국어](README.ko.md)

A voice-based mock interview where an AI listens to your presentation answers and asks questions in real time, grounded in a source passage. The default setup is a fully local pipeline: microphone → VAD → Whisper → verdict → LLM question → speech.

## Features
- **Regular questions (random):** a question bank is generated from the passage in advance. When 0–1 questions are left, it refills 3 more, never repeating earlier questions and spreading them across the start, middle, and end of the passage.
- **Follow-up questions:** takes a new term from your answer, finds the passage sentence containing it, and asks only about that sentence. Types: elaborate, verify, rationale, and sequence (skipped steps). Up to 3 follow-ups in a row.
- **Interjections:** every 3 seconds it checks for off-topic talk, repetition, or numbers that contradict the passage, and politely cuts in when needed.
- **Button-driven turns:** the mic opens when the AI finishes its question; press `Talk`, speak right away, then press `Done` (Space works for both).
- **Report:** elapsed time, follow-up and interjection counts, a color-coded record per question kind (black = regular, blue = follow-up, red = interjection), and verdicts grounded in the passage.
- Real recordings and transcripts are meant to stay local; cloud models are meant for synthetic data only.

## Setup
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python openai httpx pydantic fastapi "uvicorn[standard]" websockets numpy pytest faster-whisper silero-vad torch soundfile
ollama serve &                      # local LLM (e.g. ollama pull phi4-mini)
```
Environment variables: `OPENAI_API_KEY` (optional, for cloud comparisons; never commit it), `OLLAMA_URL` (default http://localhost:11434).

## Run
```bash
.venv/bin/python -W ignore -m qa_lab serve --port 8765   # http://127.0.0.1:8765
.venv/bin/python -m pytest -q tests
```

## Docs
`docs/` has model comparisons (`RANKING.md`, `MINI_SUITE.md`), latency and heat experiments (`LATENCY_AB.md`), and security notes (`SECURITY.md`). Most docs are written in Korean.

> This is an experimental project. Sample sizes are small, so treat the numbers as indicative only.
