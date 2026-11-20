# Security and privacy notes (qa-lab)

* **Local-only vs cloud**: Case B (all components local: Ollama, faster-whisper, AVSpeech, Silero) never sends audio/text off the machine. Case A as built is also local (loopback LiveKit dev server, local models). Case C and any `openai:` adapter send **audio and RFP/answer text to OpenAI**; use only public/synthetic RFPs for those demos. The demo UI marks `local_only` per case.
* **Data used in this repo's runs**: a synthetic RFP (`qa_lab/data/rfp.py`) and TTS-synthesised answers. No customer RFP, no human recordings.
* **Secrets**: keys only via environment (`OPENAI_API_KEY`); nothing is written to source or results. The key found in this environment was rejected by OpenAI (HTTP 401), so no cloud call succeeded.
* **Storage**: the prototype keeps session state in memory and writes only benchmark JSON under `results/` (synthetic data). A production deployment must define: recording/transcript/question-log location (encrypted bucket per tenant), access by proposal owner only, and deletion on request or after a retention window (proposal: 30 days for audio, report retained until the user deletes the proposal).
* **Access control**: the demo WebSocket has **no authentication** and binds to 127.0.0.1. Production must require a Spring-issued short-lived token scoped to one qaSessionId.
* **Before using cloud providers with real data**: confirm enterprise no-training terms, data residency, retention, and sub-processor lists; the Bedrock path already used by analysis-engine (`LLM_PROVIDER=bedrock`) is the likelier compliant route.
