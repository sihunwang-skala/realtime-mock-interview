# Integration proposal: streaming Q&A with Spring backend and React frontend

Status: **proposal for team agreement** — no existing API or type is changed. Reviewed: PR #46 (draft, `frontend/src/types/api.ts`), PR #47 (attempt/video contract), `ai/analysis-engine/analysis/qa.py` and `/api/qa*`.

## Gaps in PR #46 for streaming speech-to-speech
| PR #46 assumption | Problem for streaming Q&A | Proposal |
|---|---|---|
| `QaSessionQuestion.audioUrl` per question | follow-up questions are generated *after* the answer; no URL exists in advance | questions arrive as runtime events; audio travels on the voice channel (WebSocket/WebRTC), not as a URL |
| `QaSessionCreateRequest.questionIds` fixed list | queue is dynamic (candidates added/resolved/discarded) | `questionIds` only seeds the queue (mode SELECTED); the engine owns the live queue |
| answer = uploaded file (`QaAnswerFields`) | answers are streamed and analysed incrementally | optional recording is saved by the engine and referenced in the report |
| no runtime states, controls or interjection | UI needs AGENT_THINKING / INTERJECTED etc. | event + command contract below |
| `QaSessionQuestion.isAnswered` | follow-ups have parents and actions | `QaTurn` with `parentTurnId`, `action` |

## Ownership
* **Spring**: auth, proposal/attempt/QA-session metadata, permission, **voice token issuance**, persistent report storage/retrieval.
* **Python qa engine**: voice runtime, STT/LLM/TTS, question queue, interjection, latency metrics, report generation.
* **Frontend**: audio connection, controls, state visualisation, case selector (dev/ops only), report display.

## Identifiers
`attemptId` (presentation rehearsal, PR #47) 1—0..n `qaSessionId` (new). A QA session references the attempt whose slides/script/RFP context it was prepared from; Phase A candidate questions are produced per attempt and copied into the QA session queue at start. Question count is never assumed final.

## REST (Spring)
| Method & path | Purpose |
|---|---|
| `POST /api/proposals/{pid}/attempts/{aid}/qa-sessions` body `{mode, panels, answerLimitSec, questionIds[]}` → `{qaSessionId, voice:{transport, url, token, expiresAt}, engineSessionId}` | create session; Spring asks the engine to prepare it and returns a short-lived token |
| `POST /api/qa-sessions/{id}/end` | end (idempotent) |
| `GET /api/qa-sessions/{id}/report` → `QaReport` | `REPORT_PROCESSING` -> 202, `REPORT_READY` -> 200 |

## Voice/event channel (engine; browser connects directly with the token)
WebSocket `/ws/voice?token=…` (prototype implemented in `qa_lab/demo/server.py`; WebRTC/LiveKit equivalent in Case A).
Client -> server: binary PCM16 mono 16 kHz; JSON commands. Server -> client: binary PCM16 agent audio; JSON events.

**Event envelope** `{type, session_id, seq, ts, data}` — `seq` strictly increasing per session; client drops `seq <= lastSeq` (duplicates / late arrivals) and may request replay from `seq`.
Events: `SESSION_READY, USER_SPEAKING, AGENT_SPEAKING, AGENT_THINKING, QUESTION_STARTED, ANSWER_COMPLETED, AGENT_INTERJECTED, SESSION_ENDED, REPORT_PROCESSING, REPORT_READY, ERROR`.
**Commands** `{cmd: START|REPEAT_QUESTION|SKIP_QUESTION|FINISH_ANSWER|END_SESSION, id?}` -> `{type:"CMD_RESULT", cmd, ok, error?}` (rejected e.g. "session not started", "user is not speaking", "no more questions").
**Latency probe**: client `ping{t0}` / server `pong{t0, server_ts}` -> offset estimate (NTP style); client reports `audio_started{server_clock_ts}` when the first agent sample is actually scheduled to play. Server clocks stay monotonic; wall clocks are never compared across machines.
**Barge-in bookkeeping**: client sends `played{seconds,total}` on interruption; engine records only the played fraction as "heard".

## Report schema (`QaReport`)
```json
{ "qaSessionId": 1, "attemptId": 7, "turns": [ {"turnId":"t1","parentTurnId":null,"question":{"text":"…","action":"SWITCH","reqId":"R1","evidence":"RFP R1","source":"presentation|answer|seed"},
  "answer":{"transcript":"…","durationSec":18.2,"assessment":{"sufficiency":0.4,"contradictions":[],"offTopic":0,"unsupportedClaims":[]}},
  "interjection":{"kind":"contradiction_rfp","reason":"…","atSec":6.2,"heardText":"…"}, "latency":{"e2eSec":1.6,"stages":{"vadEnd":0.6,"stt":0.5,"firstToken":0.4,"ttsFirstChunk":0.3}} } ],
  "coverage": {"R1":{"score":0.8,"evidence":[{"quote":"…"}]}}, "queueAudit":[{"op":"resolve","id":"…","reason":"answered later"}] }
```
Question text and answer transcripts are only exposed in the report, not during practice (matches PR #46 intent).

## Open decisions for the team
1. Who mints the voice token (recommended: Spring signs a short-lived JWT the engine validates).
2. Where recordings/transcripts live (see SECURITY.md) and retention period.
3. Whether the existing `/api/qa*` batch endpoints in analysis-engine stay as the non-realtime fallback (recommended: keep, untouched).
