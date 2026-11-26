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

## Voice channel — as implemented in the prototype (added for FE integration)
Source of truth: `qa_lab/demo/server.py` (`ws_voice`), `qa_lab/cases/runtime.py` (events), `qa_lab/contracts.py` (enums), `qa_lab/demo/static/index.html` (reference client). Describes the code on `experiment/qa-lab`; the remote branch differs slightly (see Known deltas).

**Connect**: `ws://<host>/ws/voice?<query>`. The prototype does **not** use `token=` yet. The query carries the full setup: `case` (B/C/…), `turn` (`manual`|`auto`), `serverplay` (`1`|`0`), `passage` (`1` = use stored passage), plus engine options (`llm`, `stt`, `tts`, `strategy`, `interjection`). Case C additionally requires `consent=1`. Invalid combinations return `{type:"ERROR", errors:[…]}` and close. A number alone is not enough.

**Audio up (client → server)**: binary frames, mono PCM16 little-endian, 16 kHz, captured by an AudioWorklet. Frames are ignored until `START` has succeeded.

**Audio down (server → client)**: binary frames, mono PCM16, 16 kHz, streamed in chunks as TTS produces them (not one file). The client schedules each chunk in Web Audio; the first scheduled chunk triggers `audio_started`.

**Server → client JSON** (`{type, session_id, seq, ts, data}` for events; `CMD_RESULT` and `pong` are flat):
| type | when | data |
|---|---|---|
| `SESSION_READY` | after connect | `components` (chosen engine parts) |
| `USER_SPEAKING` | user speech detected | `partial` (interim text, while speaking) |
| `AGENT_THINKING` | AI is generating the next question | — |
| `AGENT_SPEAKING` | AI audio lifecycle | `state`: `started` → `synth_done` (all audio sent; browser may still be playing) → `interrupted` (barge-in) or `server_played` (server-side playback) |
| `QUESTION_STARTED` | a question is about to be spoken | `text`, `followup`, `fu_type`, `fu_depth`, `fu_max`, `interjection`, `repeat`, `control`, `skipped`, `prev_q`, `prev_answer` (fields present only when relevant) |
| `AGENT_INTERJECTED` | AI cuts in | `kind`, `reason` |
| `ANSWER_COMPLETED` | user's answer is final (after `FINISH_ANSWER`) | `text`; later the same type again with `late:true`, `verdict`, `why`, `ref` (async judgment) |
| `LATENCY` | per turn | stage timings |
| `ERROR` | rejected/unsupported | top-level `errors[]` (not under `data`), then the socket closes |
| `SESSION_ENDED`, `REPORT_PROCESSING`, `REPORT_READY` | end of session | `report` on READY |
| `CMD_RESULT` | reply to each command | `cmd`, `ok`, `error?` |
| `pong` | reply to ping | `t0`, `server_ts` |

**Client → server JSON**:
| message | meaning |
|---|---|
| `{cmd:"START"}` | start session (first question). Expected right after `SESSION_READY`. |
| `{cmd:"START_ANSWER"}` | begin user turn (manual). Present on `experiment/qa-lab`, **removed on the remote branch**. |
| `{cmd:"FINISH_ANSWER"}` | "답변 끝" button (manual mode). |
| `{cmd:"REPEAT_QUESTION"}` / `{cmd:"SKIP_QUESTION"}` | repeat or skip the current question. |
| `{cmd:"END_SESSION"}` | ends; server replies `REPORT_PROCESSING`, `SESSION_ENDED`, `REPORT_READY`, then closes. |
| `{type:"ping", t0}` | latency probe. |
| `{type:"audio_started", server_clock_ts}` | first AI sample is scheduled to play. |
| `{type:"played", seconds, total}` | sent on barge-in; server records only the heard fraction. |
| `{type:"ptt", on}` | push-to-talk state (`true` = mic should be open). |
| `{type:"pair_label", label}` | tags the next answer pair (dev/eval only; not for production UI). |

**UI state (derived, no dedicated event)**: 말하는 중 = `USER_SPEAKING`; AI가 생각 중 = `AGENT_THINKING`; AI 말하는 중 = `AGENT_SPEAKING` with state `started` until `synth_done` and audio finished, or `interrupted`; 듣는 중 = none of the above (client-side). The prototype computes it from `#state`, which is a known fragile point.

**Known deltas (must resolve before FE depends on it)**
1. Auth: docs above say `?token=…`; prototype uses query options and no token. Token is still an open decision (see below).
2. `START_ANSWER`: present locally, removed on `origin/experiment/qa-lab` (03f2546).
3. `QUESTION_STARTED` fields differ: remote carries `prev_q`/`prev_answer`/`prev_src` on normal questions; local also adds `fu_*`, `followup`, `control`, `repeat`. Decide one set.
4. Report and `ANSWER_COMPLETED.late` semantics are only verified on local.

## Open decisions for the team
1. Who mints the voice token (recommended: Spring signs a short-lived JWT the engine validates).
2. Where recordings/transcripts live (see SECURITY.md) and retention period.
3. Whether the existing `/api/qa*` batch endpoints in analysis-engine stay as the non-realtime fallback (recommended: keep, untouched).
