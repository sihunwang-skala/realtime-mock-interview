"""Streaming voice runtime used by Case B (local). Transport-agnostic: caller feeds PCM frames and receives events/audio.

Latency marks (monotonic seconds, name -> t):
 USER_SPEECH_ACTUAL_END (harness-provided), VAD_END_DETECTED, STT_FINALIZED, ANSWER_ANALYSIS_COMPLETED, QUESTION_DECISION_COMPLETED,
 QUESTION_GENERATION_STARTED, QUESTION_FIRST_TOKEN, TTS_FIRST_AUDIO_CHUNK, AUDIO_FIRST_PLAYED (browser-reported, or sink-emission estimate in harness)
"""
from __future__ import annotations
import asyncio, contextvars, re, time, numpy as np
from ..contracts import Action, Event, EventType

CLOSING_TEXT = "네~ 여기까지 들었습니다. 오늘 발표 수고하셨습니다."
TIME_UP_SILENCE_S = 10.0        # after the time limit: if nobody speaks for this long, close without waiting for an answer
from ..core.engine import QAEngine
from ..providers.tts import chunk_sentences, cached_audio, prewarm_audio

SR = 16000
class Hold:
    """Speculative response whose audio is buffered until the turn end is confirmed."""
    def __init__(self): self.held = []; self.released = False; self.marks = {}; self.deferred = []
_HOLD: contextvars.ContextVar = contextvars.ContextVar('hold', default=None)
_MK: contextvars.ContextVar = contextvars.ContextVar('marks', default=None)   # per-task latency marks (concurrent respond/interjection tasks must not clobber each other)
KIND_ACTION = {"contradiction_rfp": Action.CHALLENGE, "contradiction_prior": Action.CHALLENGE, "off_topic": Action.CLARIFY, "repetition": Action.CLARIFY, "unsupported": Action.VERIFY}

class VoiceRuntime:
    def __init__(self, engine: QAEngine, vad, stt, tts, session_id="s", on_event=None, on_audio=None, partial_every=3.0, partial_window=14.0, enable_interjection=True, preempt=True, vad_late=None, turn_mode="auto"):
        self.e = engine; self.vad = vad; self.stt = stt; self.tts = tts; self.sid = session_id
        self.on_event = on_event or (lambda ev: None); self.on_audio = on_audio or (lambda pcm, meta: None)
        self.partial_every = partial_every; self.partial_window = partial_window; self.enable_interjection = enable_interjection; self.preempt = preempt
        self.seq = 0; self.marks: dict[str, float] = {}; self.turn_marks: list[dict] = []
        self.buf: list[np.ndarray] = []; self.in_speech = False; self.speech_t0 = 0.0; self.last_partial_n = 0
        self.agent_task: asyncio.Task | None = None; self.partial_task: asyncio.Task | None = None; self.agent_speaking = False
        self.partial_text = ""; self.played_s = 0.0; self.interjected = False; self.ij_cut = 0; self.interjections: list[dict] = []
        self.pending = bytearray(); self.last_final_text = ""; self.tail = np.zeros(0, np.float32); self.responses: list[dict] = []
        self.spec: asyncio.Task | None = None; self.spec_key = ""; self.vad_late = vad_late; self.spec_hold: Hold | None = None; self.spec_count = 0; self.spec_cancelled = 0
        self.turn_mode = turn_mode; self.cut = 0; self.chunk_tasks: list[asyncio.Task] = []     # manual mode: the user ends the turn with a button

    def emit(self, t: EventType, **data):
        self.seq += 1; ev = Event(t, self.sid, self.seq, time.time(), data); self.on_event(ev); return ev
    def _cur(self): m = _MK.get(); return m if m is not None else self.marks
    def mark(self, name: str, t: float | None = None): self._cur()[name] = t if t is not None else time.monotonic()

    # ------------- input -------------
    async def feed(self, pcm: np.ndarray):
        """pcm: float32 mono 16k. Any length; processed in 512-sample frames."""
        self.pending.extend(pcm.astype(np.float32).tobytes())
        fb = 512 * 4
        while len(self.pending) >= fb:
            frame = np.frombuffer(bytes(self.pending[:fb]), np.float32); del self.pending[:fb]
            await self._frame(frame)

    async def _frame(self, frame):
        r = self.vad.feed(frame); rl = self.vad_late.feed(frame) if self.vad_late else None
        if getattr(self, 'ptt', False) and not self.in_speech: r = rl = None     # push-to-talk: only the button starts an answer
        if self.in_speech or r == "start": self.buf.append(frame)
        else:
            self.tail = np.concatenate([self.tail, frame])[-int(0.3 * SR):]      # pre-roll so first phoneme isn't clipped
        if r == "start" and not self.in_speech:
            self.in_speech = True; self.speech_t0 = time.monotonic(); self.buf = [self.tail, frame]; self.last_partial_n = 0; self.cut = 0; self.chunk_tasks = []
            self.partial_text = ""; self.interjected = False; self.ij_cut = 0; self.spec_key = ""
            if self.agent_speaking: await self._barge_in()
            self.emit(EventType.USER_SPEAKING)
        elif r == "start" and self.in_speech and self.spec_hold is not None:
            self._cancel_spec()                                                  # speech resumed before the turn was confirmed
        if self.turn_mode == "manual":                                           # silence never ends a turn; pauses only trigger background transcription
            if self.in_speech and r == "end": self._commit_chunk()
        elif self.vad_late is not None:                                          # speculative mode: early end -> prepare, late end -> confirm
            if self.in_speech and r == "end": self._start_spec()
            if self.in_speech and rl == "end": self._confirm()
        elif self.in_speech and r == "end":
            self.in_speech = False; self.marks = {}; self.mark("VAD_END_DETECTED")
            audio = np.concatenate(self.buf); self.buf = []
            self.agent_task = asyncio.create_task(self._respond(audio))
        if self.in_speech and r != "end" and not self.interjected and self.enable_interjection:
            n = sum(len(b) for b in self.buf)
            first = getattr(self, "first_partial", None)                      # experiment knob: first transcription earlier than the regular interval (None = same as partial_every)
            gap = first if (first is not None and self.last_partial_n == 0) else self.partial_every
            if n - self.last_partial_n >= gap * SR and (self.partial_task is None or self.partial_task.done()):
                self.last_partial_n = n
                self.partial_task = asyncio.create_task(self._partial(np.concatenate(self.buf)[self.ij_cut:][-int(self.partial_window * SR):], time.monotonic() - self.speech_t0))

    def _commit_chunk(self):
        """Manual mode, natural pause: transcribe the audio since the last cut in the background so the final press only needs the tail."""
        total = np.concatenate(self.buf); seg = total[self.cut:]; self.cut = len(total)
        if len(seg) >= int(0.3 * SR): self.chunk_tasks.append(asyncio.create_task(asyncio.to_thread(self.stt.transcribe, seg)))

    async def begin_answer(self):
        """Push-to-talk: the user pressed 'start answer' and speaks right away. Returns (ok, reason)."""
        if self.in_speech: return True, ""
        if self.agent_speaking: return False, "AI가 말하는 중이에요. 질문이 끝나면 눌러 주세요"
        self.in_speech = True; self.speech_t0 = time.monotonic(); self.buf = [self.tail[-int(0.1 * SR):]]; self.last_partial_n = 0; self.cut = 0; self.chunk_tasks = []
        self.partial_text = ""; self.interjected = False; self.ij_cut = 0; self.spec_key = ""
        self.emit(EventType.USER_SPEAKING); return True, ""

    async def finish_answer(self):
        """Manual mode: the user pressed 'answer done'. Returns (ok, reason). Latency clock for this turn starts at the press."""
        if not self.in_speech: return False, "아직 말씀하신 내용이 없습니다"
        if self.agent_speaking: return False, "AI가 말하는 중입니다"
        press = time.monotonic(); self.in_speech = False
        total = np.concatenate(self.buf); tail = total[self.cut:]; self.buf = []; self.cut = 0
        tasks, self.chunk_tasks = self.chunk_tasks, []
        self.marks = {"VAD_END_DETECTED": press, "BUTTON_PRESSED": press}
        self.agent_task = asyncio.create_task(self._respond(None, marks0=dict(self.marks), pre=(tasks, tail))); return True, ""

    def _start_spec(self):
        if self.spec_hold is not None: self._cancel_spec()
        self.marks = {}; self.mark("VAD_END_DETECTED")                         # early end (e.g. 600 ms)
        h = Hold(); h.marks = {}; self.spec_hold = h; self.spec_count += 1
        audio = np.concatenate(self.buf)[-int(40 * SR):]
        self.last_audio = audio; self.last_t = time.monotonic(); self.last_used = False                      # kept in case the button is pressed after the turn ended by itself
        self.agent_task = asyncio.create_task(self._respond(audio, hold=h, marks0=dict(self.marks)))
    def _cancel_spec(self):
        if self.agent_task and not self.agent_task.done(): self.agent_task.cancel()
        self.spec_hold = None; self.agent_speaking = False; self.spec_cancelled += 1
    def _confirm(self):
        """Late end (e.g. 1.5 s of silence): the turn is really over. Release held audio."""
        self.in_speech = False; h = self.spec_hold; self.spec_hold = None; self.buf = []
        if h is None: return
        h.marks["TURN_CONFIRMED"] = time.monotonic(); h.released = True
        for pcm, item in h.held:
            if "AUDIO_FIRST_PLAYED" not in h.marks: h.marks["AUDIO_FIRST_PLAYED"] = time.monotonic()
            self._spoken_total_s = getattr(self, "_spoken_total_s", 0) + len(pcm) / SR; self.on_audio(pcm, {"text": item})
        h.held = []
        for fn in h.deferred: fn()
        h.deferred = []

    async def _partial(self, audio, elapsed):
        txt = await asyncio.to_thread(self.stt.transcribe, audio)
        if not self.in_speech: return
        self.partial_text = txt
        self.emit(EventType.USER_SPEAKING, partial=txt)
        self.e.speculate_followup(txt); self._kick_spec_audio()                  # tailored follow-up is made while the person is still talking
        if time.monotonic() < getattr(self, "ij_block_until", 0): return
        d = await self.e.should_interject(self.partial_text, elapsed)
        if d.interject and self.in_speech and not self.interjected:
            self.interjected = True                                              # reserve the slot while the question is being made
            q = await self.e.make_interjection(d.kind, txt) if getattr(self.e, "keep_going", False) else None
            if getattr(self.e, "keep_going", False) and (not q or not self.in_speech):
                self.interjected = False; self.ij_block_until = time.monotonic() + 8     # nothing new to ask: stay quiet, retry later
                return
            t = time.monotonic()
            self.interjections.append({"t_after_speech_start": t - self.speech_t0, "kind": d.kind, "reason": d.reason, "policy": d.policy, "partial": txt[-600:]})
            self.emit(EventType.AGENT_INTERJECTED, kind=d.kind, reason=d.reason)
            self.agent_task = asyncio.create_task(self._speak_interjection(d, txt, t, q))

    async def _barge_in(self):
        if self.agent_task and not self.agent_task.done(): self.agent_task.cancel()
        played_frac = min(1.0, self.played_s / max(self._spoken_total_s, 1e-6)) if getattr(self, "_spoken_total_s", 0) else 0
        self.e.mark_played(int(len(self.e.spoken_full) * played_frac)); self.agent_speaking = False
        self.emit(EventType.AGENT_SPEAKING, state="interrupted", heard=self.e.heard_text())

    # ------------- output -------------
    async def _speak_stream(self, gen, key: str):
        """gen yields (action, token). Chunks text, synthesises, emits audio; records marks."""
        buf = ""; first = True; q_text = ""; self.played_s = 0.0; self._spoken_total_s = 0.0
        self.agent_speaking = True; self.emit(EventType.AGENT_THINKING); self.mark("QUESTION_GENERATION_STARTED")
        queue: asyncio.Queue = asyncio.Queue()
        async def synth_worker():
            while True:
                item = await queue.get()
                if item is None: return
                async for pcm in self._tts_stream(item):
                    cur = self._cur(); h = _HOLD.get()
                    if "TTS_FIRST_AUDIO_CHUNK" not in cur: self.mark("TTS_FIRST_AUDIO_CHUNK")
                    if h is not None and not h.released: h.held.append((pcm, item)); continue          # speculative: buffer until confirmed
                    if "AUDIO_FIRST_PLAYED" not in cur:
                        self.mark("AUDIO_FIRST_PLAYED")   # sink-emission estimate (browser overrides)
                        self.emit(EventType.AGENT_SPEAKING, state="started")
                        self._emit_latency(self._cur())
                    self._spoken_total_s += len(pcm) / SR; self.on_audio(pcm, {"text": item})
                self.played_s = self._spoken_total_s
        w = asyncio.create_task(synth_worker()); act = None
        try:
            async for act, tok in gen:
                if "QUESTION_FIRST_TOKEN" not in self._cur(): self.mark("QUESTION_FIRST_TOKEN")
                buf += tok; q_text += tok
                chunks, buf = chunk_sentences(buf, first)
                for c in chunks: first = False; queue.put_nowait(c)
            if buf.strip(): queue.put_nowait(buf.strip())
            queue.put_nowait(None); await w
            self.emit(EventType.AGENT_SPEAKING, state="synth_done")       # all audio has been sent (browser may still be playing it)
        except asyncio.CancelledError:
            w.cancel(); raise
        finally: self.agent_speaking = False
        return act, q_text

    async def _tts_stream(self, text: str):
        a = cached_audio(self.tts, text)                                         # pre-synthesised chunk: play at once
        if a is not None: yield a; return
        async for pcm in self.tts.stream(text): yield pcm

    def _kick_spec_audio(self):
        """Voice the tailored follow-up as soon as it exists (background), so it can start at once when the answer ends."""
        t = getattr(self.e, "spec_task", None); old = getattr(self, "_spec_audio", None)
        if t is None or (old is not None and not old.done()) or not getattr(self.tts, "local", False): return
        async def run():
            try:
                await t; fu = getattr(self.e, "spec_fu", None)
                if fu: await prewarm_audio(self.tts, [fu.get("say") or fu["q"]])
            except asyncio.CancelledError: raise
            except Exception: pass
        self._spec_audio = asyncio.create_task(run())

    def start_audio_prewarm(self):
        """Background loop: voice every prepared question / follow-up / interjection template ahead of time, so speaking them is instant."""
        from ..core.passage import PRIORITY, REFILL_AT, extend_bank
        if getattr(self, "_audio_task", None) or not getattr(self.tts, "local", False): return
        async def loop():
            try:
                while True:
                    await prewarm_audio(self.tts, self.e.canned_texts(), busy=lambda: PRIORITY["busy"] or getattr(self, "_urgent", False))
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError: raise
            except Exception: pass
        self._audio_task = asyncio.create_task(loop())
        if not getattr(self, "_refill_task", None):                                  # separate loop: voicing 20+ sentences can take 30 s, the refill check must not wait for it
            async def refill_loop():
                try:
                    while True:
                        bk = getattr(self.e, "_bank", None)
                        if bk: self.e.sync_bank()                                   # count the questions the bank has finished, or we would ask for more again and again
                        if bk and bk.get("done") and not bk.get("extending") and not bk.get("exhausted") and len(self.e.queue.pending()) <= REFILL_AT and not PRIORITY["busy"]:
                            await extend_bank(bk)
                        if self.in_speech and not PRIORITY["busy"] and not self.e.refill_wanted() and len(self.partial_text.strip()) >= getattr(self.e, "spec_min_chars", 15):
                            self.e.speculate_followup(self.partial_text); self._kick_spec_audio()     # not waiting for the next 4 s transcription tick
                        await asyncio.sleep(0.3)
                except asyncio.CancelledError: raise
                except Exception as e_: print("[refill error]", repr(e_), flush=True)
            self._refill_task = asyncio.create_task(refill_loop())

    def _emit_latency(self, m: dict):
        """Per-stage breakdown (seconds) of the wait before the AI started speaking: transcription / question generation / voice synthesis."""
        try:
            base = m.get("BUTTON_PRESSED") or m.get("VAD_END_DETECTED") or m.get("CONTROL_COMMAND") or m.get("QUESTION_GENERATION_STARTED"); end = m["AUDIO_FIRST_PLAYED"]
            g0 = m.get("QUESTION_GENERATION_STARTED"); g1 = m.get("QUESTION_FIRST_TOKEN"); t1 = m.get("TTS_FIRST_AUDIO_CHUNK")
            d = {"total_s": round(end - base, 2)}
            if "STT_FINALIZED" in m: d["stt_s"] = round(m["STT_FINALIZED"] - base, 2)
            if g0 is not None and g1 is not None: d["generate_s"] = round(g1 - g0, 2)
            if g1 is not None and t1 is not None: d["tts_s"] = round(t1 - g1, 2)
            self.emit(EventType.LATENCY, **d)
        except Exception: pass

    async def _respond(self, *a, **k):
        from ..core.passage import PRIORITY
        PRIORITY["busy"] = True                                                 # the background question bank steps aside while we answer
        try:                                                                      # what else was running when the answer ended (to explain a slow transcription later)
            import os as _os; bk_ = getattr(self.e, "_bank", None); sp_ = getattr(self.e, "spec_task", None)
            self._snap = {"load1": round(_os.getloadavg()[0], 1), "bank_refill_running": bool(bk_ and bk_.get("extending")), "followup_being_written": bool(sp_ is not None and not sp_.done()),
                          "unasked_questions": len(self.e.queue.pending())}
        except Exception: self._snap = {}
        t_ = getattr(self.e, "spec_task", None)
        if t_ is not None and not t_.done(): t_.cancel()                          # a follow-up still being written when the answer ends is dropped: it would slow the transcription (measured: LLM running = 2x slower Whisper)
        try: return await self._respond_inner(*a, **k)
        finally: PRIORITY["busy"] = False

    async def _respond_inner(self, audio, hold: Hold | None = None, marks0: dict | None = None, pre=None):
        self.last_used = True
        self.e.stop_prefetch()                                                  # the answer is finished: free the LLM for the next question
        mk = dict(marks0 if marks0 is not None else self.marks); _MK.set(mk); _HOLD.set(hold)
        if hold is not None: hold.marks = mk
        def ev(t, **d): (hold.deferred.append(lambda: self.emit(t, **d)) if hold is not None and not hold.released else self.emit(t, **d))
        if pre is not None:                                                     # manual mode: earlier chunks were transcribed during pauses; only the tail is left
            tasks, tail = pre; parts = [await t for t in tasks]
            if tail is not None and len(tail) >= int(0.3 * SR) and float(np.sqrt((tail ** 2).mean())) > 0.005: parts.append(await asyncio.to_thread(self.stt.transcribe, tail))
            text = " ".join(x for x in parts if x).strip()
        else: text = await asyncio.to_thread(self.stt.transcribe, audio)
        self.mark("STT_FINALIZED")
        if not text.strip(): return                                              # silence / hallucination: no turn, no question
        ev(EventType.ANSWER_COMPLETED, text=text)
        if getattr(self, "time_up", False): await self._speak_closing(); return     # time limit reached: this answer was heard, no new question
        if self.e.strategy in ("two_track", "precomputed"):
            a = self.e.fast_assess(text); self.mark("ANSWER_ANALYSIS_COMPLETED")        # fast track: heuristics only
            defer_bg = text
        else:
            a = await self.e.analyze_answer(text); self.mark("ANSWER_ANALYSIS_COMPLETED")
        prevq = self.e.cur; qprev = prevq.text if prevq else ""; prevref = self.e.ref_of(prevq)
        v = self.e.quick_verdict(text, qprev) if getattr(self.e, "keep_going", False) else None   # live experiment: cheap verdict picks the next area (no extra latency)
        gen = self.e.plan_and_stream(text, a, forced_action=(self.e.next_action(v, a) if v else None), verdict=v)
        self.mark("QUESTION_DECISION_COMPLETED")
        act, q = await self._speak_stream(gen, "followup")
        pa = self.e.get_answer(prevq) if getattr(self.e, "keep_going", False) else None     # model answer of the question that was just answered (instant)
        if self.e.strategy in ("two_track", "precomputed") and not getattr(self.e, "last_clarify", None): self.bg = asyncio.create_task(self.e.analyze_answer(text))   # deep track runs while the user answers, never competes with question generation
        from ..core.engine import MAX_FU
        _isfu = bool(act is not None and act.value in ("DEEPEN", "CLARIFY", "VERIFY", "CHALLENGE")); _cf = getattr(self.e, "cur_fu", None) if _isfu else None; self.e.cur_fu = None
        _why = self._why_action(v, prevq, act)
        self.responses.append({"stt_text": text, "fu_note": _why, "is_fu": _isfu, "fu_type": ((_cf or {}).get("type") or "새로 만든 질문") if _isfu else None, "fu_depth": ((_cf or {}).get("depth") or self.e.followups) if _isfu else None, "assessment": a.__dict__, "action": act.value if act else None, "question": q.strip(), "model_answer": (pa or {}).get("text") or "", "answered_q": qprev, "ref_text": prevref, "verdict_pending": True, "at_end": dict(getattr(self, "_snap", {}) or {}), "marks": mk if hold is not None else dict(self._cur()), "interjected_during": self.interjected, "speculative": hold is not None, "quick_verdict": v and v["verdict"], "verdict": v and v["verdict"], "why": v and v["why"], "ref": v and v["ref"], "model_answer": (pa or {}).get("text"), "model_answer_src": (pa or {}).get("src")})
        _cl = getattr(self.e, "last_clarify", None); self.e.last_clarify = None                 # "repeat" | "easier" | None: the person asked to hear it again / more simply (not an answer, never graded)
        _cq = getattr(self.e, "cur", None)                                                      # where the question just asked came from: user | bank | answer(follow-up) ... and its priority
        _meta = {"clarify": _cl} if _cl else {"q_source": "followup" if _isfu else getattr(_cq, "source", None), "q_priority": round(getattr(_cq, "priority", 0.0), 3), "q_user_text": getattr(_cq, "user_text", None)}
        self.responses[-1].update(_meta)
        ev(EventType.QUESTION_STARTED, text=q, followup=_isfu and not _cl, fu_type=((_cf or {}).get("type") or "새로 만든 질문") if _isfu else None, fu_depth=((_cf or {}).get("depth") or self.e.followups) if _isfu else None, fu_max=MAX_FU, prev_q=qprev, prev_answer=(pa or {}).get("text"), prev_src=(pa or {}).get("src"), **_meta)
        self.e.start_answer(self.e.cur)                                          # write the model answer for the question just asked, while the person answers
        # the question each answer belongs to and its source sentence are stored with every answer (answered_q, ref_text above); the report uses only these
        self.e.start_prefetch()                                                 # prepare the next questions while the person answers

    async def _bg_once(self, factory):
        """Run ONE background LLM step, but give up (return None) the moment the person starts or finishes speaking, so it can never delay a live step.
        The local model serves one request at a time: anything still running when the next answer needs it would make the next question slower."""
        from ..core.passage import PRIORITY
        task = asyncio.ensure_future(factory())
        try:
            while not task.done():
                if self.in_speech or PRIORITY["busy"]: return None
                await asyncio.sleep(0.02)
            return task.result()
        finally:
            if not task.done(): task.cancel()

    async def _late_verdict(self, entry, text, qprev, prevref=""):
        """No LLM judging during the session (it slowed the next question and heated the Mac). Keep only what the report needs: which question this answer belongs to.
        The verdict and the model answer are made when the report is opened."""
        entry["answered_q"] = qprev; entry["ref_text"] = prevref; entry["verdict_pending"] = True

    def _why_action(self, v, prevq, act) -> str:
        """One line for the report: why this turn's next question is a follow-up or a new-area question."""
        from ..core.engine import MAX_FU
        vd = (v or {}).get("verdict", "")
        if act is not None and act.value in ("deepen", "clarify", "verify", "challenge", "DEEPEN", "CLARIFY", "VERIFY", "CHALLENGE"): return getattr(self.e, "last_fu_note", "") or "꼬리질문"
        if vd in ("모름", "틀림"): return f"판정 '{vd}' → 같은 영역을 파고들지 않고 새 영역으로"
        if self.e.followups >= MAX_FU: return f"꼬리질문 {MAX_FU}번을 채워서 새 영역으로"
        return getattr(self.e, "last_fu_note", "") or "새 영역"

    async def _speak_interjection(self, d, partial, t_trig, q_ready=None):
        import re as _re
        act = KIND_ACTION.get(d.kind, Action.CLARIFY)
        from ..core.contracts_util import synthetic_assessment
        a = synthetic_assessment(d)
        _MK.set({"INTERJECTION_TRIGGERED": t_trig})
        if q_ready:                                                              # validated question (live experiment): speak it as is
            async def g():
                for w in _re.findall(r"\S+\s*", q_ready): yield act, w
            gen = g()
        else: gen = self.e.plan_and_stream(partial, a, forced_action=act, interject_kind=d.kind)
        act, q = await self._speak_stream(gen, "interjection")
        if q_ready: self.e.accept_interjection(q_ready, partial, a, act)
        self.responses.append({"interjection": True, "question": q.strip(), "marks": dict(self._cur()), "kind": d.kind})
        self.emit(EventType.QUESTION_STARTED, text=q.strip(), interjection=True, kind=d.kind)
        self.rearm_interjection()

    def rearm_interjection(self):
        """After an interjection the person may keep talking in the same turn: allow another one, judged only on what is said from now on (fresh timer, old audio ignored)."""
        if not self.in_speech: return
        self.interjected = False; self.speech_t0 = time.monotonic()
        n = sum(len(b) for b in self.buf); self.ij_cut = n; self.last_partial_n = n; self.partial_text = ""

    # ------------- controls -------------
    async def speak_text(self, text: str, action: Action = Action.SWITCH, extra: dict | None = None):
        """Speak a known question (first question, REPEAT, SKIP). No LLM call."""
        import re as _re
        async def g():
            for w in _re.findall(r"\S+\s*", text): yield action, w
        if self.agent_task and not self.agent_task.done(): self.agent_task.cancel()
        async def run():
            self._urgent = True                                                  # a question is being spoken now: background voicing steps aside
            try:
                _MK.set({"CONTROL_COMMAND": time.monotonic()}); act_, q_ = await self._speak_stream(g(), "control")
            finally: self._urgent = False
            self.emit(EventType.QUESTION_STARTED, text=q_.strip(), control=True, repeat=(action == Action.REPEAT), **(extra or {}))
            if action != Action.REPEAT: self.e.start_answer(self.e.cur)
            self.e.start_prefetch(); return act_, q_
        self.agent_task = asyncio.create_task(run()); await asyncio.sleep(0)
    async def force_end(self):
        """FINISH_ANSWER in automatic mode. Returns (ok, reason). Pressing the button must never be a dead end:
        a prepared (speculative) answer is confirmed at once; an answer already being processed counts as submitted; an utterance that just ended and was not handled yet is handled now."""
        if self.in_speech:
            if self.spec_hold is not None: self._confirm(); return True, ""                       # the answer was prepared at the first pause: release it now instead of waiting for the 1.5 s confirmation
            self.in_speech = False; self.marks = {}; self.mark("VAD_END_DETECTED"); audio = np.concatenate(self.buf); self.buf = []
            self.agent_task = asyncio.create_task(self._respond(audio)); return True, ""
        if self.agent_task and not self.agent_task.done(): return True, "답변이 이미 접수돼서 AI가 처리 중이에요"
        la = getattr(self, "last_audio", None)
        if la is not None and not getattr(self, "last_used", True) and time.monotonic() - getattr(self, "last_t", 0) < 15:
            self.marks = {}; self.mark("VAD_END_DETECTED"); self.agent_task = asyncio.create_task(self._respond(la)); return True, ""     # it ended by itself and nothing handled it
        return False, "아직 말씀하신 내용이 없어요 (마이크에 소리가 안 들어왔어요)"
    def client_played(self, t_server: float):
        """Browser-reported first audible playback (server clock). Overrides the sink-emission estimate."""
        self.marks["AUDIO_FIRST_PLAYED_BROWSER"] = t_server
    def arm_time_limit(self, seconds: float):
        """Session length (10/20/30 min). When it runs out no new question is asked. The answer being given is still heard
        (the answer is processed in _respond_inner, which then speaks the closing line). If nobody speaks for TIME_UP_SILENCE_S, the closing line is spoken anyway."""
        self.time_up = False; self.closing_done = False
        async def watch():
            await asyncio.sleep(seconds)
            self.time_up = True; self.emit(EventType.TIME_UP, state="limit_reached")
            idle = 0.0
            while not self.closing_done:
                busy = self.in_speech or self.agent_speaking or (self.agent_task is not None and not self.agent_task.done())
                idle = 0.0 if busy else idle + 0.2
                if idle >= TIME_UP_SILENCE_S: await self._speak_closing(); return
                await asyncio.sleep(0.2)
        self._limit_task = asyncio.create_task(watch())
    async def _speak_closing(self):
        """Natural ending line, spoken once. No question, no LLM call."""
        if self.closing_done: return
        self.closing_done = True
        async def g():
            for w in re.findall(r"\S+\s*", CLOSING_TEXT): yield Action.END, w
        try:
            if self.agent_task is not None and self.agent_task is not asyncio.current_task() and not self.agent_task.done(): await self.agent_task
        except Exception: pass
        self.emit(EventType.TIME_UP, state="closing")
        await self._speak_stream(g(), "closing")
        self.emit(EventType.TIME_UP, state="closed")
    async def close(self):
        for t in (self.agent_task, self.partial_task, getattr(self, "bg", None), getattr(self, "_audio_task", None), getattr(self, "_refill_task", None), getattr(self, "_limit_task", None)):
            if t and not t.done(): t.cancel()
