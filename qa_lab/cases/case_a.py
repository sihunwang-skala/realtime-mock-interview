"""Case A: LiveKit (self-hosted dev server, WebRTC) transport + the SAME modular cascade as Case B.

Scope honesty: this uses the LiveKit *rtc SDK + server* for transport (real WebRTC, jitter buffer, Opus), with our own VoiceRuntime
for VAD/STT/LLM/TTS orchestration. It does NOT use the livekit-agents framework (AgentSession, its turn-detector model, preemptive
generation, SpeechHandle). Those framework features are therefore NOT_TESTED here."""
from __future__ import annotations
import asyncio, os, subprocess, time, shutil, socket
import numpy as np
from livekit import rtc, api
from . import case_b
from .runtime import VoiceRuntime

URL = "ws://127.0.0.1:7880"; KEY, SECRET = "devkey", "secret"; ROOM = "qa-lab"
_SRV = None

def ensure_server():
    global _SRV
    s = socket.socket(); s.settimeout(0.3)
    if s.connect_ex(("127.0.0.1", 7880)) == 0: s.close(); return
    if not shutil.which("livekit-server"): raise RuntimeError("livekit-server not installed (brew install livekit)")
    _SRV = subprocess.Popen(["livekit-server", "--dev", "--bind", "127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        time.sleep(0.2); s = socket.socket(); s.settimeout(0.3)
        if s.connect_ex(("127.0.0.1", 7880)) == 0: s.close(); return
    raise RuntimeError("livekit-server failed to start")

def token(identity: str) -> str:
    return api.AccessToken(KEY, SECRET).with_identity(identity).with_grants(api.VideoGrants(room_join=True, room=ROOM)).to_jwt()

class LKSession:
    """Agent participant + (for benchmarks) a client participant in one LiveKit room."""
    def __init__(self, make_engine, make_runtime, with_client=True):
        self.make_engine = make_engine; self.make_runtime = make_runtime; self.with_client = with_client
        self.rt: VoiceRuntime | None = None; self.client_first = None; self.armed = False; self.last_loud = 0.0; self.out_q: asyncio.Queue = asyncio.Queue(); self.tasks = []
    async def start(self):
        ensure_server()
        self.agent = rtc.Room(); await self.agent.connect(URL, token("agent"))
        self.a_src = rtc.AudioSource(16000, 1); self.a_track = rtc.LocalAudioTrack.create_audio_track("agent-voice", self.a_src)
        await self.agent.local_participant.publish_track(self.a_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        @self.agent.on("track_subscribed")
        def _ts(track, pub, part):
            if track.kind == rtc.TrackKind.KIND_AUDIO: self.tasks.append(asyncio.create_task(self._agent_rx(track)))
        self.tasks.append(asyncio.create_task(self._agent_tx()))
        if self.with_client:
            self.client = rtc.Room(); await self.client.connect(URL, token("user"))
            self.c_src = rtc.AudioSource(16000, 1); self.c_track = rtc.LocalAudioTrack.create_audio_track("user-mic", self.c_src)
            await self.client.local_participant.publish_track(self.c_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
            @self.client.on("track_subscribed")
            def _cs(track, pub, part):
                if track.kind == rtc.TrackKind.KIND_AUDIO: self.tasks.append(asyncio.create_task(self._client_rx(track)))
            await asyncio.sleep(1.0)
    async def quiesce(self, quiet=1.0, timeout=40):
        """Let the previous turn's agent audio (paced in real time by the track) drain so it can't be mistaken for the next response."""
        self.armed = False; self.a_src.clear_queue()
        while not self.out_q.empty(): self.out_q.get_nowait()
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout and time.monotonic() - self.last_loud < quiet: await asyncio.sleep(0.1)
    def new_turn(self, engine):
        self.client_first = None; self.armed = True
        self.rt = self.make_runtime(engine, lambda e: None, lambda pcm, m: self.out_q.put_nowait(pcm))
        return self
    async def _agent_rx(self, track):
        async for ev in rtc.AudioStream(track, sample_rate=16000, num_channels=1):
            if self.rt: await self.rt.feed(np.frombuffer(ev.frame.data, np.int16).astype(np.float32) / 32768)
    async def _agent_tx(self):
        while True:
            pcm = await self.out_q.get(); i16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
            await self.a_src.capture_frame(rtc.AudioFrame(i16.tobytes(), 16000, 1, len(i16)))
    async def _client_rx(self, track):
        async for ev in rtc.AudioStream(track, sample_rate=16000, num_channels=1):
            x = np.frombuffer(ev.frame.data, np.int16).astype(np.float32) / 32768
            if np.sqrt((x ** 2).mean()) > 0.01:
                self.last_loud = time.monotonic()
                if self.armed and self.client_first is None: self.client_first = self.last_loud   # first audible frame at client
    async def client_send(self, frame_f32: np.ndarray):
        i16 = (np.clip(frame_f32, -1, 1) * 32767).astype(np.int16)
        await self.c_src.capture_frame(rtc.AudioFrame(i16.tobytes(), 16000, 1, len(i16)))
    async def stop(self):
        for t in self.tasks: t.cancel()
        for r in (getattr(self, "client", None), self.agent):
            if r: await r.disconnect()

class LKProxy:
    """Looks like a VoiceRuntime to the replay harness; feed() goes through the LiveKit client track; first-audio = client-side receipt."""
    def __init__(self, sess: LKSession): self.s = sess
    async def feed(self, frame): await self.s.client_send(frame)
    @property
    def _rt(self): return self.s.rt
    @property
    def interjections(self): return self._rt.interjections
    @property
    def agent_speaking(self): return self._rt.agent_speaking or not self.s.out_q.empty()
    @property
    def agent_task(self):
        class T:
            def done(_): return (self._rt.agent_task is None or self._rt.agent_task.done()) and (self.s.client_first is not None or not self._rt.responses)
        return T()
    @property
    def responses(self):
        out = []
        for r in self._rt.responses:
            r = {**r, "marks": dict(r["marks"])}
            if self.s.client_first: r["marks"]["AUDIO_FIRST_PLAYED_SINK"] = r["marks"].get("AUDIO_FIRST_PLAYED"); r["marks"]["AUDIO_FIRST_PLAYED"] = self.s.client_first
            out.append(r)
        return out

def build(opts: dict):
    """Returns (make_engine, make_runtime_factory(session), info). Benchmark driver is in benchmark/run_livekit.py."""
    make_engine, make_runtime, info = case_b.build(opts)
    return make_engine, make_runtime, {**info, "transport": "LiveKit WebRTC (self-hosted dev server, loopback)"}
