"""Subprocess worker: AVSpeechSynthesizer delivers buffers on the MAIN run loop, so it needs its own process.
stdin: JSON lines {"id":int,"text":str,"rate":float}.  stdout: frames [u32 id][u32 nbytes][float32 16k PCM]; nbytes==0 marks end."""
import sys, json, struct, threading, queue, time
import numpy as np
import AVFoundation as AV
from Foundation import NSRunLoop, NSDate
q = queue.Queue()
def reader():
    for line in sys.stdin: q.put(json.loads(line))
    q.put(None)
threading.Thread(target=reader, daemon=True).start()
syn = AV.AVSpeechSynthesizer.alloc().init()
import os
_ko = [v for v in AV.AVSpeechSynthesisVoice.speechVoices() if v.language().startswith("ko")]
_want = os.environ.get("AV_VOICE")                       # optional: exact voice identifier
voice = next((v for v in _ko if v.identifier() == _want), None) if _want else None
if voice is None: voice = max((v for v in _ko if v.name() == "Yuna"), key=lambda v: v.quality())     # best installed Yuna (premium > enhanced > compact)
sys.stderr.write(f"av_worker voice: {voice.identifier()} quality={voice.quality()}\n")
out = sys.stdout.buffer
def send(i, arr):
    b = arr.astype(np.float32).tobytes(); out.write(struct.pack("<II", i, len(b)) + b); out.flush()
send(0, np.zeros(0, np.float32))   # ready signal
cur = {"id": None, "last": 0.0}
def run(job):
    i = job["id"]; cur.update(id=i, last=time.monotonic(), done=False); carry = []
    u = AV.AVSpeechUtterance.speechUtteranceWithString_(job["text"]); u.setVoice_(voice); u.setRate_(job.get("rate", 0.5))
    def cb(buf):
        n = buf.frameLength()
        if n == 0:
            if not cur["done"]: cur["done"] = True; send(i, np.zeros(0, np.float32))
            return
        p = buf.floatChannelData()[0]; arr = np.array([p[k] for k in range(n)], np.float32)
        sr = buf.format().sampleRate(); carry.append(arr); cur["last"] = time.monotonic()
        if sum(len(c) for c in carry) >= 1024:        # ~46 ms chunks
            a = np.concatenate(carry); carry.clear()
            if sr != 16000: a = np.interp(np.linspace(0, len(a) - 1, int(len(a) * 16000 / sr)), np.arange(len(a)), a)
            send(i, a)
    syn.writeUtterance_toBufferCallback_(u, cb)
    while not cur["done"] and time.monotonic() - cur["last"] < 3:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.003))
    if carry:
        a = np.concatenate(carry)
        send(i, np.interp(np.linspace(0, len(a) - 1, int(len(a) * 16000 / 22050)), np.arange(len(a)), a))
    if not cur["done"]: send(i, np.zeros(0, np.float32))
while True:
    try: job = q.get_nowait()
    except queue.Empty:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.003)); continue
    if job is None: break
    run(job)
