from __future__ import annotations
import numpy as np, torch
from silero_vad import load_silero_vad, VADIterator
_MODEL = None
class SileroVAD:
    """Streaming Silero VAD over 512-sample (32 ms) frames at 16 kHz. Returns 'start' / 'end' / None."""
    FRAME = 512
    def __init__(self, threshold=0.5, silence_ms=600, own_model=False):
        global _MODEL
        if own_model: model = load_silero_vad()          # the model is stateful: iterators running side by side need their own copy
        else:
            if _MODEL is None: _MODEL = load_silero_vad()
            model = _MODEL
        self.it = VADIterator(model, threshold=threshold, sampling_rate=16000, min_silence_duration_ms=silence_ms, speech_pad_ms=30)
        self.silence_ms = silence_ms; self.name = f"silero-vad:{silence_ms}ms"
    def feed(self, frame: np.ndarray):
        r = self.it(torch.from_numpy(frame.astype(np.float32)))
        return None if r is None else ("start" if "start" in r else "end")
    def reset(self): self.it.reset_states()
