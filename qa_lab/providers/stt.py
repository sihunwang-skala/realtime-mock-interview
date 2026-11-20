from __future__ import annotations
import re, time, numpy as np
TERMS = "질문에 대한 답변입니다. 말 중에 RFP, API, AES, HSM, TPS 같은 용어가 나올 수 있습니다."      # a sentence, not a bare word list: with noise / very quiet speech Whisper reads a bare list back as the transcript (measured)
class FasterWhisperSTT:
    local = True
    def __init__(self, model="small", compute=None, prompt=TERMS, device="mps"):
        from faster_whisper import WhisperModel
        self.name = f"faster-whisper:{model}"; self.prompt = prompt
        try: self.m = WhisperModel(model, device=device, compute_type=compute or ("float16" if device == "mps" else "int8"), cpu_threads=8)
        except Exception: device = "cpu"; self.m = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=8)
        self.name += f"@{device}"
    def transcribe(self, audio: np.ndarray) -> str:
        segs, _ = self.m.transcribe(audio.astype(np.float32), language="ko", beam_size=1, condition_on_previous_text=False, initial_prompt=self.prompt,
                                    vad_filter=False, temperature=0.0, without_timestamps=True, no_speech_threshold=0.6, compression_ratio_threshold=2.0, log_prob_threshold=-1.0)
        return clean_transcript(strip_prompt_echo(" ".join(s.text.strip() for s in segs).strip(), self.prompt))
    def warm(self): self.transcribe(np.zeros(16000, np.float32))


def strip_prompt_echo(t: str, prompt: str | None) -> str:
    """Whisper sometimes reads its vocabulary hint back instead of the speech (noise / very quiet audio): drop any run of 3+ consecutive hint words."""
    if not prompt or not t: return t
    words = sorted({w for w in re.findall(r"[A-Za-z][A-Za-z0-9_\-]+", prompt)}, key=len, reverse=True)
    if len(words) < 3: return t
    alt = "|".join(re.escape(w) for w in words)
    t2 = re.sub(rf"(?<![A-Za-z0-9_])(?:(?:{alt})(?![A-Za-z0-9_])[\s,.]*){{3,}}", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t2).strip(" ,")


_HALLU = re.compile(r"(MBC|KBS|SBS|JTBC|YTN)\s*뉴스\s*[가-힣]{2,4}\s*(입니다|였습니다)\.?|[가-힣]{2,4}\s*기자(입니다|였습니다)\.?|시청해\s*주셔서\s*감사합니다\.?|구독과?\s*좋아요[^.]*\.?|다음\s*(영상|시간)에서\s*만나요\.?|자막\s*제공[^.]*\.?")
_SHORT_OK = {"네", "예", "응", "어", "넵"}                                       # one-syllable answers that are real answers
_HALLU_ALL = {"감사합니다", "제공기", "아멘"}
def strip_hallucination(t: str) -> str:
    """Known Whisper inventions on silence / noise (news sign-offs, 'thanks for watching', subtitle credits): removed; a transcript that is only such a phrase becomes empty."""
    t = re.sub(r"\s+", " ", _HALLU.sub(" ", t)).strip()
    toks = re.findall(r"[가-힣A-Za-z0-9]+", t)
    return "" if toks and all(w in _HALLU_ALL for w in toks) else t

def clean_transcript(t: str) -> str:
    """Drop Whisper hallucinations on silence/noise: empty, 1-2 characters, the same short token repeated many times, known sign-off phrases."""
    t = strip_hallucination(t)
    t = " ".join(w for w in t.split() if not re.search(r"(.)\1{7,}", w)).strip()   # runaway tokens such as "Hdcccccc..." are Whisper glitches, not speech
    toks = re.findall(r"[가-힣A-Za-z0-9]+", t)
    if (len(re.sub(r"\s", "", t)) < 2 and re.sub(r"[\s.,!?]", "", t) not in _SHORT_OK) or not toks: return ""
    if len(toks) >= 6 and len(set(toks)) <= max(2, len(toks) // 6): return ""
    if len(toks) >= 8:
        grams = [" ".join(toks[i:i + 2]) for i in range(len(toks) - 1)]
        if 1 - len(set(grams)) / len(grams) > 0.6: return ""
    return t


class OpenAISTT:
    """Cloud transcription (OpenAI). SYNTHETIC audio only: real recordings must not use this provider."""
    local = False
    def __init__(self, model="gpt-transcribe"):
        from openai import OpenAI
        self.name = f"openai:{model}"; self.model = model; self.c = OpenAI()
    def transcribe(self, audio: np.ndarray) -> str:
        import io, soundfile as sf
        buf = io.BytesIO(); sf.write(buf, audio.astype(np.float32), 16000, format="WAV", subtype="PCM_16")
        r = self.c.audio.transcriptions.create(model=self.model, file=("a.wav", buf.getvalue(), "audio/wav"), language="ko")
        return clean_transcript(r.text.strip())
    def warm(self): pass
