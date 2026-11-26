"""Passage mode: the user supplies a text (e.g. about AI agents); the AI asks about it aloud, the user answers, follow-ups adapt.
The passage replaces the built-in RFP: its sentences become the 'requirements' used for coverage, contradiction and interjection checks.
Long passages are normalised (hard line breaks joined) and sampled evenly so the whole text is covered, not just the beginning."""
from __future__ import annotations
import asyncio, hashlib, re
from ..contracts import Question, Requirement, Action
from .answer_analyzer import AnswerAnalyzer
from .coverage import Coverage
from .interjection import make_policy
from .planner import PASSAGE_GEN_SYS, clean_q, ko_ok, bad_question

MAX_CHARS = 150000      # was 30000; the text is only split into sentences and sampled, never sent to the model as a whole
BANK: dict = {}                    # key -> {"qs":[{"q":..,"a":..}], "done":bool}   questions (+ model answers) prepared before the session starts
PRIORITY = {"busy": False}         # set while a live answer is being processed: the bank builder pauses so it never slows the live path

def bank_key(text: str, llm: str) -> str: return hashlib.sha1((llm + "|" + text[:MAX_CHARS]).encode()).hexdigest()[:12]
async def _idle():
    while PRIORITY["busy"]: await asyncio.sleep(0.05)
async def _interruptible(factory):
    """Run ONE background LLM step, but abort it the instant a live answer needs the model (PRIORITY["busy"]), then retry once the model is free.
    The local model serves one request at a time, so without this a live follow-up waits for the whole background step (measured: up to 5 s)."""
    while True:
        await _idle()
        task = asyncio.ensure_future(factory())
        try:
            while not task.done():
                if PRIORITY["busy"]: break
                await asyncio.sleep(0.02)
            if task.done(): return task.result()
        finally:
            if not task.done():
                task.cancel()
                try: await task
                except BaseException: pass
PSYS = """당신은 발표 평가위원이다. 아래 [지문 일부]를 읽고 발표자에게 구두로 할 질문 후보를 만든다.
JSON만: {"questions":[{"text":"한국어 1문장, 말로 읽기 쉽게 50자 안팎","priority":0~1}]}
규칙: 질문 하나에 한 가지만 묻는다(두 주제를 묶지 않는다). 지문의 내용에 근거할 것. 서로 다른 관점(개념, 근거·수치, 한계·위험, 적용 방법)으로 2개. 지문에 없는 사실을 단정하지 말 것."""

FU_SYS = """당신은 까다롭지만 공정한 면접관이다. [참고 지문]만 근거로 [질문]의 후속 질문 두 개를 만든다.
JSON만: {"deeper":"답변이 맞았을 때 한 단계 더 깊이 묻는 질문","gap":"답변이 일부만 맞았을 때 빠진 부분을 확인하는 질문"}
규칙: 각각 한국어 존댓말 의문문 한 문장(반드시 '~인가요?', '~설명해 주시겠어요?'처럼 물음표로 끝남), 60자 이내. [질문]과 같은 말을 반복하지 말고 지문의 개념어만 쓴다.
사람의 답변은 아직 모른다: '언급하지 않았습니다', '알고 싶습니다'처럼 답변을 평가하거나 평서문으로 쓰지 말고 질문만 쓴다.
예시: {"deeper":"그 기준을 넘으면 시스템은 어떻게 동작하나요?","gap":"그 기준을 정한 근거나 예외는 무엇인가요?"}"""
def parse_followups(d) -> dict:
    """Validated follow-up candidates {'deeper','gap'} from the model's JSON (Korean question ending in '?', one sentence each); {} if nothing usable."""
    out = {}
    for k in ("deeper", "gap"):
        t = clean_q(str((d or {}).get(k, ""))) if isinstance(d, dict) else ""
        m = re.findall(r"[^.!?]*\?", t)                                    # only the question sentence (drop any statement the model put in front)
        if not m: continue
        t = m[0].strip()
        if 8 <= len(t) <= 110 and ko_ok(t) and not bad_question(t) and not re.search(r"(않았|알고 ?싶|없었|부족)", t): out[k] = t
    return out

def normalize(text: str) -> str:
    t = text.replace("\r", "")
    t = re.sub(r"(?m)^([ \t]*\|.*)$", r"\1\n", t)                                                              # every table row becomes its own paragraph
    t = re.sub(r"(?m)^[ \t]*\|?[ \t\-:|]+\|?[ \t]*$", "", t)                                                       # table separator / empty rows (| --- | --- |, |  |  |)
    t = re.sub(r"\*\*|__|`+", "", t); t = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]*", "", t); t = re.sub(r"[ \t]*\|[ \t]*", " · ", t); t = re.sub(r"^[ \t]*·[ \t]*|[ \t]*·[ \t]*$", "", t, flags=re.M); t = re.sub(r"(·\s*){2,}", "· ", t)      # strip markdown (bold, headings, code ticks, table pipes)
    t = re.sub(r"\n{2,}", "\u0001", t); t = re.sub(r"\s*\n\s*", " ", t); t = t.replace("\u0001", "\n")   # keep blank-line paragraph breaks, join hard wraps
    return re.sub(r"[ \t]{2,}", " ", t).strip()

_JUNK = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}|최근 수정|마지막 수정|더 보기|\[편집\]|편집\]|바로가기|로그인|이용약관|저작권|목차|분류[가-힣A-Za-z]|[♬♪★☆▶▷◆♥]|ㅣ|^\s*#|\s#\s|SBS ENTER|유튜브 채널|다음과 같습니다\.?\s*$|전체 자막")
_TS = re.compile(r"\(\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[-~–]\s*\d{1,2}:\d{2}(?::\d{2})?)?\s*\)|\[\s*(?:음악|박수|웃음|Music|Applause|Laughter)\s*\]|\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]|(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?(?![\d:])")
def is_prose(s: str) -> bool:
    """A sentence worth asking about: not page furniture (timestamps, 'more', category link blobs, tables) and not an endless unspaced run."""
    s = s.strip()
    if not (12 <= len(s) <= 600) or _JUNK.search(s): return False
    toks = s.split()
    if len(s) > 40 and (len(toks) <= 3 or len(s) / len(toks) > 14): return False                  # "분류식케이한국의남성래퍼..." : words glued together
    if sum(c.isdigit() for c in s) / len(s) > 0.35: return False
    return bool(re.search(r"(다|요|음|임|함|까|죠|[.!?])[.!?]?(\s*\[\d+\])*[.!?]?\s*$", s))          # ends like a sentence (also "…했다.[3]")

def _drop_line(l: str) -> bool:
    """Whole pasted-page lines that are furniture, removed BEFORE hard-wrapped lines are joined (otherwise they glue onto the next real sentence)."""
    l = l.strip()
    if not l: return False
    toks = l.split()
    if _JUNK.search(l) or any(len(t) >= 20 for t in toks): return True                             # timestamp / 'more' / category blob / words glued together
    ends = bool(re.search(r"(다|요|음|임|함|까|죠|[.!?])[.!?]?(\s*\[\d+\])*[.!?]?\s*$", l))
    years = sum(bool(re.fullmatch(r"(19|20)\d{2}년?|\d+", t)) for t in toks)
    return (not ends) and len(l) > 30 and years / len(toks) >= 0.1                                  # table row without a sentence ending

_GENERIC_Q = ("무엇", "어떻게", "설명", "이유", "방법", "방식", "어떤", "있나요", "하나요", "되나요", "인가요", "주시겠", "경우", "대해", "대한", "관련", "가지", "때문", "생각", "어디", "누구", "언제", "얼마")
def grounded(q: str, sent: str, soft: bool = False) -> bool:
    """The question must be answerable from its source sentence: its content words appear in the sentence, a 'why' question needs a reason in the sentence, a 'strategy / method' question needs that word in the sentence."""
    toks = [w for w in re.findall(r"[가-힣A-Za-z0-9]{2,}", q) if not any(g in w for g in _GENERIC_Q)]
    if toks and sum((w[:3] if len(w) >= 3 else w) in sent for w in toks) / len(toks) < (0.35 if soft else 0.6): return False
    if re.search(r"이유|왜", q) and not re.search(r"때문|위해|이유|므로|해서|덕분|위한|따라|라서|통해", sent): return False
    return all(k in sent for k in ("전략", "방법", "방식", "절차", "목적", "효과") if k in q)

def split_passage(text: str, max_n: int = 14) -> list[str]:
    text = _TS.sub("", text.replace("\r", ""))                                          # caption timestamps "(0:10-0:18)" and sound tags "[음악]" are not part of what was said
    _END = re.compile(r"(다|요|음|임|함|까|죠|[.!?])[.!?]?(\s*\[\d+\])*[.!?]?\s*$")
    text = "\n".join(l + ("\n" if _END.search(l.strip()) else "") for l in text.split("\n") if not _drop_line(l))     # a line that ends like a sentence is one sentence (captions have no full stops; otherwise all lines glue into one block)
    parts = [p.strip() for p in re.split(r"(?<=[.!?。])\s+|\n+", normalize(text)) if is_prose(p)]
    if len(parts) <= max_n: return parts
    step = len(parts) / max_n
    return [parts[int(i * step)] for i in range(max_n)]            # even sample across the whole passage

def chunks(text: str, size: int = 1600, max_n: int = 4) -> list[str]:
    t = normalize(text)
    if len(t) <= size: return [t]
    n = min(max_n, -(-len(t) // size)); span = len(t) // n
    return [t[i * span:(i + 1) * span][:size * 2] for i in range(n)]   # n equal slices covering start..end

def head_of(text: str, n: int = 700) -> str:
    """First few lines of the passage (cut at a sentence end) - enough to ask the first question without reading everything."""
    sents = split_passage(text, 1500)                                    # prose only: pasted pages start with an infobox / category blob that is not worth a first question
    t = ""
    for s_ in sents:
        if len(t) + len(s_) > n and t: break
        t += s_ + " "
    if t.strip(): return t.strip()
    t = normalize(text)[:n]
    m = max(t.rfind(". "), t.rfind("다. "), t.rfind("? "), t.rfind("\n"))
    return t[:m + 1] if m > n // 3 else t

async def first_question_text(eng, text: str) -> str:
    head = head_of(text)
    pr = f"[지문 앞부분]\n{head}\n\n이 앞부분의 핵심 개념을 확인하는 첫 질문 한 문장만 출력한다.\n"
    for _ in range(2):
        buf = ""
        try:
            async for tok in eng.fast.stream(PASSAGE_GEN_SYS, pr, 90):
                buf += tok
                if "?" in buf or "？" in buf: break
        except Exception: buf = ""
        q = clean_q(buf)
        if "?" in q: q = q[:q.index("?") + 1]
        if 8 <= len(q) <= 120 and ko_ok(q) and not _general_q(q): return q
    return evidence_fallback(head)

_EVIDENCE_TMPL = "{t}의 근거나 수치를 말씀해 주시겠어요?"
_EVIDENCE_GENERAL = "제안서의 수치나 기간 중 하나를 골라 그 근거를 말씀해 주시겠어요?"
def _general_q(q: str) -> bool:
    """A question that could be asked about any proposal ('핵심 내용을 본인의 말로 설명해 주시겠어요?'): not allowed, the coaching must be about the passage's own evidence."""
    return bool(re.search(r"핵심 내용|본인의 말로|이 글의|이 부분의", q))
def evidence_fallback(text: str) -> str:
    """Grounded fallback: names one concrete term from the passage and asks for its evidence or numbers. Never a generic question."""
    cands = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[가-힣]{3,}", text) if not any(g in t for g in _GENERIC_Q)]
    if not cands: return _EVIDENCE_GENERAL
    t = re.sub(r"(으로만|로만|에서|으로|에게|에|의|은|는|이|가|을|를|과|와|도)$", "", max(cands, key=len))   # "운영방안으로만" -> "운영방안"
    return _EVIDENCE_TMPL.format(t=t) if len(t) >= 2 else _EVIDENCE_GENERAL

async def apply_passage(eng, text: str) -> int:
    """Re-point an existing QAEngine at the passage and make the FIRST question from just the opening lines (about 1 s).
    The remaining questions are prepared in the background while the person is answering (QAEngine.start_prefetch)."""
    text = text[:MAX_CHARS]
    reqs = [Requirement(f"P{i+1}", s[:300], 2) for i, s in enumerate(split_passage(text))]
    all_s = split_passage(text, 1500)                                            # EVERY sentence is available for finding a follow-up's source / contradictions / next topic (eng.reqs stays a small sample for coverage and prompts)
    eng.all_reqs = [Requirement(f"G{i+1}", s[:300], 2) for i, s in enumerate(all_s)]; eng._dfc = {}
    eng.reqs = reqs; eng.cov = Coverage(reqs); eng.analyzer = AnswerAnalyzer(eng.fast, reqs)
    eng.policy = make_policy(eng.interjection_name, eng.fast, eng.all_reqs); eng.rfp_title = "사용자 지문"; eng.gen_sys = PASSAGE_GEN_SYS
    eng.setup_prefetch(chunks(text)); eng._bank = BANK.get(bank_key(text, getattr(eng.fast, "name", ""))); eng._bank_i = 0
    if eng._bank is not None: eng._bank["asked"] = eng.asked_texts                # the bank checks new questions against everything asked so far (same list object)
    eng.sync_bank()
    if not eng.queue.pending():                                                  # no bank yet: make just the first question from the opening lines
        PRIORITY["busy"] = True                                                  # the background bank steps aside so the first question is not stuck behind it
        try: q0 = await first_question_text(eng, text)
        finally: PRIORITY["busy"] = False
        eng.queue.add(Question(q0, Action.DEEPEN, None, 1.0, "passage", ""))
    return 1

def stt_terms(text: str, k: int = 6, max_chars: int = 90) -> str:
    """Whisper initial prompt built from the passage's own terms (English / snake_case / CamelCase words) so they are spelled right.
    Written as a sentence: a bare word list gets read back as the transcript when the audio is noise or very quiet (measured), a sentence does not."""
    from collections import Counter
    c = Counter(w.strip("._-") for w in re.findall(r"[A-Za-z][A-Za-z0-9_\-.]{2,}", text))
    out, n = [], 0
    for w, _ in c.most_common(k * 2):
        if n + len(w) + 2 > max_chars or len(out) >= k: break
        out.append(w); n += len(w) + 2
    return f"질문에 대한 답변입니다. 말 중에 {', '.join(out)} 같은 영어 용어가 나올 수 있습니다." if out else ""

async def _stream_one(llm, system: str, user: str, n: int = 90) -> str:
    buf = ""
    async for tok in llm.stream(system, user, n):
        buf += tok
        if "?" in buf or "？" in buf: break
    q = clean_q(buf)
    m = re.findall(r"[^.!?]*[?？]", q)                                       # the model often copies a passage sentence first: keep only the last question sentence
    return m[-1].strip() if m else q

BATCH = 3          # questions made per round (about 0.5 s each with the local model); measured: one question is used per ~35 s, so a round of 3 finishes ~20x faster than it is needed
REFILL_AT = 1      # the next round starts when 1 or fewer unasked questions are left

async def build_bank(text: str, llm_spec: str, max_q: int = BATCH):
    """Prepare the first BATCH questions in the background as soon as the passage is entered (no model answers, no follow-ups: those cost heat).
    More rounds are made by extend_bank() when the session is running low. Pauses whenever a live answer is being processed."""
    from .llm import make_llm
    key = bank_key(text, llm_spec)
    b = BANK[key] = {"qs": [], "done": False, "chars": len(text), "extending": False, "exhausted": False, "ci": 0, "si": 0, "head": False, "rounds": 0}
    try:
        b["llm"] = make_llm(llm_spec); b["text"] = text
        await _fill(b, BATCH)
    except Exception: pass
    finally: b["done"] = True

async def extend_bank(b: dict, n: int = BATCH):
    """Another round of n questions for a running session (called when 2 or fewer unasked questions remain)."""
    if b.get("extending") or b.get("exhausted") or not b.get("llm"): return
    b["extending"] = True
    try: await _fill(b, n)
    except Exception: pass
    finally: b["extending"] = False

def _kw_dup(q: str, others: list[str], th: float = 0.6) -> bool:
    """Same subject again even if worded differently: most of the smaller question's key terms already appear in an earlier one."""
    from .interjection import _kw
    k = _kw(q)
    return bool(k) and any(_kw(o) and len(k & _kw(o)) / max(min(len(k), len(_kw(o))), 1) >= th for o in others)

def _pick_pos(b: dict, n_s: int, region: int, n_regions: int = 3):
    """A sentence from the given third of the passage (early / middle / late), as far as possible from every sentence already asked about or tried."""
    lo, hi = (region * n_s) // n_regions, ((region + 1) * n_s) // n_regions
    tried = b.setdefault("tried", set()); used = b.setdefault("used_pos", [])
    cand = [i for i in range(lo, hi) if i not in tried]
    if not cand: return None
    ref = list(used) + list(tried)
    return max(cand, key=lambda i: min((abs(i - u) for u in ref), default=n_s))

def _spread_order(n: int) -> list[int]:
    """Sentence order that jumps around the passage (golden-ratio stride), so every round covers a different area instead of reading front to back."""
    out, seen, x = [], set(), 0.0
    for _ in range(n * 3):
        i = int(x * n) % n
        if i not in seen: seen.add(i); out.append(i)
        x = (x + 0.6180339887) % 1.0
    return out + [i for i in range(n) if i not in seen]

async def _fill(b: dict, n: int):
    """One round: the first round starts with a question from the opening lines, then one cheap per-sentence call per question (0.3-0.7 s each)."""
    from .planner import is_repeat, one_thing
    llm, text = b["llm"], b["text"]; target = len(b["qs"]) + n; before = len(b["qs"]); b["rounds"] += 1
    sents = [x for x in split_passage(text, 120) if len(x) >= 15]
    if "order" not in b: b["order"] = _spread_order(len(sents))
    def add(q, ref="", src=-1):
        q = clean_q(q)
        old = [x["q"] for x in b["qs"]] + list(b.get("asked", []))                       # never overlaps what was already asked (follow-ups and interjections included)
        if not (8 <= len(q) <= 120 and ko_ok(q) and not bad_question(q)) or is_repeat(q, old, th=0.5) or _kw_dup(q, old): return False
        item = {"q": q, "a": None, "ref": ref[:300], "src": src}
        if len(b["qs"]) < target and (not ref or grounded(q, ref)) and one_thing(q): b["qs"].append(item); return True
        if (not ref or grounded(q, ref, soft=True)) and one_thing(q): b.setdefault("soft", []).append(item)          # fails only the strict checks: kept as a reserve so the bank cannot run dry
        return False
    if not b["head"]:
        b["head"] = True; await _idle()
        add(await _interruptible(lambda: _stream_one(llm, PASSAGE_GEN_SYS, f"[지문 앞부분]\n{head_of(text)}\n\n이 앞부분의 핵심 개념을 확인하는 첫 질문 한 문장만 출력한다.\n")), head_of(text), 0)
    calls = 0; got = 0; used_pos = b.setdefault("used_pos", []); tried = b.setdefault("tried", set()); regions = [(b["rounds"] + i) % 3 for i in range(3)]   # every accepted question of a round comes from a different third of the passage, and the order rotates between rounds
    while len(b["qs"]) < target and calls < 4 * n:      # at most 4 model calls per wanted question (the small model passes ~1 in 3)
        pos = _pick_pos(b, len(sents), regions[got % 3])
        if pos is None: pos = next((p for r in regions for p in [_pick_pos(b, len(sents), r)] if p is not None), None)       # that third is used up: any other
        if pos is None: break
        calls += 1; tried.add(pos); b["si"] = len(tried); sent = sents[pos]; await _idle()
        try:
            if add(await _interruptible(lambda: _stream_one(llm, PASSAGE_GEN_SYS, f"Action: SWITCH\n새 영역(이 문장의 내용만으로 질문한다): {sent}\n질문 문장 한 개만 출력한다.\n")), sent, pos): got += 1; used_pos.append(pos)
        except Exception: pass
    old_ = lambda: [x["q"] for x in b["qs"]] + list(b.get("asked", []))
    while len(b["qs"]) < target and b.get("soft"):                                        # not enough strict questions: use the reserve (still from a passage sentence, lighter checks)
        it = b["soft"].pop(0)
        if not is_repeat(it["q"], old_(), th=0.5) and not _kw_dup(it["q"], old_()): b["qs"].append(it); used_pos.append(it.get("src", -1))
    if len(b["qs"]) == before and len(tried) >= len(sents): b["exhausted"] = True
