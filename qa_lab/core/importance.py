"""Which questions matter most, decided from the materials themselves.

Materials: RFP / proposal / slides / speech transcript (one passage with "[종류: 파일이름]" header lines).
The CODE computes the score (tier first, then the evaluation weight); rules find the facts, so planning takes milliseconds and never
delays a question. A model is used only optionally, at upload time, to POINT AT sentences that state a differentiator (it picks
sentence numbers, it cannot invent text).

Tiers (lower = asked first):
  1 core item, value differs          4 other number differs
  2 core item missing / no basis      5 evaluation criterion missing from the talk
  3 core item explained -> go deeper  6 evaluation criterion mentioned without basis
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from ..contracts import Action, Question
from .interjection import _kw
from .quantities import numbers, parse, is_calendar_year

TIER_BASE = {1: 0.975, 2: 0.965, 3: 0.955, 4: 0.945, 5: 0.935, 6: 0.925}   # all below the user's own (0.99) and wrong-last-time (0.985) questions
WEIGHT_STEP = 0.00008        # 100 points add 0.008 < the 0.01 tier gap: a weight never lifts a question into a higher tier
MAX_PLAN = 10                # core tiers sort first, so core questions always get their places
MAX_PER_CATEGORY = 2
BANK_BASE_WITH_PLAN = 0.6    # ordinary bank questions come after the planned ones (the opening question stays 1.0)

DOC_TYPES = ("RFP", "제안서", "발표자료", "발표 전사")
_HEADER = re.compile(r"(?m)^\[(RFP|제안요청서|제안서|발표자료|발표 전사|문서)\s*:\s*([^\]\n]*)\][ \t]*$")

CATEGORIES = {   # order = how strongly a sentence is claimed by a category
    "예산": re.compile(r"예산|사업비|비용|금액|단가|인건비|투자|원가"),
    "필수 요건": re.compile(r"필수|반드시|의무|준수|인정하지 않|불가|해야 (한다|합니다|함)|이어야"),
    "목표·성과": re.compile(r"목표|성과|KPI|지표|달성|향상|절감|정확도|만족도|수료율|취업률"),
    "차별점": re.compile(r"차별|핵심 전략|강점|특장점|독자적|최초|경쟁력"),
    "수행체계": re.compile(r"추진 ?체계|수행 ?체계|조직|투입 ?인력|전담|PM|참여 ?기관|컨소시엄|역할 분담|책임자"),
    "범위·일정": re.compile(r"대상|인원|정원|교육생|기간|일정|착수|완료|차시|범위"),
}
_BASIS = re.compile(r"근거|산출|산정|계산|때문|바탕|기준으로|기반|위해|통해|따라|분석|조사|실적|경험")
_JOSA_END = re.compile(r"(으로|로|은|는|이|가|을|를|의|에|도)$")

@dataclass
class Material:
    type: str
    name: str
    text: str

@dataclass
class Fact:
    sent: str
    source: str                      # document type
    category: str | None = None      # core category, or None
    weight: int = 0                  # evaluation points of the matching criterion
    repeated: int = 1                # in how many document types the same point appears

@dataclass
class Finding:
    tier: int
    question: str
    reason: str                      # shown in the UI and the report
    ref: str                         # source sentence (grading uses it)
    category: str = ""
    weight: int = 0
    priority: float = 0.0
    topic: str = field(default="", repr=False)
    deeper: str = field(default="", repr=False)     # what to ask instead once the basis has been explained in an answer

# ---------- materials ----------
def _doc_type(label: str, name: str, body: str) -> str:
    if label in ("발표자료", "발표 전사", "제안서"): return label
    if label in ("RFP", "제안요청서"): return "RFP"
    n = name.lower()
    if re.search(r"전사|transcript|stt|script|대본", n) or n.endswith((".srt", ".vtt")): return "발표 전사"
    if re.search(r"rfp|제안 ?요청|과업 ?지시", n): return "RFP"
    if re.search(r"제안서|proposal", n): return "제안서"
    if re.search(r"발표|slides?|deck", n): return "발표자료"
    return "RFP" if re.search(r"제안 ?요청서|과업 ?지시서|평가 ?항목.{0,20}배점", body[:1500]) else "제안서"

def split_materials(text: str) -> list[Material]:
    """Without header lines the whole text is one proposal (the previous behaviour)."""
    heads = list(_HEADER.finditer(text or ""))
    if not heads: return [Material("제안서", "", (text or "").strip())] if (text or "").strip() else []
    out = []
    if text[:heads[0].start()].strip(): out.append(Material("제안서", "", text[:heads[0].start()].strip()))
    for i, m in enumerate(heads):
        body = text[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)].strip()
        if body: out.append(Material(_doc_type(m.group(1), m.group(2), body), m.group(2).strip(), body))
    return out

def reference_text(text: str) -> str:
    """The text questions and grading are based on: every material EXCEPT the speech transcript (speech recognition errors must not become 'facts').
    Text without headers is returned unchanged. With only a transcript, the transcript is all there is."""
    if not _HEADER.search(text or ""): return text or ""
    mats = split_materials(text)
    ref = [m for m in mats if m.type != "발표 전사"] or mats
    return "\n\n".join(m.text for m in ref)

def sentences(text: str) -> list[str]:
    """Lines AND sentences: slide bullets and table rows have no sentence ending and must not be lost."""
    out = []
    for line in (text or "").split("\n"):
        line = line.strip(" \t-•·*▶○●■□")
        if re.fullmatch(r"\[슬라이드 \d+\]|\[시트 [^\]]*\]", line): continue
        out += [p.strip() for p in re.split(r"(?<=[.?!。])\s+", line) if 4 <= len(p.strip()) <= 300]
    return out

# ---------- evaluation criteria ----------
def criteria(mats: list[Material]) -> list[tuple[str, int]]:
    """(criterion name, points) from RFP score tables: "사업 이해도 20점", "수행 능력 | 30". Only in documents that talk about evaluation points."""
    out: dict[str, int] = {}
    for m in mats:
        if m.type == "발표 전사" or not re.search(r"배점|평가 ?(항목|기준|지표)", m.text): continue
        for line in m.text.split("\n"):
            line = line.strip(" \t-•·*|")
            hit = re.fullmatch(r"(.{2,40}?)\s*[\|:：(]?\s*(\d{1,3})\s*점\s*\)?(?:\s*\|.*)?", line) or re.fullmatch(r"(.{2,40}?)\s*\|\s*(\d{1,3})", line)
            if not hit: continue
            name = re.sub(r"[\|:：()]+$", "", hit.group(1)).strip(" |"); pts = int(hit.group(2))
            if 0 < pts <= 100 and not re.search(r"합계|총점|소계|배점|만점", name) and re.search(r"[가-힣A-Za-z]{2}", name): out[name.split("|")[-1].strip()] = pts
    return list(out.items())

def _weight(sent: str, crit: list[tuple[str, int]]) -> int:
    k = _kw(sent); best = 0
    for name, pts in crit:
        nk = _kw(name)
        if nk and len(nk & k) / len(nk) >= 0.5: best = max(best, pts)
    return best

def _category(sent: str) -> str | None:
    for cat, rx in CATEGORIES.items():
        if rx.search(sent): return cat
    return "예산" if any(parse(q)[0] == "원" for q in numbers(sent)) else None

# ---------- core facts ----------
def core_facts(mats: list[Material], extra: dict[str, str] | None = None) -> list[Fact]:
    """The most informative sentences per core category from the reference documents. `extra` = {sentence: category} pointed at by the model."""
    crit = criteria(mats); extra = extra or {}; facts = []
    docs = {m.type: [_kw(s) for s in sentences(m.text)] for m in mats if m.type != "발표 전사"}
    for m in mats:
        if m.type == "발표 전사": continue
        for s in sentences(m.text):
            if re.search(r"\d\s*점\s*\)?$|배점", s): continue                                    # a score-table row is a criterion, not a fact
            cat = extra.get(s) or _category(s)
            if not cat or (cat not in ("차별점", "필수 요건") and not _quantities(s) and not CATEGORIES[cat].search(s)): continue
            k = _kw(s); rep = sum(any(k and len(k & o) >= max(2, len(k) // 2) for o in lst) for lst in docs.values())
            facts.append(Fact(s, m.type, cat, _weight(s, crit), max(rep, 1)))
    order = {"RFP": 0, "제안서": 1, "발표자료": 2}
    facts.sort(key=lambda f: (-bool(_quantities(f.sent)), -f.weight, -f.repeated, order.get(f.source, 3)))
    picked, count = [], {}
    for f in facts:
        k = _kw(f.sent)
        if count.get(f.category, 0) >= MAX_PER_CATEGORY or any(k and len(k & _kw(p.sent)) / len(k) >= 0.7 for p in picked): continue
        picked.append(f); count[f.category] = count.get(f.category, 0) + 1
    return picked

def _quantities(s: str) -> list[str]:
    return sorted((q for q in numbers(s) if not is_calendar_year(q) and parse(q)[1] is not None), key=s.find)

# ---------- comparison ----------
def _related(sent: str, pool: list[str], need: int = 2) -> list[str]:
    """Sentences about the same point: share 2+ key terms, or 1 key term and a number of the same unit."""
    k = _kw(sent); units = {parse(q)[0] for q in _quantities(sent)}
    scored = []
    for t in pool:
        n = len(k & _kw(t))
        if n >= need or (n >= 1 and units & {parse(q)[0] for q in _quantities(t)}): scored.append((n, t))
    return [t for _, t in sorted(scored, key=lambda x: -x[0])]

def _conflict(ref: str, others: list[str]) -> tuple[str, str, str] | None:
    """(reference value, other value, other sentence) when a related sentence gives a DIFFERENT value of the same unit and none gives the same one.
    Conservative: 2+ shared key terms are required, and a unit that appears twice in the reference sentence is skipped (ambiguous)."""
    kr = _kw(ref); rq = _quantities(ref)
    for q in rq:
        unit, val = parse(q)
        if sum(parse(x)[0] == unit for x in rq) > 1: continue
        near = [t for t in others if len(kr & _kw(t)) >= 2]
        vals = [(x, t) for t in near for x in _quantities(t) if parse(x)[0] == unit]
        if vals and all(parse(x)[1] != val for x, _ in vals):
            return q, vals[0][0], vals[0][1]
    return None

# ---------- wording ----------
def josa(word: str, pair: str) -> str:
    """Pick the right Korean particle: josa("120명", "으로/로") -> "으로"."""
    with_b, without_b = pair.split("/")
    chars = re.findall(r"[가-힣0-9%A-Za-z]", word)
    if not chars: return without_b
    c = chars[-1]
    if "가" <= c <= "힣": final = (ord(c) - 0xAC00) % 28
    elif c.isdigit(): final = {"0": 21, "1": 8, "3": 16, "6": 1, "7": 8, "8": 8}.get(c, 0)
    else: final = 0
    if with_b == "으로" and final == 8: return without_b                  # ㄹ takes "로"
    return with_b if final else without_b

def topic_of(sent: str, q: str = "") -> str:
    """A short name for what the sentence is about: the words right before the number ("시범 대상 인원은 120명" -> "시범 대상 인원")."""
    head = sent.split(q)[0] if q else sent
    head = re.split(r"[,.:：|(\[]", head.strip(" :：|-("))[-1] or re.split(r"[:：|]", head)[0]
    words = head.split()[-3:]
    if words: words[-1] = _JOSA_END.sub("", words[-1]) if len(words[-1]) > 2 else words[-1]
    return " ".join(words).strip()

def _clause(sent: str, n: int = 45) -> str:
    """The sentence quoted as written (cut at a word only when very long): a quotation stays grammatical, a trimmed paraphrase does not."""
    s = sent.strip().rstrip(".。")
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"

def _name(f: Fact, q: str) -> str:
    t = topic_of(f.sent, q) if q else ""
    return t if len(t) >= 2 else (f.category or "해당 항목")

def _basis_question(f: Fact) -> str:
    qs = _quantities(f.sent)
    if qs and f.category == "수행체계": return f"{_name(f, qs[0])} {qs[0]} 등 인력 구성의 근거는 무엇인가요?"
    if qs: return f"{_name(f, qs[0])} {qs[0]}의 산출 근거는 무엇인가요?"
    return f"'{_clause(f.sent)}'라는 내용의 근거는 무엇인가요?"

def _deeper_question(f: Fact) -> str:
    qs = _quantities(f.sent); q = qs[0] if qs else ""
    subj = f"{_name(f, q)} {q}" if q else f"'{_clause(f.sent)}'"
    last = q or _clause(f.sent)
    return {"예산": f"{subj}에서 가장 큰 변동 위험은 무엇인가요?",
            "목표·성과": f"{subj}{josa(last, '은/는')} 어떤 방법으로 측정하실 건가요?",
            "범위·일정": f"{subj}{josa(last, '을/를')} 지키기 어려워지면 어떻게 대응하실 건가요?",
            "수행체계": f"{subj} 부분에서 문제가 생기면 누가 대응하나요?",
            "차별점": f"{subj} 부분은 경쟁 제안과 무엇이 다른가요?",
            "필수 요건": f"{subj} 요건은 어떻게 지키실 건가요?"}.get(f.category or "", f"{subj} 부분은 어떻게 확인하실 건가요?")

def _missing_question(f: Fact) -> str:
    qs = _quantities(f.sent)
    subj = f"{_name(f, qs[0])} {qs[0]} 부분은" if qs else f"'{_clause(f.sent)}'라는 내용은"
    return f"{subj} 발표에서 말씀하지 않으셨는데, 설명해 주시겠어요?"

# ---------- plan ----------
def build_plan(text: str, extra: dict[str, str] | None = None, limit: int = MAX_PLAN) -> list[Finding]:
    """The ordered list of questions that matter most. Pure rules, milliseconds."""
    mats = split_materials(text)
    if not mats: return []
    crit = criteria(mats); talk = [s for m in mats if m.type == "발표 전사" for s in sentences(m.text)]
    by_type = {t: [s for m in mats if m.type == t for s in sentences(m.text)] for t in ("RFP", "제안서", "발표자료")}
    found: list[Finding] = []
    def add(tier, question, reason, f: Fact, topic=""):
        if f.weight: reason += f" · 평가 배점 {f.weight}점"
        found.append(Finding(tier, question, reason, f.sent[:300], f.category or "", f.weight, topic=topic or f.sent, deeper=_deeper_question(f) if tier == 2 else ""))

    cores = core_facts(mats, extra)
    for f in cores:
        label = f"핵심 항목({f.category})"
        c = _conflict(f.sent, talk) if talk else None
        if c:
            a, b, _ = c
            add(1, f"{f.source}에는 {_name(f, a)}{josa(_name(f, a), '이/가')} {a}인데, 발표에서는 {b}{josa(b, '으로/로')} 들었습니다. 어느 쪽이 맞습니까?", f"{label} · 자료와 발표의 값이 다름", f); continue
        d = next((x for other, pool in by_type.items() if other != f.source for x in [_conflict(f.sent, pool)] if x), None)
        if d:
            a, b, os_ = d; other = next(t for t, pool in by_type.items() if os_ in pool)
            add(1, f"{_name(f, a)}{josa(_name(f, a), '이/가')} {f.source}에는 {a}, {other}에는 {b}{josa(b, '으로/로')} 되어 있습니다. 어느 쪽이 맞습니까?", f"{label} · 자료끼리 값이 다름", f); continue
        if not talk:
            add(2, _basis_question(f), f"{label} · 발표 전사 없음", f); continue
        rel = _related(f.sent, talk)
        if not rel: add(2, _missing_question(f), f"{label} · 발표에서 누락", f)
        elif not any(_BASIS.search(t) for t in rel): add(2, _basis_question(f), f"{label} · 발표에서 근거가 부족함", f)
        else: add(3, _deeper_question(f), f"{label} · 설명함, 한 단계 깊게", f)

    core_sents = {f.sent for f in cores}
    if talk:                                                                      # other numbers that differ from the talk
        for t_, pool in by_type.items():
            for s in pool:
                if s in core_sents or not _quantities(s): continue
                c = _conflict(s, talk)
                if c:
                    a, b, _ = c; f = Fact(s, t_, None, _weight(s, crit)); nm = _name(f, a)
                    add(4, f"{t_}에는 {nm}{josa(nm, '이/가')} {a}인데, 발표에서는 {b}{josa(b, '으로/로')} 들었습니다. 어느 쪽이 맞습니까?", "일반 항목 · 자료와 발표의 값이 다름", f)
        for name, pts in crit:                                                    # evaluation criteria the talk skipped or only asserted
            nk = _kw(name); rel = [t for t in talk if nk and len(nk & _kw(t)) / len(nk) >= 0.5]
            f = Fact(f"평가 항목: {name} {pts}점", "RFP", None, pts)
            if not rel: add(5, f"평가 항목인 '{name}'에 대한 설명이 없었는데, 어떻게 준비하셨나요?", "평가 항목 · 발표에서 누락", f, name)
            elif not any(_BASIS.search(t) for t in rel): add(6, f"'{name}' 부분의 근거를 구체적으로 말씀해 주시겠어요?", "평가 항목 · 근거가 부족함", f, name)

    found.sort(key=lambda x: (x.tier, -x.weight))
    out, seen = [], []
    for x in found:
        if x.question in seen: continue
        seen.append(x.question); out.append(x)
        if len(out) >= limit: break
    for i, x in enumerate(out):
        x.priority = round(TIER_BASE[x.tier] + min(x.weight, 100) * WEIGHT_STEP - i * 0.000001, 6)
    return out

def planned(text: str, extra: dict[str, str] | None = None) -> list[Finding]:
    """The plan used by a session: only for labelled materials (uploaded files, or "[제안서: 이름]" headers). Plain pasted text keeps the previous behaviour."""
    return build_plan(text, extra) if _HEADER.search(text or "") else []

def add_plan(eng, plan: list[Finding]) -> int:
    """Put the planned questions into the live queue. No LLM, no waiting. Template questions look alike on purpose,
    so they bypass the wording-similarity de-duplication that applies to generated questions."""
    n = 0
    for x in plan:
        if x.question in eng.asked_texts or any(q.text == x.question for q in eng.queue.items): continue
        q = Question(x.question, Action.SWITCH, None, x.priority, "plan", x.ref)
        q.ref_text = x.ref; q.reason = x.reason; q.plan = x
        eng.queue.items.append(q); n += 1
    if n:
        eng.has_plan = True; eng.queue.max_pending += n                     # planned questions must not push bank questions out of the bounded queue
        for q in eng.queue.pending():                                       # bank questions already waiting (except the opening one) move behind the plan
            if q.source == "bank" and q.priority < 0.999: q.priority = bank_priority(round((1.0 - q.priority) / 0.02))
    return n

def revise_plan(eng, answer: str) -> int:
    """After an answer, re-check the planned questions still waiting (rules only, milliseconds, so it never delays the next question).
    If the answer already explains a point WITH its basis:  core item -> ask one step deeper instead of asking for the basis;
    evaluation criterion (not core) -> drop it. Value mismatches stay: they need an explicit confirmation."""
    said = sentences(answer); changed = 0
    if not said or not getattr(eng, "has_plan", False): return 0
    for q in eng.queue.pending():
        x = getattr(q, "plan", None)
        if x is None or x.tier not in (2, 5, 6): continue
        if x.tier == 2:
            rel = _related(x.ref, said)
            if _conflict(x.ref, said): continue
        else:
            nk = _kw(x.topic); rel = [t for t in said if nk and len(nk & _kw(t)) / len(nk) >= 0.5]
        if not any(_BASIS.search(t) for t in rel): continue
        changed += 1
        if x.tier == 2 and x.deeper and x.deeper != q.text:
            q.text = x.deeper; q.priority = round(TIER_BASE[3] + min(x.weight, 100) * WEIGHT_STEP, 6)
            q.reason = f"핵심 항목({x.category}) · 앞선 답변에서 설명함, 한 단계 깊게"
            q.plan = Finding(3, x.deeper, q.reason, x.ref, x.category, x.weight, topic=x.topic)
        else: q.status = "resolved"
    return changed

def bank_priority(i: int) -> float: return BANK_BASE_WITH_PLAN - 0.005 * i

# ---------- optional model help (upload time only) ----------
PICK_SYS = """당신은 제안 평가 보조자다. 번호가 붙은 문장 중에서 이 제안의 '차별점'이나 '핵심 전략'을 말하는 문장 번호만 고른다.
JSON만: {"items":[{"n":번호,"category":"차별점"}]}  규칙: 최대 3개. 해당 문장이 없으면 빈 배열. 문장을 새로 쓰지 않는다."""

async def model_picks(llm, text: str, max_sents: int = 60) -> dict[str, str]:
    """{sentence: category}. The model only chooses numbers of existing sentences, so it cannot add facts. {} on any failure."""
    sents = [s for m in split_materials(text) if m.type in ("RFP", "제안서", "발표자료") for s in sentences(m.text)]
    if not sents: return {}
    if len(sents) > max_sents: sents = [sents[int(i * len(sents) / max_sents)] for i in range(max_sents)]
    try:
        d = await llm.json(PICK_SYS, "\n".join(f"{i + 1}. {s[:160]}" for i, s in enumerate(sents)), 150)
        out = {}
        for it in (d.get("items") or [])[:3]:
            n = int(it.get("n", 0))
            if 1 <= n <= len(sents): out[sents[n - 1]] = "차별점"
        return out
    except Exception: return {}
