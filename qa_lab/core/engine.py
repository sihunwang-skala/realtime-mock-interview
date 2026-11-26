"""QAEngine: Phase A (presentation ingest, candidate queue) + Phase B (adaptive Q&A). Transport-agnostic."""
from __future__ import annotations
import asyncio, re, time
from ..contracts import Action, Assessment, Question, Requirement
from .answer_analyzer import AnswerAnalyzer
from .coverage import Coverage
from .interjection import make_policy, Decision
from .llm import LLM
from .planner import bad_question, decide_action, gen_prompt, fu_prompt, interject_prompt, GEN_SYS, is_repeat, is_repeat_request, clean_q, ko_ok, one_thing
from .interjection import numeric_conflict, numbers, _unit, _num, _kw
from .question_queue import QuestionQueue
from . import clarify

SEG_SYS = """RFP 평가 보조기. 발표 구간을 보고 JSON만: {"covered":[{"req":"id","score":0~1,"quote":"발표 인용"}],"questions":[{"text":"평가위원이 할 꼬리질문(한국어 1문장)","req":"id","priority":0~1,"evidence":"근거"}]}
발표가 요구사항을 충분히 설명했으면 covered에 넣고 질문은 만들지 않는다. 수치 근거가 없거나 RFP와 다르거나 빠진 부분만 질문한다. 질문은 최대 2개."""

KW_NO = re.compile(r"(잘 ?모르|모르겠|모릅니다|모르는|모르[ㅁ-ㅣ가-힣]|몰라|몰라요|모름|기억이 ?(안|잘)|생각이 ?안|떠오르지 ?않)")   # clear "I don't know" phrases
KW_NO_WEAK = re.compile(r"(패스|다음 ?질문|넘어가|죄송)")                                                                  # only counts when that is ALL the person said
JUDGE_SYS = """당신은 면접 채점관이다. [참고 지문]을 정답 기준으로 [질문]에 대한 [답변]을 판정한다.
JSON만: {"verdict":"맞음|부분 맞음|틀림","why":"한국어 한 문장(틀리거나 빠진 부분이 있으면 맞는 내용이 무엇인지)"}
규칙: 지문과 같은 뜻이면 맞음, 일부만 맞거나 빠진 게 있으면 부분 맞음, 지문과 다르거나 엉뚱하면 틀림. 한국어로만 쓴다."""

IJ_TEMPLATES = {                    # fixed interjection / challenge sentences: no LLM call, and their voice is synthesised ahead of time (instant).
    # Shape (polite interruption): courtesy opener -> short reason -> hand the floor back. Each sentence is its own voice chunk.
    "repetition": ["말씀 중에 잠시만요. 같은 설명이 반복되는 것 같아요. 핵심만 한 문장으로 말씀해 주시겠어요?", "말씀 중에 잠시만요. 방금과 같은 이야기인 것 같아요. 새로 덧붙일 내용이 무엇인가요?"],
    "off_topic": ["말씀 중에 잠시만요. 지금 말씀이 질문과 조금 다른 것 같아요. 질문에 맞춰 다시 말씀해 주시겠어요?", "말씀 중에 잠시만요. 질문과 거리가 있는 것 같아요. 질문에 대한 답부터 다시 말씀해 주시겠어요?"],
    "overlong": ["말씀 중에 잠시만요. 시간이 길어졌어요. 지금까지 내용의 핵심만 정리해 주시겠어요?", "말씀 중에 잠시만요. 요점부터 짧게 다시 말씀해 주시겠어요?"],
    "unsupported": ["말씀 중에 잠시만요. 방금 말씀하신 부분의 근거가 무엇인가요?", "말씀 중에 잠시만요. 그 주장을 뒷받침하는 내용이 지문에 있나요?"],
    "contradiction_prior": ["말씀 중에 잠시만요. 앞서 하신 말씀과 지금 말씀이 다른 것 같아요. 어느 쪽이 맞나요?", "말씀 중에 잠시만요. 앞의 답변과 숫자가 달라진 것 같아요. 정확한 값이 무엇인가요?"],
    "contradiction_rfp": ["말씀 중에 잠시만요. 방금 말씀하신 수치가 지문의 내용과 다른 것 같아요. 근거가 무엇인가요?", "말씀 중에 잠시만요. 방금 수치가 지문과 맞는지 다시 확인해 주시겠어요?"],
}

# Short spoken bridge before a follow-up: tells the listener it is a follow-up to what they just said (signposting). Written by us; no Korean source found, English guides recommend the same idea.
FU_ACK = "답변 잘 들었습니다."          # first short sentence of every follow-up bridge (voiced once, cached)
FU_LEAD_TERM = {"더 자세히": "{t} 부분에 대해 조금 더 여쭤볼게요.", "근거": "{t} 부분의 근거를 여쭤볼게요.", "검증": "{t} 부분은 어떻게 확인하는지 여쭤볼게요.",
                "순서": "순서와 관련해서 여쭤볼게요.", "더 깊이": "{t} 부분을 조금 더 깊이 여쭤볼게요."}     # names the part of the answer that is being picked up
FU_LEAD = {k: f"{FU_ACK} " + (v.replace("{t} 부분", "그 부분").replace("{t} ", "그 ")) for k, v in FU_LEAD_TERM.items()}      # same without a term (when no clean term is available)
FU_LEAD_CANNED = f"{FU_ACK} 관련해서 하나 더 여쭤볼게요."
FU_LEAD_CHAIN = "좋습니다. 이어서 하나만 더 여쭤볼게요."

MAX_FU = 3          # most follow-ups in a row on one question (sources: 1-3 typical; no research standard found)

class QAEngine:
    def __init__(self, fast: LLM, reqs: list[Requirement], deep: LLM | None = None, strategy="hybrid", interjection="hybrid", rfp_title="제안요청서"):
        self.fast = fast; self.deep = deep or fast; self.reqs = reqs; self.strategy = strategy
        self.queue = QuestionQueue(); self.cov = Coverage(reqs); self.analyzer = AnswerAnalyzer(fast, reqs)
        self.policy = make_policy(interjection, fast, reqs); self.interjection_name = interjection
        self.transcript: list[dict] = []; self.turns: list[dict] = []; self.answers: list[str] = []
        self.cur: Question | None = None; self.followups = 0; self.asked_texts: list[str] = []
        self.heard_chars = 0; self.spoken_full = ""; self.t0 = time.monotonic(); self.rfp_title = rfp_title

    # ---------- Phase A ----------
    async def ingest_segment(self, text: str, ts: float | None = None) -> dict:
        ts = ts if ts is not None else time.monotonic() - self.t0
        self.transcript.append({"ts": ts, "text": text})
        reqs = "\n".join(f"{r.id}: {r.text}" for r in self.reqs)
        gaps = ", ".join(self.cov.gaps())
        d = await self.deep.json(SEG_SYS, f"[요구사항]\n{reqs}\n[아직 미충족]\n{gaps}\n[발표 구간]\n{text}", 400)
        added, resolved = [], []
        for c in d.get("covered", []) or []:
            rid = str(c.get("req", "")); 
            try: sc = float(c.get("score", 0))
            except Exception: sc = 0
            self.cov.update(rid, sc, str(c.get("quote", ""))[:120], ts)
            if sc >= 0.7 and self.queue.resolve_req(rid): resolved.append(rid)
        if self.strategy != "direct":
            for q in d.get("questions", []) or []:
                t = str(q.get("text", "")).strip()
                if len(t) < 6: continue
                try: pr = float(q.get("priority", 0.5))
                except Exception: pr = 0.5
                rid = q.get("req") if q.get("req") in self.cov.reqs else None
                if rid and self.cov.score[rid] >= 0.7: continue
                got = self.queue.add(Question(t, Action.DEEPEN, rid, pr, "presentation", str(q.get("evidence", ""))))
                if got: added.append(got.id)
        return {"added": added, "resolved": resolved, "pending": len(self.queue.pending())}

    # ---------- Phase B ----------
    # ---------- background question preparation (runs while the person is answering) ----------
    def setup_prefetch(self, chunks: list[str]):
        self._chunks = list(chunks); self.prefetch_task = None
    def start_prefetch(self):
        t = getattr(self, "prefetch_task", None)
        if (t and not t.done()) or not getattr(self, "keep_going", False): return
        if getattr(self, "_bank", None): return                     # the question bank (3 at a time, filtered) is the only source when a passage is used: this old path made extra LLM calls and unfiltered questions
        self.prefetch_task = asyncio.create_task(self._prefetch())
    def stop_prefetch(self):
        for name in ("prefetch_task", "answer_task", "fu_task", "spec_task"):
            t = getattr(self, name, None)
            if t and not t.done(): t.cancel()
    # model answer for the question that was just asked: written in the background while the person answers, shown when we move on.
    # ---------- follow-ups written in advance (independent of the answer, so they cost no time when the answer ends) ----------
    def start_followup_prep(self, q):
        if q is None or not getattr(self, "keep_going", False) or self.followups_for(q): return
        t = getattr(self, "fu_task", None)
        if t and not t.done(): t.cancel()
        self.fu_task = asyncio.create_task(self._prep_followups(q))
    async def _prep_followups(self, q):
        from .passage import FU_SYS, parse_followups, PRIORITY
        try:
            while self.refill_wanted() or PRIORITY["busy"]: await asyncio.sleep(0.1)        # priority: 1) keep 3 questions ready, 2) this question's follow-up candidates, 3) follow-up from the live answer
            fu = parse_followups(await self.fast.json(FU_SYS, f"[참고 지문]\n{self.refs_for(q.text, 3)}\n[질문]\n{q.text}", 150))
            if fu: q.fu = fu
        except asyncio.CancelledError: raise
        except Exception: pass
    def refill_wanted(self) -> bool:
        """The question bank is running low (0-1 unasked left) or is being refilled right now: that work goes before any follow-up preparation."""
        from .passage import REFILL_AT
        b = getattr(self, "_bank", None)
        if not b or b.get("exhausted") or not b.get("llm"): return False
        return bool(b.get("extending") or (b.get("done") and len(self.queue.pending()) <= REFILL_AT))
    @staticmethod
    def followups_for(q) -> dict:
        return (getattr(q, "fu", None) or (getattr(q, "bank", None) or {}).get("fu") or {}) if q is not None else {}
    def canned_followup(self, verdict: str = "") -> str | None:
        """Prepared follow-up for the current question: 'gap' (what is missing) after a partly right answer, else 'deeper'. None if nothing usable is ready."""
        fu = self.followups_for(self.cur)
        for t in ([fu.get("gap"), fu.get("deeper")] if verdict.startswith("부분") else [fu.get("deeper"), fu.get("gap")]):
            if t and not is_repeat(t, self.asked_texts, th=0.5): return t
        return None
    def canned_texts(self) -> list[str]:
        """Sentences worth voicing ahead of time, most urgent first: this question's follow-ups, the next question, interjection templates, then the rest of the queue."""
        say = lambda t: f"{FU_LEAD_CANNED} {t}"
        self.sync_bank(); pend = self.queue.pending(); out = [FU_LEAD_CANNED, FU_LEAD_CHAIN, FU_ACK, *FU_LEAD.values()]            # the short bridges first: every follow-up needs one
        out += [say(t) for t in self.followups_for(self.cur).values() if t]
        if pend: out += [pend[0].text, *[say(t) for t in self.followups_for(pend[0]).values() if t]]
        for v in IJ_TEMPLATES.values(): out += v
        for q in pend[1:]: out += [q.text, *[say(t) for t in self.followups_for(q).values() if t]]
        seen, res = set(), []
        for t in out:
            if t and t not in seen: seen.add(t); res.append(t)
        return res
    # ---------- a follow-up about what the person is saying NOW: made from the partial transcript while they are still talking, so it is ready (and voiced) when the answer ends ----------
    def speculate_followup(self, partial: str):
        if not (getattr(self, "keep_going", False) and getattr(self, "fast_followup", True)) or self.cur is None: return
        mc = getattr(self, "spec_min_chars", 15)                                       # adopted 2026-11-20: 40 -> 15 chars (follow-up ready ~41% sooner, same Whisper work; see docs/LATENCY_AB.md)
        if len(partial.strip()) < mc: self.spec_why = self.__dict__.get("spec_why") or f"말이 {mc}자 미만이라 아직 준비를 시작하지 않음"; return
        if self.refill_wanted(): self.spec_why = "질문 보충(남은 질문 0~1개)이 먼저라 대기"; return
        sp = getattr(self, "spec_fu", None)
        if sp and sp["qid"] == self.cur.id and len(partial) - len(sp["text"]) < 40: return        # nothing new worth redoing
        t = getattr(self, "spec_task", None)
        if t and not t.done(): return                                                                  # one at a time
        self.spec_task = asyncio.create_task(self._spec(partial, self.cur))
    # ---------- follow-up grounded in the passage: catch a new term from the answer -> find the passage sentence that has it -> ask about THAT sentence ----------
    @staticmethod
    def _term_ok(t: str | None) -> bool:
        """A term worth saying aloud: not an inflected word fragment like '모델 요청한'."""
        if not t or re.search(r"(한|하는|합니다|입니다|하고|하며|이다|된|되는|할|인|는|은|를|을)$", t): return False
        return bool(re.search(r"[A-Za-z0-9]", t)) or len(t.replace(" ", "")) >= 3          # a plain 2-syllable Korean word ("모델", "기업") is too generic to pick up
    def lead_for(self, typ: str | None, depth: int | None = None, term: str | None = None) -> str:
        """Spoken bridge before a follow-up: 'I heard you; about <the part you said>, one more question'. Type-specific for the first follow-up, 'one more' for later ones, generic for a prepared candidate."""
        d = self.followups if depth is None else depth
        if d >= 1: return FU_LEAD_CHAIN
        if typ in FU_LEAD_TERM and self._term_ok(term) or typ == "순서": return f"{FU_ACK} " + FU_LEAD_TERM[typ].format(t=term)
        return FU_LEAD.get(typ or "", FU_LEAD_CANNED)
    def _all(self):
        """Every passage sentence (as Requirement objects) when a passage is loaded, else the plain requirement list."""
        return getattr(self, "all_reqs", None) or self.reqs
    def _df(self, t: str) -> int:
        """In how many passage sentences a term occurs (cached): rare terms are what link an answer to its source sentence."""
        c = self.__dict__.setdefault("_dfc", {})
        if t not in c: c[t] = sum(t in x.text for x in self._all())
        return c[t]
    def find_ground(self, answer: str, cur) -> tuple[str, list[str]] | None:
        """(passage sentence, matched new terms) or None. Terms must be new (not in the question), rare in the passage (<=2 sentences) and the sentence must not already have been used for a follow-up."""
        qtok = set(re.findall(r"[가-힣A-Za-z0-9_]{2,}", cur.text if cur else ""))
        used = self.__dict__.setdefault("used_sents", set())
        josa = re.compile(r"(으로|에서|에게|까지|부터|보다|은|는|이|가|을|를|와|과|로|에|의|도|만)$")
        toks = [josa.sub("", t) if len(t) > 2 else t for t in dict.fromkeys(re.findall(r"[가-힣A-Za-z0-9_]{2,}", answer))]
        toks = [t for t in dict.fromkeys(toks) if len(t) >= 2 and t not in qtok and not any(t in x for x in qtok)]
        if not toks: return None
        best, bt = None, []
        for r in self._all():
            if r.text in used: continue                                                        # a sentence already used for a follow-up is not used again (the question's own source sentence is fine: other term / other direction)
            hit = [t for t in toks if t in r.text and self._df(t) <= 2]
            hit.sort(key=lambda t: len(next((c for c in re.split(r"[,，]|하고|하며|그리고|고 |며 ", answer) if t in c), answer)))      # the term mentioned in the shortest clause first
            hit = [t for t in hit if not any(t != u and t in u for u in hit)]                                      # "Design" inside "Component Design": keep the longer one
            hit = [t for t in hit if re.search(r"[A-Za-z0-9]", t) or self._term_ok(t)]                            # no word fragments ('모델 요청한') as the term to pick up
            if not any(re.search(r"[A-Za-z0-9]", t) for t in hit) and len(hit) < 2: continue                     # one plain Korean word ("위험한") is too weak a link to a passage sentence
            if len(hit) > len(bt): best, bt = r.text, hit
        if best and bt:                                                                               # "골든 세트": two hit words in a row are one term
            josa2 = re.compile(r"(으로|에서|에게|까지|부터|보다|은|는|이|가|을|를|와|과|로|에|의|도|만)$")
            def join(t):
                m = re.search(re.escape(t) + r"[가-힣A-Za-z0-9_]*\s+([가-힣A-Za-z0-9_]{2,})", answer); nxt = josa2.sub("", m.group(1)) if m else ""
                return f"{t} {nxt}" if nxt in bt and nxt != t else t
            bt = list(dict.fromkeys(join(t) for t in bt))
        return (best, bt) if best and bt else None
    def seq_gap(self, answer: str, sent: str) -> str | None:
        """Passage sentence with an order (A -> B -> C): returns the question when the answer stops early ("그다음 단계는 무엇인가요?") or skips a middle step ("'B'도 설명해 주시겠어요?"). Conservative: a step counts as missing only if NONE of its rare terms is in the answer."""
        cl = [c.strip() for c in re.split(r"(?<=고)\s+|(?<=뒤)\s+|(?<=며)\s+|(?<=이후)\s+|\s마지막으로\s|→|[,，]\s*|\s그리고\s|\s이후\s|\s다음\s", sent) if len(c.strip()) >= 4]
        josa = re.compile(r"(으로|에서|에게|까지|부터|보다|은|는|이|가|을|를|와|과|로|에|의|도|만)$")
        steps = []
        for c in cl:
            ks = [josa.sub("", t) if len(t) > 2 else t for t in dict.fromkeys(re.findall(r"[가-힣A-Za-z0-9_]{2,}", c))]
            ks = [k for k in ks if len(k) >= 2 and self._df(k) <= 2]
            if ks: steps.append(ks)
        if len(steps) < 3: return None
        cov = [any(k in answer for k in ks) for ks in steps]
        if not cov[0] or sum(cov) < 2: return None
        n = len(steps); last = max(i for i in range(n) if cov[i])
        miss = [i for i in range(1, last) if not cov[i]]                    # a step in the middle was skipped
        if miss: return f"'{steps[miss[0]][0]}'도 설명해 주시겠어요?"
        return "그다음 단계는 무엇인가요?" if last < n - 1 else None            # the answer stops before the last step
    def pick_fu_type(self, answer: str, sent: str, term: str) -> str | None:
        """Order: 더 자세히 (only when the clause with the new term is short) -> 검증 -> 근거 -> 순서. Never the same type twice in a row; each type needs its own evidence in the passage sentence."""
        last = getattr(self, "last_fu_type", None)
        mine = next((c for c in re.split(r"[,，]|하고|하며|그리고|고 |며 ", answer) if term in c), answer)       # the clause where the term was mentioned
        cands = []
        if len(mine.strip()) < 20 or len(mine.split()) < 8: cands.append("더 자세히")
        if re.search(r"측정|검증|확인|기준|평가|테스트|차단", sent) and not re.search(r"측정|검증|확인|테스트", mine): cands.append("검증")
        if re.search(r"때문|위해|이유|따라서|해야|필요|므로", sent) and not re.search(r"때문|위해|이유", mine): cands.append("근거")
        if self.seq_gap(answer, sent): cands.append("순서")
        cands.append("더 깊이")                                                   # last resort: a grounded sentence exists but no specific type fits -> ask one step deeper about it (so a right / partial answer still gets a follow-up)
        cands = [c for c in cands if c != last] or cands
        return cands[0] if cands else None
    FU_DIR = {"근거": "그렇게 하는 이유", "검증": "어떻게 확인·측정·검증하는지", "더 자세히": "무엇이고 어떻게 하는지 더 자세히", "더 깊이": "어떻게 동작하는지"}
    async def _spec(self, partial: str, cur):
        try:
            g = self.find_ground(partial, cur)
            if not g: self.spec_why = "내 말에서 지문에 근거가 있는 새 단어를 못 찾음"; return        # nothing in the passage to base it on: no follow-up (never an ungrounded one)
            sent, terms = g; typ = self.pick_fu_type(partial, sent, terms[0])
            if not typ: self.spec_why = f"'{terms[0]}'은(는) 찾았지만 질문 유형 조건에 해당 없음(설명이 충분히 길거나 근거 문장에 이유·확인·순서 표현이 없음)"; return
            if typ == "순서":
                q = self.seq_gap(partial, sent)
                if q and self.cur is cur and not is_repeat(q, self.asked_texts, th=0.5): self.spec_fu = {"qid": cur.id, "text": partial, "q": q, "type": typ, "sent": sent, "lead": (ld := self.lead_for(typ, term=terms[0])), "say": f"{ld} {q}"}
                else: self.spec_why = "순서 질문이 이미 한 질문과 비슷함"
                return
            prompt = f"[지문 문장]\n{sent}\n짚을 말: '{terms[0]}'\n질문 방향: '{terms[0]}'에 대해 {self.FU_DIR[typ]}를 묻는다. '이 말이'로 시작하지 않는다.\n이 지문 문장의 내용만으로 질문 한 문장만 출력한다. 40자 이내로 짧게, 쉼표나 '~며'로 두 가지를 잇지 않는다. 직전 질문을 되풀이하지 않는다. 존댓말 의문문(~나요?)으로 쓴다.\n"
            buf = ""
            async def gen():
                nonlocal buf
                async for tok in self.fast.stream(getattr(self, "gen_sys", GEN_SYS), prompt, 100):
                    buf += tok
                    if "?" in buf or "？" in buf: break
            await asyncio.wait_for(gen(), 6)
            m = re.findall(r"[^.!?]*\?", clean_q(buf)); q = m[-1].strip() if m else ""
            if self._fu_ok(q, partial, cur) and self.cur is cur: self.spec_fu = {"qid": cur.id, "text": partial, "q": q, "type": typ, "sent": sent, "lead": (ld := self.lead_for(typ, term=terms[0])), "say": f"{ld} {q}"}
            else: self.spec_why = f"만든 질문이 검사에 걸림({typ}): {q[:40]}"
        except asyncio.CancelledError: raise
        except Exception: pass
    @staticmethod
    def _copy_ratio(q: str, answer: str) -> float:
        """Share of the question's characters copied verbatim (8+ chars) from the answer. A follow-up may quote a term or two; it must not be mostly the answer read back."""
        a = re.sub(r"\s+", "", answer); h = re.sub(r"\s+", "", q)
        if len(h) < 8: return 0.0
        hit = [False] * len(h)
        for i in range(len(h) - 7):
            if h[i:i + 8] in a:
                for j in range(i, i + 8): hit[j] = True
        return sum(hit) / len(h)
    def _fu_ok(self, q: str, answer: str, cur) -> bool:
        """A real follow-up: well-formed, NOT a re-wording of anything already asked (wording similarity only: it may share topic words with its question), built on something new the person said."""
        if not (8 <= len(q) <= 110 and ko_ok(q) and one_thing(q)) or bad_question(q) or is_repeat(q, self.asked_texts, th=0.5) or self._copy_ratio(q, answer) > 0.6: return False
        if re.search(r"방금 답변|구체적인 말|답변에서", q) or not re.search(r"(나요|까요|세요|가요|어요|아요|죠|니까|시겠어요|시겠습니까|할까요)\s*[?？]$", q.strip()): return False     # prompt wording leaked / not a polite closing question
        new_terms = _kw(answer) - _kw(cur.text if cur else "")
        return bool(_kw(q) & new_terms)                                  # must mention a term that came from the answer, not only the original question's words
    def fresh_spec(self, answer: str) -> str | None:
        """The follow-up prepared during the answer, if the final answer still matches what it was made from."""
        sp = getattr(self, "spec_fu", None)
        if not sp or self.cur is None or sp["qid"] != self.cur.id: return None
        ka, kp = _kw(answer), _kw(sp["text"])
        return sp["q"] if kp and len(kp & ka) / len(kp) >= 0.6 and not is_repeat(sp["q"], self.asked_texts, th=0.5) else None
    def start_answer(self, q):
        # model answers / verdicts are made when the report is opened (writing them while the person answers heated the Mac). Only the follow-up candidates are prepared now.
        if q is None or not getattr(self, "keep_going", False): return
        self.start_followup_prep(q)
        return
        t = getattr(self, "answer_task", None)
        if t and not t.done(): t.cancel()
        self.answer_task = asyncio.create_task(self._write_answer(q))
    async def _write_answer(self, q):
        try:
            ref = self.refs_for(q.text, 3)
            buf = ""
            async for tok in self.fast.stream("당신은 면접 정답 해설자다. [참고 지문]만 근거로 [질문]에 대한 모범 답안을 한국어 존댓말 두 문장, 120자 이내로 쓴다. 지문에 없는 내용은 쓰지 않는다. 답안 문장만 출력한다.", f"[참고 지문]\n{ref}\n[질문]\n{q.text}\n모범 답안:", 110): buf += tok
            ans = buf.strip()
            if len(ans) >= 10: self.__dict__.setdefault("model_answers", {})[q.id] = ans
        except asyncio.CancelledError: raise
        except Exception: pass
    def get_answer(self, q) -> dict:
        """Model answer if it is ready; otherwise the passage excerpt the question is based on (instant, never waits)."""
        if q is None: return {"text": "", "src": ""}
        be = getattr(q, "bank", None); a = (be or {}).get("a") or getattr(self, "model_answers", {}).get(q.id)
        if a: return {"text": a, "src": "AI 모범 답안"}
        return {"text": self.refs_for(q.text, 2)[:220], "src": "지문 발췌"}
    async def _prefetch(self):
        from .passage import PSYS
        try:
            while len(self.queue.pending()) < 4:
                self.sync_bank(); b = getattr(self, "_bank", None)
                if b and not b["done"]: break                       # the bank builder is still working on it
                chunks = getattr(self, "_chunks", [])
                if chunks:
                    c = chunks.pop(0)
                    try:
                        d = await self.fast.json(PSYS, f"[지문 일부]\n{c}", 400)
                    except asyncio.CancelledError: chunks.insert(0, c); raise
                    except Exception: continue
                    for q in (d.get("questions", []) if isinstance(d, dict) else d) or []:
                        t = str(q.get("text", q) if isinstance(q, dict) else q).strip()
                        if len(t) >= 6 and not self._seen(t): self.queue.add(Question(t, Action.DEEPEN, None, 0.5, "prefetch", ""))
                else:
                    topic = self._next_topic()
                    if not topic: break
                    buf = ""
                    async for tok in self.fast.stream(getattr(self, "gen_sys", GEN_SYS), f"Action: SWITCH\n새 영역(이 문장의 내용만으로 질문한다): {topic}\n질문 문장 한 개만 출력한다.\n", 90):
                        buf += tok
                        if "?" in buf or "？" in buf: break
                    q = clean_q(buf)
                    if "?" in q: q = q[:q.index("?") + 1]
                    if 8 <= len(q) <= 120 and ko_ok(q) and not self._dup_like(q): self.queue.add(Question(q, Action.SWITCH, None, 0.4, "prefetch", ""))
        except asyncio.CancelledError: raise
        except Exception: pass

    # ---------- answer judgement + next-area policy (live experiment) ----------
    @staticmethod
    def _dont_know(answer: str) -> bool:
        a = answer.strip()
        if len(a) < 60 and KW_NO.search(a): return True                      # short answer containing a clear "don't know"
        return len(a) < 25 and bool(KW_NO_WEAK.search(a))                    # "다음 질문 부탁드립니다" etc. only when that is the whole answer
    def ref_of(self, q) -> str:
        """The passage text a question was made from: remembered when the question was created (bank: its source sentence; follow-up: its parent's). No search needed later."""
        if q is None: return ""
        r = ((getattr(q, "bank", None) or {}).get("ref")) or getattr(q, "ref_text", "") or ""
        return r[:300] if r else self.refs_for(q.text, 1)[:300]            # one sentence: a second, loosely related one misleads the judge
    def refs_for(self, text: str, k: int = 2) -> str:
        kw = _kw(text); sc = sorted(self._all(), key=lambda r: -len(kw & _kw(r.text)))
        return " ".join(r.text for r in sc[:k])
    def quick_verdict(self, answer: str, question: str | None = None) -> dict:
        """Zero-LLM verdict used to choose the next area (keeps the 2 s target); the LLM verdict is added later for the report."""
        q = question if question is not None else (self.cur.text if self.cur else ""); ref = self.refs_for(q)
        if self._dont_know(answer): return {"verdict": "모름", "why": "모른다고 답했어요.", "ref": ref}
        ov = len(_kw(answer) & _kw(ref))
        return {"verdict": "맞음" if ov >= 4 else "부분 맞음" if ov >= 2 else "틀림", "why": "", "ref": ref}
    @staticmethod
    async def grade_with(llm, ref: str, q: str, answer: str) -> dict:
        """Report time, ONE call per answer: verdict + reason + model answer together (half the calls of doing them separately). Falls back to two calls if the JSON is unusable."""
        dk = QAEngine._dont_know(answer); out = {"verdict": "모름", "why": "모른다고 답했어요.", "ref": ref, "model_answer": ""}
        sysm = "당신은 면접 채점관이다. [참고 지문]만 근거로 [질문]에 대한 [답변]을 채점하고 모범 답안을 쓴다. JSON만 출력한다. verdict는 맞음, 부분 맞음, 틀림 중 하나. why는 답변이 왜 그렇게 채점됐는지 구체적인 이유 한 문장(틀리거나 빠진 부분의 맞는 내용 포함). model_answer는 모범 답안을 한국어 존댓말 두 문장 120자 이내로(지문에 없는 내용은 쓰지 않는다). 예: {\"verdict\":\"부분 맞음\",\"why\":\"복구 목적은 맞지만 상태 저장 시점이 빠졌습니다.\",\"model_answer\":\"체크포인트는 중간 상태를 저장해 실패 시 그 지점부터 이어서 복구합니다.\"}"
        try:
            d = await llm.json(sysm, f"[참고 지문]\n{ref}\n[질문]\n{q}\n[답변]\n{answer[:600]}", 260)
            if isinstance(d, dict):
                x = str(d.get("verdict", "")).replace(" ", ""); ma = str(d.get("model_answer", "")).strip()
                if not dk: out.update(verdict="맞음" if x.startswith("맞") else "틀림" if x.startswith("틀") else "부분 맞음", why=("" if str(d.get("why", "")).startswith("한국어 한 문장") else str(d.get("why", "")))[:160])
                if len(ma) >= 10: out["model_answer"] = ma[:220]
        except Exception: pass
        if len(out["model_answer"]) < 10:
            buf = ""
            try:
                async for tok in llm.stream("당신은 면접 정답 해설자다. [참고 지문]만 근거로 [질문]에 대한 모범 답안을 한국어 존댓말 두 문장, 120자 이내로 쓴다. 지문에 없는 내용은 쓰지 않는다.", f"[참고 지문]\n{ref}\n[질문]\n{q}\n", 160): buf += tok
            except Exception: pass
            out["model_answer"] = buf.strip()
        return out
    @staticmethod
    async def judge_with(llm, ref: str, q: str, answer: str) -> dict:
        """Same judging as judge_answer, usable without a running session (report time)."""
        if QAEngine._dont_know(answer): return {"verdict": "모름", "why": "모른다고 답했어요.", "ref": ref}
        verdict, why = "부분 맞음", ""
        try:
            d = await llm.json(JUDGE_SYS, f"[참고 지문]\n{ref}\n[질문]\n{q}\n[답변]\n{answer[:600]}", 90)
            if isinstance(d, dict):
                x = str(d.get("verdict", "")).replace(" ", "")
                verdict = "맞음" if x.startswith("맞") else "틀림" if x.startswith("틀") else "부분 맞음"; why = str(d.get("why", ""))[:160]
        except Exception: pass
        return {"verdict": verdict, "why": why, "ref": ref}
    async def judge_answer(self, answer: str, question: str | None = None) -> dict:
        q = question if question is not None else (self.cur.text if self.cur else ""); ref = self.refs_for(q)
        if self._dont_know(answer): v = {"verdict": "모름", "why": "모른다고 답했어요.", "ref": ref}
        else:
            verdict, why = "부분 맞음", ""
            try:
                d = await self.fast.json(JUDGE_SYS, f"[참고 지문]\n{ref}\n[질문]\n{q}\n[답변]\n{answer[:600]}", 90)
                if isinstance(d, dict):
                    x = str(d.get("verdict", "")).replace(" ", "")
                    verdict = "맞음" if x.startswith("맞") else "틀림" if x.startswith("틀") else "부분 맞음"; why = str(d.get("why", ""))[:160]
            except Exception: pass
            v = {"verdict": verdict, "why": why, "ref": ref}
        self.__dict__.setdefault("verdicts", []).append(v); return v
    def next_action(self, v: dict, a: Assessment) -> Action:
        """Never drill one area twice: wrong / don't-know -> new area; right or partial -> at most one deeper follow-up, then new area."""
        if a.contradictions and self.followups < 1: return Action.CHALLENGE
        if v["verdict"] in ("모름", "틀림"): return Action.SWITCH
        return Action.DEEPEN if self.followups < MAX_FU else Action.SWITCH      # follow-up of a follow-up allowed, at most MAX_FU per question
    def sync_bank(self):
        """Move questions the background bank has finished since the last call into the live queue (each keeps a link to its model answer)."""
        b = getattr(self, "_bank", None)
        if not b: return
        i = getattr(self, "_bank_i", 0)
        for e in b["qs"][i:]:
            q = Question(e["q"], Action.DEEPEN if i == 0 else Action.SWITCH, None, 1.0 - 0.02 * i, "bank", ""); q.bank = e; self.queue.add(q); i += 1
        self._bank_i = i
    async def _topic_question(self, topic: str) -> str:
        buf = ""
        async for tok in self.fast.stream(getattr(self, "gen_sys", GEN_SYS), f"Action: SWITCH\n새 영역(이 문장의 내용만으로 질문한다): {topic}\n질문 문장 한 개만 출력한다.\n", 90):
            buf += tok
            if "?" in buf or "？" in buf: break
        q = clean_q(buf)
        return q[:q.index("?") + 1] if "?" in q else q
    async def next_question_now(self):
        """[질문 건너뛰기]: always returns a question quickly - prepared one, else one made on the spot (max 4 s), else a template."""
        q = self._pop_fresh()
        if q: return q
        topic = self._next_topic((self.cur.text if self.cur else ""))
        if not topic: return None
        try: txt = await asyncio.wait_for(self._topic_question(topic), 4)
        except Exception: txt = ""
        if not (8 <= len(txt) <= 120): txt = self._topic_template(topic)
        return Question(txt, Action.SWITCH, None, 0.4, "skip", "")
    def _pop_fresh(self):
        self.sync_bank()
        while True:
            q = self.queue.pop()
            if q is None: return None
            if not self._seen(q.text) and not self._blocked(q): return q
    def _blocked(self, q) -> bool:
        """Questions the person has mastered in earlier practices (ledger): never asked again."""
        fn = getattr(self, "blocked_fn", None)
        return bool(fn and fn(q))
    def _next_topic(self, hint: str = "") -> str | None:
        """Next passage sentence for a question made on the spot: the one FARTHEST from the sentences already used (early -> middle -> late, never the neighbourhood of the last topic)."""
        allr = self._all(); used = self.__dict__.setdefault("used_topics", set())
        idx = {r.id: i for i, r in enumerate(allr)}; pool = [r for r in allr if r.id not in used and self.is_topic(r.text)]
        if not pool: used.clear(); pool = [r for r in allr if self.is_topic(r.text)] or list(allr)
        if not pool: return None
        ref = [idx[u] for u in used if u in idx]
        r = max(pool, key=lambda r: min((abs(idx[r.id] - u) for u in ref), default=len(allr)))
        used.add(r.id); return r.text
    @staticmethod
    def is_topic(t: str) -> bool: return 15 <= len(t) <= 300
    def _topic_template(self, sent: str) -> str:
        """Safe question when the model could not make one: about the rarest term of the sentence (never the raw sentence, which carries timestamps / fragments)."""
        cands = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[가-힣]{3,}", sent) if re.search(r"[A-Za-z]", t) or self._term_ok(t)]
        if not cands: return "제안서의 수치나 기간 중 하나를 골라 그 근거를 말씀해 주시겠어요?"
        return f"{min(cands, key=lambda t: (self._df(t), -len(t)))}의 근거나 수치를 말씀해 주시겠어요?"

    def first_question(self) -> Question | None:
        q = self.queue.pop()
        if q: self._asked(q)
        return q
    def _asked(self, q: Question):
        self.cur = q; self.asked_texts.append(q.text); self.followups = 0; self.spoken_full = q.text; self.heard_chars = len(q.text); self.spec_why = ""
    def mark_played(self, chars: int):
        """Only the portion actually played counts as heard by the user (barge-in / interjection truncation)."""
        self.heard_chars = max(0, min(chars, len(self.spoken_full)))
    def heard_text(self): return self.spoken_full[:self.heard_chars]

    async def should_interject(self, partial: str, elapsed: float) -> Decision:
        return await self.policy.decide(partial, self.cur.text if self.cur else "", elapsed, self.answers)

    async def analyze_answer(self, answer: str) -> Assessment:
        a = await self.analyzer.analyze(self.cur.text if self.cur else "", answer, " / ".join(self.answers)[-500:])
        for rid in a.covered_reqs:
            if rid in self.cov.reqs: self.cov.update(rid, 0.8, answer[:100], time.monotonic() - self.t0); self.queue.resolve_req(rid, "answered in Q&A")
        return a

    def fast_assess(self, answer: str) -> Assessment:
        """Zero-LLM assessment for the fast track (Q2/Q4): rule contradictions + length/number/keyword sufficiency heuristics."""
        a = Assessment(); c = numeric_conflict(answer, self._all())
        if c: a.contradictions.append(c)
        for p in self.answers:
            for x in numbers(answer):
                for y in numbers(p):
                    if _unit(x) == _unit(y) and _num(x) != _num(y) and len(_kw(answer) & _kw(p)) >= 3: a.contradictions.append(f"이전 '{y}' vs '{x}'")
        w = len(answer.split()); kq = _kw(self.cur.text) if self.cur else set()
        a.sufficiency = min(1.0, 0.25 + (0.25 if numbers(answer) else 0) + (0.2 if w >= 15 else 0) + (0.2 if len(kq & _kw(answer)) >= 2 else 0) + (0.1 if w >= 30 else 0))
        a.off_topic = 0.8 if w >= 25 and len(kq & _kw(answer)) == 0 else 0.0
        a.summary = answer[:60]; return a

    async def plan_and_stream(self, answer: str, a: Assessment, forced_action: Action | None = None, verdict: dict | None = None, interject_kind: str | None = None):
        """Decide action, then yield (action, token) pairs. Question strategy changes which path is used."""
        _clar = clarify.classify(answer) if (self.cur and interject_kind is None) else None
        if _clar:                                                    # "다시 말씀해 주세요" / "더 쉽게 말씀해 주세요": not an answer; the SAME question word for word (easier wording is NOT given: the questioner keeps its stance)
            self.last_clarify = _clar                                # runtime flags the turn (never graded) and tells the screen
            for w in re.findall(r"\S+\s*", self.cur.text): yield Action.REPEAT, w
            return
        elapsed = (time.monotonic() - self.t0) / 1800
        act = forced_action or decide_action(a, self.followups, self.queue.peek() is not None, elapsed_ratio=elapsed)
        if act == Action.END and getattr(self, "keep_going", False): act = Action.DEEPEN     # live experiment: only the user ends the session (never run out of questions)
        if act == Action.END and self.strategy == "direct" and elapsed < 1.0: act = Action.DEEPEN      # Q1: never consults the bank; always generates
        if act == Action.SWITCH and self.strategy == "direct": act = Action.DEEPEN
        if act == Action.END: yield act, ""; return
        if is_repeat_request(answer) and self.cur and forced_action is None:
            for w in re.findall(r"\S+\s*", self.cur.text): yield Action.REPEAT, w
            return
        seed = None
        if act == Action.SWITCH or (self.strategy == "precomputed" and act != Action.CHALLENGE):
            q = self._pop_fresh()
            if q:
                self._asked(q); self.answers.append(answer); self._record(answer, a, act, q.text)
                for w in re.findall(r"\S+\s*", q.text): yield act, w      # precomputed question: no LLM call
                return
            if act == Action.SWITCH:
                if getattr(self, "keep_going", False): seed = self._next_topic((self.cur.text if self.cur else "") + " " + answer[-200:])     # prepared questions used up: ask about the next not-yet-used passage sentence
                else: yield Action.END, ""; return
        if getattr(self, "fast_followup", True) and not interject_kind and act in (Action.DEEPEN, Action.CLARIFY, Action.VERIFY, Action.CHALLENGE):
            if act == Action.CHALLENGE:           # contradiction: fixed sentence (no LLM); the opening words are voiced in advance
                kind = "contradiction_rfp" if any("RFP" in c for c in a.contradictions) else "contradiction_prior"
                ft = next((t for t in IJ_TEMPLATES[kind] if not self._seen(t)), None)
            else:
                sp_ = self.fresh_spec(answer); ft = sp_ or self.canned_followup((verdict or {}).get("verdict", ""))
                why_ = self.__dict__.get("spec_why") or ("답변 중에 만들 시간이 부족했거나 끝날 때까지 완성되지 않음" if self.__dict__.get("spec_task") else "")
                self.last_fu_note = (f"내 답변 기반 꼬리질문({self.spec_fu['type']}) · 지문 근거 문장 사용" if sp_ else
                                     f"미리 만든 후보 사용 (내 답변 기반은 없음: {why_})" if ft else f"꼬리질문 없음 → 다음 준비된 질문으로 (내 답변 기반: {why_})")
            if ft:                                # prepared follow-up: no LLM call, voice already synthesised -> sound within ~0.1 s of the answer ending
                nq = Question(ft, act, self.cur.req_id if self.cur else None, 1.0, "followup", "", status="asked"); nq.fu = {"-": ""}; nq.ref_text = self.ref_of(self.cur)     # prepared canned candidates belong to the original question only
                sp = getattr(self, "spec_fu", None)
                if sp and sp.get("q") == ft: nq.ref_text = sp["sent"]; self.__dict__.setdefault("used_sents", set()).add(sp["sent"]); self.last_fu_type = sp["type"]       # the chain continues from this sentence
                is_spec = bool(sp and sp.get("q") == ft)
                lead = "" if act == Action.CHALLENGE else (sp.get("lead") if is_spec and sp.get("lead") else self.lead_for(None))
                ft_say = f"{lead} {ft}".strip()
                self.cur_fu = {"type": (sp["type"] if is_spec else "미리 만든 후보") if act != Action.CHALLENGE else "모순 지적", "depth": self.followups + 1}
                self.answers.append(answer); self.followups += 1
                self.queue.items.append(nq); self.asked_texts.append(ft); self.cur = nq; self.spoken_full = ft_say; self.heard_chars = len(ft_say)
                self._record(answer, a, act, ft)
                for w in re.findall(r"\S+\s*", ft_say): yield act, w
                return
            if act != Action.CHALLENGE and getattr(self, "keep_going", False) and getattr(self, "_bank", None):     # no follow-up finished in time: drop it, ask the next prepared question instead
                q = self._pop_fresh()
                if q:
                    self._asked(q); self.answers.append(answer); self._record(answer, a, Action.SWITCH, q.text)
                    for w in re.findall(r"\S+\s*", q.text): yield Action.SWITCH, w
                    return
        ev = self.cur.evidence if self.cur else ""
        if a.contradictions and self.cur and self.cur.req_id:
            ev = ev or self.cov.reqs[self.cur.req_id].text
        if seed is None and act in (Action.DEEPEN, Action.VERIFY, Action.CLARIFY) and len(self.asked_texts) >= 2:
            nxt = self.queue.peek()                                        # a not-yet-asked prepared question as an angle hint -> keeps follow-ups from circling one topic
            if nxt and not is_repeat(nxt.text, self.asked_texts): seed = nxt.text
        prompt = gen_prompt(act, self.cur.text if self.cur else "", answer, a, ev, self.asked_texts, seed, (verdict or {}).get("verdict", ""), (verdict or {}).get("ref", "")[:300])
        if interject_kind: prompt = interject_prompt(interject_kind, answer, self.cur.text if self.cur else "", self.asked_texts)
        buf = ""; gave_up = False
        for attempt in range(2):                      # attempt 2 only if the first one starts like an already-asked question (small models loop)
            buf = ""; held: list[str] = []; ok = False; abort = False
            pr = prompt if attempt == 0 else prompt + "※ 방금 만든 질문은 이미 한 질문과 시작이 같거나 답변 문장을 베꼈다. 답변 문장을 쓰지 말고, 완전히 다른 첫 단어로, 지문의 개념어만 써서 70자 안팎의 짧은 질문 한 문장을 다시 만들어라.\n"
            agen = self.fast.stream(getattr(self, "gen_sys", GEN_SYS), pr, 100)
            async for tok in agen:
                buf += tok
                if not ok:
                    held.append(tok)
                    if len(buf.strip()) >= 14:
                        nd = getattr(self, "no_dup", True) and not interject_kind
                        if nd and not ("?" in buf or "？" in buf): continue            # hold until the whole question exists, so a duplicate is caught before anything is spoken
                        if nd and (self._seen(clean_q(buf)) or bad_question(clean_q(buf))): abort = True; gave_up = attempt == 1; break
                        if attempt == 0 and (self._same_start(buf) or (not interject_kind and self._echoes(buf, answer))): abort = True; break
                        ok = True
                        yield act, clean_q("".join(held)).lstrip("0123456789.)")       # models sometimes start with a list bullet copied from the prompt
                        held = []
                        if "?" in buf or "？" in buf: break
                else: yield act, tok
                if ok and ("?" in tok or "？" in tok): break                      # one question only (small models tend to append a second one)
            if abort:
                await agen.aclose(); continue
            for t in held: yield act, t
            break
        if gave_up:                                      # the model produced an already-asked question twice: ask a prepared, not-yet-asked one instead
            q2 = self._pop_fresh()
            if q2:
                self._asked(q2); self.answers.append(answer); self._record(answer, a, Action.SWITCH, q2.text)
                for w in re.findall(r"\S+\s*", q2.text): yield Action.SWITCH, w
                return
            topic = self._next_topic(answer[-200:])      # nothing prepared is left: ask about the next not-yet-used passage sentence (never stay silent)
            if topic:
                txt = self._topic_template(topic); self._asked(Question(txt, Action.SWITCH, None, 0.4, "fallback", "", status="asked"))
                self.answers.append(answer); self._record(answer, a, Action.SWITCH, txt)
                for w in re.findall(r"\S+\s*", txt): yield Action.SWITCH, w
                return
            for w in re.findall(r"\S+\s*", buf): yield act, w          # last resort: say what was made rather than nothing
        self.last_repeat = is_repeat(buf, self.asked_texts)   # flagged for metrics; we don't block streaming on it
        self.answers.append(answer); self.followups = 0 if act == Action.SWITCH else self.followups + 1
        nq = Question(buf.strip(), act, self.cur.req_id if self.cur else None, 1.0, "answer", ev, status="asked")
        self.queue.items.append(nq); self.asked_texts.append(nq.text); self.cur = nq; self.spoken_full = nq.text; self.heard_chars = len(nq.text)
        self._record(answer, a, act, nq.text)

    @staticmethod
    def _echoes(buf: str, answer: str) -> bool:
        """The question starts by copying the (speech-recognised, often garbled) answer."""
        h = buf.strip()[:40]; a = re.sub(r"\s+", "", answer); h = re.sub(r"\s+", "", h)
        return any(h[i:i + 8] in a for i in range(0, max(len(h) - 7, 0)))
    async def make_interjection(self, kind: str, partial: str) -> str:
        """Interjection question, generated in full first so it can be checked: it must not repeat (or paraphrase) a question already asked.
        Returns "" when no acceptable question could be made -> the AI stays quiet instead of repeating itself."""
        if getattr(self, "fast_interject", True):                                # fixed sentence per kind: no LLM call, voice synthesised in advance
            for t in IJ_TEMPLATES.get(kind, []):
                if not self._seen(t): return t
        base = interject_prompt(kind, partial, self.cur.text if self.cur else "", self.asked_texts)
        for attempt in range(2):
            pr = base if attempt == 0 else base + "※ 방금 만든 질문은 이미 한 질문과 같거나 비슷했다. 답변자가 방금 한 말 중 다른 부분을 짚어 완전히 새로운 질문을 만들어라.\n"
            buf = ""
            async for tok in self.fast.stream(getattr(self, "gen_sys", GEN_SYS), pr, 90):
                buf += tok
                if "?" in buf or "？" in buf: break
            q = clean_q(buf)
            if "?" in q: q = q[:q.index("?") + 1]
            if 8 <= len(q) <= 110 and ko_ok(q) and not self._dup_like(q): return q
        return ""
    def _seen(self, text: str) -> bool:
        """No question, follow-up or interjection is ever asked twice: compared with EVERYTHING asked so far (wording similarity, or most of its key terms already used)."""
        if is_repeat(text, self.asked_texts, th=0.5): return True
        kw = _kw(text)
        return bool(kw) and any(_kw(a) and len(kw & _kw(a)) / max(min(len(kw), len(_kw(a))), 1) >= 0.6 for a in self.asked_texts)
    def _dup_like(self, q: str) -> bool:
        """Same question again: similar wording OR (also across Korean/English wording) most of its key terms already appear in a recent question."""
        recent = self.asked_texts[-4:]
        if is_repeat(q, recent, th=0.45): return True
        kw = _kw(q)
        return any(len(kw & _kw(a)) / max(min(len(kw), len(_kw(a))), 1) >= 0.6 for a in recent if kw and _kw(a))
    def accept_interjection(self, q: str, partial: str, a: Assessment, act: Action):
        nq = Question(q, act, self.cur.req_id if self.cur else None, 1.0, "answer", "", status="asked")
        self.queue.items.append(nq); self.asked_texts.append(q); self.cur = nq; self.spoken_full = q; self.heard_chars = len(q)
        self._record(partial, a, act, q)

    def _same_start(self, buf: str) -> bool:
        """The question starts like an already-asked one (any earlier question, label prefixes ignored) or is not Korean."""
        norm = lambda t: re.sub(r"\s+", "", clean_q(t))[:14]
        h = norm(buf)
        if len(h) >= 8 and sum(1 for c in h if "가" <= c <= "힣") < 2: return True        # English-led question
        return any(norm(a) == h for a in self.asked_texts)

    def _record(self, answer, a, act, qtext):
        self.turns.append({"answer": answer, "assessment": a.__dict__, "action": act.value, "next_question": qtext, "t": round(time.monotonic() - self.t0, 2)})

    def report(self):
        return {"coverage": self.cov.report(), "turns": self.turns, "queue": self.queue.snapshot(), "queue_log": self.queue.log}
