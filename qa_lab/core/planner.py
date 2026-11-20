"""Question planner: choose the next Action and question. Memory of verified/covered content prevents repeats."""
from __future__ import annotations
import re
from ..contracts import Action, Assessment, Question
from .question_queue import QuestionQueue, similarity

def decide_action(a: Assessment, asked_followups: int, queue_has: bool, max_followups=2, elapsed_ratio=0.0) -> Action:
    if elapsed_ratio >= 1.0: return Action.END
    if a.contradictions: return Action.CHALLENGE
    if a.off_topic > 0.6: return Action.CLARIFY
    if a.unsupported_claims and asked_followups < max_followups: return Action.VERIFY
    if a.sufficiency < 0.5 and asked_followups < max_followups: return Action.CLARIFY
    if a.sufficiency < 0.8 and asked_followups < max_followups: return Action.DEEPEN
    return Action.SWITCH if queue_has else Action.END

GEN_SYS = """당신은 까다롭지만 공정한 RFP 평가위원이다. 한국어로 질문 한 개만, 한두 문장으로, 구어체 존댓말로 말한다.
Action 의미: DEEPEN=더 깊이 / CHALLENGE=모순·충돌 지적 / CLARIFY=불명확한 부분 되묻기 / VERIFY=근거·수치 요구 / SWITCH=다음 주제.
이미 한 질문과 같은 질문은 하지 않는다. 직전 질문과 같은 문장 틀·같은 끝맺음(예: '~설명해 주시겠어요?')을 반복하지 말고, 방금 답변에서 새로 나온 구체적인 말(숫자·용어·주장)을 짚어 새 관점으로 묻는다. 'Action', '초점' 같은 라벨이나 설명 없이 질문 문장만 출력한다. 한국어(한글)로만 쓰고, 질문은 한 문장·70자 안팎으로 짧고 문법에 맞게 쓴다. 답변자가 한 말을 그대로 인용하거나 이어 붙이지 않는다(받아쓰기 오류가 섞여 있다). 지문의 개념어로 새로 쓴다. 일본어·중국어·한자를 절대 쓰지 말고 영어는 고유명사·약어만 쓴다. 문장은 '~무엇인가요?', '~설명해 주시겠어요?'처럼 존댓말로 끝낸다(반말·'설명해?' 금지).
새 영역(SWITCH)이면 직전 질문과 다른 주제를 처음부터 묻는다. 직전 답변 판정이 '맞음'이면 한 단계 더 깊은 질문, '부분 맞음'이면 빠진 부분을 묻는 질문을 한다."""

ANGLES = ["수치·근거", "한계·위험", "실제 적용 방법", "대안과 비교", "측정·검증 방법", "가정이 틀렸을 때"]

PASSAGE_GEN_SYS = GEN_SYS.replace("까다롭지만 공정한 RFP 평가위원", "까다롭지만 공정한 면접관") + "\n지문 면접이다: '발표', '발표자', '제안서', 'RFP'라는 단어는 쓰지 않는다. 지문 내용과 답변자가 방금 한 말에 대해서만 묻는다.\n질문은 반드시 한 가지만 묻는다. '~와 ~'로 두 주제를 묶거나, '~하고 ~도 설명해 달라'처럼 두 개를 한 문장에 넣지 않는다. '이해하고 있나요?'처럼 예 아니오로 끝나는 질문도 하지 않고, 어떻게, 왜, 무엇을 같은 말로 시작해 50자 안팎으로 쓴다. 괄호와 같은 말 반복(예: 재현성(재현성))은 쓰지 않는다. 소리 내어 읽는 문장이다."

def gen_prompt(action: Action, last_q: str, answer: str, a: Assessment, evidence: str, asked: list[str], seed: str | None = None, verdict: str = "", ref: str = "") -> str:
    prev = "\n".join("- " + q for q in asked[-6:]) or "- (없음)"
    if action == Action.SWITCH and seed:               # new area: do NOT anchor on the previous topic
        return (f"Action: SWITCH\n새 영역(이 문장의 내용만으로 질문한다): {seed}\n질문 문장 한 개만 출력한다.\n")
    focus = "; ".join(a.contradictions or a.unsupported_claims or [a.summary])[:200]
    return (f"Action: {action.value}\n초점: {focus}\n근거: {evidence or '없음'}\n직전 질문: {last_q}\n답변 요지(받아쓰기 오류가 많을 수 있음, 문장을 베끼지 말 것): {answer[-160:]}\n"+(f"참고 지문(정확한 내용): {ref}\n" if ref else "")+f""
            + (f"직전 답변 판정: {verdict}\n" if verdict else "") + f"이번 질문의 관점: {ANGLES[len(asked) % len(ANGLES)]}\n이미 한 질문(이와 다른 관점으로):\n{prev}\n"
            + (f"다음 주제 힌트: {seed}\n" if seed else "") + "질문 문장 한 개만 출력한다.\n")

def fu_prompt(last_q: str, answer: str, asked: list[str], ref: str = "") -> str:
    """Follow-up (probe) = digs into something the person JUST SAID: picks one concrete term/number/claim from the answer and asks for the reason, an example, how it is checked, or an exception.
    It shares the topic words of the original question on purpose; it must not re-ask the original question or use a canned angle."""
    prev = "\n".join("- " + q for q in asked[-6:]) or "- (없음)"
    return (f"직전 질문: {last_q}\n방금 답변(받아쓰기 오류가 있을 수 있음): {answer[-200:]}\n" + (f"참고 지문(정확한 내용): {ref}\n" if ref else "")
            + f"이미 한 질문(되풀이 금지):\n{prev}\n"
            "규칙: 방금 답변에서 구체적인 말 하나(용어·숫자·주장)를 골라 그것에 대해 '왜 그런지', '예를 들면 어떤 경우인지', '어떻게 확인하는지', '예외는 무엇인지' 중 하나만 묻는다. "
            "직전 질문을 다시 묻지 않는다. 답변 문장을 그대로 베끼지 않는다. 60자 안팎, 존댓말 의문문 한 문장만 출력한다.\n"
            "예시: 방금 '체크포인트로 복구한다'고 하셨는데, 어느 시점의 상태까지 되돌리나요?\n")

IJ_WHY = {"contradiction_rfp": "방금 말한 숫자가 지문과 다름", "contradiction_prior": "앞서 말한 내용과 모순됨", "repetition": "같은 설명을 반복함", "off_topic": "질문과 상관없는 말이 길게 이어짐", "overlong": "너무 길게 말함", "unsupported": "근거 없이 단정함"}

def interject_prompt(kind: str, partial: str, last_q: str, asked: list[str]) -> str:
    """Interjection: react to what the person is saying right now, in one short polite sentence."""
    prev = "\n".join("- " + q for q in asked[-4:]) or "- (없음)"
    return (f"끼어드는 이유: {IJ_WHY.get(kind, '확인이 필요함')}\n원래 질문: {last_q}\n답변자가 지금 하고 있는 말: {partial[-300:]}\n이미 한 질문(같은 질문 금지):\n{prev}\n"
            "규칙: 답변자의 말에서 한 가지를 구체적으로 짚어 짧게(40자 안팎) 되묻는 질문 한 문장만 쓴다. 반드시 '말씀 중에 잠시만요,'로 시작하고 '~무엇인가요?', '~어떻게 되나요?'처럼 존댓말 의문문으로 끝낸다.\n"
            "예시: 말씀 중에 잠시만요, 방금 재시도를 세 번까지 허용한다고 하셨는데 그 기준은 무엇인가요?\n")

_DEICTIC = re.compile(r"이 문장|이 그림|이 글에서|이 표|이 단락|위 내용|해당 문장|앞서 (말한|언급)|그렇게|이렇게|아까 (말한|하던)")
_POLITE_END = re.compile(r"(나요|까요|세요|가요|어요|아요|죠|니까|시겠어요|시겠습니까|할까요)\s*[?？]$")
def stutters(q: str) -> bool:
    """The same words twice in one question ("통화 기능과 통화 기능의 중요성"): a small-model repetition glitch."""
    w = [re.sub(r"(은|는|이|가|을|를|과|와|의|에|로|도|만)$", "", t) if len(t) > 2 else t for t in re.findall(r"[가-힣A-Za-z0-9]{2,}", q)]
    bi = [(w[i], w[i + 1]) for i in range(len(w) - 1)]
    if len(bi) != len(set(bi)): return True                                          # the same two-word phrase twice
    return any(len(t) >= 3 and t in w[i + 1:i + 6] for i, t in enumerate(w))          # the same word again within a few words

def bad_question(q: str) -> bool:
    """A spoken question must stand on its own and be polite: no page artifacts (dates/times), no pointing at things the listener cannot see ('이 문장', '이 아이러니적인 유사성'), no informal ending ('알고 싶어?')."""
    q = (q or "").strip()
    if stutters(q): return True
    if re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|최근 수정|편집\]", q): return True
    if _DEICTIC.search(q): return True
    if re.search(r"[♬♪×]|ㅣ|\s#|SBS|‘|’|[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff]", q): return True     # also Chinese / Japanese characters ("이前部分에서")
    if re.search(r"전체 자막|자막 내용|영상에 포함된|본문 전체|전체 내용|지문 전체", q): return True            # asks about the document itself, not its content                                         # page / video-title debris
    if re.match(r"^이제\s", q) or re.match(r"^(이것|그것|저것)(이|은|을|과|와)", q) or re.match(r"^(이|그|저)\s", q): return True   # a spoken question must not start by pointing at something the listener cannot see
    if re.match(r"^(그|이|저)\s+(\S+\s+)?(내용|부분|상황|경우|결과|방식|이벤트|현상|유사성|차이|관계|사실|문제|이유|영향|사례|주장|점)", q): return True      # points at an unnamed thing ("이 아이러니적인 유사성은")
    return not _POLITE_END.search(q)

def clean_q(q: str) -> str:
    """Strip list bullets and label prefixes ('질문:', 'Q1.') that small models like to add."""
    q = (q or "").strip().lstrip(" -•·\n\t")
    q = re.sub(r"^(질문|Q\d*|question)\s*[:：.)]\s*", "", q, flags=re.I).strip()
    q = re.sub(r"^새\s*영역(으로\s*넘어가서|으로|에서)?\s*", "", q).strip()                    # the prompt label leaks into the question otherwise
    q = re.sub(r"^(그리고|그래서|또|그러나|하지만|그럼)\s+", "", q).strip()                        # a spoken question does not start with a conjunction
    m = re.match(r"^(?:이제\s+)?(?:그|이|저)\s+(?!문장|그림|글|표|단락|내용|부분|상황|경우|결과|방식|이벤트|현상|유사성|차이|관계|사실|문제|이유|영향|사례|주장|점)(\S.*)$", q)
    if m: q = m.group(1).strip()                                                                 # "이 마음 건강 지킴 버스의 …" -> "마음 건강 지킴 버스의 …": the thing is named, only the pointing word goes
    q = re.sub(r"([가-힣A-Za-z0-9_\-]+)\s*\(\s*\1\s*\)", r"\1", q)                      # "재현성(재현성)" -> "재현성": the model repeats the same word in brackets
    q = re.sub(r"\s*[\(（]([^()（）]{1,30})[\)）]", r" \1", q)                           # other brackets are read aloud badly by the voice: keep the words, drop the marks
    return re.sub(r"\s{2,}", " ", q).strip()

def ko_ok(q: str) -> bool:
    """Questions must be Korean: at least 20% of the letters are Hangul and 6+ Hangul syllables (English terms inside a Korean sentence are fine)."""
    ko = len(re.findall(r"[가-힣]", q)); letters = len(re.findall(r"[A-Za-z가-힣]", q))
    return letters > 0 and ko / letters >= 0.2 and ko >= 6

def one_thing(q: str) -> bool:
    """A question must ask ONE thing: short, not a yes/no check, and not two topics joined into one sentence."""
    q = q.strip()
    if len(q) > 75: return False
    if re.search(r"(이해|알고|파악|숙지)\s*(하고)?\s*있(나요|습니까|으신가요)", q): return False                         # yes/no check, not a real question
    if len(q) > 65: return False
    if len(re.findall(r"[가-힣A-Za-z0-9)]\s*(?<![결효성])(?:와|과)\s|\s(?:및|그리고|또한)\s|하고\s", q)) >= 2: return False   # two or more subjects tied together
    if len(re.findall(r"(설명|구분|비교|말씀|알려)", q)) >= 2: return False                                                    # two requests in one
    if q.count("?") + q.count("？") > 1: return False
    return True

def is_repeat(text: str, asked: list[str], th=0.6) -> bool: return any(similarity(text, q) >= th for q in asked)

import re as _re
REPEAT_RX = _re.compile(r"(다시|한 ?번 ?더).{0,8}(말씀|질문|들려|설명해 ?주)|못 ?들었|잘 ?안 ?들|무슨 ?질문")
def is_repeat_request(text: str) -> bool: return bool(REPEAT_RX.search(text)) and len(text) < 60
