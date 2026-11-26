"""Clarification requests from the answerer.

  * "다시 말씀해 주세요" / "못 들었어요" / "너무 빨라요"  -> kind "repeat": the current question is said again, unchanged.
  * "더 쉽게 말씀해 주세요" / "이해가 안 돼요"           -> kind "easier": NOT accommodated. The questioner keeps its stance:
    the same question is said again word for word (no simpler wording, no hint). The person answers as best they can and the
    answer quality is judged in the report.
Both are matched by rules (microseconds, no LLM call), so a turn never gets slower. These turns are not answers: they are flagged
`clarify` and never graded.
"""
from __future__ import annotations
import re
from .planner import is_repeat_request

EXTRA_REPEAT_RX = re.compile(r"(천천히|너무 ?빨라|한 ?번 ?만 ?더|다시 ?한 ?번)")
EASIER_RX = re.compile(r"(쉽게|쉬운 ?말|쉬운 ?표현|풀어서|간단(하게|히)|이해가 ?(안|잘 ?안|잘 ?못|못)|무슨 ?(말|뜻|소리)|뭔 ?(말|뜻|소리)|어려워)")
MAX_LEN = 60       # a clarification request is short; a long answer that merely contains such a word is a real answer

def classify(text: str) -> str | None:
    """'easier' | 'repeat' | None. 'easier' wins when both appear ("더 쉽게 다시 말씀해 주세요")."""
    t = (text or "").strip()
    if not t or len(t) >= MAX_LEN: return None
    if EASIER_RX.search(t): return "easier"
    if is_repeat_request(t) or EXTRA_REPEAT_RX.search(t): return "repeat"
    return None
