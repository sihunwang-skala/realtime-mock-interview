"""Numbers with units, read correctly: "1,200만원", "32억 5천만 원", "백만원", "120명", "35%".
Money is compared in won and periods in months, so "1,200만원" == "12,000,000원" and "1년" == "12개월"."""
from __future__ import annotations
import re

_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_SCALE = r"(?:조|억|천만|백만|십만|만|천)"
_UNIT = r"(?:퍼센트|%p|%|ms|초|개월|개소|년|개|명|원|일|주|시간|TPS|건|회|곳|차시|점)"
QTY_RX = re.compile(
    rf"(?<![\d,.])(?:(?:{_NUM})\s?{_SCALE}\s?)+(?:(?:{_NUM})\s?)?(?:{_UNIT})?"   # 32억 5천만 원 / 1,200만원 / 3만 명
    rf"|(?<![\d,.])(?:{_NUM})\s?(?:{_UNIT})"                                        # 120명 / 12,000,000원 / 35%
    rf"|(?<![가-힣\d])(?:천만|백만|십만)\s?원")                                      # 백만원
_MULT = {"조": 10**12, "억": 10**8, "천만": 10**7, "백만": 10**6, "십만": 10**5, "만": 10**4, "천": 10**3}

def numbers(text: str) -> set[str]:
    return {m.group(0).strip() for m in QTY_RX.finditer(text or "")}

def parse(q: str) -> tuple[str, float | None]:
    """(unit class, value in the base unit). Scale words without a unit ("32억") are money."""
    s = q.replace(",", "").replace(" ", "")
    m = re.search(rf"({_UNIT})$", s)
    unit = m.group(1) if m else ""; body = s[:m.start()] if m else s
    total, matched, rest = 0.0, False, body
    for num, scale in re.findall(rf"(\d+(?:\.\d+)?)?({_SCALE})", body):
        total += float(num or 1) * _MULT[scale]; matched = True
    rest = re.sub(rf"(?:\d+(?:\.\d+)?)?{_SCALE}", "", body)
    if re.fullmatch(r"\d+(?:\.\d+)?", rest): total += float(rest)
    elif not matched: return unit, None
    if not unit and matched: unit = "원" if re.search(r"조|억", body) else "수"
    if unit == "년": return "개월", total * 12
    if unit in ("퍼센트",): unit = "%"
    return unit, total

def is_calendar_year(q: str) -> bool:
    return bool(re.fullmatch(r"(19|20)\d{2}\s?년", q.strip()))
