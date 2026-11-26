"""Amounts must be read correctly: they are the core of a proposal."""
from qa_lab.core.quantities import numbers, parse

def test_comma_amount_is_not_truncated():
    assert numbers("총사업비는 1,200만원입니다") == {"1,200만원"}
    assert parse("1,200만원") == ("원", 12_000_000)

def test_same_money_in_different_writing_is_equal():
    assert parse("1,200만원") == parse("12,000,000원") == parse("1200만 원")
    assert parse("32억 5천만 원") == ("원", 3_250_000_000)
    assert parse("백만원") == ("원", 1_000_000)
    assert parse("32억") == ("원", 3_200_000_000)

def test_counts_periods_and_ratios():
    assert parse("3만 명") == ("명", 30000)
    assert parse("1년") == parse("12개월") == ("개월", 12)
    assert parse("35%") == parse("35퍼센트") == ("%", 35)
    assert numbers("교육생 120명, 기간 6개월") == {"120명", "6개월"}
