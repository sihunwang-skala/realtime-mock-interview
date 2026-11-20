"""Synthetic public-style RFP (no real company data) used for every case/benchmark."""
from ..contracts import Requirement
TITLE = "스마트시티 통합 민원 플랫폼 구축 사업 (합성 RFP)"
REQS = [
    Requirement("R1", "민원 접수 API 평균 응답시간은 2초 이내이어야 한다.", 3),
    Requirement("R2", "동시 접속자 5000명을 처리할 수 있어야 하며 가용성은 99.9% 이상이어야 한다.", 3),
    Requirement("R3", "개인정보는 암호화하여 저장하고 접근 로그를 1년 이상 보관해야 한다.", 3),
    Requirement("R4", "구축 기간은 계약 후 8개월 이내이어야 한다.", 2),
    Requirement("R5", "민원 분류에 AI를 적용하고 분류 정확도는 90% 이상이어야 한다.", 2),
    Requirement("R6", "기존 행정 시스템과 표준 API로 연계해야 한다.", 1),
]
RFP_TEXT = "\n".join(f"{r.id}. {r.text}" for r in REQS)

SEED_BANK = [
    "민원 접수 API의 평균 응답시간 2초 이내 목표를 어떤 구조로 달성하실 계획인가요?",
    "동시 접속자 5000명과 가용성 99.9퍼센트를 어떻게 검증하실 건가요?",
    "개인정보 암호화와 접근 로그 1년 보관은 어떻게 구현하시나요?",
    "구축 기간 8개월 안에 완료하기 위한 일정과 인력 계획을 말씀해 주세요.",
    "AI 민원 분류의 정확도 90퍼센트는 어떤 데이터로 어떻게 검증하실 계획인가요?",
]
