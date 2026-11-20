"""Spoken form of English words inside Korean sentences, for the local voices.
A Korean voice that meets "Retry" or "Evaluation" switches to an English-like pronunciation mid-sentence, which sounds like a different speaker.
So English words are replaced by their Hangul reading before synthesis (the text shown on screen is NOT changed).
Known words come from GLOSSARY; unknown ALL-CAPS words are spelled letter by letter; other unknown words are left as they are."""
from __future__ import annotations
import re

GLOSSARY = {w.lower(): k for w, k in {
 "Evaluation": "이밸류에이션", "Evaluations": "이밸류에이션", "Evaluate": "이밸류에이트", "Retry": "리트라이", "Retries": "리트라이", "Checkpoint": "체크포인트", "Checkpoints": "체크포인트", "Gate": "게이트",
 "Accuracy": "애큐러시", "Benchmark": "벤치마크", "Benchmarks": "벤치마크", "Timeout": "타임아웃", "Time": "타임", "out": "아웃", "Rate": "레이트", "Task": "태스크", "Tasks": "태스크", "Success": "석세스",
 "Quality": "퀄리티", "Guardrail": "가드레일", "Guardrails": "가드레일", "Reducer": "리듀서", "Runtime": "런타임", "Safety": "세이프티", "Component": "컴포넌트", "Components": "컴포넌트", "Design": "디자인",
 "Tool": "툴", "Tools": "툴", "Input": "인풋", "Output": "아웃풋", "Schema": "스키마", "State": "스테이트", "Agent": "에이전트", "Agents": "에이전트", "Workflow": "워크플로", "Workflows": "워크플로",
 "Python": "파이썬", "LangGraph": "랭그래프", "LangChain": "랭체인", "Reliability": "릴라이어빌리티", "Trade-off": "트레이드오프", "Tradeoff": "트레이드오프", "Architecture": "아키텍처",
 "Interrupt": "인터럽트", "Resume": "리줌", "Prompt": "프롬프트", "Token": "토큰", "Tokens": "토큰", "Cost": "코스트", "Latency": "레이턴시", "Throughput": "스루풋", "Parallel": "패러렐", "Send": "센드",
 "Node": "노드", "Nodes": "노드", "Graph": "그래프", "Memory": "메모리", "Retrieval": "리트리벌", "Pydantic": "파이댄틱", "Circuit": "서킷", "Breaker": "브레이커", "Fallback": "폴백", "Budget": "버짓",
 "Deploy": "디플로이", "Canary": "카나리", "Smoke": "스모크", "Rollout": "롤아웃", "Rollback": "롤백", "Pipeline": "파이프라인", "Test": "테스트", "Tests": "테스트", "Monitoring": "모니터링",
 "Dashboard": "대시보드", "Metric": "메트릭", "Metrics": "메트릭", "Log": "로그", "Logs": "로그", "Logging": "로깅", "Trace": "트레이스", "Tracing": "트레이싱", "Cache": "캐시", "Queue": "큐",
 "Server": "서버", "Client": "클라이언트", "Request": "리퀘스트", "Response": "리스펀스", "Error": "에러", "Exception": "익셉션", "Policy": "폴리시", "Permission": "퍼미션", "Human": "휴먼", "Loop": "루프",
 "Level": "레벨", "Model": "모델", "Context": "컨텍스트", "Plan": "플랜", "Planner": "플래너", "Executor": "이그제큐터", "Supervisor": "슈퍼바이저", "Router": "라우터", "Multi": "멀티", "Single": "싱글",
 "Reflection": "리플렉션", "Feedback": "피드백", "Validation": "밸리데이션", "Verification": "베리피케이션", "Guard": "가드", "Config": "컨피그", "Thread": "스레드", "Key": "키", "Id": "아이디",
 "Reasoning": "리즈닝", "Autonomy": "오토노미", "Autonomous": "오토노머스", "Production": "프로덕션", "Staging": "스테이징", "Release": "릴리스", "Version": "버전", "Data": "데이터", "Database": "데이터베이스",
 "API": "에이피아이", "LLM": "엘엘엠", "RAG": "래그", "JSON": "제이슨", "HITL": "에이치아이티엘", "CI": "씨아이", "CD": "씨디", "SQL": "에스큐엘", "HTTP": "에이치티티피", "URL": "유알엘", "UI": "유아이",
 "SDK": "에스디케이", "MCP": "엠씨피", "GPU": "지피유", "CPU": "씨피유", "RFP": "알에프피", "TPS": "티피에스", "AES": "에이이에스", "HSM": "에이치에스엠", "OCR": "오씨알", "STT": "에스티티", "TTS": "티티에스",
 "AI": "에이아이", "ML": "엠엘", "GPT": "지피티", "YAML": "야믈", "CRUD": "크러드", "SLA": "에스엘에이", "SLO": "에스엘오", "KPI": "케이피아이", "ID": "아이디", "PR": "피알", "AWS": "에이더블유에스",
}.items()}
_LET = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "에이 비 씨 디 이 에프 지 에이치 아이 제이 케이 엘 엠 엔 오 피 큐 알 에스 티 유 브이 더블유 엑스 와이 제트".split()))
_NUM = {"0": "영", "1": "원", "2": "투", "3": "쓰리", "4": "포", "5": "파이브", "6": "식스", "7": "세븐", "8": "에이트", "9": "나인"}

def _word(w: str) -> str:
    k = w.lower()
    if k in GLOSSARY: return GLOSSARY[k]
    m = re.fullmatch(r"([A-Z])(\d)", w)                                   # L1, L2 ...
    if m: return _LET[m.group(1)] + _NUM[m.group(2)]
    if w.isupper() and 2 <= len(w) <= 6: return "".join(_LET.get(c, c) for c in w)       # unknown acronym: letter by letter
    parts = re.findall(r"[A-Z][a-z]+|[a-z]+", w)                          # CamelCase: split and look each part up
    if len(parts) > 1 and all(p.lower() in GLOSSARY for p in parts): return "".join(GLOSSARY[p.lower()] for p in parts)
    return w

_JOSA = {"을": ("을", "를"), "를": ("을", "를"), "은": ("은", "는"), "는": ("은", "는"), "과": ("과", "와"), "와": ("과", "와"), "이": ("이", "가"), "가": ("이", "가")}
def _fix_josa(word: str, josa: str) -> str:
    """The particle after a converted word must match its last syllable (벤치마크을 -> 벤치마크를)."""
    last = word[-1] if word else ""
    if not ("가" <= last <= "힣") or josa not in _JOSA: return josa
    withb, without = _JOSA[josa]; return withb if (ord(last) - 0xAC00) % 28 else without

def to_spoken(text: str) -> str:
    """Hangul reading of the English words in text. Korean and digits are untouched."""
    t = re.sub(r"(?<=[A-Za-z])_(?=[A-Za-z])", " ", text)                  # thread_id -> thread id
    t = re.sub(r"(?<=[A-Za-z])\(\)", "", t)                                # interrupt() -> interrupt
    def sub(m):
        w = m.group(1); out = _word(w.rstrip("-")) + ("-" if w.endswith("-") else "")
        j = m.group(2)
        return out + (_fix_josa(out, j) if j and out != w else (j or ""))
    return re.sub(r"([A-Za-z][A-Za-z0-9\-]*)((?:을|를|은|는|과|와|이|가)(?![가-힣]))?", sub, t)
