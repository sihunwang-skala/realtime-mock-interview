"""Shared data contracts: runtime events, questions, session state."""
from __future__ import annotations
import time, uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

class Action(str, Enum):
    DEEPEN="DEEPEN"; CHALLENGE="CHALLENGE"; CLARIFY="CLARIFY"; VERIFY="VERIFY"
    SWITCH="SWITCH"; REPEAT="REPEAT"; END="END"

class EventType(str, Enum):
    SESSION_READY="SESSION_READY"; USER_SPEAKING="USER_SPEAKING"; AGENT_SPEAKING="AGENT_SPEAKING"
    AGENT_THINKING="AGENT_THINKING"; QUESTION_STARTED="QUESTION_STARTED"; ANSWER_COMPLETED="ANSWER_COMPLETED"
    AGENT_INTERJECTED="AGENT_INTERJECTED"; SESSION_ENDED="SESSION_ENDED"; REPORT_PROCESSING="REPORT_PROCESSING"
    REPORT_READY="REPORT_READY"; ERROR="ERROR"; LATENCY="LATENCY"

class Cmd(str, Enum):
    START="START"; REPEAT_QUESTION="REPEAT_QUESTION"; SKIP_QUESTION="SKIP_QUESTION"
    FINISH_ANSWER="FINISH_ANSWER"; START_ANSWER="START_ANSWER"; END_SESSION="END_SESSION"

class Status(str, Enum):
    IMPLEMENTED_AND_VERIFIED="IMPLEMENTED_AND_VERIFIED"; IMPLEMENTED_NOT_VERIFIED="IMPLEMENTED_NOT_VERIFIED"
    BLOCKED="BLOCKED"; NOT_IMPLEMENTED="NOT_IMPLEMENTED"; NOT_TESTED="NOT_TESTED"; FAILED="FAILED"

@dataclass
class Event:
    type: EventType
    session_id: str
    seq: int
    ts: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)
    def to_dict(self): d=asdict(self); d["type"]=self.type.value; return d

@dataclass
class Requirement:
    id: str; text: str; weight: int = 1

@dataclass
class Question:
    text: str
    action: Action = Action.DEEPEN
    req_id: str | None = None
    priority: float = 0.5
    source: str = "presentation"   # presentation | answer | seed
    evidence: str = ""             # RFP clause / quote that grounds it
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "pending"        # pending | asked | resolved | discarded

@dataclass
class Assessment:
    sufficiency: float = 0.0       # 0..1
    contradictions: list[str] = field(default_factory=list)
    off_topic: float = 0.0         # 0..1
    repetitive: bool = False
    unsupported_claims: list[str] = field(default_factory=list)
    covered_reqs: list[str] = field(default_factory=list)
    summary: str = ""
