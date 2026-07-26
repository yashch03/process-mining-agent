"""
Pydantic contracts for BPI Challenge 2017 event records.
Strict schema enforcement at the ingestion boundary — malformed records
are quarantined, not silently coerced or dropped.
"""
from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Optional


class EventRecord(BaseModel):
    case_id: str
    activity: str
    timestamp: datetime
    resource: Optional[str] = None

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_plausible(cls, v: datetime) -> datetime:
        if v.year < 2000 or v.year > 2030:
            raise ValueError(f"timestamp out of plausible range: {v}")
        return v

    @field_validator("case_id", "activity")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("case_id/activity cannot be empty")
        return v


def validate_record(raw: dict) -> tuple[Optional[EventRecord], Optional[str]]:
    """
    Attempt to validate one raw event dict.
    Returns (EventRecord, None) on success, or (None, error_message) on failure.
    Never raises — callers use this to route to quarantine.
    """
    try:
        return EventRecord(**raw), None
    except Exception as e:
        return None, str(e)
