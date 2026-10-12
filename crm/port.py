"""CRM port: the only interface the agent knows about.

Adapters (SQLite for the demo, Zoho for production) implement `CRMClient`.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator


class LeadStatus(str, Enum):
    NEW = "new"
    CONTACTED = "contacted"
    INTERESTED = "interested"
    QUALIFIED = "qualified"
    NOT_INTERESTED = "not_interested"
    CALLBACK = "callback"
    DO_NOT_CALL = "do_not_call"
    WRONG_NUMBER = "wrong_number"


class InterestLevel(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class PropertyType(str, Enum):
    APARTMENT = "apartment"
    VILLA = "villa"
    TOWNHOUSE = "townhouse"
    OFFICE = "office"
    LAND = "land"


class Lead(BaseModel):
    id: str
    name: str
    phone: str
    email: str | None = None
    project_interest: str | None = None
    status: LeadStatus = LeadStatus.NEW
    budget: int | None = None
    preferred_area: str | None = None
    property_type: PropertyType | None = None
    timeline: str | None = None
    interest_level: InterestLevel | None = None
    callback_at: datetime | None = None
    do_not_call: bool = False
    notes: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None


class LeadUpdate(BaseModel):
    """Fields the agent is allowed to change. Unset fields are left untouched."""

    status: LeadStatus | None = None
    budget: int | None = Field(default=None, ge=0)
    preferred_area: str | None = None
    property_type: PropertyType | None = None
    timeline: str | None = None
    interest_level: InterestLevel | None = None

    @field_validator("preferred_area", "timeline")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None

    def changes(self) -> dict:
        return self.model_dump(exclude_none=True, mode="json")


class LeadNotFoundError(LookupError):
    pass


@runtime_checkable
class CRMClient(Protocol):
    async def get_lead(self, lead_id: str) -> Lead: ...

    async def update_lead(self, lead_id: str, update: LeadUpdate) -> Lead: ...

    async def schedule_callback(self, lead_id: str, when: datetime) -> Lead: ...

    async def mark_do_not_call(self, lead_id: str) -> Lead: ...

    async def add_note(self, lead_id: str, note: str) -> Lead: ...
