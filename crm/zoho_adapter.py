"""Zoho CRM adapter (REST API v6, Leads module).

Enabled with CRM_BACKEND=zoho. Test against a Zoho *sandbox* first.

Custom fields (Budget, Preferred_Area, ...) must exist in the Leads layout; their API
names can be overridden with ZOHO_FIELD_<NAME>, e.g. ZOHO_FIELD_BUDGET=Budget_USD.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import httpx

from .port import (
    InterestLevel,
    Lead,
    LeadNotFoundError,
    LeadStatus,
    LeadUpdate,
    PropertyType,
)

# our status -> Zoho's default "Lead_Status" picklist values
STATUS_TO_ZOHO = {
    LeadStatus.NEW: "Not Contacted",
    LeadStatus.CONTACTED: "Contacted",
    LeadStatus.INTERESTED: "Pre-Qualified",
    LeadStatus.QUALIFIED: "Pre-Qualified",
    LeadStatus.NOT_INTERESTED: "Lost Lead",
    LeadStatus.CALLBACK: "Contact in Future",
    LeadStatus.DO_NOT_CALL: "Not Qualified",
    LeadStatus.WRONG_NUMBER: "Junk Lead",
}
ZOHO_TO_STATUS = {
    "Not Contacted": LeadStatus.NEW,
    "Attempted to Contact": LeadStatus.CONTACTED,
    "Contacted": LeadStatus.CONTACTED,
    "Pre-Qualified": LeadStatus.QUALIFIED,
    "Lost Lead": LeadStatus.NOT_INTERESTED,
    "Contact in Future": LeadStatus.CALLBACK,
    "Not Qualified": LeadStatus.DO_NOT_CALL,
    "Junk Lead": LeadStatus.WRONG_NUMBER,
}

_DEFAULT_FIELDS = {
    "budget": "Budget",
    "preferred_area": "Preferred_Area",
    "property_type": "Property_Type",
    "timeline": "Timeline",
    "interest_level": "Interest_Level",
    "callback_at": "Callback_Time",
    "do_not_call": "Do_Not_Call",
    "project_interest": "Project_Interest",
}


def _field(name: str) -> str:
    return os.getenv(f"ZOHO_FIELD_{name.upper()}", _DEFAULT_FIELDS[name])


class ZohoCRM:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        dc: str = "com",
        sandbox: bool = True,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._accounts_url = f"https://accounts.zoho.{dc}"
        host = "sandbox.zohoapis" if sandbox else "www.zohoapis"
        self._api = f"https://{host}.{dc}/crm/v6"
        self._token: str | None = None
        self._token_expiry = 0.0
        self._http = httpx.AsyncClient(timeout=15)

    @classmethod
    def from_env(cls) -> "ZohoCRM":
        return cls(
            client_id=os.environ["ZOHO_CLIENT_ID"],
            client_secret=os.environ["ZOHO_CLIENT_SECRET"],
            refresh_token=os.environ["ZOHO_REFRESH_TOKEN"],
            dc=os.getenv("ZOHO_DC", "com"),
            sandbox=os.getenv("ZOHO_SANDBOX", "true").lower() == "true",
        )

    async def _auth_header(self) -> dict[str, str]:
        if self._token is None or time.time() > self._token_expiry - 60:
            resp = await self._http.post(
                f"{self._accounts_url}/oauth/v2/token",
                params={
                    "refresh_token": self._refresh_token,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "grant_type": "refresh_token",
                },
            )
            resp.raise_for_status()
            body = resp.json()
            self._token = body["access_token"]
            self._token_expiry = time.time() + int(body.get("expires_in", 3600))
        return {"Authorization": f"Zoho-oauthtoken {self._token}"}

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        resp = await self._http.request(
            method, f"{self._api}{path}", headers=await self._auth_header(), **kwargs
        )
        if resp.status_code in (204, 404):
            raise LeadNotFoundError(path)
        resp.raise_for_status()
        return resp.json()

    def _to_lead(self, rec: dict) -> Lead:
        def enum_or_none(enum, value):
            try:
                return enum(str(value).lower()) if value else None
            except ValueError:
                return None

        name = rec.get("Full_Name") or " ".join(
            p for p in (rec.get("First_Name"), rec.get("Last_Name")) if p
        )
        return Lead(
            id=str(rec["id"]),
            name=name or "there",
            phone=rec.get("Phone") or rec.get("Mobile") or "",
            email=rec.get("Email"),
            project_interest=rec.get(_field("project_interest")),
            status=ZOHO_TO_STATUS.get(rec.get("Lead_Status") or "", LeadStatus.NEW),
            budget=rec.get(_field("budget")),
            preferred_area=rec.get(_field("preferred_area")),
            property_type=enum_or_none(PropertyType, rec.get(_field("property_type"))),
            timeline=rec.get(_field("timeline")),
            interest_level=enum_or_none(InterestLevel, rec.get(_field("interest_level"))),
            callback_at=rec.get(_field("callback_at")),
            do_not_call=bool(rec.get(_field("do_not_call"))),
        )

    async def _update(self, lead_id: str, fields: dict) -> Lead:
        await self._request("PUT", f"/Leads/{lead_id}", json={"data": [fields]})
        return await self.get_lead(lead_id)

    async def get_lead(self, lead_id: str) -> Lead:
        body = await self._request("GET", f"/Leads/{lead_id}")
        return self._to_lead(body["data"][0])

    async def update_lead(self, lead_id: str, update: LeadUpdate) -> Lead:
        fields: dict = {}
        for key, value in update.changes().items():
            if key == "status":
                fields["Lead_Status"] = STATUS_TO_ZOHO[LeadStatus(value)]
            else:
                fields[_field(key)] = value
        return await self._update(lead_id, fields)

    async def schedule_callback(self, lead_id: str, when: datetime) -> Lead:
        return await self._update(
            lead_id,
            {
                _field("callback_at"): when.isoformat(timespec="seconds"),
                "Lead_Status": STATUS_TO_ZOHO[LeadStatus.CALLBACK],
            },
        )

    async def mark_do_not_call(self, lead_id: str) -> Lead:
        return await self._update(
            lead_id,
            {_field("do_not_call"): True, "Lead_Status": STATUS_TO_ZOHO[LeadStatus.DO_NOT_CALL]},
        )

    async def add_note(self, lead_id: str, note: str) -> Lead:
        await self._request(
            "POST",
            f"/Leads/{lead_id}/Notes",
            json={"data": [{"Note_Title": "AI voice agent call", "Note_Content": note}]},
        )
        return await self.get_lead(lead_id)
