"""Zoho adapter request shapes, against a mocked HTTP transport (no Zoho account needed)."""

import json
from datetime import datetime, timezone

import httpx

from crm.port import LeadUpdate
from crm.zoho_adapter import ZohoCRM

RECORD = {"id": "123", "Full_Name": "Sarah Johnson", "Phone": "+1555", "Lead_Status": "Not Contacted",
          "Project_Interest": "Marina Heights", "Budget": None}


def make_crm(log):
    def handler(request: httpx.Request) -> httpx.Response:
        log.append(request)
        if request.url.path == "/oauth/v2/token":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        if request.method == "GET":
            return httpx.Response(200, json={"data": [RECORD]})
        return httpx.Response(200, json={"data": [{"code": "SUCCESS"}]})

    crm = ZohoCRM("id", "secret", "refresh", dc="com", sandbox=True)
    crm._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return crm


async def test_get_and_update_lead():
    log = []
    crm = make_crm(log)
    lead = await crm.get_lead("123")
    assert lead.name == "Sarah Johnson" and lead.status.value == "new"
    assert log[1].headers["Authorization"] == "Zoho-oauthtoken tok"
    assert log[1].url.host == "sandbox.zohoapis.com"

    await crm.update_lead("123", LeadUpdate(status="qualified", budget=800000))
    put = next(r for r in log if r.method == "PUT")
    assert json.loads(put.content) == {"data": [{"Lead_Status": "Pre-Qualified", "Budget": 800000}]}
    assert sum(r.url.path == "/oauth/v2/token" for r in log) == 1  # token is cached


async def test_callback_dnc_and_note():
    log = []
    crm = make_crm(log)
    await crm.schedule_callback("123", datetime(2026, 10, 15, 16, 0, tzinfo=timezone.utc))
    await crm.mark_do_not_call("123")
    await crm.add_note("123", "Interested in a 2-bed.")
    bodies = [json.loads(r.content) for r in log if r.method in ("PUT", "POST") and r.content]
    assert bodies[0]["data"][0] == {"Callback_Time": "2026-10-15T16:00:00+00:00", "Lead_Status": "Contact in Future"}
    assert bodies[1]["data"][0]["Do_Not_Call"] is True
    assert bodies[2]["data"][0]["Note_Content"] == "Interested in a 2-bed."
