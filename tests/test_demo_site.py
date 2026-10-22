"""The demo website: lead list, starting a call, reading the saved record, resetting a lead."""

import json

import pytest
from fastapi.testclient import TestClient

from crm.port import LeadStatus, LeadUpdate
from crm.sqlite_adapter import SQLiteCRM
from demo import server

ROOM = "lead-L003-20261022120000"


@pytest.fixture
def crm(tmp_path):
    crm = SQLiteCRM(tmp_path / "crm.db")
    crm.seed()
    return crm


@pytest.fixture
def client(crm, tmp_path):
    (tmp_path / "calls").mkdir()
    app = server.create_app(crm=crm, calls_dir=str(tmp_path / "calls"), livekit_url="ws://localhost:7880",
                            api_key="devkey", api_secret="a-long-secret-for-the-test-0123456789")
    return TestClient(app)


def test_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Start call" in r.text


def test_lists_the_twenty_demo_leads(client):
    leads = client.get("/api/leads").json()
    assert len(leads) == 20
    assert leads[2]["id"] == "L003" and leads[2]["name"] == "Emily Davis"


def test_start_call_returns_room_token_and_url(client):
    r = client.post("/api/call", json={"lead_id": "L003"})
    assert r.status_code == 200
    body = r.json()
    assert server.ROOM.match(body["room"]) and body["room"].startswith("lead-L003-")
    assert body["url"] == "ws://localhost:7880"
    assert body["token"].count(".") == 2  # a JWT


def test_start_call_for_unknown_lead_is_404(client):
    assert client.post("/api/call", json={"lead_id": "L999"}).status_code == 404


def test_call_result_is_not_done_before_the_record_exists(client):
    assert client.get(f"/api/call/{ROOM}").json() == {"done": False}


def test_call_result_ignores_a_raw_file_that_is_still_being_processed(client, tmp_path):
    (tmp_path / "calls" / f"{ROOM}-2026-10-22T120000.raw.json").write_text("{}")
    assert client.get(f"/api/call/{ROOM}").json() == {"done": False}


def test_call_result_reads_the_saved_record(client, tmp_path):
    record = {"classification": "qualified", "ended_by": "agent",
              "extraction": {"summary": "Wants a villa."},
              "transcript": [{"role": "agent", "text": "Hi"}, {"role": "user", "text": "Hello"}]}
    (tmp_path / "calls" / f"{ROOM}-2026-10-22T120000.json").write_text(json.dumps(record))
    body = client.get(f"/api/call/{ROOM}").json()
    assert body["done"] is True
    assert body["classification"] == "qualified"
    assert body["summary"] == "Wants a villa."
    assert len(body["transcript"]) == 2


def test_call_result_refuses_names_that_are_not_call_rooms(client):
    assert client.get("/api/call/..%2Fsecret").status_code in (400, 404)
    assert client.get("/api/call/lead-L003-*").status_code == 400


@pytest.mark.asyncio
async def test_reset_puts_the_lead_back(client, crm):
    await crm.update_lead("L003", LeadUpdate(status=LeadStatus.QUALIFIED, budget=500000))
    await crm.add_note("L003", "a note")
    r = client.post("/api/reset", json={"lead_id": "L003"})
    assert r.status_code == 200
    lead = await crm.get_lead("L003")
    assert lead.status == LeadStatus.NEW and lead.budget is None and lead.notes == []


def test_reset_unknown_lead_is_404(client):
    assert client.post("/api/reset", json={"lead_id": "L999"}).status_code == 404


def test_refuses_to_start_with_a_real_crm(monkeypatch):
    monkeypatch.setenv("CRM_BACKEND", "zoho")
    with pytest.raises(SystemExit):
        server.main()
