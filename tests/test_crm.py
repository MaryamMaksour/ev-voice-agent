from datetime import datetime, timedelta, timezone

import pytest

from crm.port import CRMClient, LeadNotFoundError, LeadStatus, LeadUpdate


def test_adapter_implements_port(crm):
    assert isinstance(crm, CRMClient)


async def test_get_seeded_lead(crm):
    lead = await crm.get_lead("L001")
    assert lead.name == "Sarah Johnson"
    assert lead.status == LeadStatus.NEW


async def test_unknown_lead_raises(crm):
    with pytest.raises(LeadNotFoundError):
        await crm.get_lead("NOPE")


async def test_update_only_touches_given_fields(crm):
    await crm.update_lead("L001", LeadUpdate(budget=800000, preferred_area="Marina"))
    await crm.update_lead("L001", LeadUpdate(property_type="apartment"))
    lead = await crm.get_lead("L001")
    assert (lead.budget, lead.preferred_area, lead.property_type.value) == (800000, "Marina", "apartment")
    assert lead.updated_at is not None


async def test_callback_and_dnc(crm):
    when = datetime.now(timezone.utc) + timedelta(days=1)
    lead = await crm.schedule_callback("L002", when)
    assert lead.status == LeadStatus.CALLBACK and lead.callback_at == when

    lead = await crm.mark_do_not_call("L003")
    assert lead.do_not_call and lead.status == LeadStatus.DO_NOT_CALL


async def test_notes_append(crm):
    await crm.add_note("L004", "first")
    lead = await crm.add_note("L004", "second")
    assert lead.notes == ["first", "second"]


def test_lead_update_validation():
    with pytest.raises(ValueError):
        LeadUpdate(budget=-5)
    with pytest.raises(ValueError):
        LeadUpdate(property_type="castle")
    assert LeadUpdate(preferred_area="  ").changes() == {}
