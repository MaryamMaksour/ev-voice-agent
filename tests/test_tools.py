from datetime import datetime, timedelta

from agent.tools import LeadTools


async def test_update_lead_saves_and_logs(crm):
    tools = LeadTools(crm, "L001")
    msg = await tools.update_lead(budget=750000, property_type="Villa", interest_level="HOT")
    assert msg.startswith("Saved")
    lead = await crm.get_lead("L001")
    assert lead.budget == 750000 and lead.property_type.value == "villa"
    assert tools.actions[0]["action"] == "update_lead"


async def test_invalid_values_are_reported_not_saved(crm):
    tools = LeadTools(crm, "L001")
    msg = await tools.update_lead(property_type="castle")
    assert msg.startswith("Not saved")
    assert (await crm.get_lead("L001")).property_type is None
    assert tools.actions == []


async def test_empty_update(crm):
    assert await LeadTools(crm, "L001").update_lead() == "Nothing to save."


async def test_schedule_callback(crm):
    tools = LeadTools(crm, "L002", tz="UTC")
    when = (datetime.now() + timedelta(days=2)).replace(microsecond=0).isoformat()
    assert (await tools.schedule_callback(when)).startswith("Callback booked")
    assert (await crm.get_lead("L002")).status.value == "callback"


async def test_schedule_callback_rejects_bad_input(crm):
    tools = LeadTools(crm, "L002")
    assert "ISO 8601" in await tools.schedule_callback("thursday 4pm")
    past = (datetime.now() - timedelta(days=1)).isoformat()
    assert "past" in await tools.schedule_callback(past)
    assert (await crm.get_lead("L002")).status.value == "new"


async def test_tools_are_bound_to_one_lead(crm):
    tools = LeadTools(crm, "L005")
    await tools.mark_do_not_call()
    assert (await crm.get_lead("L005")).do_not_call
    assert not (await crm.get_lead("L001")).do_not_call
