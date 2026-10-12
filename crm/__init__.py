import os

from .port import CRMClient, Lead, LeadNotFoundError, LeadUpdate


def make_crm() -> CRMClient:
    """Pick the CRM adapter from CRM_BACKEND (sqlite | zoho)."""
    backend = os.getenv("CRM_BACKEND", "sqlite").lower()
    if backend == "zoho":
        from .zoho_adapter import ZohoCRM

        return ZohoCRM.from_env()
    from .sqlite_adapter import SQLiteCRM

    return SQLiteCRM(os.getenv("CRM_SQLITE_PATH", "data/crm.db"))


__all__ = ["CRMClient", "Lead", "LeadNotFoundError", "LeadUpdate", "make_crm"]
