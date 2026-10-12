import pytest

from crm.sqlite_adapter import SQLiteCRM


@pytest.fixture
def crm(tmp_path):
    db = SQLiteCRM(tmp_path / "crm.db")
    db.seed()
    return db
