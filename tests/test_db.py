import pytest

from support_reply_gap.db import get_connection


@pytest.mark.db
def test_can_connect():
    with get_connection() as conn:
        result = conn.execute("select 1").fetchone()
    assert result == (1,)
