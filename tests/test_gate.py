"""v0.2 Fix 2 — completeness gate tests. Need the seeded local Postgres."""
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench.config import pg_url
from bench.engine_arm2 import completeness_gate

# Same Postgres the rest of the bench uses ($PG_URL, default docker :5544) —
# so this gate test isn't silently skipped for someone following the README.
URL = pg_url()


def _pg_up():
    try:
        from sqlalchemy import create_engine, text
        with create_engine(URL).connect() as c:
            c.execute(text("select 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _pg_up(), reason="local Postgres is down")


def test_gate_fires_on_missing_day():
    # case_25 / case_27: one whole day absent from p1
    for cid in ("case_25", "case_27"):
        g = completeness_gate(URL, cid)
        assert g and g["abstain_reason"] == "data_gap"
        assert "day" in g["details"]["reason"]


def test_gate_fires_on_missing_segment_source():
    # case_26: country=Sundara has ZERO p1 rows
    g = completeness_gate(URL, "case_26")
    assert g and g["abstain_reason"] == "data_gap"
    assert "Sundara" in g["details"]["reason"]


def test_gate_fires_on_mid_period_truncation():
    # case_28: grocery rows stop after day 3 of p1
    g = completeness_gate(URL, "case_28")
    assert g and g["abstain_reason"] == "data_gap"
    assert "grocery" in g["details"]["reason"]


def test_gate_is_silent_on_complete_data():
    # real demand moves, NULL-segment collapses and noise must NOT trigger it
    for cid in ("case_01",   # -60% volume in one country (still alive daily)
                "case_05",   # uniform -15% orders
                "case_13",   # NULL-geo block collapse (mapped data complete)
                "case_21"):  # noise
        assert completeness_gate(URL, cid) is None, cid
