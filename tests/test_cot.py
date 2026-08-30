"""Slice 5 RED: CFTC COT positioning with release-date semantics.

CFTC Traders in Financial Futures: positions snapshot TUESDAY, released
FRIDAY 3:30pm ET. A value may only appear on session rows dated >= release date.
"""
import pandas as pd
import pytest

from nq_research.ingest.cot import (
    add_release_dates,
    cot_asof,
    net_position_percentile,
)


def _raw_reports():
    """(tuesday_snapshot, lev net, am net, dealer net) — weekly."""
    return pd.DataFrame([
        ("2025-01-07", 50_000, -30_000, 10_000),
        ("2025-01-14", -20_000, 10_000, -5_000),
        ("2025-01-21", 80_000, 40_000, 20_000),   # released Fri Jan 24
        ("2025-01-28", -60_000, -40_000, 0),
    ], columns=["tuesday", "lev_net", "am_net", "dealer_net"])


def test_release_date_is_friday_after_snapshot():
    out = add_release_dates(_raw_reports())
    got = [d.strftime("%Y-%m-%d") for d in out["release"]]
    assert got == ["2025-01-10", "2025-01-17", "2025-01-24", "2025-01-31"]


def test_cot_asof_uses_release_not_snapshot():
    out = add_release_dates(_raw_reports())
    # Wed Jan 22: report covering Tue Jan 21 is NOT out yet (released Fri Jan 24)
    val = cot_asof(out, asof="2025-01-22", col="lev_net")
    assert val == -20_000                      # Jan 14 snapshot's report, released Jan 17
    # Fri Jan 24 close: new report visible
    val2 = cot_asof(out, asof="2025-01-24", col="lev_net")
    assert val2 == 80_000


def test_no_lookahead_injection_into_sessions():
    out = add_release_dates(_raw_reports())
    sessions = pd.DataFrame({"date": pd.to_datetime([
        "2025-01-21", "2025-01-22", "2025-01-23", "2025-01-24", "2025-01-27"
    ])})
    joined = join_cot_to_sessions(sessions, out)
    # sessions through Thu Jan 23 must still carry the Jan 14 report; Jan 24 carries the Jan 21 one
    assert (joined.loc[joined.date == "2025-01-23", "lev_net"] == -20_000).all()
    assert (joined.loc[joined.date == "2025-01-24", "lev_net"] == 80_000).all()


def test_net_position_percentile_trailing_3y():
    out = add_release_dates(_raw_reports())
    p = net_position_percentile(out, asof="2025-01-24", col="lev_net", window_years=3)
    # vs trailing history {50k, -20k}: 80k is the max -> 100th percentile
    assert p == pytest.approx(100.0)


def test_join_cot_to_sessions_no_lookahead():
    out = add_release_dates(_raw_reports())
    sessions = pd.DataFrame({"date": pd.to_datetime([
        "2025-01-10", "2025-01-13", "2025-01-20", "2025-01-21"
    ])})
    joined = join_cot_to_sessions(sessions, out)
    # Fri Jan 10 is release day itself: report IS available 3:30pm ET; conservative rule -> available same day
    assert (joined.loc[joined.date == "2025-01-10", "lev_net"] == 50_000).all()


from nq_research.ingest.cot import join_cot_to_sessions  # noqa: E402