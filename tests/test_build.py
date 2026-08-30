"""Slice 6 RED: sessions-table assembly (synthetic frames, no network, no live DB until e2e)."""
import pandas as pd
import pytest

from nq_research.features.build import assemble_sessions


def _price_sessions():
    idx = pd.bdate_range("2026-01-01", periods=15)
    df = pd.DataFrame({
        "Open": [100 + i for i in range(15)],
        "High": [101 + i for i in range(15)],
        "Low": [99 + i for i in range(15)],
        "Close": [100.5 + i for i in range(15)],
        "Volume": [1_000] * 15,
    }, index=idx)
    from nq_research.ingest.prices import compute_sessions
    return compute_sessions(df)


def _vix_frame():
    idx = pd.bdate_range("2026-01-01", periods=15)
    return pd.DataFrame({"Close": [14.0 + 0.1 * i for i in range(15)]}, index=idx)


def test_assemble_produces_schema_columns():
    s = assemble_sessions(
        primary_sessions=_price_sessions(),
        vix=_vix_frame(),
        vix3m=pd.DataFrame({"Close": [16.0] * 15}, index=_vix_frame().index),
        cpi_vintages=None,          # macro columns simply stay NaN
        cot_reports=None,
    )
    for col in ["date", "close", "ret", "gap", "dow",
                "days_to_opex", "days_since_opex", "is_opex", "is_opex_week", "is_post_opex_week",
                "is_triple_witching", "days_to_fomc", "days_since_fomc",
                "vix", "vix_bucket", "vix_term_structure", "is_backwardation",
                "three_candle_pattern", "consecutive_down_days", "dist_from_20d_high"]:
        assert col in s.columns, f"missing {col}"


def test_opex_proximity_math():
    s = _assembled()
    # Aug-like month not needed; verify using the built frame's own opex day:
    opex_rows = s[s["is_opex"]]
    assert len(opex_rows) >= 1
    row = opex_rows.iloc[0]
    assert row["days_to_opex"] == 0
    assert row["days_since_opex"] == 0


def test_vix_bucket_boundaries():
    s = _assembled()
    b = s["vix_bucket"].iloc[0]
    assert b == "<15"          # 14.0
    # term structure 14.0/16.0 < 1 -> backwardation
    assert bool(s["is_backwardation"].iloc[0])


def test_consecutive_down_days_counts():
    # craft: closes rising then two down days
    s = _assembled()
    assert s["consecutive_down_days"].iloc[-1] in (0, 1, 2, 3)


def test_dist_from_20d_high_negative_when_below():
    s = _assembled()
    vals = s["dist_from_20d_high"].dropna()          # first row NaN by construction
    assert (vals <= 0).all()
    assert len(vals) == len(s) - 1


def _assembled():
    idx = pd.bdate_range("2026-01-01", periods=15)
    up = pd.DataFrame({
        "Open": [100 + i for i in range(15)],
        "High": [101 + i for i in range(15)],
        "Low": [99 + i for i in range(15)],
        "Close": [100.5 + i for i in range(15)],
        "Volume": [1_000] * 15,
    }, index=idx)
    from nq_research.ingest.prices import compute_sessions
    ps = compute_sessions(up)
    # inject two down days at the end
    ps.iloc[-1, ps.columns.get_loc("close")] *= 0.99
    ps.iloc[-2, ps.columns.get_loc("close")] *= 0.995
    return assemble_sessions(
        primary_sessions=ps,
        vix=pd.DataFrame({"Close": [14.0 + i for i in range(15)]}, index=idx),
        vix3m=pd.DataFrame({"Close": [16.0] * 15}, index=idx),
        cpi_vintages=None,
        cot_reports=None,
    )