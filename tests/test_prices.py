"""Slice 3 RED: price ingest & session-frame construction.

Unit tests use injected fake OHLC data (no network). Live yfinance pull happens
in the e2e ingest step.
"""
import pandas as pd
import pytest

from nq_research.ingest.prices import (
    compute_sessions,
    load_parquet,
    save_parquet,
    validate_continuity,
)


def _ndx_like():
    idx = pd.bdate_range("2026-01-01", periods=6)
    return pd.DataFrame(
        {
            "Open":  [100.0, 101.0, 99.0, 100.5, 102.0, 101.0],
            "High":  [101.5, 102.0, 101.0, 103.0, 103.5, 102.5],
            "Low":   [ 99.5, 100.0, 98.0, 100.0, 101.0, 100.0],
            "Close": [101.0, 100.0, 100.5, 102.5, 101.5, 102.0],
            "Volume": [1_000] * 6,
        },
        index=idx,
    )


def test_compute_sessions_columns_and_values():
    s = compute_sessions(_ndx_like())
    assert list(s.columns) >= ["close", "ret", "ret_open_to_close", "gap", "dow"]
    # first row: ret/gap NaN
    assert pd.isna(s["ret"].iloc[0]) and pd.isna(s["gap"].iloc[0])
    # day 2: ret = 100.0/101.0 - 1
    assert s["ret"].iloc[1] == pytest.approx(100.0 / 101.0 - 1)
    assert s["gap"].iloc[1] == pytest.approx(101.0 / 101.0 - 1)          # open 101 vs prior close 101
    assert s["ret_open_to_close"].iloc[1] == pytest.approx(100.0 / 101.0 - 1)
    # dow: 1=Mon..5=Fri
    assert set(s["dow"].dropna().unique()) <= {1, 2, 3, 4, 5}
    assert s["dow"].iloc[0] == 4                                          # 2026-01-01 is a Thursday


def test_validate_continuity_flags_huge_move():
    df = _ndx_like()
    df.iloc[3, df.columns.get_loc("Close")] = 140.0                       # +36% in a day
    s = compute_sessions(df)
    bad = validate_continuity(s, max_abs_ret=0.25)
    # a spiked close pollutes BOTH adjacent returns (+39.3% spike, -27.5% reversal)
    assert len(bad) == 2
    assert list(bad["date"]) == [s.index[3], s.index[4]]


def test_validate_continuity_clean_data_passes():
    s = compute_sessions(_ndx_like())
    assert len(validate_continuity(s, max_abs_ret=0.25)) == 0


def test_parquet_roundtrip(tmp_path):
    s = compute_sessions(_ndx_like())
    p = tmp_path / "ndx.parquet"
    save_parquet(s.reset_index().rename(columns={"index": "date"}), p)
    back = load_parquet(p)
    assert len(back) == len(s)
    assert back["close"].iloc[-1] == pytest.approx(102.0)