"""Slice 4 RED: macro ingest with ALFRED point-in-time vintages.

Network is injected (fake fetchers) for unit tests.
"""
import pandas as pd
import pytest

from nq_research.ingest.macro import (
    alfred_vintage_series,
    cpi_yoy_as_reported,
    last_value_asof,
)


def _fake_alfred_cpi():
    """Monthly CPIAUCSL vintages: (observation_month, vintage_date, value).

    Modeled on real ALFRED behavior: Jan value published mid-Feb, subject to
    revisions in later vintages.
    """
    return pd.DataFrame(
        [
            # obs month, vintage (release) date, index value
            ("2024-11-01", "2024-12-11", 321.0),
            ("2024-12-01", "2025-01-15", 322.0),
            ("2025-01-01", "2025-02-12", 323.0),
            ("2025-01-01", "2025-03-12", 323.4),   # revision of Jan
            ("2025-02-01", "2025-03-12", 324.1),
        ],
        columns=["obs", "vintage", "value"],
    )


def test_alfred_vintage_asof_excludes_future_vintages():
    v = alfred_vintage_series(_fake_alfred_cpi(), obs_month="2025-01-01", asof="2025-02-28")
    # only the first (unrevised) vintage is visible
    assert float(v) == 323.0


def test_alfred_vintage_after_revision():
    v = alfred_vintage_series(_fake_alfred_cpi(), obs_month="2025-01-01", asof="2025-03-15")
    assert float(v) == 323.4   # revised value now visible


def test_alfred_vintage_missing_obs_returns_nan():
    v = alfred_vintage_series(_fake_alfred_cpi(), obs_month="2024-10-01", asof="2025-03-15")
    assert pd.isna(v)


def test_cpi_yoy_uses_yoy_of_vintage_index():
    yoy = cpi_yoy_as_reported(_fake_alfred_cpi(), asof="2025-03-15", lag_months=12)
    # latest published obs = Feb-2025 (324.1); base Feb-2024 missing -> NaN
    assert pd.isna(yoy)


def test_cpi_yoy_computes_when_base_exists():
    df = _fake_alfred_cpi()
    df.loc[len(df)] = ["2024-02-01", "2024-03-13", 310.0]   # Feb-2024 base, published Mar 2024
    yoy = cpi_yoy_as_reported(df, asof="2025-03-15", lag_months=12)
    assert yoy == pytest.approx(324.1 / 310.0 - 1, abs=1e-9)


def test_last_value_asof_monthly_series():
    v = last_value_asof(_fake_alfred_cpi(), asof="2025-02-20", freq="MS")
    # latest obs month fully published by Feb 20 2025 is Dec-2024 (vintage 2025-01-15)
    assert float(v) == 322.0