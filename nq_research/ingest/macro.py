"""Macro ingest: FRED levels + ALFRED point-in-time vintages.

Point-in-time rule: a value on session date t must come from a vintage released
on or before t. ALFRED gives (observation_date, vintage/realtime_start, value)
triples; we join sessions to the latest vintage as of the session date.

Live fetching (network) lives in fetch_*; computation is pure and unit-tested
against injected frames.
"""
from __future__ import annotations

import pandas as pd

FRED_SERIES = [
    "CPIAUCSL", "CPILFESL", "PPIACO", "PAYEMS", "UNRATE", "FEDFUNDS",
    "DGS10", "T10Y2Y", "T5YIE", "DTWEXBGS", "DCOILWTICO",
]
VINTAGE_SERIES = ["CPIAUCSL", "PAYEMS"]       # CPI + NFP get ALFRED vintages


def fetch_alfred_vintages(series_id: str, api_key: str | None = None) -> pd.DataFrame:
    """Live ALFRED pull for one series -> (obs, vintage, value) triples (network)."""
    import requests

    base = "https://api.stlouisfed.org/fred/date"
    rows = []
    start = "1990-01-01"
    # vintagedata endpoint: observation_date + realtime_start per value
    url = "https://api.stlouisfed.org/fred/series/observations"
    realtime_periods = pd.date_range("1990-01-01", pd.Timestamp.today(), freq="MS")
    # To bound requests: pull monthly realtime windows (vintage = release month start).
    # This approximates true release dates at monthly granularity; tests key on the
    # semantics (latest vintage <= asof), not exact ALFRED release days.
    vintage_ends = realtime_periods + pd.offsets.MonthEnd(1)
    vintage_ends = list(vintage_ends)
    vintage_ends[-1] = pd.Timestamp.today().normalize()
    for vs, ve in zip(realtime_periods, vintage_ends):
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": "1990-01-01",
            "realtime_start": vs.date().isoformat(),
            "realtime_end": min(ve.date(), pd.Timestamp.today().date()).isoformat(),
        }
        # 'observation_start' isn't valid for realtime pulls? it is: filter obs but keep vintages
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            continue
        for obs in r.json().get("observations", []):
            if obs.get("value") in (".", None):
                continue
            rows.append((obs["date"], vs.date().isoformat(), float(obs["value"])))
    return pd.DataFrame(rows, columns=["obs", "vintage", "value"])


def fetch_fred_series(series_id: str, api_key: str | None = None) -> pd.DataFrame:
    """Current (revised) FRED levels -> (obs, value). For non-vintage series."""
    import requests

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": "1990-01-01",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    rows = [
        (o["date"], float(o["value"]))
        for o in r.json().get("observations", [])
        if o.get("value") not in (".", None)
    ]
    return pd.DataFrame(rows, columns=["obs", "value"])


def _prepared(vintages: pd.DataFrame) -> pd.DataFrame:
    df = vintages.copy()
    df["obs"] = pd.to_datetime(df["obs"])
    df["vintage"] = pd.to_datetime(df["vintage"])
    return df.sort_values(["obs", "vintage"])


def alfred_vintage_series(vintages: pd.DataFrame, obs_month: str, asof: str) -> float:
    """Value for obs_month from the latest vintage released on/before asof. NaN if unseen."""
    df = _prepared(vintages)
    sub = df[(df["obs"] == pd.Timestamp(obs_month)) & (df["vintage"] <= pd.Timestamp(asof))]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[-1]["value"])


def cpi_yoy_as_reported(vintages: pd.DataFrame, asof: str, lag_months: int = 12) -> float:
    """YoY inflation as known at `asof` using vintage values for both endpoints."""
    asof_ts = pd.Timestamp(asof)
    cur_month = asof_ts.to_period("M").to_timestamp() - pd.offsets.MonthBegin(1)
    # latest CPI obs month published by asof: conservative - require vintage <= asof
    df = _prepared(vintages)
    visible = df[df["vintage"] <= asof_ts]
    if visible.empty:
        return float("nan")
    cur_month = visible["obs"].max()
    base_month = cur_month - pd.DateOffset(months=lag_months)
    cur = alfred_vintage_series(vintages, cur_month.strftime("%Y-%m-%d"), asof)
    base = alfred_vintage_series(vintages, base_month.strftime("%Y-%m-%d"), asof)
    if pd.isna(cur) or pd.isna(base) or base == 0:
        return float("nan")
    return cur / base - 1


def last_value_asof(vintages: pd.DataFrame, asof: str, freq: str = "MS") -> float:
    """Latest *published* observation value as of asof (by vintage date).

    "Published" requires the series' NEXT observation to also be out (or asof to
    be >= a month past its vintage), guarding against preliminary prints.
    Conservative monthly rule: only obs months whose successor has a vintage
    on/before asof, or whose vintage is >=30 days old, count as final.
    """
    df = _prepared(vintages)
    sub = df[df["vintage"] <= pd.Timestamp(asof)]
    if sub.empty:
        return float("nan")
    latest_obs = sub["obs"].max()
    # finality check: a later obs month exists, or this one is >=30 days old
    later_exists = (sub["obs"] > latest_obs).any()
    vintage_age = (pd.Timestamp(asof) - sub[sub["obs"] == latest_obs]["vintage"].max()).days
    if later_exists or vintage_age >= 30:
        return float(sub[sub["obs"] == latest_obs].iloc[-1]["value"])
    # fall back to previous obs month (current one still preliminary)
    prev = sub[sub["obs"] < latest_obs]
    if prev.empty:
        return float("nan")
    latest_prev = prev["obs"].max()
    return float(prev[prev["obs"] == latest_prev].iloc[-1]["value"])