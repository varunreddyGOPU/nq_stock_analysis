"""CFTC COT ingest: Traders in Financial Futures, Nasdaq-100 contracts.

Timing semantics (the whole point): the weekly report snapshots TUESDAY
positions and is released FRIDAY 3:30pm ET. Any join to daily sessions must
make a report visible only on/after its release date. Snapshots are never used
as of the snapshot date.

Live sources (network): Socrata API publicreporting.cftc.gov (preferred) or
historical zips fut_fin_txt_<year>.zip. Unit tests run on injected frames.
"""
from __future__ import annotations

import io
import zipfile
from datetime import timedelta

import pandas as pd

SOCRATA_URL = (
    "https://publicreporting.cftc.gov/resource/gpe5-46if.json"  # TFF futures+options combined, weekly
)
CME_CODE = "20974+"     # NASDAQ-100 futures+options combined (CME), TFF concluded-market code


def fetch_cot_tff(cme_code: str = CME_CODE) -> pd.DataFrame:
    """Live Socrata pull -> (tuesday, lev_net, am_net, dealer_net), from June 2010."""
    import requests

    params = {
        "$select": (
            "report_date_as_yyyy_mm_dd,"
            "lev_money_positions_long,lev_money_positions_short,"
            "asset_mgr_positions_long,asset_mgr_positions_short,"
            "dealer_positions_long_all,dealer_positions_short_all"
        ),
        "$where": f"cftc_contract_market_code='{cme_code}'",
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": 50000,
    }
    r = requests.get(SOCRATA_URL, params=params, timeout=120)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame([
        (
            row["report_date_as_yyyy_mm_dd"],
            float(row["lev_money_positions_long"]) - float(row["lev_money_positions_short"]),
            float(row["asset_mgr_positions_long"]) - float(row["asset_mgr_positions_short"]),
            float(row["dealer_positions_long_all"]) - float(row["dealer_positions_short_all"]),
        )
        for row in rows
    ], columns=["tuesday", "lev_net", "am_net", "dealer_net"])
    df["tuesday"] = pd.to_datetime(df["tuesday"])
    return df.sort_values("tuesday").reset_index(drop=True)


def add_release_dates(reports: pd.DataFrame) -> pd.DataFrame:
    """Attach release dates: report is out the Friday following the Tuesday snapshot."""
    out = reports.copy()
    tue = pd.to_datetime(out["tuesday"])
    days_to_friday = (4 - tue.dt.weekday) % 7
    days_to_friday = days_to_friday.where(days_to_friday > 0, 7)   # never same-day
    out["release"] = tue + pd.to_timedelta(days_to_friday, unit="D")
    return out


def cot_asof(reports: pd.DataFrame, asof: str, col: str) -> float:
    """Report value visible at close of `asof` (release <= asof). NaN if none yet."""
    df = pd.to_datetime(reports["release"])
    mask = df <= pd.Timestamp(asof)
    if not mask.any():
        return float("nan")
    return float(reports.loc[mask, col].iloc[-1])


def join_cot_to_sessions(sessions: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Left-join sessions to the latest report with release date <= session date."""
    rep = add_release_dates(reports)[["release", "lev_net", "am_net", "dealer_net"]]
    rep = rep.sort_values("release").rename(columns={"release": "date"})
    s = sessions.copy()
    s["date"] = pd.DatetimeIndex(pd.to_datetime(s["date"])).as_unit("s")
    rep["date"] = pd.DatetimeIndex(pd.to_datetime(rep["date"])).as_unit("s")
    out = pd.merge_asof(s.sort_values("date"), rep.sort_values("date"),
                        on="date", direction="backward", allow_exact_matches=True)
    return out


def net_position_percentile(reports: pd.DataFrame, asof: str, col: str, window_years: int = 3) -> float:
    """Percentile of the latest report value vs trailing `window_years` of reports."""
    df = pd.to_datetime(reports["release"])
    visible = reports[df <= pd.Timestamp(asof)]
    cur = cot_asof(reports, asof, col)
    if pd.isna(cur):
        return float("nan")
    lo = pd.Timestamp(asof) - pd.DateOffset(years=window_years)
    hist = visible.loc[df >= lo, col].astype(float)
    prev = hist.iloc[:-1] if len(hist) > 1 else hist   # trailing: exclude the current? spec: trailing window incl current
    return float((hist <= cur).mean() * 100)