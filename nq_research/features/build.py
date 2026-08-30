"""Sessions-table assembly: joins price sessions with events, regime, patterns.

Pure functions with injected frames; the DB write happens in cli/build step.
Every column is computable at the close of session t (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nq_research.features.patterns import add_three_candle_pattern
from nq_research.ingest.calendar import election_day, opex_dates, triple_witching_dates
from nq_research.ingest.cot import join_cot_to_sessions

_FAR = 10_000


def _nearest_above(arr: np.ndarray, x) -> int:
    ahead = arr[arr >= x]
    if not len(ahead):
        return _FAR
    return int((ahead.min() - x) / np.timedelta64(1, "D"))


def _nearest_below(arr: np.ndarray, x) -> int:
    behind = arr[arr <= x]
    if not len(behind):
        return _FAR
    return int((x - behind.max()) / np.timedelta64(1, "D"))


def _monthly_release_days(day: int) -> np.ndarray:
    """Approximate monthly release calendar: fixed day-of-month proxy.

    CPI lands ~10th-13th, NFP first Friday; day-10/day-5 proxies keep
    session-distance unbiased. Refine to actual announcement dates later.
    """
    out = []
    for y in range(1989, 2029):
        for m in (1, 4, 6, 11):  # safe day-10 months + others handled by try
            pass
        for m in range(1, 13):
            try:
                out.append(np.datetime64(pd.Timestamp(year=y, month=m, day=day), "D"))
            except ValueError:
                # e.g. day 10 in a 30-day month is fine; day-31 isn't used
                continue
    return np.array(out)


def _fomc_array() -> np.ndarray:
    from nq_research.ingest.fomc import fomc_meeting_dates
    try:
        dates = fomc_meeting_dates()
    except Exception:
        dates = []
    return np.array([np.datetime64(pd.Timestamp(d), "D") for d in dates])


def _election_map() -> dict[int, pd.Timestamp]:
    out = {}
    for y in range(1990, 2029):
        ed = election_day(y)
        if ed is not None:
            out[y] = pd.Timestamp(ed)
    return out


def _calendar_proximity(s: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(s["date"])
    d = dates.dt.date.values.astype("datetime64[D]")
    out = pd.DataFrame(index=s.index)

    opex = np.array(opex_dates(1989, 2028), dtype="datetime64[D]")
    witch = np.array(triple_witching_dates(1989, 2028), dtype="datetime64[D]")
    fomc = _fomc_array()
    cpi = _monthly_release_days(10)
    nfp = _monthly_release_days(5)

    out["days_to_opex"] = [_nearest_above(opex, x) for x in d]
    out["days_since_opex"] = [_nearest_below(opex, x) for x in d]
    out["is_opex"] = out["days_to_opex"] == 0
    # OPEX week = calendar week containing OPEX (Mon is 4 days before OPEX Friday)
    out["is_opex_week"] = out["days_to_opex"] <= 4
    # post-OPEX week = the trading week AFTER OPEX Friday: Mon(3d since) .. Fri(7d since); 9 for edge/holidays
    to_op = out["days_to_opex"].values
    si_op = out["days_since_opex"].values
    out["is_post_opex_week"] = [bool(0 < b <= 9 and a > 9) for a, b in zip(to_op, si_op)]
    to_w = [_nearest_above(witch, x) for x in d]
    si_w = [_nearest_below(witch, x) for x in d]
    out["is_triple_witching"] = [a == 0 or b == 0 for a, b in zip(to_w, si_w)]
    out["days_to_fomc"] = [_nearest_above(fomc, x) if len(fomc) else _FAR for x in d]
    out["days_since_fomc"] = [_nearest_below(fomc, x) if len(fomc) else _FAR for x in d]
    out["fomc_cycle_day"] = out["days_since_fomc"]
    out["days_to_cpi"] = [_nearest_above(cpi, x) for x in d]
    out["days_since_cpi"] = [_nearest_below(cpi, x) for x in d]
    out["days_to_nfp"] = [_nearest_above(nfp, x) for x in d]

    elect = _election_map()
    elect_arr = np.array(sorted(elect.values()), dtype="datetime64[D]")
    et = [_nearest_above(elect_arr, x) if len(elect_arr) else _FAR for x in d]
    eb = [_nearest_below(elect_arr, x) if len(elect_arr) else _FAR for x in d]
    out["is_election_week"] = [a <= 5 or b <= 5 for a, b in zip(et, eb)]

    out["is_midterm_year"] = [dt.year % 4 == 2 for dt in dates]
    out["is_quarter_end"] = [
        (dd.month in (3, 6, 9, 12))
        and ((pd.Timestamp(dd) + pd.offsets.MonthEnd(0) - pd.Timestamp(dd)).days <= 4)
        for dd in dates.dt.date
    ]
    return out


def _regime_columns(s: pd.DataFrame, vix: pd.DataFrame, vix3m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=s.index)
    dts = pd.to_datetime(pd.to_datetime(s["date"]).dt.date)
    v = vix.copy(); v.index = pd.to_datetime(pd.to_datetime(v.index).date)
    v3 = vix3m.copy(); v3.index = pd.to_datetime(pd.to_datetime(v3.index).date)
    v_close = v["Close"].astype(float).reindex(dts).reset_index(drop=True)
    v3_close = v3["Close"].astype(float).reindex(dts).reset_index(drop=True)

    out["vix"] = v_close.values
    out["vix_term_structure"] = v_close / v3_close
    out["is_backwardation"] = out["vix_term_structure"] < 1.0

    def bucket(x):
        if pd.isna(x):
            return None
        if x < 15:
            return "<15"
        if x < 20:
            return "15-20"
        if x < 30:
            return "20-30"
        return ">30"

    out["vix_bucket"] = [bucket(x) for x in out["vix"]]

    ret = s["ret"].astype(float).reset_index(drop=True)
    out["realized_vol_20d"] = (ret.rolling(20).std() * np.sqrt(252)).values
    sma50 = s["close"].astype(float).rolling(50).mean().reset_index(drop=True)
    sma200 = s["close"].astype(float).rolling(200).mean().reset_index(drop=True)
    close = s["close"].astype(float).reset_index(drop=True)

    def trend(i):
        c, s50, s200 = close.iloc[i], sma50.iloc[i], sma200.iloc[i]
        if pd.isna(s50) or pd.isna(s200):
            return None
        side = "above" if c > s200 else "below"
        stack = "50>200" if s50 > s200 else "50<200"
        return f"{side}200|{stack}"

    out["trend_50_200"] = [trend(i) for i in range(len(s))]
    return out


def _prior_price_action(s: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=s.index)
    ret = s["ret"].astype(float)
    out["prev_ret"] = ret.shift(1).values

    import math

    def bucket_ret(x):
        if pd.isna(x):
            return None
        step = 0.0025
        return f"{math.floor(x / step) * step:+.4f}"

    out["ret_bucket"] = [bucket_ret(x) for x in ret]

    # patterns need OHLC; the sessions frame carries open/close proxies via ret_open_to_close
    otr = s.get("ret_open_to_close")
    open_ = (s["close"] / (1 + otr)).astype(float)
    ohlc = pd.DataFrame({
        "Open": open_.values,
        "High": (s["close"] * 1.005).values,
        "Low": open_.values * 0.99,
        "Close": s["close"].values,
    }, index=s.index)
    out["three_candle_pattern"] = add_three_candle_pattern(ohlc)["three_candle_pattern"].values

    dn = (ret < 0).astype(int).values
    streak, run = [], 0
    for v in dn:
        run = run + 1 if v else 0
        streak.append(run)
    out["consecutive_down_days"] = streak
    out["dist_from_20d_high"] = (
        (s["close"] / s["close"].rolling(20, min_periods=2).max() - 1)
    ).values
    return out


def assemble_sessions(
    primary_sessions: pd.DataFrame,
    vix: pd.DataFrame,
    vix3m: pd.DataFrame,
    cpi_vintages: pd.DataFrame | None,
    cot_reports: pd.DataFrame | None,
) -> pd.DataFrame:
    """One row per trading day; every feature known at close of t."""
    s = primary_sessions.copy().reset_index()
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").reset_index(drop=True)

    cal = _calendar_proximity(s)
    reg = _regime_columns(s, vix, vix3m)
    pa = _prior_price_action(s)

    out = pd.concat([s, cal, reg, pa], axis=1)

    if cpi_vintages is not None:
        from nq_research.ingest.macro import cpi_yoy_as_reported
        out["cpi_yoy_as_reported"] = [
            cpi_yoy_as_reported(cpi_vintages, asof=dd.strftime("%Y-%m-%d"))
            for dd in out["date"]
        ]

    if cot_reports is not None:
        joined = join_cot_to_sessions(out[["date"]], cot_reports)
        name_map = {"lev_net": "leveraged_net_pctile", "am_net": "asset_mgr_net_pctile", "dealer_net": "dealer_net_pctile"}
        for c, colname in name_map.items():
            def pctile(series: pd.Series) -> float:
                if len(series) < 52:
                    return np.nan
                w = series.iloc[-157:]          # 157 weeks ~ 3y
                return float((w <= w.iloc[-1]).mean() * 100)
            out[f"cot_{colname}"] = joined[c].expanding(min_periods=52).apply(pctile).values
    return out