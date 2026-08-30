"""Price ingest: yfinance OHLCV -> Parquet cache -> base session frame.

The sessions frame here carries only price-derived columns
(date, close, ret, ret_open_to_close, gap, dow). Event/regime/macro columns
are joined in features/build.py. ^NDX is the primary return series (cash index,
no futures roll gaps); NQ=F and friends are stored but never used for returns.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PRIMARY = "^NDX"
SECONDARY = ["QQQ", "^GSPC", "NQ=F", "^VIX", "^VIX3M", "^VVIX"]
ALL_SERIES = [PRIMARY, *SECONDARY]


def fetch_yfinance(series: str, start: str | None = None) -> pd.DataFrame:
    """Live daily bars for one ticker (network). Full available history by default."""
    import yfinance as yf

    if series == "^VIX3M":
        # Yahoo only serves 1 bar; Cboe publishes the official free CSV
        return fetch_cboe_vol_index("VIX3M", start)
    df = yf.download(series, period="max", interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index).tz_localize(None).date
    df.index.name = "date"
    if start is not None:
        df = df[pd.Series(df.index).map(lambda d: pd.Timestamp(d) >= pd.Timestamp(start)).values]
    return df


def fetch_cboe_vol_index(name: str, start: str | None = None) -> pd.DataFrame:
    """Cboe free daily-history CSV for VOL indices (VIX, VIX3M, VVIX...)."""
    import requests
    from io import StringIO

    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = [c.upper() for c in df.columns]
    idx = pd.to_datetime(df["DATE"], format="%m/%d/%Y").dt.date
    out = pd.DataFrame(
        {"Open": df["OPEN"], "High": df["HIGH"], "Low": df["LOW"], "Close": df["CLOSE"]},
        index=idx,
    )
    out.index.name = "date"
    out = out.sort_index()
    if start is not None:
        out = out[pd.Series(out.index).map(lambda d: pd.Timestamp(d) >= pd.Timestamp(start)).values]
    return out


def compute_sessions(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV -> price-derived session columns.

    All returns are simple close-to-close fractions. dow: 1=Mon..5=Fri,
    weekend rows dropped (yfinance never emits them, but defensive).
    """
    df = ohlcv.copy()
    df = df[df.index.map(lambda d: pd.Timestamp(d).weekday() < 5)]
    out = pd.DataFrame(index=df.index)
    out["close"] = df["Close"].astype(float)
    c = out["close"]
    out["ret"] = c.pct_change()
    out["ret_open_to_close"] = c / df["Open"].astype(float) - 1
    out["gap"] = df["Open"].astype(float) / c.shift(1) - 1
    out["dow"] = [pd.Timestamp(d).weekday() + 1 for d in df.index]
    out.index.name = "date"
    return out


def validate_continuity(sessions: pd.DataFrame, max_abs_ret: float = 0.25) -> pd.DataFrame:
    """Rows whose |ret| exceeds max_abs_ret — data errors / accidental roll gaps."""
    bad = sessions[sessions["ret"].abs() > max_abs_ret]
    return bad.reset_index()[["date", "ret"]]


def save_parquet(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


def ingest_all(cache_dir: Path | str = "data/raw", start: str | None = None) -> dict[str, pd.DataFrame]:
    """Download every series once, cache to per-ticker Parquet, return raw frames."""
    cache = Path(cache_dir)
    out = {}
    for s in ALL_SERIES:
        p = cache / f"{s.replace('^', '_').replace('=', '_')}.parquet"
        if p.exists():
            df = load_parquet(p).set_index("date")
        else:
            df = fetch_yfinance(s, start=start)
            save_parquet(df.reset_index(), p)
        out[s] = df
    return out