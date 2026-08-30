"""Three-candle pattern classification - explicit rules, no TA-library black box.

Vocabulary (checked in this priority order, first match wins):
  1. three_down         - each of the last 3 closes strictly below the previous close
  2. three_up           - each of the last 3 closes strictly above the previous close
  3. down_down_up       - close path strictly down, down, then up over the last 3 transitions
  4. morning_star_like  - bar[-3] body down >= 0.5x its range, bar[-2] body <= 0.3x its range
                          (stall), bar[-1] closes above the midpoint of bar[-3]'s body
  5. outside_bar        - last bar's high >= prior high AND low <= prior low (strict engulf of range)
  6. inside_bar         - last bar's high < prior high AND low > prior low
  7. none               - nothing matched (classification is still produced once >=3 bars exist)

classify_three_candle returns None when fewer than 3 bars are available.
"""
from __future__ import annotations

import pandas as pd

VOCABULARY = (
    "three_down", "three_up", "down_down_up",
    "morning_star_like", "outside_bar", "inside_bar", "none",
)


def classify_three_candle(df: pd.DataFrame) -> str | None:
    """Classify the trailing sessions of an OHLC DataFrame (columns Open/High/Low/Close).

    Returns None when fewer than 3 bars exist (nothing classifiable).
    Three-transition rules (three_up/three_down/down_down_up) need 4 closes; with
    exactly 3 bars only the shape rules (morning_star_like/outside/inside) can fire.
    """
    if len(df) < 3:
        return None
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    lo = df["Low"].astype(float)
    c = df["Close"].astype(float)

    h2, h3 = h.iloc[-2], h.iloc[-1]
    l2, l3 = lo.iloc[-2], lo.iloc[-1]

    if len(df) >= 4:
        c0, c1, c2, c3 = c.iloc[-4], c.iloc[-3], c.iloc[-2], c.iloc[-1]
        o1 = o.iloc[-3]
        if c1 < c0 and c2 < c1 and c3 < c2:
            return "three_down"
        if c1 > c0 and c2 > c1 and c3 > c2:
            return "three_up"
        if c1 < c0 and c2 < c1 and c3 > c2:
            return "down_down_up"
        body3, rng3 = abs(c1 - o1), max(h.iloc[-3] - lo.iloc[-3], 1e-12)
        mid1 = (c1 + o1) / 2
        o2 = o.iloc[-2]
    else:  # exactly 3 bars: pattern rules see bars (-3,-2,-1) but 3-transition rules can't
        c1, c2, c3 = c.iloc[-3], c.iloc[-2], c.iloc[-1]
        o1, o2 = o.iloc[-3], o.iloc[-2]
        body3, rng3 = abs(c1 - o1), max(h.iloc[-3] - lo.iloc[-3], 1e-12)
        mid1 = (c1 + o1) / 2

    body2, rng2 = abs(c2 - o2), max(h2 - l2, 1e-12)
    if c1 < o1 and body3 / rng3 >= 0.5 and body2 / rng2 <= 0.3 and c3 > mid1:
        return "morning_star_like"

    if h3 >= h2 and l3 <= l2:
        return "outside_bar"
    if h3 < h2 and l3 > l2:
        return "inside_bar"
    return "none"


def add_three_candle_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a three_candle_pattern column (rolling classification; None before 4 bars)."""
    out = df.copy()
    labels = [classify_three_candle(out.iloc[: i + 1]) for i in range(len(out))]
    out["three_candle_pattern"] = labels
    return out