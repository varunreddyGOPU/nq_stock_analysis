"""Slice 2 RED: three_candle_pattern classification.

Controlled vocabulary + explicit rules (docstring in patterns.py is the spec):
- three_down: each of last 3 closes < previous close
- three_up:   each of last 3 closes > previous close
- down_down_up: c1<c0, c2<c1, c3>c2
- inside_bar: third bar's high<prev high AND low>prev low (bar 3 inside bar 2)
- outside_bar: third bar engulfs bar 2's range (high>=hh, low<=ll)
- morning_star_like: down, small-body continuation, then close above bar2 open... precise rule in module
Priority: named multi-bar shapes before generic three_up/down? NO - spec order:
three_up/three_down first, then down_down_up, morning_star_like, outside, inside.
"""

import pandas as pd
import pytest

from nq_research.features.patterns import classify_three_candle, add_three_candle_pattern


def _frame(rows):
    """rows: list of (o,h,l,c) -> DataFrame indexed by business days."""
    idx = pd.bdate_range("2026-01-01", periods=len(rows))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


def _run(rows):
    return classify_three_candle(_frame(rows))


def test_none_for_fewer_than_three_bars():
    assert _run([(10, 11, 9, 10.5)]) is None
    assert _run([(10, 11, 9, 10.5), (10.5, 12, 10, 11)]) is None


def test_three_down():
    out = _run([(10, 10.5, 9.5, 10), (10, 10.2, 9.4, 9.8), (9.8, 10, 9.0, 9.5), (9.5, 9.7, 8.8, 9.0)])
    assert out == "three_down"


def test_three_up():
    out = _run([(10, 10.5, 9.5, 10), (10, 10.6, 9.9, 10.3), (10.3, 10.9, 10.1, 10.7), (10.7, 11.4, 10.5, 11.2)])
    assert out == "three_up"


def test_down_down_up():
    out = _run([(11, 11.2, 10.6, 11), (11, 10.8, 10.3, 10.5), (10.5, 10.6, 10.0, 10.2), (10.2, 10.9, 10.1, 10.8)])
    assert out == "down_down_up"


def test_inside_bar():
    # last bar entirely inside previous bar's range (fixed low: 11.8 > 11.7)
    out = _run([(10, 10.4, 9.6, 10), (12, 12.6, 11.4, 12.2), (12.3, 12.5, 11.7, 11.9), (11.9, 12.1, 11.8, 12.0)])
    assert out == "inside_bar"


def test_outside_bar():
    # last bar's range strictly engulfs previous bar
    out = _run([(10, 10.4, 9.6, 10), (12, 12.2, 11.8, 12), (12.0, 12.8, 11.4, 12.5)])
    assert out == "outside_bar"


def test_morning_star_like():
    # trailing 3: strong down bar, small-body stall, close back above down-bar body midpoint
    out = _run([
        (12, 12.5, 11.5, 12.0),      # context
        (11, 11.1, 10.0, 10.35),     # strong down bar (body 0.65 / range 1.1 = 0.59)
        (10.6, 10.7, 10.5, 10.62),   # small stall body (0.02/0.2 = 0.1)
        (10.62, 11.4, 10.5, 11.1),   # closes above (10+10.35)/2 = 10.175
    ])
    assert out == "morning_star_like"


def test_none_when_no_pattern_matches():
    # mixed moves that fit no rule
    out = _run([(10, 10.5, 9.5, 10), (10, 10.6, 9.9, 10.2), (10.4, 10.6, 9.9, 10.1), (10.3, 10.5, 9.9, 10.4)])
    assert out == "none"


def test_add_three_candle_pattern_column():
    df = _frame([(10, 10.5, 9.5, 10), (10, 10.2, 9.4, 9.8), (9.8, 10, 9.0, 9.5), (9.5, 9.7, 8.8, 9.0)])
    out = add_three_candle_pattern(df)
    got = [None if pd.isna(v) else v for v in out["three_candle_pattern"]]
    assert got == [None, None, "none", "three_down"]