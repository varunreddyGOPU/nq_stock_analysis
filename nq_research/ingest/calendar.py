"""Event calendar: OPEX, triple witching, elections, quarter-end flags.

All computed from rules - no vendor data. Weekday convention: Monday=0 .. Friday=4
(Python default), Sunday=6.
"""
from __future__ import annotations

from datetime import date, timedelta

__all__ = [
    "third_friday",
    "is_third_friday",
    "opex_dates",
    "triple_witching_dates",
    "election_day",
    "is_quarter_end",
]


def third_friday(year: int, month: int) -> date:
    """Third Friday of the month (standard monthly OPEX).

    If the 1st falls on Friday..Sunday the third Friday is the 15th/16th;
    otherwise later. Computed directly: first Friday + 14 days.
    """
    first = date(year, month, 1)
    offset = (4 - first.weekday()) % 7          # days until first Friday
    return first + timedelta(days=offset + 14)


def is_third_friday(d: date) -> bool:
    """True when d is the third Friday of its month."""
    return d.weekday() == 4 and third_friday(d.year, d.month) == d


def opex_dates(start_year: int, end_year: int) -> list[date]:
    """Monthly OPEX (third Friday) for each month in [start_year, end_year], sorted."""
    out = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            out.append(third_friday(y, m))
    return sorted(out)


def triple_witching_dates(start_year: int, end_year: int) -> list[date]:
    """Triple-witching OPEX: third Friday of Mar/Jun/Sep/Dec, sorted."""
    out = []
    for y in range(start_year, end_year + 1):
        for m in (3, 6, 9, 12):
            out.append(third_friday(y, m))
    return sorted(out)


def election_day(year: int) -> date | None:
    """US general election: first Tuesday after the first Monday in November.

    Even years only; returns None for odd years.
    """
    if year % 2 != 0:
        return None
    first = date(year, 11, 1)
    first_monday = first + timedelta(days=(0 - first.weekday()) % 7)
    return first_monday + timedelta(days=1)     # Tuesday after first Monday


def is_quarter_end(d: date) -> bool:
    """Approximate quarter-end session flag: Mar/Jun/Sep/Dec within 4 days of month end.

    Calendar approximation (engine joins it to actual trading days upstream;
    a session-based refinement can intersect with the sessions table later).
    """
    if d.month not in (3, 6, 9, 12):
        return False
    nxt = (date(d.year + 1, 1, 1) if d.month == 12
           else date(d.year, d.month + 1, 1))
    return (nxt - d).days <= 4