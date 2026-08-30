"""Slice 1 RED: event calendar — OPEX, triple witching, elections.

Known-date spot checks verified against public calendars, including months
where the 1st falls on a Friday (hardest case for third-Friday logic).
"""
from datetime import date

import pytest

from nq_research.ingest.calendar import (
    election_day,
    is_quarter_end,
    is_third_friday,
    opex_dates,
    third_friday,
    triple_witching_dates,
)


def test_third_friday_basic():
    # Well-known OPEX dates
    assert third_friday(2024, 1) == date(2024, 1, 19)
    assert third_friday(2026, 8) == date(2026, 8, 21)   # this month, day after Friday Aug 28's week? no: the OPEX itself
    assert third_friday(2026, 9) == date(2026, 9, 18)


def test_third_friday_when_first_is_friday():
    # Aug 2026: the 1st is a SATURDAY; Fridays are 7,14,21,28 -> third = the 21st
    assert date(2026, 8, 1).weekday() == 5
    assert third_friday(2026, 8) == date(2026, 8, 21)
    # Hard case: May 2026 — May 1 IS a Friday. Fridays: 1,8,15 -> third is the 15th
    assert date(2026, 5, 1).weekday() == 4
    assert third_friday(2026, 5) == date(2026, 5, 15)
    # Jan 2021: Jan 1 is a Friday -> third Friday Jan 15 (known OPEX)
    assert third_friday(2021, 1) == date(2021, 1, 15)


def test_third_friday_dozen_known_opex():
    known = [
        (1999, 6, date(1999, 6, 18)),
        (2002, 7, date(2002, 7, 19)),
        (2008, 10, date(2008, 10, 17)),
        (2015, 3, date(2015, 3, 20)),
        (2020, 3, date(2020, 3, 20)),
        (2022, 5, date(2022, 5, 20)),
        (2023, 11, date(2023, 11, 17)),
        (2025, 1, date(2025, 1, 17)),
        (2026, 3, date(2026, 3, 20)),
        (2026, 6, date(2026, 6, 19)),
        (2026, 12, date(2026, 12, 18)),
        (2027, 3, date(2027, 3, 19)),
    ]
    for y, m, expected in known:
        got = third_friday(y, m)
        assert got == expected, f"{y}-{m}: got {got}, want {expected}"
        assert got.weekday() == 4


def test_is_third_friday():
    assert is_third_friday(date(2026, 8, 21))
    assert not is_third_friday(date(2026, 8, 28))
    assert not is_third_friday(date(2026, 8, 14))  # second Friday


def test_opex_dates_range():
    ds = opex_dates(1990, 2027)
    assert len(ds) == (2027 - 1990 + 1) * 12
    assert min(ds) == date(1990, 1, 19)   # third Friday of Jan 1990
    assert date(2026, 8, 21) in ds
    # sorted, unique
    assert all(a < b for a, b in zip(ds, ds[1:]))


def test_triple_witching_only_mar_jun_sep_dec():
    ds = triple_witching_dates(2024, 2026)
    assert all(d.month in (3, 6, 9, 12) for d in ds)
    assert len(ds) == 12
    assert date(2026, 9, 18) in ds          # Sep 2026 OPEX == witching
    assert date(2026, 8, 21) not in ds      # August OPEX is not witching


def test_election_days():
    # First Tuesday after the first Monday in November, even years only
    assert election_day(2024) == date(2024, 11, 5)      # presidential
    assert election_day(2026) == date(2026, 11, 3)      # midterm
    assert election_day(2020) == date(2020, 11, 3)
    assert election_day(2022) == date(2022, 11, 8)
    assert election_day(2023) is None                   # odd years: none
    assert election_day(2025) is None


def test_is_quarter_end_flag():
    # last trading day of quarter: here defined as date whose month in (3,6,9,12)
    # and is within 4 calendar days of month end, weekday-aware
    assert is_quarter_end(date(2026, 9, 30))            # Wed, month end
    assert is_quarter_end(date(2026, 6, 30))            # Tue, month end
    assert not is_quarter_end(date(2026, 9, 18))        # witching, not quarter end
    assert not is_quarter_end(date(2026, 8, 28))


# imported late so the test file itself is readable
from nq_research.ingest.calendar import opex_dates  # noqa: E402