"""FOMC meeting dates with a checked-in CSV fallback.

Tries to scrape federalreserve.gov historical + scheduled FOMC meeting pages.
If the scrape is unreliable (network, layout change, or empty result), falls back
to data/fomc_dates.csv (historical 1990-2024 + scheduled 2026-2027, editable).

Returns sorted list of date(date) meeting *start* dates.
"""
from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path

import requests

FED_HIST_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical2026.htm"
FED_CAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
CSV_FALLBACK = Path(__file__).resolve().parents[2] / "data" / "fomc_dates.csv"

_MONTHS = dict(
    january=1, february=2, march=3, april=4, may=5, june=6,
    july=7, august=8, september=9, october=10, november=11, december=12,
)
_MEETING_RE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:\s*[-/]\s*(\d{1,2}))?,\s*(\d{4})", re.IGNORECASE)


def fomc_meeting_dates(use_csv_fallback: bool = True) -> list[date]:
    """Historical + scheduled FOMC meeting start dates (8 per year, ~2 days each)."""
    got: set[date] = set()
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research; contact: local)"}
        for url in (FED_HIST_URL, FED_CAL_URL):
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code != 200 or len(resp.text) < 5000:
                continue
            for month, day_a, _day_b, year in _MEETING_RE.findall(resp.text):
                m = _MONTHS[month.lower()]
                try:
                    got.add(date(int(year), m, int(day_a)))
                except ValueError:
                    continue
        got = {d for d in got if 1990 <= d.year <= 2027}
    except requests.RequestException:
        pass
    if got or not use_csv_fallback:
        return sorted(got)
    return _from_csv()


def _from_csv() -> list[date]:
    if not CSV_FALLBACK.exists():
        return []
    with open(CSV_FALLBACK, newline="") as fh:
        return sorted(date.fromisoformat(row["date"]) for row in csv.DictReader(fh) if row.get("date"))