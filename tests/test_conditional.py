"""Slice 8 RED: ConditionalQuery over a real DuckDB sessions table built from synthetic data."""
import pandas as pd
import pytest

from nq_research.features.build import assemble_sessions
from nq_research.ingest.prices import compute_sessions
from nq_research.query.conditional import ConditionalQuery


def _make_db(tmp_path):
    """60 sessions, VIX flat 14, deterministic mild pattern: Fridays down."""
    idx = pd.bdate_range("2025-11-03", periods=60)
    closes = []
    px = 100.0
    for i, d in enumerate(idx):
        if d.weekday() == 4:            # Friday: -0.6%
            px *= 0.994
        else:
            px *= 1.001
        closes.append(px)
    ohlc = pd.DataFrame({
        "Open": closes, "High": [c * 1.002 for c in closes],
        "Low": [c * 0.998 for c in closes], "Close": closes,
        "Volume": [1_000] * 60,
    }, index=idx)
    s = compute_sessions(ohlc)
    vix = pd.DataFrame({"Close": [14.0] * 60}, index=idx)
    vix3m = pd.DataFrame({"Close": [16.0] * 60}, index=idx)
    table = assemble_sessions(s, vix, vix3m, None, None)

    db = tmp_path / "test.duckdb"
    import duckdb
    con = duckdb.connect(str(db))
    con.register("df", table.reset_index())
    con.execute("CREATE TABLE sessions AS SELECT * FROM df")
    con.close()
    return str(db)


def test_conditional_friday_dip_finds_matches(_db=None, tmp_path=None):
    # standalone db
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        dbp = _make_db(pathlib.Path(td))
        q = ConditionalQuery(db=dbp)
        res = q.conditional(
            filters={"dow": 5, "ret": (-0.0075, -0.0050)},
            target="next_session_return", horizon=1,
        )
        # Fridays land in the band most weeks (compounding shifts some out) — verify n>=8 and all-matching
        assert 8 <= res.n <= 12
        assert res.dates and len(res.dates) == res.n
        assert res.base_rate is not None
        assert res.ci_95[0] <= res.conditional_rate <= res.ci_95[1]
        # n=11 < 30 -> the warning MUST be populated (spec) and CI must bracket the rate
        assert res.n_warning is not None and "insufficient" in res.n_warning.lower()


def test_report_leads_with_sample_size():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        dbp = _make_db(pathlib.Path(td))
        q = ConditionalQuery(db=dbp)
        res = q.conditional(filters={"dow": 5}, target="next_session_return", horizon=1)
        txt = res.report()
        first_line = txt.splitlines()[0]
        # whatever n is, the FIRST thing shown is the sample size and its honesty label
        assert first_line.startswith("n=")
        if res.n < 30:
            assert "INSUFFICIENT SAMPLE" in first_line
        else:
            assert "CI" in first_line


def test_report_flags_insufficient_sample_loudly():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        dbp = _make_db(pathlib.Path(td))
        q = ConditionalQuery(db=dbp)
        res = q.conditional(
            filters={"dow": 5, "ret": (-0.0075, -0.0050), "is_post_opex_week": True},
            target="next_session_return", horizon=1,
        )
        assert res.n < 30
        assert "INSUFFICIENT SAMPLE" in res.report()


def test_percentiles_present():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        dbp = _make_db(pathlib.Path(td))
        q = ConditionalQuery(db=dbp)
        res = q.conditional(
            filters={"dow": 5, "ret": (-0.0075, -0.0050)},
            target="next_session_return", horizon=1,
        )
        for f in ("mean", "median", "p10", "p25", "p75", "p90"):
            assert getattr(res, f) is not None


def test_multiple_testing_tracks_trials_and_adjusts():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        dbp = _make_db(pathlib.Path(td))
        q = ConditionalQuery(db=dbp)
        for i in range(5):
            q.conditional(filters={"dow": (i % 5) + 1}, target="next_session_return", horizon=1)
        rep = q.multiple_testing_report()
        assert rep["trials"] == 5
        assert len(rep["adjusted_pvalues"]) == 5


def test_compare_subperiods_returns_windows():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        dbp = _make_db(pathlib.Path(td))
        q = ConditionalQuery(db=dbp)
        out = q.compare_subperiods({"dow": 5})
        assert isinstance(out, list) and len(out) >= 1
        first = out[0]
        assert {"window", "n", "mean"}.issubset(first.keys())


def test_targets_mfe_mae_and_triple_barrier():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        dbp = _make_db(pathlib.Path(td))
        q = ConditionalQuery(db=dbp)
        r1 = q.conditional(filters={}, target="max_favorable_excursion", horizon=3)
        assert r1.n > 0
        r2 = q.conditional(
            filters={},
            target="triple_barrier",
            horizon=10,
            barrier_params={"pt_vol": 1.0, "sl_vol": 1.0},
        )
        assert r2.n > 0
        assert set(r2.values) <= {"pt", "sl", "time"}   # NaN pre-vol rows must be dropped, not labeled