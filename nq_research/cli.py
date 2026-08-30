"""nq_research CLI: ingest -> build -> query-example.

    python -m nq_research.cli ingest   # yfinance + FRED/ALFRED + COT -> data/raw parquet
    python -m nq_research.cli build    # assemble sessions table -> data/nq_research.duckdb
    python -m nq_research.cli query-example   # Friday -0.5..-0.75% x post-OPEX question
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "nq_research.duckdb"


def cmd_ingest(args):
    from nq_research.ingest.prices import ingest_all
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"ingesting prices -> {RAW}")
    frames = ingest_all(RAW)
    for k, v in frames.items():
        print(f"  {k:8s} {len(v):6,d} bars ({v.index.min()} .. {v.index.max()})")

    api_key = os.environ.get("FRED_API_KEY")
    if api_key:
        from nq_research.ingest.macro import fetch_alfred_vintages, fetch_fred_series, FRED_SERIES, VINTAGE_SERIES
        print("ingesting FRED/ALFRED macro...")
        for sid in FRED_SERIES:
            try:
                if sid in VINTAGE_SERIES:
                    v = fetch_alfred_vintages(sid, api_key)
                    v.to_parquet(RAW / f"alfred_{sid}.parquet", index=False)
                else:
                    lv = fetch_fred_series(sid, api_key)
                    lv.to_parquet(RAW / f"fred_{sid}.parquet", index=False)
                print(f"  {sid:10s} ok")
            except Exception as e:
                print(f"  {sid:10s} FAILED ({e}) — build will leave macro cols NaN")
    else:
        print("FRED_API_KEY not set — skipping macro ingest (engine still works, macro cols NaN)")

    try:
        from nq_research.ingest.cot import fetch_cot_tff
        cot = fetch_cot_tff()
        cot.to_parquet(RAW / "cot_nq.parquet", index=False)
        print(f"  COT TFF  {len(cot):,d} weekly reports")
    except Exception as e:
        print(f"  COT failed ({e}) — positioning cols will be NaN")

    print("ingest done.")


def cmd_build(args):
    from nq_research.features.build import assemble_sessions
    from nq_research.ingest.prices import load_parquet, compute_sessions

    files = {p.name: p for p in RAW.glob("*.parquet")}
    def _pick(ticker):
        return files.get(f"{ticker.replace('^', '_').replace('=', '_')}.parquet")

    ndx_p = _pick("^NDX")
    if ndx_p is None:
        sys.exit("no ^NDX parquet — run `ingest` first")
    ndx = load_parquet(ndx_p).set_index("date")
    sessions = compute_sessions(ndx)

    def _close(t):
        p = _pick(t)
        if p is None:
            return pd.DataFrame()
        return load_parquet(p).set_index("date")[["Close"]].rename(columns={"Close": "close"})

    vix, vix3m = _pick("^VIX"), _pick("^VIX3M")
    vix_df = load_parquet(vix).set_index("date")[["Close"]] if vix else pd.DataFrame()
    vix3m_df = load_parquet(vix3m).set_index("date")[["Close"]] if vix3m else vix_df.copy()

    alfred_p = RAW / "alfred_CPIAUCSL.parquet"
    cpi_vintages = load_parquet(alfred_p) if alfred_p.exists() else None
    cot_p = RAW / "cot_nq.parquet"
    cot = load_parquet(cot_p) if cot_p.exists() else None

    table = assemble_sessions(compute_sessions(load_parquet(ndx_p).set_index("date")),
                              vix=vix_df, vix3m=vix3m_df,
                              cpi_vintages=cpi_vintages, cot_reports=cot)
    out = table.copy()
    out = out.loc[:, ~pd.Index(out.columns).duplicated()]           # drop dup column names
    if "date" not in out.columns:
        out.insert(0, "date", out.index)
    out = out.drop(columns=[c for c in ("index", "date_1") if c in out.columns]).reset_index(drop=True)

    import duckdb
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    con.register("tbl", out)
    con.execute("DROP TABLE IF EXISTS sessions")
    con.execute("CREATE TABLE sessions AS SELECT * FROM tbl")
    n = con.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM sessions").fetchone()
    con.close()
    print(f"sessions table: {n[0]:,d} rows ({n[1]} .. {n[2]}) -> {DB}")


def cmd_query_example(args):
    from nq_research.query.conditional import ConditionalQuery

    q = ConditionalQuery(db=str(DB))
    print("=" * 78)
    print("Q: Friday down -0.50%..-0.75% — what happened the following Monday?")
    print("=" * 78)
    r = q.conditional(
        filters={"dow": 5, "ret": (-0.0075, -0.005)},
        target="next_session_return", horizon=1,
    )
    print(r.report())
    print()
    print("=" * 78)
    print("Same, restricted to post-OPEX weeks:")
    print("=" * 78)
    r2 = q.conditional(
        filters={"dow": 5, "ret": (-0.0075, -0.005), "is_post_opex_week": True},
        target="next_session_return", horizon=1,
    )
    print(r2.report())
    print()
    print("Sub-period stability (5y windows), Friday-dip condition:")
    for row in q.compare_subperiods({"dow": 5, "ret": (-0.0075, -0.005)}):
        print(f"  {row['window']}: n={row['n']} mean={row['mean']}")
    print()
    rep = q.multiple_testing_report()
    print(f"multiple-testing: {rep['trials']} queries this session; BH-adjusted p-values: "
          + ", ".join(f"{v:.3f}" for v in rep["adjusted_pvalues"]))


def main(argv=None):
    p = argparse.ArgumentParser(prog="nq_research")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="download/caches all raw data")
    sub.add_parser("build", help="assemble sessions table in DuckDB")
    sub.add_parser("query-example", help="run the spec's example question")
    sub.add_parser("all", help="ingest + build")
    args = p.parse_args(argv)
    if args.cmd == "ingest":
        cmd_ingest(args)
    elif args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "query-example":
        cmd_query_example(args)


if __name__ == "__main__":
    main()