# nq-research

Conditional base-rate research engine for Nasdaq-100 daily behavior. Answers
questions like *"When Friday closes down 0.50–0.75%, what happened the following
Monday — and does it differ in the week after OPEX?"* with distributions, sample
sizes, and confidence intervals. **Not a predictor; never emits a directional call.**

## Quick start

```bash
uv sync --extra test          # or: pip install -e ".[test]"
make ingest                   # downloads price/vol/COT caches to data/raw/ (parquet)
make build                    # assembles the sessions table -> data/nq_research.duckdb
make query                    # runs the example question + multiple-testing report
make test                     # 50 tests, incl. the no-look-ahead suite
```

Optional: export `FRED_API_KEY` (free) before `make ingest` to add point-in-time
macro columns (CPI YoY as-reported via ALFRED vintages, Fed funds, curve).

## The example question, answered (1990–2026, ^NDX)

```
Friday close -0.50%..-0.75%  ->  next Monday:    n=113, up-rate 48.7% vs base 54.7%
    95% CI [39.8%, 58.4%] · mean -0.16% · p=0.20  →  no reliable edge
Same, post-OPEX week only:       n=27 — INSUFFICIENT SAMPLE, treat as anecdote
    up-rate 37.0% · mean -0.51% · p=0.066        →  suggestive, unproven
```

## What's inside

| Module | Role |
|---|---|
| `ingest/prices.py` | yfinance daily bars (9,232 ^NDX sessions) + Cboe VOL CSVs (VIX3M via official source) |
| `ingest/macro.py` | FRED levels + ALFRED **vintage** series; `cpi_yoy_as_reported()` is point-in-time |
| `ingest/calendar.py` | OPEX / triple witching / elections / quarter-end — pure computation |
| `ingest/fomc.py` | FOMC dates: checked-in CSV (1990–2026) + fed.gov scrape fallback |
| `ingest/cot.py` | CFTC TFF (NQ combined `20974+`, 846 weekly reports); **release-date join** — Tuesday snapshot visible only from Friday 3:30pm ET |
| `features/patterns.py` | `three_candle_pattern` vocabulary, explicit rules, no TA black box |
| `features/build.py` | 35-column sessions table: event proximity, VIX regime, price action, COT percentiles |
| `query/stats.py` | seeded bootstrap CIs + Benjamini-Hochberg correction |
| `query/conditional.py` | `ConditionalQuery.conditional()` / `.report()` / `.multiple_testing_report()` / `.compare_subperiods()`; targets incl. **triple-barrier** (López de Prado) |

## Guarantees (all test-enforced)

- **No look-ahead**: COT joined on release date (test), macro vintages as-of session date (test)
- **No probability without n**: every result carries `n` + bootstrapped CI; `n<30` → `INSUFFICIENT SAMPLE` headline
- **Continuity**: any |daily| return > 25% is flagged (roll-gap/data-error tripwire)
- **Reproducibility**: bootstrap is seeded; identical filters → identical results
- **Multiple testing**: session trial counter with BH-adjusted p-values in every report

## Honest limitations

- `NQ=F` is stored but never used for returns (unadjusted front-month roll)
- FOMC dates from checked-in CSV through 2026 (editable: `data/fomc_dates.csv`)
- CPI/NFP event *proximity* uses day-10/day-5 calendar proxies, not the exact BLS calendar
- TFF positioning exists only from June 2010; earlier sessions have NaN percentiles
- Macro columns stay NaN unless `FRED_API_KEY` is set before ingest