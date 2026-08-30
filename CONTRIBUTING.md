# Contributing to nq_stock_analysis

## Ground rules

1. **TDD is the law.** No production code without a failing test first (RED → GREEN → REFACTOR).
   The tests *are* the product here — they encode the no-look-ahead and honesty guarantees.
2. **No look-ahead, ever.** A feature on row `t` must be computable at the close of `t`.
   - COT joins on *release* date (Friday 3:30pm ET), never the Tuesday snapshot date.
   - Macro values must be ALFRED vintages as-of the session date.
   - If you add a data source, add a look-ahead tripwire test for it.
3. **No probability without `n`.** Any statistic surfaced to a user must carry its sample
   size and a bootstrapped CI. `n < 30` must render as `INSUFFICIENT SAMPLE`.
4. **This is not a predictor.** PRs that add "signals", directional calls, or ML score
   outputs without CI/distribution framing will be rejected on principle.

## Dev workflow

```bash
uv sync --extra test
uv run --with pytest python -m pytest tests/ -q     # must be green before any commit
```

## Branches

- `main` — protected-ish; history of the engine build
- `feature_vg_nq` — active development branch (PRs into `main` from here)

## Data sources

Free only: yfinance, Cboe public CSVs, FRED/ALFRED (free key), CFTC Socrata.
Credentials live in `.env` (gitignored) — see `.env.example`.