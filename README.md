# nq_stock_analysis

**NQ/Nasdaq-100 conditional base-rate research engine** — answers questions like
*"When Friday closes down 0.50–0.75%, what happened the following Monday — and does it
differ in the week after OPEX?"* with **distributions, sample sizes, and confidence
intervals**. Not a predictor; never emits a directional call.

Also ships a **Telegram bot (`@nqstocks_bot`)** for interactive Q&A and chart delivery.

## Example (real output, 1990–2026, ^NDX, n=9,232 sessions)

```text
Q: Friday close -0.50%..-0.75% — what happened the following Monday?

n=113 — bootstrapped 95% CI: [39.8%, 58.4%]
conditional rate 48.7% vs base 54.7% (lift -6.0%)
mean -0.16% | median -0.09% | p10 -2.3% | p90 +1.7% | p=0.20
→ no reliable edge; regime decay visible across decades

Same, post-OPEX week only:
n=27 — INSUFFICIENT SAMPLE, treat as anecdote (need ≥30)
→ suggestive (-17.6% lift), unproven. The engine refuses to call it.
```

## Architecture

```
ingest/            prices (yfinance + Cboe VOL CSVs) · FRED/ALFRED vintages ·
                   event calendar (OPEX/witching/elections) · FOMC dates ·
                   CFTC COT (TFF, release-date join)
features/          sessions-table assembly · three-candle pattern classifier
query/             ConditionalQuery + seeded bootstrap CIs + Benjamini–Hochberg
telegram_bot.py    @nqstocks_bot: /what /analyze /sessions /charts + LAN chart server
cli.py             ingest → build → query-example
tests/             50 tests — look-ahead tripwires ARE the test suite
```

35-column `sessions` table in DuckDB: price action, event proximity (OPEX/FOMC/CPI/NFP/
elections), VIX regime + term structure, candle patterns, COT positioning percentiles.

## Guarantees (all test-enforced)

- **No look-ahead** — COT joins on *release* date (Tue snapshot ≠ usable till Fri 3:30pm ET);
  macro uses point-in-time ALFRED vintages, never revised values
- **No probability without n** — every stat carries sample size + bootstrapped CI;
  `n < 30` → `INSUFFICIENT SAMPLE, treat as anecdote` in the headline
- **Continuity tripwire** — any |daily move| > 25% is flagged (data errors, roll gaps)
- **Reproducible** — seeded bootstrap, deterministic builds
- **Multiple-testing honesty** — session trial counter + BH-adjusted p-values

## Quick start

```bash
uv sync --extra test     # or: pip install -e ".[test]"
make ingest              # pulls ^NDX/QQQ/^GSPC/NQ=F/^VIX/^VIX3M/^VVIX + COT → data/raw (parquet)
make build               # assembles sessions table → data/nq_research.duckdb
make query               # runs the example question, honest report
make test                # full suite

# optional point-in-time macro
export FRED_API_KEY=...  # free from fred.gov
make ingest
```

## Telegram bot

```bash
cp .env.example .env      # NQ_BOT_TOKEN from @BotFather, NQ_CHAT_ID=your id
bash run_bot_forever.sh   # long-poll + LAN interactive-chart server :8791
```

| Command | What you get |
|---|---|
| `/what 2026-08-28` | full session card (VIX regime, pattern, events, forward 1/5/10/20d) |
| `/what 2026-08-28 13:45` | + minute-level hi/lo (recent sessions) |
| `/analyze friday retpct:-0.75..-0.5 post-opex` | conditional base-rate, n + CI first |
| `/sessions years=10` | rendered chart PNG pushed into chat |
| `/charts` | links to live hover/zoom plotly charts (LAN) |
| `/syntax` | filter vocabulary |

## Honest limitations

- `NQ=F` stored but never used for returns (unadjusted front-month roll)
- FOMC dates from a checked-in CSV (1990–2026); fed.gov scrape is fallback only
- CPI/NFP event proximity uses day-10/day-5 calendar proxies
- TFF positioning exists only from June 2010
- Macro columns stay NaN without `FRED_API_KEY`

## Status

Phase 1 complete: engine + tests + bot. Phase 2 (natural-language query layer) not started.