"""ConditionalQuery: honest conditional base-rates over the sessions table.

Design law: no probability leaves this module without n and a bootstrapped CI;
n<30 is flagged INSUFFICIENT SAMPLE in .report() text.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from nq_research.query.stats import benjamini_hochberg, bootstrap_ci_mean, bootstrap_ci_rate

MIN_N = 30
HORIZON_MAX = 60


# ---------- targets ----------

def _target_values(table: pd.DataFrame, target: str, horizon: int, barrier_params: dict | None):
    """Compute per-row target arrays aligned to table order (no look-ahead: row t gets outcomes of t+1..t+h)."""
    close = table["close"].astype(float).values
    n = len(table)
    out = {}

    if target == "next_session_return":
        h = max(1, horizon)
        fwd = np.full(n, np.nan)
        for i in range(n - h):
            fwd[i] = close[i + h] / close[i] - 1
        out["values"] = fwd
        out["kind"] = "continuous"
        out["win"] = fwd > 0

    elif target == "next_session_direction":
        fwd = np.full(n, np.nan)
        for i in range(n - max(1, horizon)):
            fwd[i] = close[i + max(1, horizon)] / close[i] - 1
        out["values"] = fwd
        out["kind"] = "binary"
        out["win"] = fwd > 0

    elif target in ("max_favorable_excursion", "max_adverse_excursion"):
        h = max(1, horizon)
        fwd = np.full(n, np.nan)
        # entries happen at next session's open approximation (close of t); measure within t+1..t+h
        highs = table["close"].astype(float).values  # conservative: close-based excursion (no raw high in sessions)
        for i in range(n - h):
            path = close[i + 1: i + 1 + h]
            if target == "max_favorable_excursion":
                fwd[i] = path.max() / close[i] - 1
            else:
                fwd[i] = path.min() / close[i] - 1
        out["values"] = fwd
        out["kind"] = "continuous"
        out["win"] = fwd > 0 if target == "max_favorable_excursion" else fwd < 0

    elif target == "triple_barrier":
        bp = barrier_params or {}
        pt_mult = float(bp.get("pt_vol", 1.0))
        sl_mult = float(bp.get("sl_vol", 1.0))
        h = max(1, horizon)
        ret = table["ret"].astype(float).values
        vol20 = pd.Series(ret).rolling(20).std().values
        labels = np.full(n, np.nan, dtype=object)
        for i in range(n - h):
            v = vol20[i]
            if v is None or np.isnan(v) or v <= 0:
                continue
            entry = close[i]
            pt = entry * (1 + pt_mult * v)
            sl = entry * (1 - sl_mult * v)
            label = "time"
            for j in range(i + 1, i + 1 + h):
                if close[j] >= pt:
                    label = "pt"
                    break
                if close[j] <= sl:
                    label = "sl"
                    break
            labels[i] = label
        out["values"] = labels
        out["kind"] = "categorical"
        out["win"] = np.array([x == "pt" for x in labels])
    else:
        raise ValueError(f"unknown target: {target}")
    return out


def _apply_filters(table: pd.DataFrame, filters: dict) -> pd.Series:
    mask = pd.Series(True, index=table.index)
    for col, spec in filters.items():
        if col not in table.columns:
            raise ValueError(f"unknown filter column: {col}")
        if isinstance(spec, tuple):
            mask &= table[col].between(spec[0], spec[1])
        elif isinstance(spec, (list, set)):
            mask &= table[col].isin(list(spec))
        else:
            mask &= table[col] == spec
    return mask


@dataclass
class ConditionalResult:
    n: int
    base_rate: float
    conditional_rate: float
    lift: float
    ci_95: tuple[float, float]
    mean: float | None = None
    median: float | None = None
    p10: float | None = None
    p25: float | None = None
    p75: float | None = None
    p90: float | None = None
    by_decade: list[dict] = field(default_factory=list)
    n_warning: str | None = None
    dates: list = field(default_factory=list)
    values: list = field(default_factory=list)
    p_value: float | None = None
    target: str = ""
    horizon: int = 1
    filters: dict = field(default_factory=dict)

    def report(self) -> str:
        lines = []
        if self.n < MIN_N:
            lines.append(f"n={self.n} — INSUFFICIENT SAMPLE, treat as anecdote (need ≥{MIN_N})")
        else:
            lines.append(f"n={self.n} — bootstrapped 95% CI: [{self.ci_95[0]:.1%}, {self.ci_95[1]:.1%}]")
        lines.append(f"target: {self.target} (h={self.horizon}) | filters: {self.filters}")
        lines.append(
            f"conditional rate {self.conditional_rate:.1%} vs base {self.base_rate:.1%} "
            f"(lift {self.lift:+.1%})"
        )
        if self.mean is not None:
            lines.append(
                f"mean {self.mean:+.4f} | median {self.median:.4f} | "
                f"p10 {self.p10:.4f} | p25 {self.p25:.4f} | p75 {self.p75:.4f} | p90 {self.p90:.4f} (fractional returns)"
            )
        if self.p_value is not None:
            lines.append(f"p-value {self.p_value:.4f}")
        if self.by_decade:
            lines.append("by 5y window:")
            for w in self.by_decade:
                lines.append(f"  {w['window']}: n={w['n']} mean {w['mean'] if w['mean'] is not None else float('nan'):+.4f}")
        return "\n".join(lines)


class ConditionalQuery:
    """Query layer over nq_research.duckdb sessions table."""

    def __init__(self, db: str = "nq_research.duckdb"):
        self.db = db
        self._trials: list[dict] = []

    def _load(self) -> pd.DataFrame:
        con = duckdb.connect(self.db, read_only=True)
        try:
            return con.execute("SELECT * FROM sessions ORDER BY date").fetchdf()
        finally:
            con.close()

    def conditional(self, filters: dict, target: str, horizon: int = 1, barrier_params: dict | None = None) -> ConditionalResult:
        table = self._load()
        tinfo = _target_values(table, target, horizon, barrier_params)
        mask = _apply_filters(table, filters)

        sel = table[mask.values].copy()
        vals = np.asarray(tinfo["values"])[mask.values]
        win = np.asarray(tinfo["win"])[mask.values]

        valid = ~pd.isna(vals) if tinfo["kind"] != "categorical" else np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in vals])
        sel = sel[valid]
        vals, win = vals[valid], win[valid]
        n = int(len(sel))

        base_valid = ~pd.isna(np.asarray(tinfo["win"], dtype=float)) if tinfo["kind"] != "categorical" else np.array(
            [v is not None for v in tinfo["win"]])
        base_rate = float(np.nanmean(np.asarray(tinfo["win"], dtype=float)))

        cond_rate = float(np.mean(win)) if n else float("nan")
        # spec: ci_95 is the bootstrapped CI on the conditional (win) RATE, always
        ci = bootstrap_ci_rate(win, seed=0) if n else (float("nan"), float("nan"))

        num = pd.to_numeric(pd.Series(vals), errors="coerce") if tinfo["kind"] != "categorical" else pd.Series([np.nan] * n)
        q = num.dropna()
        percentiles = (
            dict(mean=float(num.mean()) if len(q) else None,
                 median=float(q.median()) if len(q) else None)
            | {k: float(np.quantile(q, v)) if len(q) else None for k, v in
               dict(p10=0.10, p25=0.25, p75=0.75, p90=0.90).items()}
        )

        # 5-year sub-windows (indexed only within the selected sample)
        by_dec = []
        if n:
            sel_dates = pd.to_datetime(pd.Series(sel["date"]).reset_index(drop=True))
            vals_series = pd.to_numeric(pd.Series(vals), errors="coerce").reset_index(drop=True)
            win_series = pd.Series(win).reset_index(drop=True)
            for y0 in range(int(sel_dates.dt.year.min()), int(sel_dates.dt.year.max()) + 1, 5):
                m = (sel_dates.dt.year >= y0) & (sel_dates.dt.year < y0 + 5)
                if not m.any():
                    continue
                sub_vals = vals_series[m.values]
                by_dec.append(dict(
                    window=f"{y0}-{y0 + 4}",
                    n=int(m.sum()),
                    mean=float(sub_vals.mean()) if sub_vals.notna().any() else None,
                ))

        # p-value vs base rate (two-proportion z for binary)
        p_value = None
        if n and base_rate not in (0.0, 1.0):
            p0, ph, nh = base_rate, cond_rate, n
            se = (p0 * (1 - p0) / nh) ** 0.5
            if se > 0:
                from math import erf
                z = (ph - p0) / se
                p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))
        self._trials.append(dict(filters=filters, target=target, n=n, p_value=p_value))

        return ConditionalResult(
            n=n,
            base_rate=base_rate,
            conditional_rate=cond_rate,
            lift=cond_rate - base_rate,
            ci_95=ci,
            mean=percentiles["mean"],
            median=percentiles["median"],
            p10=percentiles["p10"],
            p25=percentiles["p25"],
            p75=percentiles["p75"],
            p90=percentiles["p90"],
            by_decade=by_dec,
            n_warning=None if n >= MIN_N else f"n={n} < {MIN_N}: insufficient sample, treat as anecdote",
            dates=[pd.Timestamp(d).date() for d in sel["date"]],
            values=[v if tinfo["kind"] == "categorical" else float(v) for v in vals],
            p_value=p_value,
            target=target,
            horizon=horizon,
            filters=filters,
        )

    def multiple_testing_report(self) -> dict:
        """BH-adjust every trial run this session; surface the trial count."""
        trials = self._trials
        pvals = [t["p_value"] if t["p_value"] is not None else 1.0 for t in trials]
        adj = benjamini_hochberg(pvals)
        return dict(trials=len(trials), adjusted_pvalues=adj, raw_pvalues=pvals)

    def compare_subperiods(self, filters: dict, horizon: int = 1, window_years: int = 5) -> list[dict]:
        """Same query across rolling non-overlapping windows to expose regime decay."""
        table = self._load()
        tinfo = _target_values(table, "next_session_return", horizon, None)
        mask = _apply_filters(table, filters)
        sel = table[mask.values].copy()
        vals = np.asarray(tinfo["values"])[mask.values]
        sel["__tgt"] = vals
        out = []
        years = pd.to_datetime(sel["date"]).dt.year
        for y0 in range(years.min(), years.max() + 1, window_years):
            m = (years >= y0) & (years < y0 + window_years)
            sub = sel[m.values]
            if sub.empty:
                continue
            v = pd.to_numeric(sub["__tgt"], errors="coerce").dropna()
            out.append(dict(
                window=f"{y0}-{y0 + window_years - 1}",
                n=int(len(v)),
                mean=float(v.mean()) if len(v) else None,
                median=float(v.median()) if len(v) else None,
            ))
        return out