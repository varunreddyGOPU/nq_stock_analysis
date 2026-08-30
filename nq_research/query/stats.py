"""Bootstrap CIs and multiple-testing correction.

Every reported probability must pass through here so n and CI are never optional.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci_rate(
    wins: "np.ndarray | list[bool]",
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI on a win rate. Deterministic for a given seed."""
    arr = np.asarray(wins, dtype=bool)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    rates = arr[idx].mean(axis=1)
    lo, hi = np.quantile(rates, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def bootstrap_ci_mean(
    x: "np.ndarray | list[float]",
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI on a mean."""
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH q-values; enforces monotonicity via the cumulative-minimum from the largest."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        raw = pvals[i] * m / rank
        prev = min(prev, raw)
        q[i] = min(prev, 1.0)
    return q