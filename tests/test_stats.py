"""Slice 7 RED: bootstrap CI (seeded) + Benjamini-Hochberg adjustment."""
import numpy as np
import pytest

from nq_research.query.stats import benjamini_hochberg, bootstrap_ci_rate, bootstrap_ci_mean


def test_bootstrap_ci_contains_known_rate_narrow():
    rng = np.random.default_rng(42)
    wins = rng.random(1000) < 0.60
    lo, hi = bootstrap_ci_rate(wins, n_resamples=10_000, seed=7)
    assert lo <= 0.60 <= hi
    assert hi - lo < 0.07           # n=1000 -> CI half-width ~3%


def test_bootstrap_ci_wide_at_small_n():
    rng = np.random.default_rng(42)
    wins = rng.random(15) < 0.60
    lo, hi = bootstrap_ci_rate(wins, n_resamples=10_000, seed=7)
    assert hi - lo > 0.3            # n=15 -> CI spans tens of points


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(1)
    wins = rng.random(500) < 0.5
    a = bootstrap_ci_rate(wins, seed=123)
    b = bootstrap_ci_rate(wins, seed=123)
    assert a == b


def test_bootstrap_ci_mean_synthetic():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 2.0, size=1000)
    lo, hi = bootstrap_ci_mean(x, n_resamples=10_000, seed=7)
    assert lo < 1.0 < hi
    assert hi - lo < 0.5


def test_benjamini_hochberg_monotone_and_bounded():
    pvals = [0.001, 0.008, 0.039, 0.041, 0.2, 0.5, 0.9]
    adj = benjamini_hochberg(pvals)
    assert len(adj) == len(pvals)
    assert all(0 <= q <= 1 for q in adj)
    # BH is nondecreasing when pvals sorted; q(i) >= p(i)
    assert all(q >= p for p, q in zip(pvals, adj))
    # first two should survive BH at FDR 0.05
    assert adj[0] < 0.05 and adj[1] < 0.05
    assert adj[-1] == pytest.approx(0.9)


def test_benjamini_hochberg_enforces_monotonicity():
    # classic case where raw q(i+1) < q(i) requires cumulative-min fix
    pvals = [0.01, 0.02, 0.03, 0.04, 0.05]
    adj = benjamini_hochberg(pvals)
    assert adj == sorted(adj)