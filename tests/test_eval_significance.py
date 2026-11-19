import numpy as np
import pandas as pd
import pytest

from nowcaster.eval import significance as S


def _ar1(rng, n, phi=0.5):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_dm_null_size_roughly_nominal():
    rng = np.random.default_rng(1)
    rej = 0
    n_sim = 300
    for _ in range(n_sim):
        d = _ar1(rng, 200, 0.3)
        r = S.dm_test(d, np.zeros_like(d), h=1, lag=S.auto_lag(200))
        rej += r["pvalue"] < 0.05
    assert 0.01 < rej / n_sim < 0.12


def test_dm_null_pvalues_uniformish():
    rng = np.random.default_rng(2)
    p = [S.dm_test(rng.standard_normal(150), rng.standard_normal(150))["pvalue"] for _ in range(300)]
    assert 0.4 < np.mean(p) < 0.6


def test_dm_detects_difference():
    rng = np.random.default_rng(3)
    a = np.abs(rng.standard_normal(300)) + 0.3
    b = np.abs(rng.standard_normal(300))
    r = S.dm_test(a, b)
    assert r["pvalue"] < 1e-3 and r["stat"] > 0 and r["mean_diff"] > 0


def test_dm_degenerate():
    assert np.isnan(S.dm_test([1, 1, 1, 1], [1, 1, 1, 1])["pvalue"])
    assert np.isnan(S.dm_test([1, 2], [0, 0])["pvalue"])


def test_dm_daily_aggregates_per_day():
    rng = np.random.default_rng(4)
    days = np.repeat(np.arange(60), 24)
    shock = np.repeat(rng.standard_normal(60) * 5, 24)  # shared within-day shock
    actual = 100 + shock + rng.standard_normal(days.size)
    df = pd.DataFrame({"op_day": days, "actual": actual,
                       "A": actual + rng.normal(0, 1.0, days.size),
                       "B": actual + rng.normal(0, 2.0, days.size)})
    r = S.dm_daily(df, "A", "B")
    assert r["n_days"] == 60 and r["n_rows"] == 1440
    assert r["stat"] < 0 and r["pvalue"] < 0.01


def test_holm():
    adj = S.holm([0.01, 0.04, 0.03, np.nan])
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)  # monotone
    assert np.isnan(adj[3])


@pytest.mark.parametrize("method,L", [("moving", 1), ("moving", 5), ("stationary", 5)])
def test_bootstrap_covers_true_mean(method, L):
    rng = np.random.default_rng(5)
    hits = 0
    for s in range(40):
        df = pd.DataFrame({"day": np.repeat(np.arange(80), 3), "x": rng.normal(1.0, 1.0, 240)})
        ci = S.block_bootstrap_ci(df, lambda d: d["x"].mean(), "day", n_boot=200,
                                  block_len=L, method=method, seed=s)
        hits += ci["lo"] <= 1.0 <= ci["hi"]
    assert hits / 40 >= 0.8


def test_bootstrap_reproducible():
    df = pd.DataFrame({"day": np.arange(30), "x": np.arange(30.0)})
    f = lambda d: d["x"].mean()  # noqa: E731
    a = S.block_bootstrap_ci(df, f, "day", n_boot=100, seed=7)
    b = S.block_bootstrap_ci(df, f, "day", n_boot=100, seed=7)
    assert a == b and a["estimate"] == pytest.approx(14.5)
