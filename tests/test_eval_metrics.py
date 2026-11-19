import numpy as np
import pandas as pd
import pytest

from nowcaster.eval import metrics as M


def test_point_metrics_nan_and_edges():
    y = [100, 200, np.nan, 0]
    f = [110, 180, 50, 5]
    assert M.mae(y, f) == pytest.approx((10 + 20 + 5) / 3)
    assert M.mape(y, f) == pytest.approx((0.1 + 0.1) / 2)  # y<=0 and NaN ignored
    assert np.isnan(M.mae([np.nan], [1]))
    assert M.skill([100, 100], [100, 100], [90, 110]) == pytest.approx(1.0)
    assert np.isnan(M.skill([100], [99], [100]))  # zero-error reference


def test_skill_uses_common_rows():
    y = np.array([100.0, 100, 100])
    a = np.array([101.0, 101, np.nan])
    ref = np.array([102.0, 102, 200])
    assert M.skill(y, a, ref) == pytest.approx(0.5)


def test_pinball_and_intervals():
    y = np.array([0.0, 1, 2, 3])
    assert M.pinball(y, y, 0.9) == 0
    assert M.pinball([1.0], [0.0], 0.9) == pytest.approx(0.9)
    assert M.pinball([0.0], [1.0], 0.9) == pytest.approx(0.1)
    assert M.coverage(y, y - 0.5, y + 0.5) == 1
    assert M.coverage([5.0, 5], [0, 6], [10, 7]) == 0.5
    assert M.mean_width([0, 1], [2, 5]) == 3


def test_brier_and_reliability():
    y = np.array([1, 0, 1, 0])
    assert M.brier(y, y) == 0
    assert M.brier_skill(y, [0.5] * 4, 0.5) == pytest.approx(0)
    assert M.brier_skill(y, y, 0.5) == pytest.approx(1)
    rel = M.reliability_bins([0, 1, 1, 1], [0.05, 0.95, 0.95, 1.0], n_bins=10)
    assert rel["n"].sum() == 4 and rel.loc[9, "n"] == 3 and rel.loc[9, "obs_freq"] == 1


def test_horizon_buckets():
    b = M.horizon_bucket([1, 3, 4, 6, 7, 12, 13, 24, 25])
    assert list(b) == ["1-3", "1-3", "4-6", "4-6", "7-12", "7-12", "13-24", "13-24", "25+"]


def test_score_table_identical_rows():
    df = pd.DataFrame({
        "ba": ["X"] * 4, "horizon": [1, 2, 5, 5],
        "actual": [100.0, 100, 100, 100],
        "M3": [101.0, np.nan, 102, 104],
        "B3": [110.0, 110, 110, 110],
        "B4": [105.0, 105, 105, 105],
    })
    t = M.score_table(df, ["M3"])
    m3 = t[(t.model == "M3") & (t.bucket == "1-3")].iloc[0]
    b3 = t[(t.model == "B3") & (t.bucket == "1-3")].iloc[0]
    assert m3.n == 1 and b3.n == 1  # NaN row dropped for every model
    assert m3.skill_vs_B3 == pytest.approx(0.9)
    b4 = t[(t.model == "B4") & (t.bucket == "4-6")].iloc[0]
    assert b4.skill_vs_B4 == 0
    assert t[(t.model == "M3") & (t.bucket == "4-6")].iloc[0].skill_vs_B4 == pytest.approx(0.4)
