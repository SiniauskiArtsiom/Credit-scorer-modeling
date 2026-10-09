"""Smoke-тесты selection.py на синтетике."""
import numpy as np
import pandas as pd
import pytest

from src.features.selection import (
    filter_by_iv, filter_correlated, filter_vif,
    review_candidates, _compute_vifs,
    PROTECTED_FEATURES,
)


@pytest.fixture
def iv_table():
    return pd.DataFrame({
        "feature": ["A", "B", "C", "D", "E", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
        "iv":      [0.30, 0.15, 0.08, 0.015, 0.005, 0.35, 0.30, 0.25],
        "n_bins":  [10, 8, 6, 4, 3, 10, 10, 10],
        "dtype":   ["numerical"] * 8,
        "trend":   ["auto"] * 8,
        "strength": ["strong", "medium", "weak", "useless", "useless",
                     "strong", "strong", "medium"],
    })


@pytest.fixture
def toy_train():
    """A и B коррелируют сильно, A и C — нет; EXT_SOURCE_* независимы."""
    n = 2000
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, n)
    b = a + rng.normal(0, 0.1, n)        # ~0.99 correlation with A
    c = rng.normal(0, 1, n)
    return pd.DataFrame({
        "A_woe": a,
        "B_woe": b,
        "C_woe": c,
        "D_woe": rng.normal(0, 1, n),
        "E_woe": rng.normal(0, 1, n),
        "EXT_SOURCE_1_woe": rng.normal(0, 1, n),
        "EXT_SOURCE_2_woe": rng.normal(0, 1, n),
        "EXT_SOURCE_3_woe": rng.normal(0, 1, n),
        "TARGET": rng.binomial(1, 0.1, n),
    })


def test_filter_by_iv(iv_table):
    kept, dropped = filter_by_iv(iv_table, threshold=0.02)
    assert "A" in kept and "B" in kept
    assert "D" in dropped and "E" in dropped
    assert len(kept) + len(dropped) == len(iv_table)


def test_filter_correlated_drops_b(toy_train, iv_table):
    # A и B коррелируют, A выше по IV → B дропается
    features = ["A", "B", "C", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    kept, dropped = filter_correlated(toy_train, features, iv_table, threshold=0.7)
    assert "A" in kept
    assert "B" in dropped
    assert "C" in kept


def test_filter_correlated_protected(toy_train, iv_table):
    """EXT_SOURCE_* не удаляются, даже если коррелируют."""
    # искусственно создаём корреляцию
    toy_train["EXT_SOURCE_2_woe"] = toy_train["EXT_SOURCE_1_woe"] * 0.99
    features = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    kept, dropped = filter_correlated(toy_train, features, iv_table, threshold=0.7)
    assert "EXT_SOURCE_1" in kept
    assert "EXT_SOURCE_2" in kept


def test_filter_vif_removes_collinear(toy_train):
    # A и B почти идентичны — один должен уйти
    features = ["A", "B", "C"]
    kept, dropped = filter_vif(toy_train, features, threshold=5.0)
    assert len(kept) < len(features)
    # должен остаться ровно один из A/B
    assert ("A" in kept) ^ ("B" in kept)


def test_filter_vif_protected(toy_train):
    """Защищённые фичи не удаляются даже при VIF > threshold."""
    toy_train["EXT_SOURCE_2_woe"] = toy_train["EXT_SOURCE_1_woe"] * 0.99
    features = ["EXT_SOURCE_1", "EXT_SOURCE_2", "C"]
    kept, dropped = filter_vif(toy_train, features, threshold=5.0)
    assert "EXT_SOURCE_1" in kept
    assert "EXT_SOURCE_2" in kept


def test_review_candidates(iv_table):
    cfg = {"columns": {"final_review_candidates": ["D", "C"]}}  # D IV=0.015 < 0.02
    features = ["A", "B", "C", "D"]
    features, dropped = review_candidates(cfg, iv_table, features)
    assert "D" in dropped
    assert "C" not in dropped  # IV 0.08 > 0.02
    assert "D" not in features


def test_compute_vifs_runs(toy_train):
    vifs = _compute_vifs(toy_train, ["A", "B", "C"])
    assert len(vifs) == 3
    # A и B почти идентичны — VIF у обоих высокий
    assert vifs["A"] > 5
    assert vifs["B"] > 5
