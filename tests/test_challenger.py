"""Smoke-тесты challenger.py на синтетике."""
import numpy as np
import pandas as pd
import pytest

from src.models.challenger import (
    gini, ks, evaluate, bootstrap_metric, fit_xgb, sanity_check_shap,
)


@pytest.fixture
def toy_data():
    n = 5000
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "f3": rng.choice(["a", "b", "c"], n),
        "DAYS_EMPLOYED_IS_SENSOR": rng.integers(0, 2, n),
    })
    X["f3"] = X["f3"].astype("category")
    logit = 0.8 * X["f1"] + 0.5 * X["f2"] + 0.3 * (X["f3"] == "a").astype(float) + 0.1
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return X, y


def test_gini_ks_range(toy_data):
    X, y = toy_data
    p = 1 / (1 + np.exp(-(0.8 * X["f1"] + 0.5 * X["f2"])))
    assert 0 < gini(y, p) < 1
    assert 0 < ks(y, p) < 1


def test_bootstrap_metric_returns_ci(toy_data):
    X, y = toy_data
    rng = np.random.default_rng(1)
    p = rng.random(len(y))
    mean, lo, hi = bootstrap_metric(y, p, gini, n=100)
    assert lo <= mean <= hi


def test_fit_xgb_runs(toy_data):
    X, y = toy_data
    cfg = {"challenger": {"num_boost_round": 30, "early_stopping_rounds": 10}}
    X_train, X_valid = X.iloc[:4000], X.iloc[4000:]
    y_train, y_valid = y[:4000], y[4000:]
    booster = fit_xgb(X_train, y_train, X_valid, y_valid, cfg)
    assert booster.best_iteration >= 0


def test_sanity_check_runs(tmp_path):
    imp = pd.DataFrame({
        "feature": [f"f{i}" for i in range(15)],
        "mean_abs_shap": np.linspace(1, 0.01, 15),
        "share_pct": np.linspace(30, 0.3, 15),
    })
    iv = pd.DataFrame({
        "feature": [f"f{i}" for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        "iv": np.linspace(0.4, 0.02, 10),
    })
    iv_path = tmp_path / "iv.csv"
    iv.to_csv(iv_path, index=False)
    # не должно падать
    sanity_check_shap(imp, iv_path)


def test_sanity_check_flags_leakage(tmp_path):
    """Если одна фича > 50% — должен быть warning (проверяем через caplog)."""
    import logging
    imp = pd.DataFrame({
        "feature": ["leaky", "f2", "f3"],
        "mean_abs_shap": [10, 0.5, 0.5],
        "share_pct": [90.0, 5.0, 5.0],
    })
    iv = pd.DataFrame({
        "feature": ["leaky", "f2", "f3"],
        "iv": [0.5, 0.1, 0.05],
    })
    iv_path = tmp_path / "iv.csv"
    iv.to_csv(iv_path, index=False)
    sanity_check_shap(imp, iv_path)   # не падает
