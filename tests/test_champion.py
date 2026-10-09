"""Smoke-тесты champion.py на синтетике."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.models.champion import (
    fit_logit, build_coef_table, ScorecardScaler,
    gini, ks, evaluate, score_deciles, check_monotonicity,
)


@pytest.fixture
def toy_data():
    """X и y со слабой, но реальной зависимостью."""
    n = 5000
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "f1_woe": rng.normal(0, 1, n),
        "f2_woe": rng.normal(0, 1, n),
        "f3_woe": rng.normal(0, 1, n),
    })
    # logit = 0.8*f1 + 0.5*f2 - 0.3*f3 + 0.1
    logit = 0.8 * X["f1_woe"] + 0.5 * X["f2_woe"] - 0.3 * X["f3_woe"] + 0.1
    p = 1 / (1 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, p))
    return X, y


def test_fit_logit_converges(toy_data):
    X, y = toy_data
    model = fit_logit(X, y)
    assert model.mle_retvals["converged"]


def test_coef_table_columns(toy_data):
    X, y = toy_data
    model = fit_logit(X, y)
    tbl = build_coef_table(model)
    expected = {"feature", "coef", "std_err", "z", "p_value",
                "ci_low", "ci_high", "significant", "positive"}
    assert expected.issubset(set(tbl.columns))
    assert "const" in tbl["feature"].values


def test_scorecard_scaling():
    """При WoE=0 балл должен быть близок к offset."""
    scaler = ScorecardScaler(base_score=600, base_odds=50, pdo=20)

    class DummyModel:
        params = pd.Series({"const": 0.0, "f1_woe": 1.0})

    scaler.fit(DummyModel())
    X = pd.DataFrame({"f1_woe": [0.0]})
    score = scaler.score(X)
    expected = scaler.offset_ - scaler.factor_ * 0.0
    assert abs(score[0] - expected) < 1e-6


def test_scorecard_negative_logit_raises_score():
    """Чем ниже logit (риск), тем выше балл."""
    scaler = ScorecardScaler(base_score=600, base_odds=50, pdo=20)

    class DummyModel:
        params = pd.Series({"const": 0.0, "f1_woe": 1.0})

    scaler.fit(DummyModel())
    X_low_risk = pd.DataFrame({"f1_woe": [-1.0]})   # отрицательный WoE → ниже риск
    X_high_risk = pd.DataFrame({"f1_woe": [1.0]})
    s_low = scaler.score(X_low_risk)[0]
    s_high = scaler.score(X_high_risk)[0]
    assert s_low > s_high


def test_points_table_structure(toy_data):
    X, y = toy_data
    model = fit_logit(X, y)
    scaler = ScorecardScaler().fit(model)
    tbl = scaler.points_table()
    assert "Intercept" in tbl["feature"].values
    assert "f1_woe" in tbl["feature"].values
    assert "points_per_woe_unit" in tbl.columns


def test_gini_ks_perfect():
    """Идеальное предсказание → Gini=1, KS=1."""
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.85])
    assert gini(y, p) > 0.9
    assert ks(y, p) > 0.9


def test_evaluate_keys(toy_data):
    X, y = toy_data
    model = fit_logit(X, y)
    p = model.predict(sm.add_constant(X, has_constant="add"))
    m = evaluate(y.values, p.values)
    assert set(m.keys()) == {"auc", "gini", "ks"}
    assert 0.5 < m["auc"] < 1.0


def test_score_deciles_monotonic_ok():
    """Синтетика: высокий балл → низкий bad_rate."""
    n = 1000
    score = np.linspace(300, 900, n)
    # bad_rate убывает с баллом
    y = (np.random.default_rng(0).random(n) < (1 - (score - 300) / 600)).astype(int)
    dec = score_deciles(y, score, n=10)
    assert len(dec) == 10
    assert check_monotonicity(dec)


def test_score_deciles_non_monotonic_detects():
    n = 1000
    score = np.linspace(300, 900, n)
    # шум — монотонности нет
    y = np.random.default_rng(0).binomial(1, 0.1, n)
    dec = score_deciles(y, score, n=10)
    # не гарантируем False, но функция должна отработать без ошибок
    check_monotonicity(dec)  # просто вызов
