"""Smoke-тесты cutoff.py на синтетике."""
import numpy as np
import pytest

from src.evaluation.cutoff import (
    metrics_at_cutoff, profit_at_cutoff, sweep_cutoffs, find_optimal,
    make_business_for_cost_ratio, sensitivity_analysis,
)


@pytest.fixture
def toy_preds():
    rng = np.random.default_rng(0)
    n = 5000
    # 8% bad rate, PD информативен
    y = rng.binomial(1, 0.08, n)
    p = np.where(y == 1, rng.beta(2, 3, n), rng.beta(1.5, 8, n))
    return y, p


@pytest.fixture
def business():
    return {"EAD_mean": 500_000, "LGD": 0.60, "rate_total": 0.15,
            "cost_ratio_range": [2, 4, 6]}


def test_metrics_at_cutoff(toy_preds):
    y, p = toy_preds
    m = metrics_at_cutoff(y, p, cutoff=0.5)
    assert m["n_approved"] + m["n_rejected"] == len(y)
    assert m["good_approved"] + m["bad_approved"] == m["n_approved"]


def test_profit_at_cutoff(toy_preds, business):
    y, p = toy_preds
    m = profit_at_cutoff(y, p, cutoff=0.5, business=business)
    assert "profit_total" in m
    assert "expected_loss_total" in m
    assert m["expected_loss_total"] >= 0


def test_sweep_returns_df(toy_preds, business):
    y, p = toy_preds
    df = sweep_cutoffs(y, p, business, n_points=50)
    assert len(df) > 10
    assert "profit_total" in df.columns
    assert "cutoff" in df.columns


def test_find_optimal_inside_range(toy_preds, business):
    y, p = toy_preds
    df = sweep_cutoffs(y, p, business, n_points=50)
    opt = find_optimal(df)
    assert df["cutoff"].min() <= opt["cutoff"] <= df["cutoff"].max()


def test_make_business_cost_ratio(business):
    b = make_business_for_cost_ratio(business, cost_ratio=4)
    # cost_ratio = LGD / rate = 4 → LGD = 4*0.15 = 0.6
    assert abs(b["LGD"] - 0.6) < 1e-9


def test_sensitivity_runs(toy_preds, business):
    y, p = toy_preds
    sens = sensitivity_analysis(y, p, business)
    assert len(sens) == len(business["cost_ratio_range"])
    # чем выше cost_ratio, тем ниже (или равен) optimal_cutoff
    # потому что FN дороже → одобряем меньше
    cutoffs = sens["optimal_cutoff"].values
    assert (np.diff(cutoffs) <= 1e-6).all()
