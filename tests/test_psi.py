"""Smoke-тесты psi.py на синтетике."""
import numpy as np
import pandas as pd
import pytest

from src.monitoring.psi import psi_numeric, psi_categorical, psi_feature, verdict


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 5000
    return pd.Series(rng.normal(0, 1, n)), pd.Series(rng.normal(0, 1, n))


def test_psi_numeric_identical(toy):
    ref, act = toy
    assert psi_numeric(ref.values, act.values) < 0.02


def test_psi_numeric_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    act = rng.normal(0.5, 1, 5000)     # сдвиг среднего
    psi = psi_numeric(ref, act)
    assert psi > 0.10


def test_psi_numeric_extreme_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    act = rng.normal(2, 1, 5000)
    psi = psi_numeric(ref, act)
    assert psi > 0.25


def test_psi_categorical_identical():
    ref = pd.Series(["a", "b", "c"] * 1000)
    act = pd.Series(["a", "b", "c"] * 1000)
    assert psi_categorical(ref, act) < 0.02


def test_psi_categorical_shift():
    ref = pd.Series(["a"] * 500 + ["b"] * 500)
    act = pd.Series(["a"] * 300 + ["b"] * 700)
    psi = psi_categorical(ref, act)
    assert psi > 0.05


def test_psi_feature_dispatches(toy):
    ref, act = toy
    assert psi_feature(ref, act) == psi_numeric(ref.values, act.values)

    ref_c = pd.Series(["a", "b"] * 1000)
    act_c = pd.Series(["a", "b"] * 1000)
    assert psi_feature(ref_c, act_c) == psi_categorical(ref_c, act_c)


def test_verdict_thresholds():
    assert verdict(0.05) == "no_drift"
    assert verdict(0.15) == "warn"
    assert verdict(0.30) == "alert"
    assert verdict(np.nan) == "insufficient_data"
