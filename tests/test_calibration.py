"""Smoke-тесты calibration.py на синтетике."""
import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from src.evaluation.calibration import (
    reliability_table, hosmer_lemeshow, gains_table,
    fit_isotonic, apply_isotonic, gini, ks, evaluate,
)


def test_reliability_table_shape():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = rng.binomial(1, p)
    tbl = reliability_table(y, p, n_bins=10)
    assert len(tbl) == 10
    assert {"n", "pred_mean", "actual_rate"}.issubset(tbl.columns)


def test_hl_calibrated():
    """Хорошо калиброванная модель → p > 0.05."""
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = rng.binomial(1, p)
    _, pval = hosmer_lemeshow(y, p, g=10)
    assert pval > 0.05


def test_hl_uncalibrated():
    """Сильно смещённые PD → p < 0.05."""
    rng = np.random.default_rng(0)
    p = rng.random(5000) * 0.5      # PD занижены
    y = rng.binomial(1, p * 2)      # реальный bad rate вдвое выше
    _, pval = hosmer_lemeshow(y, p, g=10)
    assert pval < 0.05


def test_isotonic_fixes_calibration():
    rng = np.random.default_rng(0)
    p_val = rng.random(5000)
    y_val = rng.binomial(1, p_val)
    p_test = rng.random(5000) * 0.5      # сдвинутые
    y_test = rng.binomial(1, p_test * 2)
    iso = fit_isotonic(y_val, p_val)
    p_cal = apply_isotonic(iso, p_test)
    assert p_cal.min() >= 0 and p_cal.max() <= 1
    # монотонность
    order = np.argsort(p_test)
    assert (np.diff(p_cal[order]) >= -1e-9).all()


def test_gains_monotone():
    rng = np.random.default_rng(0)
    # сильный сигнал
    y = rng.binomial(1, 0.1, 10000)
    p = y + rng.normal(0, 0.5, 10000)   # p коррелирует с y
    tbl = gains_table(y, p, n=10)
    assert len(tbl) == 10
    # capture_rate монотонно растёт
    assert (np.diff(tbl["capture_rate"]) >= 0).all()
    # top-10% capture > 10% (больше случайного)
    assert tbl.iloc[0]["capture_rate"] > 0.10


def test_gains_no_signal():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.1, 10000)
    p = rng.random(10000)
    tbl = gains_table(y, p, n=10)
    # при отсутствии сигнала top-10% capture ≈ 10%
    assert 0.05 < tbl.iloc[0]["capture_rate"] < 0.20


def test_evaluate_keys():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = rng.binomial(1, p)
    m = evaluate(y, p)
    assert set(m.keys()) == {"auc", "gini", "ks", "brier"}
