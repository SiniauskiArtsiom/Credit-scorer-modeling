"""Smoke-тесты woe.py на синтетике."""
import numpy as np
import pandas as pd
import pytest

from src.features.woe import (
    get_features, get_monotonic_trend, get_keep_nan,
    impute_value, make_binner, fit_all, transform_all, strength,
)


@pytest.fixture
def toy_cfg():
    return {
        "data": {"target": "TARGET"},
        "columns": {
            "numeric_decisions": {
                "EXT_SOURCE_1": {"monotonic_trend": "descending", "keep_nan_as_category": True},
                "DAYS_BIRTH":   {"monotonic_trend": "ascending",  "keep_nan_as_category": False},
                "AMT_CREDIT":   {"monotonic_trend": "auto",       "keep_nan_as_category": False},
            },
            "categorical_decisions": {
                "ORGANIZATION_TYPE": {"action": "keep_with_grouping"},
                "NAME_EDUCATION_TYPE": {"action": "keep", "monotonic_trend": "descending"},
                "FLAG_OWN_CAR": {"action": "recode_to_flag"},   # не должен попасть
            },
            "recode_to_categorical": ["REGION_RATING_CLIENT"],
            "categorical_extra": {
                "REGION_RATING_CLIENT": {"monotonic_trend": "ascending"},
            },
        },
    }


@pytest.fixture
def toy_train():
    n = 3000
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.1, n)
    return pd.DataFrame({
        "TARGET": y,
        "EXT_SOURCE_1": np.where(y == 1, rng.uniform(0, 0.5, n), rng.uniform(0.3, 1, n)),
        "DAYS_BIRTH": rng.integers(-25000, -7000, n),
        "AMT_CREDIT": rng.uniform(50000, 2_000_000, n),
        "ORGANIZATION_TYPE": rng.choice(
            ["Industry_high", "Industry_low", "XNA", "Other"], n),
        "NAME_EDUCATION_TYPE": rng.choice(
            ["Secondary", "Higher education", "Incomplete higher"], n, p=[0.7, 0.25, 0.05]),
        "REGION_RATING_CLIENT": pd.Categorical(rng.choice([1, 2, 3], n, p=[0.1, 0.75, 0.15])),
    })


def test_get_features(toy_cfg):
    num, cat, recode = get_features(toy_cfg)
    assert "EXT_SOURCE_1" in num
    assert "ORGANIZATION_TYPE" in cat
    assert "FLAG_OWN_CAR" not in cat   # recode_to_flag не должен попасть
    assert "REGION_RATING_CLIENT" in recode


def test_get_monotonic_trend(toy_cfg):
    assert get_monotonic_trend("EXT_SOURCE_1", "numerical", toy_cfg) == "descending"
    assert get_monotonic_trend("AMT_CREDIT", "numerical", toy_cfg) == "auto"
    assert get_monotonic_trend("REGION_RATING_CLIENT", "categorical", toy_cfg) == "ascending"
    assert get_monotonic_trend("ORGANIZATION_TYPE", "categorical", toy_cfg) == "auto"


def test_get_keep_nan(toy_cfg):
    assert get_keep_nan("EXT_SOURCE_1", "numerical", toy_cfg) is True
    assert get_keep_nan("DAYS_BIRTH", "numerical", toy_cfg) is False


def test_strength():
    assert strength(0.01) == "useless"
    assert strength(0.05) == "weak"
    assert strength(0.20) == "medium"
    assert strength(0.40) == "strong"
    assert strength(0.60) == "suspicious"


def test_fit_all_runs(toy_train, toy_cfg):
    num, cat, recode = get_features(toy_cfg)
    y = toy_train["TARGET"].values
    binners, impute, iv, failed = fit_all(
        toy_train.copy(), y, num, cat, recode, toy_cfg
    )
    assert len(binners) == len(num) + len(cat) + len(recode)
    assert failed.empty
    assert iv["iv"].notna().all()
    assert (iv["iv"] >= 0).all()


def test_transform_all(toy_train, toy_cfg):
    num, cat, recode = get_features(toy_cfg)
    y = toy_train["TARGET"].values
    binners, impute, _, _ = fit_all(toy_train.copy(), y, num, cat, recode, toy_cfg)
    dtype_map = {c: "numerical" for c in num}
    dtype_map.update({c: "categorical" for c in cat + recode})
    woe = transform_all(binners, impute, toy_train.copy(), dtype_map)
    assert all(c.endswith("_woe") for c in woe.columns)
    assert woe.notna().all().all()
    assert len(woe) == len(toy_train)


def test_days_birth_nan_imputed(toy_train, toy_cfg):
    """DAYS_BIRTH с keep_nan_as_category=False → NaN импьютируется."""
    df = toy_train.copy()
    df.loc[:10, "DAYS_BIRTH"] = np.nan
    num, cat, recode = get_features(toy_cfg)
    binners, impute, _, _ = fit_all(df, df["TARGET"].values, num, cat, recode, toy_cfg)
    assert "DAYS_BIRTH" in impute
    assert not np.isnan(impute["DAYS_BIRTH"])
