"""Smoke-тесты prepare.py на синтетических данных."""
import numpy as np
import pandas as pd
import pytest

from src.data.prepare import (
    apply_sensors, recode_to_flags, recode_to_categorical,
    apply_group_map, group_rare_categories, drop_columns,
    split_indices, prepare_test_unlabeled,
)


@pytest.fixture
def toy_train():
    n = 1000
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "TARGET": rng.binomial(1, 0.08, n),
        "DAYS_EMPLOYED": np.where(rng.random(n) < 0.2, 365243, rng.integers(-10000, -100, n)),
        "FLAG_OWN_CAR": rng.choice(["N", "Y"], n),
        "FLAG_OWN_REALTY": rng.choice(["N", "Y"], n),
        "REGION_RATING_CLIENT": rng.choice([1, 2, 3], n, p=[0.1, 0.75, 0.15]),
        "ORGANIZATION_TYPE": rng.choice(
            ["Industry: type 1", "Industry: type 2", "Industry: type 3", "XNA", "Other", "rare_cat"],
            n, p=[0.2, 0.2, 0.15, 0.2, 0.2, 0.05],
        ),
        "NAME_EDUCATION_TYPE": rng.choice(
            ["Secondary", "Higher education", "Academic degree", "Inferior"],
            n, p=[0.6, 0.3, 0.001, 0.099],
        ),
        "EXT_SOURCE_1": rng.random(n),
        "AMT_CREDIT": rng.uniform(10000, 1_000_000, n),
        "DROP_ME": rng.random(n),
    })


@pytest.fixture
def toy_test(toy_train):
    return toy_train.drop(columns=["TARGET"]).copy()


@pytest.fixture
def toy_cfg():
    return {
        "data": {"id_col": "SK_ID_CURR", "target": "TARGET"},
        "split": {"ratios": [0.7, 0.15, 0.15], "seed": 42},
        "columns": {
            "sensors": {"DAYS_EMPLOYED": {"value": 365243, "replace": None, "add_flag": True}},
            "recode_to_categorical": ["REGION_RATING_CLIENT"],
            "drop": ["DROP_ME"],
            "categorical_decisions": {
                "FLAG_OWN_CAR":   {"action": "recode_to_flag", "mapping": {"N": 0, "Y": 1}},
                "FLAG_OWN_REALTY":{"action": "recode_to_flag", "mapping": {"N": 0, "Y": 1}},
                "ORGANIZATION_TYPE": {
                    "action": "keep_with_grouping",
                    "group_map": {
                        "Industry: type 1": "Industry_high",
                        "Industry: type 2": "Industry_low",
			"Industry: type 3": "Industry_high",
                        "XNA": "XNA", "Other": "Other",
                    },
                },
                "NAME_EDUCATION_TYPE": {
                    "action": "keep",
                    "group_rare": True,
                    "rare_threshold": 0.005,
                },
            },
        },
    }


def test_sensors_creates_flag_and_nan(toy_train, toy_cfg):
    df = apply_sensors(toy_train.copy(), toy_cfg["columns"]["sensors"])
    assert "DAYS_EMPLOYED_IS_SENSOR" in df.columns
    assert df["DAYS_EMPLOYED_IS_SENSOR"].sum() > 0
    assert df.loc[df["DAYS_EMPLOYED_IS_SENSOR"] == 1, "DAYS_EMPLOYED"].isna().all()


def test_recode_to_flags(toy_train, toy_cfg):
    df = recode_to_flags(toy_train.copy(), toy_cfg["columns"]["categorical_decisions"])
    assert df["FLAG_OWN_CAR"].dtype == np.int8
    assert set(df["FLAG_OWN_CAR"].unique()).issubset({0, 1})


def test_recode_to_categorical(toy_train, toy_cfg):
    df = recode_to_categorical(toy_train.copy(), ["REGION_RATING_CLIENT"])
    assert str(df["REGION_RATING_CLIENT"].dtype) == "category"


def test_group_map_reduces_categories(toy_train, toy_cfg):
    df = apply_group_map(toy_train.copy(), toy_cfg["columns"]["categorical_decisions"])
    n_before = toy_train["ORGANIZATION_TYPE"].nunique()
    n_after = df["ORGANIZATION_TYPE"].nunique()
    assert n_after < n_before
    assert "Industry_high" in df["ORGANIZATION_TYPE"].unique()


def test_group_rare(toy_train, toy_cfg):
    df = group_rare_categories(toy_train.copy(), toy_cfg["columns"]["categorical_decisions"])
    # Academic degree (0.1%) → Other, Inferior (9.9%) остаётся
    assert "Other" in df["NAME_EDUCATION_TYPE"].unique()
    assert "Academic degree" not in df["NAME_EDUCATION_TYPE"].unique()


def test_drop_columns(toy_train, toy_cfg):
    df = drop_columns(toy_train.copy(), toy_cfg["columns"]["drop"])
    assert "DROP_ME" not in df.columns


def test_split_is_stratified(toy_train, toy_cfg):
    tr, va, te = split_indices(toy_train, toy_cfg)
    assert len(tr) + len(va) + len(te) == len(toy_train)
    # prevalence в каждой части близок к общему
    for idx in [tr, va, te]:
        assert abs(toy_train.loc[idx, "TARGET"].mean() - toy_train["TARGET"].mean()) < 0.03


def test_test_unlabeled_partial_map(toy_test, toy_cfg):
    """Категории вне group_map сохраняются как есть (partial map semantics)."""
    df_test = prepare_test_unlabeled(toy_test.copy(), toy_cfg)
    org_vals = df_test["ORGANIZATION_TYPE"].unique()
    # редкая категория, которой нет в карте, — сохраняется
    assert "rare_cat" in org_vals
    # замапленные категории — на месте
    assert "Industry_high" in org_vals
    assert "XNA" in org_vals
    # сенсор применён
    assert "DAYS_EMPLOYED_IS_SENSOR" in df_test.columns
    # DROP_ME удалён
    assert "DROP_ME" not in df_test.columns
