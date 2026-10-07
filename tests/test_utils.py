import numpy as np
import pandas as pd
import pytest

from src.utils import (
    df_summary, categorical_profile, target_rate_by_bin,
    detect_outliers_iqr, check_leakage, suggest_column_groups,
)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "TARGET": rng.binomial(1, 0.08, n),
        "AGE": rng.integers(18, 70, n).astype(float),
        "EDU": rng.choice(["low", "mid", "high"], n, p=[0.2, 0.5, 0.3]),
        "FLAG_X": rng.integers(0, 2, n),
        "CONST": 1,
        "LEAKY": np.nan,
    })
    df.loc[rng.random(n) < 0.3, "AGE"] = np.nan
    # leakage-фича: таргет + шум
    df["LEAKY"] = (df["TARGET"] + rng.integers(0, 2, n)).clip(0, 1)
    return df


def test_df_summary_has_all_rows(toy):
    s = df_summary(toy)
    assert len(s) == toy.shape[1]
    assert s.loc["CONST", "n_unique"] == 1


def test_categorical_profile_iv_positive(toy):
    p = categorical_profile(toy, "EDU", "TARGET")
    assert p.attrs["iv_total"] >= 0
    assert (p["count"].sum() == len(toy))


def test_target_rate_by_bin_monotonic_flag(toy):
    p = target_rate_by_bin(toy, "LEAKY", "TARGET", bins=5)
    assert p.attrs["iv"] > 0.1
    assert isinstance(p.attrs["monotonic"], bool)


def test_outliers_iqr(toy):
    o = detect_outliers_iqr(toy, "AGE")
    assert o["n"] > 0 and o["iqr"] >= 0


def test_check_leakage_flags_leaky(toy):
    assert check_leakage(toy, "LEAKY", "TARGET")["flag_high_iv"]


def test_suggest_column_groups(toy):
    g = suggest_column_groups(toy, id_cols=["SK_ID_CURR"], target="TARGET")
    assert "CONST" in g["drop_constants"]
    assert "EDU" in g["cat_cols"]
    assert "FLAG_X" in g["flag_cols"]

