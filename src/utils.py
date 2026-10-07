"""EDA helpers for credit-scoring PD model.

Все функции read-only: входной DataFrame не мутируется.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "df_summary",
    "plot_missing",
    "categorical_profile",
    "target_rate_by_bin",
    "detect_outliers_iqr",
    "check_leakage",
    "time_stability",
    "suggest_column_groups",
]

_NA_TOKEN = "__MISSING__"


# ---------------------------------------------------------------- #
# Шаг 1. Инвентаризация
# ---------------------------------------------------------------- #
def df_summary(df: pd.DataFrame, sort_by: str = "pct_missing") -> pd.DataFrame:
    """Один вызов — полный обзор колонок.

    Возвращает таблицу: dtype, n_missing, pct_missing, n_unique, sample.
    Сортировка по pct_missing вытаскивает проблемные колонки наверх.
    """
    summary = pd.DataFrame({
        "dtype":       df.dtypes.astype(str),
        "n_missing":   df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique":    df.nunique(dropna=True),
        "sample":      df.iloc[0] if len(df) else np.nan,
    })
    if sort_by in summary.columns:
        summary = summary.sort_values(sort_by, ascending=False)
    return summary


# ---------------------------------------------------------------- #
# Шаг 3. Пропуски
# ---------------------------------------------------------------- #
def plot_missing(
    df: pd.DataFrame,
    thresholds: Iterable[float] = (50, 90),
    figsize: tuple[float, float] = (10, 8),
):
    """Bar chart % пропусков с пороговыми линиями.

    Возвращает (fig, ax), (чтобы можно было дорисовать снаружи).
    """
    miss = df.isna().mean().mul(100)
    miss = miss[miss > 0].sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    if len(miss) == 0:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        return fig, ax

    ax.barh(miss.index, miss.values, color="#4C72B0")
    for t in thresholds:
        ax.axvline(t, color="crimson", ls="--", lw=1, label=f"{t}%")
    ax.set_xlabel("% missing")
    ax.set_title(f"Missing values: {len(miss)} of {df.shape[1]} columns")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------- #
# Шаг 6. Категориальные
# ---------------------------------------------------------------- #
def categorical_profile(
    df: pd.DataFrame,
    col: str,
    target: str,
    rare_thresh: float = 0.01,
) -> pd.DataFrame:
    """Профиль категориальной фичи: count, %, bad_rate, WoE, IV-вклад.

    NaN трактуется как отдельная категория `__MISSING__` .

    WoE считается с Laplace-smoothing (+0.5), чтобы избежать log(0).
    """
    s = df[col].astype("object").fillna(_NA_TOKEN)
    y = df[target].astype(int)

    g = pd.DataFrame({"cat": s, "y": y}).groupby("cat", observed=True)["y"]
    count = g.size()
    bad   = g.sum()
    good  = count - bad
    n_bad, n_good = int(bad.sum()), int(good.sum())
    k = len(count)

    p_bad  = (bad  + 0.5) / (n_bad  + 0.5 * k)
    p_good = (good + 0.5) / (n_good + 0.5 * k)

    prof = pd.DataFrame({
        "count":    count,
        "pct":      (count / len(s)).round(4),
        "bad":      bad.astype(int),
        "good":     good.astype(int),
        "bad_rate": (bad / count).round(4),
        "woe":      np.log(p_bad / p_good).round(4),
    })
    prof["iv_contrib"] = ((p_bad - p_good) * np.log(p_bad / p_good)).round(5)
    prof["rare"]       = prof["pct"] < rare_thresh
    prof["extreme"]    = (prof["bad_rate"] == 0) | (prof["bad_rate"] == 1)

    prof.attrs["iv_total"] = float(prof["iv_contrib"].sum())
    prof.attrs["n_categories"] = int(k)
    return prof.sort_values("bad_rate", ascending=False)


# ---------------------------------------------------------------- #
# Шаг 7. Числовые — монотонность и IV до WoE
# ---------------------------------------------------------------- #
def target_rate_by_bin(
    df: pd.DataFrame,
    col: str,
    target: str,
    bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Bad rate по бинам числовой фичи — проверка монотонности до WoE.

    strategy='quantile' → равные по количеству наблюдений (устойчиво к выбросам).
    strategy='uniform'  → равные по ширине.
    """
    s = df[col]
    y = df[target].astype(int)
    mask_na = s.isna()
    s_nn = s.dropna()

    if len(s_nn) == 0:
        raise ValueError(f"Column '{col}' is all NaN")

    n_unique = s_nn.nunique()

    if strategy == "quantile":
        if n_unique <= bins:
            uniq = np.sort(s_nn.unique())
            if len(uniq) == 1:
                edges = np.array([-np.inf, np.inf])
            else:
                edges = np.concatenate([[-np.inf], uniq[:-1], [np.inf]])
        else:
            q_edges = s_nn.quantile(np.linspace(0, 1, bins + 1)).values
            q_edges = np.unique(q_edges)
            edges = np.concatenate([[-np.inf], q_edges[1:-1], [np.inf]])
    else:  # uniform
        edges = np.linspace(s_nn.min(), s_nn.max(), bins + 1)

    binned = pd.cut(s, bins=edges, include_lowest=True, duplicates="drop")
    g = (
        pd.DataFrame({"bin": binned, "y": y})
        .dropna(subset=["bin"])
        .groupby("bin", observed=True)["y"]
    )

    count = g.size()
    bad   = g.sum()
    good  = count - bad
    n_bad, n_good = int(bad.sum()), int(good.sum())
    k = max(len(count), 1)

    p_bad  = (bad  + 0.5) / (n_bad  + 0.5 * k)
    p_good = (good + 0.5) / (n_good + 0.5 * k)

    out = pd.DataFrame({
        "count":    count,
        "bad_rate": (bad / count).round(4),
        "woe":      np.log(p_bad / p_good).round(4),
    })

    diffs = np.diff(out["bad_rate"].values)
    inc = bool((diffs >= 0).all())
    dec = bool((diffs <= 0).all())
    out.attrs["monotonic_increasing"] = inc
    out.attrs["monotonic_decreasing"] = dec
    out.attrs["monotonic"] = inc or dec
    out.attrs["iv"] = float(((p_bad - p_good) * np.log(p_bad / p_good)).sum())

    if mask_na.any():
        na_count = int(mask_na.sum())
        na_bad   = int(y[mask_na].sum())
        out.loc[_NA_TOKEN] = {
            "count":    na_count,
            "bad_rate": round(na_bad / na_count, 4) if na_count else np.nan,
            "woe":      np.nan,
        }
    return out


# ---------------------------------------------------------------- #
# Шаг 7. Числовые — выбросы и сенсоры
# ---------------------------------------------------------------- #
def detect_outliers_iqr(df: pd.DataFrame, col: str, k: float = 1.5) -> dict:
    """IQR-детектор выбросов. Возвращает dict со статистиками и примерами.
    """
    s = df[col].dropna()
    if len(s) == 0:
        return {"col": col, "n": 0, "error": "all NaN"}

    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    out_mask = (s < lower) | (s > upper)
    outliers = s[out_mask]

    return {
        "col":              col,
        "n":                int(len(s)),
        "q1":               q1,
        "q3":               q3,
        "iqr":              iqr,
        "lower":            lower,
        "upper":            upper,
        "min":              float(s.min()),
        "max":              float(s.max()),
        "n_outliers":       int(out_mask.sum()),
        "pct_outliers":     round(float(out_mask.mean() * 100), 2),
        "sample_outliers":  outliers.head(5).tolist(),
    }


# ---------------------------------------------------------------- #
# Шаг 8. Leakage / прокси таргета
# ---------------------------------------------------------------- #
def check_leakage(
    df: pd.DataFrame,
    col: str,
    target: str,
    high_iv: float = 0.5,
    extreme_bad_rate: float = 0.5,
    rare_pct: float = 0.01,
) -> dict:
    """Эвристики на leakage одной фичи.

    - IV_rough > high_iv → подозрительно сильная фича.
    - Бин с pct < rare_pct и bad_rate > extreme_bad_rate → подозрительный хвост.
    """
    is_cat = not pd.api.types.is_numeric_dtype(df[col])

    if is_cat:
        prof = categorical_profile(df, col, target)
        iv = float(prof.attrs.get("iv_total", 0.0))
        extreme = prof[(prof["pct"] < rare_pct) & (prof["bad_rate"] > extreme_bad_rate)]
    else:
        prof = target_rate_by_bin(df, col, target, bins=10)
        iv = float(prof.attrs.get("iv", 0.0))
        pct = prof["count"] / max(len(df), 1)
        extreme = prof[(pct < rare_pct) & (prof["bad_rate"] > extreme_bad_rate)]

    return {
        "col":                    col,
        "is_categorical":         is_cat,
        "iv_rough":               round(iv, 4),
        "flag_high_iv":           iv > high_iv,
        "flag_extreme_rare_bad":  len(extreme) > 0,
        "n_extreme_bins":         int(len(extreme)),
    }




# ---------------------------------------------------------------- #
# Шаг 4. Автоподсказка групп колонок
# ---------------------------------------------------------------- #
def suggest_column_groups(
    df: pd.DataFrame,
    id_cols: Iterable[str] = (),
    target: str | None = None,
) -> dict[str, list[str]]:
    """Грубая разметка колонок на группы для config.yaml.
    """
    id_cols = set(id_cols)
    reserved = id_cols | ({target} if target else set())

    all_cols = [c for c in df.columns if c not in reserved]

    def _is_cat(s: pd.Series) -> bool:
        return (
            not pd.api.types.is_numeric_dtype(s)
            and not pd.api.types.is_datetime64_any_dtype(s)
        )

    cat_cols = [c for c in all_cols if _is_cat(df[c])]
    num_all  = [c for c in all_cols if c not in cat_cols]

    flag_cols = [
        c for c in num_all
        if df[c].dropna().isin([0, 1]).all() and df[c].nunique(dropna=True) <= 2
    ]
    num_cols = [c for c in num_all if c not in flag_cols]

    drop_cols = [c for c in all_cols if df[c].nunique(dropna=True) <= 1]

    return {
        "id_cols":        sorted(id_cols),
        "cat_cols":       cat_cols,
        "flag_cols":      flag_cols,
        "num_cols":       num_cols,
        "drop_constants": drop_cols,
    }
