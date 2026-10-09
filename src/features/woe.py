"""Этап 4. WoE-биннинг фич на train, применение к valid/test/test_unlabeled.

Читает data/processed/{train,valid,test,test_unlabeled}.parquet,
обучает OptimalBinning на train, сохраняет WoE-матрицы и артефакты.

Ключевое правило: fit только на train, transform — на всех выборках.
Никакого переобучения биннеров на valid/test/test_unlabeled.

Usage:
    python -m src.features.woe --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from optbinning import OptimalBinning

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STRENGTH_THRESHOLDS = [
    (0.02, "useless"),
    (0.10, "weak"),
    (0.30, "medium"),
    (0.50, "strong"),
    (float("inf"), "suspicious"),
]


# ============================================================ #
# 1. Утилиты
# ============================================================ #
def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path) -> dict:
    with open(resolve_path(path)) as f:
        return yaml.safe_load(f)


def strength(iv: float) -> str:
    for thr, label in STRENGTH_THRESHOLDS:
        if iv < thr:
            return label
    return "suspicious"


# ============================================================ #
# 2. Загрузка данных
# ============================================================ #
def load_datasets(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    processed_dir = resolve_path(processed_dir)
    names = ["train", "valid", "test", "test_unlabeled"]
    out = {}
    for name in names:
        path = processed_dir / f"{name}.parquet"
        out[name] = pd.read_parquet(path)
        logger.info("loaded %-15s %d x %d", name, *out[name].shape)
    return out


# ============================================================ #
# 3. Определение фич и их параметров
# ============================================================ #
def get_features(cfg: dict) -> tuple[list[str], list[str], list[str]]:
    """Возвращает три списка: числовые, категориальные, recode_to_categorical."""
    cat_decisions = cfg["columns"]["categorical_decisions"]
    num_cols = list(cfg["columns"]["numeric_decisions"].keys())
    cat_cols = [
        c for c, d in cat_decisions.items()
        if d.get("action") in ("keep", "keep_with_grouping","keep_with_flag")
    ]
    recode_cols = list(cfg["columns"]["recode_to_categorical"])
    return num_cols, cat_cols, recode_cols


def get_monotonic_trend(col: str, dtype: str, cfg: dict) -> str:
    if dtype == "categorical":
        extra = cfg["columns"].get("categorical_extra", {}).get(col, {})
        if extra:
            return extra.get("monotonic_trend", "auto")
        decision = cfg["columns"]["categorical_decisions"].get(col, {})
        return decision.get("monotonic_trend", "auto")
    return cfg["columns"]["numeric_decisions"].get(col, {}).get("monotonic_trend", "auto")


def get_keep_nan(col: str, dtype: str, cfg: dict) -> bool:
    if dtype == "categorical":
        return cfg["columns"]["categorical_decisions"].get(col, {}).get(
            "keep_nan_as_category", False
        )
    return cfg["columns"]["numeric_decisions"].get(col, {}).get(
        "keep_nan_as_category", False
    )


# ============================================================ #
# 4. Импьютация NaN (для фич с keep_nan_as_category=False)
# ============================================================ #
def impute_value(df_train: pd.DataFrame, col: str, dtype: str):
    if dtype == "numerical":
        v = df_train[col].median()
        return 0.0 if pd.isna(v) else float(v)
    mode = df_train[col].mode()
    return mode.iloc[0] if len(mode) else "Other"


# ============================================================ #
# 5. Фабрика биннеров
# ============================================================ #
def make_binner(col: str, dtype: str, trend: str, n_unique: int) -> OptimalBinning:
    min_bin_size = 0.02 if dtype == "categorical" else 0.05

    return OptimalBinning(
        name=col,
        dtype=dtype,
        max_n_bins=10,
        min_bin_size=min_bin_size,
        monotonic_trend=trend,
    )


def _fit_binner(df_train: pd.DataFrame, y_train: np.ndarray,
                col: str, dtype: str, trend: str) -> OptimalBinning:
    X = df_train[col].values
    binner = make_binner(col, dtype, trend, df_train[col].nunique(dropna=True))
    binner.fit(X, y_train)
    return binner


# ============================================================ #
# 6. Основной цикл fit
# ============================================================ #
def fit_all(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
    recode_cols: list[str],
    cfg: dict,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    binners: dict[str, OptimalBinning] = {}
    impute_values: dict[str, object] = {}
    iv_rows: list[dict] = []
    failed: list[dict] = []

    feats = (
        [(c, "numerical") for c in num_cols]
        + [(c, "categorical") for c in cat_cols + recode_cols]
    )

    for col, dtype in feats:
        if col not in df_train.columns:
            failed.append({"feature": col, "error": "not in train", "trend": "-"})
            continue

        keep_nan = get_keep_nan(col, dtype, cfg)
        trend = get_monotonic_trend(col, dtype, cfg)

        # импьютация NaN, если keep_nan_as_category=False
        if not keep_nan and df_train[col].isna().any():
            val = impute_value(df_train, col, dtype)
            impute_values[col] = val
            df_train[col] = df_train[col].fillna(val)

        # cast категориальных в str (optbinning требует однородный тип)
        if dtype == "categorical":
            df_train[col] = df_train[col].astype(str)

        # основной fit, с fallback на 'auto'
        try:
            binner = _fit_binner(df_train, y_train, col, dtype, trend)
            used_trend = trend
        except Exception as e_main:
            if trend != "auto":
                logger.warning("fit failed for %s (trend=%s): %s → fallback auto",
                               col, trend, e_main)
                try:
                    binner = _fit_binner(df_train, y_train, col, dtype, "auto")
                    used_trend = "auto (fallback)"
                except Exception as e_auto:
                    logger.error("fit failed for %s even with auto: %s", col, e_auto)
                    failed.append({"feature": col, "error": str(e_auto), "trend": "auto"})
                    continue
            else:
                logger.error("fit failed for %s: %s", col, e_main)
                failed.append({"feature": col, "error": str(e_main), "trend": trend})
                continue

        tbl = binner.binning_table.build()
        iv = float(tbl.loc["Totals", "IV"])
        n_bins = len(tbl) - 1  # без Totals

        # в fit_all
        counts = tbl["Count"].iloc[:-1].astype(float)   # без последней строки Totals
        counts = counts[counts > 0]
        if len(counts) and counts.min() / counts.sum() < 0.01:
            logger.warning("%s: есть бин с долей <1%% (min=%d, total=%d)",
                        col, int(counts.min()), int(counts.sum()))

        binners[col] = binner
        iv_rows.append({
            "feature": col,
            "iv": iv,
            "n_bins": n_bins,
            "dtype": dtype,
            "trend": used_trend,
            "strength": strength(iv),
        })

    iv_table = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)
    failed_df = pd.DataFrame(failed)
    return binners, impute_values, iv_table, failed_df


# ============================================================ #
# 7. Transform
# ============================================================ #
def transform_all(
    binners: dict[str, OptimalBinning],
    impute_values: dict[str, object],
    df: pd.DataFrame,
    dtype_map: dict[str, str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col, binner in binners.items():
        X = df[col].copy()
        if col in impute_values:
            X = X.fillna(impute_values[col])
        if dtype_map[col] == "categorical":
            X = X.astype(str)
        out[f"{col}_woe"] = binner.transform(X.values, metric="woe")
    return out


# ============================================================ #
# 8. Сохранение артефактов
# ============================================================ #
def save_woe_tables(binners: dict[str, OptimalBinning], artifact_dir: Path) -> None:
    woe_dir = artifact_dir / "woe_tables"
    woe_dir.mkdir(parents=True, exist_ok=True)
    for col, binner in binners.items():
        tbl = binner.binning_table.build()
        tbl.to_csv(woe_dir / f"{col}.csv", index=False)
    logger.info("saved %d woe tables → %s", len(binners), woe_dir)


def save_artifacts(
    binners: dict,
    impute_values: dict,
    iv_table: pd.DataFrame,
    failed: pd.DataFrame,
    artifact_dir: Path,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "binners.pkl", "wb") as f:
        pickle.dump({"binners": binners, "impute_values": impute_values}, f)
    iv_table.to_csv(artifact_dir / "iv_table.csv", index=False)
    if not failed.empty:
        failed.to_csv(artifact_dir / "failed_features.csv", index=False)
    save_woe_tables(binners, artifact_dir)
    logger.info("saved binners.pkl, iv_table.csv → %s", artifact_dir)


# ============================================================ #
# 9. Отчёт
# ============================================================ #
def log_iv_report(iv_table: pd.DataFrame) -> None:
    logger.info("IV distribution:")
    counts = iv_table["strength"].value_counts()
    for _, label in STRENGTH_THRESHOLDS:
        logger.info("  %-12s %d", label, counts.get(label, 0))

    logger.info("top-10 by IV:")
    for _, r in iv_table.head(10).iterrows():
        logger.info("  %-30s IV=%.4f  bins=%d  trend=%s",
                    r["feature"], r["iv"], r["n_bins"], r["trend"])

    suspicious = iv_table.query("iv > 0.5")
    if not suspicious.empty:
        logger.warning("SUSPICIOUS (IV > 0.5): %s",
                       suspicious["feature"].tolist())


# ============================================================ #
# 10. Main
# ============================================================ #
def run(cfg: dict, processed_dir: str | Path, artifact_dir: str | Path) -> None:
    datasets = load_datasets(processed_dir)
    num_cols, cat_cols, recode_cols = get_features(cfg)
    logger.info("features to bin: %d (num=%d, cat=%d, recode=%d)",
                len(num_cols) + len(cat_cols) + len(recode_cols),
                len(num_cols), len(cat_cols), len(recode_cols))

    df_train = datasets["train"].copy()
    target = cfg["data"]["target"]
    y_train = df_train[target].values

    binners, impute_values, iv_table, failed = fit_all(
        df_train, y_train, num_cols, cat_cols, recode_cols, cfg
    )
    logger.info("fitted %d/%d features",
                len(binners), len(num_cols) + len(cat_cols) + len(recode_cols))

    dtype_map = {c: "numerical" for c in num_cols}
    dtype_map.update({c: "categorical" for c in cat_cols + recode_cols})

    for name, df in datasets.items():
        woe = transform_all(binners, impute_values, df, dtype_map)
        if target in df.columns:
            woe[target] = df[target].values
        out_path = resolve_path(processed_dir) / f"{name}_woe.parquet"
        woe.to_parquet(out_path)
        logger.info("saved %s  %d x %d", out_path.name, *woe.shape)

    save_artifacts(binners, impute_values, iv_table, failed, resolve_path(artifact_dir))
    log_iv_report(iv_table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    run(cfg, args.processed_dir, args.artifact_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
