"""Этап 7. Challenger: XGBoost на сырых фичах + SHAP.

Обучает XGB на СЫРЫХ фичах (не WoE!) с early stopping на valid.
Считает SHAP, сравнивает метрики с champion через bootstrap-CI.

Артефакты:
    artifacts/challenger_xgb.ubj       — обученный бустер
    artifacts/challenger_metrics.json  — метрики + CI
    artifacts/shap_importance.csv      — средняя |SHAP| по фичам
    artifacts/shap_summary.png         — summary-plot
    artifacts/shap_waterfall_example.png — waterfall для одного заёмщика

Usage:
    python -m src.models.challenger --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import xgboost as xgb
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================ #
# 1. Утилиты
# ============================================================ #
def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path) -> dict:
    with open(resolve_path(path)) as f:
        return yaml.safe_load(f)


def load_selected_features(artifact_dir: Path) -> list[str]:
    with open(artifact_dir / "selected_features.json") as f:
        return json.load(f)["features"]


def gini(y: np.ndarray, p: np.ndarray) -> float:
    return 2 * roc_auc_score(y, p) - 1


def ks(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y, p)),
        "gini": gini(y, p),
        "ks": ks(y, p),
    }


def bootstrap_metric(y, p, metric_fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric_fn(y[idx], p[idx]))
    return float(np.mean(vals)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


# ============================================================ #
# 2. Загрузка сырых данных
# ============================================================ #
def load_raw_data(processed_dir: Path, features: list[str], target: str):
    train = pd.read_parquet(processed_dir / "train.parquet")
    valid = pd.read_parquet(processed_dir / "valid.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")

    sensor_flag = "DAYS_EMPLOYED_IS_SENSOR"

    def pick(df, cols):
        use = list(cols)
        if sensor_flag in df.columns and sensor_flag not in use:
            use.append(sensor_flag)
        missing = [c for c in use if c not in df.columns]
        if missing:
            raise ValueError(f"Колонки отсутствуют в parquet: {missing}")
        X = df[use].copy()
        # XGB 3.x: str → category (иначе enable_categorical=True не работает)
        for c in X.columns:
            if str(X[c].dtype) in ("object", "str", "string"):
                X[c] = X[c].astype("category")
        return X, use

    X_train, use_cols = pick(train, features)
    X_valid, _ = pick(valid, features)
    X_test, _ = pick(test, features)

    y_train = train[target].astype(int).values
    y_valid = valid[target].astype(int).values
    y_test = test[target].astype(int).values

    logger.info("X_train %s, X_valid %s, X_test %s", X_train.shape, X_valid.shape, X_test.shape)
    logger.info("features: %d (+ sensor flag)", len(use_cols))
    return X_train, y_train, X_valid, y_valid, X_test, y_test, use_cols


# ============================================================ #
# 3. Fit XGB
# ============================================================ #
def fit_xgb(X_train, y_train, X_valid, y_valid, cfg: dict):
    params = cfg.get("challenger", {}).get("params", {})
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_lambda": 1.0,
        "seed": 42,
        **params,
    }
    n_rounds = cfg.get("challenger", {}).get("num_boost_round", 1000)
    early_stop = cfg.get("challenger", {}).get("early_stopping_rounds", 50)

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

    logger.info("XGB params: %s", params)
    booster = xgb.train(
        params, dtrain, num_boost_round=n_rounds,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stop,
        verbose_eval=100,
    )
    logger.info("best_iteration=%d, best_score=%.4f",
                booster.best_iteration, booster.best_score)
    return booster


# ============================================================ #
# 4. Champion predict (для сравнения)
# ============================================================ #
def champion_predict(artifact_dir: Path, processed_dir: Path,
                     features: list[str], target: str):
    """Возвращает p_champion на valid/test из сохранённого Logit."""
    with open(artifact_dir / "champion_logit.pkl", "rb") as f:
        obj = pickle.load(f)
    model = obj["model"]

    valid = pd.read_parquet(processed_dir / "valid_woe.parquet")
    test = pd.read_parquet(processed_dir / "test_woe.parquet")

    cols = [f"{f}_woe" for f in features]
    Xv = valid[cols].astype(float)
    Xt = test[cols].astype(float)

    p_valid = model.predict(sm.add_constant(Xv, has_constant="add")).values
    p_test = model.predict(sm.add_constant(Xt, has_constant="add")).values
    return p_valid, p_test


# ============================================================ #
# 5. SHAP
# ============================================================ #
def compute_shap(booster, X_valid, artifact_dir: Path, top_n: int = 15):
    logger.info("computing SHAP on %d rows...", len(X_valid))
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_valid)

    # средняя |SHAP| по фичам
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": X_valid.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance["share_pct"] = (
        100 * importance["mean_abs_shap"] / importance["mean_abs_shap"].sum()
    ).round(3)
    importance.to_csv(artifact_dir / "shap_importance.csv", index=False)
    logger.info("top-%d SHAP:", top_n)
    for _, r in importance.head(top_n).iterrows():
        logger.info("  %-32s %.4f  (%.2f%%)",
                    r["feature"], r["mean_abs_shap"], r["share_pct"])

    # summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_valid, show=False, max_display=top_n)
    plt.tight_layout()
    plt.savefig(artifact_dir / "shap_summary.png", dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("saved shap_summary.png")

    # waterfall — один заёмщик (первый)
    try:
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, shap_values[0], X_valid.iloc[0],
            show=False, max_display=10,
        )
        plt.tight_layout()
        plt.savefig(artifact_dir / "shap_waterfall_example.png", dpi=100, bbox_inches="tight")
        plt.close()
        logger.info("saved shap_waterfall_example.png")
    except Exception as e:
        logger.warning("waterfall plot failed: %s", e)

    return importance, shap_values


# ============================================================ #
# 6. Sanity check
# ============================================================ #
def sanity_check_shap(importance: pd.DataFrame, iv_table_path: Path) -> None:
    """Пересечение топ-10 SHAP и топ-10 IV."""
    iv_table = pd.read_csv(iv_table_path)
    top_shap = set(importance.head(10)["feature"])
    top_iv = set(iv_table.head(10)["feature"])

    overlap = top_shap & top_iv
    logger.info("SHAP∩IV top-10: %d фич — %s", len(overlap), sorted(overlap))
    if len(overlap) < 5:
        logger.warning("Мало пересечений SHAP и IV top-10 (<5). Проверить модели.")

    max_share = importance["share_pct"].max()
    if max_share > 50:
        top = importance.iloc[0]
        logger.warning("SUSPICIOUS: '%s' даёт %.1f%% SHAP — возможен leakage",
                       top["feature"], max_share)
    else:
        logger.info("Доминирующих фич нет (max share %.1f%%)", max_share)


# ============================================================ #
# 7. Сохранение и отчёт
# ============================================================ #
def save_artifacts(booster, use_cols, metrics, artifact_dir: Path):
    booster.save_model(str(artifact_dir / "challenger_xgb.ubj"))
    with open(artifact_dir / "challenger_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "features": use_cols}, f, indent=2)
    logger.info("saved challenger_xgb.ubj, challenger_metrics.json")


def log_report(metrics: dict) -> None:
    logger.info("=" * 60)
    logger.info("Champion vs Challenger")
    for split in ["train", "valid", "test"]:
        c = metrics["champion"].get(split)
        x = metrics["challenger"].get(split)
        if c is None or x is None:
            continue
        logger.info("%s:", split)
        logger.info("  champion  : AUC=%.4f  Gini=%.4f  KS=%.4f",
                    c["auc"], c["gini"], c["ks"])
        logger.info("  challenger: AUC=%.4f  Gini=%.4f  KS=%.4f",
                    x["auc"], x["gini"], x["ks"])
        logger.info("  ΔGini     : %+.4f", x["gini"] - c["gini"])
    logger.info("Bootstrap-CI на valid (Gini):")
    for name in ["champion", "challenger"]:
        ci = metrics[f"{name}_ci_valid"]
        logger.info("  %-11s mean=%.4f  CI=[%.4f, %.4f]",
                    name, ci["mean"], ci["lo"], ci["hi"])
    logger.info("=" * 60)


# ============================================================ #
# 8. Main
# ============================================================ #
def run(cfg: dict, processed_dir: str | Path, artifact_dir: str | Path) -> None:
    processed_dir = resolve_path(processed_dir)
    artifact_dir = resolve_path(artifact_dir)

    features = load_selected_features(artifact_dir)
    target = cfg["data"]["target"]

    X_train, y_train, X_valid, y_valid, X_test, y_test, use_cols = load_raw_data(
        processed_dir, features, target
    )

    booster = fit_xgb(X_train, y_train, X_valid, y_valid, cfg)

    # predictions
    dvalid = xgb.DMatrix(X_valid, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    p_train = booster.predict(xgb.DMatrix(X_train, enable_categorical=True))
    p_valid = booster.predict(dvalid)
    p_test = booster.predict(dtest)

    # champion predictions
    p_champ_valid, p_champ_test = champion_predict(artifact_dir, processed_dir, features, target)

    # metrics
    metrics = {
        "champion": {
            "train": evaluate(y_train, champion_predict(artifact_dir, processed_dir, features, target)[0]) if False else None,
            "valid": evaluate(y_valid, p_champ_valid),
            "test":  evaluate(y_test, p_champ_test),
        },
        "challenger": {
            "train": evaluate(y_train, p_train),
            "valid": evaluate(y_valid, p_valid),
            "test":  evaluate(y_test, p_test),
        },
    }
    # champion на train — нет прямого способа без загрузки train_woe, но добавим:
    with open(artifact_dir / "champion_logit.pkl", "rb") as f:
        champ_obj = pickle.load(f)
    train_woe = pd.read_parquet(processed_dir / "train_woe.parquet")
    cols = [f"{f}_woe" for f in features]
    p_champ_train = champ_obj["model"].predict(
        sm.add_constant(train_woe[cols].astype(float), has_constant="add")
    ).values
    metrics["champion"]["train"] = evaluate(y_train, p_champ_train)

    # bootstrap CI на valid
    metrics["champion_ci_valid"] = {
        "mean": bootstrap_metric(y_valid, p_champ_valid, gini)[0],
        "lo":   bootstrap_metric(y_valid, p_champ_valid, gini)[1],
        "hi":   bootstrap_metric(y_valid, p_champ_valid, gini)[2],
    }
    metrics["challenger_ci_valid"] = {
        "mean": bootstrap_metric(y_valid, p_valid, gini)[0],
        "lo":   bootstrap_metric(y_valid, p_valid, gini)[1],
        "hi":   bootstrap_metric(y_valid, p_valid, gini)[2],
    }

    # SHAP
    importance, _ = compute_shap(booster, X_valid, artifact_dir)
    sanity_check_shap(importance, artifact_dir / "iv_table.csv")

    # save
    save_artifacts(booster, use_cols, metrics, artifact_dir)
    log_report(metrics)


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
