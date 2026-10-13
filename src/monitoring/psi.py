"""Этап 10. PSI-мониторинг: стабильность фич и скора.

Считает PSI по WoE-значениям фич и по скору champion,
сравнивая train (reference) vs test_unlabeled (actual).

Артефакты:
    artifacts/psi_report.csv          — PSI по каждой фиче + вердикт
    artifacts/psi_distributions.png   — распределение скора train vs unlabeled
    artifacts/psi_summary.json        — сводка: max PSI, watchlist, verdict

Usage:
    python -m src.monitoring.psi --config configs/config.yaml
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
import statsmodels.api as sm
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PSI_THRESHOLDS = {
    "no_drift":   0.10,
    "warn":       0.25,
}


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


# ============================================================ #
# 2. PSI
# ============================================================ #
def psi_numeric(ref: np.ndarray, act: np.ndarray, bins: int = 10) -> float:
    """PSI для числовых значений. Бины по квантилям ref."""
    ref = ref[~np.isnan(ref)]
    act = act[~np.isnan(act)]
    if len(ref) < bins * 2 or len(act) == 0:
        return np.nan

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0

    r, _ = np.histogram(ref, bins=edges)
    a, _ = np.histogram(act, bins=edges)

    # include under/overflow into edge bins
    r = r.astype(float); a = a.astype(float)
    r_pct = np.clip(r / r.sum(), 1e-6, None)
    a_pct = np.clip(a / a.sum(), 1e-6, None)

    return float(((a_pct - r_pct) * np.log(a_pct / r_pct)).sum())


def psi_categorical(ref: pd.Series, act: pd.Series) -> float:
    """PSI для категориальных. Все категории, включая NaN как отдельный класс."""
    r = ref.value_counts(normalize=True, dropna=False)
    a = act.value_counts(normalize=True, dropna=False)
    r, a = r.align(a, fill_value=1e-6)
    r = r.clip(lower=1e-6)
    a = a.clip(lower=1e-6)
    return float(((a - r) * np.log(a / r)).sum())


def psi_feature(ref: pd.Series, act: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(ref):
        return psi_numeric(ref.values, act.values)
    return psi_categorical(ref, act)


def verdict(psi: float) -> str:
    if pd.isna(psi):
        return "insufficient_data"
    if psi < PSI_THRESHOLDS["no_drift"]:
        return "no_drift"
    if psi < PSI_THRESHOLDS["warn"]:
        return "warn"
    return "alert"


# ============================================================ #
# 3. PSI по фичам
# ============================================================ #
def compute_psi_features(
    train_woe: pd.DataFrame,
    unlabeled_woe: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []
    for f in features:
        col = f"{f}_woe"
        if col not in train_woe.columns or col not in unlabeled_woe.columns:
            logger.warning("колонка %s отсутствует, пропуск", col)
            continue
        psi = psi_feature(train_woe[col], unlabeled_woe[col])
        rows.append({
            "feature": f,
            "psi": round(psi, 5) if not pd.isna(psi) else np.nan,
            "verdict": verdict(psi),
        })
    df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return df


# ============================================================ #
# 4. PSI по скору
# ============================================================ #
def champion_score(artifact_dir: Path, df_woe: pd.DataFrame, features: list[str]) -> np.ndarray:
    with open(artifact_dir / "champion_logit.pkl", "rb") as f:
        obj = pickle.load(f)
    model = obj["model"]

    cols = [f"{f}_woe" for f in features]
    X = df_woe[cols].astype(float)
    logit = model.predict(sm.add_constant(X, has_constant="add"), which="linear").values

    # те же параметры скоркарты, что в champion.py
    factor = 40 / np.log(2)
    offset = 600 - factor * np.log(50)
    score = offset - factor * logit
    return score


# ============================================================ #
# 5. Графики
# ============================================================ #
def plot_distributions(score_train: np.ndarray, score_unlabeled: np.ndarray,
                       out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # гистограмма распределения баллов
    ax = axes[0]
    bins = np.linspace(
        min(score_train.min(), score_unlabeled.min()),
        max(score_train.max(), score_unlabeled.max()),
        40,
    )
    ax.hist(score_train, bins=bins, density=True, alpha=0.5,
            label=f"train (n={len(score_train):,})", color="C0")
    ax.hist(score_unlabeled, bins=bins, density=True, alpha=0.5,
            label=f"test_unlabeled (n={len(score_unlabeled):,})", color="C1")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution: train vs test_unlabeled")
    ax.legend()
    ax.grid(alpha=0.3)

    # CDF
    ax = axes[1]
    for s, label, color in [(score_train, "train", "C0"),
                            (score_unlabeled, "test_unlabeled", "C1")]:
        xs = np.sort(s)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, label=label, color=color)
    ax.set_xlabel("Score")
    ax.set_ylabel("CDF")
    ax.set_title("Score CDF comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("saved %s", out_path)


# ============================================================ #
# 6. Main
# ============================================================ #
def run(cfg: dict, processed_dir: str | Path, artifact_dir: str | Path) -> None:
    processed_dir = resolve_path(processed_dir)
    artifact_dir = resolve_path(artifact_dir)
    features = load_selected_features(artifact_dir)

    logger.info("loading woe datasets...")
    train_woe = pd.read_parquet(processed_dir / "train_woe.parquet")
    unlabeled_woe = pd.read_parquet(processed_dir / "test_unlabeled_woe.parquet")

    logger.info("PSI по фичам (n_features=%d)...", len(features))
    psi_df = compute_psi_features(train_woe, unlabeled_woe, features)

    logger.info("=" * 70)
    logger.info("Feature PSI (train vs test_unlabeled)")
    logger.info("=" * 70)
    logger.info("\n%s", psi_df.to_string(index=False))

    watchlist = psi_df.query("verdict == 'warn'")["feature"].tolist()
    alerts = psi_df.query("verdict == 'alert'")["feature"].tolist()

    # PSI по скору
    logger.info("=" * 70)
    logger.info("Score PSI")
    logger.info("=" * 70)
    score_train = champion_score(artifact_dir, train_woe, features)
    score_unlabeled = champion_score(artifact_dir, unlabeled_woe, features)
    psi_score = psi_numeric(score_train, score_unlabeled, bins=10)
    logger.info("score PSI = %.5f  verdict = %s", psi_score, verdict(psi_score))
    logger.info("score train : mean=%.0f  p25=%.0f  median=%.0f  p75=%.0f",
                score_train.mean(), np.percentile(score_train, 25),
                np.median(score_train), np.percentile(score_train, 75))
    logger.info("score unlab : mean=%.0f  p25=%.0f  median=%.0f  p75=%.0f",
                score_unlabeled.mean(), np.percentile(score_unlabeled, 25),
                np.median(score_unlabeled), np.percentile(score_unlabeled, 75))

    # график
    plot_distributions(score_train, score_unlabeled,
                       artifact_dir / "psi_distributions.png")

    # сохранение
    psi_df.to_csv(artifact_dir / "psi_report.csv", index=False)

    summary = {
        "n_features": len(features),
        "feature_psi_max": float(psi_df["psi"].max()),
        "feature_psi_max_name": psi_df.iloc[0]["feature"],
        "feature_psi_median": float(psi_df["psi"].median()),
        "watchlist": watchlist,
        "alerts": alerts,
        "score_psi": float(psi_score),
        "score_psi_verdict": verdict(psi_score),
        "thresholds": PSI_THRESHOLDS,
    }
    with open(artifact_dir / "psi_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("max feature PSI = %.5f (%s)", summary["feature_psi_max"],
                summary["feature_psi_max_name"])
    logger.info("median feature PSI = %.5f", summary["feature_psi_median"])
    logger.info("score PSI = %.5f", psi_score)
    logger.info("watchlist (0.10-0.25): %s", watchlist)
    logger.info("alerts (>0.25): %s", alerts if alerts else "нет")
    logger.info("saved psi_report.csv, psi_summary.json, psi_distributions.png")


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
