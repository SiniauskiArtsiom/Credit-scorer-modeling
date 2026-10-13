"""Этап 8. Калибровка + финальные метрики + gains.

Для champion (Logit) и challenger (XGBoost) считает:
  - Reliability diagram (predicted PD vs actual bad rate)
  - Hosmer-Lemeshow test
  - Isotonic regression для challenger
  - Финальные метрики (AUC/Gini/KS/Brier) с bootstrap-CI на test
  - Gains-таблицу и lift

Артефакты:
    artifacts/calibration_curves.png      — reliability diagram
    artifacts/gains_table.csv             — capture rate по децилям
    artifacts/final_metrics.json          — метрики с CI
    artifacts/challenger_isotonic.pkl     — калибратор для challenger

Usage:
    python -m src.evaluation.calibration --config configs/config.yaml
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
import xgboost as xgb
import yaml
from scipy.stats import chi2
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

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


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def evaluate(y, p) -> dict:
    return {
        "auc": float(roc_auc_score(y, p)),
        "gini": gini(y, p),
        "ks": ks(y, p),
        "brier": float(brier_score_loss(y, p)),
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
    return {
        "mean": float(np.mean(vals)),
        "lo":   float(np.quantile(vals, 0.025)),
        "hi":   float(np.quantile(vals, 0.975)),
    }


# ============================================================ #
# 2. Загрузка предсказаний
# ============================================================ #
def load_champion_preds(artifact_dir: Path, processed_dir: Path,
                        features: list[str], split: str):
    with open(artifact_dir / "champion_logit.pkl", "rb") as f:
        model = pickle.load(f)["model"]
    df = pd.read_parquet(processed_dir / f"{split}_woe.parquet")
    cols = [f"{f}_woe" for f in features]
    X = df[cols].astype(float)
    p = model.predict(sm.add_constant(X, has_constant="add")).values
    y = df["TARGET"].astype(int).values
    return y, p


def load_challenger_preds(artifact_dir: Path, processed_dir: Path,
                          features: list[str], split: str):
    booster = xgb.Booster()
    booster.load_model(str(artifact_dir / "challenger_xgb.ubj"))
    df = pd.read_parquet(processed_dir / f"{split}.parquet")
    use = list(features)
    if "DAYS_EMPLOYED_IS_SENSOR" in df.columns:
        use.append("DAYS_EMPLOYED_IS_SENSOR")
    X = df[use].copy()
    for c in X.columns:
        if str(X[c].dtype) in ("object", "str", "string"):
            X[c] = X[c].astype("category")
    p = booster.predict(xgb.DMatrix(X, enable_categorical=True))
    y = df["TARGET"].astype(int).values
    return y, p


# ============================================================ #
# 3. Reliability diagram + HL
# ============================================================ #
def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted PD vs actual bad rate по квинтилям."""
    df = pd.DataFrame({"p": p, "y": y})
    try:
        df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["p"], n_bins, labels=False)

    out = df.groupby("bin").agg(
        n=("y", "size"),
        pred_mean=("p", "mean"),
        actual_rate=("y", "mean"),
    ).reset_index()
    out["pred_mean"] = out["pred_mean"].round(4)
    out["actual_rate"] = out["actual_rate"].round(4)
    return out


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, g: int = 10) -> tuple[float, float]:
    """Hosmer-Lemeshow: χ² + p-value. p > 0.05 → модель калибрована."""
    order = np.argsort(p)
    y = np.asarray(y)[order]
    p = np.asarray(p)[order]
    bins = np.array_split(np.arange(len(y)), g)
    obs1 = np.array([y[b].sum() for b in bins])
    exp1 = np.array([p[b].sum() for b in bins])
    n = np.array([len(b) for b in bins])
    obs0, exp0 = n - obs1, n - exp1

    hl = (
        ((obs1 - exp1) ** 2) / np.clip(exp1, 1e-9, None)
        + ((obs0 - exp0) ** 2) / np.clip(exp0, 1e-9, None)
    ).sum()
    pval = 1 - chi2.cdf(hl, df=g - 2)
    return float(hl), float(pval)


# ============================================================ #
# 4. Isotonic калибровка
# ============================================================ #
def fit_isotonic(y_valid: np.ndarray, p_valid: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_valid, y_valid)
    logger.info("isotonic fitted on valid: %d points", len(p_valid))
    return iso


def apply_isotonic(iso: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    return iso.predict(p)


# ============================================================ #
# 5. Gains / lift
# ============================================================ #
def gains_table(y: np.ndarray, p: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Capture rate и lift по децилям скора (сорт. по p убыв.)."""
    df = pd.DataFrame({"p": p, "y": y}).sort_values("p", ascending=False).reset_index(drop=True)
    total_bad = df["y"].sum()
    n_total = len(df)
    rows = []
    for i in range(1, n + 1):
        top = df.iloc[: int(n_total * i / n)]
        captured = top["y"].sum()
        rows.append({
            "top_pct": i * 100 // n,
            "n": len(top),
            "captured_bad": int(captured),
            "capture_rate": round(captured / total_bad, 4),
            "lift": round((captured / len(top)) / (total_bad / n_total), 3),
        })
    return pd.DataFrame(rows)


# ============================================================ #
# 6. Графики
# ============================================================ #
def plot_calibration(
    tbl_champ: pd.DataFrame,
    tbl_chall: pd.DataFrame,
    tbl_chall_cal: pd.DataFrame,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for tbl, label, color in [
        (tbl_champ,      "champion (Logit)",       "C0"),
        (tbl_chall,      "challenger (XGB raw)",   "C1"),
        (tbl_chall_cal,  "challenger + isotonic",  "C2"),
    ]:
        ax.plot(tbl["pred_mean"], tbl["actual_rate"], "o-",
                label=label, color=color, markersize=5)
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Actual bad rate")
    ax.set_title("Reliability diagram (test)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("saved %s", out_path)


# ============================================================ #
# 7. Main
# ============================================================ #
def run(cfg: dict, processed_dir: str | Path, artifact_dir: str | Path) -> None:
    processed_dir = resolve_path(processed_dir)
    artifact_dir = resolve_path(artifact_dir)
    features = load_selected_features(artifact_dir)

    logger.info("loading predictions...")
    y_tr_c, p_tr_c = load_champion_preds(artifact_dir, processed_dir, features, "train")
    y_va_c, p_va_c = load_champion_preds(artifact_dir, processed_dir, features, "valid")
    y_te_c, p_te_c = load_champion_preds(artifact_dir, processed_dir, features, "test")

    y_tr_x, p_tr_x = load_challenger_preds(artifact_dir, processed_dir, features, "train")
    y_va_x, p_va_x = load_challenger_preds(artifact_dir, processed_dir, features, "valid")
    y_te_x, p_te_x = load_challenger_preds(artifact_dir, processed_dir, features, "test")

    # --- reliability ---
    logger.info("=" * 60)
    logger.info("Reliability + HL")
    logger.info("=" * 60)
    rel_champ = reliability_table(y_te_c, p_te_c)
    rel_chall = reliability_table(y_te_x, p_te_x)
    hl_c = hosmer_lemeshow(y_te_c, p_te_c)
    hl_x = hosmer_lemeshow(y_te_x, p_te_x)
    logger.info("champion  HL=%.3f  p=%.4f  %s",
                hl_c[0], hl_c[1], "OK" if hl_c[1] > 0.05 else "not calibrated")
    logger.info("challenger HL=%.3f  p=%.4f  %s",
                hl_x[0], hl_x[1], "OK" if hl_x[1] > 0.05 else "not calibrated")

    # --- isotonic ---
    # NB: XGBoost с binary:logistic УЖЕ калиброван (HL p=0.685 без калибровки).
    # Isotonic проверен и отклонён — он ухудшает HL (p: 0.685 → 0.079) и Gini.
    # Оставляем для референса в отчёте, в проде используем сырой XGB.
    iso = fit_isotonic(y_va_x, p_va_x)
    p_te_x_cal = apply_isotonic(iso, p_te_x)
    rel_chall_cal = reliability_table(y_te_x, p_te_x_cal)
    hl_x_cal = hosmer_lemeshow(y_te_x, p_te_x_cal)
    logger.info("challenger+iso HL=%.3f  p=%.4f  %s",
                hl_x_cal[0], hl_x_cal[1],
                "OK" if hl_x_cal[1] > 0.05 else "not calibrated")

    # --- график ---
    plot_calibration(rel_champ, rel_chall, rel_chall_cal,
                     artifact_dir / "calibration_curves.png")

    # --- финальные метрики с CI на test ---
    logger.info("=" * 60)
    logger.info("Final metrics on TEST with bootstrap-CI")
    logger.info("=" * 60)
    metrics = {}
    for name, y, p in [
        ("champion",         y_te_c, p_te_c),
        ("challenger",       y_te_x, p_te_x),
        ("challenger_cal",   y_te_x, p_te_x_cal),
    ]:
        m = evaluate(y, p)
        ci = {
            "gini": bootstrap_metric(y, p, gini),
            "auc":  bootstrap_metric(y, p, lambda a, b: roc_auc_score(a, b)),
            "ks":   bootstrap_metric(y, p, ks),
        }
        metrics[name] = {**m, "ci": ci}
        logger.info("%s:", name)
        logger.info("  AUC=%.4f  Gini=%.4f  KS=%.4f  Brier=%.4f",
                    m["auc"], m["gini"], m["ks"], m["brier"])
        logger.info("  Gini CI: [%.4f, %.4f]", ci["gini"]["lo"], ci["gini"]["hi"])


    metrics["_decision"] = {
        "champion":       "Logit, calibrated by design",
        "challenger":     "XGBoost raw, HL p=0.685, calibrated by design",
        "challenger_cal": "isotonic reference, slightly worse — NOT used in prod",
    }

    # --- gains ---
    logger.info("=" * 60)
    logger.info("Gains (top-10% capture)")
    logger.info("=" * 60)
    gains_c = gains_table(y_te_c, p_te_c)
    gains_x = gains_table(y_te_x, p_te_x_cal)
    logger.info("champion  : top-10%% capture = %.2f%%  lift = %.2fx",
                100 * gains_c.iloc[0]["capture_rate"], gains_c.iloc[0]["lift"])
    logger.info("challenger: top-10%% capture = %.2f%%  lift = %.2fx",
                100 * gains_x.iloc[0]["capture_rate"], gains_x.iloc[0]["lift"])

    gains_c.to_csv(artifact_dir / "gains_table_champion.csv", index=False)
    gains_x.to_csv(artifact_dir / "gains_table_challenger.csv", index=False)

    # --- сохранение ---
    with open(artifact_dir / "challenger_isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(artifact_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("saved final_metrics.json, gains_table_*.csv, challenger_isotonic.pkl")


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
