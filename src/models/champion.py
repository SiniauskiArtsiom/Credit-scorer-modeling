"""Этап 6. Champion: логистическая регрессия на WoE-фичах + скоркарта.

Обучает Logit (statsmodels) на 21 отобранной WoE-фиче, проверяет знаки
и значимость коэффициентов, строит скоркарту (points-to-double-odds),
считает Gini/KS на train/valid, проверяет монотонность балла.

Артефакты:
    artifacts/champion_logit.pkl   — обученный Logit
    artifacts/scorecard.csv        — таблица points по фичам
    artifacts/coef_table.csv       — коэффициенты, p-value, CI

Usage:
    python -m src.models.champion --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
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


# ============================================================ #
# 2. Загрузка и сборка X, y
# ============================================================ #
def load_data(processed_dir: Path, features: list[str], target: str):
    train = pd.read_parquet(processed_dir / "train_woe.parquet")
    valid = pd.read_parquet(processed_dir / "valid_woe.parquet")

    cols = [f"{f}_woe" for f in features]
    missing_train = [c for c in cols if c not in train.columns]
    if missing_train:
        raise ValueError(f"Колонки отсутствуют в train: {missing_train}")
    missing_valid = [c for c in cols if c not in valid.columns]
    if missing_valid:
        raise ValueError(f"Колонки отсутствуют в valid: {missing_valid}")

    X_train = train[cols].astype(float)
    y_train = train[target].astype(int)
    X_valid = valid[cols].astype(float)
    y_valid = valid[target].astype(int)

    logger.info("X_train %s, X_valid %s", X_train.shape, X_valid.shape)
    return X_train, y_train, X_valid, y_valid


# ============================================================ #
# 3. Fit Logit
# ============================================================ #
def fit_logit(X_train: pd.DataFrame, y_train: pd.Series) -> sm.Logit:
    Xc = sm.add_constant(X_train, has_constant="add")
    model = sm.Logit(y_train, Xc).fit(disp=False, maxiter=200)
    logger.info("Logit fitted: pseudo R²=%.4f, LL=%.1f",
                model.prsquared, model.llf)
    return model


# ============================================================ #
# 4. Таблица коэффициентов
# ============================================================ #
def build_coef_table(model: sm.Logit) -> pd.DataFrame:
    coefs = model.params
    pvals = model.pvalues
    conf = model.conf_int()
    conf.columns = ["ci_low", "ci_high"]

    df = pd.DataFrame({
        "feature": coefs.index,
        "coef": coefs.values,
        "std_err": model.bse.values,
        "z": model.tvalues.values,
        "p_value": pvals.values,
        "ci_low": conf["ci_low"].values,
        "ci_high": conf["ci_high"].values,
    })
    df["significant"] = df["p_value"] < 0.05
    df["positive"] = df["coef"] > 0
    return df


def check_coefficients(coef_table: pd.DataFrame) -> None:
    """Проверяет знаки и значимость. Логирует, не падает."""
    bad_sign = coef_table.query("feature != 'const' and positive")
    if not bad_sign.empty:
        logger.warning("Отрицательные коэффициенты (проверить): %s",
                       bad_sign["feature"].tolist())

    not_sig = coef_table.query("feature != 'const' and not significant")
    if not not_sig.empty:
        logger.warning("Незначимые коэффициенты (p >= 0.05): %s",
                       not_sig["feature"].tolist())

    if bad_sign.empty and not_sig.empty:
        logger.info("Все коэффициенты значимы и положительны")


# ============================================================ #
# 5. Скоркарта
# ============================================================ #
class ScorecardScaler:
    """Points-to-double-odds scaling.

    Score = Offset - Factor * logit
    где logit = beta_0 + sum(beta_i * WoE_i)
    """

    def __init__(self, base_score: float = 600.0,
                 base_odds: float = 50.0,
                 pdo: float = 40.0):
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.factor_ = pdo / np.log(2)
        self.offset_ = base_score - self.factor_ * np.log(base_odds)

    def fit(self, model: sm.Logit) -> "ScorecardScaler":
        params = model.params
        self.intercept_ = float(params["const"])
        self.coefs_ = {k: float(v) for k, v in params.items() if k != "const"}
        return self

    def score(self, X_woe: pd.DataFrame) -> np.ndarray:
        # logit = intercept + sum(beta_i * X_i)
        logit = self.intercept_ + X_woe[list(self.coefs_.keys())].mul(
            pd.Series(self.coefs_), axis=1
        ).sum(axis=1)
        return (self.offset_ - self.factor_ * logit).values

    def points_table(self) -> pd.DataFrame:
        rows = [{"feature": "Intercept", "coef": self.intercept_,
                 "points": -self.factor_ * self.intercept_}]
        for f, b in self.coefs_.items():
            rows.append({"feature": f, "coef": b,
                         "points_per_woe_unit": -self.factor_ * b})
        return pd.DataFrame(rows)

    def describe(self) -> dict:
        return {
            "base_score": self.base_score,
            "base_odds": self.base_odds,
            "pdo": self.pdo,
            "factor": self.factor_,
            "offset": self.offset_,
        }


# ============================================================ #
# 6. Метрики
# ============================================================ #
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


# ============================================================ #
# 7. Монотонность балла по децилям
# ============================================================ #
def score_deciles(y: np.ndarray, score: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Bad rate по децилям балла. Ожидаем монотонное убывание."""
    s = pd.Series(score)
    try:
        bins = pd.qcut(s, n, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(s, n, labels=False)
    df = pd.DataFrame({"score": score, "y": y, "bin": bins})
    out = df.groupby("bin").agg(
        count=("y", "size"),
        bad_rate=("y", "mean"),
        score_min=("score", "min"),
        score_max=("score", "max"),
    ).reset_index()
    out["bad_rate"] = out["bad_rate"].round(4)
    return out


def check_monotonicity(deciles: pd.DataFrame) -> bool:
    diffs = np.diff(deciles["bad_rate"].values)
    is_mono = bool((diffs <= 0).all() or (diffs >= 0).all())
    if is_mono:
        logger.info("Bad rate монотонен по децилям балла")
    else:
        logger.warning("Bad rate НЕ монотонен по децилям балла: %s",
                       deciles["bad_rate"].tolist())
    return is_mono


# ============================================================ #
# 8. Сохранение
# ============================================================ #
def save_artifacts(
    model: sm.Logit,
    scaler: ScorecardScaler,
    coef_table: pd.DataFrame,
    deciles: pd.DataFrame,
    artifact_dir: Path,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "champion_logit.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    coef_table.to_csv(artifact_dir / "coef_table.csv", index=False)
    scaler.points_table().to_csv(artifact_dir / "scorecard.csv", index=False)
    deciles.to_csv(artifact_dir / "score_deciles.csv", index=False)
    logger.info("saved champion_logit.pkl, coef_table.csv, scorecard.csv, "
                "score_deciles.csv → %s", artifact_dir)


# ============================================================ #
# 9. Отчёт
# ============================================================ #
def log_report(metrics_train: dict, metrics_valid: dict,
               scaler: ScorecardScaler, score_train: np.ndarray,
               score_valid: np.ndarray) -> None:
    logger.info("=" * 60)
    logger.info("Champion metrics")
    logger.info("  train: AUC=%.4f  Gini=%.4f  KS=%.4f",
                metrics_train["auc"], metrics_train["gini"], metrics_train["ks"])
    logger.info("  valid: AUC=%.4f  Gini=%.4f  KS=%.4f",
                metrics_valid["auc"], metrics_valid["gini"], metrics_valid["ks"])
    logger.info("Scorecard scaling:")
    for k, v in scaler.describe().items():
        logger.info("  %-12s %.4f", k, v)
    logger.info("Score distribution:")
    logger.info("  train: min=%.0f  p25=%.0f  median=%.0f  p75=%.0f  max=%.0f",
                np.min(score_train), np.percentile(score_train, 25),
                np.median(score_train), np.percentile(score_train, 75),
                np.max(score_train))
    logger.info("  valid: min=%.0f  p25=%.0f  median=%.0f  p75=%.0f  max=%.0f",
                np.min(score_valid), np.percentile(score_valid, 25),
                np.median(score_valid), np.percentile(score_valid, 75),
                np.max(score_valid))
    logger.info("=" * 60)


# ============================================================ #
# 10. Main
# ============================================================ #
def run(cfg: dict, processed_dir: str | Path, artifact_dir: str | Path) -> None:
    processed_dir = resolve_path(processed_dir)
    artifact_dir = resolve_path(artifact_dir)

    features = load_selected_features(artifact_dir)
    logger.info("selected features: %d", len(features))

    target = cfg["data"]["target"]
    X_train, y_train, X_valid, y_valid = load_data(processed_dir, features, target)

    # --- fit ---
    model = fit_logit(X_train, y_train)

    # --- coefficients ---
    coef_table = build_coef_table(model)
    check_coefficients(coef_table)

    # --- scorecard ---
    scaler = ScorecardScaler(base_score=600, base_odds=50, pdo=40).fit(model)

    score_train = scaler.score(X_train)
    score_valid = scaler.score(X_valid)

    # --- metrics ---
    p_train = model.predict(sm.add_constant(X_train, has_constant="add"))
    p_valid = model.predict(sm.add_constant(X_valid, has_constant="add"))
    metrics_train = evaluate(y_train.values, p_train.values)
    metrics_valid = evaluate(y_valid.values, p_valid.values)

    # --- deciles ---
    dec_train = score_deciles(y_train.values, score_train, n=10)
    dec_valid = score_deciles(y_valid.values, score_valid, n=10)
    logger.info("train deciles:\n%s", dec_train.to_string(index=False))
    logger.info("valid deciles:\n%s", dec_valid.to_string(index=False))
    check_monotonicity(dec_train)
    check_monotonicity(dec_valid)

    # --- save ---
    save_artifacts(model, scaler, coef_table, dec_valid, artifact_dir)
    log_report(metrics_train, metrics_valid, scaler, score_train, score_valid)


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
