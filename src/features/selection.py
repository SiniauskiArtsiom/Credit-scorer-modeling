"""Этап 5. Отбор признаков после WoE-биннинга.

Три последовательных фильтра:
  1. IV-фильтр        — отбрасываем слабые фичи (IV < 0.02)
  2. Корреляционный   — из пары |r| > 0.7 оставляем сильнейшую по IV
  3. VIF-фильтр       — итеративно убираем мультиколлинеарность (VIF > 5)

Плюс перепроверка final_review_candidates (фичи на грани IV ~0.02).

Все операции — только на train. valid/test не трогаем.

Сохраняет:
  artifacts/selected_features.json   — финальный список (для этапов 6+)
  artifacts/selected_features.csv    — то же + IV и n_bins

Usage:
    python -m src.features.selection --config configs/config.yaml
"""
from __future__ import annotations

import warnings

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Фичи, которые не удаляются никакими фильтрами.
PROTECTED_FEATURES = {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}


# ============================================================ #
# 1. Утилиты
# ============================================================ #
def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path) -> dict:
    with open(resolve_path(path)) as f:
        return yaml.safe_load(f)


# ============================================================ #
# 2. Загрузка
# ============================================================ #
def load_inputs(processed_dir: str | Path, artifact_dir: str | Path):
    processed_dir = resolve_path(processed_dir)
    artifact_dir = resolve_path(artifact_dir)

    train = pd.read_parquet(processed_dir / "train_woe.parquet")
    iv_table = pd.read_csv(artifact_dir / "iv_table.csv")
    logger.info("loaded train_woe %d x %d", *train.shape)
    logger.info("loaded iv_table %d features", len(iv_table))
    return train, iv_table


# ============================================================ #
# 3. Фильтр 1 — IV
# ============================================================ #
def filter_by_iv(
    iv_table: pd.DataFrame, threshold: float = 0.02
) -> tuple[list[str], list[str]]:
    kept = iv_table.query("iv >= @threshold")["feature"].tolist()
    dropped = iv_table.query("iv < @threshold")["feature"].tolist()
    logger.info("IV filter (>= %.3f): kept=%d, dropped=%d",
                threshold, len(kept), len(dropped))
    if dropped:
        logger.info("  dropped: %s", dropped)
    return kept, dropped


# ============================================================ #
# 4. Фильтр 2 — корреляция
# ============================================================ #
def filter_correlated(
    train: pd.DataFrame,
    features: list[str],
    iv_table: pd.DataFrame,
    threshold: float = 0.7,
) -> tuple[list[str], list[str]]:
    """Сортирует фичи по IV убыв., оставляет, если не коррелирует с уже выбранными.

    Защищённые фичи не удаляются (PROTECTED_FEATURES).
    """
    iv_rank = (
        iv_table.set_index("feature")["iv"]
        .reindex(features)
        .sort_values(ascending=False)
    )
    keep: list[str] = []
    dropped: list[str] = []
    dropped_reason: dict[str, tuple[str, float]] = {}   # f -> (parent, |r|)

    for f in iv_rank.index:
        if f in PROTECTED_FEATURES:
            keep.append(f)
            continue

        col_f = f"{f}_woe"
        conflicting = None
        max_r = 0.0
        for k in keep:
            col_k = f"{k}_woe"
            r = abs(train[col_f].corr(train[col_k]))
            if r >= threshold and r > max_r:
                max_r = r
                conflicting = k

        if conflicting is None:
            keep.append(f)
        else:
            dropped.append(f)
            dropped_reason[f] = (conflicting, max_r)

    logger.info("Correlation filter (|r| >= %.2f): kept=%d, dropped=%d",
                threshold, len(keep), len(dropped))
    for f in dropped:
        parent, r = dropped_reason[f]
        logger.info("  dropped %-30s (r=%.3f with %s)", f, r, parent)
    return keep, dropped


# ============================================================ #
# 5. Фильтр 3 — VIF
# ============================================================ #
def _compute_vifs(train: pd.DataFrame, features: list[str]) -> pd.Series:
    cols = [f"{f}_woe" for f in features]
    X = train[cols].astype(float).copy()
    X = sm.add_constant(X, has_constant="add")

    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # statsmodels ругается на вырожденные матрицы
        for i, f in enumerate(features):
            try:
                vif = variance_inflation_factor(X.values, i + 1)
                # inf → NaN, чтобы VIF-фильтр корректно пропускал
                result[f] = float(vif) if np.isfinite(vif) else np.nan
            except Exception:
                result[f] = np.nan
    return pd.Series(result)


def filter_vif(
    train: pd.DataFrame,
    features: list[str],
    threshold: float = 5.0,
    max_iter: int = 50,
) -> tuple[list[str], list[str]]:
    """Итеративно убираем фичу с максимальным VIF, пока все <= threshold.

    Защищённые фичи пропускаются — их VIF не учитывается при выборе жертвы.
    """
    current = list(features)
    dropped: list[str] = []

    for it in range(max_iter):
        vifs = _compute_vifs(train, current)
        eligible = vifs.drop(labels=[f for f in PROTECTED_FEATURES if f in vifs.index])
        if eligible.empty:
            break
        top_f, top_vif = eligible.idxmax(), eligible.max()

        if pd.isna(top_vif) or top_vif <= threshold:
            break

        current.remove(top_f)
        dropped.append(top_f)
        logger.info("  iter %d: dropped %-30s (VIF=%.2f)", it + 1, top_f, top_vif)
    else:
        logger.warning("VIF filter не сошёлся за %d итераций", max_iter)

    logger.info("VIF filter (<= %.2f): kept=%d, dropped=%d",
                threshold, len(current), len(dropped))
    return current, dropped


# ============================================================ #
# 6. Пересмотр final_review_candidates
# ============================================================ #
def review_candidates(
    cfg: dict,
    iv_table: pd.DataFrame,
    features: list[str],
) -> tuple[list[str], list[str]]:
    candidates = cfg["columns"].get("final_review_candidates", [])
    dropped: list[str] = []
    for c in candidates:
        if c in PROTECTED_FEATURES:
            continue
        if c not in features:
            continue
        iv = float(iv_table.set_index("feature").loc[c, "iv"])
        if iv < 0.02:
            features.remove(c)
            dropped.append(c)
    if dropped:
        logger.info("final_review: dropped %s", dropped)
    else:
        logger.info("final_review: ничего не дропнуто")
    return features, dropped


# ============================================================ #
# 7. Сохранение
# ============================================================ #
def save_outputs(
    features: list[str],
    iv_table: pd.DataFrame,
    artifact_dir: Path,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    iv_sub = (
        iv_table[iv_table["feature"].isin(features)]
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )
    iv_sub.to_csv(artifact_dir / "selected_features.csv", index=False)

    payload = {
        "features": features,
        "n_features": len(features),
        "iv_sum": float(iv_sub["iv"].sum()),
        "iv_top10": iv_sub.head(10)[["feature", "iv"]].to_dict(orient="records"),
    }
    with open(artifact_dir / "selected_features.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("saved selected_features.json, selected_features.csv → %s",
                artifact_dir)


# ============================================================ #
# 8. Отчёт
# ============================================================ #
def log_report(
    features: list[str],
    iv_table: pd.DataFrame,
    dropped_iv: list[str],
    dropped_corr: list[str],
    dropped_vif: list[str],
    dropped_review: list[str],
) -> None:
    iv_sub = iv_table[iv_table["feature"].isin(features)].sort_values(
        "iv", ascending=False
    )
    logger.info("=" * 60)
    logger.info("Финальный список: %d фич", len(features))
    logger.info("  - IV filter:      -%d", len(dropped_iv))
    logger.info("  - Correlation:    -%d", len(dropped_corr))
    logger.info("  - VIF:            -%d", len(dropped_vif))
    logger.info("  - final_review:   -%d", len(dropped_review))
    logger.info("  - IV sum:          %.4f", iv_sub["iv"].sum())
    logger.info("top-10 фич:")
    for _, r in iv_sub.head(10).iterrows():
        logger.info("  %-32s IV=%.4f  bins=%d  strength=%s",
                    r["feature"], r["iv"], r["n_bins"], r["strength"])
    logger.info("=" * 60)


# ============================================================ #
# 9. Main
# ============================================================ #
def run(cfg: dict, processed_dir: str | Path, artifact_dir: str | Path) -> None:
    artifact_dir = resolve_path(artifact_dir)
    train, iv_table = load_inputs(processed_dir, artifact_dir)

    # 1. IV
    features, dropped_iv = filter_by_iv(
        iv_table, threshold=cfg.get("selection", {}).get("iv_threshold", 0.02)
    )

    # 2. Correlation
    features, dropped_corr = filter_correlated(
        train, features, iv_table,
        threshold=cfg.get("selection", {}).get("corr_threshold", 0.7),
    )

    # 3. VIF
    features, dropped_vif = filter_vif(
        train, features,
        threshold=cfg.get("selection", {}).get("vif_threshold", 5.0),
    )

    # 4. final_review
    features, dropped_review = review_candidates(cfg, iv_table, features)

    # 5. Сохранение + отчёт
    save_outputs(features, iv_table, artifact_dir)
    log_report(features, iv_table, dropped_iv, dropped_corr, dropped_vif, dropped_review)


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
