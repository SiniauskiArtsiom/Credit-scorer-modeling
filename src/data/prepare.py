"""Этап 2+3. Подготовка данных: сенсоры, recode, group_map, drop, split.

Читает configs/config.yaml, превращает сырые CSV в 4 parquet-файла:
    data/processed/train.parquet
    data/processed/valid.parquet
    data/processed/test.parquet
    data/processed/test_unlabeled.parquet

Порядок операций критичен:
    sensors → recode_to_flag → recode_to_categorical → group_map
    → group_rare → drop → split → apply_to_test_unlabeled → save

Модуль идемпотентен: повторный запуск с тем же seed даёт те же данные.

Usage:
    python -m src.data.prepare --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ============================================================ #
# 1. Загрузка и валидация
# ============================================================ #
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path) -> Path:
    """Резолвит путь относительно корня проекта.

    Абсолютные пути возвращаются как есть.
    Относительные — от PROJECT_ROOT, а не от cwd.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_and_validate(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает train и test_unlabeled, проверяет схему."""
    train_path = resolve_path(cfg["data"]["train_path"])
    test_path = resolve_path(cfg["data"]["test_unlabeled_path"])
    id_col = cfg["data"]["id_col"]
    target = cfg["data"]["target"]

    df = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    logger.info("loaded train:  %d x %d", *df.shape)
    logger.info("loaded test:   %d x %d", *df_test.shape)

    assert target in df.columns, f"{target} отсутствует в train"
    assert target not in df_test.columns, f"{target} не должен быть в test_unlabeled"
    assert set(df.columns) - set(df_test.columns) == {target}, \
        "Разница колонок train/test должна быть ровно {TARGET}"
    assert set(df_test.columns).issubset(set(df.columns)), \
        "test_unlabeled содержит колонки, которых нет в train"
    assert df[id_col].is_unique, f"{id_col} не уникален в train"
    assert df_test[id_col].is_unique, f"{id_col} не уникален в test"

    overlap = set(df[id_col]) & set(df_test[id_col])
    assert not overlap, f"Пересечение ID между train/test: {len(overlap)}"

    return df, df_test


# ============================================================ #
# 2. Сенсоры
# ============================================================ #
def apply_sensors(df: pd.DataFrame, sensors: dict) -> pd.DataFrame:
    """Заменяет сенсорные значения на NaN, создаёт флаг *_IS_SENSOR."""
    for col, rule in sensors.items():
        if col not in df.columns:
            logger.warning("sensor: колонка %s не найдена, пропуск", col)
            continue
        value = rule["value"]
        n_before = int((df[col] == value).sum())
        if rule.get("add_flag", False):
            flag_col = f"{col}_IS_SENSOR"
            df[flag_col] = (df[col] == value).astype("int8")
            logger.info("sensor: %s → flag %s (%d записей)", col, flag_col, n_before)
        if rule.get("replace", None) is None:
            df.loc[df[col] == value, col] = np.nan
    return df


# ============================================================ #
# 3. Recode to flag
# ============================================================ #
def recode_to_flags(df: pd.DataFrame, cat_decisions: dict) -> pd.DataFrame:
    """N/Y → 0/1 для фич с action='recode_to_flag'."""
    for col, decision in cat_decisions.items():
        if decision.get("action") != "recode_to_flag":
            continue
        if col not in df.columns:
            continue
        mapping = decision["mapping"]
        n_before = df[col].nunique(dropna=False)
        df[col] = df[col].map(mapping).astype("int8")
        logger.info("recode_to_flag: %s (%d → 2)", col, n_before)
    return df


# ============================================================ #
# 4. Recode to categorical
# ============================================================ #
def recode_to_categorical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Приводит числовые дискретные фичи к категориальному типу."""
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype("category")
        logger.info("recode_to_categorical: %s", col)
    return df


# ============================================================ #
# 5. Group map
# ============================================================ #
def apply_group_map(df: pd.DataFrame, cat_decisions: dict) -> pd.DataFrame:
    """Применяет group_map: 58 → 21 для ORGANIZATION_TYPE и т.д."""
    for col, decision in cat_decisions.items():
        group_map = decision.get("group_map", {})
        if not group_map or col not in df.columns:
            continue
        n_before = df[col].nunique(dropna=False)
        mapped = df[col].map(group_map)
        df[col] = mapped.where(mapped.notna(), df[col])
        n_after = df[col].nunique(dropna=False)
        logger.info("group_map: %s (%d → %d)", col, n_before, n_after)
    return df


# ============================================================ #
# 6. Group rare
# ============================================================ #
def group_rare_categories(df: pd.DataFrame, cat_decisions: dict) -> pd.DataFrame:
    """Редкие категории (< threshold) → 'Other'."""
    for col, decision in cat_decisions.items():
        if not decision.get("group_rare", False) or col not in df.columns:
            continue
        thr = decision.get("rare_threshold", 0.01)
        freq = df[col].value_counts(normalize=True, dropna=False)
        rare = freq[freq < thr].index.tolist()
        # не трогаем NaN
        rare = [r for r in rare if pd.notna(r)]
        if rare:
            df[col] = df[col].replace(rare, "Other")
            logger.info("group_rare: %s (%d редких → Other, thr=%.3f)",
                        col, len(rare), thr)
    return df


# ============================================================ #
# 7. Drop
# ============================================================ #
def drop_columns(df: pd.DataFrame, drop_list: list[str]) -> pd.DataFrame:
    existing = [c for c in drop_list if c in df.columns]
    missing = set(drop_list) - set(existing)
    if missing:
        logger.warning("drop: %d колонок из config не найдены", len(missing))
    df = df.drop(columns=existing)
    logger.info("drop: %d колонок удалено", len(existing))
    return df


# ============================================================ #
# 8. Сплит
# ============================================================ #
def split_indices(df: pd.DataFrame, cfg: dict) -> tuple:
    target = cfg["data"]["target"]
    seed = cfg["split"]["seed"]
    val_ratio, test_ratio = cfg["split"]["ratios"][1], cfg["split"]["ratios"][2]

    tr_idx, tmp_idx = train_test_split(
        df.index, test_size=val_ratio + test_ratio,
        stratify=df[target], random_state=seed,
    )
    rel_test = test_ratio / (val_ratio + test_ratio)
    val_idx, te_idx = train_test_split(
        tmp_idx, test_size=rel_test,
        stratify=df.loc[tmp_idx, target], random_state=seed,
    )
    logger.info("split: train=%d, valid=%d, test=%d",
                len(tr_idx), len(val_idx), len(te_idx))
    return tr_idx, val_idx, te_idx


# ============================================================ #
# 9. Применение к test_unlabeled
# ============================================================ #
def prepare_test_unlabeled(df_test: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Те же преобразования, что и для train."""
    cat_decisions = cfg["columns"]["categorical_decisions"]

    df_test = apply_sensors(df_test, cfg["columns"]["sensors"])
    df_test = recode_to_flags(df_test, cat_decisions)
    df_test = recode_to_categorical(df_test, cfg["columns"]["recode_to_categorical"])
    df_test = apply_group_map(df_test, cat_decisions)
    df_test = group_rare_categories(df_test, cat_decisions)
    df_test = drop_columns(df_test, cfg["columns"]["drop"])

    # Проверка: нет ли колонок, которых не было в train
    # (простая проверка — пересечение с финальной схемой)
    return df_test


# ============================================================ #
# 10. Логирование финальной схемы
# ============================================================ #
def log_final_schema(df: pd.DataFrame, cfg: dict) -> None:
    cat_cols = cfg["columns"]["categorical_decisions"].keys()
    all_cols = set(df.columns)
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    cat = df.select_dtypes(include="category").columns.tolist()
    obj = df.select_dtypes(include=["object","str"]).columns.tolist()
    flags = [c for c in numeric if df[c].dropna().isin([0, 1]).all()
             and df[c].nunique(dropna=True) <= 2]

    logger.info("final schema: %d columns", len(all_cols))
    logger.info("  numeric: %d", len(numeric))
    logger.info("  category: %d", len(cat))
    logger.info("  object (для WoE): %d", len(obj))
    logger.info("  flags (0/1): %d", len(flags))


# ============================================================ #
# 11. Сохранение
# ============================================================ #
def save_outputs(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    tr_idx, val_idx, te_idx,
    cfg: dict,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    id_col = cfg["data"]["id_col"]

    df.loc[tr_idx].set_index(id_col).to_parquet(out_dir / "train.parquet")
    df.loc[val_idx].set_index(id_col).to_parquet(out_dir / "valid.parquet")
    df.loc[te_idx].set_index(id_col).to_parquet(out_dir / "test.parquet")
    df_test.set_index(id_col).to_parquet(out_dir / "test_unlabeled.parquet")

    logger.info("saved 4 parquet → %s", out_dir)


# ============================================================ #
# Main
# ============================================================ #
def prepare(cfg: dict, out_dir: str | Path = "data/processed") -> None:
    df, df_test = load_and_validate(cfg)

    cat_decisions = cfg["columns"]["categorical_decisions"]

    # --- train ---
    df = apply_sensors(df, cfg["columns"]["sensors"])
    df = recode_to_flags(df, cat_decisions)
    df = recode_to_categorical(df, cfg["columns"]["recode_to_categorical"])
    df = apply_group_map(df, cat_decisions)
    df = group_rare_categories(df, cat_decisions)
    df = drop_columns(df, cfg["columns"]["drop"])

    df = df.copy()
    log_final_schema(df, cfg)

    # --- split ---
    tr_idx, val_idx, te_idx = split_indices(df, cfg)

    # --- test_unlabeled ---
    df_test = prepare_test_unlabeled(df_test, cfg)

    # --- schema consistency check ---
    target = cfg["data"]["target"]
    train_cols = set(df.columns) - {target}
    test_cols = set(df_test.columns)
    if train_cols != test_cols:
        diff = train_cols.symmetric_difference(test_cols)
        logger.warning("schema mismatch между train и test_unlabeled: %s", diff)

    # --- save ---
    save_outputs(df, df_test, tr_idx, val_idx, te_idx, cfg, out_dir)

    # --- sanity: prevalence ---
    target = cfg["data"]["target"]
    for name, idx in [("train", tr_idx), ("valid", val_idx), ("test", te_idx)]:
        prev = df.loc[idx, target].mean()
        logger.info("%s prevalence: %.4f", name, prev)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    prepare(cfg, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
