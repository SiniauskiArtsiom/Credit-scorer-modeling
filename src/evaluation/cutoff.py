"""Этап 9. Cutoff-анализ + Expected Loss.

Превращает PD в бизнес-решение:
  - Profit curve: прибыль как функция cutoff
  - Optimal cutoff: максимум прибыли
  - Expected Loss портфеля при выбранном cutoff
  - Sensitivity analysis: cost_ratio ∈ {2, 4, 6, 10}

Артефакты:
    artifacts/profit_curve.png         — кривая прибыли
    artifacts/cutoff_analysis.csv      — метрики по всем cutoff
    artifacts/cutoff_summary.json      — оптимальные cutoff для разных cost_ratio

Usage:
    python -m src.evaluation.cutoff --config configs/config.yaml
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

# Параметры бизнеса по умолчанию (можно переопределить в config)
DEFAULT_BUSINESS = {
    "EAD_mean": 500_000,      # средняя сумма кредита, CZK
    "LGD": 0.60,               # loss given default
    "rate_total": 0.15,        # прибыль с хорошего за весь срок / EAD
    "cost_ratio_range": [2, 4, 6, 10],   # FN/FP для sensitivity
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
# 2. Загрузка предсказаний champion
# ============================================================ #
def load_champion_test_preds(artifact_dir: Path, processed_dir: Path,
                             features: list[str]):
    with open(artifact_dir / "champion_logit.pkl", "rb") as f:
        model = pickle.load(f)["model"]
    df = pd.read_parquet(processed_dir / "test_woe.parquet")
    cols = [f"{f}_woe" for f in features]
    X = df[cols].astype(float)
    p = model.predict(sm.add_constant(X, has_constant="add")).values
    y = df["TARGET"].astype(int).values
    return y, p


# ============================================================ #
# 3. Метрики при cutoff
# ============================================================ #
def metrics_at_cutoff(y: np.ndarray, p: np.ndarray, cutoff: float) -> dict:
    """Все бизнес-метрики для одного cutoff."""
    approved = p < cutoff              # одобряем заявки с низким PD
    n_approved = int(approved.sum())
    n_rejected = int((~approved).sum())

    if n_approved == 0:
        return None

    y_appr = y[approved]
    tp = int(((y_appr == 0)).sum())    # true positive = одобренный хороший
    fp = int(((y_appr == 1)).sum())    # false positive = одобренный плохой

    return {
        "cutoff": round(float(cutoff), 4),
        "n_approved": n_approved,
        "n_rejected": n_rejected,
        "approval_rate": round(n_approved / len(y), 4),
        "good_approved": tp,
        "bad_approved": fp,
        "bad_rate_approved": round(fp / n_approved, 4) if n_approved else 0.0,
    }


# ============================================================ #
# 4. Прибыль
# ============================================================ #
def profit_at_cutoff(y: np.ndarray, p: np.ndarray, cutoff: float,
                     business: dict) -> dict:
    """Прибыль при данном cutoff."""
    m = metrics_at_cutoff(y, p, cutoff)
    if m is None:
        return None

    EAD = business["EAD_mean"]
    LGD = business["LGD"]
    rate = business["rate_total"]

    good_profit = rate * EAD           # прибыль с хорошего
    bad_loss = LGD * EAD               # потеря с плохого

    profit = m["good_approved"] * good_profit - m["bad_approved"] * bad_loss
    el_total = m["bad_approved"] * LGD * EAD       # ожидаемые потери на одобренном портфеле

    return {
        **m,
        "good_profit_per_loan": good_profit,
        "bad_loss_per_loan": bad_loss,
        "profit_total": float(profit),
        "expected_loss_total": float(el_total),
        "profit_per_approved": float(profit / m["n_approved"]) if m["n_approved"] else 0.0,
    }


# ============================================================ #
# 5. Sweep по cutoff
# ============================================================ #
def sweep_cutoffs(y: np.ndarray, p: np.ndarray, business: dict,
                  n_points: int = 200) -> pd.DataFrame:
    cutoffs = np.unique(np.quantile(p, np.linspace(0.001, 0.999, n_points)))
    rows = []
    for c in cutoffs:
        m = profit_at_cutoff(y, p, c, business)
        if m is not None:
            rows.append(m)
    df = pd.DataFrame(rows)
    logger.info("sweep: %d cutoffs computed", len(df))
    return df


def find_optimal(df: pd.DataFrame) -> pd.Series:
    return df.loc[df["profit_total"].idxmax()]


# ============================================================ #
# 6. Sensitivity: разные cost_ratio
# ============================================================ #
def make_business_for_cost_ratio(base: dict, cost_ratio: float) -> dict:
    """Cost_ratio = bad_loss / good_profit. Подбираем LGD при фиксированных EAD, rate."""
    b = dict(base)
    EAD = b["EAD_mean"]
    rate = b["rate_total"]
    b["LGD"] = min(1.0, cost_ratio * rate)   # так, что LGD·EAD / (rate·EAD) = cost_ratio
    return b


def sensitivity_analysis(y: np.ndarray, p: np.ndarray, base: dict) -> pd.DataFrame:
    rows = []
    for cr in base["cost_ratio_range"]:
        business = make_business_for_cost_ratio(base, cr)
        df = sweep_cutoffs(y, p, business, n_points=100)
        opt = find_optimal(df)
        rows.append({
            "cost_ratio": cr,
            "LGD": round(business["LGD"], 3),
            "optimal_cutoff": round(opt["cutoff"], 4),
            "approval_rate": round(opt["approval_rate"], 4),
            "bad_rate_approved": round(opt["bad_rate_approved"], 4),
            "profit_total": round(opt["profit_total"], 0),
        })
    return pd.DataFrame(rows)


# ============================================================ #
# 7. Графики
# ============================================================ #
def plot_profit_curve(df: pd.DataFrame, optimal: pd.Series, out_path: Path):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["cutoff"], df["profit_total"] / 1e6, "C0-", label="Profit (M CZK)")
    ax1.axvline(optimal["cutoff"], color="C3", ls="--",
                label=f"opt cutoff={optimal['cutoff']:.3f}")
    ax1.set_xlabel("PD cutoff")
    ax1.set_ylabel("Profit, M CZK", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(df["cutoff"], df["approval_rate"], "C2-", alpha=0.5,
             label="Approval rate")
    ax2.plot(df["cutoff"], df["bad_rate_approved"], "C1-", alpha=0.5,
             label="Bad rate (approved)")
    ax2.set_ylabel("Rate", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    plt.title("Profit vs Cutoff")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("saved %s", out_path)


# ============================================================ #
# 8. Main
# ============================================================ #
def run(cfg: dict, processed_dir: str | Path, artifact_dir: str | Path) -> None:
    processed_dir = resolve_path(processed_dir)
    artifact_dir = resolve_path(artifact_dir)
    features = load_selected_features(artifact_dir)

    logger.info("loading champion test predictions...")
    y, p = load_champion_test_preds(artifact_dir, processed_dir, features)

    business = {**DEFAULT_BUSINESS, **cfg.get("business", {})}
    logger.info("business params: %s", business)

    # --- sweep ---
    df = sweep_cutoffs(y, p, business, n_points=200)
    optimal = find_optimal(df)

    logger.info("=" * 60)
    logger.info("Optimal cutoff (base case)")
    logger.info("  cutoff          = %.4f", optimal["cutoff"])
    logger.info("  approval_rate   = %.2f%%", 100 * optimal["approval_rate"])
    logger.info("  bad_rate (appr) = %.2f%%", 100 * optimal["bad_rate_approved"])
    logger.info("  profit_total    = %.0f CZK  (%.2f M)",
                optimal["profit_total"], optimal["profit_total"] / 1e6)
    logger.info("  expected_loss   = %.0f CZK  (%.2f M)",
                optimal["expected_loss_total"], optimal["expected_loss_total"] / 1e6)

    # --- график ---
    plot_profit_curve(df, optimal, artifact_dir / "profit_curve.png")

    # --- save sweep ---
    df.to_csv(artifact_dir / "cutoff_analysis.csv", index=False)

    # --- sensitivity ---
    logger.info("=" * 60)
    logger.info("Sensitivity analysis (different cost ratios)")
    sens = sensitivity_analysis(y, p, business)
    logger.info("\n%s", sens.to_string(index=False))

    # --- summary json ---
    summary = {
        "business_params": business,
        "optimal_base": {
            "cutoff": float(optimal["cutoff"]),
            "approval_rate": float(optimal["approval_rate"]),
            "bad_rate_approved": float(optimal["bad_rate_approved"]),
            "profit_total": float(optimal["profit_total"]),
            "expected_loss_total": float(optimal["expected_loss_total"]),
        },
        "sensitivity": sens.to_dict(orient="records"),
    }
    with open(artifact_dir / "cutoff_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("saved cutoff_summary.json, cutoff_analysis.csv, profit_curve.png")


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
