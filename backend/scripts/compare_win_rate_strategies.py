import argparse
import logging
import random

import numpy as np
import pandas as pd
import torch

from app.config import SEED
from app.core.feature_creator import calculate_qualifying_features, engineer_features
from app.core.loader import load_data, load_qualifying_data, load_standings_data
from app.core.predictor import create_target_variable
from app.core.trainer import get_default_feature_cols, train_models

log = logging.getLogger(__name__)

SERIES_MAPPING = {
    "F3": "F2",
    "F2": "F1",
}


def build_trainable_df(feeder_series: str) -> pd.DataFrame:
    parent_series = SERIES_MAPPING[feeder_series]

    feeder_df = load_data(feeder_series)
    parent_df = load_standings_data(parent_series, "drivers")

    qualifying_df = load_qualifying_data(feeder_series)
    if not qualifying_df.empty and {"Driver", "year"}.issubset(qualifying_df.columns):
        feeder_df = calculate_qualifying_features(feeder_df, qualifying_df)
    else:
        feeder_df["avg_quali_pos"] = np.nan
        feeder_df["std_quali_pos"] = np.nan

    feeder_df = create_target_variable(feeder_df, parent_df, parent_series)

    features_df = engineer_features(feeder_df)
    features_df["promoted"] = feeder_df["promoted"]

    return features_df


def add_weighted_win_rate(df: pd.DataFrame, sprint_weight: float) -> pd.DataFrame:
    weighted_df = df.copy()

    numerator = weighted_df["feature_wins"].fillna(0) + sprint_weight * weighted_df[
        "sprint_wins"
    ].fillna(0)
    denominator = weighted_df["feature_races"].fillna(0) + sprint_weight * weighted_df[
        "sprint_races"
    ].fillna(0)

    weighted_df["weighted_win_rate"] = np.where(
        denominator > 0, numerator / denominator, 0
    )
    return weighted_df


def build_feature_variants(base_feature_cols: list[str]) -> dict[str, list[str]]:
    def replace_with_single(cols: list[str], replacement: str) -> list[str]:
        out = []
        inserted = False
        for col in cols:
            if col in {
                "sprint_win_rate",
                "feature_win_rate",
                "win_rate",
                "weighted_win_rate",
            }:
                if not inserted:
                    out.append(replacement)
                    inserted = True
            else:
                out.append(col)
        if not inserted:
            out.append(replacement)
        return out

    split_cols = []
    split_inserted = False
    for col in base_feature_cols:
        if col in {
            "sprint_win_rate",
            "feature_win_rate",
            "win_rate",
            "weighted_win_rate",
        }:
            if not split_inserted:
                split_cols.extend(["sprint_win_rate", "feature_win_rate"])
                split_inserted = True
        else:
            split_cols.append(col)
    if not split_inserted:
        split_cols.extend(["sprint_win_rate", "feature_win_rate"])

    return {
        "overall_win_rate": replace_with_single(base_feature_cols, "win_rate"),
        "split_sprint_feature": split_cols,
        "weighted_win_rate": replace_with_single(
            base_feature_cols, "weighted_win_rate"
        ),
    }


def evaluate_strategy(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[float, dict[str, float]]:
    # Reset all RNGs before each fit so strategy runs are comparable and reproducible.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    _, _, _, per_model_metrics = train_models(
        df,
        feature_cols_override=feature_cols,
        include_metrics=True,
    )

    mean_pr_auc = float(np.mean(list(per_model_metrics.values())))
    return mean_pr_auc, per_model_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare win-rate feature strategies using calibrated test PR-AUC "
            "across all models in train_models."
        ),
    )
    parser.add_argument(
        "--series",
        choices=["F3", "F2"],
        default="F2",
        help="Feeder series to evaluate (F3 evaluates promotion to F2, F2 to F1).",
    )
    parser.add_argument(
        "--sprint-weight",
        type=float,
        default=0.5,
        help="Weight applied to sprint wins/races in weighted_win_rate.",
    )
    parser.add_argument(
        "--sweep-weighted",
        action="store_true",
        help="Sweep weighted_win_rate sprint weights and report the best one.",
    )
    parser.add_argument(
        "--sweep-start",
        type=float,
        default=0.0,
        help="Start value for weighted sweep (inclusive).",
    )
    parser.add_argument(
        "--sweep-end",
        type=float,
        default=1.0,
        help="End value for weighted sweep (inclusive).",
    )
    parser.add_argument(
        "--sweep-step",
        type=float,
        default=0.05,
        help="Step size for weighted sweep.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    trainable_df = build_trainable_df(args.series)

    base_features = get_default_feature_cols(args.series)
    variants = build_feature_variants(base_features)

    sweep_results = []
    best_weight = args.sprint_weight
    if args.sweep_weighted:
        if args.sweep_step <= 0:
            raise ValueError("--sweep-step must be > 0")
        if args.sweep_end < args.sweep_start:
            raise ValueError("--sweep-end must be >= --sweep-start")

        weight = args.sweep_start
        best_score = float("-inf")
        while weight <= args.sweep_end + 1e-9:
            weighted_df = add_weighted_win_rate(trainable_df, round(weight, 6))
            weighted_features = build_feature_variants(base_features)[
                "weighted_win_rate"
            ]
            mean_pr_auc, per_model = evaluate_strategy(weighted_df, weighted_features)
            rounded_weight = round(weight, 6)
            sweep_results.append((rounded_weight, mean_pr_auc, per_model))

            if mean_pr_auc > best_score:
                best_score = mean_pr_auc
                best_weight = rounded_weight

            weight += args.sweep_step

        log.info("Best weighted sprint weight from sweep: %s", best_weight)

    trainable_df = add_weighted_win_rate(trainable_df, best_weight)

    results = []
    for strategy_name, feature_cols in variants.items():
        log.info("Evaluating %s with features: %s", strategy_name, feature_cols)
        mean_pr_auc, per_model = evaluate_strategy(trainable_df, feature_cols)
        results.append((strategy_name, mean_pr_auc, per_model))

    results.sort(key=lambda item: item[1], reverse=True)

    print("\n=== Win-Rate Strategy Comparison ===")
    print(f"Series: {args.series} | Sprint weight: {best_weight}")
    print("Rank | Strategy              | Mean PR-AUC")
    print("-----+-----------------------+------------")

    for idx, (name, mean_score, _) in enumerate(results, start=1):
        print(f"{idx:>4} | {name:<21} | {mean_score:.4f}")

    print("\nPer-model PR-AUC:")
    for name, _, per_model in results:
        print(f"\n{name}:")
        for model_name, score in sorted(per_model.items()):
            print(f"  - {model_name:<20} {score:.4f}")

    if sweep_results:
        print("\n=== Weighted Sweep Results ===")
        print("Weight | Mean PR-AUC")
        print("-------+------------")
        for weight, score, _ in sorted(sweep_results, key=lambda x: x[0]):
            print(f"{weight:>6.2f} | {score:.4f}")
        print(f"\nBest weighted sprint weight: {best_weight}")


if __name__ == "__main__":
    main()
