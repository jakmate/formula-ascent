import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from app.config import SEED
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import warnings


def analyze_features(df, feature_cols, temporal_split_col="year", seed=SEED):
    """
    Analyze feature importance, redundancy, and recommend a robust subset.

    Args:
        df: DataFrame with features and 'promoted' target
        feature_cols: list of feature column names
        temporal_split_col: column to use for temporal validation split (e.g., 'year')
        seed: random seed for reproducibility
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS: Redundancy + Predictive Importance")
    print("=" * 60)

    # Filter out rows with NaN in target variable
    df_clean = df.dropna(subset=["promoted"]).copy()

    if df_clean.empty:
        print("\n⚠️  No valid data after removing NaN targets")
        return None

    print(f"\nUsing {len(df_clean)} samples ()")
    print(f"Removed {len(df) - len(df_clean)} with NaN targets")

    X = df_clean[feature_cols].copy()
    y = df_clean["promoted"].copy()

    # Handle missing values safely
    X = X.fillna(0)

    # ===== 1. Redundancy Analysis =====
    print("\n🔍 REDUNDANCY ANALYSIS")
    print("-" * 40)

    # Correlation matrix
    corr_matrix = X.corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append(
                    (
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j],
                    )
                )

    if high_corr_pairs:
        print("\n⚠️  Highly Correlated Pairs (|r| > 0.8):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  • {feat1} ↔ {feat2}: {corr:.3f}")
    else:
        print("\n✅ No highly correlated feature pairs (|r| ≤ 0.8)")

    # VIF (only if not singular matrix)
    try:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(X.shape[1])
        ]
        high_vif = vif_data[vif_data["VIF"] > 10]

        if not high_vif.empty:
            print("\n⚠️  High Multicollinearity (VIF > 10):")
            print(high_vif[["feature", "VIF"]].to_string(index=False))
        else:
            print("\n✅ No severe multicollinearity (VIF ≤ 10)")
    except np.linalg.LinAlgError:
        print("\n⚠️  VIF skipped: singular matrix (perfect collinearity)")
        vif_data = pd.DataFrame(
            {"feature": X.columns, "VIF": [np.nan] * len(X.columns)}
        )

    # ===== 2. Predictive Power Analysis =====
    print("\n🎯 PREDICTIVE IMPORTANCE (Imbalanced-Aware)")
    print("-" * 40)

    # Temporal split: use last 20% of years as validation
    years = df_clean[temporal_split_col].unique()
    years = sorted(years)
    n_val_years = max(1, int(len(years) * 0.2))
    val_years = years[-n_val_years:]

    train_mask = ~df_clean[temporal_split_col].isin(val_years)
    val_mask = df_clean[temporal_split_col].isin(val_years)

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    if y_val.sum() == 0:
        # Fallback to stratified random split if no positives in temporal val
        print("\n⚠️  No promotions in temporal validation → using stratified split")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )

    # Mutual Information (univariate signal)
    mi_scores = mutual_info_classif(X_train, y_train, random_state=seed)
    mi_df = pd.DataFrame({"feature": X.columns, "MI_score": mi_scores}).sort_values(
        "MI_score", ascending=False
    )

    # Permutation Importance (true predictive power)
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=seed,
        max_depth=5,  # Reduce overfitting on small data
    )
    model.fit(X_train, y_train)

    perm_imp = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="average_precision",  # Critical for imbalanced data
        n_repeats=10,
        random_state=seed,
        n_jobs=-1,
    )

    perm_df = pd.DataFrame(
        {
            "feature": X.columns,
            "perm_importance": perm_imp.importances_mean,
            "perm_std": perm_imp.importances_std,
        }
    ).sort_values("perm_importance", ascending=False)

    # Combine results
    importance_df = mi_df.merge(perm_df, on="feature")
    print("\n📊 Feature Importance Summary:")
    print(importance_df.to_string(index=False, float_format="%.4f"))

    # ===== 3. Recommendations =====
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)

    # Identify redundant groups
    redundant_groups = []
    used = set()
    for feat1, feat2, _ in high_corr_pairs:
        if feat1 in used or feat2 in used:
            continue
        # Find all features correlated with feat1
        group = {feat1, feat2}
        for f1, f2, r in high_corr_pairs:
            if f1 in group or f2 in group:
                group.add(f1)
                group.add(f2)
        redundant_groups.append(group)
        used.update(group)

    # For each redundant group, keep the one with highest permutation importance
    to_remove = set()
    for group in redundant_groups:
        group_df = importance_df[importance_df["feature"].isin(group)]
        best_in_group = group_df.loc[group_df["perm_importance"].idxmax(), "feature"]
        to_remove.update(group - {best_in_group})
        print(f"\n🔁 Redundant group: {sorted(group)}")
        print(
            f"   → Keep: {best_in_group} (perm_importance={group_df.loc[group_df['feature'] == best_in_group, 'perm_importance'].iloc[0]:.4f})"  # noqa: F501
        )
        for f in group - {best_in_group}:
            print(f"   → Remove: {f}")

    # Also remove features with non-positive permutation importance
    non_predictive = importance_df[importance_df["perm_importance"] <= 0][
        "feature"
    ].tolist()
    non_predictive = [
        f for f in non_predictive if f not in to_remove
    ]  # avoid duplicates
    if non_predictive:
        print("\n🗑️  Non-predictive features (perm_importance ≤ 0):")
        for f in non_predictive:
            imp = importance_df.loc[
                importance_df["feature"] == f, "perm_importance"
            ].iloc[0]
            print(f"   • {f} ({imp:.4f})")
        to_remove.update(non_predictive)

    # Final recommended set
    recommended = [f for f in feature_cols if f not in to_remove]
    print(f"\n✅ Recommended feature set ({len(recommended)}/{len(feature_cols)}):")
    for i, f in enumerate(recommended, 1):
        print(f"  {i}. {f}")

    return {
        "correlation_matrix": corr_matrix,
        "vif_data": vif_data,
        "importance_df": importance_df,
        "redundant_groups": redundant_groups,
        "recommended_features": recommended,
        "features_to_remove": list(to_remove),
    }
