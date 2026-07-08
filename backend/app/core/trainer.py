import logging

import torch
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from torch import nn, optim

from app.config import SEED
from app.core.pytorch_model import RacingPredictor

log = logging.getLogger(__name__)


def train_pytorch_model(x_train_sub, y_train_sub, x_val, y_val, x_test, feature_cols):
    """Train a PyTorch neural network model for binary classification."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scale features
    scaler = RobustScaler()
    x_train_sub_scaled = scaler.fit_transform(x_train_sub)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # Convert to PyTorch tensors
    x_train_torch = torch.FloatTensor(x_train_sub_scaled).to(device)
    y_train_torch = torch.FloatTensor(y_train_sub.values.copy()).to(device)
    x_val_torch = torch.FloatTensor(x_val_scaled).to(device)
    y_val_torch = torch.FloatTensor(y_val.values.copy()).to(device)
    x_test_torch = torch.FloatTensor(x_test_scaled).to(device)

    pytorch_model = RacingPredictor(x_train_sub_scaled.shape[1]).to(device)

    # Calculate class weights for imbalanced data
    n_neg = (y_train_sub == 0).sum()
    n_pos = (y_train_sub == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)

    # Loss function, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(pytorch_model.parameters(), lr=0.01, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for _ in range(30):
        pytorch_model.train()
        optimizer.zero_grad()

        outputs = pytorch_model(x_train_torch).squeeze()
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()

        # Validation
        pytorch_model.eval()
        with torch.no_grad():
            val_outputs = pytorch_model(x_val_torch).squeeze()
            val_loss = criterion(val_outputs, y_val_torch)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = pytorch_model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 3:
                break

    # Load best model and evaluate
    pytorch_model.load_state_dict(best_state)
    pytorch_model.eval()

    # Validation and Test evaluation
    with torch.no_grad():
        val_probas = torch.sigmoid(pytorch_model(x_val_torch)).cpu().numpy().flatten()
        test_probas = torch.sigmoid(pytorch_model(x_test_torch)).cpu().numpy().flatten()

    # Calibrate PyTorch model using validation set
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(val_probas, y_val)
    pytorch_model.calibrator = iso_reg

    # Store additional attributes for inference
    pytorch_model.scaler = scaler
    pytorch_model.feature_cols = feature_cols
    pytorch_model.device = device

    return pytorch_model, scaler, test_probas


def train_models(df):
    """Training function."""
    if df.empty:
        log.warning("No data available for training")
        return {}, None, None

    df_clean = df.dropna(subset=["promoted"])

    if df_clean["series"][0] == "F2":
        feature_cols = [
            "experience",
            "std_quali_pos",
            "feature_win_rate",
            "champ_pos_pct",
            "nationality_encoded",
        ]
    else:
        feature_cols = [
            "avg_quali_pos",
            "sprint_win_rate",
            "feature_win_rate",
            "experience",
            "teammate_h2h_rate",
            "nationality_encoded",
            "participation_rate",
            "dnf_rate",
            "champ_pos_pct",
        ]

    x = df_clean[feature_cols].fillna(0)
    y = df_clean["promoted"]
    years = df_clean["year"]

    # Temporal split
    unique_years = sorted(years.unique())
    n_train_years = int(len(unique_years) * 0.8)
    train_years = unique_years[:n_train_years]

    train_mask = years.isin(train_years)
    x_train, x_test = x[train_mask], x[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(skf.split(x_train, y_train))

    x_train_sub, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
    y_train_sub, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    log.debug(
        "Training subset: %s samples, %s promotions (%.2f)",
        len(x_train_sub),
        y_train_sub.sum(),
        y_train_sub.mean(),
    )
    log.debug(
        "Validation: %s samples, %s promotions (%.2f)",
        len(x_val),
        y_val.sum(),
        y_val.mean(),
    )
    log.debug(
        "Test: %s samples, %s promotions (%.2f)",
        len(x_test),
        y_test.sum(),
        y_test.mean(),
    )

    # Traditional ML pipelines
    traditional_pipelines = {
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
            ],
        ),
        "LightGBM": Pipeline(
            [
                (
                    "classifier",
                    LGBMClassifier(
                        random_state=SEED,
                        class_weight="balanced",
                        verbosity=-1,
                        n_jobs=-1,
                    ),
                ),
            ],
        ),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=SEED,
                        class_weight="balanced",
                        max_iter=10000,
                    ),
                ),
            ],
        ),
        "Naive Bayes": Pipeline([("classifier", GaussianNB())]),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        random_state=SEED,
                        C=1,
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                    ),
                ),
            ],
        ),
        "Random Forest": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=SEED,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                    ),
                ),
            ],
        ),
    }

    results = {}

    # Train traditional models
    for name, pipeline in traditional_pipelines.items():
        log.info("Training %s", name)
        log.debug("-" * 40)

        # Fit on training subset
        pipeline.fit(x_train_sub, y_train_sub)

        # Evaluate on validation set
        probas_val = pipeline.predict_proba(x_val)[:, 1]

        # Evaluate on test set
        y_pred = pipeline.predict(x_test)
        probas_test = pipeline.predict_proba(x_test)[:, 1]

        # Calibration using validation set
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(probas_val, y_val)
        pipeline.calibrator = iso_reg

        log.debug(classification_report(y_test, y_pred, zero_division=0))
        calibrated_probas = pipeline.calibrator.transform(probas_test)
        pr_auc_calibrated = average_precision_score(y_test, calibrated_probas)
        log.debug("Test PR-AUC (calibrated): %.4f", pr_auc_calibrated)

        results[name] = pipeline

    # Train PyTorch Model
    log.info("Training PyTorch Model")
    log.debug("-" * 40)
    pytorch_model, scaler, test_probas = train_pytorch_model(
        x_train_sub,
        y_train_sub,
        x_val,
        y_val,
        x_test,
        feature_cols,
    )

    calibrated_probas = pytorch_model.calibrator.transform(test_probas)
    y_pred = (test_probas > 0.5).astype(int)
    log.debug(classification_report(y_test, y_pred, zero_division=0))

    pr_auc_calibrated = average_precision_score(y_test, calibrated_probas)
    log.debug("Test PR-AUC (calibrated): %.4f", pr_auc_calibrated)
    results["PyTorch"] = pytorch_model

    return results, feature_cols, scaler
