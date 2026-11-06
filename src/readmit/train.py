"""Train a baseline and a gradient-boosted model, pick one on the validation block,
calibrate it, and save it with a model card."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from .features import build_features, make_preprocessor
from .schema import TARGET

SEED = 42


def candidates() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=100,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                random_state=SEED)),
        ]),
    }


def fit_and_select(train_df, valid_df):
    Xtr, ytr = build_features(train_df), train_df[TARGET].values
    Xva, yva = build_features(valid_df), valid_df[TARGET].values
    results, fitted = {}, {}
    for name, model in candidates().items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
        results[name] = {"valid_auroc": float(roc_auc_score(yva, p)),
                         "valid_auprc": float(average_precision_score(yva, p))}
        fitted[name] = model
    best = max(results, key=lambda k: results[k]["valid_auroc"])
    # Calibrate on the held-out validation block so risk scores read as probabilities.
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted[best]), method="isotonic")
    calibrated.fit(Xva, yva)
    return calibrated, best, results


def save(model, best, results, train_df, out_dir: Path, cleaning: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    card = {
        "model_name": "readmit-30d",
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "selected_algorithm": best,
        "candidate_validation_metrics": results,
        "calibration": "isotonic, fit on validation block",
        "training_rows": int(len(train_df)),
        "training_positive_rate": float(train_df[TARGET].mean()),
        "intended_use": "Rank adult diabetic inpatients at discharge for transitional-care "
                        "outreach. Decision support only; not for denying care.",
        "not_features": ["race", "gender"],
        "cleaning": cleaning,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
