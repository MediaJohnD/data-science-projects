"""XGBoost model utilities with cross-validation."""

from typing import Any, Dict, Tuple

import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier


def train_model_cv(X, y, param_grid: Dict[str, Any] | None = None, cv: int = 5) -> Tuple[XGBClassifier, Dict[str, Any] | None, float | None]:
    """Train an XGBoost model with optional grid search CV."""
    base_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    if param_grid:
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        search = GridSearchCV(
            base_model,
            param_grid,
            scoring="roc_auc",
            cv=cv_strategy,
            n_jobs=-1,
        )
        search.fit(X, y)
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        return best_model, best_params, best_score

    base_model.fit(X, y)
    return base_model, None, None


def evaluate_model(model: XGBClassifier, X, y) -> Dict[str, float]:
    """Return accuracy and ROC AUC for the given dataset."""
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, preds),
        "roc_auc": roc_auc_score(y, proba),
    }


def classification_summary(model: XGBClassifier, X, y) -> str:
    preds = model.predict(X)
    return classification_report(y, preds)


def save_model(model: XGBClassifier, path: str):
    joblib.dump(model, path)


def load_model(path: str) -> XGBClassifier:
    return joblib.load(path)
