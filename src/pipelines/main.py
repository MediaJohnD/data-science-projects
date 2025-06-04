from pathlib import Path

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shap = None
from src.ingestion.csv_loader import load_csv
from src.features.feature_generator import generate_features
from src.models import (
    save_model,
    train_model_cv,
    train_knn_classifier,
    train_random_forest,
    train_dbscan,
    train_isolation_forest,
    train_autoencoder,
    train_rnn_classifier,
    train_knn_classifier_cv,
    train_random_forest_cv,
    train_dbscan_cv,
    classification_metrics,
    clustering_metrics,
    save_metrics,
    time_aware_split,
    geo_holdout_split,
)


def run_pipeline(config):
    data_path = Path(config['data_path'])
    target_col = config['target_column']
    model_path = Path(config.get('model_path', 'model.joblib'))

    df = load_csv(data_path)
    X, y, _ = generate_features(df, target_col)

    train_idx = list(range(len(df)))
    test_idx = []
    if "time_column" in config:
        train_idx, test_idx = time_aware_split(df, config["time_column"], config.get("test_size", 0.2))
    if "geo_column" in config and "holdout_geos" in config:
        train_idx, test_idx = geo_holdout_split(df, config["geo_column"], config["holdout_geos"])

    X_train, y_train = X[train_idx], y.iloc[train_idx]
    X_test = y_test = None
    if len(test_idx) > 0:
        X_test, y_test = X[test_idx], y.iloc[test_idx]

    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "xgboost")
    param_grid = config.get("param_grid") or model_cfg.get("param_grid")
    cv = config.get("cv", 5)

    metrics = {}
    if model_type == "xgboost":
        model, best_params, best_score = train_model_cv(X_train, y_train, param_grid, cv=cv)
        y_pred = model.predict(X_train)
        proba = model.predict_proba(X_train)[:, 1]
        metrics["train"] = classification_metrics(y_train, y_pred, proba)
    elif model_type == "knn":
        if param_grid:
            model, best_params, best_score = train_knn_classifier_cv(X_train, y_train, param_grid, cv=cv)
        else:
            n_neighbors = model_cfg.get("n_neighbors", 5)
            model = train_knn_classifier(X_train, y_train, n_neighbors)
            best_params = best_score = None
        y_pred = model.predict(X_train)
        metrics["train"] = classification_metrics(y_train, y_pred)
    elif model_type == "random_forest":
        if param_grid:
            model, best_params, best_score = train_random_forest_cv(X_train, y_train, param_grid, cv=cv)
        else:
            model = train_random_forest(X_train, y_train, **model_cfg.get("params", {}))
            best_params = best_score = None
        y_pred = model.predict(X_train)
        proba = model.predict_proba(X_train)[:, 1]
        metrics["train"] = classification_metrics(y_train, y_pred, proba)
    elif model_type == "dbscan":
        if param_grid:
            model, best_params, best_score = train_dbscan_cv(X_train, param_grid)
        else:
            model = train_dbscan(X_train, **model_cfg.get("params", {}))
            best_params = best_score = None
        metrics["train"] = clustering_metrics(X_train, model.labels_)
    elif model_type == "isolation_forest":
        model = train_isolation_forest(X_train, **model_cfg.get("params", {}))
        metrics["train"] = {}
        best_params = best_score = None
    elif model_type == "autoencoder":
        model, encoded = train_autoencoder(X_train, **model_cfg.get("params", {}))
        metrics["train"] = {}
        best_params = best_score = None
    elif model_type == "rnn":
        model = train_rnn_classifier(X_train, y_train, **model_cfg.get("params", {}))
        y_pred = model.predict(X_train)
        metrics["train"] = classification_metrics(y_train, y_pred.ravel() > 0.5)
        best_params = best_score = None
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if X_test is not None:
        if model_type in {"dbscan"}:
            metrics["test"] = clustering_metrics(X_test, model.fit_predict(X_test))
        else:
            y_pred = model.predict(X_test)
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]
            metrics["test"] = classification_metrics(y_test, y_pred, proba)

    if best_params:
        print(f"Best params: {best_params}, CV score: {best_score:.3f}")

    metrics_path = config.get("metrics_path", "metrics.json")
    save_metrics(metrics, metrics_path)
    save_model(model, model_path)
    if shap is not None and model_type in {"xgboost", "random_forest"}:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_train[:100])
        shap.save("shap_values.pkl", shap_values)
    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    example_config = {
        'data_path': 'data/raw/example.csv',
        'target_column': 'target',
        'model_path': 'model.joblib',
        'param_grid': {
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0],
        }
    }
    run_pipeline(example_config)
