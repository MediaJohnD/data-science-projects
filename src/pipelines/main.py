from pathlib import Path
from src.ingestion.csv_loader import load_csv
from src.features.feature_generator import generate_features
from src.models import (
    classification_summary,
    evaluate_model,
    save_model,
    train_model_cv,
    train_knn_classifier,
    train_random_forest,
    train_dbscan,
    train_isolation_forest,
    train_autoencoder,
    train_rnn_classifier,
    evaluate_classifier,
)


def run_pipeline(config):
    data_path = Path(config['data_path'])
    target_col = config['target_column']
    model_path = Path(config.get('model_path', 'model.joblib'))

    df = load_csv(data_path)
    X, y, _ = generate_features(df, target_col)

    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "xgboost")
    param_grid = config.get("param_grid") or model_cfg.get("param_grid")
    cv = config.get("cv", 5)

    if model_type == "xgboost":
        model, best_params, best_score = train_model_cv(X, y, param_grid, cv=cv)
        metrics = evaluate_model(model, X, y)
        print(f"Training metrics: {metrics}")
        if best_params:
            print(f"Best params: {best_params}, CV score: {best_score:.3f}")
            print(classification_summary(model, X, y))
    elif model_type == "knn":
        n_neighbors = model_cfg.get("n_neighbors", 5)
        model = train_knn_classifier(X, y, n_neighbors)
        acc = evaluate_classifier(model, X, y)
        print(f"KNN accuracy: {acc:.3f}")
        best_params = best_score = None
    elif model_type == "random_forest":
        model = train_random_forest(X, y, **model_cfg.get("params", {}))
        acc = evaluate_classifier(model, X, y)
        print(f"Random Forest accuracy: {acc:.3f}")
        best_params = best_score = None
    elif model_type == "dbscan":
        model = train_dbscan(X, **model_cfg.get("params", {}))
        print(f"DBSCAN labels: {set(model.labels_)}")
        best_params = best_score = None
    elif model_type == "isolation_forest":
        model = train_isolation_forest(X, **model_cfg.get("params", {}))
        print("Isolation Forest model trained")
        best_params = best_score = None
    elif model_type == "autoencoder":
        model, encoded = train_autoencoder(X, **model_cfg.get("params", {}))
        print(f"Autoencoder trained, encoded shape: {encoded.shape}")
        best_params = best_score = None
    elif model_type == "rnn":
        model = train_rnn_classifier(X, y, **model_cfg.get("params", {}))
        acc = evaluate_classifier(model, X, y)
        print(f"RNN accuracy: {acc:.3f}")
        best_params = best_score = None
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    save_model(model, model_path)
    print(f"Model saved to {model_path}")


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
