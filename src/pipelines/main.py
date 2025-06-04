from pathlib import Path
from src.ingestion.csv_loader import load_csv
from src.features.feature_generator import generate_features
from src.models.xgb_model import (
    classification_summary,
    evaluate_model,
    save_model,
    train_model_cv,
)


def run_pipeline(config):
    data_path = Path(config['data_path'])
    target_col = config['target_column']
    model_path = Path(config.get('model_path', 'model.joblib'))

    df = load_csv(data_path)
    X, y, _ = generate_features(df, target_col)

    param_grid = config.get("param_grid")
    if param_grid is None and isinstance(config.get("model"), dict):
        param_grid = config["model"].get("param_grid")
    cv = config.get("cv", 5)
    model, best_params, best_score = train_model_cv(X, y, param_grid, cv=cv)
    metrics = evaluate_model(model, X, y)
    print(f"Training metrics: {metrics}")
    if best_params:
        print(f"Best params: {best_params}, CV score: {best_score:.3f}")
        print(classification_summary(model, X, y))

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
