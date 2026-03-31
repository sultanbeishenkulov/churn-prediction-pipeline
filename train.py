import joblib
import mlflow
import mlflow.xgboost
import optuna
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "churn-xgboost"


def objective(trial, X_train, y_train, scale_pos_weight):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "random_state": 42,
        "verbosity": 0,
    }
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    return scores.mean()


def train(X_train, y_train, X_test, y_test, n_trials=50):
    # Compute once, reuse everywhere
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda t: objective(t, X_train, y_train, scale_pos_weight),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_params = {
            **study.best_params,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "auc",
            "random_state": 42,
            "verbosity": 0,
        }
        mlflow.log_params(best_params)

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_avg_precision", ap)

        print(f"\nTest AUC-ROC       : {auc:.4f}")
        print(f"Test Avg Precision : {ap:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

        # Save model
        model_path = MODEL_DIR / "model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        # Save feature names for inference alignment
        feature_path = MODEL_DIR / "feature_names.txt"
        feature_path.write_text("\n".join(X_train.columns.tolist()))

    return model


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from load import load_raw
    from engineer import engineer_features

    df = load_raw()
    df = engineer_features(df)
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train(X_train, y_train, X_test, y_test)