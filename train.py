import xgboost as xgb
import optuna
import mlflow
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, X_train, y_train):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "auc",
        "random_state": 42,
    }
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    return scores.mean()

def train(X_train, y_train, X_test, y_test, n_trials=50):
    with mlflow.start_run():
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, X_train, y_train), n_trials=n_trials)

        best_params = study.best_params
        best_params["scale_pos_weight"] = (y_train == 0).sum() / (y_train == 1).sum()
        mlflow.log_params(best_params)

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("test_auc", auc)
        print(f"\nTest AUC: {auc:.4f}")
        print(classification_report(y_test, model.predict(X_test)))

        joblib.dump(model, "model.joblib")
        mlflow.log_artifact("model.joblib")

    return model
