import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import load
from src.features import FEATURES, build_features

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection")


def evaluate(name, y_true, score, flag, params):
    metrics = {
        "val_pr_auc": average_precision_score(y_true, score),
        "val_recall_at_0.5": recall_score(y_true, flag),
        "val_precision_at_0.5": precision_score(y_true, flag, zero_division=0),
    }
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    print(f"{name:20s} PR-AUC {metrics['val_pr_auc']:.3f}")


def main():
    df = load().sort_values("step", kind="stable").reset_index(drop=True)
    X, y, step, amount = build_features(df)
    tr = step <= 300
    va = (step > 300) & (step <= 350)

    rule = df["isFlaggedFraud"]
    evaluate("old_rule", y[va], rule[va], rule[va], {"model": "rule"})

    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    lr.fit(X[tr], y[tr])
    p = lr.predict_proba(X[va])[:, 1]
    evaluate("logistic_regression", y[va], p, p >= 0.5, {"model": "logreg", "class_weight": "balanced"})

    params = {"n_estimators": 2000, "learning_rate": 0.05, "max_depth": 6,
              "subsample": 0.8, "colsample_bytree": 0.8}
    spw = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    xgb = XGBClassifier(**params, scale_pos_weight=spw, eval_metric="aucpr",
                        early_stopping_rounds=100, tree_method="hist", n_jobs=-1)
    xgb.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=100)
    p = xgb.predict_proba(X[va])[:, 1]
    evaluate("xgboost", y[va], p, p >= 0.5,
             {**params, "model": "xgboost", "best_iteration": xgb.best_iteration})

    joblib.dump({"model": xgb, "features": FEATURES}, "models/model.joblib")
    print("Saved models/model.joblib")


if __name__ == "__main__":
    main()
