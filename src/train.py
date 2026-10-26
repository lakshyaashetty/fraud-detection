import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import load
from src.features import FEATURES, build_features


def report(name, y_true, score, flag):
    print(f"{name:20s} PR-AUC {average_precision_score(y_true, score):.3f} | "
          f"recall {recall_score(y_true, flag):.3f} | "
          f"precision {precision_score(y_true, flag, zero_division=0):.3f}")


def main():
    df = load().sort_values("step", kind="stable").reset_index(drop=True)
    X, y, step, amount = build_features(df)

    # Time-based split (see build log: steps 401+ are a separate stress test)
    tr = step <= 300
    va = (step > 300) & (step <= 350)
    print(f"train rows {tr.sum():,} | valid rows {va.sum():,}")

    # Baseline 0: the old rule
    rule = df["isFlaggedFraud"]
    report("Old rule", y[va], rule[va], rule[va])

    # Baseline 1: logistic regression
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(class_weight="balanced", max_iter=1000))
    lr.fit(X[tr], y[tr])
    p = lr.predict_proba(X[va])[:, 1]
    report("Logistic regression", y[va], p, p >= 0.5)

    # Main model: XGBoost
    spw = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    xgb = XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="aucpr", early_stopping_rounds=100,
        tree_method="hist", n_jobs=-1,
    )
    xgb.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=100)
    p = xgb.predict_proba(X[va])[:, 1]
    report("XGBoost", y[va], p, p >= 0.5)

    joblib.dump({"model": xgb, "features": FEATURES}, "models/model.joblib")
    print("Saved models/model.joblib")


if __name__ == "__main__":
    main()
