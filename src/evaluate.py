import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from src.data import load
from src.features import build_features

ALERT_BUDGET = 0.01  # fraud team can review 1% of transactions


def summarize(name, p, y, amt, th):
    flag = p >= th
    share = amt[flag & (y == 1)].sum() / amt[y == 1].sum()
    print(f"{name:6s} PR-AUC {average_precision_score(y, p):.3f} | ROC-AUC {roc_auc_score(y, p):.3f} | "
          f"recall {recall_score(y, flag):.3f} | precision {precision_score(y, flag, zero_division=0):.3f} | "
          f"flagged per 1,000 {flag.mean() * 1000:.1f} | fraud $ caught {share:.1%}")


def main():
    bundle = joblib.load("models/model.joblib")
    model, features = bundle["model"], bundle["features"]

    df = load().sort_values("step", kind="stable").reset_index(drop=True)
    X, y, step, amount = build_features(df)
    va = ((step > 300) & (step <= 350)).to_numpy()
    te = ((step > 350) & (step <= 400)).to_numpy()
    st = (step > 400).to_numpy()
    p = model.predict_proba(X[features])[:, 1]
    y, amount = y.to_numpy(), amount.to_numpy()

    # 1. Trade-off table: how much fraud can each review budget catch? (validation)
    print("Review budget -> recall, precision, fraud $ caught (validation):")
    for budget in [0.002, 0.005, 0.01, 0.02, 0.05]:
        t = float(np.quantile(p[va], 1 - budget))
        flag = p[va] >= t
        share = amount[va][flag & (y[va] == 1)].sum() / amount[va][y[va] == 1].sum()
        print(f"  {budget:5.1%} of transactions -> recall {recall_score(y[va], flag):.3f} | "
              f"precision {precision_score(y[va], flag, zero_division=0):.3f} | fraud $ caught {share:.1%}")

    # 2. Operating threshold: flag the top ALERT_BUDGET of validation scores
    th = float(np.quantile(p[va], 1 - ALERT_BUDGET))
    print(f"\nOperating point: review {ALERT_BUDGET:.0%} of transactions -> threshold {th:.4f}\n")

    # 3. Score each period at that threshold (test is touched only here)
    summarize("valid", p[va], y[va], amount[va], th)
    summarize("test", p[te], y[te], amount[te], th)
    summarize("stress", p[st], y[st], amount[st], th)

    # 4. SHAP on a sample of test rows
    sample = X[features][te].sample(5000, random_state=0)
    shap.summary_plot(shap.TreeExplainer(model).shap_values(sample), sample, show=False)
    plt.savefig("docs/shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("\nSaved docs/shap_summary.png")

    # 5. Error analysis on test
    fraud = te & (y == 1)
    caught, missed = fraud & (p >= th), fraud & (p < th)
    print(f"\nTest frauds: {fraud.sum()} | caught {caught.sum()} | missed {missed.sum()}")
    cols = ["log_amount", "is_transfer", "hour", "dest_prev_count", "amount_vs_dest_mean"]
    print(X.loc[caught, cols].median().to_frame("caught").join(X.loc[missed, cols].median().to_frame("missed")))

    bundle["threshold"] = th
    joblib.dump(bundle, "models/model.joblib")
    print(f"\nSaved threshold {th:.4f} into models/model.joblib")


if __name__ == "__main__":
    main()
