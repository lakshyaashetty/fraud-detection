import math

import numpy as np
import pandas as pd

FEATURES = [
    "log_amount", "is_transfer", "hour", "is_night", "round_amount",
    "dest_prev_count", "dest_prev_mean", "amount_vs_dest_mean",
    "steps_since_dest_last", "orig_prev_count",
]


def build_features(df: pd.DataFrame):
    """Vectorized features for training. Returns X, y, step, amount."""
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    X = pd.DataFrame(index=df.index)
    X["log_amount"] = np.log1p(df["amount"])
    X["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    X["hour"] = df["step"] % 24
    X["is_night"] = (X["hour"] <= 5).astype(int)
    X["round_amount"] = (df["amount"] % 1000 == 0).astype(int)

    g = df.groupby("nameDest", sort=False)
    X["dest_prev_count"] = g.cumcount()
    prev_sum = g["amount"].cumsum() - df["amount"]
    X["dest_prev_mean"] = (prev_sum / X["dest_prev_count"]).fillna(0.0)
    X["amount_vs_dest_mean"] = df["amount"] / (X["dest_prev_mean"] + 1)
    X["steps_since_dest_last"] = g["step"].diff().fillna(-1)
    X["orig_prev_count"] = df.groupby("nameOrig", sort=False).cumcount()
    return X[FEATURES], df["isFraud"], df["step"], df["amount"]


def features_for_one(t: dict, state: dict) -> dict:
    """Same features for ONE live transaction, from running account state."""
    d = state["dest"].get(t["nameDest"], {"count": 0, "sum": 0.0, "last_step": None})
    prev_mean = d["sum"] / d["count"] if d["count"] else 0.0
    hour = t["step"] % 24
    return {
        "log_amount": math.log1p(t["amount"]),
        "is_transfer": int(t["type"] == "TRANSFER"),
        "hour": hour,
        "is_night": int(hour <= 5),
        "round_amount": int(t["amount"] % 1000 == 0),
        "dest_prev_count": d["count"],
        "dest_prev_mean": prev_mean,
        "amount_vs_dest_mean": t["amount"] / (prev_mean + 1),
        "steps_since_dest_last": (t["step"] - d["last_step"]) if d["last_step"] is not None else -1,
        "orig_prev_count": state["orig"].get(t["nameOrig"], 0),
    }


def update_state(t: dict, state: dict) -> None:
    d = state["dest"].setdefault(t["nameDest"], {"count": 0, "sum": 0.0, "last_step": None})
    d["count"] += 1
    d["sum"] += t["amount"]
    d["last_step"] = t["step"]
    state["orig"][t["nameOrig"]] = state["orig"].get(t["nameOrig"], 0) + 1
