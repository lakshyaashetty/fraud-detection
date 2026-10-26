import joblib

from src.data import load

RECENT_STEPS = 72  # keep accounts active in the last 3 days of data


def main():
    df = load()
    cutoff = df["step"].max() - RECENT_STEPS

    dest = df.groupby("nameDest").agg(count=("amount", "size"), sum=("amount", "sum"),
                                      last_step=("step", "max"))
    dest = dest[dest["last_step"] > cutoff]

    orig = df.groupby("nameOrig").agg(count=("amount", "size"), last_step=("step", "max"))
    orig = orig[orig["last_step"] > cutoff]

    state = {
        "dest": {k: {"count": int(v["count"]), "sum": float(v["sum"]), "last_step": int(v["last_step"])}
                 for k, v in dest.to_dict("index").items()},
        "orig": {k: int(v) for k, v in orig["count"].items()},
    }
    joblib.dump(state, "models/state.joblib", compress=3)
    print(f"Saved state for {len(state['dest']):,} receivers and {len(state['orig']):,} senders")


if __name__ == "__main__":
    main()
