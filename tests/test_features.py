import numpy as np
import pandas as pd

from src.features import FEATURES, build_features, features_for_one, update_state


def sample_df():
    return pd.DataFrame({
        "step": [1, 1, 2, 3, 3],
        "type": ["TRANSFER", "CASH_OUT", "TRANSFER", "CASH_OUT", "TRANSFER"],
        "amount": [100.0, 5000.0, 2000.0, 7000.0, 1000.0],
        "nameOrig": ["A", "B", "A", "C", "D"],
        "nameDest": ["X", "Y", "X", "X", "Y"],
        "isFraud": [0, 0, 1, 0, 0],
    })


def test_batch_and_live_features_match():
    """Training features and API features must be identical (no training/serving skew)."""
    df = sample_df()
    X, *_ = build_features(df)
    state = {"dest": {}, "orig": {}}
    for i, t in enumerate(df.to_dict("records")):
        live = features_for_one(t, state)
        update_state(t, state)
        for f in FEATURES:
            assert np.isclose(live[f], X.iloc[i][f]), f"{f} differs on row {i}"


def test_no_future_leakage():
    """Changing the LAST transaction must not change any earlier row's features."""
    df = sample_df()
    X1, *_ = build_features(df)
    df2 = df.copy()
    df2.loc[4, "amount"] = 1e9
    X2, *_ = build_features(df2)
    pd.testing.assert_frame_equal(X1.iloc[:4], X2.iloc[:4])


def test_new_receiver_has_no_history():
    state = {"dest": {}, "orig": {}}
    f = features_for_one({"step": 5, "type": "TRANSFER", "amount": 500.0,
                          "nameOrig": "A", "nameDest": "NEW"}, state)
    assert f["dest_prev_count"] == 0
    assert f["steps_since_dest_last"] == -1
