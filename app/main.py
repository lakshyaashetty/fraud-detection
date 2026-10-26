from typing import Literal

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import features_for_one, update_state

bundle = joblib.load("models/model.joblib")
MODEL, FEATURES, THRESHOLD = bundle["model"], bundle["features"], bundle["threshold"]
STATE = joblib.load("models/state.joblib")
EXPLAINER = shap.TreeExplainer(MODEL)

app = FastAPI(title="Fraud Scoring API")


class Transaction(BaseModel):
    step: int = Field(ge=1, description="Hour since start of data")
    type: Literal["TRANSFER", "CASH_OUT"]
    amount: float = Field(gt=0)
    nameOrig: str
    nameDest: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(t: Transaction):
    txn = t.model_dump()
    x = pd.DataFrame([features_for_one(txn, STATE)])[FEATURES]
    p = float(MODEL.predict_proba(x)[0, 1])
    contrib = EXPLAINER.shap_values(x)[0]
    top = sorted(zip(FEATURES, contrib), key=lambda kv: -abs(kv[1]))[:3]
    update_state(txn, STATE)
    return {
        "fraud_score": round(p, 4),
        "decision": "review" if p >= THRESHOLD else "approve",
        "top_reasons": [{"feature": f, "impact": round(float(v), 3)} for f, v in top],
    }
