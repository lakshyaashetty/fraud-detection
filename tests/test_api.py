import uuid

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def txn(**overrides):
    body = {"step": 710, "type": "CASH_OUT", "amount": 9000.0,
            "nameOrig": f"TEST_{uuid.uuid4().hex}", "nameDest": f"TEST_{uuid.uuid4().hex}"}
    body.update(overrides)
    return body


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_score_returns_valid_response():
    r = client.post("/score", json=txn())
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_score"] <= 1
    assert body["decision"] in {"approve", "review"}
    assert len(body["top_reasons"]) == 3


def test_rejects_bad_input():
    assert client.post("/score", json=txn(amount=-5)).status_code == 422
    assert client.post("/score", json=txn(type="PAYMENT")).status_code == 422


def test_big_night_transfer_to_new_receiver_is_reviewed():
    """Regression test: this pattern is fraud ~89% of the time in the data."""
    r = client.post("/score", json=txn(step=700, type="TRANSFER", amount=2_437_189.53))
    assert r.json()["decision"] == "review"


def test_repeat_request_changes_score():
    """Documents a known design limit: the API updates account history on every call."""
    body = txn(step=700, type="TRANSFER", amount=2_437_189.53)
    first = client.post("/score", json=body).json()["fraud_score"]
    second = client.post("/score", json=body).json()["fraud_score"]
    assert second < first
