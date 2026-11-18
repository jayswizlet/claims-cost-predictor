"""
Requests that are valid JSON but describe an impossible member must be refused
with a 422 rather than scored.

These sit alongside test_api.py, which covers the course's required checks.
"""

import pytest

from conftest import MODEL_PATH

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None

VALID = {
    "age_bucket": "50-59",
    "sex": "F",
    "region": "Northeast",
    "total_claims": 18,
    "specialty_drug_fills": 2,
    "inpatient_claims": 1,
    "has_specialty_drug": True,
}


@pytest.fixture(scope="module")
def client():
    if TestClient is None:
        pytest.skip("pip install -r requirements-dev.txt to get the FastAPI test client")
    if not MODEL_PATH.exists():
        pytest.skip("model.joblib not found — run `python train.py` first")
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_a_realistic_member_is_scored(client):
    assert client.post("/predict", json=VALID).status_code == 200


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"total_claims": 3, "inpatient_claims": 3, "specialty_drug_fills": 3},
         "inpatient and specialty claims never overlap, so together they cannot exceed the total"),
        ({"total_claims": 5, "specialty_drug_fills": 6}, "more specialty fills than claims"),
        ({"total_claims": 5, "inpatient_claims": 6}, "more inpatient claims than claims"),
        ({"has_specialty_drug": False}, "has_specialty_drug contradicts specialty_drug_fills"),
        ({"total_claims": 50_000}, "a claim count far outside anything the model was trained on"),
        ({"medical_claims": 5}, "a field the model does not use"),
    ],
)
def test_impossible_members_are_rejected_with_422(client, change, reason):
    response = client.post("/predict", json={**VALID, **change})
    assert response.status_code == 422, f"sent {reason} and got {response.status_code}"


def test_counts_that_exactly_fill_the_total_are_allowed(client):
    """The sum rule is <=, so a member whose claims are all inpatient or specialty is fine."""
    body = {**VALID, "total_claims": 3, "inpatient_claims": 1, "specialty_drug_fills": 2}
    assert client.post("/predict", json=body).status_code == 200


def test_the_ceiling_itself_is_allowed(client):
    from app.schemas import MAX_CLAIMS

    assert client.post("/predict", json={**VALID, "total_claims": MAX_CLAIMS}).status_code == 200
