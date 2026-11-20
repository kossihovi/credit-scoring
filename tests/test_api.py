"""Tests de l'API : validation, encodage et cohérence des réponses."""

import pytest
from fastapi.testclient import TestClient

import api

client = TestClient(api.app)

VALID = {
    "age_years": 40, "amt_credit": 500_000, "amt_annuity": 25_000, "amt_goods_price": 450_000,
    "years_employed": 5, "ext_source_2": 0.5, "ext_source_3": 0.5,
    "name_family_status": "Married", "name_education_type": "Higher education",
    "organization_type": "Business Entity Type 3",
}


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["n_features"] == len(api.FEATURES)


def test_predict_returns_consistent_decision():
    body = client.post("/predict", json=VALID).json()
    assert 0 <= body["probability"] <= 1
    assert len(body["top_factors"]) == 5
    expected = "refuser" if body["probability"] >= body["threshold"] else "accepter"
    assert body["decision"] == expected


def test_minimal_payload_is_accepted():
    r = client.post("/predict", json={"age_years": 30, "amt_credit": 100_000, "amt_annuity": 8_000})
    assert r.status_code == 200


def test_unknown_field_is_rejected_not_ignored():
    """Un champ que le modèle n'utilise pas doit déclencher une erreur, pas être ignoré."""
    r = client.post("/predict", json={**VALID, "amt_income_total": 150_000})
    assert r.status_code == 422


@pytest.mark.parametrize("field,value", [
    ("age_years", 17), ("amt_credit", 0), ("ext_source_1", 1.5), ("years_employed", 61),
    ("organization_type", "Not an organization"), ("name_family_status", "Complicated"),
])
def test_invalid_values_are_rejected(field, value):
    assert client.post("/predict", json={**VALID, field: value}).status_code == 422


def test_no_job_sets_employment_anomaly_flag():
    row = api.build_row(api.Client(**{**VALID, "years_employed": None}))
    assert row.loc[0, "DAYS_EMPLOYED_ANOM"] == 1 and row["DAYS_EMPLOYED"].isna().all()
    row = api.build_row(api.Client(**VALID))
    assert row.loc[0, "DAYS_EMPLOYED_ANOM"] == 0 and row.loc[0, "DAYS_EMPLOYED"] == -5 * 365


def test_better_external_score_lowers_risk():
    low = client.post("/predict", json={**VALID, "ext_source_2": 0.2}).json()["probability"]
    high = client.post("/predict", json={**VALID, "ext_source_2": 0.8}).json()["probability"]
    assert high < low
