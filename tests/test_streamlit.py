"""Vérifie que l'interface Streamlit envoie exactement ce que l'API attend."""

from pathlib import Path

import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import api

APP = str(Path(__file__).resolve().parents[1] / "deploy" / "streamlit_app.py")


def test_form_payload_matches_api_schema(monkeypatch):
    api_client = TestClient(api.app)
    sent = {}

    def fake_post(url, json=None, timeout=None):
        sent["payload"] = json
        resp = api_client.post("/predict", json=json)
        sent["status"] = resp.status_code
        return resp

    monkeypatch.setattr(requests, "post", fake_post)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button[0].click().run()

    assert not at.exception
    assert sent["status"] == 200
    # champ pour champ : ni oubli, ni champ en trop
    assert set(sent["payload"]) == set(api.Client.model_fields)
    assert len(at.session_state.history) == 1


def test_optional_fields_are_none_only_when_flagged_unknown(monkeypatch):
    api_client = TestClient(api.app)
    sent = {}

    def fake_post(url, json=None, timeout=None):
        sent["payload"] = json
        return api_client.post("/predict", json=json)

    monkeypatch.setattr(requests, "post", fake_post)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.checkbox(key="unk_e1").check()          # score externe inconnu
    at.checkbox(key="unk_bureau_ratio").uncheck()  # historique renseigné
    at.button[0].click().run()

    assert sent["payload"]["ext_source_1"] is None
    assert sent["payload"]["bureau_debt_ratio"] == 0.3
    assert sent["payload"]["own_car_age"] is None  # pas de véhicule coché par défaut
