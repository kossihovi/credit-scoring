"""Tests du pipeline : features, entraînement, scoring sur données synthétiques."""

import json

import numpy as np
import pandas as pd
import pytest

from credit_scoring import pipeline as pl


def test_clean_application_handles_employment_anomaly():
    app = pd.DataFrame({"DAYS_EMPLOYED": [-1000, 365243], "DAYS_BIRTH": [-14600, -18250]})
    out = pl.clean_application(app)
    assert out["DAYS_EMPLOYED_ANOM"].tolist() == [0, 1]
    assert np.isnan(out.loc[1, "DAYS_EMPLOYED"])
    assert out["AGE_YEARS"].round(1).tolist() == [40.0, 50.0]


def test_build_features_keeps_rows_and_adds_flags(raw_data):
    app = pd.read_csv(raw_data / "application_train.csv")
    tables = pl.load_tables(raw_data)
    df = pl.build_features(app, tables)
    assert len(df) == len(app)
    assert {"HAS_BUREAU", "HAS_BB_STATUS", "BUREAU_DEBT_RATIO", "INST_IS_LATE_MEAN",
            "PREV_PREV_CREDIT_RATIO_MEAN", "PREV_DAYS_LAST_DUE_1ST_VERSION_MAX"} <= set(df.columns)
    # 365243 dans previous_application est bien traité comme manquant
    assert df["PREV_DAYS_LAST_DUE_1ST_VERSION_MAX"].max() < 365243
    # clients sans bureau : compteur à 0 et indicateur à 0
    no_bureau = df[df["HAS_BUREAU"] == 0]
    assert len(no_bureau) > 0 and (no_bureau["BUREAU_N_CREDITS"] == 0).all()


def test_build_features_works_without_history_tables():
    app = pd.DataFrame({"SK_ID_CURR": [1, 2], "DAYS_EMPLOYED": [-10, -20], "DAYS_BIRTH": [-12000, -13000]})
    df = pl.build_features(app, {})
    assert df["HAS_BUREAU"].tolist() == [0, 0]
    assert df["HAS_BB_STATUS"].tolist() == [0, 0]


def test_build_features_rejects_duplicate_ids(raw_data):
    app = pd.read_csv(raw_data / "application_train.csv")
    dup = pd.concat([app, app.iloc[:1]])
    with pytest.raises(ValueError):
        pl.build_features(dup, pl.load_tables(raw_data))


def test_load_tables_filters_on_client_ids(raw_data):
    ids = pd.read_csv(raw_data / "application_train.csv")["SK_ID_CURR"].iloc[:50]
    tables = pl.load_tables(raw_data, client_ids=ids)
    assert tables["bureau"]["SK_ID_CURR"].isin(ids).all()
    assert tables["bureau_balance"]["SK_ID_BUREAU"].isin(tables["bureau"]["SK_ID_BUREAU"]).all()


def test_optimal_threshold_matches_theory_on_calibrated_scores():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.4, 200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    thr = pl.optimal_threshold(y, p, c_fn=10, c_fp=1)
    assert abs(thr - 1 / 11) < 0.03  # seuil théorique c_fp / (c_fp + c_fn)


def test_train_then_score_roundtrip(raw_data, tmp_path):
    metrics = pl.train(raw_data, tmp_path, n_trials=0, n_estimators=40)
    for f in ("model.json", "model_meta.joblib", "metrics.json"):
        assert (tmp_path / f).exists()
    assert 0.5 < metrics["auc_test"] <= 1.0
    assert json.loads((tmp_path / "metrics.json").read_text())["threshold"] == metrics["threshold"]

    new = pd.read_csv(raw_data / "application_train.csv").drop(columns="TARGET").iloc[:20]
    out = pl.score(new, pl.load_tables(raw_data, client_ids=new["SK_ID_CURR"]), tmp_path)
    assert list(out.columns) == ["SK_ID_CURR", "probability", "decision"]
    assert out["probability"].between(0, 1).all()
    assert set(out["decision"]) <= {"accepter", "refuser"}


def test_score_with_unknown_category_and_missing_columns(raw_data, tmp_path):
    pl.train(raw_data, tmp_path, n_estimators=20)
    app = pd.read_csv(raw_data / "application_train.csv").drop(columns="TARGET").iloc[:5].copy()
    app["CODE_GENDER"] = "ZZZ"  # modalité jamais vue
    out = pl.score(app, {}, tmp_path)
    assert out["probability"].notna().all()


def test_score_with_shipped_model_on_minimal_application(tmp_path):
    """Le modèle livré dans deploy/ doit rester utilisable via le pipeline."""
    app = pd.DataFrame({"SK_ID_CURR": [1], "DAYS_EMPLOYED": [-1000], "DAYS_BIRTH": [-14600],
                        "EXT_SOURCE_2": [0.5], "AMT_CREDIT": [500_000], "AMT_ANNUITY": [25_000]})
    out = pl.score(app, {}, pl.Path(__file__).resolve().parents[1] / "deploy")
    assert 0 <= out.loc[0, "probability"] <= 1


def test_train_can_exclude_protected_columns(raw_data, tmp_path):
    pl.train(raw_data, tmp_path, n_estimators=20, drop_cols=["CODE_GENDER"])
    meta = pl.joblib.load(tmp_path / "model_meta.joblib")
    assert "CODE_GENDER" not in meta["features"] and "CODE_GENDER" not in meta["cat_cols"]
