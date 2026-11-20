"""Tests des analyses (variable sensible, validation croisée, équité)."""

import numpy as np
import pandas as pd

from credit_scoring import analysis as an
from credit_scoring import pipeline as pl


def test_fairness_table_computes_group_rates():
    # groupe A : 4 clients ; 1 défaut (refusé) ; 3 bons dont 1 refusé à tort
    # groupe B : 4 clients ; 2 défauts (1 refusé) ; 2 bons, aucun refusé
    y = pd.Series([1, 0, 0, 0, 1, 1, 0, 0])
    proba = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
    group = pd.Series(list("AAAABBBB"))
    t = an.fairness_table(y, proba, group, threshold=0.5, min_group_size=1)
    assert t.loc["A", "taux_refus"] == 0.5 and t.loc["B", "taux_refus"] == 0.25
    assert t.loc["A", "bons_clients_refuses"] == 1 / 3 and t.loc["B", "bons_clients_refuses"] == 0
    assert t.loc["A", "defauts_detectes"] == 1.0 and t.loc["B", "defauts_detectes"] == 0.5
    gaps = an.fairness_gaps(t)
    assert round(gaps["ecart_taux_refus"], 1) == 25.0


def test_fairness_table_ignores_tiny_groups():
    y = pd.Series([0, 1, 0, 1])
    t = an.fairness_table(y, np.array([0.1, 0.9, 0.1, 0.9]), pd.Series(list("AABB")),
                          0.5, min_group_size=3)
    assert t.empty


def test_age_groups_keep_natural_order():
    g = an.age_groups(pd.Series([25.0, 35.0, 50.0, 70.0]))
    assert list(g.cat.categories) == ["< 30 ans", "30-44 ans", "45-59 ans", "60 ans et +"]
    assert g.tolist() == ["< 30 ans", "30-44 ans", "45-59 ans", "60 ans et +"]


def test_summarize_cv_interval_contains_mean_and_counts_wins():
    cv = pd.DataFrame({"auc_xgboost": [0.79, 0.78, 0.80, 0.79, 0.78],
                       "auc_logistique": [0.77, 0.77, 0.78, 0.78, 0.77]})
    cv["écart"] = cv["auc_xgboost"] - cv["auc_logistique"]
    s = an.summarize_cv(cv)
    assert s["ic95_bas"] < s["ecart_moyen"] < s["ic95_haut"]
    assert s["xgboost_meilleur_dans"] == 5 and s["ic95_bas"] > 0


def test_cross_validation_and_removal_on_synthetic_data(raw_data):
    X_train, X_test, y_train, y_test, cat_cols = pl.prepare_train_test(raw_data)
    cv = an.cross_validate_models(X_train, y_train, cat_cols, n_folds=3, n_estimators=15)
    assert len(cv) == 3 and cv[["auc_xgboost", "auc_logistique"]].gt(0.3).all().all()

    comparison, fitted = an.compare_without_column(X_train, y_train, X_test, y_test, n_estimators=15)
    assert list(comparison.index) == ["avec CODE_GENDER", "sans CODE_GENDER"]
    assert set(fitted) == {"avec", "sans"}


def test_full_report_is_written(raw_data, tmp_path):
    report = an.run_analysis(raw_data, tmp_path, n_folds=3, n_estimators=15,
                             cv_estimators=15, min_group_size=20)
    for title in ("1. EFFET DU RETRAIT", "2. VALIDATION CROISÉE", "3. ÉQUITÉ"):
        assert title in report
    assert (tmp_path / "analysis.txt").exists() and (tmp_path / "cross_validation.csv").exists()


def test_report_without_cv(raw_data, tmp_path):
    report = an.run_analysis(raw_data, tmp_path, n_estimators=15, skip_cv=True, min_group_size=20)
    assert "2. VALIDATION CROISÉE" not in report and "3. ÉQUITÉ" in report
