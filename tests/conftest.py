"""Fixtures partagées : mini-jeu de données synthétique au format Home Credit."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy"))


@pytest.fixture(scope="session")
def raw_data(tmp_path_factory) -> Path:
    """Écrit des CSV synthétiques (mêmes colonnes que les tables Kaggle utilisées)."""
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("raw")
    n = 1500
    ids = np.arange(100_000, 100_000 + n)

    ext2 = rng.uniform(0, 1, n)
    app = pd.DataFrame({
        "SK_ID_CURR": ids,
        "TARGET": (rng.uniform(0, 1, n) < 0.25 * (1 - ext2)).astype(int),
        "DAYS_BIRTH": -rng.integers(7000, 25000, n),
        "DAYS_EMPLOYED": np.where(rng.uniform(size=n) < 0.15, 365243, -rng.integers(100, 9000, n)),
        "AMT_INCOME_TOTAL": rng.uniform(30_000, 400_000, n),
        "EXT_SOURCE_2": ext2,
        "CODE_GENDER": rng.choice(["F", "M"], n),
        "NAME_EDUCATION_TYPE": rng.choice(["Higher education", "Lower secondary"], n),
    })
    app.to_csv(d / "application_train.csv", index=False)

    nb = 3000
    bureau = pd.DataFrame({
        "SK_ID_CURR": rng.choice(ids[:1000], nb),
        "SK_ID_BUREAU": np.arange(nb),
        "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], nb),
        "DAYS_CREDIT": -rng.integers(10, 3000, nb),
        "CREDIT_DAY_OVERDUE": rng.integers(0, 5, nb),
        "AMT_CREDIT_MAX_OVERDUE": rng.uniform(0, 1000, nb),
        "AMT_CREDIT_SUM": rng.uniform(1000, 100_000, nb),
        "AMT_CREDIT_SUM_DEBT": rng.uniform(0, 50_000, nb),
        "AMT_CREDIT_SUM_OVERDUE": rng.uniform(0, 100, nb),
        "CNT_CREDIT_PROLONG": rng.integers(0, 2, nb),
    })
    bureau.to_csv(d / "bureau.csv", index=False)

    pd.DataFrame({
        "SK_ID_BUREAU": rng.choice(bureau["SK_ID_BUREAU"][:2000], 8000),
        "MONTHS_BALANCE": -rng.integers(0, 60, 8000),
        "STATUS": rng.choice(["C", "X", "0", "1", "2"], 8000),
    }).to_csv(d / "bureau_balance.csv", index=False)

    npv = 3000
    prev_ids = np.arange(npv)
    pd.DataFrame({
        "SK_ID_PREV": prev_ids,
        "SK_ID_CURR": rng.choice(ids[:1200], npv),
        "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused", "Canceled"], npv),
        "AMT_CREDIT": rng.uniform(1000, 100_000, npv),
        "AMT_APPLICATION": rng.uniform(1000, 100_000, npv),
        "DAYS_FIRST_DRAWING": rng.choice([365243, -300, -800], npv),
        "DAYS_FIRST_DUE": -rng.integers(10, 2000, npv),
        "DAYS_LAST_DUE_1ST_VERSION": rng.choice([365243, -100, -500], npv),
        "DAYS_LAST_DUE": -rng.integers(10, 2000, npv),
        "DAYS_TERMINATION": rng.choice([365243, -50, -400], npv),
    }).to_csv(d / "previous_application.csv", index=False)

    pd.DataFrame({
        "SK_ID_PREV": rng.choice(prev_ids, 6000),
        "SK_ID_CURR": rng.choice(ids[:1200], 6000),
        "MONTHS_BALANCE": -rng.integers(1, 80, 6000),
        "SK_DPD": rng.integers(0, 10, 6000),
    }).to_csv(d / "POS_CASH_balance.csv", index=False)

    ni = 6000
    due = -rng.integers(10, 2000, ni)
    pd.DataFrame({
        "SK_ID_PREV": rng.choice(prev_ids, ni),
        "SK_ID_CURR": rng.choice(ids[:1200], ni),
        "DAYS_INSTALMENT": due,
        "DAYS_ENTRY_PAYMENT": due + rng.integers(-5, 20, ni),
        "AMT_INSTALMENT": rng.uniform(100, 5000, ni),
        "AMT_PAYMENT": rng.uniform(100, 5000, ni),
    }).to_csv(d / "installments_payments.csv", index=False)

    pd.DataFrame({
        "SK_ID_PREV": rng.choice(prev_ids, 2000),
        "SK_ID_CURR": rng.choice(ids[:500], 2000),
        "MONTHS_BALANCE": -rng.integers(1, 40, 2000),
        "AMT_BALANCE": rng.uniform(0, 50_000, 2000),
    }).to_csv(d / "credit_card_balance.csv", index=False)
    return d
