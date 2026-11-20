"""API de scoring de risque de crédit (FastAPI).

Le modèle XGBoost entraîné sur 258 variables est exposé via un formulaire
réduit aux 16 variables les plus influentes (sélectionnées avec SHAP). Les
autres variables sont transmises au modèle comme valeurs manquantes.

Endpoints
---------
``GET /health``
    État du service, nombre de variables et seuil de décision.
``POST /predict``
    Probabilité de défaut, décision (accepter / refuser) et cinq principaux
    facteurs explicatifs (contributions SHAP du modèle XGBoost).

Lancement ::

    uvicorn api:app --reload

Les artefacts ``model.json`` et ``model_meta.joblib`` (produits par
``python -m credit_scoring.pipeline train``) doivent se trouver à côté de ce fichier.
"""

from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

BASE = Path(__file__).parent
META = joblib.load(BASE / "model_meta.joblib")
BOOSTER = xgb.Booster()
BOOSTER.load_model(str(BASE / "model.json"))

FEATURES: list[str] = META["features"]
CAT_COLS: list[str] = META["cat_cols"]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

#: Valeurs imposées aux indicateurs de présence d'historique : le formulaire ne
#: collecte pas l'historique détaillé, on suppose donc « pas de dossier bureau ».
DEFAULTS = {"HAS_BUREAU": 0, "HAS_BB_STATUS": 0, "BUREAU_N_CREDITS": 0}


class Client(BaseModel):
    """Données d'une demande de crédit.

    Seuls ``age_years``, ``amt_credit`` et ``amt_annuity`` sont obligatoires ;
    toute autre information absente (``None``) est traitée comme manquante par
    le modèle. Les champs inconnus sont refusés (HTTP 422) afin qu'aucune
    information envoyée ne soit ignorée silencieusement.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Obligatoires ---
    age_years: float = Field(..., ge=18, le=100, description="Âge en années")
    amt_credit: float = Field(..., gt=0, description="Montant du crédit demandé")
    amt_annuity: float = Field(..., gt=0, description="Annuité (remboursement par période)")

    # --- Situation du client ---
    amt_goods_price: Optional[float] = Field(None, ge=0, description="Prix du bien financé")
    years_employed: Optional[float] = Field(
        None, ge=0, le=60,
        description="Ancienneté dans l'emploi actuel (années). Vide = sans emploi / retraité.",
    )
    own_car_age: Optional[float] = Field(None, ge=0, description="Âge du véhicule (vide = pas de véhicule)")
    organization_type: Optional[str] = Field(None, description="Type d'organisation de l'employeur")
    name_family_status: Optional[str] = Field(None, description="Situation familiale")
    name_education_type: Optional[str] = Field(None, description="Niveau d'éducation")

    # --- Scores externes (système tiers) ---
    ext_source_1: Optional[float] = Field(None, ge=0, le=1, description="Score externe 1 (0 à 1)")
    ext_source_2: Optional[float] = Field(None, ge=0, le=1, description="Score externe 2 (0 à 1)")
    ext_source_3: Optional[float] = Field(None, ge=0, le=1, description="Score externe 3 (0 à 1)")

    # --- Agrégats d'historique (optionnels) ---
    prev_days_last_due_1st_version_max: Optional[float] = Field(
        None, description="Demandes précédentes : jours (relatifs à la demande) de la dernière échéance prévue, max"
    )
    prev_prev_credit_ratio_mean: Optional[float] = Field(
        None, ge=0, description="Demandes précédentes : ratio moyen crédit accordé / demandé"
    )
    bureau_debt_ratio: Optional[float] = Field(
        None, ge=0, description="Bureau de crédit : dette restante / total emprunté"
    )
    inst_is_late_mean: Optional[float] = Field(
        None, ge=0, le=1, description="Paiements passés : part des échéances payées en retard"
    )

    @field_validator("organization_type", "name_family_status", "name_education_type")
    @classmethod
    def _known_category(cls, value: Optional[str], info) -> Optional[str]:
        """Refuse une modalité absente des catégories vues à l'entraînement."""
        allowed = META["categories"][info.field_name.upper()]
        if value is not None and value not in allowed:
            raise ValueError(f"modalité inconnue {value!r}, attendu l'une de {allowed}")
        return value


class Factor(BaseModel):
    """Contribution d'une variable à la prédiction."""

    feature: str
    value: Optional[str]
    contribution: float  # > 0 : augmente le risque ; < 0 : le diminue (échelle log-odds)


class Prediction(BaseModel):
    """Réponse de ``POST /predict``."""

    probability: float
    threshold: float
    decision: Literal["accepter", "refuser"]
    top_factors: list[Factor]


# Garde-fou : chaque champ du formulaire doit correspondre à une variable du modèle.
# `years_employed` est la seule exception : il est converti en DAYS_EMPLOYED.
FORM_ONLY = {"years_employed"}
_missing = [n.upper() for n in Client.model_fields if n not in FORM_ONLY and n.upper() not in FEATURES]
assert not _missing, f"Champs absents du modèle : {_missing}"


def build_row(c: Client) -> pd.DataFrame:
    """Construit la ligne d'entrée du modèle à partir d'un formulaire client.

    - toutes les variables non renseignées valent NaN ;
    - ``years_employed`` est converti en ``DAYS_EMPLOYED`` (négatif) ; s'il est
      vide, on reproduit le traitement d'entraînement (NaN + ``DAYS_EMPLOYED_ANOM = 1``) ;
    - les variables catégorielles sont encodées avec les modalités de l'entraînement.

    Paramètres
    ----------
    c : Client
        Demande validée.

    Retourne
    --------
    pd.DataFrame
        Une ligne, colonnes dans l'ordre ``FEATURES``.
    """
    row = {f: np.nan for f in FEATURES}
    row.update(DEFAULTS)

    for name, value in c.model_dump().items():
        if name != "years_employed":
            row[name.upper()] = np.nan if value is None else value

    if c.years_employed is None:
        row["DAYS_EMPLOYED"], row["DAYS_EMPLOYED_ANOM"] = np.nan, 1
    else:
        row["DAYS_EMPLOYED"], row["DAYS_EMPLOYED_ANOM"] = -c.years_employed * 365, 0

    X = pd.DataFrame([row], columns=FEATURES)
    X[NUM_COLS] = X[NUM_COLS].astype(float)
    for col in CAT_COLS:  # les modalités ont déjà été validées dans `Client`
        X[col] = pd.Categorical(X[col], categories=META["categories"][col])
    return X


app = FastAPI(title="Credit scoring API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    """Vérifie que le service est up et renvoie la configuration du modèle."""
    return {"status": "ok", "n_features": len(FEATURES), "threshold": META["threshold"]}


@app.post("/predict", response_model=Prediction)
def predict(client: Client) -> Prediction:
    """Évalue une demande de crédit.

    La décision est ``refuser`` si la probabilité de défaut est supérieure ou
    égale au seuil optimisé sur un coût métier (faux négatif = 10 × faux positif).
    Les facteurs renvoyés sont les 5 variables dont la contribution (en valeur
    absolue) à la prédiction est la plus forte.
    """
    X = build_row(client)
    dm = xgb.DMatrix(X, enable_categorical=True)
    proba = float(BOOSTER.predict(dm)[0])

    contrib = BOOSTER.predict(dm, pred_contribs=True)[0][:-1]  # dernière colonne = biais
    top = np.argsort(-np.abs(contrib))[:5]

    factors = []
    for i in top:
        v = X.iloc[0, i]
        factors.append(Factor(
            feature=FEATURES[i],
            value=None if pd.isna(v) else str(v),
            contribution=round(float(contrib[i]), 4),
        ))
    thr = META["threshold"]
    return Prediction(
        probability=round(proba, 4),
        threshold=thr,
        decision="refuser" if proba >= thr else "accepter",
        top_factors=factors,
    )
