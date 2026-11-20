"""Pipeline de scoring de risque de crédit (Home Credit Default Risk).

Ce module reprend, sous forme de fonctions réutilisables, le travail du notebook
``notebooks/home_credit.ipynb`` :

1. **Chargement** des tables brutes (demandes + historique de crédit) ;
2. **Feature engineering** : nettoyage de la table principale et agrégation des
   tables d'historique au niveau client (``SK_ID_CURR``) ;
3. **Entraînement** d'un modèle XGBoost (optimisation Optuna optionnelle) ;
4. **Choix du seuil de décision** par minimisation d'un coût métier
   (un défaut non détecté coûte ``COST_FN`` fois plus qu'un bon client refusé) ;
5. **Sauvegarde** des artefacts consommés par l'API (``model.json`` et
   ``model_meta.joblib``) ;
6. **Scoring** de nouvelles demandes avec le modèle sauvegardé.

Les deux usages se font en ligne de commande ::

    # Entraînement (les CSV Kaggle sont dans ./data)
    python -m credit_scoring.pipeline train --data-dir data --out-dir deploy

    # Scoring de nouvelles demandes
    python -m credit_scoring.pipeline predict \\
        --app data/application_test.csv --data-dir data \\
        --model-dir deploy --output predictions.csv

ou depuis Python ::

    from credit_scoring.pipeline import load_tables, build_features, score
    app = pd.read_csv("data/application_test.csv")
    tables = load_tables("data", client_ids=app["SK_ID_CURR"])
    predictions = score(app, tables, model_dir="deploy")
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #

SEED = 42
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"

#: Valeur sentinelle de ``DAYS_EMPLOYED`` (≈ 1000 ans) : retraités / sans emploi.
DAYS_EMPLOYED_ANOMALY = 365243

#: Coût d'un faux négatif (défaut accepté) et d'un faux positif (bon client refusé).
COST_FN, COST_FP = 10, 1

#: Meilleurs hyperparamètres trouvés par Optuna dans le notebook (15 essais).
BEST_PARAMS = {
    "max_depth": 6,
    "min_child_weight": 17.39807471129172,
    "subsample": 0.9878338511058234,
    "colsample_bytree": 0.6875664116805573,
    "reg_lambda": 39.46212980759094,
    "reg_alpha": 3.795853142670642,
}

#: Tables d'historique (nom de fichier sans extension).
AUX_TABLES = (
    "bureau",
    "bureau_balance",
    "previous_application",
    "POS_CASH_balance",
    "installments_payments",
    "credit_card_balance",
)

#: Colonnes de ``previous_application`` où 365243 signifie « valeur manquante ».
PREV_SENTINEL_COLS = (
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
)

#: Agrégations appliquées à la table ``bureau``.
BUREAU_NUM_AGGS = {
    "DAYS_CREDIT": ["min", "max"],
    "CREDIT_DAY_OVERDUE": ["max"],
    "AMT_CREDIT_MAX_OVERDUE": ["max"],
    "AMT_CREDIT_SUM": ["sum", "max"],
    "AMT_CREDIT_SUM_DEBT": ["sum", "max"],
    "AMT_CREDIT_SUM_OVERDUE": ["sum"],
    "CNT_CREDIT_PROLONG": ["sum"],
}


# --------------------------------------------------------------------------- #
# 1. Chargement des données
# --------------------------------------------------------------------------- #

def _read_csv(
    path: Path,
    key: Optional[str] = None,
    ids: Optional[Iterable] = None,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Lit un CSV, éventuellement en ne gardant que les lignes dont ``key`` est dans ``ids``.

    Le filtrage se fait par blocs pour éviter de charger en mémoire les grosses
    tables (``installments_payments`` dépasse 13 millions de lignes) quand on ne
    score que quelques clients.

    Paramètres
    ----------
    path : Path
        Chemin du fichier CSV.
    key : str, optionnel
        Colonne utilisée pour le filtrage.
    ids : itérable, optionnel
        Valeurs de ``key`` à conserver. Si ``None``, tout le fichier est lu.
    chunksize : int
        Nombre de lignes lues par bloc.

    Retourne
    --------
    pd.DataFrame
        Lignes lues (ou filtrées).
    """
    if key is None or ids is None:
        return pd.read_csv(path)
    wanted = pd.Index(pd.unique(np.asarray(ids)))
    parts = [chunk[chunk[key].isin(wanted)] for chunk in pd.read_csv(path, chunksize=chunksize)]
    return pd.concat(parts, ignore_index=True)


def load_tables(
    data_dir: str | Path,
    client_ids: Optional[Iterable] = None,
    names: Sequence[str] = AUX_TABLES,
) -> dict[str, pd.DataFrame]:
    """Charge les tables d'historique de crédit.

    Une table absente du dossier est ignorée (avec un avertissement) : les
    variables correspondantes seront simplement manquantes, ce que le modèle
    sait gérer.

    Paramètres
    ----------
    data_dir : str ou Path
        Dossier contenant les CSV (``bureau.csv``, ``previous_application.csv``…).
    client_ids : itérable, optionnel
        Identifiants ``SK_ID_CURR`` à conserver. Utile pour scorer de nouveaux
        clients sans charger tout l'historique en mémoire. ``None`` = tout charger.
    names : séquence de str
        Tables à charger (par défaut, les six tables d'historique).

    Retourne
    --------
    dict[str, pd.DataFrame]
        Dictionnaire ``{nom_de_table: DataFrame}``.
    """
    data_dir = Path(data_dir)
    tables: dict[str, pd.DataFrame] = {}
    for name in names:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            logger.warning("Table absente, ignorée : %s", path)
            continue
        if name == "bureau_balance":
            # Pas de SK_ID_CURR dans cette table : on filtre via les crédits du bureau.
            bureau = tables.get("bureau")
            if client_ids is not None and bureau is not None:
                tables[name] = _read_csv(path, "SK_ID_BUREAU", bureau["SK_ID_BUREAU"])
            else:
                tables[name] = _read_csv(path)
        else:
            tables[name] = _read_csv(path, ID_COL, client_ids)
        logger.info("%s : %s lignes", name, f"{len(tables[name]):,}")
    return tables


# --------------------------------------------------------------------------- #
# 2. Feature engineering
# --------------------------------------------------------------------------- #

def clean_application(app: pd.DataFrame) -> pd.DataFrame:
    """Nettoie la table principale des demandes de crédit.

    - ``DAYS_EMPLOYED == 365243`` (retraités, sans emploi) est remplacé par NaN
      et signalé par l'indicateur ``DAYS_EMPLOYED_ANOM`` (l'absence d'emploi est
      en soi informative) ;
    - ``AGE_YEARS`` est dérivé de ``DAYS_BIRTH``.

    Paramètres
    ----------
    app : pd.DataFrame
        Table ``application_train`` / ``application_test`` (avec ou sans ``TARGET``).

    Retourne
    --------
    pd.DataFrame
        Copie nettoyée de ``app``.
    """
    df = app.copy()
    anomaly = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY
    df["DAYS_EMPLOYED_ANOM"] = anomaly.astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].mask(anomaly)
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    return df


def summarize(table: pd.DataFrame, prefix: str, stats: Sequence[str] = ("mean", "max")) -> pd.DataFrame:
    """Agrège toutes les colonnes numériques d'une table au niveau client.

    Paramètres
    ----------
    table : pd.DataFrame
        Table contenant ``SK_ID_CURR`` (ex. ``previous_application``).
    prefix : str
        Préfixe des colonnes produites (ex. ``"PREV"`` → ``PREV_AMT_CREDIT_MEAN``).
    stats : séquence de str
        Statistiques à calculer pour chaque colonne.

    Retourne
    --------
    pd.DataFrame
        Une ligne par client (index ``SK_ID_CURR``), colonnes en majuscules,
        plus ``<PREFIX>_N_ROWS`` (nombre de lignes de la table par client).
    """
    num = table.select_dtypes(include="number").drop(columns=["SK_ID_PREV"], errors="ignore")
    grouped = num.groupby(ID_COL)
    agg = grouped.agg(list(stats))
    agg.columns = [f"{prefix}_{col}_{stat}".upper() for col, stat in agg.columns]
    agg[f"{prefix}_N_ROWS".upper()] = grouped.size()
    return agg


def aggregate_bureau(bureau: pd.DataFrame, bureau_balance: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Construit les variables « bureau de crédit » au niveau client.

    Combine :

    - des agrégats sur les crédits déclarés au bureau (montants, retards, dette
      restante, nombre de crédits actifs, ratio dette / total emprunté) ;
    - si ``bureau_balance`` est fourni, des agrégats sur l'historique mensuel de
      statut (part de mois en retard, pire statut observé…).

    Paramètres
    ----------
    bureau : pd.DataFrame
        Table ``bureau``.
    bureau_balance : pd.DataFrame, optionnel
        Table ``bureau_balance`` (statut mensuel de chaque crédit du bureau).

    Retourne
    --------
    pd.DataFrame
        Une ligne par client (index ``SK_ID_CURR``), colonnes ``BUREAU_*`` et ``BB_*``.
    """
    by_client = bureau.groupby(ID_COL)
    agg = by_client.agg(BUREAU_NUM_AGGS)
    agg.columns = ["BUREAU_" + "_".join(col).upper() for col in agg.columns]
    agg["BUREAU_N_CREDITS"] = by_client.size()

    active = bureau[bureau["CREDIT_ACTIVE"] == "Active"].groupby(ID_COL).agg(
        BUREAU_ACTIVE_N=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
    )
    agg = agg.join(active, how="left")
    agg[["BUREAU_ACTIVE_N", "BUREAU_ACTIVE_DEBT_SUM"]] = agg[
        ["BUREAU_ACTIVE_N", "BUREAU_ACTIVE_DEBT_SUM"]
    ].fillna(0)
    agg["BUREAU_DEBT_RATIO"] = (
        agg["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"] / agg["BUREAU_AMT_CREDIT_SUM_SUM"].replace(0, np.nan)
    )

    if bureau_balance is not None:
        bb = bureau_balance.copy()
        bb["IS_LATE"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
        bb["STATUS_NUM"] = pd.to_numeric(bb["STATUS"], errors="coerce")  # 'C' et 'X' -> NaN
        per_credit = bb.groupby("SK_ID_BUREAU").agg(
            BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            BB_LATE_COUNT=("IS_LATE", "sum"),
            BB_LATE_RATE=("IS_LATE", "mean"),
            BB_WORST_STATUS=("STATUS_NUM", "max"),
        )
        link = bureau.set_index("SK_ID_BUREAU")[ID_COL]
        per_credit = per_credit.join(link, how="inner")
        per_client = per_credit.groupby(ID_COL).agg(
            BB_LATE_COUNT_SUM=("BB_LATE_COUNT", "sum"),
            BB_LATE_RATE_MEAN=("BB_LATE_RATE", "mean"),
            BB_LATE_RATE_MAX=("BB_LATE_RATE", "max"),
            BB_WORST_STATUS_MAX=("BB_WORST_STATUS", "max"),
            BB_MONTHS_TOTAL=("BB_MONTHS_COUNT", "sum"),
        )
        agg = agg.join(per_client, how="left")
    return agg


def aggregate_previous(prev: pd.DataFrame) -> pd.DataFrame:
    """Agrège les demandes de crédit précédentes (préfixe ``PREV_``).

    Ajoute trois variables dérivées avant agrégation : ``PREV_APPROVED``,
    ``PREV_REFUSED`` (statut de la demande) et ``PREV_CREDIT_RATIO``
    (crédit accordé / montant demandé).
    """
    prev = prev.copy()
    for col in PREV_SENTINEL_COLS:
        if col in prev:
            prev[col] = prev[col].replace(DAYS_EMPLOYED_ANOMALY, np.nan)
    prev["PREV_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["PREV_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    prev["PREV_CREDIT_RATIO"] = prev["AMT_CREDIT"] / prev["AMT_APPLICATION"].replace(0, np.nan)
    return summarize(prev, "PREV")


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """Agrège l'historique des échéances payées (préfixe ``INST_``).

    Variables dérivées : jours de retard (``DAYS_LATE``), indicateur de retard
    (``IS_LATE``) et ratio payé / dû (``PAY_RATIO``).
    """
    inst = inst.copy()
    inst["DAYS_LATE"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    inst["IS_LATE"] = (inst["DAYS_LATE"] > 0).astype(int)
    inst["PAY_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"].replace(0, np.nan)
    return summarize(inst, "INST")


def _add_features(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Joint ``feats`` (indexé par ``SK_ID_CURR``) à ``df`` sans dupliquer de colonnes."""
    return df.drop(columns=feats.columns, errors="ignore").join(feats, on=ID_COL, how="left")


def build_features(app: pd.DataFrame, tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Construit la table de variables complète à partir des données brutes.

    La fonction est utilisée à l'identique pour l'entraînement et pour le
    scoring de nouvelles demandes. Les tables absentes de ``tables`` sont
    tolérées : les clients concernés auront des valeurs manquantes (et
    ``HAS_BUREAU = 0``), comme les clients sans historique dans les données
    d'origine.

    Paramètres
    ----------
    app : pd.DataFrame
        Demandes de crédit (une ligne par ``SK_ID_CURR``).
    tables : mapping
        Tables d'historique, telles que renvoyées par :func:`load_tables`.

    Retourne
    --------
    pd.DataFrame
        Une ligne par demande (même ordre que ``app``), avec ``SK_ID_CURR``
        et, s'il était présent, ``TARGET``.

    Lève
    ----
    ValueError
        Si ``SK_ID_CURR`` n'est pas unique dans ``app``, ou si une jointure duplique des lignes.
    """
    if not app[ID_COL].is_unique:
        raise ValueError("SK_ID_CURR doit être unique dans `app` (une ligne par demande).")
    df = clean_application(app)

    if "bureau" in tables:
        df = _add_features(df, aggregate_bureau(tables["bureau"], tables.get("bureau_balance")))
    if "previous_application" in tables:
        df = _add_features(df, aggregate_previous(tables["previous_application"]))
    if "POS_CASH_balance" in tables:
        df = _add_features(df, summarize(tables["POS_CASH_balance"], "POS"))
    if "installments_payments" in tables:
        df = _add_features(df, aggregate_installments(tables["installments_payments"]))
    if "credit_card_balance" in tables:
        df = _add_features(df, summarize(tables["credit_card_balance"], "CC"))

    # Indicateurs de présence : « pas d'historique » est une information en soi.
    if "BUREAU_N_CREDITS" not in df:
        df["BUREAU_N_CREDITS"] = np.nan
    df["HAS_BUREAU"] = df["BUREAU_N_CREDITS"].notna().astype(int)
    df["BUREAU_N_CREDITS"] = df["BUREAU_N_CREDITS"].fillna(0)
    if "BB_WORST_STATUS_MAX" not in df:
        df["BB_WORST_STATUS_MAX"] = np.nan
    df["HAS_BB_STATUS"] = df["BB_WORST_STATUS_MAX"].notna().astype(int)

    if len(df) != len(app):  # garde-fou : une jointure ne doit jamais ajouter de lignes
        raise ValueError("Une jointure a dupliqué des lignes.")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """Sépare la matrice de variables ``X`` de la cible ``y``.

    Retire ``TARGET``, ``SK_ID_CURR`` (identifiant) et ``DAYS_BIRTH`` (remplacé
    par ``AGE_YEARS``). Si ``TARGET`` est absent (nouvelles données), ``y`` vaut ``None``.
    """
    y = df[TARGET] if TARGET in df else None
    X = df.drop(columns=[c for c in (TARGET, ID_COL, "DAYS_BIRTH") if c in df])
    return X, y


# --------------------------------------------------------------------------- #
# 3. Encodage des variables catégorielles
# --------------------------------------------------------------------------- #

def infer_cat_cols(X: pd.DataFrame) -> list[str]:
    """Liste les colonnes catégorielles (toute colonne non numérique)."""
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def to_categorical(values: pd.Series, categories: Sequence) -> pd.Categorical:
    """Encode ``values`` avec des catégories fixées ; toute modalité inconnue devient NaN."""
    known = values.where(values.isin(categories))
    return pd.Categorical(known, categories=list(categories))


def align_to_meta(X: pd.DataFrame, meta: Mapping) -> pd.DataFrame:
    """Met ``X`` exactement au format attendu par le modèle sauvegardé.

    - réordonne les colonnes selon ``meta["features"]`` (colonnes absentes → NaN) ;
    - convertit les colonnes numériques en ``float`` ;
    - encode les colonnes catégorielles avec les catégories vues à l'entraînement
      (une modalité inconnue devient NaN au lieu de faire échouer la prédiction).

    Paramètres
    ----------
    X : pd.DataFrame
        Variables brutes (sortie de :func:`build_features` sans ``TARGET`` ni ID).
    meta : mapping
        Métadonnées du modèle (``features``, ``cat_cols``, ``categories``).

    Retourne
    --------
    pd.DataFrame
        Matrice prête pour ``xgboost.DMatrix(..., enable_categorical=True)``.
    """
    features, cat_cols = list(meta["features"]), list(meta["cat_cols"])
    num_cols = [c for c in features if c not in cat_cols]
    out = X.reindex(columns=features)
    out[num_cols] = out[num_cols].astype(float)
    for col in cat_cols:
        out[col] = to_categorical(out[col], meta["categories"][col])
    return out


# --------------------------------------------------------------------------- #
# 4. Entraînement, seuil et évaluation
# --------------------------------------------------------------------------- #

def tune_hyperparameters(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 15,
    timeout_minutes: Optional[float] = 45,
) -> dict:
    """Optimise les hyperparamètres XGBoost par optimisation bayésienne (Optuna, TPE).

    Chaque essai entraîne un modèle avec arrêt précoce et est évalué par l'AUC
    sur le jeu de validation.

    Paramètres
    ----------
    X_fit, y_fit : jeu d'apprentissage.
    X_val, y_val : jeu de validation (arrêt précoce et score de l'essai).
    n_trials : int
        Nombre maximal d'essais.
    timeout_minutes : float, optionnel
        Durée maximale de l'optimisation.

    Retourne
    --------
    dict
        Meilleurs hyperparamètres (même clés que :data:`BEST_PARAMS`).
    """
    import optuna  # import local : dépendance nécessaire seulement pour le tuning

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_weight=trial.suggest_float("min_child_weight", 10, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 0.8),
            reg_lambda=trial.suggest_float("reg_lambda", 1, 50, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        )
        model = _make_classifier(n_estimators=1000, **params)
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        trial.set_user_attr("best_iter", model.best_iteration)
        return model.best_score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    timeout = None if timeout_minutes is None else timeout_minutes * 60
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    logger.info("Optuna : meilleur AUC validation = %.5f", study.best_value)
    return study.best_params


def _make_classifier(
    n_estimators: int = 3000, early_stopping_rounds: Optional[int] = 100, **params
) -> xgb.XGBClassifier:
    """Crée un ``XGBClassifier`` avec la configuration du projet.

    ``early_stopping_rounds=None`` désactive l'arrêt précoce (le modèle n'a alors
    pas besoin de jeu de validation : utile en validation croisée).
    """
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=-1,
        random_state=SEED,
        **params,
    )


def total_cost(y: np.ndarray, p: np.ndarray, threshold: float, c_fn: float = COST_FN, c_fp: float = COST_FP) -> float:
    """Coût total d'une politique « refuser si probabilité >= seuil ».

    Paramètres
    ----------
    y : array
        Cibles réelles (1 = défaut).
    p : array
        Probabilités de défaut prédites.
    threshold : float
        Seuil de refus.
    c_fn, c_fp : float
        Coût d'un faux négatif (défaut accepté) et d'un faux positif (bon client refusé).
    """
    y = np.asarray(y)
    refused = np.asarray(p) >= threshold
    false_neg = ((~refused) & (y == 1)).sum()
    false_pos = (refused & (y == 0)).sum()
    return float(c_fn * false_neg + c_fp * false_pos)


def optimal_threshold(
    y: np.ndarray,
    p: np.ndarray,
    c_fn: float = COST_FN,
    c_fp: float = COST_FP,
    grid: Optional[np.ndarray] = None,
) -> float:
    """Cherche le seuil de refus qui minimise le coût total sur une grille.

    À titre de repère, le seuil théorique pour un modèle parfaitement calibré
    est ``c_fp / (c_fp + c_fn)``.
    """
    grid = np.linspace(0.01, 0.60, 120) if grid is None else grid
    costs = np.array([total_cost(y, p, t, c_fn, c_fp) for t in grid])
    return float(grid[costs.argmin()])


def evaluate(
    model: xgb.XGBClassifier,
    sets: Mapping[str, tuple[pd.DataFrame, pd.Series]],
    threshold: float,
    c_fn: float = COST_FN,
    c_fp: float = COST_FP,
) -> dict:
    """Calcule les métriques de performance et de coût.

    Paramètres
    ----------
    model : xgb.XGBClassifier
        Modèle entraîné.
    sets : mapping
        ``{"train": (X, y), "val": (X, y), "test": (X, y)}``. Les métriques de
        seuil et de coût sont calculées sur ``"test"``.
    threshold : float
        Seuil de refus.

    Retourne
    --------
    dict
        AUC par jeu, Brier / AUC-PR sur le test, taux de refus, rappel, précision
        et coût par client (modèle vs « tout accepter » vs « tout refuser »).
    """
    metrics: dict = {}
    probas = {name: model.predict_proba(X)[:, 1] for name, (X, _) in sets.items()}
    for name, (_, y) in sets.items():
        metrics[f"auc_{name}"] = float(roc_auc_score(y, probas[name]))

    y_test = np.asarray(sets["test"][1])
    p_test = probas["test"]
    refused = p_test >= threshold
    tn, fp, fn, tp = confusion_matrix(y_test, refused).ravel()
    n = len(y_test)
    metrics.update(
        brier_test=float(brier_score_loss(y_test, p_test)),
        auc_pr_test=float(average_precision_score(y_test, p_test)),
        threshold=float(threshold),
        refusal_rate=float(refused.mean()),
        recall=float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        precision=float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        cost_per_client_model=total_cost(y_test, p_test, threshold, c_fn, c_fp) / n,
        cost_per_client_accept_all=float(c_fn * y_test.sum() / n),
        cost_per_client_refuse_all=float(c_fp * (y_test == 0).sum() / n),
    )
    return metrics


def prepare_train_test(
    data_dir: str | Path, drop_cols: Sequence[str] = ()
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Charge les données brutes et les découpe en jeux d'apprentissage et de test.

    Découpage 80 / 20, stratifié sur la cible, graine fixe (donc reproductible).
    Les modalités des variables catégorielles sont figées sur le jeu d'apprentissage.

    Paramètres
    ----------
    data_dir : str ou Path
        Dossier contenant ``application_train.csv`` et les tables d'historique.
    drop_cols : séquence de str
        Variables à exclure.

    Retourne
    --------
    tuple
        ``(X_train, X_test, y_train, y_test, cat_cols)``.
    """
    data_dir = Path(data_dir)
    app = pd.read_csv(data_dir / "application_train.csv")
    logger.info("Demandes : %s", app.shape)
    tables = load_tables(data_dir)
    X, y = split_features_target(build_features(app, tables))
    X = X.drop(columns=list(drop_cols), errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_test = X_train.copy(), X_test.copy()
    cat_cols = infer_cat_cols(X_train)
    for col in cat_cols:  # les catégories sont figées sur le jeu d'entraînement
        X_train[col] = X_train[col].astype("category")
        X_test[col] = to_categorical(X_test[col], X_train[col].cat.categories)
    return X_train, X_test, y_train, y_test, cat_cols


def split_fit_val(X_train: pd.DataFrame, y_train: pd.Series):
    """Découpe l'apprentissage en ``fit`` (85 %) et ``validation`` (15 %), stratifié.

    La validation sert à l'arrêt précoce et au choix du seuil, jamais à l'évaluation finale.

    Retourne
    --------
    tuple
        ``(X_fit, X_val, y_fit, y_val)``.
    """
    return train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[dict] = None,
    n_estimators: int = 3000,
    c_fn: float = COST_FN,
    c_fp: float = COST_FP,
    verbose: int | bool = 100,
):
    """Entraîne un XGBoost avec arrêt précoce et choisit le seuil de refus.

    Paramètres
    ----------
    X_train, y_train : jeu d'apprentissage (découpé en interne par :func:`split_fit_val`).
    params : dict, optionnel
        Hyperparamètres (défaut : :data:`BEST_PARAMS`).
    n_estimators : int
        Nombre maximal d'arbres.
    c_fn, c_fp : float
        Coûts d'un faux négatif / faux positif pour le seuil.
    verbose : int ou bool
        Affiche l'AUC de validation tous les ``verbose`` arbres (``False`` : silencieux).

    Retourne
    --------
    tuple
        ``(model, threshold, (X_fit, y_fit, X_val, y_val))``.
    """
    X_fit, X_val, y_fit, y_val = split_fit_val(X_train, y_train)
    params = dict(BEST_PARAMS if params is None else params)
    model = _make_classifier(n_estimators=n_estimators, **params)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=verbose)
    logger.info("Arbres retenus : %s", model.best_iteration)
    threshold = optimal_threshold(y_val, model.predict_proba(X_val)[:, 1], c_fn, c_fp)
    return model, threshold, (X_fit, y_fit, X_val, y_val)


def train(
    data_dir: str | Path,
    out_dir: str | Path = "deploy",
    n_trials: int = 0,
    timeout_minutes: Optional[float] = 45,
    params: Optional[dict] = None,
    n_estimators: int = 3000,
    c_fn: float = COST_FN,
    c_fp: float = COST_FP,
    drop_cols: Sequence[str] = (),
) -> dict:
    """Entraîne le modèle de bout en bout et sauvegarde les artefacts de déploiement.

    Étapes : chargement → features → découpage train/test (80/20, stratifié) →
    découpage fit/validation (85/15) → (optionnel) Optuna → entraînement avec
    arrêt précoce → choix du seuil sur la validation → évaluation sur le test →
    sauvegarde.

    Fichiers écrits dans ``out_dir`` :

    - ``model.json`` : modèle XGBoost ;
    - ``model_meta.joblib`` : variables, colonnes catégorielles et leurs
      modalités, seuil, coûts (lus par ``deploy/api.py``), plus métriques et
      hyperparamètres ;
    - ``metrics.json`` : métriques lisibles.

    Paramètres
    ----------
    data_dir : str ou Path
        Dossier contenant ``application_train.csv`` et les tables d'historique.
    out_dir : str ou Path
        Dossier de sortie des artefacts.
    n_trials : int
        Nombre d'essais Optuna. ``0`` (défaut) : utilise ``params`` ou
        :data:`BEST_PARAMS` sans optimisation.
    timeout_minutes : float, optionnel
        Durée maximale d'Optuna.
    params : dict, optionnel
        Hyperparamètres à utiliser quand ``n_trials == 0``.
    n_estimators : int
        Nombre maximal d'arbres (l'arrêt précoce retient le meilleur).
    c_fn, c_fp : float
        Coûts d'un faux négatif / faux positif pour le choix du seuil.
    drop_cols : séquence de str
        Variables à exclure du modèle (ex. ``("CODE_GENDER",)`` pour ne pas
        utiliser une caractéristique protégée).

    Retourne
    --------
    dict
        Les métriques d'évaluation (voir :func:`evaluate`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, cat_cols = prepare_train_test(data_dir, drop_cols)

    if n_trials > 0:
        X_fit, X_val, y_fit, y_val = split_fit_val(X_train, y_train)
        params = tune_hyperparameters(X_fit, y_fit, X_val, y_val, n_trials, timeout_minutes)
    params = dict(BEST_PARAMS if params is None else params)

    model, threshold, (X_fit, y_fit, X_val, y_val) = fit_model(
        X_train, y_train, params, n_estimators, c_fn, c_fp
    )
    metrics = evaluate(
        model,
        {"train": (X_fit, y_fit), "val": (X_val, y_val), "test": (X_test, y_test)},
        threshold, c_fn, c_fp,
    )
    logger.info("Métriques : %s", json.dumps(metrics, indent=2))

    model.save_model(str(out_dir / "model.json"))
    joblib.dump(
        {
            "features": list(X_train.columns),
            "cat_cols": cat_cols,
            "categories": {c: list(X_train[c].cat.categories) for c in cat_cols},
            "threshold": threshold,
            "costs": {"FN": c_fn, "FP": c_fp},
            "params": params,
            "best_iteration": int(model.best_iteration),
            "metrics": metrics,
        },
        out_dir / "model_meta.joblib",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Artefacts écrits dans %s", out_dir)
    return metrics


# --------------------------------------------------------------------------- #
# 5. Scoring de nouvelles données
# --------------------------------------------------------------------------- #

def load_model(model_dir: str | Path = "deploy") -> tuple[xgb.Booster, dict]:
    """Charge le modèle et ses métadonnées depuis ``model_dir``.

    Retourne
    --------
    tuple
        ``(booster, meta)`` où ``meta`` contient ``features``, ``cat_cols``,
        ``categories`` et ``threshold``.
    """
    model_dir = Path(model_dir)
    booster = xgb.Booster()
    booster.load_model(str(model_dir / "model.json"))
    return booster, joblib.load(model_dir / "model_meta.joblib")


def predict_proba(features: pd.DataFrame, model_dir: str | Path = "deploy") -> np.ndarray:
    """Probabilité de défaut pour chaque ligne de ``features``.

    Paramètres
    ----------
    features : pd.DataFrame
        Variables au format de :func:`build_features` (sans ``TARGET``). Les
        colonnes manquantes sont traitées comme des valeurs manquantes.
    model_dir : str ou Path
        Dossier contenant ``model.json`` et ``model_meta.joblib``.
    """
    booster, meta = load_model(model_dir)
    X = align_to_meta(features, meta)
    return booster.predict(xgb.DMatrix(X, enable_categorical=True))


def score(
    app: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    model_dir: str | Path = "deploy",
) -> pd.DataFrame:
    """Score de nouvelles demandes de crédit de bout en bout.

    Paramètres
    ----------
    app : pd.DataFrame
        Nouvelles demandes (même format que ``application_train``, sans ``TARGET``).
    tables : mapping
        Tables d'historique de ces clients (voir :func:`load_tables`).
    model_dir : str ou Path
        Dossier des artefacts du modèle.

    Retourne
    --------
    pd.DataFrame
        Colonnes ``SK_ID_CURR``, ``probability`` (risque de défaut) et
        ``decision`` (``"refuser"`` si ``probability >= seuil``, sinon ``"accepter"``).
    """
    _, meta = load_model(model_dir)
    X, _ = split_features_target(build_features(app, tables))
    proba = predict_proba(X, model_dir)
    return pd.DataFrame(
        {
            ID_COL: app[ID_COL].to_numpy(),
            "probability": proba,
            "decision": np.where(proba >= meta["threshold"], "refuser", "accepter"),
        }
    )


# --------------------------------------------------------------------------- #
# 6. Ligne de commande
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments (sous-commandes ``train`` et ``predict``)."""
    parser = argparse.ArgumentParser(description="Pipeline de scoring de risque de crédit.")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Entraîne le modèle et sauvegarde les artefacts.")
    t.add_argument("--data-dir", default="data", help="Dossier des CSV Kaggle (défaut : data).")
    t.add_argument("--out-dir", default="deploy", help="Dossier de sortie (défaut : deploy).")
    t.add_argument("--n-trials", type=int, default=0,
                   help="Essais Optuna ; 0 = hyperparamètres du notebook (défaut).")
    t.add_argument("--timeout", type=float, default=45, help="Durée max. d'Optuna en minutes.")
    t.add_argument("--c-fn", type=float, default=COST_FN, help="Coût d'un faux négatif.")
    t.add_argument("--c-fp", type=float, default=COST_FP, help="Coût d'un faux positif.")
    t.add_argument("--drop-cols", nargs="*", default=[], metavar="COL",
                   help="Variables à exclure du modèle (ex. CODE_GENDER).")

    p = sub.add_parser("predict", help="Score de nouvelles demandes.")
    p.add_argument("--app", required=True, help="CSV des nouvelles demandes (format application_*).")
    p.add_argument("--data-dir", default=None,
                   help="Dossier des tables d'historique ; sans lui, scoring sur la seule demande.")
    p.add_argument("--model-dir", default="deploy", help="Dossier du modèle (défaut : deploy).")
    p.add_argument("--output", default="predictions.csv", help="CSV de sortie.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Point d'entrée de la ligne de commande."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    if args.command == "train":
        train(args.data_dir, args.out_dir, n_trials=args.n_trials,
              timeout_minutes=args.timeout, c_fn=args.c_fn, c_fp=args.c_fp,
              drop_cols=args.drop_cols)
    else:
        app = pd.read_csv(args.app)
        tables = load_tables(args.data_dir, client_ids=app[ID_COL]) if args.data_dir else {}
        result = score(app, tables, args.model_dir)
        result.to_csv(args.output, index=False)
        logger.info("%s prédictions écrites dans %s (taux de refus : %.1f %%)",
                    len(result), args.output, 100 * (result["decision"] == "refuser").mean())


if __name__ == "__main__":
    main()
