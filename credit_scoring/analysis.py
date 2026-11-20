"""Trois analyses pour juger le modèle de façon honnête.

1. **Retirer une variable sensible** (``CODE_GENDER``) : que perd-on en performance ?
2. **Validation croisée** : l'avantage de XGBoost sur la régression logistique est-il
   réel, ou dû au hasard du découpage train / test ?
3. **Équité** : le modèle refuse-t-il plus souvent certains groupes (genre, âge) ?

Lancement (les CSV Kaggle sont dans ``data/``) ::

    python -m credit_scoring.analysis --data-dir data

Le rapport est écrit dans ``reports/analysis.txt``. Comptez 20 à 40 minutes sur les
données complètes ; ``--folds 3`` accélère la validation croisée.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from credit_scoring.pipeline import (
    BEST_PARAMS,
    SEED,
    _make_classifier,
    evaluate,
    fit_model,
    prepare_train_test,
)

logger = logging.getLogger(__name__)

#: Nombre d'arbres retenu par l'arrêt précoce dans le notebook.
BEST_N_TREES = 743


# --------------------------------------------------------------------------- #
# 1. Que coûte le retrait d'une variable sensible ?
# --------------------------------------------------------------------------- #

def compare_without_column(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    column: str = "CODE_GENDER",
    n_estimators: int = 3000,
) -> tuple[pd.DataFrame, dict]:
    """Entraîne le modèle avec, puis sans ``column``, et compare les résultats sur le test.

    Les deux modèles utilisent le même découpage et les mêmes hyperparamètres : la
    seule différence est la variable retirée. Chaque modèle a son propre seuil de refus.

    Retourne
    --------
    tuple
        - un tableau (une ligne par modèle : AUC, seuil, taux de refus, coût par client) ;
        - un dictionnaire ``{"avec": (modèle, seuil), "sans": (modèle, seuil)}`` réutilisable
          pour l'analyse d'équité.
    """
    rows, fitted = [], {}
    for label, drop in (("avec", []), ("sans", [column])):
        Xtr, Xte = X_train.drop(columns=drop), X_test.drop(columns=drop)
        model, threshold, (X_fit, y_fit, X_val, y_val) = fit_model(
            Xtr, y_train, n_estimators=n_estimators, verbose=False
        )
        m = evaluate(
            model,
            {"train": (X_fit, y_fit), "val": (X_val, y_val), "test": (Xte, y_test)},
            threshold,
        )
        rows.append({
            "modèle": f"{label} {column}",
            "AUC test": m["auc_test"],
            "seuil de refus": m["threshold"],
            "taux de refus": m["refusal_rate"],
            "coût par client": m["cost_per_client_model"],
        })
        fitted[label] = (model, threshold)
    return pd.DataFrame(rows).set_index("modèle"), fitted


# --------------------------------------------------------------------------- #
# 2. Validation croisée : XGBoost vs régression logistique
# --------------------------------------------------------------------------- #

def make_logistic_baseline(num_cols: Sequence[str], cat_cols: Sequence[str]) -> Pipeline:
    """Régression logistique de référence (même préparation que dans le notebook).

    Numériques : imputation par la médiane (+ indicateur de manquant) puis
    transformation en rangs ; catégorielles : modalité « MISSING » puis one-hot.
    """
    num = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("qt", QuantileTransformer(output_distribution="normal", n_quantiles=200,
                                   subsample=100_000, random_state=SEED)),
    ])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
    ])
    prep = ColumnTransformer([("num", num, list(num_cols)), ("cat", cat, list(cat_cols))])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=0.1, max_iter=5000, tol=1e-3))])


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: Sequence[str],
    n_folds: int = 5,
    n_estimators: int = BEST_N_TREES,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """Compare XGBoost et régression logistique sur les **mêmes** découpages.

    Les données sont coupées en ``n_folds`` parts ; chaque part sert tour à tour de
    test pendant que les autres servent à entraîner. XGBoost utilise un nombre
    d'arbres fixe (pas d'arrêt précoce), pour ne jamais « regarder » la part de test.

    Retourne
    --------
    pd.DataFrame
        Une ligne par part : ``auc_xgboost``, ``auc_logistique`` et leur écart.
    """
    params = dict(BEST_PARAMS if params is None else params)
    num_cols = [c for c in X.columns if c not in cat_cols]
    X_lr = X.astype({c: object for c in cat_cols})  # la régression logistique veut du texte, pas des catégories

    rows = []
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for i, (tr, te) in enumerate(folds.split(X, y), start=1):
        xgb_model = _make_classifier(n_estimators=n_estimators, early_stopping_rounds=None, **params)
        xgb_model.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        auc_xgb = roc_auc_score(y.iloc[te], xgb_model.predict_proba(X.iloc[te])[:, 1])

        lr = make_logistic_baseline(num_cols, cat_cols).fit(X_lr.iloc[tr], y.iloc[tr])
        auc_lr = roc_auc_score(y.iloc[te], lr.predict_proba(X_lr.iloc[te])[:, 1])

        rows.append({"part": i, "auc_xgboost": auc_xgb, "auc_logistique": auc_lr,
                     "écart": auc_xgb - auc_lr})
        logger.info("Part %s/%s : XGBoost %.4f | logistique %.4f", i, n_folds, auc_xgb, auc_lr)
    return pd.DataFrame(rows).set_index("part")


def summarize_cv(cv: pd.DataFrame) -> dict:
    """Résume la validation croisée : moyenne, dispersion et intervalle de confiance à 95 %.

    L'intervalle porte sur l'**écart** XGBoost − logistique. S'il ne contient pas 0,
    l'avantage de XGBoost est probablement réel. C'est une approximation : les parts
    partagent des données d'entraînement, l'intervalle est donc un peu trop étroit.
    """
    n = len(cv)
    diff = cv["écart"]
    half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
    return {
        "auc_xgboost_moyen": cv["auc_xgboost"].mean(),
        "auc_xgboost_ecart_type": cv["auc_xgboost"].std(ddof=1),
        "auc_logistique_moyen": cv["auc_logistique"].mean(),
        "ecart_moyen": diff.mean(),
        "ic95_bas": diff.mean() - half,
        "ic95_haut": diff.mean() + half,
        "xgboost_meilleur_dans": int((diff > 0).sum()),
        "n_parts": n,
    }


# --------------------------------------------------------------------------- #
# 3. Équité : qui est refusé ?
# --------------------------------------------------------------------------- #

def fairness_table(
    y_true: pd.Series,
    proba: np.ndarray,
    group: pd.Series,
    threshold: float,
    min_group_size: int = 500,
) -> pd.DataFrame:
    """Compare, groupe par groupe, les décisions du modèle.

    Pour chaque groupe (ex. femmes / hommes) :

    - ``taux_defaut_reel`` : part des clients qui ont réellement fait défaut ;
    - ``taux_refus`` : part des clients refusés par le modèle ;
    - ``defauts_detectes`` : part des vrais défauts qui ont été refusés (rappel) ;
    - ``bons_clients_refuses`` : part des clients fiables refusés à tort.

    La dernière colonne est la plus parlante : à comportement identique (aucun
    défaut), un groupe est-il refusé plus souvent qu'un autre ?

    Paramètres
    ----------
    y_true : cibles réelles (1 = défaut).
    proba : probabilités de défaut prédites.
    group : groupe de chaque client (même index que ``y_true``).
    threshold : seuil de refus.
    min_group_size : int
        Les groupes plus petits sont ignorés (chiffres trop instables).
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "refuse": np.asarray(proba) >= threshold,
                       "groupe": pd.Series(group).reset_index(drop=True)})
    rows = {}
    for name, g in df.groupby("groupe", observed=True):
        if len(g) < min_group_size:
            continue
        good, bad = g[g["y"] == 0], g[g["y"] == 1]
        rows[name] = {
            "clients": len(g),
            "taux_defaut_reel": g["y"].mean(),
            "taux_refus": g["refuse"].mean(),
            "defauts_detectes": bad["refuse"].mean() if len(bad) else np.nan,
            "bons_clients_refuses": good["refuse"].mean() if len(good) else np.nan,
        }
    return pd.DataFrame(rows).T


def age_groups(age_years: pd.Series) -> pd.Series:
    """Tranches d'âge : moins de 30 ans, 30-44, 45-59, 60 ans et plus."""
    return pd.cut(age_years, [0, 30, 45, 60, 200], right=False,
                  labels=["< 30 ans", "30-44 ans", "45-59 ans", "60 ans et +"])


def fairness_gaps(table: pd.DataFrame) -> dict:
    """Écarts maximaux entre groupes, en points de pourcentage."""
    return {
        "ecart_taux_refus": 100 * (table["taux_refus"].max() - table["taux_refus"].min()),
        "ecart_bons_clients_refuses": 100 * (
            table["bons_clients_refuses"].max() - table["bons_clients_refuses"].min()
        ),
    }


# --------------------------------------------------------------------------- #
# Rapport complet
# --------------------------------------------------------------------------- #

def _fmt(df: pd.DataFrame) -> str:
    return df.round(4).to_string()


def run_analysis(
    data_dir: str | Path,
    out_dir: str | Path = "reports",
    column: str = "CODE_GENDER",
    n_folds: int = 5,
    n_estimators: int = 3000,
    cv_estimators: int = BEST_N_TREES,
    skip_cv: bool = False,
    min_group_size: int = 500,
) -> str:
    """Exécute les trois analyses et écrit le rapport dans ``out_dir/analysis.txt``.

    Paramètres
    ----------
    data_dir : dossier des CSV Kaggle.
    out_dir : dossier du rapport.
    column : variable sensible à retirer (défaut ``CODE_GENDER``).
    n_folds : nombre de parts de la validation croisée.
    n_estimators : nombre maximal d'arbres pour l'analyse 1.
    cv_estimators : nombre d'arbres fixe pour la validation croisée.
    skip_cv : ignorer la validation croisée (la plus longue).
    min_group_size : taille minimale d'un groupe pour l'analyse d'équité.

    Retourne
    --------
    str
        Le texte du rapport.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, cat_cols = prepare_train_test(data_dir)
    parts: list[str] = []

    # --- 1. variable sensible ---
    logger.info("Analyse 1/3 : modèle avec / sans %s", column)
    comparison, fitted = compare_without_column(X_train, y_train, X_test, y_test, column, n_estimators)
    loss = comparison["AUC test"].iloc[0] - comparison["AUC test"].iloc[1]
    parts += [
        f"1. EFFET DU RETRAIT DE {column}", _fmt(comparison),
        f"-> Retirer {column} change l'AUC de {-loss:+.4f}"
        + (" (perte négligeable : la variable n'apporte presque rien)." if abs(loss) < 0.002
           else " (la variable apportait de l'information au modèle)."),
        "",
    ]

    # --- 2. validation croisée ---
    if not skip_cv:
        logger.info("Analyse 2/3 : validation croisée (%s parts)", n_folds)
        X_all = pd.concat([X_train, X_test])
        y_all = pd.concat([y_train, y_test])
        X_cv = X_all.drop(columns=[column])  # on compare les modèles tels qu'ils seraient déployés
        cv = cross_validate_models(X_cv, y_all, [c for c in cat_cols if c != column],
                                   n_folds, cv_estimators)
        s = summarize_cv(cv)
        verdict = ("l'intervalle exclut 0 : l'avantage de XGBoost est probablement réel."
                   if s["ic95_bas"] > 0 else
                   "l'intervalle contient 0 : l'avantage de XGBoost n'est pas démontré.")
        parts += [
            "2. VALIDATION CROISÉE : XGBOOST vs RÉGRESSION LOGISTIQUE", _fmt(cv),
            f"AUC moyen : XGBoost {s['auc_xgboost_moyen']:.4f} (± {s['auc_xgboost_ecart_type']:.4f}) "
            f"| logistique {s['auc_logistique_moyen']:.4f}",
            f"Écart moyen {s['ecart_moyen']:+.4f}, intervalle à 95 % "
            f"[{s['ic95_bas']:+.4f} ; {s['ic95_haut']:+.4f}]",
            f"XGBoost est meilleur dans {s['xgboost_meilleur_dans']} parts sur {s['n_parts']} ; {verdict}",
            "",
        ]
        cv.to_csv(out_dir / "cross_validation.csv")

    # --- 3. équité ---
    logger.info("Analyse 3/3 : équité")
    parts.append("3. ÉQUITÉ (jeu de test)")
    groups = {"genre": X_test[column] if column in X_test else None,
              "âge": age_groups(X_test["AGE_YEARS"])}
    for label, (model, thr) in fitted.items():
        parts.append(f"\nModèle {label} {column} (seuil {thr:.3f})")
        proba = model.predict_proba(X_test.drop(columns=[column] if label == "sans" else []))[:, 1]
        for attr, grp in groups.items():
            if grp is None:
                continue
            table = fairness_table(y_test, proba, grp, thr, min_group_size)
            if table.empty:
                parts.append(f"\nPar {attr} : aucun groupe d'au moins {min_group_size} clients.")
                continue
            gaps = fairness_gaps(table)
            parts += [f"\nPar {attr} :", _fmt(table),
                      f"-> écart de taux de refus : {gaps['ecart_taux_refus']:.1f} points ; "
                      f"écart de bons clients refusés à tort : {gaps['ecart_bons_clients_refuses']:.1f} points"]
            table.to_csv(out_dir / f"fairness_{attr}_{label}.csv")
    parts += [
        "",
        "Lecture : un écart de taux de refus peut refléter de vraies différences de risque "
        "(voir taux_defaut_reel). L'écart de « bons clients refusés à tort » est plus "
        "révélateur : il mesure qui est refusé alors qu'il aurait remboursé.",
    ]

    report = "\n".join(parts)
    (out_dir / "analysis.txt").write_text(report, encoding="utf-8")
    comparison.to_csv(out_dir / "feature_removal.csv")
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Point d'entrée de la ligne de commande."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Analyses : variable sensible, validation croisée, équité.")
    p.add_argument("--data-dir", default="data", help="Dossier des CSV Kaggle (défaut : data).")
    p.add_argument("--out-dir", default="reports", help="Dossier du rapport (défaut : reports).")
    p.add_argument("--column", default="CODE_GENDER", help="Variable sensible à retirer.")
    p.add_argument("--folds", type=int, default=5, help="Parts de la validation croisée (défaut : 5).")
    p.add_argument("--skip-cv", action="store_true", help="Ignorer la validation croisée (la plus longue).")
    args = p.parse_args(argv)
    print(run_analysis(args.data_dir, args.out_dir, args.column, n_folds=args.folds, skip_cv=args.skip_cv))


if __name__ == "__main__":
    main()
