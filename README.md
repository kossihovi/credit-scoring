# Scoring de risque de crédit — Home Credit Default Risk

Projet de bout en bout sur le jeu de données Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) :
exploration, feature engineering sur 7 tables, modèle XGBoost optimisé avec Optuna, **choix du seuil de décision selon un coût métier**,
explicabilité (SHAP), puis déploiement via une **API FastAPI** et une **interface Streamlit**.

## Résultats

Évaluation sur un jeu de test de 20 % (stratifié), jamais utilisé pour l'entraînement ni le réglage.

| Indicateur | Valeur |
|---|---|
| AUC ROC (test) | **0,786** (régression logistique de référence : 0,774) |
| Brier (test) | 0,0657 (sans modèle : 0,0742) |
| Seuil de refus | 0,079 (coût d'un défaut non détecté = 10 × celui d'un bon client refusé) |
| Taux de refus / rappel / précision | 30,7 % / 70,4 % / 18,5 % |
| Coût moyen par client | **0,489** (tout accepter : 0,807 ; tout refuser : 0,919) |

Le seuil est choisi en minimisant le coût total sur un jeu de validation, et non à 0,5 : avec 8 % de défauts, un seuil « classique » n'aurait aucun sens métier.

## Architecture

```
.
├── credit_scoring/
│   ├── pipeline.py          # Features, entraînement, seuil, scoring de nouvelles données (+ CLI)
│   ├-- analysis.py          # Analyse d'équité
├── deploy/
│   ├── api.py               # API FastAPI (/health, /predict)
│   ├── streamlit_app.py     # Interface de saisie et de monitoring
│   ├── model.json           # Modèle XGBoost entraîné
│   ├── model_meta.joblib    # Variables, catégories, seuil, coûts
│   └── requirements.txt
├── notebooks/
│   └── home_credit.ipynb    # Exploration, modélisation, SHAP, calibration
├── tests/                   # pytest : pipeline, API, cohérence Streamlit ↔ API
└── .github/workflows/ci.yml # Tests automatiques à chaque push
```

Le notebook sert à explorer et à expliquer la démarche ; `credit_scoring/pipeline.py` en est la version réutilisable et testée.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # ajouter requirements-dev.txt pour les tests
```

## Utiliser l'application

```bash
cd deploy
uvicorn api:app --reload                 # terminal 1 → http://localhost:8000/docs
streamlit run streamlit_app.py           # terminal 2 → http://localhost:8501
```

Exemple d'appel direct à l'API :

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age_years": 40, "amt_credit": 500000, "amt_annuity": 25000, "amt_goods_price": 450000,
  "years_employed": 5, "ext_source_2": 0.5, "ext_source_3": 0.5,
  "name_family_status": "Married", "name_education_type": "Higher education",
  "organization_type": "Business Entity Type 3"
}'
```

La réponse contient la probabilité de défaut, la décision (`accepter` / `refuser`) et les 5 variables qui ont le plus pesé (contributions SHAP).
L'API rejette (HTTP 422) tout champ inconnu ou toute modalité non vue à l'entraînement, plutôt que de l'ignorer silencieusement.

## Ré-entraîner le modèle ou scorer de nouvelles données

Placer les CSV Kaggle dans `data/` (non versionné), puis :

```bash
# Entraînement (hyperparamètres du notebook ; ajouter --n-trials 15 pour relancer Optuna)
python -m credit_scoring.pipeline train --data-dir data --out-dir deploy

# Scoring d'un fichier de nouvelles demandes
python -m credit_scoring.pipeline predict \
    --app data/application_test.csv --data-dir data \
    --model-dir deploy --output predictions.csv
```

Ou depuis Python :

```python
import pandas as pd
from credit_scoring.pipeline import load_tables, score

app = pd.read_csv("data/application_test.csv")
tables = load_tables("data", client_ids=app["SK_ID_CURR"])   # ne charge que les lignes utiles
predictions = score(app, tables, model_dir="deploy")          # SK_ID_CURR, probability, decision
```

## Analyses de robustesse et d'équité

Trois questions que se posera tout lecteur critique, regroupées dans un seul rapport :

```bash
python -m credit_scoring.analysis --data-dir data     # ~20-40 min ; --folds 3 pour aller plus vite
```

| Question | Méthode |
|---|---|
| Que perd-on en retirant `CODE_GENDER` ? | Même modèle entraîné avec puis sans la variable, comparé sur le jeu de test |
| L'avantage de XGBoost sur la régression logistique est-il réel ? | Validation croisée à 5 parts, mêmes découpages pour les deux modèles, intervalle de confiance à 95 % sur l'écart d'AUC |
| Le modèle refuse-t-il plus souvent certains groupes ? | Taux de refus, défauts détectés et **bons clients refusés à tort**, par genre et par tranche d'âge |

Le rapport (`reports/analysis.txt`) et les tableaux CSV sont écrits dans `reports/`.

## Tests

```bash
pytest
```

Les tests utilisent un jeu de données synthétique et vérifient notamment que le formulaire Streamlit envoie exactement les champs du schéma de l'API.

## Choix et limites

- **Formulaire partiel.** Le modèle complet utilise 258 variables ; le formulaire n'en demande que 16 (les plus importantes selon SHAP). Les autres sont transmises comme valeurs manquantes. Sur le jeu de test, l'AUC passe de 0,786 à 0,746 et la probabilité moyenne prédite de 0,079 à 0,153 : **le risque est surestimé** et le seuil (calibré sur le modèle complet) refuse donc plus souvent qu'attendu. Un recalibrage du modèle partiel est la suite logique.
- **Historique de crédit.** L'API force `HAS_BUREAU = 0` : les agrégats d'historique saisis dans le formulaire sont optionnels et ne signalent pas la présence d'un dossier bureau.
- **Variables sensibles.** `CODE_GENDER` figure parmi les variables du modèle livré (le formulaire ne l'envoie pas). Pour un usage réel, l'exclure : `python -m credit_scoring.pipeline train --drop-cols CODE_GENDER` ; le rapport d'analyse mesure ce que ce retrait coûte et audite les écarts de décision par genre et par âge. Un écart de décision entre groupes ne prouve pas à lui seul une discrimination, mais appelle une explication.
- **Données.** Jeu de données public de compétition ; ce projet est une démonstration, pas un outil d'octroi de crédit.

## Pistes d'amélioration

Recalibration (isotonique / Platt) du modèle partiel, suivi de dérive des données en production (le monitoring actuel ne couvre que la session Streamlit), conteneurisation Docker, mesures d'équité plus fines (Fairlearn).
