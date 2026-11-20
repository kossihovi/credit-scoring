"""Interface Streamlit du scoring de crédit.

Le formulaire envoie à l'API (``api.py``) **exactement** les champs de son
schéma ``Client`` : tout champ supplémentaire serait refusé (HTTP 422).

Lancement (l'API doit tourner) ::

    streamlit run streamlit_app.py

L'URL de l'API se règle avec la variable d'environnement ``API_URL``
(défaut : ``http://localhost:8000``).
"""

import os
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Modalités acceptées par l'API (celles vues à l'entraînement du modèle).
FAMILY_STATUS = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
EDUCATION = ["Secondary / secondary special", "Higher education", "Incomplete higher",
             "Lower secondary", "Academic degree"]
ORGANIZATION_TYPES = [
    'Advertising', 'Agriculture', 'Bank', 'Business Entity Type 1', 'Business Entity Type 2',
    'Business Entity Type 3', 'Cleaning', 'Construction', 'Culture', 'Electricity', 'Emergency',
    'Government', 'Hotel', 'Housing', 'Industry: type 1', 'Industry: type 10',
    'Industry: type 11', 'Industry: type 12', 'Industry: type 13', 'Industry: type 2',
    'Industry: type 3', 'Industry: type 4', 'Industry: type 5', 'Industry: type 6',
    'Industry: type 7', 'Industry: type 8', 'Industry: type 9', 'Insurance', 'Kindergarten',
    'Legal Services', 'Medicine', 'Military', 'Mobile', 'Other', 'Police', 'Postal', 'Realtor',
    'Religion', 'Restaurant', 'School', 'Security', 'Security Ministries', 'Self-employed',
    'Services', 'Telecom', 'Trade: type 1', 'Trade: type 2', 'Trade: type 3', 'Trade: type 4',
    'Trade: type 5', 'Trade: type 6', 'Trade: type 7', 'Transport: type 1', 'Transport: type 2',
    'Transport: type 3', 'Transport: type 4', 'University', 'XNA',
]

st.set_page_config(page_title="Scoring crédit", layout="wide")
st.title("Scoring de risque de crédit")
tab_pred, tab_mon = st.tabs(["Prédiction", "Monitoring"])
if "history" not in st.session_state:
    st.session_state.history = []


def score_input(label: str, key: str) -> Optional[float]:
    """Curseur 0-1 pour un score externe ; renvoie ``None`` si « inconnu » est coché."""
    value = st.slider(label, 0.0, 1.0, 0.5, 0.01, key=f"val_{key}")
    unknown = st.checkbox("inconnu", key=f"unk_{key}")
    return None if unknown else value


def optional_number(label: str, key: str, default: float, min_value: Optional[float] = None,
                    max_value: Optional[float] = None, step: float = 0.1,
                    unknown_by_default: bool = True) -> Optional[float]:
    """Champ numérique facultatif ; renvoie ``None`` si « non renseigné » est coché."""
    value = st.number_input(label, min_value=min_value, max_value=max_value,
                            value=default, step=step, key=f"val_{key}")
    unknown = st.checkbox("non renseigné", value=unknown_by_default, key=f"unk_{key}")
    return None if unknown else value


def render_result(res: dict) -> None:
    """Affiche la jauge de risque, la décision et les principaux facteurs."""
    thr = res["threshold"] * 100
    prob = res["probability"] * 100
    upper = max(50, int(-(-prob // 10) * 10))  # l'échelle s'étend si le risque dépasse 50 %
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob,
        number={"suffix": " %"},
        gauge={"axis": {"range": [0, upper]},
               "steps": [{"range": [0, thr], "color": "#c8e6c9"},
                         {"range": [thr, upper], "color": "#ffcdd2"}],
               "threshold": {"line": {"color": "red", "width": 4}, "value": thr}},
    ))
    fig.update_layout(height=300, margin=dict(t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    msg = f"(risque {res['probability']:.1%}, seuil {res['threshold']:.1%})"
    if res["decision"] == "accepter":
        st.success(f"Décision : ACCEPTER {msg}")
    else:
        st.error(f"Décision : REFUSER {msg}")
    st.caption(
        "Le modèle est alimenté par un formulaire réduit (16 variables sur 258) : les autres "
        "sont traitées comme manquantes, ce qui tend à surestimer le risque."
    )

    st.subheader("Principaux facteurs")
    fac = pd.DataFrame(res["top_factors"])
    fac["value"] = fac["value"].fillna("non renseignée")
    st.bar_chart(fac.set_index("feature")["contribution"])
    st.dataframe(fac, hide_index=True)
    st.caption("Contribution positive : augmente le risque. Négative : le diminue.")


with tab_pred:
    with st.form("client"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Profil**")
            age = st.number_input("Âge", 18, 100, 40)
            family = st.selectbox("Situation familiale", FAMILY_STATUS)
            education = st.selectbox("Études", EDUCATION)
            organization = st.selectbox("Type d'organisation de l'employeur", ORGANIZATION_TYPES,
                                        index=ORGANIZATION_TYPES.index("Business Entity Type 3"))

        with c2:
            st.markdown("**Emploi et véhicule**")
            years_emp = st.number_input("Ancienneté dans l'emploi (années)", 0.0, 60.0, 5.0, 0.5)
            no_job = st.checkbox("Sans emploi / retraité (ancienneté non applicable)")
            has_car = st.checkbox("Possède un véhicule")
            car_age = st.number_input("Âge du véhicule (années)", 0.0, 80.0, 5.0, 1.0)

        with c3:
            st.markdown("**Crédit demandé**")
            credit = st.number_input("Montant du crédit", 1000.0, value=500000.0, step=10000.0)
            annuity = st.number_input("Annuité (remboursement annuel)", 100.0, value=25000.0, step=500.0)
            goods = st.number_input("Prix du bien financé", 0.0, value=450000.0, step=10000.0)

        st.markdown("**Scores externes** (0 = risqué, 1 = sûr)")
        s1, s2, s3 = st.columns(3)
        with s1:
            e1 = score_input("EXT_SOURCE_1", "e1")
        with s2:
            e2 = score_input("EXT_SOURCE_2", "e2")
        with s3:
            e3 = score_input("EXT_SOURCE_3", "e3")

        with st.expander("Historique de crédit (optionnel)"):
            h1, h2 = st.columns(2)
            with h1:
                prev_due = optional_number(
                    "Demandes précédentes : jours de la dernière échéance prévue (max, relatif à la demande)",
                    "prev_due", -500.0, step=50.0)
                prev_ratio = optional_number(
                    "Demandes précédentes : ratio moyen crédit accordé / demandé",
                    "prev_ratio", 1.0, min_value=0.0, step=0.05)
            with h2:
                bureau_ratio = optional_number(
                    "Bureau de crédit : dette restante / total emprunté",
                    "bureau_ratio", 0.3, min_value=0.0, step=0.05)
                inst_late = optional_number(
                    "Paiements passés : part des échéances payées en retard",
                    "inst_late", 0.1, min_value=0.0, max_value=1.0, step=0.01)

        submitted = st.form_submit_button("Évaluer")

    if submitted:
        # Les clés correspondent champ pour champ au schéma `Client` de api.py.
        payload = {
            "age_years": age,
            "amt_credit": credit,
            "amt_annuity": annuity,
            "amt_goods_price": goods,
            "years_employed": None if no_job else years_emp,
            "own_car_age": car_age if has_car else None,
            "organization_type": organization,
            "name_family_status": family,
            "name_education_type": education,
            "ext_source_1": e1,
            "ext_source_2": e2,
            "ext_source_3": e3,
            "prev_days_last_due_1st_version_max": prev_due,
            "prev_prev_credit_ratio_mean": prev_ratio,
            "bureau_debt_ratio": bureau_ratio,
            "inst_is_late_mean": inst_late,
        }

        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            r.raise_for_status()
            res = r.json()
        except requests.RequestException as e:
            detail = e.response.text if getattr(e, "response", None) is not None else ""
            st.error(f"API indisponible ou requête refusée : {e} {detail}")
            st.stop()

        render_result(res)
        st.session_state.history.append(
            {"probabilité": res["probability"], "décision": res["decision"]}
        )

with tab_mon:
    h = pd.DataFrame(st.session_state.history)
    if h.empty:
        st.info("Aucune prédiction pour l'instant.")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Prédictions", len(h))
        m2.metric("Taux de refus", f"{(h['décision'] == 'refuser').mean():.0%}")
        st.bar_chart(h["probabilité"])
        st.dataframe(h, hide_index=True)
