import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

REFERENCES = Path("data/references")
API_URL = "http://127.0.0.1:8000"
MODEL_NAME = "stacking_model"


def load_json(filename):
    path = REFERENCES / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


st.set_page_config(
    page_title="Home Credit Default Risk",
    page_icon="🏦",
    layout="wide",
)

st.sidebar.title("Home Credit Risk")
page = st.sidebar.radio(
    "Navigation",
    ["Prédiction", "Performance", "Interprétabilité", "Fairness"],
)


def render_prediction_page():
    st.title("Prédiction de risque de défaut")
    st.markdown("Renseignez les informations du client pour obtenir une prédiction.")

    st.subheader("Informations financières")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        income_amount = st.number_input("Revenu annuel (€)", min_value=0.0, value=30000.0, step=1000.0)
    with col2:
        credit_amount = st.number_input("Montant du crédit (€)", min_value=0.0, value=900000.0, step=1000.0)
    with col3:
        annuity_amount = st.number_input("Annuité mensuelle (€)", min_value=0.0, value=80000.0, step=100.0)
    with col4:
        amt_goods_price = st.number_input("Prix du bien financé (€)", min_value=0.0, value=920000.0, step=1000.0)

    ratio_income_credit = income_amount / credit_amount if credit_amount else 0.0
    credit_to_annuity_ratio = credit_amount / annuity_amount if annuity_amount else 0.0
    credit_to_goods_ratio = credit_amount / amt_goods_price if amt_goods_price else 0.0

    st.caption(
        f"Ratio revenu/crédit : **{ratio_income_credit:.3f}** · "
        f"Ratio crédit/annuité : **{credit_to_annuity_ratio:.1f}** · "
        f"Ratio crédit/prix du bien : **{credit_to_goods_ratio:.3f}**"
    )

    st.divider()

    with st.form("prediction_form"):
        st.subheader("Informations personnelles")
        col5, col6, col7 = st.columns(3)
        with col5:
            age_years = st.slider("Âge", min_value=18, max_value=70, value=22)
            age_group = "young" if age_years < 30 else "adult" if age_years <= 50 else "senior"
        with col6:
            is_employed = st.selectbox("Statut emploi", [1, 0], index=1, format_func=lambda x: "Employé" if x == 1 else "Non employé")
            days_employed = st.number_input("Jours d'emploi (négatif)", value=-1000.0, step=100.0) if is_employed else None
        with col7:
            owns_realty = st.selectbox("Propriétaire immobilier", [0, 1], index=0, format_func=lambda x: "Oui" if x == 1 else "Non")
            owns_car = st.selectbox("Propriétaire véhicule", [0, 1], index=0, format_func=lambda x: "Oui" if x == 1 else "Non")

        st.subheader("Informations sociales")
        col8, col9, col10 = st.columns(3)
        with col8:
            education_type = st.selectbox("Niveau d'éducation", [
                "Lower secondary",
                "Secondary / secondary special",
                "Incomplete higher",
                "Higher education",
                "Academic degree",
            ])
            occupation_type = st.selectbox("Profession", [
                "Low-skill Laborers", "Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
                "High skill tech staff", "Accountants", "Medicine staff",
                "Security staff", "Cooking staff", "Cleaning staff",
                "Private service staff", "Waiters/barmen staff",
                "Secretaries", "Realty agents", "HR staff", "IT staff",
            ])
        with col9:
            family_status = st.selectbox("Situation familiale", [
                "Single / not married", "Married", "Civil marriage",
                "Separated", "Widow",
            ])
            organization_type = st.selectbox("Secteur d'activité", [
                "Business Entity Type 3", "Business Entity Type 2", "Business Entity Type 1",
                "Self-employed", "Government", "School", "Kindergarten", "Medicine",
                "Construction", "Trade: type 7", "Military", "Bank", "Police",
                "Transport: type 4", "Housing", "Restaurant", "Industry: type 9",
                "Security Ministries", "University", "XNA", "Other",
            ])
        with col10:
            cnt_children = st.number_input("Nombre d'enfants", min_value=0, max_value=10, value=4)
            cnt_fam_members = st.number_input("Membres du foyer", min_value=1, max_value=15, value=6)

        st.subheader("Scores externes")
        col11, col12, col13 = st.columns(3)
        with col11:
            ext_source_2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.02, 0.01)
        with col12:
            ext_source_3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.02, 0.01)
        with col13:
            region_rating = st.selectbox("Note région", [1, 2, 3], index=2)

        st.subheader("Historique bureau")
        col14, col15, col16 = st.columns(3)
        with col14:
            bureau_credit_count = st.number_input("Nombre de crédits bureau", min_value=0, value=12)
            bureau_active_count = st.number_input("Crédits actifs", min_value=0, value=10)
        with col15:
            bureau_debt_total = st.number_input("Dette totale bureau (€)", min_value=0.0, value=850000.0, step=1000.0)
            bureau_overdue_max = st.number_input("Retard max bureau (jours)", min_value=0.0, value=90.0)
        with col16:
            debt_credit_ratio = st.slider("Ratio dette/crédit", 0.0, 1.0, 0.98, 0.01)
            overdue_debt_ratio = st.slider("Ratio retard/dette", 0.0, 1.0, 0.5, 0.01)
            loan_types_bureau = st.number_input("Types de crédit", min_value=0, value=5)

        st.subheader("Historique de remboursement")
        col17, col18, col19 = st.columns(3)
        with col17:
            inst_late_ratio = st.slider("Ratio de paiements en retard", 0.0, 1.0, 0.8, 0.01)
        with col18:
            prev_credit_ratio = st.slider("Ratio de crédits précédents bien remboursés", 0.0, 1.0, 0.2, 0.01)
        with col19:
            cc_utilization_avg = st.slider("Taux d'utilisation moyen carte de crédit", 0.0, 1.0, 0.95, 0.01)

        submitted = st.form_submit_button("Prédire", use_container_width=True)

    if not submitted:
        return

    payload = {
        "income_amount": income_amount,
        "credit_amount": credit_amount,
        "annuity_amount": annuity_amount,
        "ratio_income_credit": ratio_income_credit,
        "credit_to_annuity_ratio": credit_to_annuity_ratio,
        "credit_to_goods_ratio": credit_to_goods_ratio,
        "age_years": float(age_years),
        "age_group": age_group,
        "days_employed": float(days_employed) if days_employed else None,
        "is_employed": is_employed,
        "owns_car": owns_car,
        "owns_realty": owns_realty,
        "cnt_children": float(cnt_children),
        "cnt_fam_members": float(cnt_fam_members),
        "education_type": education_type,
        "occupation_type": occupation_type,
        "family_status": family_status,
        "organization_type": organization_type,
        "region_rating": float(region_rating),
        "ext_source_2": ext_source_2,
        "ext_source_3": ext_source_3,
        "bureau_credit_count": float(bureau_credit_count),
        "bureau_active_count": float(bureau_active_count),
        "bureau_debt_total": bureau_debt_total,
        "bureau_overdue_max": bureau_overdue_max,
        "debt_credit_ratio": debt_credit_ratio,
        "overdue_debt_ratio": overdue_debt_ratio,
        "loan_types_bureau": float(loan_types_bureau),
        "inst_late_ratio": inst_late_ratio,
        "prev_credit_ratio": prev_credit_ratio,
        "cc_utilization_avg": cc_utilization_avg,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        result = response.json()
    except requests.exceptions.ConnectionError:
        st.error("API non disponible — assurez-vous que uvicorn tourne sur le port 8000.")
        return
    except Exception as e:
        st.error(f"Erreur : {e}")
        return

    proba = result["default_probability"]
    decision = result["decision"]
    risk = result["risk_level"]
    threshold = result["threshold"]

    st.divider()
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Probabilité de défaut", f"{proba:.1%}")
    col_r2.metric("Seuil", f"{threshold:.2f}")
    col_r3.metric("Décision", "Défaut prédit" if decision == 1 else "Pas de défaut")

    if risk == "high":
        st.error(f"Risque élevé, probabilité de défaut : {proba:.1%}")
    elif risk == "medium":
        st.warning(f"Risque moyen, probabilité de défaut : {proba:.1%}")
    else:
        st.success(f"Risque faible, probabilité de défaut : {proba:.1%}")

    st.progress(min(proba, 1.0))


def render_performance_page():
    st.title("Performance du modèle")

    metrics = load_json(f"metrics_{MODEL_NAME}.json")
    if metrics is None:
        st.warning(f"Métriques non trouvées — lancez d'abord `python src/evaluate.py {MODEL_NAME}`")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC test", f"{metrics['auc']:.4f}")
    col2.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    col3.metric("FPR", f"{metrics['fpr'] * 100:.1f}%")
    col4.metric("Threshold", f"{metrics['threshold']:.2f}")

    st.divider()

    metrics_img = REFERENCES / f"metrics_{MODEL_NAME}.png"
    if metrics_img.exists():
        st.image(str(metrics_img), caption="Courbe ROC · Matrice de confusion · Precision-Recall", use_container_width=True)

    st.divider()
    st.subheader("Résumé des métriques")

    metrics_data = {
        "Métrique": ["AUC test", "Recall (TVP)", "FPR", "FNR", "Accuracy"],
        "Valeur": [
            f"{metrics['auc']:.4f}",
            f"{metrics['recall'] * 100:.1f}%",
            f"{metrics['fpr'] * 100:.1f}%",
            f"{metrics['fnr'] * 100:.1f}%",
            f"{metrics['accuracy'] * 100:.1f}%",
        ],
        "Interprétation": [
            "Capacité discriminante globale",
            f"{metrics['recall'] * 100:.1f}% des défauts détectés",
            f"{metrics['fpr'] * 100:.1f}% des bons clients refusés",
            f"{metrics['fnr'] * 100:.1f}% des défauts ratés",
            "Taux de bonnes prédictions global",
        ],
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)


def render_interpretability_page():
    st.title("Interprétabilité SHAP")
    st.markdown("Importance des features basée sur les valeurs SHAP (TreeExplainer sur LightGBM).")

    shap_img = REFERENCES / f"shap_summary_{MODEL_NAME}.png"
    if shap_img.exists():
        st.image(str(shap_img), caption="SHAP Summary Plot", use_container_width=True)
    else:
        st.warning(f"Image non trouvée — lancez d'abord `python src/evaluate.py {MODEL_NAME}`")

    st.divider()
    st.subheader("Top 15 features par importance SHAP")

    shap_csv = REFERENCES / f"shap_importance_{MODEL_NAME}.csv"
    if not shap_csv.exists():
        st.warning(f"CSV non trouvé — lancez d'abord `python src/evaluate.py {MODEL_NAME}`")
        return

    df_shap = pd.read_csv(shap_csv).head(15)
    st.bar_chart(df_shap.set_index("feature")["mean_abs_shap"])
    st.dataframe(df_shap, use_container_width=True, hide_index=True)


def render_fairness_page():
    st.title("Analyse de fairness")
    st.markdown("Audit d'équité par genre via Fairlearn. `gender_clean` n'est jamais utilisée comme feature du modèle.")

    fairness = load_json(f"fairness_{MODEL_NAME}.json")
    if fairness is None:
        st.warning(f"Métriques non trouvées — lancez d'abord `python src/evaluate.py {MODEL_NAME}`")
        return

    auc_f = fairness["auc_by_gender"]["F"]
    auc_m = fairness["auc_by_gender"]["M"]

    col1, col2 = st.columns(2)
    col1.metric("AUC Femmes", f"{auc_f:.4f}")
    col2.metric("AUC Hommes", f"{auc_m:.4f}")

    st.divider()

    fairness_img = REFERENCES / f"fairness_{MODEL_NAME}.png"
    if fairness_img.exists():
        st.image(str(fairness_img), caption="Fairness Analysis by Gender", use_container_width=True)

    st.divider()
    st.subheader("Métriques par genre")

    by_group = fairness["by_group"]
    fairness_data = {
        "Genre": ["F", "M"],
        "Selection Rate": [f"{by_group['selection_rate'][g] * 100:.1f}%" for g in ["F", "M"]],
        "False Positive Rate": [f"{by_group['false_positive_rate'][g] * 100:.1f}%" for g in ["F", "M"]],
        "False Negative Rate": [f"{by_group['false_negative_rate'][g] * 100:.1f}%" for g in ["F", "M"]],
        "AUC": [f"{auc_f:.4f}", f"{auc_m:.4f}"],
    }
    st.dataframe(pd.DataFrame(fairness_data), use_container_width=True, hide_index=True)

    fnr_gap = by_group["false_negative_rate"]["F"] - by_group["false_negative_rate"]["M"]
    higher_fnr_group = "féminins" if fnr_gap > 0 else "masculins"

    st.info(
        f"**Observation :** l'écart AUC F/M est de {abs(auc_f - auc_m):.4f}. "
        f"Le FNR est plus élevé pour les défauts {higher_fnr_group}, "
        "le modèle en rate davantage dans ce groupe. "
        "Piste d'amélioration : post-processing avec contrainte d'équité Fairlearn."
    )


PAGES = {
    "Prédiction": render_prediction_page,
    "Performance": render_performance_page,
    "Interprétabilité": render_interpretability_page,
    "Fairness": render_fairness_page,
}

PAGES[page]()