"""
Application Streamlit — Prédiction du risque d'AVC
====================================================

Cette application charge le modèle entraîné (LogisticRegression optimisée pour le F2)
ainsi que les objets de prétraitement associés (Winsorizer, SMOTEENN, etc.) issus
du notebook `stroke_ml_pipeline_v2.ipynb`, et permet d'estimer le risque d'AVC d'un patient
à partir de son profil médical et de ses habitudes de vie.

Pour lancer l'application :
    streamlit run app.py

Arborescence attendue :
    .
    ├── app.py
    └── model_artifacts/
        ├── modele_final.joblib
        └── metadata.json
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_antd_components as sac  # Pour les icônes Material Design dans les encadrés
from sklearn.base import BaseEstimator, TransformerMixin

# ============================================================
# CLASSE PERSONNALISÉE (OBLIGATOIRE POUR CHARGER LE MODÈLE)
# ============================================================
class Winsorizer(BaseEstimator, TransformerMixin):
    """Winsorise chaque colonne selon des bornes de percentile apprises sur le train (fit),
    puis appliquées telles quelles au transform — aucune statistique n'est recalculée sur le test."""

    def __init__(self, limits=(0.01, 0.01)):
        self.limits = limits

    def fit(self, x, y=None):
        X = np.asarray(x, dtype=float)
        low, high = self.limits
        self.lower_ = np.nanquantile(X, low, axis=0)
        self.upper_ = np.nanquantile(X, 1 - high, axis=0)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        return self

    def transform(self, x):
        X = np.asarray(x, dtype=float).copy()
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


# ============================================================
# Configuration générale de la page
# ============================================================

# Chemin dynamique (portable pour Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOSSIER_ARTEFACTS = os.path.join(BASE_DIR, "model_artifacts")

st.set_page_config(
    page_title="Prédiction du risque d'AVC",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e0.svg", # Icône cerveau en ligne
    layout="centered",
)

# Injection du CSS pour la police Material Symbols (pour les icônes dans les titres HTML)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
<style>
.material-symbols-outlined {
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
  vertical-align: middle;
  font-size: 1.2em;
}
</style>
""", unsafe_allow_html=True)

LIBELLES_CLASSES = {0: "Pas d'AVC", 1: "AVC"}
COULEURS_CLASSES = {0: "#2ecc71", 1: "#e74c3c"}


# ============================================================
# Chargement des artefacts (mis en cache pour éviter de recharger à chaque interaction)
# ============================================================
@st.cache_resource
def charger_artefacts():
    """Charge le pipeline complet et ses métadonnées."""
    # 1. Charger le pipeline complet (Winsorizer + Imputer + Scaler + SMOTEENN + Modèle)
    chemin_pipeline = os.path.join(DOSSIER_ARTEFACTS, "modele_final.joblib")
    if not os.path.exists(chemin_pipeline):
        raise FileNotFoundError(
            f"Fichier manquant : {chemin_pipeline}. "
            f"Assurez-vous que le dossier 'model_artifacts' est bien présent à côté de app.py."
        )
    pipeline = joblib.load(chemin_pipeline)

    # 2. Charger les métadonnées (si présentes)
    metadonnees = {}
    chemin_metadata = os.path.join(DOSSIER_ARTEFACTS, "metadata.json")
    if os.path.exists(chemin_metadata):
        with open(chemin_metadata, "r", encoding="utf-8") as f:
            metadonnees = json.load(f)

    return pipeline, metadonnees


try:
    PIPELINE, METADONNEES = charger_artefacts()
    artefacts_charges = True
except FileNotFoundError as erreur:
    artefacts_charges = False
    st.error(str(erreur))


# ============================================================
# En-tête de l'application
# ============================================================
# Titre principal avec icône Material Design (Style Churn)
sac.alert(
    label="Prédiction du risque d'AVC",
    description="Modèle LogisticRegression optimisé pour le F2, avec rééquilibrage SMOTEENN.",
    color="blue",
    icon="psychology"
)

st.markdown(
    """
Cette application utilise un modèle de **Machine Learning** (LogisticRegression, optimisée par
recherche sur grille et rééquilibrage SMOTEENN) entraîné sur le jeu de données
*Healthcare Stroke Dataset* pour estimer le risque d'AVC d'un patient à partir de son
profil médical et de ses habitudes de vie.
"""
)

if not artefacts_charges:
    st.stop()

# Message d'avertissement basé sur VOTRE notebook (Métriques réelles de votre modèle)
sac.alert(
    label="Modèle en phase de test",
    description="Un rappel de **0.84** (84% des AVC réels détectés) mais une précision de **0.12** (12% des alertes correctes). Utilisez le seuil pour ajuster le compromis.",
    color="warning",
    icon="warning"
)

st.divider()

# ============================================================
# Formulaire de saisie
# ============================================================
with st.form("formulaire_prediction"):
    
    # Section Profil du patient (Style Churn)
    sac.alert(
        label="Profil du patient",
        color="gray",
        icon="person"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        genre = st.selectbox(
            "Genre", ["Female", "Male"],
            format_func=lambda x: "Femme" if x == "Female" else "Homme",
        )
        age = st.number_input("Âge (années)", min_value=0, max_value=120, value=45)
        marie = st.selectbox(
            "Marié(e) (ou l'a déjà été)", ["Yes", "No"],
            format_func=lambda x: "Oui" if x == "Yes" else "Non",
        )
    with col2:
        residence = st.selectbox(
            "Type de résidence", ["Urban", "Rural"],
            format_func=lambda x: "Urbaine" if x == "Urban" else "Rurale",
        )
        work_type = st.selectbox(
            "Type d'activité professionnelle",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
            format_func=lambda x: {
                "Private": "Secteur privé",
                "Self-employed": "Indépendant",
                "Govt_job": "Fonction publique",
                "children": "Enfant (non actif)",
                "Never_worked": "N'a jamais travaillé",
            }[x],
        )

    # Section Indicateurs médicaux
    sac.alert(
        label="Indicateurs médicaux",
        color="gray",
        icon="stethoscope"
    )
    
    col3, col4 = st.columns(2)
    with col3:
        hypertension = st.selectbox(
            "Hypertension", [0, 1],
            format_func=lambda x: "Oui" if x == 1 else "Non",
        )
        heart_disease = st.selectbox(
            "Maladie cardiaque", [0, 1],
            format_func=lambda x: "Oui" if x == 1 else "Non",
        )
    with col4:
        avg_glucose_level = st.number_input(
            "Glycémie moyenne (mg/dL)", min_value=40.0, max_value=300.0, value=100.0, step=0.5
        )
        bmi = st.number_input("IMC (BMI)", min_value=10.0, max_value=80.0, value=25.0, step=0.1)

    # Section Habitudes de vie
    sac.alert(
        label="Habitudes de vie",
        color="gray",
        icon="smoking_rooms"
    )
    
    smoking_status = st.selectbox(
        "Statut tabagique",
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
        format_func=lambda x: {
            "never smoked": "N'a jamais fumé",
            "formerly smoked": "Ancien fumeur",
            "smokes": "Fumeur actuel",
            "Unknown": "Inconnu",
        }[x],
    )

    bouton_predire = st.form_submit_button(
        "Estimer le risque d'AVC", 
        icon=":material/monitor_heart:",
        use_container_width=True
    )


# ============================================================
# Prédiction et affichage des résultats
# ============================================================
if bouton_predire:
    reponses = {
        "gender": genre,
        "age": age,
        "ever_married": marie,
        "Residence_type": residence,
        "work_type": work_type,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }

    # 1. Construire le DataFrame (le pipeline gère le reste)
    observation_preparee = preparer_observation(reponses)

    # 2. Appliquer le pipeline complet (il fera l'imputation, le scaling, etc., puis la prédiction)
    proba_avc = PIPELINE.predict_proba(observation_preparee)[0][1]

    st.divider()
    
    # Section Résultats (Style Churn)
    sac.alert(
        label="Résultat de la prédiction",
        color="blue",
        icon="monitoring"
    )

    seuil = st.slider(
        "Seuil de décision (probabilité à partir de laquelle on classe le patient à risque)",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="Un seuil plus bas augmente la sensibilité (plus d'AVC détectés) mais aussi les fausses alertes.",
    )
    classe_predite = 1 if proba_avc >= seuil else 0

    couleur = COULEURS_CLASSES[classe_predite]
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {couleur}22;
                    border: 2px solid {couleur};">
            <h3 style="color: {couleur}; margin: 0;">
                {LIBELLES_CLASSES[classe_predite]}
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    col_a.metric("Probabilité estimée d'AVC", f"{proba_avc * 100:.1f} %")
    col_b.metric("Seuil appliqué", f"{seuil * 100:.0f} %")

    st.markdown("#### Probabilité d'AVC")
    fig = go.Figure(
        go.Bar(
            x=[proba_avc, 1 - proba_avc],
            y=["AVC", "Pas d'AVC"],
            orientation="h",
            marker_color=[COULEURS_CLASSES[1], COULEURS_CLASSES[0]],
            text=[f"{proba_avc * 100:.1f} %", f"{(1 - proba_avc) * 100:.1f} %"],
            textposition="auto",
        )
    )
    fig.update_layout(
        xaxis_title="Probabilité",
        yaxis_title="",
        xaxis_range=[0, 1],
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Cette prédiction est fournie à titre indicatif et s'appuie sur un modèle statistique "
        "entraîné sur un jeu de données spécifique. Elle ne remplace en aucun cas un avis médical "
        "professionnel."
    )

st.divider()
with st.expander("À propos du modèle"):
    modele_nom = METADONNEES.get("modele", type(PIPELINE).__name__)
    metriques = METADONNEES.get("metriques_test", {})
    
    st.markdown(
        f"""
- **Modèle utilisé** : `{modele_nom}` (optimisé par GridSearchCV, données rééquilibrées via SMOTEENN)
- **Métrique d'optimisation** : F2-score (beta=2) — priorité à la détection des cas d'AVC réels
- **F2-score (validation croisée)** : {METADONNEES.get('f2_cv', 'N/A')}
- **Performances mesurées (seuil 0.5)** :
"""
    )
    if metriques:
        cols = st.columns(len(metriques))
        for col, (k, v) in zip(cols, metriques.items()):
            col.metric(k, f"{v:.4f}")
    
    st.markdown(
        f"""
- **Source du pipeline** : notebook `stroke_ml_pipeline_v2.ipynb`
"""
    )