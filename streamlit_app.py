"""
Application Streamlit — Prédiction du churn client (Telco)
------------------------------------------------------------
Charge le pipeline entraîné (prétraitement + rééquilibrage + modèle,
sauvegardé via joblib) et ses métadonnées (seuil de décision, métriques),
puis permet de scorer un client saisi manuellement ou un fichier CSV
de plusieurs clients.

Arborescence attendue (celle produite par le notebook) :
    model_artifacts/
        Churn_model_pipeline.joblib
        Churn_model_metadata.json
    streamlit_app.py   <- ce fichier

Lancement :
    streamlit run streamlit_app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_antd_components as sac  # <--- NOUVEAU
from sklearn.base import BaseEstimator, TransformerMixin


# ----------------------------------------------------------------------
# IMPORTANT : le pipeline sauvegardé (.joblib) contient un transformer
# personnalisé (Winsorizer) défini dans le notebook d'entraînement.
# joblib/pickle a besoin de retrouver cette classe, identique, dans
# l'environnement qui charge le fichier — sinon `joblib.load()` échoue
# avec une AttributeError ("Can't get attribute 'Winsorizer'").
# Cette définition doit donc rester strictement synchronisée avec celle
# du notebook si jamais la classe y est modifiée.
# ----------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Winsorise chaque colonne selon des bornes de percentile apprises sur le train (fit),
    puis appliquées telles quelles au transform — aucune statistique n'est recalculée sur le test."""

    def __init__(self, limits=(0.01, 0.01)):
        self.limits = limits

    def fit(self, x, y=None):
        X = np.asarray(x)
        low, high = self.limits
        self.lower_ = np.nanquantile(x, low, axis=0)
        self.upper_ = np.nanquantile(x, 1 - high, axis=0)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        return self

    def transform(self, x):
        return np.clip(x, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

# ----------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Prédiction du churn client — Telco",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c9.svg", # Icône graphique en baisse
    layout="centered",
)

ARTIFACTS_DIR = Path(__file__).parent / "model_artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "Churn_model_pipeline.joblib"
METADATA_PATH = ARTIFACTS_DIR / "Churn_model_metadata.json"

# Valeurs catégorielles exactes du dataset d'entraînement (Telco-Customer-Churn)
OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
    "StreamingMovies": ["No", "No internet service", "Yes"],
}

# Colonnes attendues par le pipeline, dans un ordre quelconque (le ColumnTransformer
# sélectionne par nom, pas par position) — sert à construire le DataFrame d'entrée.
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


# ----------------------------------------------------------------------
# Chargement du pipeline et des métadonnées (mis en cache)
# ----------------------------------------------------------------------
@st.cache_resource
def charger_pipeline():
    if not PIPELINE_PATH.exists():
        return None
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def charger_metadata():
    if not METADATA_PATH.exists():
        return None
    # CORRECTION : lecture explicitement en UTF-8, avec repli si le fichier a été
    # écrit sans encodage explicite sur un système Windows (cp1252 par défaut) —
    # cas typique : "hypothèses" -> octet 0xe9 invalide en UTF-8 strict.
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        st.warning(
            "`Churn_model_metadata.json` ne semble pas encodé en UTF-8 (probable "
            "artefact Windows). Lecture de secours en cp1252 — régénérez ce fichier "
            "depuis le notebook (cellule de sauvegarde corrigée) pour éviter ce message."
        )
        with open(METADATA_PATH, "r", encoding="cp1252") as f:
            return json.load(f)


pipeline = charger_pipeline()
metadata = charger_metadata()

# ----------------------------------------------------------------------
# En-tête avec icône Material
# ----------------------------------------------------------------------
sac.alert(
    label="Prédiction du churn client",
    description="Estime la probabilité qu'un client résilie son abonnement, à partir du pipeline entraîné (prétraitement + rééquilibrage SMOTEENN + modèle).",
    color="blue",
    icon=True
)

if pipeline is None or metadata is None:
    st.error(
        "Artefacts introuvables. Placez `Churn_model_pipeline.joblib` et "
        "`Churn_model_metadata.json` dans un dossier `model_artifacts/` à côté de ce script "
        "(c'est l'emplacement où le notebook les sauvegarde)."
    )
    st.stop()

seuil_defaut = metadata.get("seuil_decision", 0.5)

# ----------------------------------------------------------------------
# Section modèle chargé avec icône
# ----------------------------------------------------------------------
with st.expander("À propos du modèle chargé"):
    st.write(f"**Modèle** : {metadata.get('modele', 'inconnu')}")
    st.write(f"**Rééquilibrage** : {metadata.get('resampling', 'inconnu')}")
    metriques = metadata.get("metriques_test", {})
    if metriques:
        cols = st.columns(len(metriques))
        for col, (k, v) in zip(cols, metriques.items()):
            col.metric(k, f"{v:.3f}" if isinstance(v, float) else str(v))
    st.caption(f"Seuil de décision par défaut (optimisé F2) : {seuil_defaut:.2f}")

st.divider()

# ----------------------------------------------------------------------
# Section Seuil de décision avec icône Material (⚙️ remplacé)
# ----------------------------------------------------------------------
# sac.alert(
#     label="Seuil de décision",
#     description="Ajustez le seuil au-delà duquel un client est classé à risque.",
#     color="gray",
#     icon="gear"
# )

# seuil = st.slider(
#     "Seuil au-delà duquel un client est classé « à risque de churn »",
#     min_value=0.0, max_value=1.0,
#     value=float(seuil_defaut), step=0.01,
#     help=(
#         "Le pipeline seul (.predict()) utiliserait 0.5 par défaut. Le seuil optimisé "
#         f"lors de l'entraînement est {seuil_defaut:.2f} (métrique F2)."
#     ),
# )

# st.divider()

# ----------------------------------------------------------------------
# Deux modes : saisie manuelle d'un client, ou import CSV pour plusieurs clients
# ----------------------------------------------------------------------
mode = st.radio("Mode", ["Client unique (formulaire)", "Plusieurs clients (fichier CSV)"], horizontal=True)


def predire(df_clients: pd.DataFrame, seuil: float) -> pd.DataFrame:
    """Applique le pipeline et retourne le DataFrame enrichi des prédictions."""
    proba = pipeline.predict_proba(df_clients)[:, 1]
    resultat = df_clients.copy()
    resultat["probabilite_churn"] = proba
    resultat["prediction"] = (proba >= seuil).astype(int)
    resultat["prediction_label"] = resultat["prediction"].map({0: "Reste", 1: "Churn (à risque)"})
    return resultat


# ----------------------------------------------------------------------
# MODE 1 : formulaire pour un client unique
# ----------------------------------------------------------------------
if mode == "Client unique (formulaire)":
    # Remplacement de l'icône 👤 par une icône Material
    sac.alert(
        label="Informations du client",
        description="Renseignez les caractéristiques du client.",
        color="blue",
        icon="person"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Genre", OPTIONS["gender"])
        senior = st.selectbox("Senior (65 ans ou +)", ["Non", "Oui"])
        partner = st.selectbox("En couple (Partner)", OPTIONS["Partner"])
        dependents = st.selectbox("Personnes à charge (Dependents)", OPTIONS["Dependents"])
        tenure = st.number_input("Ancienneté (mois)", min_value=0, max_value=100, value=12)

    with col2:
        phone_service = st.selectbox("Service téléphonique", OPTIONS["PhoneService"])
        multiple_lines = st.selectbox("Lignes multiples", OPTIONS["MultipleLines"])
        internet_service = st.selectbox("Service internet", OPTIONS["InternetService"])
        online_security = st.selectbox("Sécurité en ligne", OPTIONS["OnlineSecurity"])
        online_backup = st.selectbox("Sauvegarde en ligne", OPTIONS["OnlineBackup"])
        device_protection = st.selectbox("Protection des appareils", OPTIONS["DeviceProtection"])

    with col3:
        tech_support = st.selectbox("Support technique", OPTIONS["TechSupport"])
        streaming_tv = st.selectbox("Streaming TV", OPTIONS["StreamingTV"])
        streaming_movies = st.selectbox("Streaming Films", OPTIONS["StreamingMovies"])
        contract = st.selectbox("Type de contrat", OPTIONS["Contract"])
        paperless_billing = st.selectbox("Facturation dématérialisée", OPTIONS["PaperlessBilling"])
        payment_method = st.selectbox("Moyen de paiement", OPTIONS["PaymentMethod"])

    st.markdown("**Facturation**")
    col4, col5 = st.columns(2)
    with col4:
        monthly_charges = st.number_input("Charges mensuelles ($)", min_value=0.0, value=70.0, step=0.5)
    with col5:
        total_charges = st.number_input(
            "Total facturé à date ($)", min_value=0.0,
            value=float(monthly_charges * max(tenure, 1)), step=1.0,
        )

    # Bouton avec icône Material native Streamlit
    if st.button("Prédire", icon=":material/search:", type="primary"):
        client = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Oui" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }])[FEATURE_COLUMNS]

        resultat = predire(client, seuil)
        proba = resultat.loc[0, "probabilite_churn"]
        est_churn = resultat.loc[0, "prediction"] == 1

        st.divider()
        if est_churn:
            st.error(f"Client à risque de churn — probabilité estimée : **{proba:.1%}**")
        else:
            st.success(f"Client jugé stable — probabilité de churn estimée : **{proba:.1%}**")

        st.progress(min(float(proba), 1.0))
        st.caption(f"Seuil de décision utilisé : {seuil:.2f}")

# ----------------------------------------------------------------------
# MODE 2 : import CSV pour scorer plusieurs clients d'un coup
# ----------------------------------------------------------------------
else:
    # Remplacement de l'icône 📄 par une icône Material
    sac.alert(
        label="Import d'un fichier CSV",
        description="Le fichier doit contenir les colonnes suivantes (mêmes noms que le dataset d'entraînement) : " + ", ".join(FEATURE_COLUMNS),
        color="green",
        icon="upload_file"
    )

    fichier = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if fichier is not None:
        df_import = pd.read_csv(fichier)

        colonnes_manquantes = [c for c in FEATURE_COLUMNS if c not in df_import.columns]
        if colonnes_manquantes:
            st.error(f"Colonnes manquantes dans le fichier : {colonnes_manquantes}")
        else:
            # TotalCharges peut contenir des espaces/valeurs vides comme dans le dataset original
            df_import["TotalCharges"] = pd.to_numeric(df_import["TotalCharges"], errors="coerce")
            lignes_invalides = df_import["TotalCharges"].isna().sum()
            if lignes_invalides:
                st.warning(f"{lignes_invalides} ligne(s) avec TotalCharges invalide seront ignorées.")
                df_import = df_import.dropna(subset=["TotalCharges"])

            # Bouton avec icône Material native Streamlit
            if st.button("Scorer tous les clients", icon=":material/query_stats:", type="primary"):
                resultat = predire(df_import[FEATURE_COLUMNS], seuil)

                nb_churn = int(resultat["prediction"].sum())
                nb_total = len(resultat)
                col1, col2, col3 = st.columns(3)
                col1.metric("Clients analysés", nb_total)
                col2.metric("Clients à risque", nb_churn)
                col3.metric("Taux à risque", f"{nb_churn / nb_total:.1%}" if nb_total else "—")

                st.divider()
                st.dataframe(
                    resultat[["probabilite_churn", "prediction_label"] + FEATURE_COLUMNS]
                    .sort_values("probabilite_churn", ascending=False)
                    .reset_index(drop=True),
                    use_container_width=True,
                )

                csv_export = resultat.to_csv(index=False).encode("utf-8")
                # Bouton de téléchargement avec icône Material native
                st.download_button(
                    "Télécharger les résultats (CSV)",
                    data=csv_export,
                    file_name="predictions_churn.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )

st.divider()
st.caption(
    "Le seuil par défaut (F2) privilégie le rappel : il génère volontairement plus de "
    "faux positifs pour rater le moins de churners possible. Ajustez le seuil selon la "
    "capacité réelle de contact de l'équipe de rétention."
)