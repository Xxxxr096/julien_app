import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import os


# --- Charger et préparer les données ---
@st.cache_data
def charger_modele():
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "merged_data_spv.csv")
    )
    df = pd.read_csv(data_path)

    for col in ["IMC", "Taille", "Poids"]:
        df[col] = df[col].str.replace(",", ".").astype(float)

    df = df[
        (df["Tension artérielle systol"] > 70)
        & (df["Tension artérielle systol"] < 200)
        & (df["Tension artérielle Diastol"] > 40)
        & (df["Tension artérielle Diastol"] < 130)
        & (df["IMC"] > 10)
        & (df["IMC"] < 60)
    ]

    df["Risque"] = df["Incendie et port de l'ARI Toutes missions_y"].apply(
        lambda x: 1 if x != "Apte" else 0
    )

    features = [
        "Age_x",
        "IMC",
        "Taille",
        "Poids",
        "Périmètre abdominal",
        "Tension artérielle systol",
        "Tension artérielle Diastol",
        "Luc léger",
        "Pompes",
        "Tractions",
    ]
    X = df[features]
    y = df["Risque"]

    # 🔧 Nettoyage des données avant apprentissage
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = y.fillna(0)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)
    return model


model = charger_modele()

# --- Interface utilisateur ---
st.title("🧠 Détection du Risque Santé / Blessure")

st.markdown(
    "**Remplis les informations ci-dessous pour estimer ton niveau de risque de blessure ou inaptitude.**"
)

# Entrée utilisateur
age = st.slider("Âge", 18, 70, 30)
imc = st.number_input("IMC", min_value=10.0, max_value=60.0, value=22.0)
taille = st.number_input("Taille (cm)", min_value=140.0, max_value=210.0, value=180.0)
poids = st.number_input("Poids (kg)", min_value=40.0, max_value=150.0, value=72.0)
abdomen = st.slider("Périmètre abdominal (cm)", 60, 140, 85)
systol = st.slider("Tension artérielle systolique", 90, 180, 120)
diastol = st.slider("Tension artérielle diastolique", 50, 120, 80)
luc = st.slider("Test Luc léger", 0, 15, 6)
pompes = st.slider("Nombre de pompes", 0, 60, 20)
tractions = st.slider("Nombre de tractions", 0, 30, 5)

if st.button("📊 Évaluer le Risque"):
    input_data = pd.DataFrame(
        [[age, imc, taille, poids, abdomen, systol, diastol, luc, pompes, tractions]],
        columns=[
            "Age_x",
            "IMC",
            "Taille",
            "Poids",
            "Périmètre abdominal",
            "Tension artérielle systol",
            "Tension artérielle Diastol",
            "Luc léger",
            "Pompes",
            "Tractions",
        ],
    )

    proba = model.predict_proba(input_data)[0][1]
    st.metric("🩺 Score de Risque", f"{proba * 100:.2f} %")

    if proba > 0.5:
        st.error("⚠️ Risque élevé détecté. Une attention médicale est recommandée.")
    elif proba > 0.3:
        st.warning("🟠 Risque modéré. Une surveillance pourrait être utile.")
    else:
        st.success("🟢 Risque faible. Continuez vos bonnes habitudes !")
