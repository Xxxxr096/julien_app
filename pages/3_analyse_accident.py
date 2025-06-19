import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import chardet
import os

# Titre
st.title("Analyse de l'accidentologie")


# Fonction de chargement des données avec détection automatique de l'encodage
def load_data():
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "accidentologie.csv")
    )

    # Détection de l'encodage
    with open(data_path, "rb") as f:
        result = chardet.detect(f.read())
    encoding_detected = result["encoding"]

    # Lecture du fichier avec l'encodage détecté et le séparateur correct
    df = pd.read_csv(data_path, sep=";", encoding=encoding_detected)
    return df


# Lecture des données
data = load_data()

# Nettoyage des données
data.columns = data.columns.str.replace("*", "", regex=False).str.strip()
data.drop(columns=["Agent"], inplace=True)
data["Date de l'accident"] = pd.to_datetime(
    data["Date de l'accident"], errors="coerce", dayfirst=True
)
data["Année"] = data["Date de l'accident"].dt.year
data["Mois"] = data["Date de l'accident"].dt.month
data["Jour"] = data["Date de l'accident"].dt.day
data["Jour_semaine"] = data["Date de l'accident"].dt.day_name()
data["Durée totale arrêt"] = pd.to_numeric(data["Durée totale arrêt"], errors="coerce")

# Affichage du tableau
st.subheader("Aperçu des données")
st.dataframe(data.head())

# Graphique: accidents par année
st.subheader("Nombre d'accidents par année")
fig1, ax1 = plt.subplots()
data["Année"].value_counts().sort_index().plot(kind="bar", ax=ax1)
st.pyplot(fig1)

# Graphique: accidents par jour de la semaine
st.subheader("Nombre d'accidents par jour de la semaine")
fig2, ax2 = plt.subplots()
data["Jour_semaine"].value_counts().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
).plot(kind="bar", ax=ax2)
st.pyplot(fig2)

# Statistiques durée arrêt
st.subheader("Statistiques sur la durée totale d'arrêt")
st.write(data["Durée totale arrêt"].describe())
