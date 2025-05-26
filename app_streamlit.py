import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
from folium.plugins import MiniMap


# --- Chargement des données ---
@st.cache_data
def load_data():
    return pd.read_csv("data_clean_spp_spv_with_type.csv")


df = load_data()

st.title("Application d'Analyse de la Condition Physique et de la Santé")

# --- SIDEBAR ---
st.sidebar.header("Filtres dynamiques")
cie = st.sidebar.multiselect("Cie:", df["Cie_x"].dropna().unique())
ut = st.sidebar.multiselect("UT:", df["UT_x"].dropna().unique())
aptitude = st.sidebar.multiselect(
    "Aptitude Générale:", df["Aptitude générale"].unique()
)
type_options = st.sidebar.multiselect("Type:", df["type"].dropna().unique())

age_min, age_max = st.sidebar.slider(
    "Tranche d'âge:", int(df["age"].min()), int(df["age"].max()), (20, 60)
)
imc_min, imc_max = st.sidebar.slider(
    "IMC:", float(df["imc"].min()), float(df["imc"].max()), (18.0, 30.0)
)
poids_min, poids_max = st.sidebar.slider(
    "Poids:", float(df["poids"].min()), float(df["poids"].max()), (67.0, 80.0)
)
palier_min, palier_max = st.sidebar.slider(
    "Luc Léger (valeur réelle):",
    int(df["luc_leger"].min()),
    int(df["luc_leger"].max()),
    (int(df["luc_leger"].min()), int(df["luc_leger"].max())),
)

# --- Application des filtres ---
df_filtered = df.copy()
if cie:
    df_filtered = df_filtered[df_filtered["Cie_x"].isin(cie)]
if ut:
    df_filtered = df_filtered[df_filtered["UT_x"].isin(ut)]
if aptitude:
    df_filtered = df_filtered[df_filtered["Aptitude générale"].isin(aptitude)]
if type_options:
    df_filtered = df_filtered[df_filtered["type"].isin(type_options)]

df_filtered = df_filtered[
    (df_filtered["age"] >= age_min) & (df_filtered["age"] <= age_max)
]
df_filtered = df_filtered[
    (df_filtered["imc"] >= imc_min) & (df_filtered["imc"] <= imc_max)
]
df_filtered = df_filtered[
    (df_filtered["poids"] >= poids_min) & (df_filtered["poids"] <= poids_max)
]
df_filtered = df_filtered[
    (df_filtered["luc_leger"] >= palier_min) & (df_filtered["luc_leger"] <= palier_max)
]

# --- VISUALISATIONS ---
st.subheader("Statistiques Globales sur les Données Filtrées")
st.write(f"Nombre d'individus: {df_filtered.shape[0]}")

features = ["imc", "taille", "poids", "tension_sys", "tension_dia"]
for feature in features:
    st.subheader(f"Distribution de {feature.upper()}")
    if feature in df_filtered.columns and not df_filtered[feature].dropna().empty:
        fig, ax = plt.subplots()
        sns.histplot(df_filtered[feature], kde=True, ax=ax)
        ax.set_title(f"Histogramme de {feature.upper()}")
        st.pyplot(fig)
    else:
        st.info(f"Aucune donnée disponible pour {feature.upper()}.")

phys_tests = ["luc_leger", "pompes", "tractions"]
for test in phys_tests:
    st.subheader(f"{test.replace('_', ' ').title()} par Cie")
    if not df_filtered.empty and test in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="Cie_x", y=test, data=df_filtered, ax=ax, palette="Set2")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    else:
        st.info(f"Aucune donnée disponible pour {test}.")

level_cols = {
    "luc_leger": " Palier Luc Léger",
    "Niveau pompes": "Niveau Pompes",
    "Niveau tractions": "Niveau Tractions",
}
for col, label in level_cols.items():
    if col in df_filtered.columns:
        st.subheader(f"{label} moyen par Cie")
        moyenne = df_filtered.groupby("Cie_x")[col].mean().sort_values(ascending=False)
        if not moyenne.empty:
            fig, ax = plt.subplots()
            colors = plt.cm.viridis(moyenne.rank() / moyenne.count())
            moyenne.plot(kind="bar", ax=ax, color=colors)
            ax.set_ylabel("Niveau moyen")
            ax.set_title(f"{label} moyen par Cie")
            st.pyplot(fig)

st.subheader("Analyse par Type")
if "type" in df_filtered.columns and not df_filtered["type"].dropna().empty:
    type_counts = df_filtered["type"].value_counts()
    fig, ax = plt.subplots()
    type_counts.plot(kind="bar", ax=ax)
    ax.set_title("Répartition des effectifs par type")
    ax.set_ylabel("Nombre d'individus")
    st.pyplot(fig)

    indicateurs_physiques = [
        "imc",
        "poids",
        "taille",
        "luc_leger",
        "pompes",
        "tractions",
    ]
    for indicateur in indicateurs_physiques:
        if indicateur in df_filtered.columns:
            st.subheader(f"{indicateur.upper()} par Type")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x="type", y=indicateur, data=df_filtered, ax=ax, palette="Set3")
            ax.set_title(f"Distribution de {indicateur.upper()} par Type")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

    if "luc_leger" in df_filtered.columns:
        st.subheader("Niveau moyen Luc Léger par Type")
        moyennes_type = (
            df_filtered.groupby("type")["luc_leger"].mean().sort_values(ascending=False)
        )
        fig, ax = plt.subplots()
        moyennes_type.plot(
            kind="bar",
            ax=ax,
            color=plt.cm.plasma(moyennes_type.rank() / len(moyennes_type)),
        )
        ax.set_ylabel("Palier moyen")
        ax.set_title("Luc Léger moyen par Type")
        st.pyplot(fig)
# --- NOUVELLES VISUALISATIONS : Luc Léger, Aptitude, Incendie/ARI ---

st.subheader("Luc Léger selon l'Aptitude Générale et l'Exposition Incendie")

# Histogramme Luc Léger par Incendie et port de l'ARI, coloré par aptitude
if (
    "luc_leger" in df_filtered.columns
    and "Aptitude générale" in df_filtered.columns
    and "Incendie et port de l'ARI Toutes missions_x" in df_filtered.columns
):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        df_filtered,
        x="luc_leger",
        hue="Aptitude générale",
        multiple="stack",
        bins=15,
        palette="Set2",
        kde=False,
    )
    ax.set_title("Répartition du palier Luc Léger selon l'aptitude générale")
    ax.set_xlabel("Palier Luc Léger")
    ax.set_ylabel("Nombre d'individus")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        df_filtered,
        x="luc_leger",
        hue="Incendie et port de l'ARI Toutes missions_x",
        multiple="stack",
        bins=15,
        palette="Set2",
        kde=False,
    )
    ax.set_title(
        "Répartition du palier Luc Léger selon Incendie et port de l'ARI Toutes missions"
    )
    ax.set_xlabel("Palier Luc Léger")
    ax.set_ylabel("Nombre d'individus")
    st.pyplot(fig)

# Histogramme Luc Léger par aptitude, coloré par type (SPP/SPV)
if "type" in df_filtered.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        df_filtered,
        x="luc_leger",
        hue="type",
        multiple="stack",
        bins=15,
        palette="Set3",
        kde=False,
    )
    ax.set_title("Répartition du palier Luc Léger selon l'aptitude générale (par type)")
    ax.set_xlabel("Palier Luc Léger")
    ax.set_ylabel("Nombre d'individus")
    st.pyplot(fig)


# Boxplot Luc Léger par aptitude et Incendie/ARI
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=df_filtered,
    x="Aptitude générale",
    y="luc_leger",
    hue="Incendie et port de l'ARI Toutes missions_x",
    palette="pastel",
)
ax.set_title("Luc Léger par Aptitude Générale et Incendie/ARI")
ax.set_ylabel("Palier Luc Léger")
ax.set_xlabel("Aptitude Générale")
ax.tick_params(axis="x", rotation=45)
st.pyplot(fig)

# --- Indicateurs IMC, poids, taille par aptitude ---
st.subheader("Indicateurs physiques par Aptitude Générale")

indicateurs = ["imc", "poids", "taille"]
for ind in indicateurs:
    if ind in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=df_filtered,
            x="Aptitude générale",
            y=ind,
            palette="coolwarm",
        )
        ax.set_title(f"{ind.upper()} par Aptitude Générale")
        ax.set_ylabel(ind.upper())
        ax.set_xlabel("Aptitude Générale")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
st.subheader("🔗 Corrélations avec le Palier Luc Léger")

# Sélection des colonnes numériques pertinentes
cols_corr = [
    "luc_leger",
    "imc",
    "poids",
    "taille",
    "tension_sys",
    "tension_dia",
    "pompes",
    "tractions",
    "Niveau Luc léger",
    "Niveau pompes",
    "Niveau tractions",
]

# Filtrage des colonnes existantes dans le dataframe filtré
cols_corr = [col for col in cols_corr if col in df_filtered.columns]
df_corr = df_filtered[cols_corr].dropna()

# Calcul de la matrice de corrélation
corr_matrix = df_corr.corr()

# Affichage d'une heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Matrice de Corrélation - Indicateurs Physiques et Luc Léger")
st.pyplot(fig)


# --- Carte interactive ---
st.subheader("Carte Interactive des UT")


@st.cache_data
def load_geojson():
    with open("alsace_map.geojson", "r", encoding="utf-8") as f:
        return json.load(f)


try:
    geojson_data = load_geojson()
    ut_mapping = {
        "UT STRASBOURG OUEST": "STRASBOURG-3",
        "UT HAGUENAU": "HAGUENAU",
        "UT MOLSHEIM": "MOLSHEIM",
        "UT INGWILLER": "INGWILLER",
        "UT OBERNAI": "OBERNAI",
        "UT LINGOLSHEIM": "LINGOLSHEIM",
        "UT BISCHWILLER": "BISCHWILLER",
        "UT SÉLESTAT": "SÉLESTAT",
        "UT STRASBOURG NORD": "STRASBOURG-3",
        "UT SAVERNE": "SAVERNE",
        "UT BRUMATH": "BRUMATH",
        "UT ERSTEIN": "ERSTEIN",
        "UT STRASBOURG FINKWI": "STRASBOURG-3",
        "UT WISSEMBOURG": "WISSEMBOURG",
        "UT STRASBOURG SUD": "STRASBOURG-3",
        "UT BOUXWILLER": "BOUXWILLER",
    }
    df_filtered["UT_clean"] = (
        df_filtered["UT_x"].astype(str).str.strip().str.upper().replace(ut_mapping)
    )
    luc_leger_moyen = df_filtered.groupby("UT_clean")["luc_leger"].mean().reset_index()
    luc_leger_moyen.columns = ["nom", "luc_leger_moyen"]
    effectif_ut = df_filtered["UT_clean"].value_counts().reset_index()
    effectif_ut.columns = ["nom", "effectif"]

    geo_features = [
        {**f["properties"], "geometry": f["geometry"]} for f in geojson_data["features"]
    ]
    geo_df = pd.DataFrame(geo_features)
    geo_df["nom"] = geo_df["nom"].str.strip().str.upper()

    geo_df = geo_df.merge(effectif_ut, on="nom", how="left")
    geo_df = geo_df.merge(luc_leger_moyen, on="nom", how="left")
    geo_df.fillna({"effectif": 0, "luc_leger_moyen": 0}, inplace=True)

    m = folium.Map(location=[48.6, 7.6], zoom_start=9, control_scale=True)
    colormap = cm.linear.YlOrRd_09.scale(
        geo_df["luc_leger_moyen"].min(), geo_df["luc_leger_moyen"].max()
    )
    colormap.caption = "Luc Léger moyen"
    colormap.add_to(m)

    folium.Choropleth(
        geo_data=geojson_data,
        data=geo_df,
        columns=["nom", "luc_leger_moyen"],
        key_on="feature.properties.nom",
        fill_color="YlGnBu",
        fill_opacity=0.6,
        line_opacity=0.5,
        legend_name="Luc Léger moyen",
    ).add_to(m)

    for _, row in geo_df.iterrows():
        if row["effectif"] > 0:
            geom = row["geometry"]
            coords = (
                geom["coordinates"][0]
                if geom["type"] == "Polygon"
                else geom["coordinates"][0][0]
            )
            lon = sum(pt[0] for pt in coords) / len(coords)
            lat = sum(pt[1] for pt in coords) / len(coords)
            folium.CircleMarker(
                location=(lat, lon),
                radius=8,
                color=colormap(row["luc_leger_moyen"]),
                fill=True,
                fill_color=colormap(row["luc_leger_moyen"]),
                fill_opacity=0.9,
                tooltip=folium.Tooltip(
                    f"""
<b>UT : {row['nom']}</b><br>
Effectif : {int(row['effectif'])}<br>
Luc Léger moyen : {row['luc_leger_moyen']:.2f}<br>
<hr style='margin: 4px 0;'>
<b>Filtres actifs</b><br>
- Âge : {age_min} - {age_max} ans<br>
- IMC : {imc_min:.1f} - {imc_max:.1f}<br>
- Poids : {poids_min:.1f} - {poids_max:.1f} kg<br>
- Luc Léger : {palier_min} - {palier_max}
""",
                    sticky=True,
                ),
            ).add_to(m)

    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, height=700)

except Exception as e:
    st.error(f"Erreur de chargement de la carte : {e}")


st.markdown(
    """
**Utilisation suggérée :**
- Comparez les performances physiques entre différentes compagnies
- Repérez les zones avec des tensions élevées ou IMC critiques
- Analysez la progression ou la performance moyenne par région ou groupe d'âge
- Identifiez les corrélations entre les indicateurs (ex : poids vs IMC, ou IMC vs Luc Léger)
- Visualisez clairement la répartition des niveaux de Luc Léger par unité ou compagnie
"""
)

if not df_filtered.empty:
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger les données filtrées (CSV)",
        data=csv,
        file_name="donnees_filtrees.csv",
        mime="text/csv",
    )
