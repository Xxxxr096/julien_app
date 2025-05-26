import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
from folium.plugins import MiniMap


# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("data_clean_spp_spv.csv")


df = load_data()

st.title("Application d'Analyse de la Condition Physique et de la Santé")

# --- SIDEBAR ---
st.sidebar.header("Filtres dynamiques")

# Filtres disponibles
cie = st.sidebar.multiselect("Cie:", df["Cie_x"].dropna().unique())
ut = st.sidebar.multiselect("UT:", df["UT_x"].dropna().unique())
aptitude = st.sidebar.multiselect(
    "Aptitude Générale:", df["Aptitude générale"].unique()
)
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
    "Niveau Luc Léger:",
    int(df["Niveau Luc léger"].min()),
    int(df["Niveau Luc léger"].max()),
    (1, int(df["Niveau Luc léger"].max())),
)

# Application des filtres
df_filtered = df.copy()
if cie:
    df_filtered = df_filtered[df_filtered["Cie_x"].isin(cie)]
if ut:
    df_filtered = df_filtered[df_filtered["UT_x"].isin(ut)]
if aptitude:
    df_filtered = df_filtered[df_filtered["Aptitude générale"].isin(aptitude)]
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
    (df_filtered["Niveau Luc léger"] >= palier_min)
    & (df_filtered["Niveau Luc léger"] <= palier_max)
]


# --- VISUALISATIONS ---
st.subheader("Statistiques Globales sur les Données Filtrées")
st.write(f"Nombre d'individus: {df_filtered.shape[0]}")

# Histogrammes
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

# Boxplots par Cie
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

# Moyennes par Cie pour les niveaux

# Moyennes par Cie pour les niveaux
level_cols = {
    "Niveau Luc léger": "Palier Luc Léger",
    "Niveau pompes": "Palier Pompes",
    "Niveau tractions": "Palier Tractions",
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
        else:
            st.info(
                "Aucune donnée disponible pour ce graphique avec les filtres actuels."
            )


# Répartition des paliers Luc Léger par Cie avec couleurs
st.subheader("Répartition des Paliers Luc Léger par Cie")
palier_counts = (
    df_filtered.groupby(["Cie_x", "Niveau Luc léger"]).size().unstack(fill_value=0)
)
fig, ax = plt.subplots(figsize=(12, 6))
palier_counts.plot(kind="bar", stacked=True, colormap="tab20", ax=ax)
ax.set_title("Distribution des niveaux Luc Léger par Cie")
ax.set_ylabel("Nombre d'individus")
ax.legend(title="Palier")
st.pyplot(fig)

# Corrélations
st.subheader("Corrélations entre indicateurs physiques")
corr_cols = [
    "imc",
    "taille",
    "poids",
    "tension_sys",
    "tension_dia",
    "luc_leger",
    "pompes",
    "tractions",
]
if not df_filtered[corr_cols].dropna().empty:
    corr_matrix = df_filtered[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Matrice de corrélation")
    st.pyplot(fig)
else:
    st.info("Pas assez de données pour calculer les corrélations.")


st.subheader("Carte Interactive des UT filtrées")


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

    geo_features = [
        {**f["properties"], "geometry": f["geometry"]} for f in geojson_data["features"]
    ]
    geo_df = pd.DataFrame(geo_features)
    geo_df["nom"] = geo_df["nom"].str.strip().str.upper()

    luc_leger_moyen = (
        df_filtered.groupby("UT_clean")["Niveau Luc léger"].mean().reset_index()
    )
    luc_leger_moyen.columns = ["nom", "luc_leger_moyen"]
    charge_par_ut = df_filtered["UT_clean"].value_counts().reset_index()
    charge_par_ut.columns = ["nom", "effectif"]

    geo_df = geo_df.merge(charge_par_ut, on="nom", how="left")
    geo_df = geo_df.merge(luc_leger_moyen, on="nom", how="left")
    geo_df.fillna({"effectif": 0, "luc_leger_moyen": 0}, inplace=True)
    geo_df["taux_charge"] = (geo_df["effectif"] / geo_df["effectif"].sum() * 100).round(
        2
    )

    # --- Création de la carte ---
    m = folium.Map(location=[48.6, 7.6], zoom_start=9, control_scale=True)

    # ColorMap continue
    colormap = cm.linear.YlOrRd_09.scale(
        geo_df["luc_leger_moyen"].min(), geo_df["luc_leger_moyen"].max()
    )
    colormap.caption = "Palier moyen Luc Léger"
    colormap.add_to(m)

    # Choroplèthe
    folium.Choropleth(
        geo_data=geojson_data,
        data=geo_df,
        columns=["nom", "luc_leger_moyen"],
        key_on="feature.properties.nom",
        fill_color="YlGnBu",
        fill_opacity=0.6,
        line_opacity=0.5,
        legend_name="Palier moyen Luc Léger",
    ).add_to(m)

    # GeoJSON avec tooltip stylisé
    folium.GeoJson(
        geojson_data,
        name="Contours",
        style_function=lambda x: {"color": "black", "weight": 1.2, "fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["nom"],
            aliases=["UT:"],
            sticky=True,
            labels=True,
            style=(
                "background-color: white; color: #333; font-family: Arial; "
                "font-size: 12px; padding: 5px;"
            ),
        ),
    ).add_to(m)

    # Cercle + Tooltip dynamique
    for _, row in geo_df.iterrows():
        if row["effectif"] > 0:
            geom = row["geometry"]
            coords = (
                geom["coordinates"][0]
                if geom["type"] == "Polygon"
                else geom["coordinates"][0][0]
            )
            lon_center = sum(pt[0] for pt in coords) / len(coords)
            lat_center = sum(pt[1] for pt in coords) / len(coords)

            couleur = colormap(row["luc_leger_moyen"])

            tooltip_text = f"""
            <b>UT : {row['nom']}</b><br>
            Effectif : {int(row['effectif'])}<br>
            Taux de charge : {row['taux_charge']}%<br>
            Palier moyen : {row['luc_leger_moyen']:.2f}<br>
            <hr style='margin: 4px 0;'>
            <b>Filtres actifs</b><br>
            - Âge : {age_min}-{age_max} ans<br>
            - IMC : {imc_min:.1f}-{imc_max:.1f}<br>
            - Poids : {poids_min:.1f}-{poids_max:.1f} kg<br>
            - Palier : {palier_min}-{palier_max}
            """

            folium.CircleMarker(
                location=(lat_center, lon_center),
                radius=8,
                color=couleur,
                fill=True,
                fill_color=couleur,
                fill_opacity=0.9,
                tooltip=folium.Tooltip(tooltip_text, sticky=True),
            ).add_to(m)

    # Mini-carte en bas à droite
    MiniMap(toggle_display=True, position="bottomright").add_to(m)

    # Contrôle des couches
    folium.LayerControl().add_to(m)

    st_folium(m, use_container_width=True, height=750)

except Exception as e:
    st.error(f"Erreur lors du chargement de la carte : {e}")

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
