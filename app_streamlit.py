import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
from folium.plugins import MiniMap
import numpy as np
from streamlit_folium import folium_static


# --- Chargement des données ---
@st.cache_data
def load_data():
    df = pd.read_csv("merged_data_spv.csv")

    # Standardiser les noms de colonnes : minuscules, sans espace
    df.columns = df.columns.str.strip().str.lower()

    # Colonnes à convertir de 'xx,yy' → float
    cols_to_fix = [
        "poids",
        "taille",
        "imc",
        "luc léger",
        "tension artérielle systol",
        "tension artérielle diastol",
    ]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    return df


df = load_data()
df.columns = df.columns.str.strip().str.lower()

# Ajoute dans le chargement si ce n’est pas fait :
if "périmètre abdominal" in df.columns:
    df["périmètre abdominal"] = (
        df["périmètre abdominal"].astype(str).str.replace(",", ".").astype(float)
    )
df["taille"] = df["taille"].astype(str).str.replace(",", ".").astype(float)

# Remplacer les valeurs aberrantes (0 ou > 250 cm) par NaN
df.loc[(df["taille"] <= 100) | (df["taille"] > 250), "taille"] = None

# Convertir les tailles en mètres (optionnel, selon ton besoin)
df["taille"] = df["taille"] / 100

st.title("Application d'Analyse de la Condition Physique et de la Santé")
with st.expander("📘 Guide d'utilisation de l'application", expanded=False):
    st.markdown(
        """
### 🧭 Guide d'utilisation

Bienvenue dans l'application d'analyse de la condition physique et de la santé.

---

#### 🔍 1. Filtres (colonne de gauche)
Utilisez les filtres pour explorer les données :
- **Cie / UT** : sélectionnez une ou plusieurs compagnies ou unités territoriales.
- **Sexe, Aptitude générale** : filtrez selon les caractéristiques individuelles.
- **Âge, IMC, Poids** : ciblez des groupes spécifiques.
- **Luc Léger - Paliers** : filtrez par niveau d’endurance.

Tous les graphiques et la carte s’adaptent automatiquement aux filtres choisis.

---

#### 📊 2. Visualisations
Plusieurs types de graphiques sont proposés :
- **Histogrammes** : IMC, taille, poids.
- **Boxplots** : luc léger, pompes, tractions par compagnie ou aptitude.
- **Graphiques empilés** : analyse croisée IMC / luc léger ou tour de taille / sexe.
- **Matrice de corrélation** : explorez les relations entre les variables.

---

#### 🗺️ 3. Carte Interactive
La carte affiche l’**IMC moyen** par UT, en fonction des filtres appliqués.

⚠️ **À noter** :
- La carte peut prendre quelques secondes à se recharger après un déplacement, un zoom ou un changement de filtre.
- Cela est dû au recalcul dynamique des données.
- En cas de blocage ou lenteur :
- Essayez de **rafraîchir la page** du navigateur.

---

#### 💾 4. Export
Vous pouvez télécharger les **données filtrées au format CSV** en bas de la page.

---

#### 🆘 En cas de problème
- Vérifiez les filtres sélectionnés (trop restrictifs = données vides).
- Si un graphique ou la carte ne s'affiche pas, utilisez le bouton de **rafraîchissement du navigateur**.


---
    """
    )

# --- SIDEBAR ---
st.sidebar.header("Filtres dynamiques")
cie = st.sidebar.multiselect("Cie:", df["cie_x"].dropna().unique())
ut = st.sidebar.multiselect("UT:", df["ut_x"].dropna().unique())
aptitude = st.sidebar.multiselect(
    "Aptitude Générale:", df["aptitude générale"].unique()
)

sexe_options = st.sidebar.multiselect(
    "sexe :", df["sexe"].dropna().unique(), default=df["sexe"].dropna().unique()
)


st.sidebar.markdown("**Age - Catégories**")
age_category = st.sidebar.multiselect(
    "Selectionnez une catégorie d'Age : ",
    [
        "Tous",
        "16 à 29",
        "30 à 39",
        "40 à 49",
        "50 à 57",
        "plus de 57",
    ],
)
st.sidebar.markdown("**imc - Catégories**")
imc_category = st.sidebar.multiselect(
    "Sélectionnez une catégorie d'imc :",
    [
        "Tous",
        "Normal (18.5 - 24.9)",
        "Surpoids (25.0 - 29.9)",
        "Obésité modérée (30.0 - 34.9)",
        "Obésité sévère (35.0 - 39.9)",
        "Obésité massive (>40)",
    ],
)
poids_min, poids_max = st.sidebar.slider(
    "poids:", float(df["poids"].min()), float(df["poids"].max()), (0.0, 144.0)
)

# --- Application des filtres ---
df_filtered = df.copy()
if cie:
    df_filtered = df_filtered[df_filtered["cie_x"].isin(cie)]
if ut:
    df_filtered = df_filtered[df_filtered["ut_x"].isin(ut)]
if aptitude:
    df_filtered = df_filtered[df_filtered["aptitude générale"].isin(aptitude)]

if age_category:
    filtres_age = []
    for cat in age_category:
        if cat == "16 à 29":
            filtres_age.append(
                (df_filtered["age_x"] >= 16) & (df_filtered["age_x"] <= 29)
            )
        elif cat == "30 à 39":
            filtres_age.append(
                (df_filtered["age_x"] >= 30) & (df_filtered["age_x"] <= 39)
            )
        elif cat == "40 à 49":
            filtres_age.append(
                (df_filtered["age_x"] >= 40) & (df_filtered["age_x"] <= 49)
            )
        elif cat == "50 à 57":
            filtres_age.append(
                (df_filtered["age_x"] >= 50) & (df_filtered["age_x"] <= 57)
            )
        elif cat == "Plus de 57":
            filtres_age.append(df_filtered["age_x"] > 57)

    if filtres_age:
        df_filtered = df_filtered[pd.concat(filtres_age, axis=1).any(axis=1)]


df_filtered = df_filtered[
    (df_filtered["poids"] >= poids_min) & (df_filtered["poids"] <= poids_max)
]

st.sidebar.markdown("**Luc Léger - Paliers**")
luc_leger_categories = st.sidebar.multiselect(
    "Sélectionnez une ou plusieurs catégories de palier Luc Léger :",
    ["1", "2", "3", "4", "5", "plus de 6"],
)

if sexe_options:
    df_filtered = df_filtered[df_filtered["sexe"].isin(sexe_options)]


# Application du filtre imc par classe
if imc_category:
    filtres_imc = []
    for cat in imc_category:
        if "Normal" in cat:
            filtres_imc.append(
                (df_filtered["imc"] >= 18.5) & (df_filtered["imc"] <= 24.9)
            )
        elif "Surpoids" in cat:
            filtres_imc.append(
                (df_filtered["imc"] >= 25.0) & (df_filtered["imc"] <= 29.9)
            )
        elif "modérée" in cat:
            filtres_imc.append(
                (df_filtered["imc"] >= 30.0) & (df_filtered["imc"] <= 34.9)
            )
        elif "sévère" in cat:
            filtres_imc.append(
                (df_filtered["imc"] >= 35.0) & (df_filtered["imc"] <= 39.9)
            )
        elif "massive" in cat:
            filtres_imc.append(df_filtered["imc"] >= 40.0)

    if filtres_imc:
        df_filtered = df_filtered[pd.concat(filtres_imc, axis=1).any(axis=1)]

if luc_leger_categories:
    filtres_luc = []
    for cat in luc_leger_categories:
        if cat == "1":
            filtres_luc.append(df_filtered["luc léger"] == 1)
        elif cat == "2":
            filtres_luc.append(df_filtered["luc léger"] == 2)
        elif cat == "3":
            filtres_luc.append(df_filtered["luc léger"] == 3)
        elif cat == "4":
            filtres_luc.append(df_filtered["luc léger"] == 4)
        elif cat == "5":
            filtres_luc.append(df_filtered["luc léger"] == 5)
        elif cat == "plus de 6":
            filtres_luc.append(df_filtered["luc léger"] >= 6)

    if filtres_luc:
        df_filtered = df_filtered[pd.concat(filtres_luc, axis=1).any(axis=1)]

# --- VISUALISATIONS ---
st.subheader("Statistiques Globales sur les Données Filtrées")
st.write(f"Nombre d'individus: {df_filtered.shape[0]}")


st.subheader("Distribution de l’imc empilée selon le niveau luc léger")

if "imc" in df_filtered.columns and "niveau luc léger" in df_filtered.columns:
    df_imc = df_filtered[["imc", "niveau luc léger"]].dropna()

    # Définir les bins
    bins = np.histogram_bin_edges(df_imc["imc"], bins=20)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Initialiser les comptages pour chaque niveau
    niveaux = [1, 2, 3]
    couleurs = {1: "red", 2: "orange", 3: "green"}
    bar_data = {
        niv: np.histogram(df_imc[df_imc["niveau luc léger"] == niv]["imc"], bins=bins)[
            0
        ]
        for niv in niveaux
    }

    # Créer le graphique empilé
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros_like(bin_centers)
    for niv in niveaux:
        ax.bar(
            bin_centers,
            bar_data[niv],
            width=np.diff(bins),
            bottom=bottom,
            color=couleurs[niv],
            edgecolor="black",
            label=f"Niveau {niv}",
        )
        bottom += bar_data[niv]

    ax.set_title("Distribution empilée de l’imc par niveau luc léger")
    ax.set_xlabel("imc")
    ax.set_ylabel("Nombre d’individus")
    ax.legend(title="niveau luc léger")
    st.pyplot(fig)
else:
    st.info(
        "Les données nécessaires pour afficher cette visualisation sont incomplètes."
    )

st.subheader("Distribution du Palier Luc Léger par Catégorie d'IMC")

if "luc léger" in df_filtered.columns and "imc" in df_filtered.columns:
    # 1. Créer la catégorie d’IMC
    def classify_imc(imc):
        if pd.isna(imc):
            return "Inconnu"
        elif imc < 18.5:
            return "Insuffisance pondérale"
        elif imc >= 18.5 and imc <= 25.0:
            return "Normal"
        elif imc >= 25 and imc <= 29.9:
            return "Surpoids"
        elif imc >= 30 and imc <= 34.9:
            return "Obésité modérée"
        elif imc >= 35.0 and imc <= 39.9:
            return "Obésité sévère"
        else:
            return "Obésité massive"

    df_viz = df_filtered[["luc léger", "imc"]].dropna()
    if df_viz.empty:
        st.info("Aucune donnée disponible pour cette combinaison de filtres.")
    else:
        df_viz["imc_cat"] = df_viz["imc"].apply(classify_imc)

        palette = {
            "Normal": "green",
            "Surpoids": "orange",
            "Obésité modérée": "red",
            "Obésité sévère": "darkred",
            "Obésité massive": "black",
            "Insuffisance pondérale": "blue",
            "Inconnu": "gray",
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data=df_viz,
            x="luc léger",
            hue="imc_cat",
            multiple="stack",
            palette=palette,
            bins=15,
            edgecolor="white",
        )
        ax.set_title("Distribution du Palier Luc Léger par Catégorie d'IMC")
        ax.set_xlabel("Palier Luc Léger")
        ax.set_ylabel("Nombre d'individus")
        st.pyplot(fig)
else:
    st.warning("Les colonnes nécessaires 'luc léger' et 'imc' sont manquantes.")

st.subheader("Distribution du Tour de Taille selon le Sexe et les Normes de Santé")

if "périmètre abdominal" in df_filtered.columns and "sexe" in df_filtered.columns:
    df_tour = df_filtered[["périmètre abdominal", "sexe"]].dropna()

    def couleur_tour(row):
        if row["sexe"].lower() == "m":
            return "green" if row["périmètre abdominal"] < 94 else "red"
        elif row["sexe"].lower() == "f":
            return "green" if row["périmètre abdominal"] < 80 else "red"
        else:
            return "gray"

    df_tour["couleur"] = df_tour.apply(couleur_tour, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for couleur in ["green", "red", "gray"]:
        subset = df_tour[df_tour["couleur"] == couleur]
        if not subset.empty:
            ax.hist(
                subset["périmètre abdominal"],
                bins=15,
                alpha=0.7,
                label=couleur.capitalize(),
                color=couleur,
                edgecolor="black",
            )

    ax.set_title("Distribution du Tour de Taille (coloré selon les seuils OMS)")
    ax.set_xlabel("Tour de Taille (cm)")
    ax.set_ylabel("Nombre d'individus")
    ax.legend(title="État de santé")
    st.pyplot(fig)
else:
    st.warning(
        "La colonne 'périmètre abdominal' ou 'sexe' est manquante dans les données."
    )

features = [
    "imc",
    "taille",
    "poids",
]


for feature in features:
    st.subheader(f"Distribution de {feature.upper()}")
    if feature in df_filtered.columns and not df_filtered[feature].dropna().empty:
        fig, ax = plt.subplots()
        sns.histplot(df_filtered[feature], kde=True, ax=ax)
        ax.set_title(f"Histogramme de {feature.upper()}")
        st.pyplot(fig)
    else:
        st.info(f"Aucune donnée disponible pour {feature.upper()}.")

phys_tests = ["luc léger", "pompes", "tractions"]
for test in phys_tests:
    st.subheader(f"{test.replace('_', ' ').title()} par Cie")
    if not df_filtered.empty and test in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="cie_x", y=test, data=df_filtered, ax=ax, palette="Set2")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    else:
        st.info(f"Aucune donnée disponible pour {test}.")

level_cols = {
    "luc léger": " Palier luc léger",
    "niveau pompes": "niveau pompes",
    "niveau tractions": "niveau tractions",
}
for col, label in level_cols.items():
    if col in df_filtered.columns:
        st.subheader(f"{label} moyen par Cie")
        moyenne = df_filtered.groupby("cie_x")[col].mean().sort_values(ascending=False)
        if not moyenne.empty:
            fig, ax = plt.subplots()
            colors = plt.cm.viridis(moyenne.rank() / moyenne.count())
            moyenne.plot(kind="bar", ax=ax, color=colors)
            ax.set_ylabel("niveau moyen")
            ax.set_title(f"{label} moyen par Cie")
            st.pyplot(fig)


# --- NOUVELLES VISUALISATIONS : luc léger, Aptitude, Incendie/ARI ---
st.subheader("Distribution de la Tension Artérielle (Systolique & Diastolique)")

if (
    "tension artérielle systol" in df_filtered.columns
    and "tension artérielle diastol" in df_filtered.columns
):
    df_tension = df_filtered[
        ["tension artérielle systol", "tension artérielle diastol"]
    ].dropna()

    # Création des catégories selon les seuils OMS
    df_tension["sys_couleur"] = df_tension["tension artérielle systol"].apply(
        lambda x: "red" if x > 140 else "green"
    )
    df_tension["dia_couleur"] = df_tension["tension artérielle diastol"].apply(
        lambda x: "red" if x > 90 else "green"
    )

    # Histogramme tension systolique
    fig, ax = plt.subplots(figsize=(10, 6))
    for couleur in ["green", "red"]:
        subset = df_tension[df_tension["sys_couleur"] == couleur]
        if not subset.empty:
            ax.hist(
                subset["tension artérielle systol"],
                bins=15,
                alpha=0.7,
                label=f"Systolique ({couleur})",
                color=couleur,
                edgecolor="black",
            )
    ax.set_title("Distribution de la Tension Artérielle Systolique")
    ax.set_xlabel("Tension Systolique (mmHg)")
    ax.set_ylabel("Nombre d'individus")
    ax.legend(title="État (140 mmHg seuil)")
    st.pyplot(fig)

    # Histogramme tension diastolique
    fig, ax = plt.subplots(figsize=(10, 6))
    for couleur in ["green", "red"]:
        subset = df_tension[df_tension["dia_couleur"] == couleur]
        if not subset.empty:
            ax.hist(
                subset["tension artérielle diastol"],
                bins=15,
                alpha=0.7,
                label=f"Diastolique ({couleur})",
                color=couleur,
                edgecolor="black",
            )
    ax.set_title("Distribution de la Tension Artérielle Diastolique")
    ax.set_xlabel("Tension Diastolique (mmHg)")
    ax.set_ylabel("Nombre d'individus")
    ax.legend(title="État (90 mmHg seuil)")
    st.pyplot(fig)

else:
    st.warning("Les colonnes de tension artérielle sont manquantes ou incomplètes.")

st.subheader("luc léger selon l'Aptitude Générale et l'Exposition Incendie")

# Histogramme luc léger par Incendie et port de l'ARI, coloré par aptitude
if (
    "luc léger" in df_filtered.columns
    and "aptitude générale" in df_filtered.columns
    and "incendie et port de l'ari toutes missions_y" in df_filtered.columns
):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        df_filtered,
        x="luc léger",
        hue="aptitude générale",
        multiple="stack",
        bins=15,
        palette="Set2",
        kde=False,
    )
    ax.set_title("Répartition du palier luc léger selon l'aptitude générale")
    ax.set_xlabel("Palier luc léger")
    ax.set_ylabel("Nombre d'individus")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        df_filtered,
        x="luc léger",
        hue="incendie et port de l'ari toutes missions_y",
        multiple="stack",
        bins=15,
        palette="Set2",
        kde=False,
    )
    ax.set_title(
        "Répartition du palier luc léger selon Incendie et port de l'ARI Toutes missions"
    )
    ax.set_xlabel("Palier luc léger")
    ax.set_ylabel("Nombre d'individus")
    st.pyplot(fig)


# Boxplot luc léger par aptitude et Incendie/ARI
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=df_filtered,
    x="aptitude générale",
    y="luc léger",
    hue="incendie et port de l'ari toutes missions_y",
    palette="pastel",
)
ax.set_title("luc léger par Aptitude Générale et Incendie/ARI")
ax.set_ylabel("Palier luc léger")
ax.set_xlabel("Aptitude Générale")
ax.tick_params(axis="x", rotation=45)
st.pyplot(fig)

# --- Indicateurs imc, poids, taille par aptitude ---
st.subheader("Indicateurs physiques par Aptitude Générale")

indicateurs = ["imc", "poids", "taille"]
for ind in indicateurs:
    if ind in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=df_filtered,
            x="aptitude générale",
            y=ind,
            palette="coolwarm",
        )
        ax.set_title(f"{ind.upper()} par Aptitude Générale")
        ax.set_ylabel(ind.upper())
        ax.set_xlabel("Aptitude Générale")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
st.subheader("🔗 Corrélations avec le Palier luc léger")

# Sélection des colonnes numériques pertinentes
cols_corr = [
    "luc léger",
    "imc",
    "poids",
    "taille",
    "tension artérielle systol",
    "tension artérielle diastol",
    "pompes",
    "tractions",
    "niveau luc léger",
    "niveau pompes",
    "niveau tractions",
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
ax.set_title("Matrice de Corrélation - Indicateurs Physiques et luc léger")
st.pyplot(fig)

st.subheader("Carte Interactive des UT")


@st.cache_data
def load_geojson():
    with open("alsace_map.geojson", "r", encoding="utf-8") as f:
        return json.load(f)


try:
    geojson_data = load_geojson()

    # Nettoyage des noms d’UT
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
        df_filtered["ut_x"].astype(str).str.strip().str.upper().replace(ut_mapping)
    )

    # Moyenne d'IMC par UT
    imc_moyen = df_filtered.groupby("UT_clean")["imc"].mean().reset_index()
    imc_moyen.columns = ["nom", "imc_moyen"]

    # Effectif par UT
    effectif_ut = df_filtered["UT_clean"].value_counts().reset_index()
    effectif_ut.columns = ["nom", "effectif"]

    # Construction des features géographiques
    geo_features = [
        {**f["properties"], "geometry": f["geometry"]} for f in geojson_data["features"]
    ]
    geo_df = pd.DataFrame(geo_features)
    geo_df["nom"] = geo_df["nom"].str.strip().str.upper()

    # Fusion avec données
    geo_df = geo_df.merge(effectif_ut, on="nom", how="left")
    geo_df = geo_df.merge(imc_moyen, on="nom", how="left")
    geo_df.fillna({"effectif": 0, "imc_moyen": 0}, inplace=True)

    # Carte
    m = folium.Map(location=[48.6, 7.6], zoom_start=9, control_scale=True)

    colormap = cm.linear.YlOrRd_09.scale(
        geo_df["imc_moyen"].min(), geo_df["imc_moyen"].max()
    )
    colormap.caption = "IMC moyen"
    colormap.add_to(m)

    folium.Choropleth(
        geo_data=geojson_data,
        data=geo_df,
        columns=["nom", "imc_moyen"],
        key_on="feature.properties.nom",
        fill_color="YlOrRd",
        fill_opacity=0.6,
        line_opacity=0.5,
        line_color="black",
        legend_name="IMC moyen par UT",
        highlight=True,
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

            tooltip_text = f"""
<b>UT : {row['nom']}</b><br>
Effectif : {int(row['effectif'])}<br>
IMC moyen : {row['imc_moyen']:.2f}
"""
            folium.CircleMarker(
                location=(lat, lon),
                radius=7,
                color=colormap(row["imc_moyen"]),
                fill=True,
                fill_color=colormap(row["imc_moyen"]),
                fill_opacity=0.9,
                tooltip=folium.Tooltip(tooltip_text, sticky=True),
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
- Repérez les zones avec des tensions élevées ou imc critiques
- Analysez la progression ou la performance moyenne par région ou groupe d'âge
- Identifiez les corrélations entre les indicateurs (ex : poids vs imc, ou imc vs luc léger)
- Visualisez clairement la répartition des niveaux de luc léger par unité ou compagnie
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
