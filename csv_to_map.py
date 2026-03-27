import os
import warnings

import matplotlib; matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import geopandas as gpd
from matplotlib.lines import Line2D
from pyproj import Geod

try:
    import folium
    from folium.plugins import HeatMap, HeatMapWithTime
    from folium.plugins import BeautifyIcon
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))

colors = ['blue', 'red', 'darkgreen', 'purple']

# Bounding box Burkina Faso + marge
BF_EXTENT = (-5.7, 2.5, 9.3, 15.2)   # (lon_min, lon_max, lat_min, lat_max)

#==========================
def loadCSVData(_filepath, _separator=","):
    with open(_filepath) as csv_file:
        df = pd.read_csv(
            _filepath, delimiter=_separator, header=0,
            names=['place','type','latitude','longitude',
                   'Flood','Fire','Drought','Storm','Tornado','FloodPeak',
                   'Thunderstorm','Desertification','Deforestation','Landslide',
                   'Collapse','AgriculturalCampaign','PriceIncrease']
        )
        return df

#==========================
def spatialViz(_data):
    """Carte interactive Folium (HTML)."""
    if not _FOLIUM_OK:
        print("  [warn] folium non installé — carte HTML ignorée. "
              "Installez-le avec : pip install folium")
        return
    data = _data.dropna(subset=['latitude', 'longitude'])
    locations = data[['latitude', 'longitude']]
    locationlist = locations.values.tolist()
    print("data size:", len(locationlist))

    places = data[['place', 'type']]
    placeslist = places.values.tolist()
    envrisk = data[['Flood', 'Fire', 'AgriculturalCampaign', 'PriceIncrease']]
    envrisklist = envrisk.values.tolist()
    print(envrisk)

    map = folium.Map(location=(12.27793, -1.573062), zoom_start=6, tiles="OpenStreetMap")

    radius = 10
    for i in range(len(data)):
        for k in range(4):
            if envrisklist[i][k] == 0:
                continue
            shape = None
            if placeslist[i][1] == "region":
                shape = "circle"
            elif placeslist[i][1] == "departement":
                shape = "star"
            elif placeslist[i][1] == "village":
                shape = "marker"
            folium.Marker(
                [locationlist[i][0], locationlist[i][1]],
                icon=BeautifyIcon(
                    location=[locationlist[i][0], locationlist[i][1]],
                    border_color=colors[k],
                    text_color=colors[k],
                    number=int(envrisklist[i][k]),
                    inner_icon_style="margin-top:0;font-size:16px",
                    icon_shape=shape,
                    icon_size=(35, 35)
                ),
                tooltip=placeslist[i][0] + ":" + "{} events".format(int(envrisklist[i][k]))
            ).add_to(map)

    out_dir = os.path.join(current_dir, "output_data", "dataviz")
    os.makedirs(out_dir, exist_ok=True)
    map.save(os.path.join(out_dir, "carto_all_events.html"))

#==========================
# ── Helpers cartographiques (éléments requis par Reviewer 3) ─────────────────

def _add_north_arrow(ax, x=0.06, y=0.18, size=0.07):
    """(1) Flèche Nord en coordonnées axes."""
    ax.annotate(
        "", xy=(x, y + size), xytext=(x, y),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=18),
    )
    ax.text(
        x, y + size + 0.015, "N",
        transform=ax.transAxes,
        ha="center", va="bottom", fontsize=11, fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )


def _add_scale_bar(ax, length_km=100, location=(0.04, 0.07)):
    """(2) Barre d'échelle. Longueur calculée en degrés via pyproj à la latitude médiane."""
    geod = Geod(ellps="WGS84")
    lon_min, lon_max, lat_min, lat_max = BF_EXTENT
    lat_mid = (lat_min + lat_max) / 2
    lon_ref = (lon_min + lon_max) / 2

    _, _, dist_1deg = geod.inv(lon_ref, lat_mid, lon_ref + 1.0, lat_mid)
    deg_per_km = 1.0 / (dist_1deg / 1000.0)
    bar_frac   = (length_km * deg_per_km) / (lon_max - lon_min)

    x0, y0 = location
    bar_h  = 0.008

    # Moitié droite noire + moitié gauche blanche (alternance standard)
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), bar_frac, bar_h, boxstyle="square,pad=0",
        transform=ax.transAxes, facecolor="black", edgecolor="black", lw=0.5, zorder=5,
    ))
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), bar_frac / 2, bar_h, boxstyle="square,pad=0",
        transform=ax.transAxes, facecolor="white", edgecolor="black", lw=0.5, zorder=6,
    ))
    ax.text(x0,            y0 - 0.013, "0",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
    ax.text(x0 + bar_frac, y0 - 0.013, f"{length_km} km",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5)


def _build_legend(ax, present_types):
    handles = []

    # ── SECTION 1 : RISKS ─────────────────────────────
    handles.append(Line2D([], [], linestyle="none", label="Risks"))
    for label, c in [
        ("Flood",  "#2171b5"),
        ("Fire", "#d73027"),
        ("Agricultural Campaign", "#1a9641"),
        ("Inflation", "#7b2d8b"),
    ]:
        handles.append(
            mpatches.Patch(facecolor=c, edgecolor="white", lw=0.5,
                           label="   " + label)   # indentation
        )

    # espace
    handles.append(Line2D([], [], linestyle="none", label=" "))

    # ── SECTION 2 : TYPE DE LIEUX ─────────────────────
    handles.append(Line2D([], [], linestyle="none", label="Type of Locations"))
    for geo_type, mk, lbl in [
        ("region", "o", "Region"),
        ("province", "s", "Province"),
        ("departement", "^", "Departement"),
        ("village", "D", "Village"),
        ("country", "P", "Country"),
    ]:
        if geo_type not in present_types:
            continue
        handles.append(
            Line2D([0], [0],
                   marker=mk, linestyle="none",
                   markerfacecolor="#555555",
                   markeredgecolor="white",
                   markeredgewidth=0.4,
                   markersize=8,
                   label="   " + lbl)
        )

    # espace
    handles.append(Line2D([], [], linestyle="none", label=" "))

    # ── SECTION 3 : NOMBRE D'ÉVÉNEMENTS ───────────────
    handles.append(Line2D([], [], linestyle="none", label="Number of Events"))
    for n, sz in [(1, 5), (5, 9), (15, 13)]:
        handles.append(
            Line2D([0], [0],
                   marker="o", linestyle="none",
                   markerfacecolor="#888888",
                   markeredgecolor="white",
                   markeredgewidth=0.4,
                   markersize=sz,
                   label="   " + str(n))
        )

    # ── CRÉATION LÉGENDE ──────────────────────────────
    leg = ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.97,
        edgecolor="#bbbbbb",
        fontsize=8,
        title="Legend",
        title_fontsize=10,
        borderpad=1,
        labelspacing=0.5,
        handletextpad=0.6,
    )

    # alignement gauche
    leg._legend_box.align = "left"

    # titres en gras
    for text in leg.get_texts():
        if text.get_text().strip() in [
            "Risks", "Type of Locations", "Number of Events"
        ]:
            text.set_fontweight("bold")

    # style boîte
    leg.get_frame().set_linewidth(0.8)

    return leg

#==========================
def spatialVizStatic(_data, output_path=None, dpi=300):
    """Carte statique publication (PNG) avec les 4 éléments cartographiques.

    Reviewer 3 — éléments inclus :
      (1) Flèche Nord
      (2) Barre d'échelle (100 km)
      (3) Légende complète (risque, type de lieu, nb événements)
      (4) Grille de coordonnées WGS84 (EPSG:4326)
    """
    if output_path is None:
        out_dir = os.path.join(current_dir, "output_data", "dataviz")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "figure5_vulnerable_zones.png")

    data = _data.dropna(subset=['latitude', 'longitude']).copy()
    data['latitude']  = pd.to_numeric(data['latitude'],  errors='coerce')
    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    data = data.dropna(subset=['latitude', 'longitude'])

    # Filtrer les coordonnées hors Burkina Faso (erreurs Wikidata)
    lon_min, lon_max, lat_min, lat_max = BF_EXTENT
    data = data[
        data['longitude'].between(lon_min, lon_max) &
        data['latitude'].between(lat_min, lat_max)
    ].copy()

    risk_cols   = ['Flood', 'Fire', 'AgriculturalCampaign', 'PriceIncrease']
    risk_colors = ['#2171b5', '#d73027', '#1a9641', '#7b2d8b']
    data['n_total'] = data[risk_cols].sum(axis=1)

    # ── Fond cartographique (Natural Earth via cartopy) ───────────────────
    print("  Chargement des couches cartographiques (Natural Earth)...")
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 16), subplot_kw={"projection": proj})
    plt.subplots_adjust(right=0.78)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax.set_facecolor("#d4e6f1")

    countries_shp = shpreader.natural_earth(resolution='50m', category='cultural',
                                            name='admin_0_countries')
    countries_gdf = gpd.read_file(countries_shp)

    # Pays voisins
    for geom in countries_gdf[
        countries_gdf['NAME'].isin(["Mali","Niger","Benin","Togo","Ghana","Ivory Coast"])
    ].geometry:
        ax.add_geometries([geom], crs=proj, facecolor="#f0f0f0",
                          edgecolor="#aaaaaa", linewidth=0.4)

    # Burkina Faso fond
    bf_gdf = countries_gdf[countries_gdf['NAME'] == 'Burkina Faso']
    for geom in bf_gdf.geometry:
        ax.add_geometries([geom], crs=proj, facecolor="#fff7ec", edgecolor="none")

    # Régions admin Burkina Faso
    admin1_shp = shpreader.natural_earth(resolution='10m', category='cultural',
                                         name='admin_1_states_provinces')
    bf_admin = gpd.read_file(admin1_shp)
    bf_admin = bf_admin[bf_admin['admin'] == 'Burkina Faso']
    for geom in bf_admin.geometry:
        ax.add_geometries([geom], crs=proj, facecolor="none",
                          edgecolor="#999999", linewidth=0.35, linestyle="--")
    for _, row in bf_admin.iterrows():
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        if lon_min < cx < lon_max and lat_min < cy < lat_max:
            ax.text(cx, cy, row['name'], transform=proj,
                    fontsize=5.5, ha="center", va="center",
                    color="#666666", style="italic",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    # Frontière Burkina Faso (trait épais)
    for geom in bf_gdf.geometry:
        ax.add_geometries([geom], crs=proj, facecolor="none",
                          edgecolor="#222222", linewidth=1.2)

    # ── Données événements ────────────────────────────────────────────────
    marker_map = {"region": "o", "province": "s",
                  "departement": "^", "village": "D", "country": "P"}
    n_max = data['n_total'].replace(0, np.nan).max()
    if not n_max or np.isnan(n_max):
        n_max = 1

    for _, row in data.sort_values('n_total', ascending=False).iterrows():
        mk  = marker_map.get(str(row['type']).strip(), "o")
        col = next((rc for rc, c in zip(risk_cols, risk_colors) if row[rc] > 0), "#888888")
        if isinstance(col, str) and col in risk_cols:
            col = risk_colors[risk_cols.index(col)]
        sz  = max(18, 18 + 180 * (max(row['n_total'], 0) / n_max) ** 0.5)
        ax.scatter(row['longitude'], row['latitude'], transform=proj,
                   s=sz, c=col, marker=mk,
                   edgecolors="white", linewidths=0.5, alpha=0.87, zorder=4)

    # Labels des lieux les plus touchés (n_total >= 3)
    for _, row in data[data['n_total'] >= 3].sort_values('n_total', ascending=False).iterrows():
        ax.text(row['longitude'] + 0.05, row['latitude'] + 0.06,
                row['place'], transform=proj,
                fontsize=6.5, ha="left", va="bottom", color="#1a1a1a",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

    # ── (4) Grille de coordonnées — WGS84 EPSG:4326 ──────────────────────
    gl = ax.gridlines(crs=proj, draw_labels=True,
                      linewidth=0.5, color="#aaaaaa", alpha=0.6, linestyle=":",
                      x_inline=False, y_inline=False)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7.5, "color": "#333333"}
    gl.ylabel_style = {"size": 7.5, "color": "#333333"}
    gl.xlocator = plt.FixedLocator([-4, -2, 0, 2])
    gl.ylocator = plt.FixedLocator([10, 11, 12, 13, 14, 15])

    # ── (1) Flèche Nord ───────────────────────────────────────────────────
    _add_north_arrow(ax, x=0.06, y=0.04, size=0.07)

    # ── (2) Barre d'échelle ───────────────────────────────────────────────
    _add_scale_bar(ax, length_km=100, location=(0.04, 0.03))

    # Mention système de coordonnées
    ax.text(0.99, 0.01, "Coord. WGS84 (EPSG:4326)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#444444",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    # ── (3) Légende complète ──────────────────────────────────────────────
    _build_legend(ax, set(data['type'].astype(str).str.strip().unique()))

    # ── Titre & source ────────────────────────────────────────────────────
    # ax.set_title(
    #     ""
    #     "",
    #     #fontsize=11, fontweight="bold", pad=10,
    # )
    # fig.text(
    #     # 0.5, 0.01,
    #     # "Source : STKGFS Knowledge Graph — données issues de la presse burkinabè (2009). "
    #     # "Fond cartographique : Natural Earth.",
    #     # ha="center", fontsize=6.5, color="#666666",
    # )
    
    plt.tight_layout(rect=[0, 0, 0.78, 1])

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  [save] {output_path}  (dpi={dpi})")

#==========================
if __name__ == '__main__':
    dep_risks_file = os.path.join(
        current_dir, "Events_to_Graph", "data", "eco-env-agrii.csv"
    )
    data = loadCSVData(dep_risks_file)

    # Carte interactive HTML (Folium)
    spatialViz(data)

    # Carte statique PNG publication (avec éléments cartographiques Reviewer 3)
    spatialVizStatic(data)
