#!/usr/bin/env python3

import pandas as pd
import ast
import json
import math
import time
import re
import sys
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "BurkinaGeoEnricher/1.0 (Research Project; contact@example.com)"

BURKINA_FASO_QID = "Q965"


NEIGHBOR_RADIUS_KM = 10

REQUEST_DELAY_SECONDS = 0.5


EXTERNAL_LOCATIONS = {
    "bamako", "mali", "europe", "cuba", "indochine", "guinée", "burundi",
    "france", "niger", "côte d'ivoire", "ghana", "togo", "bénin", "sénégal"
}

TYPES_TO_REPLACE_WITH_ORIGINAL = {"autre", "ville"}

# Nom local → Nom Wikidata (pour la recherche)
LOCAL_TO_WIKIDATA_NAMES = {
    "Hauts-Bassins": "Guiriko",
    "hauts-bassins": "Guiriko",
    "Hauts-bassins": "Guiriko",
    "HAUTS-BASSINS": "Guiriko",
    "Hauts Bassins": "Guiriko",
}

# Nom Wikidata → Nom local (pour l'affichage dans la hiérarchie)
WIKIDATA_TO_LOCAL_NAMES = {
    "Guiriko": "Hauts-Bassins",
}

# Mapping des types administratifs Wikidata vers nos types
TYPE_MAPPING = {
    "région du burkina faso": "region",
    "region of burkina faso": "region",
    "province du burkina faso": "province",
    "province of burkina faso": "province",
    "département du burkina faso": "departement",
    "department of burkina faso": "departement",
    "commune du burkina faso": "departement",
    "commune of burkina faso": "departement",
    "ville": "ville",
    "city": "ville",
    "town": "ville",
    "village": "village",
    "human settlement": "village",
    "locality": "village",
    "country": "country",
    "pays": "country",
}

# =============================================================================
# CLASSES DE DONNÉES
# =============================================================================

@dataclass
class LocationInfo:
    """Informations géographiques d'un lieu."""
    name: str
    wikidata_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_type: Optional[str] = None
    hierarchy_departement: Optional[str] = None
    hierarchy_province: Optional[str] = None
    hierarchy_region: Optional[str] = None
    hierarchy_pays: str = "Burkina Faso"
    neighbors: List[Dict] = None
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = []

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def clean_location_name(name: str) -> str:
    """Nettoie et normalise le nom d'un lieu."""
    name = name.strip()
    
    # Normalisation des variantes courantes
    normalizations = {
        "OUAGADOUGOU": "Ouagadougou",
        "ouagadougou": "Ouagadougou",
        "Ouaga": "Ouagadougou",
        "ouaga": "Ouagadougou",
        "Bobo": "Bobo-Dioulasso",
        "bobo": "Bobo-Dioulasso",
        "Tougan.": "Tougan",
    }
    
    if name in normalizations:
        return normalizations[name]
    
    # Remplacer les tirets spéciaux
    name = re.sub(r'[–—]', '-', name)
    
    return name


def get_wikidata_search_name(name: str) -> str:
    """
    Convertit un nom local en nom Wikidata pour la recherche.
    Ex: "Hauts-Bassins" → "Guiriko"
    """
    # Vérifier dans le mapping (insensible à la casse pour plus de robustesse)
    if name in LOCAL_TO_WIKIDATA_NAMES:
        return LOCAL_TO_WIKIDATA_NAMES[name]
    
    # Essayer avec différentes variantes de casse
    name_lower = name.lower()
    for local_name, wikidata_name in LOCAL_TO_WIKIDATA_NAMES.items():
        if local_name.lower() == name_lower:
            return wikidata_name
    
    return name


def get_local_display_name(wikidata_name: str) -> str:
    """
    Convertit un nom Wikidata en nom local pour l'affichage.
    Ex: "Guiriko" → "Hauts-Bassins"
    """
    if wikidata_name in WIKIDATA_TO_LOCAL_NAMES:
        return WIKIDATA_TO_LOCAL_NAMES[wikidata_name]
    return wikidata_name


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcule la distance en kilomètres entre deux points géographiques
    en utilisant la formule de Haversine.
    """
    R = 6371  # Rayon de la Terre en km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def parse_wikidata_coordinates(coord_string: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse les coordonnées Wikidata au format 'Point(lon lat)'."""
    if not coord_string:
        return None, None
    
    match = re.search(r'Point\(([^\s]+)\s+([^\)]+)\)', coord_string)
    if match:
        lon = float(match.group(1))
        lat = float(match.group(2))
        return lat, lon
    return None, None


def classify_location_type(wikidata_type: str) -> str:
    """Classifie le type de lieu selon les catégories standard."""
    if not wikidata_type:
        return "autre"
    
    type_lower = wikidata_type.lower()
    
    for pattern, classification in TYPE_MAPPING.items():
        if pattern in type_lower:
            return classification
    
    # Patterns additionnels
    if "région" in type_lower or "region" in type_lower:
        return "region"
    elif "province" in type_lower:
        return "province"
    elif "département" in type_lower or "commune" in type_lower:
        return "departement"
    elif "village" in type_lower or "settlement" in type_lower:
        return "village"
    elif "ville" in type_lower or "city" in type_lower or "town" in type_lower:
        return "ville"
    
    return "autre"


def determine_location_type(wikidata_type: str, original_label: str) -> str:
    """
    Détermine le type final d'un lieu.
    
    Si Wikidata retourne un type générique ("autre" ou "ville"), on conserve
    le label original du DataFrame d'entrée (s'il existe).
    
    Args:
        wikidata_type: Type brut retourné par Wikidata
        original_label: Label original du DataFrame d'entrée
    
    Returns:
        Le type à utiliser pour ce lieu
    """
    classified_type = classify_location_type(wikidata_type)
    
    # Si le type Wikidata est générique et qu'on a un label original, on le conserve
    if classified_type in TYPES_TO_REPLACE_WITH_ORIGINAL and original_label:
        return original_label
    
    return classified_type

# =============================================================================
# CLASSE PRINCIPALE - CLIENT WIKIDATA
# =============================================================================

class WikidataGeoClient:
    """Client pour interroger Wikidata et récupérer les données géographiques."""
    
    def __init__(self):
        self.sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
        self.sparql.setReturnFormat(JSON)
        self.sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
        self.cache = {}  # Cache des résultats
    
    def _execute_query(self, query: str) -> Optional[Dict]:
        """Exécute une requête SPARQL avec gestion des erreurs."""
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            time.sleep(REQUEST_DELAY_SECONDS)
            return results
        except Exception as e:
            logger.warning(f"Erreur SPARQL: {e}")
            time.sleep(REQUEST_DELAY_SECONDS * 2)  # Attendre plus en cas d'erreur
            return None
    
    def search_location(self, location_name: str) -> Optional[Dict]:
        """
        Recherche un lieu sur Wikidata par son nom.
        Retourne l'ID Wikidata, les coordonnées et le type.
        """
        clean_name = clean_location_name(location_name)
        # Convertir le nom local en nom Wikidata si nécessaire
        search_name = get_wikidata_search_name(clean_name)
        
        # Vérifier le cache (avec le nom original pour éviter les doublons)
        cache_key = f"search_{clean_name.lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Log si le nom a été converti
        if search_name != clean_name:
            logger.info(f"    [Mapping] '{clean_name}' → '{search_name}' (nom Wikidata)")
        
        # Requête 1: Recherche avec filtre Burkina Faso
        query = f"""
        SELECT DISTINCT ?item ?itemLabel ?coord ?typeLabel WHERE {{
          {{
            ?item rdfs:label "{search_name}"@fr .
          }} UNION {{
            ?item rdfs:label "{search_name}"@en .
          }} UNION {{
            ?item skos:altLabel "{search_name}"@fr .
          }}
          ?item wdt:P17 wd:{BURKINA_FASO_QID} .
          OPTIONAL {{ ?item wdt:P625 ?coord . }}
          OPTIONAL {{ ?item wdt:P31 ?type . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "fr,en". }}
        }}
        LIMIT 5
        """
        
        results = self._execute_query(query)
        
        if results and results["results"]["bindings"]:
            result = results["results"]["bindings"][0]
            entity_id = result["item"]["value"].split("/")[-1]
            coord = result.get("coord", {}).get("value")
            type_label = result.get("typeLabel", {}).get("value", "")
            
            data = {
                "wikidata_id": entity_id,
                "coordinates": coord,
                "wikidata_type": type_label
            }
            self.cache[cache_key] = data
            return data
        
        # Requête 2: Recherche plus large (sans filtre pays)
        query_broad = f"""
        SELECT DISTINCT ?item ?itemLabel ?coord ?typeLabel ?countryLabel WHERE {{
          {{
            ?item rdfs:label "{search_name}"@fr .
          }} UNION {{
            ?item rdfs:label "{search_name}"@en .
          }}
          OPTIONAL {{ ?item wdt:P625 ?coord . }}
          OPTIONAL {{ ?item wdt:P31 ?type . }}
          OPTIONAL {{ ?item wdt:P17 ?country . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "fr,en". }}
        }}
        LIMIT 10
        """
        
        results = self._execute_query(query_broad)
        
        if results and results["results"]["bindings"]:
            # Filtrer pour privilégier les résultats du Burkina Faso
            for result in results["results"]["bindings"]:
                country = result.get("countryLabel", {}).get("value", "").lower()
                if "burkina" in country:
                    entity_id = result["item"]["value"].split("/")[-1]
                    coord = result.get("coord", {}).get("value")
                    type_label = result.get("typeLabel", {}).get("value", "")
                    
                    data = {
                        "wikidata_id": entity_id,
                        "coordinates": coord,
                        "wikidata_type": type_label
                    }
                    self.cache[cache_key] = data
                    return data
            
            # Si pas de résultat Burkina, prendre le premier
            result = results["results"]["bindings"][0]
            entity_id = result["item"]["value"].split("/")[-1]
            coord = result.get("coord", {}).get("value")
            type_label = result.get("typeLabel", {}).get("value", "")
            
            data = {
                "wikidata_id": entity_id,
                "coordinates": coord,
                "wikidata_type": type_label
            }
            self.cache[cache_key] = data
            return data
        
        self.cache[cache_key] = None
        return None
    
    def get_administrative_hierarchy(self, entity_id: str) -> Dict:
        """
        Récupère l'arborescence administrative complète pour une entité.
        Utilise P131 (situé dans l'entité territoriale administrative).
        """
        cache_key = f"hierarchy_{entity_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        query = f"""
        SELECT ?parent1 ?parent1Label ?parent1TypeLabel
               ?parent2 ?parent2Label ?parent2TypeLabel
               ?parent3 ?parent3Label ?parent3TypeLabel
               ?parent4 ?parent4Label ?parent4TypeLabel
        WHERE {{
          OPTIONAL {{
            wd:{entity_id} wdt:P131 ?parent1 .
            OPTIONAL {{ ?parent1 wdt:P31 ?parent1Type . }}
            OPTIONAL {{
              ?parent1 wdt:P131 ?parent2 .
              OPTIONAL {{ ?parent2 wdt:P31 ?parent2Type . }}
              OPTIONAL {{
                ?parent2 wdt:P131 ?parent3 .
                OPTIONAL {{ ?parent3 wdt:P31 ?parent3Type . }}
                OPTIONAL {{
                  ?parent3 wdt:P131 ?parent4 .
                  OPTIONAL {{ ?parent4 wdt:P31 ?parent4Type . }}
                }}
              }}
            }}
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "fr,en". }}
        }}
        LIMIT 1
        """
        
        hierarchy = {
            "departement": None,
            "province": None,
            "region": None,
            "pays": "Burkina Faso"
        }
        
        results = self._execute_query(query)
        
        if results and results["results"]["bindings"]:
            result = results["results"]["bindings"][0]
            
            # Parcourir les niveaux et identifier leur type
            for i in range(1, 5):
                label = result.get(f"parent{i}Label", {}).get("value", "")
                type_label = result.get(f"parent{i}TypeLabel", {}).get("value", "").lower()
                
                if label:
                    # Convertir le nom Wikidata en nom local si nécessaire
                    display_label = get_local_display_name(label)
                    
                    if "région" in type_label or "region" in type_label:
                        hierarchy["region"] = display_label
                    elif "province" in type_label:
                        hierarchy["province"] = display_label
                    elif "département" in type_label or "commune" in type_label:
                        if not hierarchy["departement"]:
                            hierarchy["departement"] = display_label
        
        self.cache[cache_key] = hierarchy
        return hierarchy
    
    def get_nearby_entities(self, lat: float, lon: float, radius_km: float = 10) -> List[Dict]:
        """
        Trouve les entités spatiales voisines dans un rayon donné.
        Utilise le service géospatial de Wikidata.
        
        Note: Les voisins conservent le type Wikidata car ce sont de nouveaux
        lieux découverts (non présents dans le DataFrame d'entrée).
        """
        cache_key = f"nearby_{lat}_{lon}_{radius_km}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        query = f"""
        SELECT DISTINCT ?place ?placeLabel ?distance ?typeLabel ?coord WHERE {{
          SERVICE wikibase:around {{
            ?place wdt:P625 ?coord .
            bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
            bd:serviceParam wikibase:radius "{radius_km}" .
            bd:serviceParam wikibase:distance ?distance .
          }}
          ?place wdt:P17 wd:{BURKINA_FASO_QID} .
          OPTIONAL {{ ?place wdt:P31 ?type . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "fr,en". }}
        }}
        ORDER BY ?distance
        LIMIT 50
        """
        
        neighbors = []
        results = self._execute_query(query)
        
        if results and results["results"]["bindings"]:
            for binding in results["results"]["bindings"]:
                name = binding.get("placeLabel", {}).get("value", "")
                distance = binding.get("distance", {}).get("value", "")
                entity_type = binding.get("typeLabel", {}).get("value", "")
                coord_str = binding.get("coord", {}).get("value", "")
                
                n_lat, n_lon = parse_wikidata_coordinates(coord_str)
                # Pour les voisins, on garde le type Wikidata tel quel
                # (ce sont de nouveaux lieux, pas de label original disponible)
                classified_type = classify_location_type(entity_type)
                
                # Convertir le nom Wikidata en nom local si nécessaire
                display_name = get_local_display_name(name)
                
                if display_name and distance:
                    neighbors.append({
                        "nom": display_name,
                        "type": classified_type,
                        "type_wikidata": entity_type,
                        "distance_km": round(float(distance), 2),
                        "lat": n_lat,
                        "lon": n_lon
                    })
        
        self.cache[cache_key] = neighbors
        return neighbors

# =============================================================================
# FONCTION PRINCIPALE D'ENRICHISSEMENT
# =============================================================================

def enrich_location(client: WikidataGeoClient, location_name: str, location_label: str) -> LocationInfo:
    """
    Enrichit un lieu avec toutes ses informations géographiques.
    
    Args:
        client: Client Wikidata
        location_name: Nom du lieu
        location_label: Label original du DataFrame d'entrée (ex: "departement", "province")
    
    Returns:
        LocationInfo avec les données enrichies
    """
    clean_name = clean_location_name(location_name)
    info = LocationInfo(name=clean_name)
    
    # Vérifier si c'est un lieu externe
    if clean_name.lower() in EXTERNAL_LOCATIONS:
        info.hierarchy_pays = clean_name
        logger.info(f"  {clean_name}: Lieu hors Burkina Faso")
        return info
    
    # Cas spécial: le pays lui-même
    if clean_name.lower() in ["burkina", "burkina faso"]:
        info.wikidata_id = BURKINA_FASO_QID
        info.location_type = "country"
        info.latitude = 12.2383
        info.longitude = -1.5616
        logger.info(f"  {clean_name}: Pays (Burkina Faso)")
        return info
    
    # Cas spécial: axes routiers
    if "axe" in clean_name.lower():
        logger.info(f"  {clean_name}: Axe routier (non géolocalisable)")
        return info
    
    # Rechercher sur Wikidata
    logger.info(f"  Recherche: {clean_name}")
    search_result = client.search_location(clean_name)
    
    if search_result:
        info.wikidata_id = search_result["wikidata_id"]
        
        # Déterminer le type géographique
        wikidata_type_raw = search_result.get("wikidata_type", "")
        wikidata_type_classified = classify_location_type(wikidata_type_raw)
        
        # Si Wikidata retourne "autre" ou "ville", conserver le label original du DataFrame
        # (sauf si le label original est vide)
        if wikidata_type_classified in TYPES_TO_REPLACE_WITH_ORIGINAL and location_label:
            info.location_type = location_label
            logger.info(f"    → Type Wikidata '{wikidata_type_classified}' ('{wikidata_type_raw}') → conservé label original '{location_label}'")
        else:
            info.location_type = wikidata_type_classified
        
        # Coordonnées
        if search_result.get("coordinates"):
            info.latitude, info.longitude = parse_wikidata_coordinates(search_result["coordinates"])
            logger.info(f"    → Trouvé: {info.wikidata_id} ({info.latitude}, {info.longitude}) - type: {info.location_type}")
        else:
            logger.info(f"    → Trouvé: {info.wikidata_id} (sans coordonnées) - type: {info.location_type}")
        
        # Hiérarchie administrative
        hierarchy = client.get_administrative_hierarchy(info.wikidata_id)
        info.hierarchy_departement = hierarchy.get("departement")
        info.hierarchy_province = hierarchy.get("province")
        info.hierarchy_region = hierarchy.get("region")
        
        if hierarchy.get("region"):
            logger.info(f"    → Hiérarchie: {info.hierarchy_departement} → {info.hierarchy_province} → {info.hierarchy_region}")
        
        # Voisins (si coordonnées disponibles)
        if info.latitude and info.longitude:
            neighbors = client.get_nearby_entities(info.latitude, info.longitude, NEIGHBOR_RADIUS_KM)
            # Exclure le lieu lui-même
            info.neighbors = [n for n in neighbors if n["nom"].lower() != clean_name.lower()]
            logger.info(f"    → {len(info.neighbors)} voisins dans {NEIGHBOR_RADIUS_KM}km")
    else:
        # Lieu non trouvé sur Wikidata : conserver le label original
        if location_label:
            info.location_type = location_label
            logger.warning(f"    → Non trouvé sur Wikidata, conservé label original '{location_label}'")
        else:
            logger.warning(f"    → Non trouvé sur Wikidata")
    
    return info


def process_dataframe(df: pd.DataFrame, location_column: str = 'location') -> pd.DataFrame:
    """
    Traite un DataFrame et enrichit chaque lieu.
    """
    client = WikidataGeoClient()
    
    # Extraire les lieux uniques
    unique_locations = {}
    for idx, row in df.iterrows():
        try:
            loc = ast.literal_eval(row[location_column])
            key = (loc['text'].strip(), loc.get('label', ''))
            if key not in unique_locations:
                unique_locations[key] = loc
        except:
            pass
    
    logger.info(f"Traitement de {len(unique_locations)} lieux uniques...")
    
    # Traiter chaque lieu unique
    location_data = {}
    for i, ((name, label), loc_info) in enumerate(unique_locations.items()):
        logger.info(f"\n[{i+1}/{len(unique_locations)}] {name} ({label})")
        location_data[(name, label)] = enrich_location(client, name, label)
    
    # Créer les nouvelles colonnes
    new_columns = {
        "lieu_nettoye": [],
        "wikidata_id": [],
        "latitude": [],
        "longitude": [],
        "type_geo": [],
        "hierarchy_departement": [],
        "hierarchy_province": [],
        "hierarchy_region": [],
        "hierarchy_pays": [],
        "neighbors_10km": [],
        "neighbors_10km_count": [],
        "neighbors_10km_types": []
    }
    
    # Remplir les colonnes pour chaque ligne
    for idx, row in df.iterrows():
        try:
            loc = ast.literal_eval(row[location_column])
            key = (loc['text'].strip(), loc.get('label', ''))
            info = location_data.get(key)
        except:
            info = None
        
        if info:
            new_columns["lieu_nettoye"].append(info.name)
            new_columns["wikidata_id"].append(info.wikidata_id)
            new_columns["latitude"].append(info.latitude)
            new_columns["longitude"].append(info.longitude)
            new_columns["type_geo"].append(info.location_type)
            new_columns["hierarchy_departement"].append(info.hierarchy_departement)
            new_columns["hierarchy_province"].append(info.hierarchy_province)
            new_columns["hierarchy_region"].append(info.hierarchy_region)
            new_columns["hierarchy_pays"].append(info.hierarchy_pays)
            new_columns["neighbors_10km"].append(json.dumps(info.neighbors, ensure_ascii=False))
            new_columns["neighbors_10km_count"].append(len(info.neighbors))
            
            # Compter les types de voisins
            type_counts = {}
            for n in info.neighbors:
                t = n.get("type", "autre")
                type_counts[t] = type_counts.get(t, 0) + 1
            new_columns["neighbors_10km_types"].append(json.dumps(type_counts, ensure_ascii=False))
        else:
            for col in new_columns:
                if col == "neighbors_10km":
                    new_columns[col].append("[]")
                elif col == "neighbors_10km_count":
                    new_columns[col].append(0)
                elif col == "neighbors_10km_types":
                    new_columns[col].append("{}")
                elif col == "hierarchy_pays":
                    new_columns[col].append("Burkina Faso")
                else:
                    new_columns[col].append(None)
    
    # Ajouter les colonnes au DataFrame
    for col_name, col_data in new_columns.items():
        df[col_name] = col_data
    
    return df


def print_statistics(df: pd.DataFrame):
    """Affiche les statistiques de l'enrichissement."""
    print("\n" + "=" * 70)
    print("STATISTIQUES DE L'ENRICHISSEMENT")
    print("=" * 70)
    
    print(f"\n📊 GÉNÉRAL")
    print(f"   Total enregistrements: {len(df)}")
    print(f"   Lieux uniques: {df['lieu_nettoye'].nunique()}")
    
    print(f"\n📍 GÉOLOCALISATION")
    print(f"   Avec coordonnées: {df['latitude'].notna().sum()} ({df['latitude'].notna().sum()*100/len(df):.1f}%)")
    print(f"   Avec Wikidata ID: {df['wikidata_id'].notna().sum()} ({df['wikidata_id'].notna().sum()*100/len(df):.1f}%)")
    
    print(f"\n🏛️ HIÉRARCHIE ADMINISTRATIVE")
    print(f"   Avec région: {df['hierarchy_region'].notna().sum()} ({df['hierarchy_region'].notna().sum()*100/len(df):.1f}%)")
    print(f"   Avec province: {df['hierarchy_province'].notna().sum()} ({df['hierarchy_province'].notna().sum()*100/len(df):.1f}%)")
    print(f"   Avec département: {df['hierarchy_departement'].notna().sum()} ({df['hierarchy_departement'].notna().sum()*100/len(df):.1f}%)")
    
    print(f"\n🔗 VOISINAGE ({NEIGHBOR_RADIUS_KM} KM)")
    print(f"   Lieux avec voisins: {(df['neighbors_10km_count'] > 0).sum()}")
    print(f"   Moyenne voisins/lieu: {df['neighbors_10km_count'].mean():.1f}")
    print(f"   Max voisins: {df['neighbors_10km_count'].max()}")
    
    # Répartition par région
    print(f"\n🗺️ RÉPARTITION PAR RÉGION")
    region_counts = df['hierarchy_region'].value_counts().head(10)
    for region, count in region_counts.items():
        print(f"   {region or 'Non défini'}: {count}")
    
    # Répartition par type géographique
    print(f"\n📌 RÉPARTITION PAR TYPE GÉOGRAPHIQUE")
    type_counts = df['type_geo'].value_counts()
    for geo_type, count in type_counts.items():
        print(f"   {geo_type or 'Non défini'}: {count}")


def main():
    """Fonction principale."""
    print("=" * 70)
    print("ENRICHISSEMENT GÉOGRAPHIQUE VIA WIKIDATA")
    print("Burkina Faso - Analyse de sécurité alimentaire")
    print("=" * 70)
    
    # Arguments en ligne de commande
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        output_file = input_file.replace('.csv', '_enriched.csv')
    else:
        # Fichier par défaut
        input_file = 'talamasca.csv'
        output_file = 'df_processed_enriched.csv'
    
    print(f"\n📂 Fichier d'entrée: {input_file}")
    print(f"📂 Fichier de sortie: {output_file}")
    
    # Charger les données
    print("\n1. Chargement des données...")
    try:
        df = pd.read_csv(input_file)
        print(f"   ✓ {len(df)} enregistrements chargés")
    except FileNotFoundError:
        print(f"   ✗ Fichier non trouvé: {input_file}")
        sys.exit(1)
    
    # Vérifier la colonne location
    if 'location' not in df.columns:
        print("   ✗ Colonne 'location' non trouvée dans le CSV")
        sys.exit(1)
    
    # Traiter le DataFrame
    print("\n2. Enrichissement via Wikidata...")
    print("   (Cela peut prendre plusieurs minutes selon le nombre de lieux)")
    df_enriched = process_dataframe(df)
    
    # Sauvegarder
    print(f"\n3. Sauvegarde: {output_file}")
    df_enriched.to_csv(output_file, index=False, encoding='utf-8')
    print("   ✓ Fichier sauvegardé")
    
    # Statistiques
    print_statistics(df_enriched)
    
    print("\n" + "=" * 70)
    print("ENRICHISSEMENT TERMINÉ")
    print("=" * 70)
    
    return df_enriched


if __name__ == "__main__":
    df_result = main()