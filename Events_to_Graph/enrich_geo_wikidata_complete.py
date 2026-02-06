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

# Local name -> Wikidata name (for search)
LOCAL_TO_WIKIDATA_NAMES = {
    "Hauts-Bassins": "Guiriko",
    "hauts-bassins": "Guiriko",
    "Hauts-bassins": "Guiriko",
    "HAUTS-BASSINS": "Guiriko",
    "Hauts Bassins": "Guiriko",
}

# Wikidata name -> Local display name (for hierarchy display)
WIKIDATA_TO_LOCAL_NAMES = {
    "Guiriko": "Hauts-Bassins",
}

# Mapping of Wikidata administrative types to standard types
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
# DATA CLASSES
# =============================================================================

@dataclass
class LocationInfo:
    """Geographic information for a location."""
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
# UTILITY FUNCTIONS
# =============================================================================

def clean_location_name(name: str) -> str:
    """Clean and normalize a location name."""
    name = name.strip()

    # Common variant normalization
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

    # Replace special dashes
    name = re.sub(r'[–—]', '-', name)

    return name


def get_wikidata_search_name(name: str) -> str:
    """Convert a local name to the Wikidata search name.

    Example: "Hauts-Bassins" -> "Guiriko"
    """
    # Check mapping (case-insensitive)
    if name in LOCAL_TO_WIKIDATA_NAMES:
        return LOCAL_TO_WIKIDATA_NAMES[name]

    # Try case variants
    name_lower = name.lower()
    for local_name, wikidata_name in LOCAL_TO_WIKIDATA_NAMES.items():
        if local_name.lower() == name_lower:
            return wikidata_name

    return name


def get_local_display_name(wikidata_name: str) -> str:
    """Convert a Wikidata name to the local display name.

    Example: "Guiriko" -> "Hauts-Bassins"
    """
    if wikidata_name in WIKIDATA_TO_LOCAL_NAMES:
        return WIKIDATA_TO_LOCAL_NAMES[wikidata_name]
    return wikidata_name


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance (km) between two geographic points
    using the Haversine formula.
    """
    R = 6371  # Earth radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def parse_wikidata_coordinates(coord_string: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse Wikidata coordinates from 'Point(lon lat)' format."""
    if not coord_string:
        return None, None

    match = re.search(r'Point\(([^\s]+)\s+([^\)]+)\)', coord_string)
    if match:
        lon = float(match.group(1))
        lat = float(match.group(2))
        return lat, lon
    return None, None


def classify_location_type(wikidata_type: str) -> str:
    """Classify a location type according to standard categories."""
    if not wikidata_type:
        return "autre"

    type_lower = wikidata_type.lower()

    for pattern, classification in TYPE_MAPPING.items():
        if pattern in type_lower:
            return classification

    # Additional patterns
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
    """Determine the final location type.

    If Wikidata returns a generic type ("autre" or "ville"), the original
    label from the input DataFrame is preserved instead.

    Args:
        wikidata_type: Raw type returned by Wikidata.
        original_label: Original label from the input DataFrame.

    Returns:
        The type to use for this location.
    """
    classified_type = classify_location_type(wikidata_type)

    # If Wikidata type is generic and an original label exists, keep it
    if classified_type in TYPES_TO_REPLACE_WITH_ORIGINAL and original_label:
        return original_label

    return classified_type

# =============================================================================
# MAIN CLASS - WIKIDATA CLIENT
# =============================================================================

class WikidataGeoClient:
    """Client for querying Wikidata to retrieve geographic data."""

    def __init__(self):
        self.sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
        self.sparql.setReturnFormat(JSON)
        self.sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
        self.cache = {}

    def _execute_query(self, query: str) -> Optional[Dict]:
        """Execute a SPARQL query with error handling."""
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            time.sleep(REQUEST_DELAY_SECONDS)
            return results
        except Exception as e:
            logger.warning(f"SPARQL error: {e}")
            time.sleep(REQUEST_DELAY_SECONDS * 2)
            return None

    def search_location(self, location_name: str) -> Optional[Dict]:
        """Search for a location on Wikidata by name.

        Returns the Wikidata ID, coordinates, and type.
        """
        clean_name = clean_location_name(location_name)
        search_name = get_wikidata_search_name(clean_name)

        # Check cache (using original name to avoid duplicates)
        cache_key = f"search_{clean_name.lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if search_name != clean_name:
            logger.info(f"    [Mapping] '{clean_name}' -> '{search_name}' (Wikidata name)")

        # Query 1: Search with Burkina Faso filter
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

        # Query 2: Broader search (without country filter)
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
            # Prioritize Burkina Faso results
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

            # If no Burkina result, take the first one
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
        """Retrieve the full administrative hierarchy for an entity.

        Uses P131 (located in the administrative territorial entity).
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

            # Walk through hierarchy levels and identify their type
            for i in range(1, 5):
                label = result.get(f"parent{i}Label", {}).get("value", "")
                type_label = result.get(f"parent{i}TypeLabel", {}).get("value", "").lower()

                if label:
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
        """Find nearby spatial entities within a given radius.

        Uses the Wikidata geospatial service. Neighbors retain the Wikidata
        type since they are newly discovered locations not present in the
        input DataFrame.
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
                classified_type = classify_location_type(entity_type)

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
# MAIN ENRICHMENT FUNCTION
# =============================================================================

def enrich_location(client: WikidataGeoClient, location_name: str, location_label: str) -> LocationInfo:
    """Enrich a location with all its geographic information.

    Args:
        client: Wikidata client instance.
        location_name: Name of the location.
        location_label: Original label from the input DataFrame (e.g., "departement", "province").

    Returns:
        LocationInfo populated with enriched data.
    """
    clean_name = clean_location_name(location_name)
    info = LocationInfo(name=clean_name)

    # Check if external location
    if clean_name.lower() in EXTERNAL_LOCATIONS:
        info.hierarchy_pays = clean_name
        logger.info(f"  {clean_name}: Location outside Burkina Faso")
        return info

    # Special case: country itself
    if clean_name.lower() in ["burkina", "burkina faso"]:
        info.wikidata_id = BURKINA_FASO_QID
        info.location_type = "country"
        info.latitude = 12.2383
        info.longitude = -1.5616
        logger.info(f"  {clean_name}: Country (Burkina Faso)")
        return info

    # Special case: road axes
    if "axe" in clean_name.lower():
        logger.info(f"  {clean_name}: Road axis (not geolocatable)")
        return info

    # Search on Wikidata
    logger.info(f"  Searching: {clean_name}")
    search_result = client.search_location(clean_name)

    if search_result:
        info.wikidata_id = search_result["wikidata_id"]

        # Determine geographic type
        wikidata_type_raw = search_result.get("wikidata_type", "")
        wikidata_type_classified = classify_location_type(wikidata_type_raw)

        # If Wikidata returns "autre" or "ville", keep the original DataFrame label
        if wikidata_type_classified in TYPES_TO_REPLACE_WITH_ORIGINAL and location_label:
            info.location_type = location_label
            logger.info(f"    -> Wikidata type '{wikidata_type_classified}' ('{wikidata_type_raw}') -> kept original label '{location_label}'")
        else:
            info.location_type = wikidata_type_classified

        # Coordinates
        if search_result.get("coordinates"):
            info.latitude, info.longitude = parse_wikidata_coordinates(search_result["coordinates"])
            logger.info(f"    -> Found: {info.wikidata_id} ({info.latitude}, {info.longitude}) - type: {info.location_type}")
        else:
            logger.info(f"    -> Found: {info.wikidata_id} (no coordinates) - type: {info.location_type}")

        # Administrative hierarchy
        hierarchy = client.get_administrative_hierarchy(info.wikidata_id)
        info.hierarchy_departement = hierarchy.get("departement")
        info.hierarchy_province = hierarchy.get("province")
        info.hierarchy_region = hierarchy.get("region")

        if hierarchy.get("region"):
            logger.info(f"    -> Hierarchy: {info.hierarchy_departement} -> {info.hierarchy_province} -> {info.hierarchy_region}")

        # Neighbors (if coordinates available)
        if info.latitude and info.longitude:
            neighbors = client.get_nearby_entities(info.latitude, info.longitude, NEIGHBOR_RADIUS_KM)
            # Exclude the location itself
            info.neighbors = [n for n in neighbors if n["nom"].lower() != clean_name.lower()]
            logger.info(f"    -> {len(info.neighbors)} neighbors within {NEIGHBOR_RADIUS_KM}km")
    else:
        # Location not found on Wikidata: keep the original label
        if location_label:
            info.location_type = location_label
            logger.warning(f"    -> Not found on Wikidata, kept original label '{location_label}'")
        else:
            logger.warning(f"    -> Not found on Wikidata")

    return info


def process_dataframe(df: pd.DataFrame, location_column: str = 'location') -> pd.DataFrame:
    """Process a DataFrame and enrich each location."""
    client = WikidataGeoClient()

    # Extract unique locations
    unique_locations = {}
    for idx, row in df.iterrows():
        try:
            loc = ast.literal_eval(row[location_column])
            key = (loc['text'].strip(), loc.get('label', ''))
            if key not in unique_locations:
                unique_locations[key] = loc
        except:
            pass

    logger.info(f"Processing {len(unique_locations)} unique locations...")

    # Process each unique location
    location_data = {}
    for i, ((name, label), loc_info) in enumerate(unique_locations.items()):
        logger.info(f"\n[{i+1}/{len(unique_locations)}] {name} ({label})")
        location_data[(name, label)] = enrich_location(client, name, label)

    # Create new columns
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

    # Fill columns for each row
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

            # Count neighbor types
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

    # Add columns to DataFrame
    for col_name, col_data in new_columns.items():
        df[col_name] = col_data

    return df


def print_statistics(df: pd.DataFrame):
    """Print enrichment statistics."""
    print("\n" + "=" * 70)
    print("ENRICHMENT STATISTICS")
    print("=" * 70)

    print(f"\nGENERAL")
    print(f"   Total records: {len(df)}")
    print(f"   Unique locations: {df['lieu_nettoye'].nunique()}")

    print(f"\nGEOLOCATION")
    print(f"   With coordinates: {df['latitude'].notna().sum()} ({df['latitude'].notna().sum()*100/len(df):.1f}%)")
    print(f"   With Wikidata ID: {df['wikidata_id'].notna().sum()} ({df['wikidata_id'].notna().sum()*100/len(df):.1f}%)")

    print(f"\nADMINISTRATIVE HIERARCHY")
    print(f"   With region: {df['hierarchy_region'].notna().sum()} ({df['hierarchy_region'].notna().sum()*100/len(df):.1f}%)")
    print(f"   With province: {df['hierarchy_province'].notna().sum()} ({df['hierarchy_province'].notna().sum()*100/len(df):.1f}%)")
    print(f"   With departement: {df['hierarchy_departement'].notna().sum()} ({df['hierarchy_departement'].notna().sum()*100/len(df):.1f}%)")

    print(f"\nNEIGHBORHOOD ({NEIGHBOR_RADIUS_KM} KM)")
    print(f"   Locations with neighbors: {(df['neighbors_10km_count'] > 0).sum()}")
    print(f"   Average neighbors/location: {df['neighbors_10km_count'].mean():.1f}")
    print(f"   Max neighbors: {df['neighbors_10km_count'].max()}")

    # Distribution by region
    print(f"\nDISTRIBUTION BY REGION")
    region_counts = df['hierarchy_region'].value_counts().head(10)
    for region, count in region_counts.items():
        print(f"   {region or 'Undefined'}: {count}")

    # Distribution by geographic type
    print(f"\nDISTRIBUTION BY GEOGRAPHIC TYPE")
    type_counts = df['type_geo'].value_counts()
    for geo_type, count in type_counts.items():
        print(f"   {geo_type or 'Undefined'}: {count}")


def main():
    """Main entry point."""
    print("=" * 70)
    print("GEOGRAPHIC ENRICHMENT VIA WIKIDATA")
    print("Burkina Faso - Food Security Analysis")
    print("=" * 70)

    # Command-line arguments
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        output_file = input_file.replace('.csv', '_enriched.csv')
    else:
        input_file = 'talamasca.csv'
        output_file = 'df_processed_enriched.csv'

    print(f"\nInput file: {input_file}")
    print(f"Output file: {output_file}")

    # Load data
    print("\n1. Loading data...")
    try:
        df = pd.read_csv(input_file)
        print(f"   {len(df)} records loaded")
    except FileNotFoundError:
        print(f"   File not found: {input_file}")
        sys.exit(1)

    # Check location column
    if 'location' not in df.columns:
        print("   Column 'location' not found in CSV")
        sys.exit(1)

    # Process the DataFrame
    print("\n2. Enriching via Wikidata...")
    print("   (This may take several minutes depending on the number of locations)")
    df_enriched = process_dataframe(df)

    # Save
    print(f"\n3. Saving: {output_file}")
    df_enriched.to_csv(output_file, index=False, encoding='utf-8')
    print("   File saved")

    # Statistics
    print_statistics(df_enriched)

    print("\n" + "=" * 70)
    print("ENRICHMENT COMPLETE")
    print("=" * 70)

    return df_enriched


if __name__ == "__main__":
    df_result = main()
