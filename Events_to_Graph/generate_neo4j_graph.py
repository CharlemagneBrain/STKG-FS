import pandas as pd
import json
import re
import sys
import ast
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

REQUIRED_COLUMNS = [
    'term', 'theme', 'concept', 'phase', 'lieu_nettoye', 'type_geo',
    'date', 'date_x', 'article_id', 'contexte_enrichi', 'wikidata_id', 'label',
    'latitude', 'longitude', 'hierarchy_region', 'hierarchy_province',
    'hierarchy_departement', 'neighbors_10km', 'publication_date'
]

# Mapping des anciens noms de régions vers les nouveaux noms
# Cascades est l'ancien nom de la région Tannounyan
REGION_NAME_MAPPING = {
    'Cascades': 'Tannounyan',
    # Ajouter d'autres mappings si nécessaire
}

# =============================================================================
# CORRECTION V5: Mapping des noms de régions internes vers officiels
# Kadiogo est une PROVINCE de la région Centre, pas une région
# =============================================================================
REGION_INTERNAL_TO_OFFICIAL = {
    # Noms internes (bassins versants) → Noms officiels
    'Bankui': 'Boucle du Mouhoun',
    'Kuilsé': 'Centre-Nord',
    'Nakambé': 'Centre-Est',
    'Nakambé (région)': 'Centre-Est',
    'Nazinon': 'Centre-Sud',
    'Oubri': 'Plateau Central',
    'Tannounyan': 'Cascades',
    'Yaadga': 'Nord',
    'Djôrô': 'Sud-Ouest',
    'Plateau': 'Plateau Central',
    'Nando': 'Centre-Ouest',
    'Liptako': 'Sahel',
    'Goulmou': 'Est',
    'Guiriko': 'Hauts-Bassins',
    # CORRECTION V5: Kadiogo est une province, pas une région
    # Dans le CSV, hierarchy_region='Kadiogo' est incorrect
    # La vraie région est 'Centre'
    'Kadiogo': 'Centre',
}

# Correction des types géographiques incorrects
TYPE_GEO_CORRECTIONS = {
    'Cascades': 'region',  # Cascades était marqué comme province mais c'est une région
}

# =============================================================================
# UTILITIES FUNCTIONS
# =============================================================================

def escape_cypher_string(s: str) -> str:
    """Échappe les caractères spéciaux pour Cypher."""
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.replace("\\", "\\\\")
    s = s.replace("'", "\\'")
    s = s.replace('"', '\\"')
    s = s.replace("\n", " ")
    s = s.replace("\r", " ")
    return s


def normalize_location_name(name: str) -> str:
    """Normalise le nom d'un lieu (gère les anciens noms de régions)."""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    return REGION_NAME_MAPPING.get(name, name)


def normalize_region_name(region: str) -> str:
    """Convertit un nom de région interne vers le nom officiel."""
    if pd.isna(region):
        return ""
    region = str(region).strip()
    return REGION_INTERNAL_TO_OFFICIAL.get(region, region)


def correct_type_geo(lieu: str, type_geo: str) -> str:
    """Corrige le type géographique si nécessaire."""
    if pd.isna(type_geo):
        return ""
    type_geo = str(type_geo).strip()
    # Vérifier si ce lieu nécessite une correction de type
    if lieu in TYPE_GEO_CORRECTIONS:
        return TYPE_GEO_CORRECTIONS[lieu]
    return type_geo


def parse_date_column(date_val: Any) -> Tuple[Optional[str], Optional[str]]:
    """Parse la colonne 'date' qui contient un dict JSON."""
    if pd.isna(date_val):
        return None, None
    
    date_str = str(date_val)
    
    try:
        if date_str.startswith('{'):
            date_dict = ast.literal_eval(date_str)
            if isinstance(date_dict, dict):
                text = date_dict.get('text', '')
                return text, None
    except:
        pass
    
    return date_str, None


def parse_normalized_date(date_str: str) -> Optional[datetime]:
    """Parse une date normalisée (date_x) en différents formats."""
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    formats = [
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%Y/%m/%d',
        '%d-%m-%Y',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            pass
    
    match = re.search(r'(\d{4})', date_str)
    if match:
        try:
            return datetime(int(match.group(1)), 1, 1)
        except:
            pass
    
    return None


def format_event_title(term: str, lieu: str, dt: Optional[datetime]) -> str:
    """Crée un titre formaté: risk_lieu_year-month"""
    term_clean = re.sub(r'[^a-zA-Z0-9àâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ]', '_', str(term))
    lieu_clean = re.sub(r'[^a-zA-Z0-9àâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ]', '_', str(lieu))
    
    if dt:
        year_month = f"{dt.year}-{dt.month:02d}"
    else:
        year_month = "unknown"
    
    return f"{term_clean}_{lieu_clean}_{year_month}"


# =============================================================================
# PRE-PROCESSING: Normalisation des données
# =============================================================================

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prétraitement du DataFrame: normalisation des lieux et correction des types."""
    df = df.copy()
    
    # Normaliser les noms de lieux (Cascades → Tannounyan)
    df['lieu_nettoye_original'] = df['lieu_nettoye']
    df['lieu_nettoye'] = df['lieu_nettoye'].apply(normalize_location_name)
    
    # Corriger les types géographiques
    df['type_geo_original'] = df['type_geo']
    df['type_geo'] = df.apply(
        lambda row: correct_type_geo(row['lieu_nettoye_original'], row['type_geo']), 
        axis=1
    )
    
    # Normaliser aussi hierarchy_region si nécessaire
    df['hierarchy_region'] = df['hierarchy_region'].apply(normalize_location_name)
    
    # V5: Normaliser les noms de régions internes vers officiels
    df['hierarchy_region_original'] = df['hierarchy_region']
    df['hierarchy_region'] = df['hierarchy_region'].apply(normalize_region_name)
    
    # Log des corrections effectuées
    corrections = df[df['lieu_nettoye'] != df['lieu_nettoye_original']]
    if len(corrections) > 0:
        print(f"  Corrections de noms de lieux: {len(corrections)} entrées")
        for old, new in corrections[['lieu_nettoye_original', 'lieu_nettoye']].drop_duplicates().values:
            print(f"    - '{old}' → '{new}'")
    
    type_corrections = df[df['type_geo'] != df['type_geo_original']]
    if len(type_corrections) > 0:
        print(f"  Corrections de types géo: {len(type_corrections)} entrées")
        for _, row in type_corrections[['lieu_nettoye_original', 'type_geo_original', 'type_geo']].drop_duplicates().iterrows():
            print(f"    - '{row['lieu_nettoye_original']}': '{row['type_geo_original']}' → '{row['type_geo']}'")
    
    # V5: Log des corrections de régions
    region_corrections = df[df['hierarchy_region'] != df['hierarchy_region_original']]
    if len(region_corrections) > 0:
        print(f"  Corrections de noms de régions: {len(region_corrections)} entrées")
        for old, new in region_corrections[['hierarchy_region_original', 'hierarchy_region']].drop_duplicates().values:
            if pd.notna(old) and str(old).strip():
                print(f"    - '{old}' → '{new}'")
    
    return df


# =============================================================================
# Generate Cypher Graph - Version 5
# =============================================================================

class Neo4jGraphGeneratorV5:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.events = {}          # event_id -> event_data
        self.risks = {}           # risk_name -> risk_data
        self.locations = {}       # location_name -> location_data
        self.times = {}           # date_normalized -> time_data
        
        # Relations
        self.event_risk_relations = []      # (event_id, risk_name, article_count)
        self.event_location_relations = []
        self.event_time_relations = []
        self.recurrent_relations = []       # (e1_id, e2_id, duration_days)
        self.synchronous_relations = []     # V5: (e1_id, e2_id) - sera créé dans les deux sens
        self.precedes_relations = []
        self.location_hierarchy_relations = []
        self.location_neighbor_relations = []
    
    def process(self):
        """Traite le DataFrame et génère le graphe."""
        print("1. Extraction et agrégation des entités...")
        self._extract_and_aggregate_entities()
        
        print("2. Création des relations temporelles (IS_RECURRENT, IS_SYNCHRONOUS, PRECEDES)...")
        self._create_temporal_relations()
        
        print("3. Création des relations spatiales...")
        self._create_spatial_relations()
        
        self._print_statistics()
    
    def _extract_and_aggregate_entities(self):
        """Extrait les entités uniques avec agrégation correcte."""
        event_counter = 0
        
        grouped = self.df.groupby(['term', 'lieu_nettoye', 'date_x'])
        
        for (term, lieu, date_normalized), group in grouped:
            event_id = f"E{event_counter}"
            event_counter += 1
            
            # Collecter les informations
            articles = set()
            publication_dates_map = {}  # article_id -> publication_date
            contexts = []
            frequency_by_article = defaultdict(int)
            
            for _, row in group.iterrows():
                article_id = str(row['article_id'])
                articles.add(article_id)
                frequency_by_article[article_id] += 1
                
                # Date de publication par article
                pub_date = row.get('publication_date', '')
                if pd.notna(pub_date) and str(pub_date).strip():
                    publication_dates_map[article_id] = str(pub_date).strip()
                
                # Contexte
                context = row.get('contexte_enrichi', '')
                if pd.notna(context) and str(context).strip():
                    context_escaped = escape_cypher_string(str(context)[:500])
                    if context_escaped not in [c.get('context', '') for c in contexts]:
                        contexts.append({
                            "article_id": article_id,
                            "context": context_escaped
                        })
            
            # Parser la date normalisée
            dt = parse_normalized_date(date_normalized)
            date_str = str(date_normalized) if pd.notna(date_normalized) else ""
            
            # Créer le titre: risk_lieu_year-month
            title = format_event_title(term, lieu, dt)
            
            # Créer l'événement (avec date)
            first_row = group.iloc[0]
            self.events[event_id] = {
                "id": event_id,
                "title": escape_cypher_string(title),
                "term": escape_cypher_string(term),
                "lieu": escape_cypher_string(lieu),
                "date": date_str,  # Date normalisée de l'événement
                "frequency_total": len(group),
                "frequency_by_article": dict(frequency_by_article),
                "article_count": len(articles),  # Gardé pour la relation CONCERNS
                "publication_dates": publication_dates_map,  # {"article_id": "date", ...}
                "contexts": contexts[:10],
                "_date_str": date_str,  # Pour les relations temporelles
                "_dt": dt  # Pour les relations temporelles
            }
            
            # Créer/mettre à jour le nœud Risk
            risk_key = escape_cypher_string(term)
            if risk_key not in self.risks:
                self.risks[risk_key] = {
                    "name": risk_key,
                    "theme": escape_cypher_string(first_row.get('theme', '')),
                    "concept": escape_cypher_string(first_row.get('concept', '')),
                    "phase": escape_cypher_string(first_row.get('phase', ''))
                }
            
            # Créer/mettre à jour le nœud Location
            location_key = escape_cypher_string(lieu)
            if location_key not in self.locations:
                self.locations[location_key] = {
                    "name": location_key,
                    "type": escape_cypher_string(first_row.get('type_geo', '')),
                    "wikidata_id": escape_cypher_string(first_row.get('wikidata_id', '')),
                    "latitude": first_row.get('latitude') if pd.notna(first_row.get('latitude')) else None,
                    "longitude": first_row.get('longitude') if pd.notna(first_row.get('longitude')) else None,
                    "region": escape_cypher_string(first_row.get('hierarchy_region', '')),
                    "province": escape_cypher_string(first_row.get('hierarchy_province', '')),
                    "departement": escape_cypher_string(first_row.get('hierarchy_departement', '')),
                    "neighbors_raw": first_row.get('neighbors_10km', '[]')
                }
            
            # Créer/mettre à jour le nœud Time
            date_key = escape_cypher_string(date_str)
            if date_key and date_key not in self.times:
                self.times[date_key] = {
                    "datetime": date_key,
                    "year": dt.year if dt else None,
                    "month": dt.month if dt else None,
                    "day": dt.day if dt else None
                }
            
            # Relations de base (avec article_count pour CONCERNS)
            article_count = len(articles)
            self.event_risk_relations.append((event_id, risk_key, article_count))
            if location_key:
                self.event_location_relations.append((event_id, location_key))
            if date_key:
                self.event_time_relations.append((event_id, date_key))
    
    def _create_temporal_relations(self):
        """Crée les relations temporelles entre événements."""
        # Grouper les événements par lieu et risque (pour IS_RECURRENT)
        events_by_lieu_risk = defaultdict(list)
        # Grouper par lieu et date (pour IS_SYNCHRONOUS)
        events_by_lieu_date = defaultdict(list)
        # Grouper par lieu et année (pour PRECEDES)
        events_by_lieu_year = defaultdict(list)
        
        for event_id, event_data in self.events.items():
            lieu = event_data["lieu"]
            risk = event_data["term"]
            dt = event_data["_dt"]
            date_str = event_data["_date_str"]
            
            if dt:
                events_by_lieu_risk[(lieu, risk)].append((event_id, dt))
                events_by_lieu_date[(lieu, date_str)].append((event_id, dt, risk))
                events_by_lieu_year[(lieu, dt.year)].append((event_id, dt, risk))
        
        # === IS_RECURRENT (même lieu, même risque, dates différentes) ===
        # Avec duration_days pour stocker l'intervalle
        seen_recurrent = set()
        for (lieu, risk), events in events_by_lieu_risk.items():
            if len(events) > 1:
                events_sorted = sorted(events, key=lambda x: x[1])
                for i in range(len(events_sorted) - 1):
                    e1_id, e1_dt = events_sorted[i]
                    e2_id, e2_dt = events_sorted[i + 1]
                    
                    key = tuple(sorted([e1_id, e2_id]))
                    if key not in seen_recurrent:
                        # Calculer la durée en jours entre les occurrences
                        duration_days = (e2_dt - e1_dt).days
                        self.recurrent_relations.append((e1_id, e2_id, duration_days))
                        seen_recurrent.add(key)
        
        # === IS_SYNCHRONOUS (même lieu, même date, risques différents) ===
        # V5: Relation SANS DIRECTION - on stocke une seule fois (e1, e2)
        # mais on créera les deux sens dans le Cypher
        seen_synchronous = set()
        for (lieu, date_str), events in events_by_lieu_date.items():
            if len(events) > 1:
                # Tous les événements du même lieu à la même date
                for i in range(len(events)):
                    for j in range(i + 1, len(events)):
                        e1_id, e1_dt, e1_risk = events[i]
                        e2_id, e2_dt, e2_risk = events[j]
                        
                        # Seulement si risques différents (sinon ce serait le même événement)
                        if e1_risk != e2_risk:
                            key = tuple(sorted([e1_id, e2_id]))
                            if key not in seen_synchronous:
                                self.synchronous_relations.append((e1_id, e2_id))
                                seen_synchronous.add(key)
        
        # === PRECEDES (même lieu, même année, risques différents, dates différentes) ===
        seen_precedes = set()
        for (lieu, year), events in events_by_lieu_year.items():
            if len(events) > 1:
                events_sorted = sorted(events, key=lambda x: x[1])
                for i in range(len(events_sorted)):
                    for j in range(i + 1, len(events_sorted)):
                        e1_id, e1_dt, e1_risk = events_sorted[i]
                        e2_id, e2_dt, e2_risk = events_sorted[j]
                        
                        # Seulement si risques différents ET dates différentes
                        # (si même date → IS_SYNCHRONOUS, pas PRECEDES)
                        if e1_risk != e2_risk and e1_dt != e2_dt:
                            key = (e1_id, e2_id)
                            if key not in seen_precedes:
                                # Calculer la durée en jours
                                duration_days = (e2_dt - e1_dt).days
                                self.precedes_relations.append((e1_id, e2_id, duration_days))
                                seen_precedes.add(key)
    
    def _create_spatial_relations(self):
        """Crée les relations spatiales."""
        parent_locations = {}
        
        for loc_name, loc_data in list(self.locations.items()):
            dept = loc_data.get("departement", "")
            if dept and dept != loc_name and dept not in self.locations:
                parent_locations[dept] = {
                    "name": dept,
                    "type": "departement",
                    "wikidata_id": "",
                    "latitude": None,
                    "longitude": None,
                    "region": loc_data.get("region", ""),
                    "province": loc_data.get("province", ""),
                    "departement": dept,
                    "neighbors_raw": "[]"
                }
            
            prov = loc_data.get("province", "")
            if prov and prov != loc_name and prov not in self.locations:
                parent_locations[prov] = {
                    "name": prov,
                    "type": "province",
                    "wikidata_id": "",
                    "latitude": None,
                    "longitude": None,
                    "region": loc_data.get("region", ""),
                    "province": prov,
                    "departement": "",
                    "neighbors_raw": "[]"
                }
            
            reg = loc_data.get("region", "")
            if reg and reg != loc_name and reg not in self.locations:
                parent_locations[reg] = {
                    "name": reg,
                    "type": "region",
                    "wikidata_id": "",
                    "latitude": None,
                    "longitude": None,
                    "region": reg,
                    "province": "",
                    "departement": "",
                    "neighbors_raw": "[]"
                }
        
        self.locations.update(parent_locations)
        
        # Relations hiérarchiques
        seen_hierarchy = set()
        for loc_name, loc_data in self.locations.items():
            loc_type = loc_data.get("type", "")
            
            if loc_type in ["village"]:
                dept = loc_data.get("departement", "")
                if dept and dept in self.locations:
                    key = (loc_name, dept, "IS_FROM_DEPARTEMENT")
                    if key not in seen_hierarchy:
                        self.location_hierarchy_relations.append((loc_name, dept, "IS_FROM_DEPARTEMENT"))
                        seen_hierarchy.add(key)
            
            if loc_type == "departement":
                prov = loc_data.get("province", "")
                if prov and prov in self.locations:
                    key = (loc_name, prov, "IS_FROM_PROVINCE")
                    if key not in seen_hierarchy:
                        self.location_hierarchy_relations.append((loc_name, prov, "IS_FROM_PROVINCE"))
                        seen_hierarchy.add(key)
            
            if loc_type == "province":
                reg = loc_data.get("region", "")
                if reg and reg in self.locations:
                    key = (loc_name, reg, "IS_FROM_REGION")
                    if key not in seen_hierarchy:
                        self.location_hierarchy_relations.append((loc_name, reg, "IS_FROM_REGION"))
                        seen_hierarchy.add(key)
            
            if not loc_type or loc_type == "autre":
                dept = loc_data.get("departement", "")
                prov = loc_data.get("province", "")
                reg = loc_data.get("region", "")
                
                if dept and dept != loc_name and dept in self.locations:
                    key = (loc_name, dept, "IS_FROM_DEPARTEMENT")
                    if key not in seen_hierarchy:
                        self.location_hierarchy_relations.append((loc_name, dept, "IS_FROM_DEPARTEMENT"))
                        seen_hierarchy.add(key)
                elif prov and prov != loc_name and prov in self.locations:
                    key = (loc_name, prov, "IS_FROM_PROVINCE")
                    if key not in seen_hierarchy:
                        self.location_hierarchy_relations.append((loc_name, prov, "IS_FROM_PROVINCE"))
                        seen_hierarchy.add(key)
                elif reg and reg != loc_name and reg in self.locations:
                    key = (loc_name, reg, "IS_FROM_REGION")
                    if key not in seen_hierarchy:
                        self.location_hierarchy_relations.append((loc_name, reg, "IS_FROM_REGION"))
                        seen_hierarchy.add(key)
        
        # Relations IS_NEAR_TO
        seen_neighbors = set()
        locations_copy = dict(self.locations)
        for loc_name, loc_data in locations_copy.items():
            neighbors_raw = loc_data.get("neighbors_raw", "[]")
            
            if neighbors_raw and neighbors_raw != "[]":
                try:
                    neighbors = json.loads(neighbors_raw)
                    for neighbor in neighbors[:10]:
                        neighbor_name = escape_cypher_string(neighbor.get("nom", ""))
                        distance = neighbor.get("distance_km", 0)
                        neighbor_type = neighbor.get("type", "autre")
                        
                        if neighbor_name:
                            if neighbor_name not in self.locations:
                                self.locations[neighbor_name] = {
                                    "name": neighbor_name,
                                    "type": escape_cypher_string(neighbor_type),
                                    "wikidata_id": "",
                                    "latitude": neighbor.get("lat"),
                                    "longitude": neighbor.get("lon"),
                                    "region": "",
                                    "province": "",
                                    "departement": "",
                                    "neighbors_raw": "[]"
                                }
                            
                            key = tuple(sorted([loc_name, neighbor_name]))
                            if key not in seen_neighbors and loc_name != neighbor_name:
                                self.location_neighbor_relations.append((loc_name, neighbor_name, distance))
                                seen_neighbors.add(key)
                except:
                    pass
    
    def _print_statistics(self):
        """Affiche les statistiques du graphe."""
        print("\n" + "=" * 60)
        print("STATISTIQUES DU GRAPHE (V5)")
        print("=" * 60)
        print(f"Nœuds Event: {len(self.events)}")
        print(f"Nœuds Risk: {len(self.risks)}")
        print(f"Nœuds Location: {len(self.locations)}")
        print(f"Nœuds Time: {len(self.times)}")
        
        total_mentions = sum(e['frequency_total'] for e in self.events.values())
        avg_mentions = total_mentions / len(self.events) if self.events else 0
        max_mentions = max((e['frequency_total'] for e in self.events.values()), default=0)
        multi_article = sum(1 for e in self.events.values() if e['article_count'] > 1)
        
        print(f"\n--- Statistiques de fréquence ---")
        print(f"Total mentions (avant agrégation): {total_mentions}")
        print(f"Moyenne mentions/événement: {avg_mentions:.2f}")
        print(f"Max mentions pour un événement: {max_mentions}")
        print(f"Événements mentionnés dans plusieurs articles: {multi_article}")
        
        print(f"\n--- Relations ---")
        print(f"Relations Event-Risk (CONCERNS avec article_count): {len(self.event_risk_relations)}")
        print(f"Relations Event-Location (LOCATED_IN): {len(self.event_location_relations)}")
        print(f"Relations Event-Time (OCCURRED_ON): {len(self.event_time_relations)}")
        print(f"Relations IS_RECURRENT (avec duration_days): {len(self.recurrent_relations)}")
        print(f"Relations IS_SYNCHRONOUS (bidirectionnelles): {len(self.synchronous_relations)} paires = {len(self.synchronous_relations) * 2} relations")
        print(f"Relations PRECEDES (avec duration_days): {len(self.precedes_relations)}")
        
        # Statistiques sur IS_RECURRENT
        if self.recurrent_relations:
            durations = [d for _, _, d in self.recurrent_relations]
            print(f"  - Durée min récurrence: {min(durations)} jours")
            print(f"  - Durée max récurrence: {max(durations)} jours")
            print(f"  - Durée moyenne récurrence: {sum(durations)/len(durations):.1f} jours")
        
        hierarchy_counts = {}
        for _, _, rel_type in self.location_hierarchy_relations:
            hierarchy_counts[rel_type] = hierarchy_counts.get(rel_type, 0) + 1
        print(f"Relations IS_FROM_DEPARTEMENT: {hierarchy_counts.get('IS_FROM_DEPARTEMENT', 0)}")
        print(f"Relations IS_FROM_PROVINCE: {hierarchy_counts.get('IS_FROM_PROVINCE', 0)}")
        print(f"Relations IS_FROM_REGION: {hierarchy_counts.get('IS_FROM_REGION', 0)}")
        print(f"Relations IS_NEAR_TO: {len(self.location_neighbor_relations)}")
        
    
    def generate_cypher(self) -> str:
        """Génère le code Cypher complet."""
        lines = []
        
    
        
        # Contraintes et index
        lines.append("// ===== CONTRAINTES ET INDEX =====")
        lines.append("CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE;")
        lines.append("CREATE CONSTRAINT risk_name IF NOT EXISTS FOR (r:Risk) REQUIRE r.name IS UNIQUE;")
        lines.append("CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE;")
        lines.append("CREATE CONSTRAINT time_datetime IF NOT EXISTS FOR (t:Time) REQUIRE t.datetime IS UNIQUE;")
        lines.append("CREATE INDEX event_term IF NOT EXISTS FOR (e:Event) ON (e.term);")
        lines.append("CREATE INDEX event_title IF NOT EXISTS FOR (e:Event) ON (e.title);")
        lines.append("CREATE INDEX location_type IF NOT EXISTS FOR (l:Location) ON (l.type);")
        lines.append("CREATE INDEX time_year IF NOT EXISTS FOR (t:Time) ON (t.year);")
        lines.append("")
        
        # Nœuds Risk
        lines.append("// ===== NŒUDS RISK =====")
        for risk_name, risk_data in self.risks.items():
            lines.append(f"MERGE (r:Risk {{name: '{risk_data['name']}'}})")
            lines.append(f"SET r.theme = '{risk_data['theme']}', r.concept = '{risk_data['concept']}', r.phase = '{risk_data['phase']}';")
        lines.append("")
        
        # Nœuds Location
        lines.append("// ===== NŒUDS LOCATION =====")
        for loc_name, loc_data in self.locations.items():
            lat_str = f", l.latitude = {loc_data['latitude']}" if loc_data['latitude'] else ""
            lon_str = f", l.longitude = {loc_data['longitude']}" if loc_data['longitude'] else ""
            
            lines.append(f"MERGE (l:Location {{name: '{loc_data['name']}'}})")
            lines.append(f"SET l.type = '{loc_data['type']}', l.wikidata_id = '{loc_data['wikidata_id']}'{lat_str}{lon_str};")
        lines.append("")
        
        # Nœuds Time
        lines.append("// ===== NŒUDS TIME =====")
        for date_str, time_data in self.times.items():
            year_str = f", t.year = {time_data['year']}" if time_data['year'] else ""
            month_str = f", t.month = {time_data['month']}" if time_data['month'] else ""
            day_str = f", t.day = {time_data['day']}" if time_data['day'] else ""
            
            lines.append(f"MERGE (t:Time {{datetime: '{time_data['datetime']}'}})")
            lines.append(f"SET t.datetime = '{time_data['datetime']}'{year_str}{month_str}{day_str};")
        lines.append("")
        
        # Nœuds Event
        lines.append("// ===== NŒUDS EVENT =====")
        for event_id, event_data in self.events.items():
            frequency_by_article_json = json.dumps(event_data['frequency_by_article'], ensure_ascii=False).replace("'", "\\'")
            publication_dates_json = json.dumps(event_data['publication_dates'], ensure_ascii=False).replace("'", "\\'")
            contexts_json = json.dumps(event_data['contexts'], ensure_ascii=False).replace("'", "\\'")
            
            lines.append(f"MERGE (e:Event {{id: '{event_id}'}})")
            lines.append(f"SET e.title = '{event_data['title']}',")
            lines.append(f"    e.term = '{event_data['term']}',")
            lines.append(f"    e.lieu = '{event_data['lieu']}',")
            lines.append(f"    e.date = '{event_data['date']}',")
            lines.append(f"    e.frequency_total = {event_data['frequency_total']},")
            lines.append(f"    e.frequency_by_article = '{frequency_by_article_json}',")
            lines.append(f"    e.publication_dates = '{publication_dates_json}',")
            lines.append(f"    e.contexts = '{contexts_json}';")
        lines.append("")
        
        # Relations Event -> Risk (CONCERNS avec article_count)
        lines.append("// ===== RELATIONS EVENT -> RISK (CONCERNS) =====")
        for event_id, risk_name, article_count in self.event_risk_relations:
            lines.append(f"MATCH (e:Event {{id: '{event_id}'}}), (r:Risk {{name: '{risk_name}'}})")
            lines.append(f"MERGE (e)-[:CONCERNS {{article_count: {article_count}}}]->(r);")
        lines.append("")
        
        # Relations Event -> Location
        lines.append("// ===== RELATIONS EVENT -> LOCATION (LOCATED_IN) =====")
        for event_id, loc_name in self.event_location_relations:
            lines.append(f"MATCH (e:Event {{id: '{event_id}'}}), (l:Location {{name: '{loc_name}'}})")
            lines.append("MERGE (e)-[:LOCATED_IN]->(l);")
        lines.append("")
        
        # Relations Event -> Time
        lines.append("// ===== RELATIONS EVENT -> TIME (OCCURRED_ON) =====")
        for event_id, date_str in self.event_time_relations:
            lines.append(f"MATCH (e:Event {{id: '{event_id}'}}), (t:Time {{datetime: '{date_str}'}})")
            lines.append("MERGE (e)-[:OCCURRED_ON]->(t);")
        lines.append("")
        
        # Relations IS_RECURRENT (avec duration_days)
        lines.append("// ===== RELATIONS IS_RECURRENT (même risque, même lieu, dates différentes) =====")
        lines.append("// duration_days: nombre de jours entre les deux occurrences")
        for e1_id, e2_id, duration_days in self.recurrent_relations:
            lines.append(f"MATCH (e1:Event {{id: '{e1_id}'}}), (e2:Event {{id: '{e2_id}'}})")
            lines.append(f"MERGE (e1)-[:IS_RECURRENT {{duration_days: {duration_days}}}]->(e2);")
        lines.append("")
        
        # Relations IS_SYNCHRONOUS (BIDIRECTIONNELLES - V5)
        lines.append("// ===== RELATIONS IS_SYNCHRONOUS (même lieu, même date, risques différents) =====")
        lines.append("// V5: Relation BIDIRECTIONNELLE - créée dans les deux sens")
        lines.append("// Deux événements survenus le même jour au même endroit sans relation de causalité")
        for e1_id, e2_id in self.synchronous_relations:
            lines.append(f"MATCH (e1:Event {{id: '{e1_id}'}}), (e2:Event {{id: '{e2_id}'}})")
            lines.append("MERGE (e1)-[:IS_SYNCHRONOUS]->(e2)")
            lines.append("MERGE (e2)-[:IS_SYNCHRONOUS]->(e1);")
        lines.append("")
        
        # Relations PRECEDES (avec durée en jours)
        lines.append("// ===== RELATIONS PRECEDES (même lieu, risques différents, dates différentes) =====")
        lines.append("// duration_days: nombre de jours entre les deux événements")
        for e1_id, e2_id, duration_days in self.precedes_relations:
            lines.append(f"MATCH (e1:Event {{id: '{e1_id}'}}), (e2:Event {{id: '{e2_id}'}})")
            lines.append(f"MERGE (e1)-[:PRECEDES {{duration_days: {duration_days}}}]->(e2);")
        lines.append("")
        
        # Relations hiérarchiques spatiales
        lines.append("// ===== RELATIONS HIÉRARCHIQUES SPATIALES =====")
        for loc1, loc2, rel_type in self.location_hierarchy_relations:
            lines.append(f"MATCH (l1:Location {{name: '{loc1}'}}), (l2:Location {{name: '{loc2}'}})")
            lines.append(f"MERGE (l1)-[:{rel_type}]->(l2);")
        lines.append("")
        
        # Relations IS_NEAR_TO
        lines.append("// ===== RELATIONS IS_NEAR_TO =====")
        for loc1, loc2, distance in self.location_neighbor_relations[:500]:
            lines.append(f"MATCH (l1:Location {{name: '{loc1}'}}), (l2:Location {{name: '{loc2}'}})")
            lines.append(f"MERGE (l1)-[:IS_NEAR_TO {{distance_km: {distance}}}]->(l2);")
        
        return "\n".join(lines)


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    
    
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        output_file = input_file.replace('.csv', '_neo4j_v5.cypher')
    else:
        input_file = 'greek_final.csv'
        output_file = 'knowledge_graph_v5.cypher'
    
    print(f"\nFichier d'entrée: {input_file}")
    print(f"Fichier de sortie: {output_file}")
    
    print("\nChargement des données...")
    df = pd.read_csv(input_file)
    print(f"  Total enregistrements: {len(df)}")
    
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"  ATTENTION: Colonnes manquantes: {missing_cols}")
    
    print("\nPrétraitement (normalisation des lieux et régions)...")
    df = preprocess_dataframe(df)
    
    print("\nFiltrage des données...")
    df_filtered = df[(df['wikidata_id'].notna()) & (df['label'] == 1)]
    print(f"  Après filtrage (wikidata_id non vide ET label=1): {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        print("ERREUR: Aucun enregistrement après filtrage!")
        sys.exit(1)
    
    print("\nGénération du graphe...")
    generator = Neo4jGraphGeneratorV5(df_filtered)
    generator.process()
    
    print("\nGénération du code Cypher...")
    cypher_code = generator.generate_cypher()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cypher_code)
    print(f"\nFichier Cypher sauvegardé: {output_file}")
    
    # CSV des événements
    events_csv = output_file.replace('.cypher', '_events.csv')
    events_data = []
    for event_id, event_data in generator.events.items():
        events_data.append({
            'event_id': event_id,
            'title': event_data['title'],
            'term': event_data['term'],
            'lieu': event_data['lieu'],
            'date': event_data['date'],
            'frequency_total': event_data['frequency_total'],
            'article_count': event_data['article_count'],
            'publication_dates': json.dumps(event_data['publication_dates'])
        })
    
    pd.DataFrame(events_data).to_csv(events_csv, index=False)
    print(f"CSV des événements: {events_csv}")
    
    return generator


if __name__ == "__main__":
    generator = main()