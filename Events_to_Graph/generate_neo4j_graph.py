import pandas as pd
import json
import re
import sys
import ast
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any


REQUIRED_COLUMNS = [
    'term', 'theme', 'concept', 'phase', 'lieu_nettoye', 'type_geo',
    'date', 'date_x', 'article_id', 'contexte_enrichi', 'wikidata_id', 'label',
    'latitude', 'longitude', 'hierarchy_region', 'hierarchy_province',
    'hierarchy_departement', 'neighbors_10km', 'publication_date'
]


REGION_NAME_MAPPING = {
    'Cascades': 'Tannounyan',
}


REGION_INTERNAL_TO_OFFICIAL = {
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
    'Kadiogo': 'Centre',
}

# Geographic type corrections
TYPE_GEO_CORRECTIONS = {
    'Cascades': 'region',
}


def escape_cypher_string(s: str) -> str:

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
    """Normalize a location name (handles old region names)."""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    return REGION_NAME_MAPPING.get(name, name)


def normalize_region_name(region: str) -> str:
    """Convert an internal region name to its official name."""
    if pd.isna(region):
        return ""
    region = str(region).strip()
    return REGION_INTERNAL_TO_OFFICIAL.get(region, region)


def correct_type_geo(lieu: str, type_geo: str) -> str:
    """Correct the geographic type if needed."""
    if pd.isna(type_geo):
        return ""
    type_geo = str(type_geo).strip()
    if lieu in TYPE_GEO_CORRECTIONS:
        return TYPE_GEO_CORRECTIONS[lieu]
    return type_geo


def parse_date_column(date_val: Any) -> Tuple[Optional[str], Optional[str]]:
    """Parse the 'date' column which contains a JSON dict."""
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
    """Parse a normalized date (date_x) in various formats."""
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
    """Create a formatted title: risk_location_year-month."""
    term_clean = re.sub(r'[^a-zA-Z0-9àâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ]', '_', str(term))
    lieu_clean = re.sub(r'[^a-zA-Z0-9àâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ]', '_', str(lieu))

    if dt:
        year_month = f"{dt.year}-{dt.month:02d}"
    else:
        year_month = "unknown"

    return f"{term_clean}_{lieu_clean}_{year_month}"




def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame: normalize location names and correct types."""
    df = df.copy()

    # Normalize location names (Cascades -> Tannounyan)
    df['lieu_nettoye_original'] = df['lieu_nettoye']
    df['lieu_nettoye'] = df['lieu_nettoye'].apply(normalize_location_name)

    # Correct geographic types
    df['type_geo_original'] = df['type_geo']
    df['type_geo'] = df.apply(
        lambda row: correct_type_geo(row['lieu_nettoye_original'], row['type_geo']),
        axis=1
    )

    # Normalize hierarchy_region
    df['hierarchy_region'] = df['hierarchy_region'].apply(normalize_location_name)

    # V5: Normalize internal region names to official names
    df['hierarchy_region_original'] = df['hierarchy_region']
    df['hierarchy_region'] = df['hierarchy_region'].apply(normalize_region_name)

    # Log applied corrections
    corrections = df[df['lieu_nettoye'] != df['lieu_nettoye_original']]
    if len(corrections) > 0:
        print(f"  Location name corrections: {len(corrections)} entries")
        for old, new in corrections[['lieu_nettoye_original', 'lieu_nettoye']].drop_duplicates().values:
            print(f"    - '{old}' -> '{new}'")

    type_corrections = df[df['type_geo'] != df['type_geo_original']]
    if len(type_corrections) > 0:
        print(f"  Geographic type corrections: {len(type_corrections)} entries")
        for _, row in type_corrections[['lieu_nettoye_original', 'type_geo_original', 'type_geo']].drop_duplicates().iterrows():
            print(f"    - '{row['lieu_nettoye_original']}': '{row['type_geo_original']}' -> '{row['type_geo']}'")

    # Log region name corrections
    region_corrections = df[df['hierarchy_region'] != df['hierarchy_region_original']]
    if len(region_corrections) > 0:
        print(f"  Region name corrections: {len(region_corrections)} entries")
        for old, new in region_corrections[['hierarchy_region_original', 'hierarchy_region']].drop_duplicates().values:
            if pd.notna(old) and str(old).strip():
                print(f"    - '{old}' -> '{new}'")

    return df




class Neo4jGraphGeneratorV5:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.events = {}
        self.risks = {}
        self.locations = {}
        self.times = {}

        # Relations
        self.event_risk_relations = []
        self.event_location_relations = []
        self.event_time_relations = []
        self.recurrent_relations = []
        self.synchronous_relations = []
        self.precedes_relations = []
        self.location_hierarchy_relations = []

    def process(self):

        self._extract_and_aggregate_entities()

        self._create_temporal_relations()

        self._create_spatial_relations()

        self._print_statistics()

    def _extract_and_aggregate_entities(self):

        event_counter = 0

        grouped = self.df.groupby(['term', 'lieu_nettoye', 'date_x'])

        for (term, lieu, date_normalized), group in grouped:
            event_id = f"E{event_counter}"
            event_counter += 1

            articles = set()
            publication_dates_map = {}
            contexts = []
            frequency_by_article = defaultdict(int)

            for _, row in group.iterrows():
                article_id = str(row['article_id'])
                articles.add(article_id)
                frequency_by_article[article_id] += 1

                pub_date = row.get('publication_date', '')
                if pd.notna(pub_date) and str(pub_date).strip():
                    publication_dates_map[article_id] = str(pub_date).strip()

                context = row.get('contexte_enrichi', '')
                if pd.notna(context) and str(context).strip():
                    context_escaped = escape_cypher_string(str(context)[:500])
                    if context_escaped not in [c.get('context', '') for c in contexts]:
                        contexts.append({
                            "article_id": article_id,
                            "context": context_escaped
                        })

            dt = parse_normalized_date(date_normalized)
            date_str = str(date_normalized) if pd.notna(date_normalized) else ""

            # Title format: risk_location_year-month
            title = format_event_title(term, lieu, dt)

            first_row = group.iloc[0]
            self.events[event_id] = {
                "id": event_id,
                "title": escape_cypher_string(title),
                "term": escape_cypher_string(term),
                "lieu": escape_cypher_string(lieu),
                "date": date_str,
                "frequency_total": len(group),
                "frequency_by_article": dict(frequency_by_article),
                "article_count": len(articles),
                "publication_dates": publication_dates_map,
                "contexts": contexts[:10],
                "_date_str": date_str,
                "_dt": dt
            }

            # Create/update Risk node
            risk_key = escape_cypher_string(term)
            if risk_key not in self.risks:
                self.risks[risk_key] = {
                    "name": risk_key,
                    "theme": escape_cypher_string(first_row.get('theme', '')),
                    "concept": escape_cypher_string(first_row.get('concept', '')),
                    "phase": escape_cypher_string(first_row.get('phase', ''))
                }

            # Create/update Location node
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

            # Create/update Time node
            date_key = escape_cypher_string(date_str)
            if date_key and date_key not in self.times:
                self.times[date_key] = {
                    "datetime": date_key,
                    "year": dt.year if dt else None,
                    "month": dt.month if dt else None,
                    "day": dt.day if dt else None
                }

            # Base relations (with article_count for CONCERNS)
            article_count = len(articles)
            self.event_risk_relations.append((event_id, risk_key, article_count))
            if location_key:
                self.event_location_relations.append((event_id, location_key))
            if date_key:
                self.event_time_relations.append((event_id, date_key))

    def _create_temporal_relations(self):

        # Group events by location+risk (for IS_RECURRENT)
        events_by_lieu_risk = defaultdict(list)
        # Group by location+date (for IS_SYNCHRONOUS)
        events_by_lieu_date = defaultdict(list)
        # Group by location+year (for PRECEDES)
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

        # === IS_RECURRENT (same location, same risk, different dates) ===
        seen_recurrent = set()
        for (lieu, risk), events in events_by_lieu_risk.items():
            if len(events) > 1:
                events_sorted = sorted(events, key=lambda x: x[1])
                for i in range(len(events_sorted) - 1):
                    e1_id, e1_dt = events_sorted[i]
                    e2_id, e2_dt = events_sorted[i + 1]

                    key = tuple(sorted([e1_id, e2_id]))
                    if key not in seen_recurrent:
                        duration_days = (e2_dt - e1_dt).days
                        self.recurrent_relations.append((e1_id, e2_id, duration_days))
                        seen_recurrent.add(key)

        # === IS_SYNCHRONOUS (same location, same date, different risks) ===
        # V5: Non-directional relation - stored once (e1, e2)
        # but created in both directions in Cypher
        seen_synchronous = set()
        for (lieu, date_str), events in events_by_lieu_date.items():
            if len(events) > 1:
                for i in range(len(events)):
                    for j in range(i + 1, len(events)):
                        e1_id, e1_dt, e1_risk = events[i]
                        e2_id, e2_dt, e2_risk = events[j]

                        # Only if different risks (otherwise same event)
                        if e1_risk != e2_risk:
                            key = tuple(sorted([e1_id, e2_id]))
                            if key not in seen_synchronous:
                                self.synchronous_relations.append((e1_id, e2_id))
                                seen_synchronous.add(key)

        # === PRECEDES (same location, same year, different risks, different dates) ===
        seen_precedes = set()
        for (lieu, year), events in events_by_lieu_year.items():
            if len(events) > 1:
                events_sorted = sorted(events, key=lambda x: x[1])
                for i in range(len(events_sorted)):
                    for j in range(i + 1, len(events_sorted)):
                        e1_id, e1_dt, e1_risk = events_sorted[i]
                        e2_id, e2_dt, e2_risk = events_sorted[j]

                        # Only if different risks AND different dates
                        # (same date -> IS_SYNCHRONOUS, not PRECEDES)
                        if e1_risk != e2_risk and e1_dt != e2_dt:
                            key = (e1_id, e2_id)
                            if key not in seen_precedes:
                                duration_days = (e2_dt - e1_dt).days
                                self.precedes_relations.append((e1_id, e2_id, duration_days))
                                seen_precedes.add(key)

    def _create_spatial_relations(self):

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

        # Hierarchical relations
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


    def _print_statistics(self):

        print("\n" + "=" * 60)
        print("GRAPH STATISTICS")
        print("=" * 60)
        print(f"Event nodes: {len(self.events)}")
        print(f"Risk nodes: {len(self.risks)}")
        print(f"Location nodes: {len(self.locations)}")
        print(f"Time nodes: {len(self.times)}")

        total_mentions = sum(e['frequency_total'] for e in self.events.values())
        avg_mentions = total_mentions / len(self.events) if self.events else 0
        max_mentions = max((e['frequency_total'] for e in self.events.values()), default=0)
        multi_article = sum(1 for e in self.events.values() if e['article_count'] > 1)

        print(f"\n--- Frequency statistics ---")
        print(f"Total mentions (before aggregation): {total_mentions}")
        print(f"Average mentions/event: {avg_mentions:.2f}")
        print(f"Max mentions for a single event: {max_mentions}")
        print(f"Events mentioned in multiple articles: {multi_article}")

        print(f"\n--- Relations ---")
        print(f"Event-Risk (CONCERNS with article_count): {len(self.event_risk_relations)}")
        print(f"Event-Location (LOCATED_IN): {len(self.event_location_relations)}")
        print(f"Event-Time (OCCURRED_ON): {len(self.event_time_relations)}")
        print(f"IS_RECURRENT (with duration_days): {len(self.recurrent_relations)}")
        print(f"IS_SYNCHRONOUS (bidirectional): {len(self.synchronous_relations)} pairs = {len(self.synchronous_relations) * 2} relations")
        print(f"PRECEDES (with duration_days): {len(self.precedes_relations)}")

        if self.recurrent_relations:
            durations = [d for _, _, d in self.recurrent_relations]
            print(f"  - Min recurrence duration: {min(durations)} days")
            print(f"  - Max recurrence duration: {max(durations)} days")
            print(f"  - Mean recurrence duration: {sum(durations)/len(durations):.1f} days")

        hierarchy_counts = {}
        for _, _, rel_type in self.location_hierarchy_relations:
            hierarchy_counts[rel_type] = hierarchy_counts.get(rel_type, 0) + 1
        print(f"IS_FROM_DEPARTEMENT: {hierarchy_counts.get('IS_FROM_DEPARTEMENT', 0)}")
        print(f"IS_FROM_PROVINCE: {hierarchy_counts.get('IS_FROM_PROVINCE', 0)}")
        print(f"IS_FROM_REGION: {hierarchy_counts.get('IS_FROM_REGION', 0)}")



    def generate_cypher(self) -> str:
        """Generate the complete Cypher code."""
        lines = []


        lines.append("// ===== CONSTRAINTS AND INDEXES =====")
        lines.append("CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE;")
        lines.append("CREATE CONSTRAINT risk_name IF NOT EXISTS FOR (r:Risk) REQUIRE r.name IS UNIQUE;")
        lines.append("CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE;")
        lines.append("CREATE CONSTRAINT time_datetime IF NOT EXISTS FOR (t:Time) REQUIRE t.datetime IS UNIQUE;")
        lines.append("CREATE INDEX event_term IF NOT EXISTS FOR (e:Event) ON (e.term);")
        lines.append("CREATE INDEX event_title IF NOT EXISTS FOR (e:Event) ON (e.title);")
        lines.append("CREATE INDEX location_type IF NOT EXISTS FOR (l:Location) ON (l.type);")
        lines.append("CREATE INDEX time_year IF NOT EXISTS FOR (t:Time) ON (t.year);")
        lines.append("")


        lines.append("// ===== RISK NODES =====")
        for risk_name, risk_data in self.risks.items():
            lines.append(f"MERGE (r:Risk {{name: '{risk_data['name']}'}})")
            lines.append(f"SET r.theme = '{risk_data['theme']}', r.concept = '{risk_data['concept']}', r.phase = '{risk_data['phase']}';")
        lines.append("")

        lines.append("// ===== LOCATION NODES =====")
        for loc_name, loc_data in self.locations.items():
            lat_str = f", l.latitude = {loc_data['latitude']}" if loc_data['latitude'] else ""
            lon_str = f", l.longitude = {loc_data['longitude']}" if loc_data['longitude'] else ""

            lines.append(f"MERGE (l:Location {{name: '{loc_data['name']}'}})")
            lines.append(f"SET l.type = '{loc_data['type']}', l.wikidata_id = '{loc_data['wikidata_id']}'{lat_str}{lon_str};")
        lines.append("")

        lines.append("// ===== TIME NODES =====")
        for date_str, time_data in self.times.items():
            year_str = f", t.year = {time_data['year']}" if time_data['year'] else ""
            month_str = f", t.month = {time_data['month']}" if time_data['month'] else ""
            day_str = f", t.day = {time_data['day']}" if time_data['day'] else ""

            lines.append(f"MERGE (t:Time {{datetime: '{time_data['datetime']}'}})")
            lines.append(f"SET t.datetime = '{time_data['datetime']}'{year_str}{month_str}{day_str};")
        lines.append("")

        lines.append("// ===== EVENT NODES =====")
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

        lines.append("// ===== EVENT -> RISK RELATIONS (CONCERNS) =====")
        for event_id, risk_name, article_count in self.event_risk_relations:
            lines.append(f"MATCH (e:Event {{id: '{event_id}'}}), (r:Risk {{name: '{risk_name}'}})")
            lines.append(f"MERGE (e)-[:CONCERNS {{article_count: {article_count}}}]->(r);")
        lines.append("")

        lines.append("// ===== EVENT -> LOCATION RELATIONS (LOCATED_IN) =====")
        for event_id, loc_name in self.event_location_relations:
            lines.append(f"MATCH (e:Event {{id: '{event_id}'}}), (l:Location {{name: '{loc_name}'}})")
            lines.append("MERGE (e)-[:LOCATED_IN]->(l);")
        lines.append("")

        lines.append("// ===== EVENT -> TIME RELATIONS (OCCURRED_ON) =====")
        for event_id, date_str in self.event_time_relations:
            lines.append(f"MATCH (e:Event {{id: '{event_id}'}}), (t:Time {{datetime: '{date_str}'}})")
            lines.append("MERGE (e)-[:OCCURRED_ON]->(t);")
        lines.append("")


        lines.append("// ===== IS_RECURRENT RELATIONS (same risk, same location, different dates) =====")
        lines.append("// duration_days: number of days between the two occurrences")
        for e1_id, e2_id, duration_days in self.recurrent_relations:
            lines.append(f"MATCH (e1:Event {{id: '{e1_id}'}}), (e2:Event {{id: '{e2_id}'}})")
            lines.append(f"MERGE (e1)-[:IS_RECURRENT {{duration_days: {duration_days}}}]->(e2);")
        lines.append("")

        lines.append("// ===== IS_SYNCHRONOUS RELATIONS (same location, same date, different risks) =====")
        lines.append("// V5: BIDIRECTIONAL relation - created in both directions")
        lines.append("// Two events occurring on the same day at the same location without causal relation")
        for e1_id, e2_id in self.synchronous_relations:
            lines.append(f"MATCH (e1:Event {{id: '{e1_id}'}}), (e2:Event {{id: '{e2_id}'}})")
            lines.append("MERGE (e1)-[:IS_SYNCHRONOUS]->(e2)")
            lines.append("MERGE (e2)-[:IS_SYNCHRONOUS]->(e1);")
        lines.append("")

        lines.append("// ===== PRECEDES RELATIONS (same location, different risks, different dates) =====")
        lines.append("// duration_days: number of days between the two events")
        for e1_id, e2_id, duration_days in self.precedes_relations:
            lines.append(f"MATCH (e1:Event {{id: '{e1_id}'}}), (e2:Event {{id: '{e2_id}'}})")
            lines.append(f"MERGE (e1)-[:PRECEDES {{duration_days: {duration_days}}}]->(e2);")
        lines.append("")


        lines.append("// ===== SPATIAL HIERARCHY RELATIONS =====")
        for loc1, loc2, rel_type in self.location_hierarchy_relations:
            lines.append(f"MATCH (l1:Location {{name: '{loc1}'}}), (l2:Location {{name: '{loc2}'}})")
            lines.append(f"MERGE (l1)-[:{rel_type}]->(l2);")
        lines.append("")

        return "\n".join(lines)



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

    print(f"\nInput file: {input_file}")
    print(f"Output file: {output_file}")

    print("\nLoading data...")
    df = pd.read_csv(input_file)
    print(f"  Total records: {len(df)}")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"  WARNING: Missing columns: {missing_cols}")

    print("\nPreprocessing (normalizing locations and regions)...")
    df = preprocess_dataframe(df)

    print("\nFiltering data...")
    df_filtered = df[(df['wikidata_id'].notna()) & (df['label'] == 1)]
    print(f"  After filtering (wikidata_id not null AND label=1): {len(df_filtered)}")

    if len(df_filtered) == 0:
        print("ERROR: No records after filtering!")
        sys.exit(1)

    print("\nGenerating graph...")
    generator = Neo4jGraphGeneratorV5(df_filtered)
    generator.process()

    print("\nGenerating Cypher code...")
    cypher_code = generator.generate_cypher()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cypher_code)
    print(f"\nCypher file saved: {output_file}")

    # Events CSV
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
    print(f"Events CSV: {events_csv}")

    return generator


if __name__ == "__main__":
    generator = main()
