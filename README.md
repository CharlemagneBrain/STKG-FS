# STKGFS: Spatio-Temporal Knowledge Graph from Unstructured Texts for Food Security Monitoring

A framework for constructing spatio-temporal knowledge graphs from French news articles to monitor food security in Burkina Faso.

> **Paper**: *Spatio-Temporal Knowledge Graph from Unstructured Texts: A Multi-Scale Approach for Food Security Monitoring*
> Submitted to **AGILE: GIScience Series, 2026**

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Reproducibility Guide](#reproducibility-guide)
- [Neo4j Queries and Visualizations](#neo4j-queries-and-visualizations)
- [Results and Interpretation](#results-and-interpretation)
- [Pre-trained Models](#pre-trained-models)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Food security monitoring in West Africa requires timely, fine-grained, and interpretable information from diverse sources. This project implements an integrated NLP and knowledge graph pipeline that extracts, structures, and analyzes food security-related information from **15,000 French press articles** published in 2009 from Burkina Faso.

The approach combines three specialized components:

1. **Spatial entity recognition** -- Fine-tuned CamemBERT for village-level geographic extraction across the full Burkina Faso administrative hierarchy
2. **Temporal normalization** -- HeidelTime-based extraction and normalization of temporal expressions (TIMEX3 standard)
3. **Knowledge graph construction** -- Projection-based triplet formation and Neo4j graph with spatial, contextual, and temporal inference relationships

### Key Features

- **Fine-grained spatial entity extraction**: Village-level geographic recognition using fine-tuned CamemBERT, achieving F1=0.96--0.99 across all administrative levels
- **Temporal expression normalization**: French temporal extraction with HeidelTime (TIMEX3 standard)
- **Triplet formation**: Projection-based algorithm creating *(Risk Indicator, Location, Date)* triplets with 81.91% accuracy
- **Knowledge graph construction**: Neo4j-based graph with spatial hierarchies and temporal inference relations (IS_RECURRENT, IS_SYNCHRONOUS, PRECEDES)

### Pipeline Architecture

```
1. Data Collection          2. Data Processing              3. KG Exploitation
┌─────────────────┐    ┌──────────────────────────┐    ┌─────────────────────┐
│ Press Articles   │    │ Spatial & Temporal Entity │    │ Evaluation          │
│ (15,000 articles)│───>│ Recognition (CamemBERT + │───>│ Indicators          │
│                  │    │ HeidelTime)              │    │                     │
│ Food Security    │    │          │               │    │ Spatial             │
│ Lexicon (433     │───>│ Risk Indicator Detection │    │ Visualization       │
│ terms)           │    │ (Lexicon-based)          │    │ of Risk Indicators  │
│                  │    │          │               │    └─────────────────────┘
│ Open Data        │    │ Relationship Rules       │
│ (Wikidata)       │───>│ Definition & Generation  │
│                  │    │          │               │
│ Administrative   │    │ Knowledge Graph          │
│ Hierarchy        │───>│ Construction & Update    │
└─────────────────┘    └──────────────────────────┘
```

### Performance Highlights

**Spatial Entity Recognition (Table 1 in paper)**

| Entity Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Country     | 0.98      | 1.00   | 0.99     |
| Region      | 0.99      | 0.99   | 0.99     |
| Province    | 0.99      | 0.98   | 0.99     |
| Department  | 0.99      | 0.99   | 0.99     |
| Village     | 0.94      | 0.98   | 0.96     |

---

## Installation

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python      | 3.10+   | Core runtime |
| Neo4j       | 4.4+    | Knowledge graph storage and querying |
| Java        | 8+      | Required by HeidelTime |
| CUDA GPU    | Optional| Recommended for model training (~2-3h on RTX 3090) |

### Step-by-step Setup

```bash
# 1. Clone the repository
git clone https://github.com/[username]/stkgfs.git
cd stkgfs

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: venv\Scripts\activate     # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download spaCy French model
python -m spacy download fr_core_news_sm

# 5. Verify Java installation (required for HeidelTime)
java -version
```

### HeidelTime Configuration

HeidelTime requires a TreeTagger installation. Update the path in `Temporal_Entities_Detection/config.props`:

```properties
treeTaggerHome = /path/to/your/TreeTaggerLinux
```

The `py-heideltime` package handles the integration automatically. Install it via:

```bash
pip install py-heideltime
```

### Neo4j Setup

1. Download and install [Neo4j Desktop](https://neo4j.com/download/) or use [Neo4j Aura](https://neo4j.com/cloud/aura/)
2. Create a new database
3. Import the generated Cypher file (see [Step 6](#step-6-import-to-neo4j))

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `cykhash` / `pyrobuf` build failure | These are optional dependencies. Core functionality is unaffected. |
| HeidelTime `FileNotFoundError` | Update `treeTaggerHome` path in `config.props` |
| CUDA out of memory | Reduce `per_device_train_batch_size` to 16 or 8 in training arguments |
| spaCy model not found | Run `python -m spacy download fr_core_news_sm` |
| Neo4j connection error | Ensure Neo4j is running and credentials are correct |

---

## Project Structure

```
stkgfs/
├── Fine-Tuning/                          # Step 1: CamemBERT fine-tuning for NER
│   ├── CamemBERT_FT.ipynb               # Training and evaluation notebook
│   └── annotations/                     # BIO-tagged training data
│       ├── train_extended_bio_feb.json   # 59,900 training sentences
│       ├── val_extended_bio_feb.json     # 14,758 validation sentences
│       └── test_extended_bio_feb.json    # 11,594 test sentences
│
├── Spatial_Annotation_Detection/         # Step 2: Spatial entity detection
│   ├── Spatial_Pipeline.ipynb           # CamemBERT vs GLiNER evaluation
│   └── data/
│       └── df_sample.csv               # 1,000 annotated articles for evaluation
│
├── Temporal_Entities_Detection/          # Step 3: Temporal extraction
│   ├── Heideltime_Detection.ipynb       # HeidelTime processing notebook
│   ├── config.props                     # HeidelTime/TreeTagger configuration
│   └── data/
│       ├── df_extended_intersection.csv # Segments with spatial + temporal annotations
│       └── new_heaideltime_today.csv    # HeidelTime extraction results
│
├── Triplet_Formation/                    # Step 4: Triplet extraction
│   ├── Triplet_Algo.ipynb               # Projection-based triplet algorithm
│   └── data/
│       ├── lexique_enrichi_final_july.csv  # 433-term food security lexicon
│       ├── annotated_data.csv              # 1,133 expert-annotated triplets
│       └── reconstruct_df_new.csv          # Reconstructed articles
│
├── Events_to_Graph/                      # Step 5: Knowledge graph construction
│   ├── Preprocessing_for_Graph.ipynb    # Data analysis and visualization
│   ├── enrich_geo_wikidata_complete.py  # Wikidata geocoding enrichment
│   ├── generate_neo4j_graph.py          # Neo4j Cypher query generation
│   └── data/
│       ├── df_preprocessed.csv          # Preprocessed triplets with labels
│       ├── processed_wd.csv             # Wikidata-enriched locations
│       ├── annotations_df.csv           # Consolidated annotations
│       ├── stkg_dtb.cypher              # Generated Neo4j import file
│       └── stkg_dtb_events.csv          # Event summary CSV
│
├── Preprocessing/                        # Text preprocessing utilities
│   └── Preprocess_News_Papers.py        # Article segmentation with overlap
│
├── results_csv/                          # Intermediate evaluation results
│   ├── entites_df.csv
│   ├── filtered_df_nelson.csv
│   ├── segmented_csvfile.csv
│   └── spatial_entities_with_pos.csv
│
├── requirements.txt                      # Python dependencies
└── README.md                            # This file
```

---

## Data Description

### Source Corpus

- **15,000 unannotated French news articles** published in 2009
- Collected from major West African news outlets (LeFaso.net, Burkina24)
- Articles segmented into overlapping windows (configurable length and overlap)

### Training Data for NER

The CamemBERT NER model is trained on the complete administrative hierarchy of Burkina Faso, derived from the [2022 Statistical Yearbook of Territorial Administration](http://cns.bf/IMG/pdf/matds_annuaire_at_2022.pdf):

| Administrative Level | Count | Examples |
|---------------------|-------|----------|
| Regions             | 13    | Centre, Hauts-Bassins, Sahel |
| Provinces           | 45    | Kadiogo, Houet, Soum |
| Departments         | 351   | Ouagadougou, Bobo-Dioulasso, Koudougou |
| Villages            | 7,936 | Pabre, Koubri, Sya |

Training data was generated via **distant supervision**: sentences containing place names from the official gazetteer are annotated with their corresponding administrative type using BIO tags.

| Split      | Sentences | Purpose |
|------------|-----------|---------|
| Train      | 59,900    | Model training |
| Validation | 14,758    | Hyperparameter tuning |
| Test       | 11,594    | Final evaluation (contains ~20% unseen entities) |

### Food Security Lexicon

Domain-specific lexicon defined and validated by experts:

- **433 unique terms** organized into **92 concepts** and **8 thematic categories**
- Categories: Agriculture, Environment, Economic, Sociopolitical, Dietary, Health, General Crisis, Human Health
- Includes crisis phase annotations (pre-crisis, crisis, post-crisis) where applicable
- **8 vague terms** requiring contextual filtering: *production*, *campagne*, *prix*, *cout*, *crise*, *stock*, *pluie*, *pluviometrie*

### Manual Annotations and Corrections

> **Important**: Results in this repository may differ slightly from values reported in the associated paper. The conclusions and interpretations remain unchanged. Values will be adjusted in the final paper.

The following manual interventions were applied by domain experts:

1. **Spatial annotation corrections**:
   - **Homonymous villages**: Village names appearing in multiple provinces were disambiguated based on article context (e.g., a village named "Bissa" exists in several provinces)
   - **Orthographic variations**: West African toponyms exhibit significant spelling variability (e.g., "Ouagadougou" vs. "Ouaga")
   - **False positives**: Place names coincidentally matching common French words were removed

2. **Triplet validation** (annotation campaign on 1,133 triplets):
   - **Term relevance**: Each detected risk indicator was assessed for food-security relevance (46.9% deemed relevant, 53.1% non-relevant)
   - **Spatial association**: Correct linkage of risk term to actual location (84.29% accuracy before correction)
   - **Temporal association**: Correct linkage to event date (96.56% accuracy before correction)

3. **Post-processing refinements**:
   - Multi-location event associations: 21 additional triplets generated when annotators identified events affecting multiple locations
   - Region name normalization: Internal Wikidata names mapped to official names (e.g., "Guiriko" -> "Hauts-Bassins", "Kadiogo" -> "Centre")

---

## Reproducibility Guide

Follow these steps sequentially to reproduce the complete pipeline. Each step produces outputs consumed by subsequent steps.

### Step 1: Preprocess News Articles

```bash
cd Preprocessing
```

The `Preprocess_News_Papers.py` module segments raw articles into overlapping windows:

```python
from Preprocess_News_Papers import Preprocess_News_Papers

preprocessor = Preprocess_News_Papers("path/to/raw_articles.csv")
segmented_df = preprocessor.apply_splitting(
    max_length=200,    # Maximum words per segment
    min_words=10,      # Minimum words threshold
    overlap=0.2        # 20% overlap between segments
)
```

### Step 2: Fine-tune CamemBERT for Spatial NER

```bash
cd Fine-Tuning
jupyter notebook CamemBERT_FT.ipynb
```

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Base model | `camembert-base` |
| Learning rate | 5e-5 |
| Batch size | 32 |
| Epochs | 70 |
| Weight decay | 0.01 |
| Frozen layers | Embedding layers only |
| Trainable parameters | 85,062,923 / 110,039,819 (77.3%) |

**Expected training time**: ~2-3 hours on NVIDIA RTX 3090

**What the notebook does**:
1. Loads BIO-tagged training data from `annotations/`
2. Tokenizes with CamemBERT tokenizer and aligns labels
3. Freezes embedding layers to preserve pre-trained French representations
4. Trains for 70 epochs with evaluation every 100 steps
5. Evaluates on held-out test set with `seqeval` classification report

### Step 3: Evaluate Spatial Entity Detection

```bash
cd Spatial_Annotation_Detection
jupyter notebook Spatial_Pipeline.ipynb
```

**What the notebook does**:
1. Loads 1,000 manually annotated articles (`data/df_sample.csv`)
2. Compares fine-tuned CamemBERT vs. GLiNER (zero-shot)
3. Filters predictions against validated entity sets
4. Computes character-level BIO metrics with `seqeval`

### Step 4: Extract Temporal Entities

```bash
cd Temporal_Entities_Detection
jupyter notebook Heideltime_Detection.ipynb
```

**Prerequisites**: Java 8+ and correct `config.props` paths.

**What the notebook does**:
1. Extracts publication dates from article headers using regex
2. Runs HeidelTime with French language rules and `news` document type
3. Extracts DATE-type TIMEX3 expressions with character spans
4. Saves results to `data/new_heaideltime_today.csv`

### Step 5: Form Triplets

```bash
cd Triplet_Formation
jupyter notebook Triplet_Algo.ipynb
```

**What the notebook does**:
1. Reconstructs full articles from segments (with span offset correction)
2. Splits into sentences using `wtpsplit` (SaT-12l, French)
3. Detects food security terms via lexicon matching with morphological variants
4. Applies **vague term filtering**: retains ambiguous terms only when co-occurring with non-vague terms or variation triggers (augmentation/diminution)
5. **Spatial projection**: assigns the most specific location to each term using hierarchical preference (village > department > province > region > country)
6. **Temporal projection**: assigns closest valid date not exceeding publication date
7. Enriches each triplet with context window (1 sentence before + after)

### Step 6: Build Knowledge Graph

```bash
cd Events_to_Graph

# 6a. Analyze and visualize preprocessed data
jupyter notebook Preprocessing_for_Graph.ipynb

# 6b. Enrich locations with Wikidata coordinates and hierarchy
python enrich_geo_wikidata_complete.py data/df_preprocessed.csv data/processed_wd.csv

# 6c. Generate Neo4j Cypher import file
python generate_neo4j_graph.py data/processed_wd.csv data/stkg_dtb.cypher
```

**Wikidata enrichment** (`enrich_geo_wikidata_complete.py`):
- Queries Wikidata SPARQL endpoint for each unique location
- Retrieves: Wikidata Q-ID, coordinates (lat/lon), administrative type
- Builds administrative hierarchy via `P131` (located in administrative entity)
- Finds neighboring entities within 10km radius
- Handles name mappings (e.g., "Hauts-Bassins" <-> "Guiriko" in Wikidata)

**Graph generation** (`generate_neo4j_graph.py`):
- Filters triplets: keeps only validated entries (`label=1`) with Wikidata IDs
- Aggregates triplets sharing same (term, location, date) into unique Event nodes
- Creates 4 node types and 3 relationship categories (see [Results](#results-and-interpretation))
- Normalizes region names from Wikidata internal names to official names

**Expected runtime**: Wikidata enrichment takes ~10-20 minutes (depends on API rate limits).

### Step 7: Import to Neo4j

```cypher
// In Neo4j Browser or cypher-shell:
:source /path/to/stkg_dtb.cypher
```

Or via command line:
```bash
cat data/stkg_dtb.cypher | cypher-shell -u neo4j -p <password>
```

---

## Neo4j Queries and Visualizations

### Graph Schema

The knowledge graph contains **4 node types** and **3 relationship categories**:

**Node types:**
| Node | Properties | Description |
|------|-----------|-------------|
| `Event` | id, title, term, lieu, date, frequency_total, contexts | Food security situation (risk + location + date) |
| `Risk` | name, theme, concept, phase | Thematic risk indicator |
| `Location` | name, type, wikidata_id, latitude, longitude | Geographic entity with administrative level |
| `Time` | datetime, year, month, day | Normalized temporal entity |

**Relationship categories:**

1. **Contextual relationships** (Event <-> entities):
   - `CONCERNS`: Event -> Risk (with `article_count`)
   - `LOCATED_IN`: Event -> Location
   - `OCCURRED_ON`: Event -> Time

2. **Spatial relationships** (Location hierarchy):
   - `IS_FROM_DEPARTEMENT`: Village -> Department
   - `IS_FROM_PROVINCE`: Department -> Province
   - `IS_FROM_REGION`: Province -> Region
   - `IS_NEAR_TO`: Location <-> Location (with `distance_km`)

3. **Temporal inference relationships** (Event <-> Event):
   - `IS_RECURRENT`: Same risk + same location, different dates (with `duration_days`)
   - `IS_SYNCHRONOUS`: Same location + same date, different risks (bidirectional)
   - `PRECEDES`: Same location + same year, different risks, sequential dates (with `duration_days`)

### Queries for Paper Figures

**Query 1: Temporal monitoring of events in Ouagadougou (Figure 4)**
```cypher
MATCH (e:Event)-[:LOCATED_IN]->(l:Location {name: 'Ouagadougou'})
MATCH (e)-[:CONCERNS]->(r:Risk)
MATCH (e)-[:OCCURRED_ON]->(t:Time)
WHERE t.year = 2009
RETURN e, r, t, l
```

**Query 2: Vulnerable areas by crisis type (Figure 5)**
```cypher
// Environmental crises (floods and fires)
MATCH (e:Event)-[:CONCERNS]->(r:Risk)
MATCH (e)-[:LOCATED_IN]->(l:Location)
WHERE r.theme = 'environment'
  AND l.latitude IS NOT NULL
RETURN l.name, l.type, l.latitude, l.longitude, r.name, count(e) AS event_count
ORDER BY event_count DESC
```

**Query 3: Recurring events at same location (IS_RECURRENT)**
```cypher
MATCH (e1:Event)-[rec:IS_RECURRENT]->(e2:Event)
WHERE rec.duration_days < 180
MATCH (e1)-[:CONCERNS]->(r:Risk)
MATCH (e1)-[:LOCATED_IN]->(l:Location)
RETURN e1.title, e2.title, rec.duration_days, r.name, l.name
ORDER BY rec.duration_days
LIMIT 20
```

**Query 4: Synchronous events -- compound crises**
```cypher
MATCH (e1:Event)-[:IS_SYNCHRONOUS]->(e2:Event)
MATCH (e1)-[:CONCERNS]->(r1:Risk)
MATCH (e2)-[:CONCERNS]->(r2:Risk)
MATCH (e1)-[:LOCATED_IN]->(l:Location)
WHERE r1.theme <> r2.theme
RETURN l.name, r1.name, r2.name, e1.date
```

**Query 5: Regional event distribution (Table 3)**
```cypher
MATCH (e:Event)-[:LOCATED_IN]->(l:Location)
MATCH (e)-[:CONCERNS]->(r:Risk)
OPTIONAL MATCH (l)-[:IS_FROM_DEPARTEMENT|IS_FROM_PROVINCE|IS_FROM_REGION*1..3]->(region:Location)
WHERE region.type = 'region'
WITH coalesce(region.name, l.name) AS region_name, r.theme AS theme, count(DISTINCT e) AS event_count
RETURN region_name, theme, event_count
ORDER BY event_count DESC
```

**Query 6: Cascading events -- PRECEDES chains**
```cypher
MATCH path = (e1:Event)-[:PRECEDES*1..3]->(e2:Event)
WHERE e1 <> e2
MATCH (e1)-[:LOCATED_IN]->(l:Location)
MATCH (e1)-[:CONCERNS]->(r1:Risk)
MATCH (e2)-[:CONCERNS]->(r2:Risk)
RETURN e1.title, r1.name, e2.title, r2.name,
       length(path) AS chain_length, l.name
ORDER BY chain_length DESC
LIMIT 20
```

**Query 7: Spatial hierarchy traversal**
```cypher
// From village to region: Sya -> Bobo-Dioulasso -> Houet -> Hauts-Bassins
MATCH path = (v:Location)-[:IS_FROM_DEPARTEMENT|IS_FROM_PROVINCE|IS_FROM_REGION*1..3]->(r:Location)
WHERE v.name = 'Sya'
RETURN path
```

**Query 8: Multi-scale event aggregation**
```cypher
// Count events at each administrative level for a region
MATCH (e:Event)-[:LOCATED_IN]->(l:Location)
MATCH path = (l)-[:IS_FROM_DEPARTEMENT|IS_FROM_PROVINCE|IS_FROM_REGION*0..3]->(region:Location {name: 'Hauts-Bassins'})
RETURN l.name, l.type, count(e) AS events
ORDER BY events DESC
```

---

## Results and Interpretation

### Spatial Entity Recognition (Table 1)

Fine-tuned CamemBERT evaluated on a held-out test set containing ~20% unseen entities at each hierarchical level:

| Entity Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Country     | 0.98      | 1.00   | 0.99     | 4,648   |
| Region      | 0.99      | 0.99   | 0.99     | 6,744   |
| Province    | 0.99      | 0.98   | 0.99     | 541     |
| Department  | 0.99      | 0.99   | 0.99     | 1,433   |
| Village     | 0.94      | 0.98   | 0.96     | 3,236   |
| **Micro avg** | **0.98** | **0.98** | **0.98** | **16,602** |

**Comparison with baselines** (on 1,000 test articles):
- Baseline CamemBERT (no fine-tuning): P=0.87, R=0.58, **F1=0.70**
- GLiNER (zero-shot): P=0.82, R=0.77, **F1=0.79**
- Fine-tuned CamemBERT: P=0.98, R=0.98, **F1=0.98**

### Triplet Formation Evaluation (Table 2)

Expert annotation campaign on 1,133 extracted triplets:

| Category | Count | Percentage |
|----------|-------|------------|
| Location and Date correct | 928 | 81.91% |
| Location corrected only | 166 | 14.65% |
| Date corrected only | 27 | 2.38% |
| Both corrected | 12 | 1.06% |
| **Total** | **1,133** | **100%** |

- **Spatial projection accuracy**: 84.29% (955/1,133 correct associations)
- **Temporal projection accuracy**: 96.56% (1,094/1,133 correct associations)

**Main spatial error patterns:**
1. Over-aggregation to capital/country level when specific locations appear outside the 2-sentence projection window
2. Ambiguous location references when multiple locations of similar granularity are mentioned

### Knowledge Graph Statistics

| Metric | Value |
|--------|-------|
| Unique events | 376 |
| Risk types | 77 |
| Distinct locations | 71 (415 including hierarchy and neighbors) |
| Temporal entities | 147 |
| Source articles | 200 |
| IS_RECURRENT relationships | 141 |
| IS_SYNCHRONOUS relationship pairs | 263 (526 directed edges) |
| PRECEDES relationships | 9,726 |
| Spatial hierarchy relations | 91 |
| IS_NEAR_TO relations | 354 |

**Corpus distribution by theme:**
- Sociopolitical: 29.9%
- Environmental: 23.8%
- Agricultural: 16.3%
- General crisis: 11.4%
- Economic: 10.9%

**Top risk indicators**: inondation (flood, 28.1%), catastrophe (disaster, 13.1%), insecurite (insecurity, 10.9%), corruption (10.9%), incendie (fire, 10.9%)

### Metrics Explained

- **Precision**: Proportion of correctly identified entities among all detected entities. High precision = few false alarms.
- **Recall**: Proportion of actual entities successfully detected. High recall = few missed entities.
- **F1-Score**: Harmonic mean of precision and recall, providing a single balanced metric (0-1).
- **BIO tagging**: Begin-Inside-Outside scheme for sequence labeling. B-tag marks entity start, I-tag marks continuation, O marks non-entity tokens.

### Notes on Divergences Between Code and Paper

Some numerical values produced by running the code may differ slightly from those reported in the paper. This is due to:

1. **Manual corrections applied post-extraction**: Expert annotations corrected spatial and temporal associations after initial algorithmic extraction
2. **Relevance filtering**: The 53.1% non-relevant triplets were filtered during the annotation campaign, a step not fully automated in the pipeline
3. **Region name normalization**: Wikidata returns internal region names (e.g., "Guiriko") that are mapped to official names (e.g., "Hauts-Bassins") in post-processing

These differences do not affect the conclusions or interpretations presented in the paper.

---

## Pre-trained Models

### HuggingFace Model

The fine-tuned CamemBERT model for Burkina Faso spatial NER will be available at:

**[HuggingFace Model Hub -- Link to be added upon publication]**

#### Model Card Summary

| Property | Value |
|----------|-------|
| **Model name** | CamemBERT-BurkinaFaso-NER |
| **Base model** | `camembert-base` |
| **Task** | Token classification (NER) |
| **Language** | French |
| **Entity types** | country, region, province, departement, village |
| **Training data** | 59,900 sentences (BIO-tagged, distant supervision) |
| **F1-Score** | 0.96--0.99 per entity type |
| **License** | MIT |

#### Usage Example

```python
from transformers import CamembertTokenizerFast, CamembertForTokenClassification
import torch

# Load model and tokenizer
tokenizer = CamembertTokenizerFast.from_pretrained("username/CamemBERT-BurkinaFaso-NER")
model = CamembertForTokenClassification.from_pretrained("username/CamemBERT-BurkinaFaso-NER")

# Entity labels
label_list = [
    "O",
    "B-country", "I-country",
    "B-region", "I-region",
    "B-departement", "I-departement",
    "B-province", "I-province",
    "B-village", "I-village"
]
id2label = {i: label for i, label in enumerate(label_list)}

# Inference
text = "Les inondations ont touche Ouagadougou et les villages de Pabre et Koubri."
inputs = tokenizer(text, return_tensors="pt", truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=2)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

for token, pred_id in zip(tokens, predictions[0]):
    label = id2label[pred_id.item()]
    if label != "O":
        print(f"{token}: {label}")
```

**Expected output:**
```
Ouagadougou: B-departement
Pabre: B-village
Koubri: B-village
```

#### Publishing to HuggingFace

```python
from huggingface_hub import HfApi, create_repo

# Create repository
create_repo("username/CamemBERT-BurkinaFaso-NER", private=False)

# Upload model files
api = HfApi()
api.upload_folder(
    folder_path="path/to/model/checkpoint",
    repo_id="username/CamemBERT-BurkinaFaso-NER",
    repo_type="model"
)
```

---

## Limitations

1. **Lexicon-based detection**: May produce false positives when food security terminology appears in non-relevant contexts (policy discussions, historical references, prevention campaigns). The 53.1% non-relevance rate reflects this challenge.
2. **Media coverage bias**: Ouagadougou (402 occurrences) and major cities receive disproportionate coverage compared to rural areas, reflecting the structure of the press corpus rather than actual risk distribution.
3. **Spatial projection window**: The 2-sentence context window for spatial projection may miss longer-range location references, leading to over-aggregation to capital or country level.
4. **Single country focus**: The model is trained specifically on the Burkina Faso administrative hierarchy and may not generalize to other West African countries without retraining.
5. **Temporal scope**: The current corpus covers only 2009 articles. Extending to multiple years would require additional data collection.

---

## Citation

If you use this code, data, or model in your research, please cite:

```bibtex
@inproceedings{author2026stkgfs,
  title={Spatio-Temporal Knowledge Graph from Unstructured Texts:
         A Multi-Scale Approach for Food Security Monitoring},
  author={[Authors]},
  booktitle={AGILE: GIScience Series},
  year={2026},
  doi={[DOI]}
}
```

---

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Administrative hierarchy data from the [2022 Statistical Yearbook of Territorial Administration](http://cns.bf/IMG/pdf/matds_annuaire_at_2022.pdf), Burkina Faso Ministry of Territorial Administration
- Geographic coordinates and entity linking via [Wikidata](https://www.wikidata.org/)
- French NER foundation from [CamemBERT](https://camembert-model.fr/) (Martin et al., 2020)
- Temporal extraction via [HeidelTime](https://github.com/HeidelTime/heideltime) (Strotgen et al., 2013)
- Sentence segmentation via [wtpsplit](https://github.com/segment-any-text/wtpsplit) (SaT model)

---

## Contact

For questions about this research, please contact: [email address]
