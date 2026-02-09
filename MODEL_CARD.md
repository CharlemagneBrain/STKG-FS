---
language: fr
license: mit
tags:
- token-classification
- ner
- french
- burkina-faso
- food-security
- camembert
datasets:
- CharlesAbdoulaye/BF_NER_datasets
metrics:
- precision
- recall
- f1
model-index:
- name: BF_NER
  results:
  - task:
      type: token-classification
      name: Named Entity Recognition
    dataset:
      name: Burkina Faso Administrative Hierarchy
      type: CharlesAbdoulaye/BF_NER_datasets
    metrics:
    - type: f1
      value: 0.98
      name: F1 (micro avg)
    - type: precision
      value: 0.98
      name: Precision (micro avg)
    - type: recall
      value: 0.98
      name: Recall (micro avg)
---

# BF_NER: Burkina Faso Named Entity Recognition

Fine-tuned CamemBERT model for extracting geographic entities from French text, specialized for the Burkina Faso administrative hierarchy.

## Model Description

This model is a fine-tuned version of [`camembert-base`](https://huggingface.co/camembert-base) for Named Entity Recognition (NER) of geographic locations in French news articles. It recognizes five administrative levels specific to Burkina Faso:

- **Country**: Burkina Faso, regional country references
- **Region**: 13 regions (e.g., Centre, Hauts-Bassins, Sahel)
- **Province**: 45 provinces (e.g., Kadiogo, Houet, Soum)
- **Department**: 351 departments (e.g., Ouagadougou, Bobo-Dioulasso, Koudougou)
- **Village**: 7,936 villages (e.g., Pabre, Koubri, Sya)

### Model Details

- **Developed by**: Charles Abdoulaye Ngom, Landy Rajaonarivo, Sarah Valentin, Maguelonne Teisseire
- **Model type**: Token Classification (NER)
- **Language**: French
- **Base model**: `camembert-base`
- **License**: MIT
- **Paper**: *Spatio-Temporal Knowledge Graph from Unstructured Texts: A Multi-Scale Approach for Food Security Monitoring* (AGILE 2026)
- **DOI (model)**: [10.57967/hf/7766](https://doi.org/10.57967/hf/7766)
- **DOI (datasets)**: [10.57967/hf/7767](https://doi.org/10.57967/hf/7767)

## Intended Use

### Primary Use Cases

- **Food security monitoring**: Extract location mentions from news articles to track food security events
- **Geographic information extraction**: Identify and classify locations in French West African texts
- **Multi-scale spatial analysis**: Enable analysis from village to country level
- **Crisis mapping**: Support humanitarian and development organizations in monitoring regional events

### Out-of-Scope Use

- This model is **NOT suitable** for:
  - Named entity recognition in other countries (limited to Burkina Faso administrative entities)
  - Non-French languages
  - Person, organization, or other non-location entity types
  - Real-time applications without additional validation

## Training Data

The model was trained using **distant supervision** on 15,000 French news articles from 2009:

| Split | Sentences | Description |
|-------|-----------|-------------|
| Train | 59,900 | Sentences containing administrative place names |
| Validation | 14,758 | Used for hyperparameter tuning |
| Test | 11,594 | Held-out set with ~20% unseen entities per level |

**Data Source**: Official gazetteer from the [2022 Statistical Yearbook of Territorial Administration](http://cns.bf/IMG/pdf/matds_annuaire_at_2022.pdf), Burkina Faso Ministry of Territorial Administration.

**Annotation Scheme**: BIO tagging (Begin-Inside-Outside)
- `B-{type}`: Beginning of an entity
- `I-{type}`: Inside/continuation of an entity
- `O`: Outside any entity

**Entity types**: `country`, `region`, `province`, `departement`, `village`

## Training Procedure

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | `camembert-base` |
| Learning rate | 5e-5 |
| Batch size | 32 |
| Epochs | 70 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Frozen layers | Embedding layers only |
| Trainable parameters | 85,062,923 / 110,039,819 (77.3%) |

### Training Environment

- **Hardware**: NVIDIA RTX 3090 GPU
- **Training time**: ~2-3 hours
- **Framework**: Transformers 4.45.2, PyTorch 2.5.1

## Evaluation

### Test Set Performance

Evaluated on a held-out test set containing ~20% unseen entities at each hierarchical level:

| Entity Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Country     | 0.98      | 1.00   | 0.99     | 4,648   |
| Region      | 0.99      | 0.99   | 0.99     | 6,744   |
| Province    | 0.99      | 0.98   | 0.99     | 541     |
| Department  | 0.99      | 0.99   | 0.99     | 1,433   |
| Village     | 0.94      | 0.98   | 0.96     | 3,236   |
| **Micro avg** | **0.98** | **0.98** | **0.98** | **16,602** |

### Comparison with Baselines

Tested on 1,000 manually annotated news articles:

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Baseline CamemBERT (no fine-tuning) | 0.87 | 0.58 | 0.70 |
| GLiNER (zero-shot) | 0.82 | 0.77 | 0.79 |
| **BF_NER (this model)** | **0.98** | **0.98** | **0.98** |

## Usage

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import CamembertTokenizerFast, CamembertForTokenClassification
import torch

# Load model and tokenizer
tokenizer = CamembertTokenizerFast.from_pretrained("CharlesAbdoulaye/BF_NER")
model = CamembertForTokenClassification.from_pretrained("CharlesAbdoulaye/BF_NER")
model.eval()

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

# Example text with all 5 entity types
text = "La crise alimentaire au Burkina Faso a frappe la region des Hauts-Bassins et la province du Houet. La ville de Bobo-Dioulasso et le village de Sya sont particulierement touches."

# Tokenize with offset mapping for span extraction
inputs = tokenizer(text, return_tensors="pt", truncation=True, return_offsets_mapping=True)
offset_mapping = inputs.pop("offset_mapping")[0].tolist()

with torch.no_grad():
    outputs = model(**inputs)

preds = torch.argmax(outputs.logits, dim=2)[0].tolist()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Reconstruct entities with character spans
entities, current = [], None
for idx, (pred_id, (start, end)) in enumerate(zip(preds, offset_mapping)):
    if start == 0 and end == 0:
        if current: entities.append(current); current = None
        continue
    label = id2label[pred_id]
    is_subword = not tokens[idx].startswith("\u2581")
    if label.startswith("B-"):
        if current: entities.append(current)
        current = {"type": label[2:], "start": start, "end": end}
    elif label.startswith("I-") or (is_subword and current):
        if current: current["end"] = end
    else:
        if current: entities.append(current); current = None
if current: entities.append(current)

# Merge consecutive same-type entities (handles hyphenated names)
merged = []
for ent in entities:
    if merged and merged[-1]["type"] == ent["type"]:
        gap = text[merged[-1]["end"]:ent["start"]]
        if gap in ("", "-", " -", "- "):
            merged[-1]["end"] = ent["end"]; continue
    merged.append(dict(ent))

for ent in merged:
    ent["text"] = text[ent["start"]:ent["end"]].rstrip(".,;:!?")
    print(f'{ent["text"]:20s} | {ent["type"]:15s} | span=({ent["start"]}, {ent["end"]})')
```

**Expected output:**
```
Burkina Faso         | country         | span=(24, 36)
Hauts-Bassins        | region          | span=(60, 73)
Houet                | province        | span=(92, 97)
Bobo-Dioulasso       | departement     | span=(111, 125)
Sya                  | village         | span=(143, 146)
```

## Limitations

1. **Geographic scope**: The model is trained exclusively on Burkina Faso administrative entities. It will not recognize locations from other countries with the same accuracy.

2. **Temporal coverage**: Training data is from 2009. Administrative boundaries and place names may have changed since then.

3. **Homonyms**: Village names that exist in multiple provinces may be ambiguous. The model does not perform disambiguation based on context.

4. **Spelling variations**: West African toponyms exhibit significant spelling variability (e.g., "Ouagadougou" vs "Ouaga"). The model handles common variations but may miss rare spellings not present in training data.

5. **Language**: Only French text is supported. The model will not work on texts in local languages (Mooré, Dioula, Fulfulde, etc.).

## Ethical Considerations

### Potential Biases

- **Media coverage bias**: Urban areas (especially Ouagadougou and Bobo-Dioulasso) are overrepresented in news articles compared to rural villages.
- **Administrative changes**: Administrative boundaries and names may have changed since the 2022 gazetteer was published.
- **Language bias**: French-language bias excludes indigenous language place names and local toponyms.

### Responsible Use

This model is intended for research and humanitarian applications:
- ✅ Food security monitoring and early warning systems
- ✅ Geographic information extraction for development organizations
- ✅ Academic research on crisis mapping and NLP

## Citation

If you use this model in your research, please cite:

```bibtex
@article{ngom2026stkgfs,
  title={Spatio-Temporal Knowledge Graph from Unstructured Texts: A Multi-Scale Approach for Food Security Monitoring},
  author={Ngom, Charles Abdoulaye and Rajaonarivo, Landy and Valentin, Sarah and Teisseire, Maguelonne},
  journal={AGILE: GIScience Series},
  year={2026},
  
}
```

## Contact

For questions about this model:
- Charles Abdoulaye Ngom: [charles.ngom@inrae.fr](mailto:charles.ngom@inrae.fr)
- Landy Rajaonarivo: [landy.rajaonarivo@inrae.fr](mailto:landy.rajaonarivo@inrae.fr)
- Sarah Valentin: [sarah.valentin@cirad.fr](mailto:sarah.valentin@cirad.fr)
- Maguelonne Teisseire: [maguelonne.teisseire@inrae.fr](mailto:maguelonne.teisseire@inrae.fr)

## Acknowledgments

- Administrative hierarchy data: [2022 Statistical Yearbook](http://cns.bf/IMG/pdf/matds_annuaire_at_2022.pdf), Burkina Faso Ministry of Territorial Administration
- Base model: [CamemBERT](https://camembert-model.fr/) (Martin et al., 2020)
- Geographic enrichment: [Wikidata](https://www.wikidata.org/)

## License

This model is released under the MIT License. See the [LICENSE](LICENSE) file for details.
