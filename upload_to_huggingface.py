#!/usr/bin/env python3
"""
Script to upload the BF_NER model and datasets to HuggingFace Hub.

Usage:
    1. Install dependencies: pip install huggingface_hub
    2. Login: huggingface-cli login
    3. Run: python upload_to_huggingface.py
"""

import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path


def upload_model():
    """Upload the fine-tuned CamemBERT model to HuggingFace."""
    print("=" * 70)
    print("UPLOADING MODEL TO HUGGINGFACE")
    print("=" * 70)

    model_path = "/data/charles/agile/camembert-ner-finetuned/checkpoint-15000"
    repo_id = "CharlesAbdoulaye/BF_NER"

    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model directory not found: {model_path}")
        return False

    print(f"\nModel path: {model_path}")
    print(f"Target repository: {repo_id}")

    try:
        # Create repository (or get existing)
        print("\n1. Creating repository on HuggingFace...")
        create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        print("   Repository ready")

        # Upload model files
        print("\n2. Uploading model checkpoint...")
        api = HfApi()
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print("   Model uploaded successfully")

        # Upload model card
        model_card_path = "MODEL_CARD.md"
        if os.path.exists(model_card_path):
            print("\n3. Uploading model card (README.md)...")
            api.upload_file(
                path_or_fileobj=model_card_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model"
            )
            print("   Model card uploaded successfully")

        print(f"\nMODEL UPLOAD COMPLETE")
        print(f"View at: https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"\nERROR during model upload: {e}")
        return False


def upload_datasets():
    """Upload the training datasets to HuggingFace."""
    print("\n" + "=" * 70)
    print("UPLOADING DATASETS TO HUGGINGFACE")
    print("=" * 70)

    repo_id = "CharlesAbdoulaye/BF_NER_datasets"

    dataset_files = [
        "Fine-Tuning/annotations/train_extended_bio_feb.json",
        "Fine-Tuning/annotations/val_extended_bio_feb.json",
        "Fine-Tuning/annotations/test_extended_bio_feb.json"
    ]

    # Check if files exist
    missing_files = [f for f in dataset_files if not os.path.exists(f)]
    if missing_files:
        print(f"\nERROR: Dataset files not found:")
        for f in missing_files:
            print(f"  - {f}")
        return False

    print(f"\nTarget repository: {repo_id}")
    print(f"Files to upload: {len(dataset_files)}")

    try:
        # Create repository
        print("\n1. Creating dataset repository on HuggingFace...")
        create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
        print("   Repository ready")

        # Upload each dataset file
        print("\n2. Uploading dataset files...")
        api = HfApi()

        for i, file_path in enumerate(dataset_files, 1):
            filename = os.path.basename(file_path)
            print(f"   [{i}/{len(dataset_files)}] Uploading {filename}...")

            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"       Uploaded successfully")

        # Create a simple dataset card
        print("\n3. Creating dataset card...")
        dataset_card = """---
language: fr
license: mit
task_categories:
- token-classification
tags:
- ner
- french
- burkina-faso
- bio-tagging
---

# BF_NER Training Datasets

BIO-tagged training data for the [BF_NER model](https://huggingface.co/CharlesAbdoulaye/BF_NER).

## Dataset Description

This dataset contains 86,252 sentences with BIO tags for geographic Named Entity Recognition in French, specifically for Burkina Faso administrative entities.

### Splits

| Split | Sentences | Description |
|-------|-----------|-------------|
| Train | 59,900 | Training set |
| Validation | 14,758 | Validation set for hyperparameter tuning |
| Test | 11,594 | Held-out test set with ~20% unseen entities |

### Entity Types

- `country`: Country-level entities
- `region`: 13 regions of Burkina Faso
- `province`: 45 provinces
- `departement`: 351 departments
- `village`: 7,936 villages

### Data Format

Each JSON file contains a list of examples with:
- `tokens`: List of word tokens
- `tags`: List of BIO tags (B-{type}, I-{type}, O)

Example:
```json
{
  "tokens": ["Les", "inondations", "touchent", "Ouagadougou"],
  "tags": ["O", "O", "O", "B-departement"]
}
```

## Citation

```bibtex
@inproceedings{ngom2026stkgfs,
  title={Spatio-Temporal Knowledge Graph from Unstructured Texts:
         A Multi-Scale Approach for Food Security Monitoring},
  author={Ngom, Charles Abdoulaye and Rajaonarivo, Landy and Valentin, Sarah and Teisseire, Maguelonne},
  booktitle={AGILE: GIScience Series},
  year={2026}
}
```

## License

MIT License
"""

        api.upload_file(
            path_or_fileobj=dataset_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("   Dataset card uploaded")

        print(f"\nDATASET UPLOAD COMPLETE")
        print(f"View at: https://huggingface.co/datasets/{repo_id}")
        return True

    except Exception as e:
        print(f"\nERROR during dataset upload: {e}")
        return False


def main():
    """Main execution."""
    print("\n" + "=" * 70)
    print("HUGGINGFACE UPLOAD SCRIPT")
    print("BF_NER Model and Datasets")
    print("=" * 70)

    print("\nIMPORTANT: Make sure you have:")
    print("  1. Installed huggingface_hub: pip install huggingface_hub")
    print("  2. Logged in: huggingface-cli login")
    print()

    response = input("Continue with upload? (y/n): ").strip().lower()
    if response != 'y':
        print("Upload cancelled.")
        return

    # Upload model
    model_success = upload_model()

    # Upload datasets
    dataset_success = upload_datasets()

    # Summary
    print("\n" + "=" * 70)
    print("UPLOAD SUMMARY")
    print("=" * 70)
    print(f"Model upload: {'SUCCESS' if model_success else 'FAILED'}")
    print(f"Dataset upload: {'SUCCESS' if dataset_success else 'FAILED'}")

    if model_success and dataset_success:
        print("\nAll uploads completed successfully!")
        print("\nNext steps:")
        print("  1. Visit https://huggingface.co/CharlesAbdoulaye/BF_NER")
        print("  2. Review the model card and add any additional information")
        print("  3. Test the model with the usage examples")
    else:
        print("\nSome uploads failed. Please check the errors above.")


if __name__ == "__main__":
    main()
