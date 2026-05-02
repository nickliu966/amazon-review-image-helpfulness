# amazon-review-image-helpfulness

This repository contains the code for thesis project on how user-generated review images contribute to online review helpfulness. The project compares multiple image representation families within a common predictive framework:

1. Interpretable visual quality and photographic descriptors  
2. Object-detection-based semantic and layout features  
3. Neural network-based image representations  

The main empirical setting is Amazon product reviews, with Electronics used as the example category throughout the running instructions.

---

## Repository Structure

```text
YOUR_PROJECT_FOLDER_NAME/
  src/
    __init__.py
    utils.py
    image_loading.py
    text_features.py
    visual_quality.py
    semantic_layout.py
    neural_representations.py
    clip_similarity.py
    product_features.py
    dataset_building.py
    model_utils.py
    qualitative.py
    visual_sanity.py
    plotting.py

  scripts/
    01_text_features.py
    02_image_features.py
    03_build_model_data.py
    04_presence_models.py
    05_image_representation_models.py
    06_qualitative_analysis.py
    07_figures.py
    09_visual_sanity_check.py


  README.md
  requirements.txt
  .gitignore
```

Generated data, model outputs, logs, figures, and model weights are not tracked in Git.

---

## Data

The raw Amazon review data and product metadata are not included in this repository. They can be found [here](https://amazon-reviews-2023.github.io/)

Example review file:

```text
path/to/raw_data/Electronics.jsonl
```

Example metadata file:

```text
path/to/raw_data/meta_Electronics.jsonl
```

Replace these paths with the actual raw-data location on your machine.

---

## Environment

Create or activate the Python environment used for the project:

```bash
conda activate amazon_review_image_helpfulness
```

Install package dependencies:

```bash
pip install -r requirements.txt
```

---

## Pipeline Overview

The main workflow is:

```text
Raw reviews + product metadata
        ↓
01_text_features.py
        ↓
02_image_features.py
        ↓
03_build_model_data.py
        ↓
04_presence_models.py
05_image_representation_models.py
        ↓
06_qualitative_analysis.py
07_figures.py
09_visual_sanity_check.py
```

---

## Example: Electronics Pipeline

Run all commands from the project root:

```bash
cd path/to/YOUR_PROJECT_FOLDER_NAME
```

### 1. Extract text features

```bash
python scripts/01_text_features.py \
  --category electronics \
  --input "path/to/raw_data/Electronics.jsonl" \
  --out-dir "data/text_electronics" \
  --chunk-size 500000
```

### 2. Extract image features

```bash
python scripts/02_image_features.py \
  --category electronics \
  --text-dir "data/text_electronics" \
  --product-meta "path/to/raw_data/meta_Electronics.jsonl" \
  --out-dir "data/image_electronics" \
  --min-reviews 50 \
  --max-review-images 6 \
  --max-product-images 3 \
  --workers 32 \
  --max-inflight 128 \
  --gpu-batch 64
```

### 3. Build model data

```bash
python scripts/03_build_model_data.py \
  --category electronics \
  --text-dir "data/text_electronics" \
  --image-scalar-path "data/image_electronics/electronics_image_scalars.parquet" \
  --image-vector-path "data/image_electronics/electronics_image_vectors.npz" \
  --product-meta "path/to/raw_data/meta_Electronics.jsonl" \
  --out-dir "data/model_electronics" \
  --master-csv "data/master_electronics.csv" \
  --min-reviews 50
```

This produces:

```text
data/model_electronics/
  model_df_scalars_electronics.parquet
  model_df_vectors_electronics.npz
  feature_blocks_electronics.json
  product_split_electronics.joblib

data/master_electronics.csv
```

The scalar parquet file is used for tabular modeling. The vector `.npz` file stores low-level and high-level CNN representations. The master CSV is used for qualitative analysis and figure generation.

### 4. Run image-presence models

```bash
python scripts/04_presence_models.py \
  --category electronics \
  --scalars "data/model_electronics/model_df_scalars_electronics.parquet" \
  --feature-blocks "data/model_electronics/feature_blocks_electronics.json" \
  --out-dir "outputs/electronics" \
  --split-path "data/model_electronics/product_split_electronics.joblib" \
  --n-param-samples 40 \
  --save-predictions
```

### 5. Run image-representation models

```bash
python scripts/05_image_representation_models.py \
  --category electronics \
  --scalars "data/model_electronics/model_df_scalars_electronics.parquet" \
  --vectors "data/model_electronics/model_df_vectors_electronics.npz" \
  --feature-blocks "data/model_electronics/feature_blocks_electronics.json" \
  --out-dir "outputs/electronics" \
  --split-path "data/model_electronics/product_split_electronics.joblib" \
  --n-low-pcs 10 \
  --n-high-pcs 10 \
  --n-param-samples 40 \
  --save-importance \
  --save-all-predictions
```

### 6. Run qualitative analysis

```bash
python scripts/06_qualitative_analysis.py \
  --category electronics \
  --category-label "Electronics" \
  --prediction-path "outputs/electronics/M0_img_baseline_diagnostic_predictions_electronics_image_only.csv" \
  --raw-csv "data/master_electronics.csv" \
  --out-dir "outputs/qualitative" \
  --top-pool 500 \
  --n-final 20 \
  --min-age-days 180
```

### 7. Generate figures

```bash
python scripts/07_figures.py \
  --category electronics \
  --category-label "Electronics" \
  --out-dir "outputs/figures_electronics" \
  --master-csv "data/master_electronics.csv" \
  --product-meta "path/to/raw_data/meta_Electronics.jsonl" \
  --scalars "data/model_electronics/model_df_scalars_electronics.parquet" \
  --results "outputs/electronics/all_model_results_electronics_image_only.csv" \
  --pos-cases "outputs/qualitative/qual_pos20_electronics.csv" \
  --neg-cases "outputs/qualitative/qual_neg20_electronics.csv" \
  --pos-case-numbers "1,2,6,10" \
  --neg-case-numbers "1,3,11,16"
```

This creates EDA figures, model-performance figures, and qualitative case grids.

### 8. Run visual sanity check

```bash
python scripts/09_visual_sanity_check.py \
  --category electronics \
  --category-label "Electronics" \
  --baseline-predictions "outputs/electronics/M0_img_baseline_diagnostic_predictions_electronics_image_only.csv" \
  --full-predictions "outputs/electronics/H6_full_diagnostic_predictions_electronics_image_only.csv" \
  --master-csv "data/master_electronics.csv" \
  --out-dir "outputs/visual_sanity_electronics" \
  --n-cases 10 \
  --max-base-proba 0.60 \
  --min-helpful-vote 10 \
  --min-review-age-days 180
```

The visual sanity check compares image-baseline predictions with full image-content model predictions and selects cases where the full model most increases or decreases predicted helpfulness.

---

## Running Other Categories

The same pipeline can be run for other categories by changing the category name and file paths.

For Beauty & Personal Care, use:

```text
--category bc
--category-label "Beauty & Personal Care"
--input "path/to/raw_data/Beauty_and_Personal_Care.jsonl"
--product-meta "path/to/raw_data/meta_Beauty_and_Personal_Care.jsonl"
```

For quick development runs, use smaller values such as:

```text
--min-reviews 2
--sample-n 1000
--n-low-pcs 2
--n-high-pcs 2
--n-param-samples 2
```

---

## Outputs

Main outputs include:

```text
data/model_electronics/
  model_df_scalars_electronics.parquet
  model_df_vectors_electronics.npz
  feature_blocks_electronics.json

outputs/electronics/
  all_model_results_electronics_image_only.csv
  model prediction files
  feature importance files

outputs/qualitative/
  qualitative case CSVs

outputs/figures_electronics/
  EDA figures
  model comparison figures
  qualitative image grids

outputs/visual_sanity_electronics/
  visual sanity-check CSVs and image grids
```

---
