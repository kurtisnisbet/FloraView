# image2biomass

Predicting pasture biomass from field photographs using multi-modal deep learning.

---

## Overview

Pasture biomass measurement is critical for livestock farm management but is expensive and time-consuming when done manually. This project builds a machine learning pipeline that predicts dry biomass yield (in grams) from smartphone images of pasture combined with tabular field measurements.

The model is trained on the CSIRO Pasture Biomass dataset and predicts four biomass components simultaneously:

| Target | Description |
|---|---|
| `Dry_Clover_g` | Dry clover biomass |
| `Dry_Dead_g` | Dry dead material biomass |
| `Dry_Green_g` | Dry green (live) biomass |
| `GDM_g` | Total green dry matter |

---

## Dataset

- **Source:** CSIRO Pasture Biomass dataset
- **Samples:** 357 field observations across multiple Australian states (NSW, VIC, TAS, SA, WA)
- **Inputs:** Pasture photograph + tabular features
- **Tabular features:** NDVI (Pre_GSHH_NDVI), average sward height (Height_Ave_cm), Australian state, month, season, and one-hot encoded species presence (14 pasture species including Ryegrass, Clover, Phalaris, SubClover, and others)

The raw data consists of per-species rows that were pivoted to a wide format (one row per image/observation) during feature engineering.

### Sample images

![Sample pasture images](assets/eda_sample_images.png)

### Species and geographic distribution

![Species distribution](assets/eda_species_distribution.png)

![State and season distribution](assets/eda_state_season_heatmap.png)

### Target distributions

Targets are heavily right-skewed with zero inflation — many observations have zero clover or dead material. A log1p transform is applied before training.

![Target distributions (raw)](assets/eda_target_distributions_raw.png)

![Target distributions (log-transformed)](assets/eda_target_distributions_log.png)

### Feature correlations

![Correlation matrix](assets/eda_correlation_matrix.png)

---

## Methodology

### Model

[AutoGluon MultiModalPredictor](https://auto.gluon.ai/stable/tutorials/multimodal/index.html) is used for all targets. It fuses image and tabular features in a late-fusion MLP architecture — a pretrained vision encoder processes the image, a separate branch handles tabular features, and both representations are concatenated before the final regression head.

A separate predictor is trained per target. AutoGluon does not natively support multi-output regression, so this is the recommended approach.

### Key design decisions

**Log1p target transform:** All four targets are right-skewed with zero inflation. Targets are log1p-transformed before training and expm1-reversed at evaluation, which stabilises training and improves RMSE on the original grams scale.

**Backbone comparison:** Three image encoder architectures were compared to identify the best feature extractor for pasture imagery. The default AutoGluon backbone outperformed alternatives on the two highest-R² targets.

**Cross-validation:** Given the small dataset size (357 samples), 5-fold CV was used to obtain more reliable performance estimates than a single train/val split.

### Infrastructure

Training was run on Azure ML using a Tesla T4 GPU cluster (`Standard_NC4as_T4_v3`), providing approximately 35× speedup over local CPU. MLflow was used for experiment tracking and run comparison.

---

## Results

### Backbone comparison

Three architectures were compared on the same 80/20 train/val split.

![Backbone comparison](assets/backbone_comparison.png)

| Target | Default | Swin-Base | EfficientNet-B4 |
|---|---|---|---|
| Dry_Clover_g | 0.563 | **0.629** | 0.607 |
| Dry_Dead_g | 0.280 | **0.429** | 0.194 |
| Dry_Green_g | **0.726** | 0.626 | 0.614 |
| GDM_g | **0.825** | 0.808 | 0.516 |

The default AutoGluon backbone performs best overall, particularly on GDM_g and Dry_Green_g. Swin-Base outperforms on the harder, low-signal targets (Dry_Clover_g and Dry_Dead_g).

### 5-Fold cross-validation

![CV results](assets/cv_results.png)

| Target | R² mean | R² std |
|---|---|---|
| Dry_Clover_g | 0.457 | ± 0.026 |
| Dry_Dead_g | 0.285 | ± 0.080 |
| Dry_Green_g | 0.695 | ± 0.059 |
| GDM_g | 0.726 | ± 0.036 |

The low std on GDM_g and Dry_Green_g confirms these predictions are stable across different data splits. The higher std on Dry_Dead_g reflects the difficulty of that target — dead material has weak visual and tabular signal.

### Predicted vs actual

![Predicted vs actual](assets/pred_vs_actual.png)

### Residuals

![Residuals](assets/residuals.png)

### Total biomass

Total biomass (sum of all four predicted targets) vs actual.

![Total biomass](assets/total_biomass.png)

### Interpretation

- **GDM_g** (total green dry matter) is the most predictable target (R²=0.73–0.83), which is the metric most relevant to practical farm management
- **Dry_Dead_g** is the hardest target (R²=0.28–0.43) — dead biomass has limited visual distinction from soil and varies with weather history
- The model beats a naive mean-prediction baseline on all four targets across all backbones

---

## Project structure

```
image2biomass/
├── assets/                    # plots and figures for this README
├── data/
│   ├── raw/                   # CSIRO pasture images + train.csv (not tracked in git)
│   └── processed/
│       ├── df_wide.csv        # pivoted wide format (pre feature engineering)
│       └── df_model.csv       # final model-ready dataset (357 rows × 26 cols)
├── models/                    # trained model checkpoints (not tracked in git)
│   └── azure_default/         # best Azure ML run (default backbone, Tesla T4)
├── notebooks/
│   ├── 01_eda.ipynb           # exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # pivot, species encoding, target inspection
│   ├── 03_modelling.ipynb     # local CPU smoke test (120 s per target)
│   ├── 04_azure_ml_job.ipynb  # Azure ML job submission and experiment management
│   └── 05_evaluation.ipynb    # results analysis, backbone comparison, CV summary
├── results/                   # downloaded Azure ML output CSVs and plots
├── src/
│   └── train.py               # training script (runs on Azure ML GPU cluster)
├── requirements.txt
└── .env                       # Azure credentials — not committed to git
```

---

## Reproducing the experiments

### Prerequisites

- Python 3.11
- An Azure ML workspace with a GPU compute cluster (tested on `Standard_NC4as_T4_v3`, Tesla T4)
- Azure CLI installed and authenticated (`az login`)

### Setup

```bash
git clone https://github.com/<your-username>/image2biomass.git
cd image2biomass
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=<your-resource-group>
AZURE_WORKSPACE_NAME=<your-workspace-name>
```

### Running notebooks

Run notebooks in order from `notebooks/`:

1. `01_eda.ipynb` — explore the raw data
2. `02_feature_engineering.ipynb` — produce `df_model.csv`
3. `03_modelling.ipynb` — local smoke test to verify the pipeline end-to-end
4. `04_azure_ml_job.ipynb` — submit training jobs to Azure ML
5. `05_evaluation.ipynb` — analyse results

> **Note:** Raw image data is not included in this repository due to size. Contact CSIRO for dataset access.

### Training script arguments

`src/train.py` accepts the following arguments when submitted as an Azure ML job:

| Argument | Default | Description |
|---|---|---|
| `--data_path` | required | Path to mounted data asset |
| `--time_limit` | 14400 | Seconds per target per fold |
| `--backbone` | None | timm backbone name (None = AutoGluon default) |
| `--n_folds` | 1 | Number of CV folds (1 = standard split) |

---

## Limitations and future work

- **Dataset size:** 357 samples is small for deep learning. Collecting additional labelled images, particularly for underrepresented species and seasons, would likely improve all targets.
- **Dry_Dead_g ceiling:** Dead biomass prediction appears near its ceiling with current inputs. Additional features such as time-since-rain or spectral indices may help.
- **Deployment:** The trained models are not yet deployed. A natural next step is a simple inference API or mobile-friendly interface that accepts a photo and returns biomass predictions.
- **Temporal generalisation:** The current train/val split does not account for temporal structure. A time-based split would give a stricter estimate of out-of-sample performance.

---

## Technologies

Python · AutoGluon · PyTorch · Azure ML · MLflow · scikit-learn · pandas · matplotlib
