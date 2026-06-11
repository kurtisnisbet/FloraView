"""
train.py -- AutoGluon MultiModalPredictor training script for Azure ML.

This script is the cloud equivalent of the 03_modelling.ipynb file. It is submitted as
an Azure ML command job from 04_azure_ml_job.ipynb, which passes arguments via
argparse.

Key design decisions:
- One predictor is trained per target (4 total). AutoGluon MultiModalPredictor
  does not natively support multi-output regression, so this was the supported
  approach.

- Targets are log1p-transformed before training to compress the right skew and
  handle zero inflation found during 01_eda. A copy of the original and untransformed
  DataFrame is kept for computing metrics in the original grams scale in 05_evaluation.

- The backbone model is configurable via --backbone. Passing None
  uses AutoGluon's default. Named timm backbones can be specified to run
  architecture comparison experiments.

- Cross-validation is supported via --n_folds. When n_folds > 1, the dataset
  is split into k folds and one predictor is trained per fold per target.
  Per-fold and aggregate (mean +/- std) metrics are logged to MLflow.

- All outputs are saved to outputs/ which Azure ML captures to cloud storage.
"""

import subprocess
import sys

# Install setuptools before any other imports -- the base Docker image ships
# with setuptools 82 which is missing pkg_resources in some conda envs.
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "setuptools==69.5.1", "--force-reinstall", "--quiet",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "azureml-mlflow", "--quiet",
])

import argparse
import json
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    help="Path to the mounted data directory (set by Azure ML at runtime).",
)

# Time limit in seconds
parser.add_argument(
    "--time_limit",
    type=int,
    default=14400,
    help="Seconds AutoGluon is allowed to train per target (or per fold-target "
         "when using cross-validation). Default: 14400 (4 hours).",
)

# Using the default backbone, as determined by AutoGluon based on dataset size properties.
parser.add_argument(
    "--backbone",
    type=str,
    default=None,
    help="timm checkpoint name for the image encoder (e.g. "
         "'swin_base_patch4_window7_224'). None = AutoGluon default.",
)

# Number of cross-validation folds.
parser.add_argument(
    "--n_folds",
    type=int,
    default=1,
    help="Number of cross-validation folds. 1 = standard 80/20 train/val "
         "split. 5 = 5-fold CV.",
)
args = parser.parse_args()


# Constants
TARGETS     = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g"]
RANDOM_SEED = 42
VAL_SIZE    = 0.2


# Load data and resolve image paths
data_path = Path(args.data_path)

df = pd.read_csv(data_path / "processed" / "df_model.csv")

# Image paths in df_model.csv are relative ("train/<id>.jpg").
# Resolve to absolute paths so AutoGluon can open each file.
df["image_path"] = df["image_path"].apply(
    lambda x: str(data_path / "raw" / x)
)

# Fail fast if any images are missing rather than silently training on a broken dataset.
missing = [p for p in df["image_path"] if not Path(p).exists()]
if missing:
    raise FileNotFoundError(
        f"{len(missing)} image(s) not found. First missing: {missing[0]}"
    )
print(f"Loaded {len(df)} samples. All images verified.")


# Log-transform targets
# Keep df_original (untransformed) for computing metrics in grams.
df_original = df.copy()
for target in TARGETS:
    df[target] = np.log1p(df[target])


# Build AutoGluon hyperparameters dict
# Only populated when a specific backbone is requested. When empty,
# AutoGluon uses its own default architecture selection.
hyperparameters = {}
if args.backbone:
    hyperparameters["model.timm_image.checkpoint_name"] = args.backbone


# MLflow setup
mlflow.autolog()
mlflow.log_params({
    "backbone"                : args.backbone or "autogluon_default",
    "n_folds"                 : args.n_folds,
    "time_limit_per_target_s" : args.time_limit,
    "target_transform"        : "log1p",
    "targets"                 : str(TARGETS),
    "random_seed"             : RANDOM_SEED,
})

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
Path("outputs").mkdir(exist_ok=True)


# Helper to train and evaluate one predictor per target
# Accepts log-transformed train/val DataFrames and the original-scale val
# DataFrame for metric computation. 
def train_and_evaluate(df_train, df_val_log, df_val_orig, fold_label=""):
    """Train one MultiModalPredictor per target and return results."""
    predictors = {}
    results    = {}

    for target in TARGETS:
        print(f"\n--- Training: {target}{fold_label} ---")
        predictor = MultiModalPredictor(
            label=target,
            problem_type="regression",
            path=f"outputs/autogluon_{target}{fold_label}",
            eval_metric="rmse",
        )
        predictor.fit(
            train_data=df_train,
            time_limit=args.time_limit,
            hyperparameters=hyperparameters if hyperparameters else None,
        )
        predictors[target] = predictor

        # y_true is already in original grams scale (from df_val_orig).
        # y_pred is in log1p space -- reverse with expm1.
        y_true = df_val_orig[target].values
        y_pred = np.expm1(predictor.predict(df_val_log).values)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        results[target] = {"RMSE": rmse, "R2": r2}

    return predictors, results


# Standard train/val split (n_folds == 1)
if args.n_folds == 1:

    df_train, df_val = train_test_split(
        df, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    df_val_orig = df_original.loc[df_val.index]
    print(f"Train: {len(df_train)}  |  Val: {len(df_val)}")

    predictors, results = train_and_evaluate(df_train, df_val, df_val_orig)

    for target, metrics in results.items():
        mlflow.log_metrics({
            f"{target}_rmse": metrics["RMSE"],
            f"{target}_r2"  : metrics["R2"],
        })

    results_df = pd.DataFrame(results).T.round(4)
    print("\n--- Validation results (original scale) ---")
    print(results_df.to_string())
    results_df.to_csv(f"outputs/results_{timestamp}.csv")

    # Save fit summaries for all targets
    fit_summaries = {}
    for target in TARGETS:
        fit_summaries[target] = predictors[target].fit_summary(show_plot=False)
    with open(f"outputs/fit_summary_{timestamp}.json", "w") as f:
        json.dump(fit_summaries, f, indent=2, default=str)

    print(f"\nSaved results_{timestamp}.csv and fit_summary_{timestamp}.json to outputs/")


# K-fold cross-validation (n_folds > 1)
# One predictor is trained per fold per target. Per-fold metrics are logged
# individually to MLflow, and aggregate mean +/- std are logged as summaries.
else:

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_results = {target: {"RMSE": [], "R2": []} for target in TARGETS}
    indices    = np.arange(len(df))

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1} / {args.n_folds}  "
              f"(train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'=' * 50}")

        df_train_fold    = df.iloc[train_idx]
        df_val_fold      = df.iloc[val_idx]
        df_val_orig_fold = df_original.iloc[val_idx]

        _, fold_results = train_and_evaluate(
            df_train_fold,
            df_val_fold,
            df_val_orig_fold,
            fold_label=f"_fold{fold}",
        )

        for target, metrics in fold_results.items():
            cv_results[target]["RMSE"].append(metrics["RMSE"])
            cv_results[target]["R2"].append(metrics["R2"])
            mlflow.log_metrics({
                f"{target}_fold{fold}_rmse": metrics["RMSE"],
                f"{target}_fold{fold}_r2"  : metrics["R2"],
            })

    # Aggregate across folds and log summary metrics
    cv_summary = {}
    for target in TARGETS:
        rmse_vals = cv_results[target]["RMSE"]
        r2_vals   = cv_results[target]["R2"]
        cv_summary[target] = {
            "RMSE_mean" : round(float(np.mean(rmse_vals)), 4),
            "RMSE_std"  : round(float(np.std(rmse_vals)),  4),
            "R2_mean"   : round(float(np.mean(r2_vals)),   4),
            "R2_std"    : round(float(np.std(r2_vals)),    4),
        }
        mlflow.log_metrics({
            f"{target}_cv_rmse_mean" : cv_summary[target]["RMSE_mean"],
            f"{target}_cv_rmse_std"  : cv_summary[target]["RMSE_std"],
            f"{target}_cv_r2_mean"   : cv_summary[target]["R2_mean"],
            f"{target}_cv_r2_std"    : cv_summary[target]["R2_std"],
        })

    results_df = pd.DataFrame(cv_summary).T
    print("\n--- CV Results (original scale, mean +/- std across folds) ---")
    print(results_df.to_string())
    results_df.to_csv(f"outputs/cv_results_{timestamp}.csv")
    print(f"\nSaved cv_results_{timestamp}.csv to outputs/")
