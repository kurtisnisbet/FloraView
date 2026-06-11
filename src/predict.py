'''predict.py - this script takes an observation (a photo + tabular fields) and
returns the four biomass predictions.'''

import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.multimodal import MultiModalPredictor

TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g"]

# Project root is two levels up from this file (src/predict.py -> src -> root)
ROOT_DIR  = Path(__file__).parent.parent
MODEL_DIR = ROOT_DIR / "models" / "azure"
DATA_FILE = ROOT_DIR / "data" / "processed" / "df_model.csv"
RAW_DIR   = ROOT_DIR / "data" / "raw"

def load_predictors():
    # Load only the models whose folders are actually present. This lets the
    # same code run locally (all 4 models) and on the deployed Space (a subset,
    # to fit the 1GB free-tier storage limit).
    predictors = {}
    for target in TARGETS:
        model_path = MODEL_DIR / f"autogluon_{target}"
        if model_path.exists():
            predictors[target] = MultiModalPredictor.load(str(model_path))
    return predictors

def predict(input_df, predictors, targets=TARGETS):
    # targets defaults to all four, but the caller can pass a subset
    # (e.g. ["GDM_g"]) to run fewer models -- this is what "fast mode" uses.
    results = {}
    for target in targets:
        # predict on the log1p scale (mirrors training).
        # as_pandas=False -> get a raw NumPy array. AutoGluon's pandas conversion
        # breaks on single-row regression inputs (it tries to wrap a lone scalar
        # in a DataFrame), so we skip it and handle the array ourselves.
        # realtime=True -> use AutoGluon's lightweight inference path. Without it,
        # every call spins up a full PyTorch Lightning Trainer + DataLoader workers,
        # which costs minutes per prediction for a single image (esp. on Windows).
        log_pred = predictors[target].predict(input_df, as_pandas=False, realtime=True)

        # reverse the transform: expm1 to get back to the original scale
        grams = np.expm1(log_pred)

        # biomass cannot be negative, so clip anything below zero
        grams = np.clip(grams, 0, None)

        # ravel flattens to 1-D, then take the first (only) value as a plain float
        results[target] = float(np.ravel(grams)[0])
    return results

if __name__ == "__main__":
    # load one real row from the model-ready dataset as a test input
    df = pd.read_csv(DATA_FILE)
    sample = df.head(1).copy()

    # the CSV stores image_path relative to data/raw (e.g. "train/ID123.jpg");
    # AutoGluon needs an absolute path to open the file, so rewrite the column
    sample["image_path"] = sample["image_path"].apply(lambda p: str(RAW_DIR / p))

    predictors = load_predictors()
    preds = predict(sample, predictors)

    image_file = df.iloc[0]["image_path"]   # e.g. "train/ID1011485656.jpg"
    print("Image:", image_file)
    print("Predicted (g):", {k: round(v, 2) for k, v in preds.items()})
    # eyeball accuracy: print the TRUE values from the same row
    print("Actual    (g):", {t: float(df.iloc[0][t]) for t in TARGETS})

    # Save a persistent record so we have a reproducible sanity-check artifact.
    out = pd.DataFrame({
        "image_file": image_file,
        "target":     TARGETS,
        "predicted_g": [round(preds[t], 4) for t in TARGETS],
        "actual_g":    [float(df.iloc[0][t]) for t in TARGETS],
    })
    RESULTS_DIR = ROOT_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "predict_test.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")