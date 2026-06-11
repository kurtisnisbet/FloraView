'''app.py - Gradio demo for FloraView.

Upload a pasture photo + a few simple field measurements, get back predicted
biomass (grams). Designed to be usable on a phone in the field, so we ask for
as little as possible and impute / default the rest.

Run locally with:  python app.py
'''

import csv
from datetime import datetime
from pathlib import Path
import calendar

import gradio as gr
import pandas as pd

# Month dropdown: show month names and pass the integer (label, value) to the function.
MONTH_CHOICES = [(calendar.month_name[m], m) for m in range(1, 13)]

# Reuse the inference logic from src/predict.py.
from src.predict import load_predictors, predict, TARGETS

# --- Load models ONCE at startup (module level = a single time when app boots). ---
print("Loading models (this takes a moment)...")
PREDICTORS = load_predictors()
# Whichever targets actually loaded: all 4 locally, GDM + Clover on the Space.
AVAILABLE_TARGETS = [t for t in TARGETS if t in PREDICTORS]
print(f"Models loaded: {AVAILABLE_TARGETS}. Launching app...")

# --- Full species one-hot column list, in df_model.csv order. ---
ALL_SPECIES = [
    "BarleyGrass", "Barleygrass", "Bromegrass", "Capeweed", "Clover", "CrumbWeed",
    "Fescue", "Lucerne", "Mixed", "Phalaris", "Ryegrass", "SilverGrass",
    "SpearGrass", "SubcloverDalkeith", "SubcloverLosa", "WhiteClover",
]

# Only the  common species are offered in the UI. This app is to be used in the field, and simplicity is king.
COMMON_SPECIES = ["Clover", "Ryegrass", "Phalaris", "Fescue", "Lucerne"]

# Defaults for fields we deliberately don't ask the user for:
NDVI_DEFAULT = 0.66                       # dataset average (range 0.16-0.91)
DEFAULT_SPECIES = ["Clover", "Ryegrass"]  # imputed when user selects no species

STATES = ["NSW", "Vic", "Tas", "WA"]

# Southern-hemisphere season mapping.
MONTH_TO_SEASON = {
    12: "Summer", 1: "Summer", 2: "Summer",
    3: "Autumn", 4: "Autumn", 5: "Autumn",
    6: "Winter", 7: "Winter", 8: "Winter",
    9: "Spring", 10: "Spring", 11: "Spring",
}

# Optional CSV log (ephemeral on the Space; handy locally). One row per prediction.
LOG_PATH = Path(__file__).parent / "results" / "app_predictions_log.csv"

# Columns for the per-session history table.
HISTORY_COLS = ["Photo ID", "State", "Month", "Height", "Species present", "GDM (g)", "Clover (g)", "Green (g)"]

INSTRUCTIONS = """**Instructions:** Upload a pasture photo and enter a few field details to estimate
total Green Dry Matter (GDM), Clover biomass, and non-Clover biomass, in grams.

A fourth component, dead biomass, is not included in this hosted demo due to storage limits on HuggingFace.
The full four-component model and instructions are available at:
https://github.com/kurtisnisbet/pasture-biomass-predictor"""


def log_prediction(image, state, month, season, height, species, preds):
    """Append one row to the CSV log, writing the header on first use."""
    LOG_PATH.parent.mkdir(exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["timestamp", "image_file", "State", "Month", "Season",
                 "Height_Ave_cm", "species"] + TARGETS
            )
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            Path(image).name, state, month, season, height,
            "|".join(species),
            *[round(preds[t], 4) if t in preds else "" for t in TARGETS],
        ])


def run_prediction(image, state, month, height, species_selected, history):
    """Predict biomass for one observation and append it to the session history.

    Returns three things (matching the three outputs wired to the button):
      1. the results table for THIS prediction
      2. the updated history table (to display)
      3. the updated history list (to store back in gr.State)
    """
    # Gradio passes None if no image was uploaded. Leave history untouched.
    if image is None:
        return ([["Please upload a pasture photo.", ""]],
                pd.DataFrame(history or [], columns=HISTORY_COLS), history)

    species = species_selected if species_selected else DEFAULT_SPECIES
    season = MONTH_TO_SEASON[int(month)]

    # Build a single model-ready row (all 26 columns df_model.csv expects).
    row = {
        "image_path": image,
        "State": state,
        "Pre_GSHH_NDVI": NDVI_DEFAULT,
        "Height_Ave_cm": float(height),
        "Month": int(month),
        "Season": season,
    }
    for sp in ALL_SPECIES:
        row[sp] = 1 if sp in species else 0
    for t in TARGETS:                       # label cols must exist; values ignored
        row[t] = 0

    input_df = pd.DataFrame([row])
    preds = predict(input_df, PREDICTORS, AVAILABLE_TARGETS)

    # Headline numbers. Green is derived: GDM = green + clover  ->  green = GDM - clover.
    gdm = preds["GDM_g"]
    clover = preds.get("Dry_Clover_g", 0.0)
    green = max(gdm - clover, 0.0)

    results = [
        ["Total green dry matter (GDM)", round(gdm, 1)],
        ["Clover", round(clover, 1)],
        ["Green grass (GDM - clover)", round(green, 1)],
    ]
    # If the dead model is present (local full version), show it on the bottom.
    if "Dry_Dead_g" in preds:
        results.append(["Dead material", round(preds["Dry_Dead_g"], 1)])

    log_prediction(image, state, int(month), season, float(height), species, preds)

    # Append a compact row to the per-session history (a NEW list each time).
    new_row = [
        Path(image).name, state, int(month), float(height),
        "|".join(species), round(gdm, 1), round(clover, 1), round(green, 1),
    ]
    history = (history or []) + [new_row]

    # Give the display table its own object (a DataFrame), separate
    # from the list stored in gr.State. Returning the same list to both a
    # Dataframe output and a State output lets Gradio's postprocessing mutate
    # the stored state.
    history_df = pd.DataFrame(history, columns=HISTORY_COLS)
    return results, history_df, history


# Build the UI with gr.Blocks (layout + history needs this setup). 
with gr.Blocks(
    theme=gr.themes.Soft(
        # Carlito is a free Calibri-metric-compatible Google font, so it renders
        # the same for every visitor; Calibri/sans-serif are fallbacks.
        font=[gr.themes.GoogleFont("Carlito"), "Calibri", "sans-serif"],
    ),
    title="FloraView",
) as demo:
    gr.Markdown("# FloraView - pasture biomass from a photo")
    gr.Markdown(INSTRUCTIONS)

    # Per-session memory: separate for each visitor and is wiped when they leave.
    history_state = gr.State([])

    with gr.Row():
        # Left column: the input form.
        with gr.Column(scale=1):
            image_in = gr.Image(type="filepath", label="Pasture photo")
            state_in = gr.Dropdown(STATES, label="State", value="Vic")
            month_in = gr.Dropdown(MONTH_CHOICES, label="Month", value=6)
            height_in = gr.Number(label="Average plant height (cm)", value=8)
            species_in = gr.Dropdown(
                COMMON_SPECIES, multiselect=True,
                label="Pasture species present (leave empty to assume ryegrass/clover)",
            )
            submit = gr.Button("Estimate biomass", variant="primary")

        # Right column: results on top, session history below (bottom-right).
        with gr.Column(scale=1):
            results_out = gr.Dataframe(
                headers=["Component", "Predicted (g)"], label="Results",
            )
            history_out = gr.Dataframe(
                headers=HISTORY_COLS,
                label="This session's predictions",
            )

    # pass the form values + history IN, get results + history OUT.
    submit.click(
        fn=run_prediction,
        inputs=[image_in, state_in, month_in, height_in, species_in, history_state],
        outputs=[results_out, history_out, history_state],
    )


if __name__ == "__main__":
    # 0.0.0.0 makes it reachable outside the container; port matches the Dockerfile.
    demo.launch(server_name="0.0.0.0", server_port=7860)
