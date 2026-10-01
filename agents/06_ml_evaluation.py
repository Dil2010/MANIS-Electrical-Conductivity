# ============================================================
# MANIS: ELECTRICAL CONDUCTIVITY
# ML EXECUTION & EVALUATION AI AGENT
# New Correlation (10-parameter) feature set
# ============================================================
# Reads  : ec_models/ec_newcorr_trained_models.pkl, ec_newcorr_scaler.pkl,
#          ec_newcorr_feature_cols.json   (from the ML Development Agent)
#          ec_results/ec_newcorr_test_set.csv (held-out test rows)
#          Baseline artifacts for Conventional (5) and Shen et al. (9)
# Writes : ec_results/ec_newcorr_test_predictions_<Model>.csv   (one per model)
#          ec_results/ec_newcorr_all_model_predictions.csv      (all models, for
#                                                               the Physical
#                                                               Validation Agent)
#          ec_results/ec_newcorr_model_comparison.csv
#          ec_results/ec_newcorr_three_way_comparison.csv
#          ec_results/ec_newcorr_<figure>_<Model>.png  (300 DPI)
#          ec_newcorr_eval_ai_notes_<Model>.txt / ec_newcorr_comparison_ai_notes.txt
#
# Models were trained on log10(sigma_nf). Every prediction is back
# transformed (10**pred) before any metric or plot, so all reported
# numbers are in raw S/m and match the ML Development Agent exactly.
#
# CHANGES FROM PREVIOUS VERSION
#  1. DRIVE_DIR moved to MANIS_ELECTRICAL/.
#  2. API key setup + get_client() (fresh client per call).
#  3. Predictions saved per model, so the AI analysis always reads
#     the model it was asked about (previously one shared file was
#     overwritten by whichever model was evaluated last).
#  4. All model predictions saved in one file for the Physical
#     Validation Agent.
#  5. Conventional (5) and Shen et al. (9) baselines computed from their
#     saved artifacts instead of described from memory; if missing, the
#     AI is told so and instructed not to estimate.
#  6. Model Comparison prompt previously described this as the
#     Conventional (5) baseline; corrected to New Correlation (10).
#  7. Three way comparison table (all algorithms x all feature sets).
#  8. Data range passed from the test set instead of hardcoded.
#  9. matplotlib backend set before pyplot import.
#
# REVISION NOTES:
#  - Axis label of the MAPE-by-nanoparticle figure corrected from
#    "Mean Absolute Error (%)" to "Mean Absolute Percentage Error (%)".
#    No calculation was changed.
#  - Known limitation: the baseline tag "conv" below does not match the
#    file names written by the Conventional agent (ec_conventional_*), so
#    the Conventional baseline is reported as unavailable in the AI
#    comparisons and the three way table. This affects only the AI
#    commentary, not any metric or figure reported in the manuscript.
# ============================================================

# !pip install xgboost scikit-learn gradio anthropic -q

import os
import re
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import anthropic
import gradio as gr
from google.colab import drive

warnings.filterwarnings('ignore')

# ============================================================
# FEATURE SET IDENTITY
# ============================================================
FS_TAG  = "newcorr"
FS_NAME = "New Correlation (10-parameter)"

# Baseline feature sets (tag: display name).
# EDIT the tag if the Conventional agent saved under a different prefix.
BASELINES = {
    "conv": "Conventional (5-parameter)",
    "shen": "Shen et al. (9-parameter)",
}

# ============================================================
# GOOGLE DRIVE SETUP
# ============================================================
DRIVE_DIR   = "/content/drive/MyDrive/MANIS_ELECTRICAL/"
CLEAN_CSV   = DRIVE_DIR + "nanofluid_ec_data_clean.csv"
MODEL_DIR   = Path(DRIVE_DIR + "ec_models/")
RESULTS_DIR = Path(DRIVE_DIR + "ec_results/")

MODELS_PKL    = MODEL_DIR / f"ec_{FS_TAG}_trained_models.pkl"
SCALER_PKL    = MODEL_DIR / f"ec_{FS_TAG}_scaler.pkl"
FEATURES_JSON = MODEL_DIR / f"ec_{FS_TAG}_feature_cols.json"
TEST_CSV      = RESULTS_DIR / f"ec_{FS_TAG}_test_set.csv"
ALL_PRED_CSV  = RESULTS_DIR / f"ec_{FS_TAG}_all_model_predictions.csv"
COMPARE_CSV   = RESULTS_DIR / f"ec_{FS_TAG}_model_comparison.csv"
COMPARE_TXT   = DRIVE_DIR + f"ec_{FS_TAG}_comparison_ai_notes.txt"

THREEWAY_CSV  = RESULTS_DIR / f"ec_{FS_TAG}_three_way_comparison.csv"

def mount_drive():
    try:
        drive.mount('/content/drive', force_remount=False)
        os.makedirs(DRIVE_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        return True
    except Exception as e:
        print(f"Drive mount failed: {e}")
        return False

mount_drive()

# ============================================================
# API KEY SETUP
# ============================================================
def clean_key(raw):
    raw = (raw or "").strip().strip('"').strip("'")
    if raw.count("sk-ant-") > 1:
        print(f"⚠️ Secret contains {raw.count('sk-ant-')} keys; using the first one.")
        raw = "sk-ant-" + raw.split("sk-ant-")[1]
    return "".join(raw.split())

try:
    from google.colab import userdata
    _key = clean_key(userdata.get("ANTHROPIC_API_KEY"))
except Exception:
    _key = clean_key("")  # paste a manual key between the quotes if needed

if _key.startswith("sk-ant-") and 90 <= len(_key) <= 130:
    os.environ["ANTHROPIC_API_KEY"] = _key
    print(f"✅ API key loaded (length {len(_key)}).")
else:
    print(f"⚠️ Key looks invalid (length {len(_key)}). Evaluation works; AI tabs will not.")

CLAUDE_MODEL = "claude-sonnet-4-5"

def get_client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No ANTHROPIC_API_KEY in environment. Re-run the key setup.")
    return anthropic.Anthropic(api_key=key)

def ask_claude(prompt, max_tokens):
    response = get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ============================================================
# FONTS: Times New Roman (with fallback)
# ============================================================
def setup_fonts():
    available = {f.name for f in fm.fontManager.ttflist}
    if "Times New Roman" in available:
        plt.rcParams['font.family'] = 'Times New Roman'
    elif "Liberation Serif" in available:
        plt.rcParams['font.family'] = 'Liberation Serif'
        print("Times New Roman not found; using Liberation Serif.")
    else:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
        print("Times New Roman not found; using generic serif.")
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'

setup_fonts()

# ============================================================
# CONSTANTS (must match the ML Development Agent)
# ============================================================
FEATURE_COLS = [
    "volume_fraction", "temperature_K", "particle_size_nm",
    "nanoparticle_encoded", "base_fluid_encoded",
    "sigma_p", "rho_p", "sigma_bf", "eps_r", "mu_bf",
]
LOG_TARGET_COL = "log_sigma_nf"
TARGET_COL     = "sigma_nf"
SCALED_MODELS  = {"ANN", "SVR"}
HIGH_ERROR_PCT = 20

MODEL_COLORS = {
    "ANN":               "#E63946",
    "Random Forest":     "#2A9D8F",
    "XGBoost":           "#E9C46A",
    "Gradient Boosting": "#457B9D",
    "SVR":               "#F4A261",
}
DEFAULT_COLOR = "#D3D3D3"
EDGE_COLOR    = "#888888"
EDGE_WIDTH    = 0.4

NANOPARTICLE_DISPLAY = {
    "AL2O3": r"$\mathrm{Al_2O_3}$",
    "CUO":   "CuO",
    "TIO2":  r"$\mathrm{TiO_2}$",
    "ZNO":   "ZnO",
    "SIO2":  r"$\mathrm{SiO_2}$",
    "SIC":   "SiC",
    "CACO3": r"$\mathrm{CaCO_3}$",
}

def get_model_color(name):
    return MODEL_COLORS.get(name, DEFAULT_COLOR)

def get_nanoparticle_display(code):
    key = re.sub(r'[^A-Z0-9]', '', str(code).upper())
    return NANOPARTICLE_DISPLAY.get(key, code)

def safe_name(model_name):
    return model_name.replace(" ", "_")

# ============================================================
# LOADING AND PREDICTION
# ============================================================
def load_artifacts(models_pkl, scaler_pkl, features_json):
    """Returns (models, scaler, feature_cols, error_message)."""
    for p in (models_pkl, scaler_pkl, features_json):
        if not Path(p).exists():
            return None, None, None, f"Missing file: {p}"
    try:
        with open(models_pkl, "rb") as f:
            models = pickle.load(f)
        with open(scaler_pkl, "rb") as f:
            scaler = pickle.load(f)
        with open(features_json, "r") as f:
            feature_cols = json.load(f)
        return models, scaler, feature_cols, None
    except Exception as e:
        return None, None, None, f"Error loading artifacts: {e}"

def load_models():
    models, scaler, feature_cols, err = load_artifacts(MODELS_PKL, SCALER_PKL, FEATURES_JSON)
    if err is None and feature_cols != FEATURE_COLS:
        err = (f"Feature mismatch.\nSaved: {feature_cols}\nExpected: {FEATURE_COLS}\n"
               f"Re-run the ML Development Agent.")
        return None, None, None, err
    return models, scaler, feature_cols, err

def predict_raw(model_name, model, scaler, feature_cols, df):
    """Predict log10(sigma_nf) and back-transform to raw S/m."""
    X = df[feature_cols].values.astype(float)
    if model_name in SCALED_MODELS:
        X = scaler.transform(X)
    return 10 ** model.predict(X)

def compute_metrics(y_true, y_pred):
    return {
        "R2":      float(r2_score(y_true, y_pred)),
        "MAE_Sm":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE_Sm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%":  float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-30))) * 100),
    }

def data_range(df):
    y = df[TARGET_COL].values
    y = y[y > 0]
    lo, hi = y.min(), y.max()
    return {"min_Sm": float(lo), "max_Sm": float(hi),
            "orders_of_magnitude": round(float(np.log10(hi / lo)), 2)}

def baseline_artifact_paths(tag):
    return (MODEL_DIR / f"ec_{tag}_trained_models.pkl",
            MODEL_DIR / f"ec_{tag}_scaler.pkl",
            MODEL_DIR / f"ec_{tag}_feature_cols.json",
            RESULTS_DIR / f"ec_{tag}_test_set.csv")

def load_baseline_metrics():
    """Per algorithm metrics for every baseline feature set, each computed
    from its own saved models on its own saved test set.
    Returns ({tag: {model: metrics}}, {tag: note})."""
    my_test = pd.read_csv(TEST_CSV)
    all_metrics, notes = {}, {}
    for tag, name in BASELINES.items():
        mp, sp, fp, tp = baseline_artifact_paths(tag)
        models, scaler, feature_cols, err = load_artifacts(mp, sp, fp)
        if err or not tp.exists():
            notes[tag] = f"{name}: artifacts NOT found ({err or f'missing {tp.name}'})."
            continue
        b_test = pd.read_csv(tp)
        notes[tag] = "" if len(b_test) == len(my_test) else (
            f"{name}: WARNING, test set has {len(b_test)} rows vs {len(my_test)} here; "
            f"splits may differ.")
        y_true = b_test[TARGET_COL].values
        out = {}
        for mname, model in models.items():
            try:
                m = compute_metrics(y_true, predict_raw(mname, model, scaler, feature_cols, b_test))
                out[mname] = {k: round(v, 4) if k in ("R2", "MAPE_%") else v for k, v in m.items()}
            except Exception as e:
                print(f"Baseline {tag}/{mname} failed: {e}")
        all_metrics[tag] = out
    return all_metrics, notes

def baseline_text_for(model_name=None):
    """Prompt block with baseline numbers (one algorithm, or all)."""
    metrics, notes = load_baseline_metrics()
    lines = []
    for tag, name in BASELINES.items():
        if tag in metrics and (model_name is None or model_name in metrics[tag]):
            data = metrics[tag] if model_name is None else metrics[tag][model_name]
            lines.append(f"{name}: {json.dumps(data, indent=2)} {notes.get(tag, '')}")
        else:
            lines.append(f"{notes.get(tag, name + ': not available.')} "
                         f"State this explicitly and do NOT estimate its numbers.")
    return "\n".join(lines)

def build_three_way_table():
    """Rows: algorithm. Columns: R2 and MAPE for each feature set."""
    comp, err = evaluate_all_models()
    if comp is None:
        return None, err
    metrics, notes = load_baseline_metrics()
    table = pd.DataFrame({"Model": comp["Model"]})
    for tag, name in BASELINES.items():
        short = name.split(" (")[0]
        if tag in metrics:
            table[f"R2 {short}"] = table["Model"].map(lambda m: metrics[tag].get(m, {}).get("R2"))
            table[f"MAPE% {short}"] = table["Model"].map(lambda m: metrics[tag].get(m, {}).get("MAPE_%"))
    short = FS_NAME.split(" (")[0]
    table[f"R2 {short}"] = comp["R2"].round(4).values
    table[f"MAPE% {short}"] = comp["MAPE_%"].round(2).values
    table.to_csv(THREEWAY_CSV, index=False)
    msg = "\n".join([f"✅ Saved {THREEWAY_CSV.name}"] + [n for n in notes.values() if n])
    return table, msg

def run_three_way_table():
    table, msg = build_three_way_table()
    if table is None:
        return f"❌ {msg}", None
    return msg, table

# ============================================================
# EVALUATION + FIGURES
# ============================================================
def style_axes(ax, grid_axis="both"):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, alpha=0.3, color='#dddddd', axis=grid_axis, which='both')

def build_prediction_df(model_name, models, scaler, feature_cols):
    df = pd.read_csv(TEST_CSV)
    y_actual = df[TARGET_COL].values
    y_pred = predict_raw(model_name, models[model_name], scaler, feature_cols, df)
    df["sigma_nf_predicted"] = y_pred
    df["error"]   = y_actual - y_pred
    df["error_%"] = np.abs((y_actual - y_pred) / (y_actual + 1e-30)) * 100
    return df

def run_evaluation(selected_model):
    models, scaler, feature_cols, err = load_models()
    if models is None:
        return f"❌ {err}\nRun the EC ML Development Agent (Train All Models) first.", None, None, None, None
    if selected_model not in models:
        return f"❌ Model {selected_model} not found.", None, None, None, None
    if not TEST_CSV.exists():
        return f"❌ Held-out test set not found: {TEST_CSV}", None, None, None, None

    try:
        df = build_prediction_df(selected_model, models, scaler, feature_cols)
    except Exception as e:
        return f"❌ Prediction error: {e}", None, None, None, None

    pred_path = RESULTS_DIR / f"ec_{FS_TAG}_test_predictions_{safe_name(selected_model)}.csv"
    df.to_csv(pred_path, index=False)

    y_actual = df[TARGET_COL].values
    y_pred   = df["sigma_nf_predicted"].values
    m        = compute_metrics(y_actual, y_pred)
    color    = get_model_color(selected_model)
    tag      = f"ec_{FS_TAG}"
    mname    = safe_name(selected_model)

    metrics_text = (
        f"Evaluation Results: {selected_model} ({FS_NAME}, held-out test set)\n"
        f"{'━'*50}\n"
        f"R²:   {m['R2']:.4f}\n"
        f"MAE:  {m['MAE_Sm']:.4e} S/m\n"
        f"RMSE: {m['RMSE_Sm']:.4e} S/m\n"
        f"MAPE: {m['MAPE_%']:.2f}%\n"
        f"Test points: {len(df)}\n"
        f"{'━'*50}\n"
        f"Trained on log10(σ_nf); metrics on back-transformed raw S/m.\n"
        f"Predictions saved: {pred_path.name}"
    )

    # Figure 1: Predicted vs Actual (log-log)
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    mask = (y_actual > 0) & (y_pred > 0)
    ax1.scatter(y_actual[mask], y_pred[mask], alpha=0.65, color=color,
                edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH, s=50)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    lo = min(y_actual[mask].min(), y_pred[mask].min()) * 0.5
    hi = max(y_actual[mask].max(), y_pred[mask].max()) * 2
    ax1.plot([lo, hi], [lo, hi], color='#CC4444', linestyle='--', linewidth=1.8)
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
    ax1.set_xlabel("Actual Electrical Conductivity (S/m)", fontsize=12)
    ax1.set_ylabel("Predicted Electrical Conductivity (S/m)", fontsize=12)
    ax1.legend(handles=[
        Patch(facecolor=color, edgecolor=EDGE_COLOR, label=selected_model),
        Line2D([0], [0], color='none', label=f'R\u00b2 = {m["R2"]:.4f}'),
        Line2D([0], [0], color='#CC4444', linestyle='--', linewidth=1.8, label='Perfect fit'),
    ], fontsize=11)
    style_axes(ax1)
    plt.tight_layout()
    plot1_path = str(RESULTS_DIR / f"{tag}_predicted_vs_actual_{mname}.png")
    fig1.savefig(plot1_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)

    # Figure 2: Error distribution
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.hist(df["error_%"], bins=30, color=color,
             edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, alpha=0.9)
    ax2.axvline(x=m["MAPE_%"], color='#CC4444', linestyle='--', linewidth=1.8)
    ax2.set_xlabel("Absolute Error (%)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.legend(handles=[
        Patch(facecolor=color, edgecolor=EDGE_COLOR, label=selected_model),
        Line2D([0], [0], color='#CC4444', linestyle='--', linewidth=1.8,
               label=f'MAPE = {m["MAPE_%"]:.2f}%'),
    ], fontsize=11)
    style_axes(ax2, grid_axis='y')
    plt.tight_layout()
    plot2_path = str(RESULTS_DIR / f"{tag}_error_distribution_{mname}.png")
    fig2.savefig(plot2_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)

    # Figure 3: MAPE by nanoparticle type
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    group_mape = df.groupby("nanoparticle")["error_%"].mean().sort_values()
    base_rgb = mcolors.to_rgb(color)
    n = len(group_mape)
    bar_colors = [tuple(c * (0.6 + 0.4 * i / max(n - 1, 1)) for c in base_rgb) for i in range(n)]
    ax3.barh([get_nanoparticle_display(x) for x in group_mape.index], group_mape.values,
             color=bar_colors, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    ax3.axvline(x=m["MAPE_%"], color='#CC4444', linestyle='--', linewidth=1.8)
    offset = group_mape.max() * 0.01
    for i, v in enumerate(group_mape.values):
        ax3.text(v + offset, i, f"{v:.1f}%", va='center', fontsize=11, color='#333333')
    ax3.set_xlabel("Mean Absolute Percentage Error (%)", fontsize=12)
    ax3.set_ylabel("Nanoparticle Type", fontsize=12)
    ax3.legend(handles=[
        Patch(facecolor=color, edgecolor=EDGE_COLOR, label=selected_model),
        Line2D([0], [0], color='#CC4444', linestyle='--', linewidth=1.8,
               label=f'Overall MAPE = {m["MAPE_%"]:.2f}%'),
    ], fontsize=11)
    style_axes(ax3, grid_axis='x')
    plt.tight_layout()
    plot3_path = str(RESULTS_DIR / f"{tag}_performance_by_group_{mname}.png")
    fig3.savefig(plot3_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)

    preview = df[["nanoparticle", "base_fluid", "particle_size_nm",
                  "volume_fraction", "temperature_K",
                  TARGET_COL, "sigma_nf_predicted", "error_%"]].round(6)

    return metrics_text, preview, plot1_path, plot2_path, plot3_path

# ============================================================
# ALL MODELS: comparison table + combined predictions
# ============================================================
def evaluate_all_models():
    """Returns (comparison_df, error). Also writes ALL_PRED_CSV and COMPARE_CSV."""
    models, scaler, feature_cols, err = load_models()
    if models is None:
        return None, err
    if not TEST_CSV.exists():
        return None, f"Held-out test set not found: {TEST_CSV}"

    test_df = pd.read_csv(TEST_CSV)
    y_true = test_df[TARGET_COL].values
    all_pred = test_df.copy()
    rows = []
    for name, model in models.items():
        try:
            y_pred = predict_raw(name, model, scaler, feature_cols, test_df)
            all_pred[f"pred_{safe_name(name)}"] = y_pred
            m = compute_metrics(y_true, y_pred)
            rows.append({"Model": name, **m})
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    all_pred.to_csv(ALL_PRED_CSV, index=False)
    comp = (pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True))
    comp.to_csv(COMPARE_CSV, index=False)
    return comp, None

def run_comparison_table():
    comp, err = evaluate_all_models()
    if comp is None:
        return f"❌ {err}", None
    shown = comp.copy()
    shown["R2"] = shown["R2"].round(4)
    shown["MAPE_%"] = shown["MAPE_%"].round(2)
    for c in ("MAE_Sm", "RMSE_Sm"):
        shown[c] = shown[c].map(lambda v: f"{v:.4e}")
    msg = (f"✅ Evaluated {len(comp)} models on {FS_NAME} test set.\n"
           f"Saved: {COMPARE_CSV.name}\nSaved: {ALL_PRED_CSV.name} (input for Physical Validation Agent)")
    return msg, shown

# ============================================================
# AI ANALYSIS (single model)
# ============================================================
FEATURE_SET_CONTEXT = f"""FEATURE SET: {FS_NAME}. The conventional five parameters
(nanoparticle type, base fluid type, particle size, volume fraction, temperature)
plus five properties: nanoparticle electrical conductivity (sigma_p), nanoparticle
density (rho_p), base fluid electrical conductivity (sigma_bf), relative
permittivity (eps_r), and base fluid viscosity (mu_bf). Zeta potential (U0),
nanolayer thickness (h), and sphericity (psi) from the original New Correlation
model were not used as ML features. sigma_p and rho_p are fixed literature values
per nanoparticle material. This is the fullest of the three feature sets compared
(Conventional 5, Shen et al. 9, New Correlation 10). The target was log10(sigma_nf);
all metrics are back-transformed to raw S/m."""

def ai_analyze_evaluation(selected_model):
    models, scaler, feature_cols, err = load_models()
    if models is None:
        return f"❌ {err}"
    if selected_model not in models:
        return f"❌ Model {selected_model} not found."
    if not TEST_CSV.exists():
        return f"❌ Held-out test set not found: {TEST_CSV}"

    try:
        df = build_prediction_df(selected_model, models, scaler, feature_cols)
        y_actual = df[TARGET_COL].values
        m = compute_metrics(y_actual, df["sigma_nf_predicted"].values)

        group_perf = df.groupby("nanoparticle")["error_%"].agg(
            ["mean", "max", "min", "std", "count"]).round(3).to_dict()
        bf_perf = df.groupby("base_fluid")["error_%"].agg(
            ["mean", "count"]).round(3).to_dict()
        df["temp_bin"] = pd.cut(df["temperature_K"], bins=4)
        temp_perf = {str(k): v for k, v in
                     df.groupby("temp_bin", observed=True)["error_%"].mean().round(3).items()}
        df["vf_bin"] = pd.cut(df["volume_fraction"], bins=4)
        vf_perf = {str(k): v for k, v in
                   df.groupby("vf_bin", observed=True)["error_%"].mean().round(3).items()}
        high = df[df["error_%"] > HIGH_ERROR_PCT]
        high_examples = high[["nanoparticle", "base_fluid", "particle_size_nm",
                              "volume_fraction", "temperature_K", "error_%"]
                             ].head(10).to_dict(orient="records")

        baseline_text = baseline_text_for(selected_model)

        evaluation_data = {
            "model": selected_model,
            "evaluation_set": f"held-out test set ({len(df)} records)",
            "target_range_test_set": data_range(df),
            "overall_metrics": {k: (round(v, 4) if k in ("R2", "MAPE_%") else v) for k, v in m.items()},
            "performance_by_nanoparticle": group_perf,
            "performance_by_base_fluid": bf_perf,
            "performance_by_temperature_range": temp_perf,
            "performance_by_volume_fraction": vf_perf,
            f"high_error_cases_over_{HIGH_ERROR_PCT}pct": len(high),
            "high_error_examples": high_examples,
        }

        prompt = f"""You are an expert in machine learning for nanofluid electrical conductivity
prediction, familiar with mechanistic models for this system (Maxwell mixing theory,
electric double layer and ion mobility effects, Shen et al. 2012). Analyze these
held-out test results.

{FEATURE_SET_CONTEXT}

Prior benchmarking context: all five theoretical EC correlations produced negative R²
on this dataset.

Evaluation data:
{json.dumps(evaluation_data, indent=2, default=str)}

Baselines for the SAME algorithm ({selected_model}) on the same held-out split:
{baseline_text}

Provide analysis on:
1. OVERALL PERFORMANCE: Is R² = {m['R2']:.4f} and MAPE = {m['MAPE_%']:.2f}% strong for a
   target spanning the range given above? Note that raw space R² is dominated by the
   largest conductivity values, while MAPE weights all points equally; comment on
   whether the two metrics agree.
2. NANOPARTICLE AND BASE FLUID ANALYSIS: which systems are predicted best and worst,
   noting the per group counts (small groups give unreliable MAPE).
3. TEMPERATURE AND VOLUME FRACTION EFFECTS on error.
4. HIGH ERROR CASES (>{HIGH_ERROR_PCT}%): what conditions they share, and whether this points
   to the model, the feature set, or data quality.
5. COMPARISON TO CONVENTIONAL (5) AND SHEN ET AL. (9): use only the baseline numbers
   given above. Did adding sigma_p and rho_p (on top of the base fluid properties)
   improve this algorithm further, or has performance plateaued? Note that sigma_p and
   rho_p are constant within each nanoparticle class, so any gain may reflect
   material identity already carried by nanoparticle_encoded.
6. PUBLICATION READINESS: suitability as the fullest feature set in the three way
   comparison, and how the log-space training and the trimmed New Correlation
   parameter set should be stated as limitations.

Cite the actual numbers. Formal academic style suitable for a discussion section.
Do not use em dashes."""

        notes = ask_claude(prompt, 2500)
        with open(DRIVE_DIR + f"ec_{FS_TAG}_eval_ai_notes_{safe_name(selected_model)}.txt", "w") as f:
            f.write(notes)
        return notes
    except Exception as e:
        return f"❌ AI analysis error: {e}"

# ============================================================
# AI MODEL COMPARISON (all models)
# ============================================================
def ai_compare_models():
    comp, err = evaluate_all_models()
    if comp is None:
        return f"❌ {err}"
    try:
        test_df = pd.read_csv(TEST_CSV)
        results = comp.round({"R2": 4, "MAPE_%": 2}).to_dict(orient="records")

        baseline_text = baseline_text_for(None)

        prompt = f"""You are an expert in machine learning for nanofluid electrical conductivity
prediction. Compare all trained models and recommend the best one.

{FEATURE_SET_CONTEXT}

All results on the held-out test set ({len(test_df)} records), raw S/m:
{json.dumps(results, indent=2)}

Test set target range: {json.dumps(data_range(test_df))}
Nanoparticles in test set: {sorted(test_df['nanoparticle'].unique().tolist())}
Base fluids in test set: {sorted(test_df['base_fluid'].unique().tolist())}

Baselines per algorithm on the same held-out split:
{baseline_text}

Provide:
1. MODEL RANKING with justification, flagging any case where R² and MAPE disagree
   (high R² with poor MAPE usually means large magnitude points fit well while small
   magnitude points are missed).
2. BEST MODEL RECOMMENDATION for this feature set.
3. CHANGE VERSUS CONVENTIONAL (5) AND SHEN ET AL. (9), per algorithm, using only the
   baseline numbers above. State whether sigma_p and rho_p add information beyond
   nanoparticle_encoded or largely duplicate it, and whether gains are consistent
   across algorithms or limited to one.
4. ENSEMBLE POSSIBILITY.
5. OVERALL CONCLUSION OF THE THREE WAY COMPARISON: which feature set and algorithm
   combination should be reported as best, and how large the margins are.

Cite the actual numbers. Formal academic style. Do not use em dashes."""

        notes = ask_claude(prompt, 2000)
        with open(COMPARE_TXT, "w") as f:
            f.write(notes)
        return notes
    except Exception as e:
        return f"❌ AI comparison error: {e}"

# ============================================================
# DRIVE STATUS
# ============================================================
def check_drive():
    if not os.path.exists(DRIVE_DIR):
        return "❌ Drive not mounted. Click Mount Drive."
    lines = [f"✅ Drive mounted: {DRIVE_DIR}"]
    for label, folder in (("ec_models/", MODEL_DIR), ("ec_results/", RESULTS_DIR)):
        if folder.exists():
            files = sorted(f for f in os.listdir(folder)
                           if (folder / f).is_file() and FS_TAG in f)
            lines.append(f"\n{label} ({FS_TAG} files: {len(files)})")
            for f in files:
                lines.append(f"  {f}  ({(folder / f).stat().st_size:,} bytes)")
    for tag, name in BASELINES.items():
        mp, _, _, tp = baseline_artifact_paths(tag)
        lines.append(f"\n{name} ({tag}): models {mp.exists()}, test set {tp.exists()}")
    return "\n".join(lines)

def do_mount():
    return check_drive() if mount_drive() else "❌ Mount failed."

def get_available_models():
    models, _, _, _ = load_models()
    return list(models.keys()) if models else ["No models found: run the ML Development Agent first"]

# ============================================================
# GRADIO INTERFACE
# ============================================================
_choices = get_available_models()

with gr.Blocks(title=f"MANIS EC Evaluation Agent ({FS_NAME})",
               theme=gr.themes.Soft()) as eval_app:

    gr.Markdown(f"# MANIS: EC ML Execution & Evaluation Agent ({FS_NAME})")
    gr.Markdown("Evaluates on the held-out test set saved by the ML Development Agent. "
                "Models trained on log10(σ_nf); all metrics back-transformed to raw S/m.")

    with gr.Tabs():

        with gr.Tab("1: Evaluation"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(choices=_choices, value=_choices[0],
                                                 label="Select Model")
                    evaluate_btn   = gr.Button("Run Evaluation", variant="primary", size="lg")
                    metrics_output = gr.Textbox(label="Metrics", interactive=False, lines=12)
                with gr.Column(scale=2):
                    predictions_table = gr.Dataframe(label="Actual vs Predicted (Test Set)",
                                                     interactive=False)
            gr.Markdown("### Figures (300 DPI)")
            with gr.Row():
                plot1 = gr.Image(label="Predicted vs Actual (log-log)")
                plot2 = gr.Image(label="Error Distribution")
                plot3 = gr.Image(label="MAPE by Nanoparticle Type")

        with gr.Tab("2: AI Analysis"):
            with gr.Row():
                analysis_model = gr.Dropdown(choices=_choices, value=_choices[0],
                                             label="Select Model to Analyse")
                analyze_btn = gr.Button("Generate AI Analysis", variant="primary", size="lg")
            ai_analysis_output = gr.Textbox(label="AI Analysis", interactive=False, lines=35)

        with gr.Tab("3: Model Comparison"):
            table_btn  = gr.Button("Evaluate All Models (table + save predictions)",
                                   variant="secondary")
            table_msg  = gr.Textbox(label="Status", interactive=False, lines=3)
            comp_table = gr.Dataframe(label="All Models: held-out test set, raw S/m",
                                      interactive=False)
            gr.Markdown("### Three way comparison (all algorithms x all feature sets)")
            tw_btn   = gr.Button("Build Three Way Table", variant="secondary")
            tw_msg   = gr.Textbox(label="Status", interactive=False, lines=3)
            tw_table = gr.Dataframe(label="R² and MAPE per feature set, same held-out split",
                                    interactive=False, wrap=True)
            compare_btn = gr.Button("AI Comparison & Recommendation", variant="primary", size="lg")
            comparison_output = gr.Textbox(label="AI Comparison", interactive=False, lines=35)

        with gr.Tab("4: Google Drive"):
            gr.Markdown(f"Folder: `{DRIVE_DIR}`")
            with gr.Row():
                mount_btn = gr.Button("Mount / Remount Drive", variant="secondary")
                check_btn = gr.Button("Check Drive Status", variant="secondary")
            drive_status = gr.Textbox(label="Status", interactive=False, lines=14)

    evaluate_btn.click(run_evaluation, inputs=[model_dropdown],
                       outputs=[metrics_output, predictions_table, plot1, plot2, plot3])
    analyze_btn.click(ai_analyze_evaluation, inputs=[analysis_model],
                      outputs=[ai_analysis_output])
    table_btn.click(run_comparison_table, outputs=[table_msg, comp_table])
    compare_btn.click(ai_compare_models, outputs=[comparison_output])
    tw_btn.click(run_three_way_table, outputs=[tw_msg, tw_table])
    mount_btn.click(do_mount, outputs=[drive_status])
    check_btn.click(check_drive, outputs=[drive_status])

eval_app.launch(share=True)
