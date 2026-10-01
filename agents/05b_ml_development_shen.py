# ############################################################
# CELL A — GUARD (its own cell, directly ABOVE Cell B)
# Checks that the features file content is identical to the
# verified copy in MANIS_REVISION/00_inputs_v3.
# ############################################################
from google.colab import drive
import pandas as pd, numpy as np
drive.mount('/content/drive')

F = "/content/drive/MyDrive/MANIS_ELECTRICAL/nanofluid_ec_features.xlsx"
V = "/content/drive/MyDrive/MANIS_REVISION/00_inputs_v3/nanofluid_ec_features.xlsx"
read = lambda p: pd.read_excel(p, sheet_name="EC Features", header=1, skiprows=[2]).dropna(how="all")

cur, ver = read(F), read(V)
assert cur.shape == ver.shape, f"Shape differs: current {cur.shape} vs verified {ver.shape}. STOP."
assert list(cur.columns) == list(ver.columns), "Columns differ from the verified copy. STOP."

bad = []
for c in cur.columns:
    a, b = cur[c].values, ver[c].values
    if pd.api.types.is_numeric_dtype(cur[c]):
        if not np.allclose(a.astype(float), b.astype(float), rtol=1e-9, atol=0, equal_nan=True):
            bad.append(c)
    elif not (pd.Series(a).astype(str).values == pd.Series(b).astype(str).values).all():
        bad.append(c)

assert not bad, f"Content differs from the verified copy in columns: {bad}. STOP."
print(f"✅ Features content identical to the verified v3 copy ({len(cur)} rows). Safe to train.")


# ############################################################
# CELL B — SHEN ML DEVELOPMENT AGENT
# ############################################################
# ============================================================
# MANIS — ELECTRICAL CONDUCTIVITY
# COLAB 4: ML DEVELOPMENT AI AGENT  (Shen et al. 9-parameter feature set)
# ============================================================
# Reads  : nanofluid_ec_features.xlsx  (Feature Engineering Agent output)
# Models : ANN, Random Forest, XGBoost, Gradient Boosting, SVR
# Split  : Stratified by group_id (80/20, random_state=42)
# Target : log_sigma_nf (log10 of electrical conductivity, S/m) —
#          dataset spans ~10 orders of magnitude, so training happens
#          in log-space; MAPE is additionally reported in raw S/m space
#          (back-transformed) since log-space MAPE has no clean physical
#          meaning (log values can be negative/near-zero).
# Output : Figures saved to Google Drive
#          + trained_models.pkl / scaler.pkl / feature_cols.json
#          + results/ec_shen_test_set.csv  (held-out test rows)
#          + ec_pretraining_ai_notes.txt / ec_posttraining_ai_notes.txt
#
# REVISION CHANGE (2026-09-30):
#   Metrics table now saved to ec_results/ec_shen_dev_metrics.csv
#   (previously only displayed, never saved).
#
# SCOPE: Shen et al. (9-parameter) feature set —
#   particle_size_nm, volume_fraction, temperature_K,
#   nanoparticle_encoded, base_fluid_encoded,
#   sigma_bf, eps_r, rho_bf, upsilon
# Reuses the EXACT SAME train/test split indices as Colab 3
# (Conventional 5), since load_data() derives the split
# deterministically from the same group_id order and random_state
# in the same source Excel file — only FEATURE_COLS differs, so
# all three feature sets (Colab 3/4/5) are compared fairly on
# identical held-out rows.
#
# AI LAYER (three functional modules, mirrors the TC Colab 3 structure):
#   1. Pre-training module  — reviews dataset characteristics and
#      training considerations BEFORE any model is fit.
#   2. Training module      — trains the 5 algorithms.
#   3. Post-training module — structured scientific commentary on
#      performance, contextualised against nanofluid electrical
#      conductivity physics.
#
# FIX (this version): the Anthropic client is now created FRESH inside
# get_client(), called at the moment of each API request, rather than
# once at script startup. This prevents a stale/invalid key captured
# during an early failed load (e.g. before Secrets permission was
# granted) from silently persisting for the rest of the session —
# previously, a bad client built once at the top of the script kept
# being reused by every button click even after the underlying key
# was fixed, requiring a full runtime restart to pick up the change.
# ============================================================

# ── CELL 1: INSTALL ─────────────────────────────────────────
# !pip install xgboost scikit-learn openpyxl gradio anthropic -q

# ── CELL 2: IMPORTS ─────────────────────────────────────────

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import anthropic
from pathlib import Path
from google.colab import drive

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import xgboost as xgb
import gradio as gr

warnings.filterwarnings('ignore')

# ── Google Drive ──────────────────────────────────────────────
DRIVE_DIR    = "/content/drive/MyDrive/MANIS_ELECTRICAL/"
FEATURE_XLS  = DRIVE_DIR + "nanofluid_ec_features.xlsx"
FIGURES_DIR  = DRIVE_DIR + "EC_ML_Figures/"
MODEL_DIR    = Path(DRIVE_DIR + "ec_models/")
RESULTS_DIR  = Path(DRIVE_DIR + "ec_results/")
TEST_CSV     = RESULTS_DIR / "ec_shen_test_set.csv"
PRETRAIN_TXT = DRIVE_DIR + "ec_pretraining_ai_notes.txt"
POSTTRAIN_TXT= DRIVE_DIR + "ec_posttraining_ai_notes.txt"

def mount_drive():
    try:
        drive.mount('/content/drive', force_remount=False)
        os.makedirs(DRIVE_DIR,   exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR,   exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        return True
    except Exception as e:
        print(f"Drive mount failed: {e}")
        return False

mount_drive()

# ── API KEY SETUP (fix for AuthenticationError) ─────────────────
# Preferred: store the key once in Colab's Secrets manager (key icon,
# left sidebar), named ANTHROPIC_API_KEY, and toggle "Notebook access" on.
try:
    from google.colab import userdata
    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
    print("✅ API key loaded from Colab Secrets.")
except Exception as e:
    manual_key = ""  # ← paste your key here only if Secrets isn't set up
    if manual_key:
        os.environ["ANTHROPIC_API_KEY"] = manual_key
        print("✅ API key set manually for this session.")
    else:
        print("⚠️  No API key found via Secrets, and no manual key provided.")
        print("   Model training (Tab 1) will still work without it —")
        print("   only the AI commentary tabs need it.")

def get_client():
    """Builds a fresh Anthropic client at call time, reading whatever key
    is CURRENTLY in os.environ. This means re-running just the key-setup
    cell above (after fixing Secrets access, rotating a key, etc.) is
    enough to fix subsequent AI calls — no full runtime restart needed."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "No ANTHROPIC_API_KEY found in the environment. Re-run the "
            "API key setup cell above after confirming Colab Secrets access."
        )
    return anthropic.Anthropic(api_key=key)

# ============================================================
# CONSTANTS — SHEN ET AL. (9) FEATURE SET
# ============================================================

FEATURE_COLS = [
    "volume_fraction",
    "temperature_K",
    "particle_size_nm",
    "nanoparticle_encoded",
    "base_fluid_encoded",
    "sigma_bf",
    "eps_r",
    "rho_bf",
    "upsilon",
]

FEATURE_LABELS = [
    "φ (vol. fraction)",
    "T (K)",
    "d_p (nm)",
    "NP encoded",
    "BF encoded",
    "σ_bf (S/m)",
    "ε_r",
    "ρ_bf (kg/m³)",
    "υ (m²/s)",
]

TARGET_COL   = "log_sigma_nf"     # train in log-space
RAW_TARGET_COL = "sigma_nf"       # for back-transformed MAPE reporting
RANDOM_STATE = 42

MODEL_COLORS = {
    "ANN":                "#E63946",
    "Random Forest":      "#2A9D8F",
    "XGBoost":            "#E9C46A",
    "Gradient Boosting":  "#457B9D",
    "SVR":                "#F4A261",
}

# ============================================================
# DATA LOADING & SPLITTING
# ============================================================

def load_data():
    if not Path(FEATURE_XLS).exists():
        return (None, None, None, None, None, None, None, None, None,
                "❌ nanofluid_ec_features.xlsx not found. Run the Feature Engineering Agent first.")
    try:
        xl     = pd.ExcelFile(FEATURE_XLS)
        sheets = xl.sheet_names
        sheet  = next((s for s in sheets if "Feature" in s and "Set" not in s), sheets[0])
        df     = pd.read_excel(FEATURE_XLS, sheet_name=sheet, header=1, skiprows=[2])

        print(f"   Sheet: '{sheet}'  |  Raw columns: {list(df.columns)}")

        COL_MAP = {
            "Group\nID":                 "group_id",
            "Sub\nGroup ID":             "subgroup_id",
            "Nanoparticle":              "nanoparticle",
            "Base Fluid":                "base_fluid",
            "d_p":                       "particle_size_nm",
            "φ":                         "volume_fraction",
            "T":                         "temperature_K",
            "σ_bf":                      "sigma_bf",
            "ε_r":                       "eps_r",
            "ρ_bf":                      "rho_bf",
            "υ":                         "upsilon",
            "σ_p":                       "sigma_p",
            "ρ_p":                       "rho_p",
            "μ_bf":                      "mu_bf",
            "µ_bf":                      "mu_bf",
            "NP\nEncoded":               "nanoparticle_encoded",
            "BF\nEncoded":               "base_fluid_encoded",
            "σ_nf  ← TARGET":            "sigma_nf",
            "log10(σ_nf)  ← TARGET":     "log_sigma_nf",
        }
        df = df.rename(columns=COL_MAP)
        print(f"   Mapped columns: {list(df.columns)}")

        if "nanoparticle_encoded" not in df.columns:
            print("   ⚠️  nanoparticle_encoded missing — computing now")
            le = LabelEncoder()
            df["nanoparticle_encoded"] = le.fit_transform(df["nanoparticle"].astype(str))
        if "base_fluid_encoded" not in df.columns:
            print("   ⚠️  base_fluid_encoded missing — computing now")
            le = LabelEncoder()
            df["base_fluid_encoded"] = le.fit_transform(df["base_fluid"].astype(str))

        required = FEATURE_COLS + [TARGET_COL, RAW_TARGET_COL, "group_id"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return (None, None, None, None, None, None, None, None, None,
                    f"❌ Missing columns in Excel:\n{missing}\n\n"
                    f"Available columns:\n{list(df.columns)}\n\n"
                    f"Re-run the Feature Engineering Agent to regenerate the Excel.")

        df = df.dropna(subset=required).reset_index(drop=True)

        X       = df[FEATURE_COLS].values.astype(float)
        y_log   = df[TARGET_COL].values.astype(float)
        y_raw   = df[RAW_TARGET_COL].values.astype(float)

        # Split by group_id, 80/20 — SPLIT ONCE, reused for every feature
        # set benchmarked (Conventional now; Shen et al./New Correlation
        # later use these exact same train_idx/test_idx).
        train_idx, test_idx = [], []
        for gid in df["group_id"].unique():
            idx = df[df["group_id"] == gid].index.tolist()
            if len(idx) < 3:
                train_idx.extend(idx)
                continue
            n_test = max(1, int(len(idx) * 0.20))
            np.random.seed(RANDOM_STATE)
            np.random.shuffle(idx)
            test_idx.extend(idx[:n_test])
            train_idx.extend(idx[n_test:])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_log[train_idx], y_log[test_idx]
        y_train_raw, y_test_raw = y_raw[train_idx], y_raw[test_idx]

        test_df = df.loc[test_idx].reset_index(drop=True)

        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        msg = (f"✅ Data loaded from sheet: '{sheet}'\n"
               f"   Rows: {len(df)}  |  Groups: {df['group_id'].nunique()}\n"
               f"   Train: {len(X_train)}  |  Test: {len(X_test)}\n"
               f"   Features: {len(FEATURE_COLS)} (Shen et al. set)  |  Target: {TARGET_COL} (log-space)")

        return (X_train, X_test, y_train, y_test,
                X_train_sc, X_test_sc, scaler, df, test_df, msg,
                y_train_raw, y_test_raw)

    except Exception as e:
        return (None, None, None, None, None, None, None, None, None,
                f"❌ Error reading Excel: {str(e)}", None, None)

# ============================================================
# MODULE 1 — PRE-TRAINING AI RECOMMENDATION
# ============================================================

def ai_pretraining_analysis(df, X_train, X_test, y_train, y_test):
    group_sizes = df.groupby("group_id").size()
    np_counts   = df["nanoparticle"].value_counts().to_dict()
    bf_counts   = df["base_fluid"].value_counts().to_dict()

    feature_stats = {}
    for col, label in zip(FEATURE_COLS, FEATURE_LABELS):
        feature_stats[label] = {
            "min":      float(np.nanmin(df[col])),
            "max":      float(np.nanmax(df[col])),
            "mean":     float(np.nanmean(df[col])),
            "variance": float(np.nanvar(df[col])),
        }

    summary = {
        "total_records":       len(df),
        "total_groups":        int(df["group_id"].nunique()),
        "train_size":          len(X_train),
        "test_size":           len(X_test),
        "smallest_group_size": int(group_sizes.min()),
        "largest_group_size":  int(group_sizes.max()),
        "nanoparticle_counts": np_counts,
        "base_fluid_counts":   bf_counts,
        "target_range_log10":  f"{y_train.min():.2f} to {y_train.max():.2f}  (log10 S/m)",
        "target_range_raw":    f"~10^{y_train.min():.1f} to 10^{y_train.max():.1f} S/m",
        "feature_stats":       feature_stats,
    }

    prompt = f"""You are an expert in machine learning model development for nanofluid
electrical conductivity prediction.

Before training begins, review this dataset summary and provide recommendations.
This is the SHEN ET AL. (9-parameter) physics-informed feature set — the
conventional 5 parameters (nanoparticle type, base fluid type, particle
size, volume fraction, temperature) plus 4 physics-derived base-fluid
properties: base fluid electrical conductivity (sigma_bf), relative
permittivity (eps_r), density (rho_bf), and kinematic viscosity (upsilon).
Note: this is NOT the full original Shen et al. (2012) parameter set —
Zeta potential (U0), viscosity decay index (lambda), and reference
temperature (T0) were dropped, since they were fixed constants specific
to one ZnO-in-oil experiment with zero variance across this dataset,
unusable as ML training columns (see prior benchmarking discussion).

The target is log10(sigma_nf) — the raw electrical conductivity spans roughly
10 orders of magnitude across this dataset, so training happens in log-space.

Dataset summary:
{json.dumps(summary, indent=2)}

Planned models: Random Forest, Gradient Boosting, XGBoost, Support Vector
Regression, and an Artificial Neural Network (MLP), trained on these 9
features, with the SAME 80/20 group-aware train/test split used for the
Conventional (5) feature set in Colab 3, so results are directly comparable.

Provide concise commentary (3-4 sentences each):

1. DATASET SUITABILITY
   Is this dataset adequate in size and diversity for training these five
   algorithms on just 5 conventional features? Flag any group with too few
   records, or any nanoparticle/base fluid category that is under-represented.

2. FEATURE SCALE CONSIDERATIONS
   Given the very different scales/ranges across all 9 features (e.g. sigma_bf
   in S/m spans several orders of magnitude on its own, vs. temperature in K,
   vs. encoded categorical integers), confirm whether standard scaling is
   appropriate for ANN and SVR, and flag any feature likely to dominate
   distance-based models if left unscaled.

3. EXPECTED CHALLENGES
   With sigma_bf, eps_r, rho_bf, and upsilon now added, note that sigma_bf is
   largely determined by which base fluid was used (i.e. may correlate
   strongly with base_fluid_encoded) — flag the risk of redundant/collinear
   features rather than genuinely new information. Also note that
   nanoparticle_encoded and base_fluid_encoded remain arbitrary integer
   labels — flag the risk of tree-based models finding spurious ordinal
   relationships in these encodings.

4. RECOMMENDED FOCUS
   Which algorithm(s) are likely best suited to this 9-parameter feature set,
   and why? Note any expectation for how this should perform relative to the
   Conventional (5) feature set already benchmarked in Colab 3.

Be specific, use the actual numbers, and keep it concise."""

    response = get_client().messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models():
    return {
        "ANN": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
            verbose=False,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "SVR": SVR(
            kernel="rbf",
            C=100,
            gamma="scale",
            epsilon=0.001,
        ),
    }

# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true_log, y_pred_log, y_true_raw):
    """Training happens in log-space (necessary given the ~10 orders of
    magnitude spanned by sigma_nf), but R², MAE, RMSE, and MAPE are all
    reported here in back-transformed RAW S/m space, so this table's
    columns match the mechanistic model benchmarking table exactly and
    the two can be compared directly."""
    y_pred_raw = 10 ** y_pred_log

    r2   = r2_score(y_true_raw, y_pred_raw)
    mae  = mean_absolute_error(y_true_raw, y_pred_raw)
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    mape = float(np.mean(np.abs((y_true_raw - y_pred_raw) / (y_true_raw + 1e-30))) * 100)

    return {
        "R²":     round(r2,   4),
        "MAE":    round(mae,  10),
        "RMSE":   round(rmse, 10),
        "MAPE %": round(mape, 2),
    }

# ============================================================
# TRAINING (MODULE 2)
# ============================================================

def train_all_models(X_train, X_test, y_train, y_test,
                     X_train_sc, X_test_sc, y_test_raw):
    models    = build_models()
    results   = {}
    metrics_rows = []

    for name, model in models.items():
        print(f"  Training {name}...", end=" ")

        if name in ("ANN", "SVR"):
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        m = compute_metrics(y_test, y_pred, y_test_raw)
        m["Model"] = name
        metrics_rows.append(m)
        results[name] = {"model": model, "y_pred": y_pred, "metrics": m}
        print(f"R²={m['R²']:.4f}  MAPE={m['MAPE %']:.2f}%")

    metrics_df = (pd.DataFrame(metrics_rows)
                  [["Model","R²","MAE","RMSE","MAPE %"]]
                  .sort_values("R²", ascending=False)
                  .reset_index(drop=True))

    return results, metrics_df

# ============================================================
# MODULE 3 — POST-TRAINING AI INTERPRETATION
# ============================================================

def ai_posttraining_analysis(metrics_df, results, X_test):
    best_row  = metrics_df.iloc[0]
    worst_row = metrics_df.iloc[-1]

    best_name  = best_row["Model"]
    best_model = results[best_name]["model"]
    importance_summary = None
    if best_name in ("Random Forest", "XGBoost", "Gradient Boosting"):
        importances = best_model.feature_importances_
        importances = importances / (importances.sum() + 1e-12)
        importance_summary = {
            label: round(float(val), 4)
            for label, val in sorted(
                zip(FEATURE_LABELS, importances),
                key=lambda x: -x[1]
            )
        }

    prompt = f"""You are an expert in machine learning for nanofluid electrical
conductivity prediction.

Training has completed for five algorithms (Random Forest, Gradient Boosting,
XGBoost, SVR, ANN) using the SHEN ET AL. 9-parameter feature set (the
conventional 5 parameters plus base fluid electrical conductivity, relative
permittivity, density, and kinematic viscosity). Target was log10(sigma_nf),
since the raw electrical conductivity spans about 10 orders of magnitude.
This uses the SAME held-out test rows as the Conventional (5) feature set
run in Colab 3.

Full results table (sorted by R², best first — R2/MAE/RMSE/MAPE all in raw S/m space):
{metrics_df.to_json(orient="records", indent=2)}

Best model: {best_name}  (R² = {best_row['R²']}, MAPE = {best_row['MAPE %']}%)
Weakest model: {worst_row['Model']}  (R² = {worst_row['R²']}, MAPE = {worst_row['MAPE %']}%)

{"Feature importance for the best model (normalised): " + json.dumps(importance_summary, indent=2) if importance_summary else ""}

Provide structured commentary (3-4 sentences each):

1. BEST-PERFORMING MODEL
   Identify the best-performing algorithm and state why its architecture
   suits this small, conventional feature set for a target spanning many
   orders of magnitude.

2. WEAKEST-PERFORMING MODEL
   Explain why the weakest algorithm likely underperforms here.

3. INTERPRETATION OF FEATURE IMPORTANCE
   {"Interpret the feature importance ranking above. In particular: do sigma_bf, eps_r, rho_bf, or upsilon rank ahead of the arbitrary nanoparticle_encoded/base_fluid_encoded category labels? If the physics-derived features genuinely outrank the category labels, that supports the case these features encode real transferable information rather than the model just memorising which system each row belongs to." if importance_summary else "Feature importances aren't available for this model type — note this and suggest permutation importance as a next step."}

4. COMPARISON TO CONVENTIONAL (5) BASELINE
   Compare this 9-parameter result to the Conventional (5) feature set's best
   result from Colab 3 (Random Forest/XGBoost/etc., R² around 0.97-0.995 in
   raw S/m space depending on model). Did adding sigma_bf, eps_r, rho_bf,
   upsilon meaningfully improve R²/MAPE, or is the improvement marginal —
   suggesting these 4 physics-derived features mostly duplicate information
   already available through base_fluid_encoded?

Be specific, cite the actual numbers, and keep it concise."""

    response = get_client().messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ============================================================
# PERSIST ARTIFACTS
# ============================================================

def save_artifacts(models_results, scaler, test_df):
    models_only = {name: r["model"] for name, r in models_results.items()}

    with open(MODEL_DIR / "ec_shen_trained_models.pkl", "wb") as f:
        pickle.dump(models_only, f)
    with open(MODEL_DIR / "ec_shen_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "ec_shen_feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    keep_cols = list(dict.fromkeys(
        ["group_id", "subgroup_id", "nanoparticle", "base_fluid"]
        + FEATURE_COLS + [TARGET_COL, RAW_TARGET_COL]
    ))
    keep_cols = [c for c in keep_cols if c in test_df.columns]
    test_df[keep_cols].to_csv(TEST_CSV, index=False)

    print(f"✅ Saved models  → {MODEL_DIR}")
    print(f"✅ Saved test set → {TEST_CSV}")

# ============================================================
# FIGURES
# ============================================================

BG     = "white"
GRID_C = "#EEEEEE"

def _ax_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, color="#1E3A5F")
    ax.set_xlabel(xlabel, fontsize=10, color="#333333")
    ax.set_ylabel(ylabel, fontsize=10, color="#333333")
    ax.tick_params(colors="#333333", labelsize=9)
    ax.grid(True, color=GRID_C, linewidth=0.7, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")


def fig_predicted_vs_actual(results, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Predicted vs Experimental log10(σ_nf)",
                 fontsize=14, fontweight="bold", color="#1E3A5F", y=1.02)

    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["y_pred"]
        color  = MODEL_COLORS[name]

        lim = [min(y_test.min(), y_pred.min())-0.3,
               max(y_test.max(), y_pred.max())+0.3]

        ax.scatter(y_test, y_pred, color=color, alpha=0.6,
                   s=25, edgecolors="none", zorder=3)
        ax.plot(lim, lim, "k--", linewidth=1.2, alpha=0.6, zorder=2)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_facecolor(BG)

        r2   = res["metrics"]["R²"]
        mape = res["metrics"]["MAPE %"]
        ax.text(0.05, 0.93,
                f"R² = {r2}\nMAPE (raw) = {mape}%",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8,
                          edgecolor="#CCCCCC", boxstyle="round,pad=0.4"))

        _ax_style(ax, name,
                  "Experimental log10(σ_nf)",
                  "Predicted log10(σ_nf)")

    plt.tight_layout()
    path = FIGURES_DIR + "fig1_pred_vs_actual.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    return path


def fig_metrics_comparison(metrics_df):
    models = metrics_df["Model"].tolist()
    colors = [MODEL_COLORS[m] for m in models]
    x      = np.arange(len(models))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("ML Model Performance Comparison — Shen et al. (9) Feature Set",
                 fontsize=14, fontweight="bold", color="#1E3A5F")

    for ax, col in zip(axes, ["R²","MAE","RMSE","MAPE %"]):
        vals = metrics_df[col].values.astype(float)
        bars = ax.bar(x, vals, color=colors, edgecolor="white",
                      linewidth=0.5, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.set_facecolor(BG)
        ax.grid(True, axis="y", color=GRID_C, linewidth=0.7, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")
        ax.set_title(col, fontsize=12, fontweight="bold", color="#1E3A5F")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(vals.max())*0.01,
                    f"{v:.4f}", ha="center", fontsize=8, color="#333333")

    plt.tight_layout()
    path = FIGURES_DIR + "fig2_metrics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    return path


def fig_feature_importance(results, X_test, y_test, X_test_sc):
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Feature Importance — Shen et al. (9) Feature Set",
                 fontsize=14, fontweight="bold", color="#1E3A5F", y=1.02)

    for ax, (name, res) in zip(axes, results.items()):
        model  = res["model"]
        color  = MODEL_COLORS[name]

        if name in ("Random Forest", "XGBoost", "Gradient Boosting"):
            importances = model.feature_importances_
        else:
            perm = permutation_importance(
                model, X_test_sc, y_test,
                n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
            )
            importances = perm.importances_mean

        importances = importances / (importances.sum() + 1e-12)
        order = np.argsort(importances)
        labels_ord = [FEATURE_LABELS[i] for i in order]

        ax.barh(labels_ord, importances[order],
                color=color, edgecolor="none", zorder=3)
        ax.set_facecolor(BG)
        ax.grid(True, axis="x", color=GRID_C, linewidth=0.7, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")
        _ax_style(ax, name, "Relative Importance", "")

    plt.tight_layout()
    path = FIGURES_DIR + "fig3_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    return path


def fig_residual_distribution(results, y_test):
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Prediction Error Distribution (log-space residuals)",
                 fontsize=14, fontweight="bold", color="#1E3A5F", y=1.02)

    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["y_pred"]
        errors = y_pred - y_test   # log-space residual (dex)
        color  = MODEL_COLORS[name]

        ax.hist(errors, bins=30, color=color, edgecolor="white",
                alpha=0.85, linewidth=0.5, zorder=3)
        ax.axvline(0, color="#1E3A5F", linestyle="--",
                   linewidth=1.5, label="Zero error")
        ax.axvline(errors.mean(), color="#E63946", linestyle="--",
                   linewidth=1.2, label=f"Mean={errors.mean():.3f} dex")
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID_C, linewidth=0.7, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")
        ax.legend(fontsize=8)
        _ax_style(ax, name, "Error (log10 units / dex)", "Count")

    plt.tight_layout()
    path = FIGURES_DIR + "fig4_residual_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    return path


def fig_error_vs_features(results, X_test, y_test):
    best_name = max(results, key=lambda n: results[n]["metrics"]["R²"])
    y_pred    = results[best_name]["y_pred"]
    errors    = np.abs(y_pred - y_test)   # absolute log-space error (dex)
    phi_vals  = X_test[:, 0]
    T_vals    = X_test[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Prediction Error vs Input Features  ({best_name})",
                 fontsize=13, fontweight="bold", color="#1E3A5F")

    sc1 = ax1.scatter(phi_vals*100, errors,
                      c=errors, cmap="RdYlGn_r",
                      s=30, alpha=0.7, edgecolors="none",
                      vmin=0, vmax=errors.max(), zorder=3)
    plt.colorbar(sc1, ax=ax1, label="Absolute Error (dex)")
    ax1.set_facecolor(BG)
    _ax_style(ax1, "Error vs Volume Fraction",
              "Volume Fraction (%)", "Absolute Error (log10 units)")

    sc2 = ax2.scatter(T_vals, errors,
                      c=errors, cmap="RdYlGn_r",
                      s=30, alpha=0.7, edgecolors="none",
                      vmin=0, vmax=errors.max(), zorder=3)
    plt.colorbar(sc2, ax=ax2, label="Absolute Error (dex)")
    ax2.set_facecolor(BG)
    _ax_style(ax2, "Error vs Temperature",
              "Temperature (K)", "Absolute Error (log10 units)")

    plt.tight_layout()
    path = FIGURES_DIR + "fig5_error_vs_features.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    return path


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="MANIS — EC ML Development Agent (Colab 4 — Shen et al. 9-param)", theme=gr.themes.Soft()) as app:

    gr.Markdown("# 🤖 MANIS — Electrical Conductivity ML Development Agent (Colab 4)")
    gr.Markdown(
        "**Shen et al. (9-parameter) feature set** — conventional 5 parameters plus "
        "base fluid electrical conductivity, permittivity, density, kinematic viscosity.  \n"
        "Trains on log10(σ_nf) given the ~10 orders of magnitude spanned by "
        "electrical conductivity in this dataset. MAPE reported in back-transformed "
        "raw S/m space for physical interpretability.  \n"
        "Reads `nanofluid_ec_features.xlsx` → AI pre-training review → trains 5 models "
        "→ AI post-training interpretation → saves figures + model artifacts + "
        "held-out test set to Drive."
    )

    results_state    = gr.State(None)
    metrics_state    = gr.State(None)
    data_arrays_state= gr.State(None)

    with gr.Tabs():

        with gr.Tab("1 — Load & Train"):
            gr.Markdown("### Step 1 — Load Feature Dataset")
            load_btn    = gr.Button("📂 Load Dataset", variant="secondary")
            load_status = gr.Textbox(label="Load Status", interactive=False, lines=4)

            gr.Markdown("### Step 2 — AI Pre-Training Review")
            pretrain_btn    = gr.Button("🤖 Generate Pre-Training AI Review", variant="secondary")
            pretrain_status = gr.Textbox(label="Pre-Training AI Recommendation",
                                          interactive=False, lines=14)

            gr.Markdown("### Step 3 — Train All Models")
            train_btn    = gr.Button("▶  Train All Models", variant="primary", size="lg")
            train_status = gr.Textbox(label="Training Log", interactive=False, lines=10)
            metrics_tbl  = gr.Dataframe(
                label="📊 Model Metrics — Shen et al. (9) — sorted by R², raw S/m space",
                interactive=False, wrap=True
            )

        with gr.Tab("2 — AI Interpretation"):
            gr.Markdown("### Post-Training AI Interpretation")
            posttrain_btn    = gr.Button("🤖 Generate Post-Training Interpretation",
                                          variant="primary", size="lg")
            posttrain_status = gr.Textbox(label="Post-Training AI Interpretation",
                                           interactive=False, lines=26)

        with gr.Tab("3 — Figures"):
            gr.Markdown("### Generate & Save All Figures to Google Drive")
            figs_btn    = gr.Button("📈 Generate All Figures", variant="primary", size="lg")
            figs_status = gr.Textbox(label="Status", interactive=False, lines=3)

            with gr.Row():
                fig1_out = gr.Image(label="Fig 1 — Predicted vs Experimental (log-space)")
                fig2_out = gr.Image(label="Fig 2 — Metrics Comparison")
            with gr.Row():
                fig3_out = gr.Image(label="Fig 3 — Feature Importance")
                fig4_out = gr.Image(label="Fig 4 — Residual Distribution")
            fig5_out = gr.Image(label="Fig 5 — Error vs Volume Fraction & Temperature")

        with gr.Tab("4 — Google Drive"):
            gr.Markdown(f"Reads: `{FEATURE_XLS}`  \nFigures saved to: `{FIGURES_DIR}`  \n"
                       f"Models saved to: `{MODEL_DIR}`  \nTest set saved to: `{TEST_CSV}`  \n"
                       f"AI notes saved to: `{PRETRAIN_TXT}`, `{POSTTRAIN_TXT}`")
            with gr.Row():
                mount_btn = gr.Button("Mount / Remount Drive", variant="secondary")
                check_btn = gr.Button("Check Drive Status",    variant="secondary")
            drive_status = gr.Textbox(label="Status", interactive=False, lines=8)

            def check_drive_status():
                if not os.path.exists(DRIVE_DIR):
                    return "❌ Drive not mounted."
                lines = ["✅ Drive mounted:"]
                for f in sorted(os.listdir(DRIVE_DIR)):
                    full = os.path.join(DRIVE_DIR, f)
                    if os.path.isfile(full):
                        sz = os.path.getsize(full)
                        lines.append(f"  {f}  ({sz:,} bytes)")
                if os.path.exists(MODEL_DIR):
                    lines.append(f"\nModels folder ({MODEL_DIR}):")
                    for f in sorted(os.listdir(MODEL_DIR)):
                        sz = os.path.getsize(os.path.join(MODEL_DIR, f))
                        lines.append(f"  {f}  ({sz:,} bytes)")
                if os.path.exists(RESULTS_DIR):
                    lines.append(f"\nResults folder ({RESULTS_DIR}):")
                    for f in sorted(os.listdir(RESULTS_DIR)):
                        sz = os.path.getsize(os.path.join(RESULTS_DIR, f))
                        lines.append(f"  {f}  ({sz:,} bytes)")
                if os.path.exists(FIGURES_DIR):
                    lines.append(f"\nFigures folder ({FIGURES_DIR}):")
                    for f in sorted(os.listdir(FIGURES_DIR)):
                        sz = os.path.getsize(os.path.join(FIGURES_DIR, f))
                        lines.append(f"  {f}  ({sz:,} bytes)")
                return "\n".join(lines)

            def do_mount():
                mount_drive()
                return check_drive_status()

            mount_btn.click(do_mount,             outputs=[drive_status])
            check_btn.click(check_drive_status,   outputs=[drive_status])

    # ── CALLBACKS ─────────────────────────────────────────────

    def do_load():
        try:
            result = load_data()
            msg = result[9]
            if result[0] is None:
                return msg, None

            (X_train, X_test, y_train, y_test,
             X_train_sc, X_test_sc, scaler, df, test_df, _,
             y_train_raw, y_test_raw) = result

            data = {
                "X_train":     X_train,
                "X_test":      X_test,
                "y_train":     y_train,
                "y_test":      y_test,
                "y_train_raw": y_train_raw,
                "y_test_raw":  y_test_raw,
                "X_train_sc":  X_train_sc,
                "X_test_sc":   X_test_sc,
                "scaler":      scaler,
                "df":          df,
                "test_df":     test_df,
            }
            return msg, data
        except Exception as e:
            return f"❌ Error loading data: {str(e)}", None

    def do_pretraining_review(data_arrays):
        if data_arrays is None:
            return "❌ Load data first (Step 1)."
        try:
            notes = ai_pretraining_analysis(
                data_arrays["df"], data_arrays["X_train"], data_arrays["X_test"],
                data_arrays["y_train"], data_arrays["y_test"]
            )
            with open(PRETRAIN_TXT, "w") as f:
                f.write(notes)
            return notes
        except Exception as e:
            return f"❌ Pre-training review error: {str(e)}"

    def do_train(data_arrays):
        if data_arrays is None:
            return ("❌ Load data first.",
                    pd.DataFrame(), None, None)
        try:
            print("Training models...")
            results, metrics_df = train_all_models(
                data_arrays["X_train"], data_arrays["X_test"],
                data_arrays["y_train"], data_arrays["y_test"],
                data_arrays["X_train_sc"], data_arrays["X_test_sc"],
                data_arrays["y_test_raw"],
            )

            save_artifacts(results, data_arrays["scaler"], data_arrays["test_df"])
            # --- REVISION: save the metrics table (was never saved before) ---
            metrics_df.to_csv(RESULTS_DIR / "ec_shen_dev_metrics.csv", index=False)

            best = metrics_df.iloc[0]
            status = (f"✅ Training complete!\n"
                      f"Best model: {best['Model']}  "
                      f"R²={best['R²']}  MAPE={best['MAPE %']}%\n"
                      f"Models saved → {MODEL_DIR}\n"
                      f"Test set saved → {TEST_CSV}\n"
                      f"Metrics saved → {RESULTS_DIR / 'ec_shen_dev_metrics.csv'}\n"
                      f"Go to Tab 2 for AI post-training interpretation.\n"
                      f"Figures ready — go to Tab 3.")
            return status, metrics_df, results, metrics_df
        except Exception as e:
            return f"❌ Training error: {str(e)}", pd.DataFrame(), None, None

    def do_posttraining_interpretation(results, metrics_df, data_arrays):
        if results is None or metrics_df is None or len(metrics_df) == 0:
            return "❌ Train models first (Step 3)."
        try:
            notes = ai_posttraining_analysis(metrics_df, results, data_arrays["X_test"])
            with open(POSTTRAIN_TXT, "w") as f:
                f.write(notes)
            return notes
        except Exception as e:
            return f"❌ Post-training interpretation error: {str(e)}"

    def do_figures(results, data_arrays):
        if results is None:
            return ("❌ Train models first.",
                    None, None, None, None, None)
        try:
            y_test    = data_arrays["y_test"]
            X_test    = data_arrays["X_test"]
            X_test_sc = data_arrays["X_test_sc"]
            metrics_df= pd.DataFrame([r["metrics"] for r in results.values()])

            print("Generating figures...")
            p1 = fig_predicted_vs_actual(results, y_test)
            p2 = fig_metrics_comparison(metrics_df)
            p3 = fig_feature_importance(results, X_test, y_test, X_test_sc)
            p4 = fig_residual_distribution(results, y_test)
            p5 = fig_error_vs_features(results, X_test, y_test)

            status = (f"✅ 5 figures saved to:\n{FIGURES_DIR}\n"
                      f"  fig1_pred_vs_actual.png\n"
                      f"  fig2_metrics_comparison.png\n"
                      f"  fig3_feature_importance.png\n"
                      f"  fig4_residual_distribution.png\n"
                      f"  fig5_error_vs_features.png")
            return status, p1, p2, p3, p4, p5
        except Exception as e:
            return f"❌ Figure error: {str(e)}", None, None, None, None, None

    load_btn.click(
        do_load,
        outputs=[load_status, data_arrays_state]
    )

    pretrain_btn.click(
        do_pretraining_review,
        inputs=[data_arrays_state],
        outputs=[pretrain_status]
    )

    train_btn.click(
        do_train,
        inputs=[data_arrays_state],
        outputs=[train_status, metrics_tbl, results_state, metrics_state]
    )

    posttrain_btn.click(
        do_posttraining_interpretation,
        inputs=[results_state, metrics_state, data_arrays_state],
        outputs=[posttrain_status]
    )

    figs_btn.click(
        do_figures,
        inputs=[results_state, data_arrays_state],
        outputs=[figs_status, fig1_out, fig2_out,
                 fig3_out, fig4_out, fig5_out]
    )

app.launch(share=True)
