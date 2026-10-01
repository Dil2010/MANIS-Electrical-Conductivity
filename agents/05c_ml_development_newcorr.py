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
# CELL B — NEW CORRELATION ML DEVELOPMENT AGENT
# ############################################################
# ============================================================
# MANIS: ELECTRICAL CONDUCTIVITY
# ML DEVELOPMENT AI AGENT: New Correlation (10-parameter) feature set
# ============================================================
# Reads  : nanofluid_ec_features.xlsx  (Feature Engineering Agent output)
# Models : ANN, Random Forest, XGBoost, Gradient Boosting, SVR
# Split  : by group_id (80/20, random_state=42), identical to the
#          Conventional (5) and Shen et al. (9) runs
# Target : log_sigma_nf (log10 of sigma_nf, S/m). Metrics are reported
#          on back-transformed raw S/m values.
# Output (all under MANIS_ELECTRICAL/):
#   ec_models/ec_newcorr_trained_models.pkl, _scaler.pkl, _feature_cols.json
#   ec_results/ec_newcorr_test_set.csv
#   ec_results/ec_newcorr_dev_metrics.csv
#   EC_ML_Figures/ec_newcorr_fig1..fig5_*.png
#   ec_newcorr_pretraining_ai_notes.txt / ec_newcorr_posttraining_ai_notes.txt
#
# FEATURES: particle_size_nm, volume_fraction, temperature_K,
#   nanoparticle_encoded, base_fluid_encoded,
#   sigma_p, rho_p, sigma_bf, eps_r, mu_bf
#
# CHANGES FROM PREVIOUS VERSION
#  1. DRIVE_DIR moved to MANIS_ELECTRICAL/.
#  2. API key setup + get_client() (fresh client per call).
#  3. Every output file carries the ec_newcorr_ prefix. Previously the
#     figures and AI notes had generic names and were overwritten by
#     whichever feature set ran last in the shared folder.
#  4. Metrics table saved to ec_newcorr_dev_metrics.csv.
#  5. Split consistency check: the held-out rows are compared with the
#     Conventional and Shen et al. test sets if those files exist. This
#     matters because rows missing sigma_p, rho_p or mu_bf are dropped
#     before splitting, which would silently change the held-out rows.
#  6. Baseline results in the post-training prompt are read from saved
#     files instead of being described from memory; if missing, the AI is
#     told not to estimate them.
#  7. Stale "Conventional (5)" wording removed from constants and prompts.
#
# REVISION CHANGE (2026-09-30):
#  8. Baseline tag "conv" renamed to "conventional" so the Conventional
#     metrics file (ec_conventional_dev_metrics.csv) is found, and the
#     split check falls back to ec_test_set.csv, the name the
#     Conventional agent uses for its held-out rows.
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

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import xgboost as xgb
import gradio as gr

warnings.filterwarnings('ignore')

# ============================================================
# FEATURE SET IDENTITY
# ============================================================
FS_TAG  = "newcorr"
FS_NAME = "New Correlation (10-parameter)"

# Other feature sets used for split check and baseline comparison.
# REVISION: "conv" renamed to "conventional" to match the file names the
# Conventional agent actually writes (ec_conventional_dev_metrics.csv).
BASELINES = {
    "conventional": "Conventional (5-parameter)",
    "shen": "Shen et al. (9-parameter)",
}

# ============================================================
# GOOGLE DRIVE
# ============================================================
DRIVE_DIR     = "/content/drive/MyDrive/MANIS_ELECTRICAL/"
FEATURE_XLS   = DRIVE_DIR + "nanofluid_ec_features.xlsx"
FIGURES_DIR   = DRIVE_DIR + "EC_ML_Figures/"
MODEL_DIR     = Path(DRIVE_DIR + "ec_models/")
RESULTS_DIR   = Path(DRIVE_DIR + "ec_results/")
TEST_CSV      = RESULTS_DIR / f"ec_{FS_TAG}_test_set.csv"
METRICS_CSV   = RESULTS_DIR / f"ec_{FS_TAG}_dev_metrics.csv"
PRETRAIN_TXT  = DRIVE_DIR + f"ec_{FS_TAG}_pretraining_ai_notes.txt"
POSTTRAIN_TXT = DRIVE_DIR + f"ec_{FS_TAG}_posttraining_ai_notes.txt"

def mount_drive():
    try:
        drive.mount('/content/drive', force_remount=False)
        for d in (DRIVE_DIR, FIGURES_DIR, MODEL_DIR, RESULTS_DIR):
            os.makedirs(d, exist_ok=True)
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
    print(f"⚠️ Key looks invalid (length {len(_key)}). Training works; AI tabs will not.")

CLAUDE_MODEL = "claude-sonnet-4-5"

def get_client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No ANTHROPIC_API_KEY in environment. Re-run the key setup.")
    return anthropic.Anthropic(api_key=key)

def ask_claude(prompt, max_tokens):
    return get_client().messages.create(
        model=CLAUDE_MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    ).content[0].text

# ============================================================
# CONSTANTS: NEW CORRELATION (10) FEATURE SET
# ============================================================
FEATURE_COLS = [
    "volume_fraction", "temperature_K", "particle_size_nm",
    "nanoparticle_encoded", "base_fluid_encoded",
    "sigma_p", "rho_p", "sigma_bf", "eps_r", "mu_bf",
]
FEATURE_LABELS = [
    "φ (vol. fraction)", "T (K)", "d_p (nm)", "NP encoded", "BF encoded",
    "σ_p (S/m)", "ρ_p (kg/m³)", "σ_bf (S/m)", "ε_r", "μ_bf (Pa·s)",
]
TARGET_COL     = "log_sigma_nf"
RAW_TARGET_COL = "sigma_nf"
RANDOM_STATE   = 42

MODEL_COLORS = {
    "ANN":               "#E63946",
    "Random Forest":     "#2A9D8F",
    "XGBoost":           "#E9C46A",
    "Gradient Boosting": "#457B9D",
    "SVR":               "#F4A261",
}

COL_MAP = {
    "Group\nID": "group_id", "Sub\nGroup ID": "subgroup_id",
    "Nanoparticle": "nanoparticle", "Base Fluid": "base_fluid",
    "d_p": "particle_size_nm", "φ": "volume_fraction", "T": "temperature_K",
    "σ_bf": "sigma_bf", "ε_r": "eps_r", "ρ_bf": "rho_bf", "υ": "upsilon",
    "σ_p": "sigma_p", "ρ_p": "rho_p", "μ_bf": "mu_bf", "µ_bf": "mu_bf",
    "NP\nEncoded": "nanoparticle_encoded", "BF\nEncoded": "base_fluid_encoded",
    "σ_nf  ← TARGET": "sigma_nf",
    "log10(σ_nf)  ← TARGET": "log_sigma_nf",
}

# ============================================================
# SPLIT CONSISTENCY CHECK
# ============================================================
def _row_keys(d):
    return sorted(zip(d["group_id"].astype(int),
                      d["volume_fraction"].astype(float).round(8),
                      d["temperature_K"].astype(float).round(3),
                      d[RAW_TARGET_COL].astype(float).map(lambda v: float(f"{v:.6e}"))))

def check_split_consistency(test_df):
    lines = []
    mine = _row_keys(test_df)
    for tag, name in BASELINES.items():
        p = RESULTS_DIR / f"ec_{tag}_test_set.csv"
        # REVISION: the Conventional agent saves its held-out rows as ec_test_set.csv
        if not p.exists() and tag == "conventional":
            p = RESULTS_DIR / "ec_test_set.csv"
        if not p.exists():
            lines.append(f"   {name}: test set not found, cannot check")
            continue
        other = _row_keys(pd.read_csv(p))
        if other == mine:
            lines.append(f"   ✅ {name}: identical held-out rows ({len(mine)})")
        else:
            diff = len(set(mine) ^ set(other))
            lines.append(f"   ❌ {name}: held-out rows DIFFER ({len(mine)} here vs "
                         f"{len(other)}, {diff} rows not shared)")
    return "\n".join(lines)

# ============================================================
# DATA LOADING & SPLITTING
# ============================================================
def _fail(msg):
    return (None,) * 9 + (msg, None, None)

def load_data():
    if not Path(FEATURE_XLS).exists():
        return _fail("❌ nanofluid_ec_features.xlsx not found. Run the Feature Engineering Agent first.")
    try:
        xl     = pd.ExcelFile(FEATURE_XLS)
        sheet  = next((s for s in xl.sheet_names if "Feature" in s and "Set" not in s),
                      xl.sheet_names[0])
        df     = pd.read_excel(FEATURE_XLS, sheet_name=sheet, header=1, skiprows=[2])
        df     = df.rename(columns=COL_MAP)

        if "nanoparticle_encoded" not in df.columns:
            print("   ⚠️  nanoparticle_encoded missing, computing now")
            df["nanoparticle_encoded"] = LabelEncoder().fit_transform(df["nanoparticle"].astype(str))
        if "base_fluid_encoded" not in df.columns:
            print("   ⚠️  base_fluid_encoded missing, computing now")
            df["base_fluid_encoded"] = LabelEncoder().fit_transform(df["base_fluid"].astype(str))

        required = FEATURE_COLS + [TARGET_COL, RAW_TARGET_COL, "group_id"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return _fail(f"❌ Missing columns in Excel:\n{missing}\n\n"
                         f"Available columns:\n{list(df.columns)}")

        n_before = len(df)
        df = df.dropna(subset=required).reset_index(drop=True)
        n_dropped = n_before - len(df)

        X     = df[FEATURE_COLS].values.astype(float)
        y_log = df[TARGET_COL].values.astype(float)
        y_raw = df[RAW_TARGET_COL].values.astype(float)

        # Group-aware 80/20 split, identical logic across all feature sets
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
               f"   Rows: {len(df)} ({n_dropped} dropped for missing values)  |  "
               f"Groups: {df['group_id'].nunique()}\n"
               f"   Train: {len(X_train)}  |  Test: {len(X_test)}\n"
               f"   Features: {len(FEATURE_COLS)} ({FS_NAME})  |  Target: {TARGET_COL}\n"
               f"   Split check against other feature sets:\n{check_split_consistency(test_df)}")

        return (X_train, X_test, y_train, y_test, X_train_sc, X_test_sc,
                scaler, df, test_df, msg, y_train_raw, y_test_raw)
    except Exception as e:
        return _fail(f"❌ Error reading Excel: {e}")

# ============================================================
# BASELINE RESULTS (read from saved files, never from memory)
# ============================================================
def load_baseline_results():
    """Looks for each baseline's dev metrics, then its evaluation agent
    comparison table. Returns a text block for the prompt."""
    blocks = []
    for tag, name in BASELINES.items():
        found = None
        for fname in (f"ec_{tag}_dev_metrics.csv", f"ec_{tag}_model_comparison.csv"):
            p = RESULTS_DIR / fname
            if p.exists():
                found = (fname, pd.read_csv(p))
                break
        if found:
            blocks.append(f"{name} (from {found[0]}):\n{found[1].to_json(orient='records', indent=2)}")
        else:
            blocks.append(f"{name}: results file NOT found. State this and do NOT estimate its numbers.")
    return "\n\n".join(blocks)

# ============================================================
# MODULE 1: PRE-TRAINING AI RECOMMENDATION
# ============================================================
FEATURE_SET_CONTEXT = f"""FEATURE SET: {FS_NAME}. The conventional five parameters
(nanoparticle type, base fluid type, particle size, volume fraction, temperature)
plus five properties: nanoparticle electrical conductivity (sigma_p), nanoparticle
density (rho_p), base fluid electrical conductivity (sigma_bf), relative
permittivity (eps_r), and base fluid viscosity (mu_bf). Zeta potential (U0),
nanolayer thickness (h), and sphericity (psi) from the original New Correlation
model were not used as ML features. The target is log10(sigma_nf) because raw
electrical conductivity spans many orders of magnitude in this dataset."""

def ai_pretraining_analysis(df, X_train, X_test, y_train, y_test):
    group_sizes = df.groupby("group_id").size()
    feature_stats = {
        label: {"min": float(np.nanmin(df[c])), "max": float(np.nanmax(df[c])),
                "mean": float(np.nanmean(df[c])), "variance": float(np.nanvar(df[c])),
                "unique_values": int(df[c].nunique())}
        for c, label in zip(FEATURE_COLS, FEATURE_LABELS)
    }
    summary = {
        "total_records": len(df), "total_groups": int(df["group_id"].nunique()),
        "train_size": len(X_train), "test_size": len(X_test),
        "smallest_group_size": int(group_sizes.min()),
        "largest_group_size": int(group_sizes.max()),
        "nanoparticle_counts": df["nanoparticle"].value_counts().to_dict(),
        "base_fluid_counts": df["base_fluid"].value_counts().to_dict(),
        "target_range_log10": f"{y_train.min():.2f} to {y_train.max():.2f} (log10 S/m)",
        "feature_stats": feature_stats,
    }

    prompt = f"""You are an expert in machine learning model development for nanofluid
electrical conductivity prediction. Review this dataset before training.

{FEATURE_SET_CONTEXT}

Dataset summary:
{json.dumps(summary, indent=2)}

Planned models: Random Forest, Gradient Boosting, XGBoost, SVR, and an ANN (MLP),
trained on these 10 features with the same group-aware 80/20 split used for the
Conventional (5) and Shen et al. (9) feature sets.

Provide concise commentary (3 to 4 sentences each):

1. DATASET SUITABILITY: is the dataset adequate in size and diversity for these
   five algorithms on 10 features? Flag small groups and under-represented
   nanoparticle or base fluid categories.
2. FEATURE SCALE: sigma_p and sigma_bf each span orders of magnitude, unlike
   temperature or the integer encodings. Is standard scaling adequate for ANN and
   SVR, and which features could dominate distance-based models?
3. EXPECTED CHALLENGES: sigma_p and rho_p are fixed literature values per
   nanoparticle material (see unique_values), so they may be collinear with
   nanoparticle_encoded. sigma_bf, eps_r, and mu_bf may be collinear with
   base_fluid_encoded. Flag whether these add information or duplicate the
   category labels, and the risk of spurious ordinal splits on integer encodings.
4. RECOMMENDED FOCUS: which algorithm(s) suit this feature set, and what to
   expect relative to the Conventional (5) and Shen et al. (9) sets.

Use the actual numbers. Be concise. Do not use em dashes."""
    return ask_claude(prompt, 1200)

# ============================================================
# MODEL DEFINITIONS
# ============================================================
def build_models():
    return {
        "ANN": MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                            solver="adam", max_iter=1000, learning_rate_init=0.001,
                            early_stopping=True, validation_fraction=0.1,
                            random_state=RANDOM_STATE, verbose=False),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=None,
                                               min_samples_split=2,
                                               random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=RANDOM_STATE, verbosity=0),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                       learning_rate=0.05, subsample=0.8,
                                                       random_state=RANDOM_STATE),
        "SVR": SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.001),
    }

# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true_log, y_pred_log, y_true_raw):
    """Trained in log space; R², MAE, RMSE and MAPE reported on
    back-transformed raw S/m, matching the mechanistic benchmarking table."""
    y_pred_raw = 10 ** y_pred_log
    return {
        "R²":     round(r2_score(y_true_raw, y_pred_raw), 4),
        "MAE":    round(mean_absolute_error(y_true_raw, y_pred_raw), 10),
        "RMSE":   round(float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))), 10),
        "MAPE %": round(float(np.mean(np.abs((y_true_raw - y_pred_raw) /
                                             (y_true_raw + 1e-30))) * 100), 2),
    }

# ============================================================
# MODULE 2: TRAINING
# ============================================================
def train_all_models(X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, y_test_raw):
    results, rows = {}, []
    for name, model in build_models().items():
        print(f"  Training {name}...", end=" ")
        if name in ("ANN", "SVR"):
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        m = compute_metrics(y_test, y_pred, y_test_raw)
        m["Model"] = name
        rows.append(m)
        results[name] = {"model": model, "y_pred": y_pred, "metrics": m}
        print(f"R²={m['R²']:.4f}  MAPE={m['MAPE %']:.2f}%")

    metrics_df = (pd.DataFrame(rows)[["Model", "R²", "MAE", "RMSE", "MAPE %"]]
                  .sort_values("R²", ascending=False).reset_index(drop=True))
    return results, metrics_df

# ============================================================
# MODULE 3: POST-TRAINING AI INTERPRETATION
# ============================================================
def ai_posttraining_analysis(metrics_df, results):
    best_row, worst_row = metrics_df.iloc[0], metrics_df.iloc[-1]
    best_name = best_row["Model"]
    importance_summary = None
    if best_name in ("Random Forest", "XGBoost", "Gradient Boosting"):
        imp = results[best_name]["model"].feature_importances_
        imp = imp / (imp.sum() + 1e-12)
        importance_summary = {l: round(float(v), 4)
                              for l, v in sorted(zip(FEATURE_LABELS, imp), key=lambda x: -x[1])}

    importance_block = ("Feature importance for the best model (normalised):\n"
                        + json.dumps(importance_summary, indent=2)) if importance_summary else ""
    importance_task = (
        "Interpret the ranking above. Do sigma_p and rho_p (fixed per nanoparticle material) "
        "or sigma_bf, eps_r, mu_bf (largely set by base fluid choice) outrank the "
        "nanoparticle_encoded and base_fluid_encoded labels? Note that impurity-based "
        "importance splits credit arbitrarily between collinear features, so a high rank "
        "does not by itself prove new information."
        if importance_summary else
        "Importances are not available for this model type; note this and suggest "
        "permutation importance.")

    prompt = f"""You are an expert in machine learning for nanofluid electrical conductivity
prediction. Training has completed for five algorithms.

{FEATURE_SET_CONTEXT}

Results on the held-out test set, sorted by R² (all metrics in raw S/m):
{metrics_df.to_json(orient="records", indent=2)}

Best model: {best_name} (R² = {best_row['R²']}, MAPE = {best_row['MAPE %']}%)
Weakest model: {worst_row['Model']} (R² = {worst_row['R²']}, MAPE = {worst_row['MAPE %']}%)

{importance_block}

Baseline results on the same held-out split:
{load_baseline_results()}

Provide structured commentary (3 to 4 sentences each):

1. BEST-PERFORMING MODEL: why its architecture suits this feature set and a target
   spanning many orders of magnitude.
2. WEAKEST-PERFORMING MODEL: why it underperforms here.
3. FEATURE IMPORTANCE: {importance_task}
4. COMPARISON TO BASELINES: using only the baseline numbers given above, does the
   New Correlation set improve R² and MAPE over Conventional (5) and Shen et al. (9),
   per algorithm, or has performance plateaued?

Cite the actual numbers. Be concise. Do not use em dashes."""
    return ask_claude(prompt, 1500)

# ============================================================
# PERSIST ARTIFACTS
# ============================================================
def save_artifacts(models_results, scaler, test_df, metrics_df):
    with open(MODEL_DIR / f"ec_{FS_TAG}_trained_models.pkl", "wb") as f:
        pickle.dump({n: r["model"] for n, r in models_results.items()}, f)
    with open(MODEL_DIR / f"ec_{FS_TAG}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / f"ec_{FS_TAG}_feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    keep = list(dict.fromkeys(["group_id", "subgroup_id", "nanoparticle", "base_fluid"]
                              + FEATURE_COLS + [TARGET_COL, RAW_TARGET_COL]))
    test_df[[c for c in keep if c in test_df.columns]].to_csv(TEST_CSV, index=False)

    out = metrics_df.rename(columns={"R²": "R2", "MAE": "MAE_Sm",
                                     "RMSE": "RMSE_Sm", "MAPE %": "MAPE_%"})
    out.to_csv(METRICS_CSV, index=False)

    print(f"✅ Saved models   → {MODEL_DIR}")
    print(f"✅ Saved test set → {TEST_CSV}")
    print(f"✅ Saved metrics  → {METRICS_CSV}")

# ============================================================
# FIGURES
# ============================================================
BG, GRID_C = "white", "#EEEEEE"

def fig_path(name):
    return FIGURES_DIR + f"ec_{FS_TAG}_{name}.png"

def _ax_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, color="#1E3A5F")
    ax.set_xlabel(xlabel, fontsize=10, color="#333333")
    ax.set_ylabel(ylabel, fontsize=10, color="#333333")
    ax.tick_params(colors="#333333", labelsize=9)
    ax.grid(True, color=GRID_C, linewidth=0.7, zorder=0)
    for s in ax.spines.values():
        s.set_edgecolor("#CCCCCC")


def fig_predicted_vs_actual(results, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Predicted vs Experimental log10(σ_nf): {FS_NAME}",
                 fontsize=14, fontweight="bold", color="#1E3A5F", y=1.02)
    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["y_pred"]
        lim = [min(y_test.min(), y_pred.min()) - 0.3, max(y_test.max(), y_pred.max()) + 0.3]
        ax.scatter(y_test, y_pred, color=MODEL_COLORS[name], alpha=0.6, s=25,
                   edgecolors="none", zorder=3)
        ax.plot(lim, lim, "k--", linewidth=1.2, alpha=0.6, zorder=2)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_facecolor(BG)
        ax.text(0.05, 0.93, f"R² = {res['metrics']['R²']}\nMAPE (raw) = {res['metrics']['MAPE %']}%",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="#CCCCCC",
                          boxstyle="round,pad=0.4"))
        _ax_style(ax, name, "Experimental log10(σ_nf)", "Predicted log10(σ_nf)")
    plt.tight_layout()
    path = fig_path("fig1_pred_vs_actual")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def fig_metrics_comparison(metrics_df):
    models = metrics_df["Model"].tolist()
    colors = [MODEL_COLORS[m] for m in models]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"ML Model Performance Comparison: {FS_NAME}",
                 fontsize=14, fontweight="bold", color="#1E3A5F")
    for ax, col in zip(axes, ["R²", "MAE", "RMSE", "MAPE %"]):
        vals = metrics_df[col].values.astype(float)
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.set_facecolor(BG)
        ax.grid(True, axis="y", color=GRID_C, linewidth=0.7, zorder=0)
        for s in ax.spines.values():
            s.set_edgecolor("#CCCCCC")
        ax.set_title(col, fontsize=12, fontweight="bold", color="#1E3A5F")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(vals.max())*0.01,
                    f"{v:.4g}", ha="center", fontsize=8, color="#333333")
    plt.tight_layout()
    path = fig_path("fig2_metrics_comparison")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def fig_feature_importance(results, y_test, X_test_sc):
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Feature Importance: {FS_NAME}",
                 fontsize=14, fontweight="bold", color="#1E3A5F", y=1.02)
    for ax, (name, res) in zip(axes, results.items()):
        model = res["model"]
        if name in ("Random Forest", "XGBoost", "Gradient Boosting"):
            imp = model.feature_importances_
        else:
            imp = permutation_importance(model, X_test_sc, y_test, n_repeats=10,
                                         random_state=RANDOM_STATE, n_jobs=-1).importances_mean
        imp = imp / (imp.sum() + 1e-12)
        order = np.argsort(imp)
        ax.barh([FEATURE_LABELS[i] for i in order], imp[order],
                color=MODEL_COLORS[name], edgecolor="none", zorder=3)
        ax.set_facecolor(BG)
        _ax_style(ax, name, "Relative Importance", "")
    plt.tight_layout()
    path = fig_path("fig3_feature_importance")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def fig_residual_distribution(results, y_test):
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Prediction Error Distribution (log-space residuals): {FS_NAME}",
                 fontsize=14, fontweight="bold", color="#1E3A5F", y=1.02)
    for ax, (name, res) in zip(axes, results.items()):
        errors = res["y_pred"] - y_test
        ax.hist(errors, bins=30, color=MODEL_COLORS[name], edgecolor="white",
                alpha=0.85, linewidth=0.5, zorder=3)
        ax.axvline(0, color="#1E3A5F", linestyle="--", linewidth=1.5, label="Zero error")
        ax.axvline(errors.mean(), color="#E63946", linestyle="--", linewidth=1.2,
                   label=f"Mean={errors.mean():.3f} dex")
        ax.set_facecolor(BG)
        ax.legend(fontsize=8)
        _ax_style(ax, name, "Error (log10 units / dex)", "Count")
    plt.tight_layout()
    path = fig_path("fig4_residual_distribution")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def fig_error_vs_features(results, X_test, y_test):
    best_name = max(results, key=lambda n: results[n]["metrics"]["R²"])
    errors = np.abs(results[best_name]["y_pred"] - y_test)
    phi_vals = X_test[:, FEATURE_COLS.index("volume_fraction")]
    T_vals = X_test[:, FEATURE_COLS.index("temperature_K")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Prediction Error vs Input Features ({best_name}, {FS_NAME})",
                 fontsize=13, fontweight="bold", color="#1E3A5F")
    for ax, xv, xl, title in ((ax1, phi_vals * 100, "Volume Fraction (%)", "Error vs Volume Fraction"),
                              (ax2, T_vals, "Temperature (K)", "Error vs Temperature")):
        sc = ax.scatter(xv, errors, c=errors, cmap="RdYlGn_r", s=30, alpha=0.7,
                        edgecolors="none", vmin=0, vmax=errors.max(), zorder=3)
        plt.colorbar(sc, ax=ax, label="Absolute Error (dex)")
        ax.set_facecolor(BG)
        _ax_style(ax, title, xl, "Absolute Error (log10 units)")
    plt.tight_layout()
    path = fig_path("fig5_error_vs_features")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path

# ============================================================
# DRIVE STATUS
# ============================================================
def check_drive_status():
    if not os.path.exists(DRIVE_DIR):
        return "❌ Drive not mounted."
    lines = [f"✅ Drive mounted: {DRIVE_DIR}"]
    for label, folder in (("Models", MODEL_DIR), ("Results", RESULTS_DIR), ("Figures", FIGURES_DIR)):
        if os.path.exists(folder):
            files = sorted(f for f in os.listdir(folder) if FS_TAG in f)
            lines.append(f"\n{label} ({FS_TAG} files: {len(files)})")
            lines += [f"  {f}  ({os.path.getsize(os.path.join(folder, f)):,} bytes)" for f in files]
    return "\n".join(lines)

# ============================================================
# GRADIO INTERFACE
# ============================================================
with gr.Blocks(title=f"MANIS EC ML Development Agent ({FS_NAME})", theme=gr.themes.Soft()) as app:

    gr.Markdown(f"# 🤖 MANIS: EC ML Development Agent ({FS_NAME})")
    gr.Markdown(
        "Conventional five parameters plus nanoparticle conductivity and density, and base "
        "fluid conductivity, permittivity and viscosity. Trains on log10(σ_nf); metrics "
        "reported on back-transformed raw S/m.  \n"
        f"Folder: `{DRIVE_DIR}`"
    )

    results_state, metrics_state, data_arrays_state = gr.State(None), gr.State(None), gr.State(None)

    with gr.Tabs():
        with gr.Tab("1: Load & Train"):
            gr.Markdown("### Step 1: Load Feature Dataset")
            load_btn    = gr.Button("📂 Load Dataset", variant="secondary")
            load_status = gr.Textbox(label="Load Status", interactive=False, lines=8)

            gr.Markdown("### Step 2: AI Pre-Training Review")
            pretrain_btn    = gr.Button("🤖 Generate Pre-Training AI Review", variant="secondary")
            pretrain_status = gr.Textbox(label="Pre-Training AI Recommendation",
                                         interactive=False, lines=14)

            gr.Markdown("### Step 3: Train All Models")
            train_btn    = gr.Button("▶  Train All Models", variant="primary", size="lg")
            train_status = gr.Textbox(label="Training Log", interactive=False, lines=8)
            metrics_tbl  = gr.Dataframe(label=f"📊 Model Metrics: {FS_NAME}, sorted by R², raw S/m",
                                        interactive=False, wrap=True)

        with gr.Tab("2: AI Interpretation"):
            posttrain_btn    = gr.Button("🤖 Generate Post-Training Interpretation",
                                         variant="primary", size="lg")
            posttrain_status = gr.Textbox(label="Post-Training AI Interpretation",
                                          interactive=False, lines=26)

        with gr.Tab("3: Figures"):
            figs_btn    = gr.Button("📈 Generate All Figures", variant="primary", size="lg")
            figs_status = gr.Textbox(label="Status", interactive=False, lines=7)
            with gr.Row():
                fig1_out = gr.Image(label="Fig 1: Predicted vs Experimental (log space)")
                fig2_out = gr.Image(label="Fig 2: Metrics Comparison")
            with gr.Row():
                fig3_out = gr.Image(label="Fig 3: Feature Importance")
                fig4_out = gr.Image(label="Fig 4: Residual Distribution")
            fig5_out = gr.Image(label="Fig 5: Error vs Volume Fraction & Temperature")

        with gr.Tab("4: Google Drive"):
            with gr.Row():
                mount_btn = gr.Button("Mount / Remount Drive", variant="secondary")
                check_btn = gr.Button("Check Drive Status", variant="secondary")
            drive_status = gr.Textbox(label="Status", interactive=False, lines=14)
            mount_btn.click(lambda: (mount_drive(), check_drive_status())[1], outputs=[drive_status])
            check_btn.click(check_drive_status, outputs=[drive_status])

    # ── CALLBACKS ─────────────────────────────────────────────
    def do_load():
        try:
            r = load_data()
            if r[0] is None:
                return r[9], None
            keys = ["X_train", "X_test", "y_train", "y_test", "X_train_sc", "X_test_sc",
                    "scaler", "df", "test_df", "_msg", "y_train_raw", "y_test_raw"]
            data = dict(zip(keys, r))
            return r[9], data
        except Exception as e:
            return f"❌ Error loading data: {e}", None

    def do_pretraining_review(d):
        if d is None:
            return "❌ Load data first (Step 1)."
        try:
            notes = ai_pretraining_analysis(d["df"], d["X_train"], d["X_test"],
                                            d["y_train"], d["y_test"])
            with open(PRETRAIN_TXT, "w") as f:
                f.write(notes)
            return notes
        except Exception as e:
            return f"❌ Pre-training review error: {e}"

    def do_train(d):
        if d is None:
            return "❌ Load data first.", pd.DataFrame(), None, None
        try:
            results, metrics_df = train_all_models(d["X_train"], d["X_test"], d["y_train"],
                                                   d["y_test"], d["X_train_sc"], d["X_test_sc"],
                                                   d["y_test_raw"])
            save_artifacts(results, d["scaler"], d["test_df"], metrics_df)
            best = metrics_df.iloc[0]
            status = (f"✅ Training complete.\nBest model: {best['Model']}  "
                      f"R²={best['R²']}  MAPE={best['MAPE %']}%\n"
                      f"Models → {MODEL_DIR}\nTest set → {TEST_CSV.name}\n"
                      f"Metrics → {METRICS_CSV.name}")
            return status, metrics_df, results, metrics_df
        except Exception as e:
            return f"❌ Training error: {e}", pd.DataFrame(), None, None

    def do_posttraining_interpretation(results, metrics_df):
        if results is None or metrics_df is None or len(metrics_df) == 0:
            return "❌ Train models first (Step 3)."
        try:
            notes = ai_posttraining_analysis(metrics_df, results)
            with open(POSTTRAIN_TXT, "w") as f:
                f.write(notes)
            return notes
        except Exception as e:
            return f"❌ Post-training interpretation error: {e}"

    def do_figures(results, metrics_df, d):
        if results is None or d is None:
            return ("❌ Train models first.",) + (None,) * 5
        try:
            p1 = fig_predicted_vs_actual(results, d["y_test"])
            p2 = fig_metrics_comparison(metrics_df)
            p3 = fig_feature_importance(results, d["y_test"], d["X_test_sc"])
            p4 = fig_residual_distribution(results, d["y_test"])
            p5 = fig_error_vs_features(results, d["X_test"], d["y_test"])
            status = f"✅ 5 figures saved to {FIGURES_DIR}\n" + "\n".join(
                f"  {os.path.basename(p)}" for p in (p1, p2, p3, p4, p5))
            return status, p1, p2, p3, p4, p5
        except Exception as e:
            return (f"❌ Figure error: {e}",) + (None,) * 5

    load_btn.click(do_load, outputs=[load_status, data_arrays_state])
    pretrain_btn.click(do_pretraining_review, inputs=[data_arrays_state], outputs=[pretrain_status])
    train_btn.click(do_train, inputs=[data_arrays_state],
                    outputs=[train_status, metrics_tbl, results_state, metrics_state])
    posttrain_btn.click(do_posttraining_interpretation, inputs=[results_state, metrics_state],
                        outputs=[posttrain_status])
    figs_btn.click(do_figures, inputs=[results_state, metrics_state, data_arrays_state],
                   outputs=[figs_status, fig1_out, fig2_out, fig3_out, fig4_out, fig5_out])

app.launch(share=True)
