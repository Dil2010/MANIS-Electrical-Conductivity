# ============================================================
# MANIS: ELECTRICAL CONDUCTIVITY
# PHYSICAL VALIDATION AI AGENT
# New Correlation (10-parameter) feature set
# ============================================================
# Reads  : nanofluid_ec_features.xlsx
#          ec_models/ec_newcorr_trained_models.pkl, ec_newcorr_scaler.pkl,
#          ec_newcorr_feature_cols.json
#          ec_results/ec_newcorr_test_set.csv (to mark held-out points)
# Writes : ec_results/physical_validation_ec_newcorr/
#            ec_newcorr_<analysis>_...png                (300 DPI)
#            selections_ec_newcorr.json                  (figure registry)
#            ec_newcorr_physical_validation_summary.csv  (MAPE + trend agreement)
#
# Analysis 1: two figures, sigma_nf vs volume fraction, all 5 models
# Analysis 2: one figure, two subplots, same nanofluid, two particle sizes
# Analysis 3: one figure, sigma_nf vs temperature, up to 3 volume fractions
# Each group is locked after use and cannot be reused in another analysis.
#
# Models were trained on log10(sigma_nf); predict_for_rows() back-transforms
# (10**pred), so every figure compares raw S/m against raw experimental data.
#
# CHANGES FROM PREVIOUS VERSION
#  1. DRIVE_DIR moved to MANIS_ELECTRICAL/; API key setup + get_client().
#  2. Experimental points are split into training (filled) and held-out
#     test (open) markers. Previously all points were plotted alike, although
#     most were seen during training.
#  3. Each figure reports per model MAPE on the plotted points and a trend
#     agreement score (share of intervals where the predicted slope sign
#     matches the experimental one), saved to a summary CSV.
#  4. Particle size candidates are now pairs that share at least 3 volume
#     fractions AT THE SAME temperature. Previously common volume fractions
#     were counted across all temperatures, and the fallback plotted all
#     temperatures on one vs-volume-fraction line. Only the two plotted
#     groups are locked (previously a triplet locked three but plotted two).
#  5. Model comparison and temperature scans count points at the plotted
#     condition, not across the whole group.
#  6. Temperature figure: model line styles now match the legend.
#  7. AI ranking output shows which candidate each reason refers to.
#  8. Selection registry records every figure's exact parameters, so the
#     same selections can be replicated in the Conventional (5) and
#     Shen et al. (9) agents for a like for like comparison.
#
# REVISION NOTE: this is the version used to produce Figs. 8 to 11 of the
# revised manuscript; the code is unchanged. Candidate systems are scanned
# deterministically (scan_for_*), and the AI ranking (ai_rank) receives only
# data-coverage information (volume fractions, temperature range, particle
# size contrast, number of held-out test points) plus the system identity;
# no model predictions or errors are passed to it. The researcher makes the
# final selection.
# ============================================================

# !pip install xgboost scikit-learn openpyxl gradio anthropic -q

import os
import re
import json
import pickle
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as fm
from pathlib import Path
import anthropic
import gradio as gr
from google.colab import drive

warnings.filterwarnings('ignore')

# ============================================================
# FEATURE SET IDENTITY
# ============================================================
FS_TAG  = "newcorr"
FS_NAME = "New Correlation (10-parameter)"
REPLICATE_FROM_TAGS = ["shen", "conv"]

# ============================================================
# PATHS
# ============================================================
DRIVE_DIR    = "/content/drive/MyDrive/MANIS_ELECTRICAL/"
FEATURE_XLS  = DRIVE_DIR + "nanofluid_ec_features.xlsx"
MODEL_DIR    = Path(DRIVE_DIR + "ec_models/")
RESULTS_ROOT = Path(DRIVE_DIR + "ec_results/")

def val_dir(tag=FS_TAG):
    return RESULTS_ROOT / f"physical_validation_ec_{tag}"

def registry_path(tag=FS_TAG):
    return val_dir(tag) / f"selections_ec_{tag}.json"

VAL_DIR       = val_dir()
TEST_CSV      = RESULTS_ROOT / f"ec_{FS_TAG}_test_set.csv"
SUMMARY_CSV   = VAL_DIR / f"ec_{FS_TAG}_physical_validation_summary.csv"
MODELS_PKL    = MODEL_DIR / f"ec_{FS_TAG}_trained_models.pkl"
SCALER_PKL    = MODEL_DIR / f"ec_{FS_TAG}_scaler.pkl"
FEATURES_JSON = MODEL_DIR / f"ec_{FS_TAG}_feature_cols.json"

def mount_drive():
    try:
        drive.mount('/content/drive', force_remount=False)
        os.makedirs(VAL_DIR, exist_ok=True)
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
    print(f"⚠️ Key looks invalid (length {len(_key)}). Figures work; AI ranking will not.")

CLAUDE_MODEL = "claude-sonnet-4-5"

def get_client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No ANTHROPIC_API_KEY in environment. Re-run the key setup.")
    return anthropic.Anthropic(api_key=key)

# ============================================================
# FONTS AND STYLE
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

MODEL_COLORS = {
    "ANN":               "#E63946",
    "Random Forest":     "#2A9D8F",
    "XGBoost":           "#E9C46A",
    "Gradient Boosting": "#457B9D",
    "SVR":               "#F4A261",
}
EXP_COLOR     = "#2C2C2A"
EXP_MARKER    = "^"
SCALED_MODELS = {"ANN", "SVR"}

FEATURE_COLS = [
    "volume_fraction", "temperature_K", "particle_size_nm",
    "nanoparticle_encoded", "base_fluid_encoded",
    "sigma_p", "rho_p", "sigma_bf", "eps_r", "mu_bf",
]
TARGET_COL = "sigma_nf"

NANOPARTICLE_DISPLAY = {
    "AL2O3": r"$\mathrm{Al_2O_3}$",
    "CUO":   "CuO",
    "TIO2":  r"$\mathrm{TiO_2}$",
    "ZNO":   "ZnO",
    "SIO2":  r"$\mathrm{SiO_2}$",
    "SIC":   "SiC",
    "CACO3": r"$\mathrm{CaCO_3}$",
}

def get_nanoparticle_display(code):
    key = re.sub(r'[^A-Z0-9]', '', str(code).upper())
    return NANOPARTICLE_DISPLAY.get(key, code)

def sanitize(text):
    return re.sub(r'[^0-9a-zA-Z]+', '_', str(text)).strip('_')

def fmt_num(value):
    return str(round(float(value), 2)).replace('.', '_')

def style_ax(ax, xlabel, ylabel, title=None):
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color='#000000', pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color='#000000')
    ax.set_ylabel(ylabel, fontsize=10, color='#000000')
    ax.tick_params(axis='both', labelsize=9, colors='#000000')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.grid(True, linestyle='--', alpha=0.4, color='#dddddd')
    ax.set_axisbelow(True)

# ============================================================
# SELECTION REGISTRY (replaces the plain locked-groups list)
# Each record: {"analysis", "figure", "group_ids", "temperature_K",
#               "volume_fractions", "file"}
# ============================================================
def load_registry(tag=FS_TAG):
    p = registry_path(tag)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []

def save_registry(records):
    os.makedirs(VAL_DIR, exist_ok=True)
    with open(registry_path(), "w") as f:
        json.dump(records, f, indent=2)

def add_record(record):
    recs = [r for r in load_registry()
            if not (r["analysis"] == record["analysis"] and r["figure"] == record["figure"])]
    recs.append(record)
    save_registry(recs)

def load_locked():
    return {int(g) for r in load_registry() for g in r["group_ids"]}

def reset_locked():
    if registry_path().exists():
        registry_path().unlink()
    return "✅ All locks reset. All groups available again."

def unlock_groups(text):
    try:
        ids = {int(x) for x in re.findall(r'\d+', text or "")}
    except Exception:
        ids = set()
    if not ids:
        return "Enter one or more group IDs, e.g. 3, 7"
    recs = load_registry()
    keep = [r for r in recs if not ids & set(r["group_ids"])]
    save_registry(keep)
    return f"✅ Removed {len(recs) - len(keep)} figure record(s) using groups {sorted(ids)}.\n\n" + get_locked_display()

def get_locked_display():
    recs = load_registry()
    if not recs:
        return "No groups locked yet."
    df, _ = load_features_df()
    lines = ["Figure registry (groups locked):"]
    for r in sorted(recs, key=lambda x: (x["analysis"], str(x["figure"]))):
        desc = []
        for gid in r["group_ids"]:
            if df is not None and (df["group_id"] == gid).any():
                row = df[df["group_id"] == gid].iloc[0]
                desc.append(f"G{gid} {row['nanoparticle']}/{row['base_fluid']}/{row['particle_size_nm']} nm")
            else:
                desc.append(f"G{gid}")
        cond = f"T={r['temperature_K']} K" if r.get("temperature_K") is not None else \
               f"φ={r.get('volume_fractions')}"
        lines.append(f"  {r['analysis']} fig {r['figure']}: {'; '.join(desc)} | {cond}")
    return "\n".join(lines)

# ============================================================
# DATA LOADING
# ============================================================
COL_MAP = {
    "Group\nID": "group_id", "Group ID": "group_id",
    "Sub\nGroup ID": "subgroup_id", "Sub Group ID": "subgroup_id",
    "Nanoparticle": "nanoparticle", "Base Fluid": "base_fluid",
    "d_p": "particle_size_nm", "φ": "volume_fraction", "T": "temperature_K",
    "σ_bf": "sigma_bf", "ε_r": "eps_r", "ρ_bf": "rho_bf", "υ": "upsilon",
    "σ_p": "sigma_p", "ρ_p": "rho_p", "μ_bf": "mu_bf", "µ_bf": "mu_bf",
    "NP\nEncoded": "nanoparticle_encoded", "NP Encoded": "nanoparticle_encoded",
    "BF\nEncoded": "base_fluid_encoded", "BF Encoded": "base_fluid_encoded",
    "σ_nf  ← TARGET": "sigma_nf",
    "log10(σ_nf)  ← TARGET": "log_sigma_nf",
}

def _row_keys(d):
    return list(zip(d["group_id"].astype(int),
                    d["volume_fraction"].astype(float).round(8),
                    d["temperature_K"].astype(float).round(3),
                    d[TARGET_COL].astype(float).map(lambda v: float(f"{v:.6e}"))))

def load_features_df():
    """Returns (df, msg). Adds is_test = row belongs to the held-out test set."""
    if not Path(FEATURE_XLS).exists():
        return None, "❌ nanofluid_ec_features.xlsx not found. Run the Feature Engineering Agent first."
    try:
        xl = pd.ExcelFile(FEATURE_XLS)
        sheet = next((s for s in xl.sheet_names if "Feature" in s and "Set" not in s),
                     xl.sheet_names[0])
        df = pd.read_excel(FEATURE_XLS, sheet_name=sheet, header=1, skiprows=[2])
        df = df.rename(columns=COL_MAP)
        required = FEATURE_COLS + [TARGET_COL, "group_id", "nanoparticle", "base_fluid"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, f"❌ Missing columns: {missing}"
        df = df.dropna(subset=required).reset_index(drop=True)
        df["group_id"] = df["group_id"].astype(int)
        df["T_round"] = df["temperature_K"].astype(float).round(0)

        if TEST_CSV.exists():
            test_keys = set(_row_keys(pd.read_csv(TEST_CSV)))
            df["is_test"] = [k in test_keys for k in _row_keys(df)]
            note = f" | held-out test rows matched: {int(df['is_test'].sum())}"
        else:
            df["is_test"] = False
            note = " | ⚠️ test set file not found, all points shown as training"
        return df, f"✅ Loaded {len(df)} rows, {df['group_id'].nunique()} groups{note}."
    except Exception as e:
        return None, f"❌ Error reading Excel: {e}"

def load_models():
    """Returns (models, scaler, error)."""
    for p in (MODELS_PKL, SCALER_PKL, FEATURES_JSON):
        if not p.exists():
            return None, None, f"Missing {p.name}. Run the EC ML Development Agent first."
    with open(MODELS_PKL, "rb") as f:
        models = pickle.load(f)
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_JSON) as f:
        saved = json.load(f)
    if saved != FEATURE_COLS:
        return None, None, f"Feature mismatch. Saved {saved}, expected {FEATURE_COLS}."
    return models, scaler, None

def predict_for_rows(sub, models, scaler):
    """Predict log10(sigma_nf) and back-transform to raw S/m."""
    X = sub[FEATURE_COLS].values.astype(float)
    Xs = scaler.transform(X)
    return {name: 10 ** m.predict(Xs if name in SCALED_MODELS else X)
            for name, m in models.items()}

# ============================================================
# QUANTITATIVE CHECKS
# ============================================================
def trend_agreement(y_exp, y_pred):
    """Share of consecutive intervals where the predicted slope sign
    matches the experimental slope sign (rows already sorted by x)."""
    de, dp = np.diff(np.asarray(y_exp, float)), np.diff(np.asarray(y_pred, float))
    mask = de != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(de[mask]) == np.sign(dp[mask])))

def summarize(sub, preds, analysis, figure, panel=""):
    y = sub[TARGET_COL].values
    rows = []
    for name, yp in preds.items():
        rows.append({
            "feature_set": FS_TAG, "analysis": analysis, "figure": figure, "panel": panel,
            "group_ids": ",".join(str(g) for g in sorted(sub["group_id"].unique())),
            "model": name, "n_points": len(sub), "n_test_points": int(sub["is_test"].sum()),
            "MAPE_%": round(float(np.mean(np.abs((y - yp) / (y + 1e-30))) * 100), 2),
            "trend_agreement": round(trend_agreement(y, yp), 3),
        })
    return rows

def save_summary(rows, analysis, figure):
    new = pd.DataFrame(rows)
    if SUMMARY_CSV.exists():
        old = pd.read_csv(SUMMARY_CSV)
        old = old[~((old["analysis"] == analysis) & (old["figure"].astype(str) == str(figure)))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(SUMMARY_CSV, index=False)

def summary_text(rows):
    lines = ["Model              MAPE %   Trend agreement"]
    for r in rows:
        tag = f" [{r['panel']}]" if r["panel"] else ""
        lines.append(f"{r['model'] + tag:<22}{r['MAPE_%']:>7}   {r['trend_agreement']}")
    return "\n".join(lines)

# ============================================================
# PLOTTING HELPERS
# ============================================================
def plot_experimental(ax, x, sub, marker=EXP_MARKER, label_prefix="Experimental"):
    train = ~sub["is_test"].values
    test = sub["is_test"].values
    if train.any():
        ax.scatter(np.asarray(x)[train], sub[TARGET_COL].values[train], color=EXP_COLOR,
                   marker=marker, s=80, zorder=5, label=f"{label_prefix} (training)")
    if test.any():
        ax.scatter(np.asarray(x)[test], sub[TARGET_COL].values[test], facecolors='none',
                   edgecolors=EXP_COLOR, linewidths=1.4, marker=marker, s=80, zorder=6,
                   label=f"{label_prefix} (held-out test)")

def save_fig(fig, fname):
    path = str(VAL_DIR / fname)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path

# ============================================================
# SCANNING
# ============================================================
def scan_for_model_comparison():
    df, msg = load_features_df()
    if df is None:
        return [], msg
    locked = load_locked()
    cands = []
    for gid in sorted(df["group_id"].unique()):
        if gid in locked:
            continue
        g = df[df["group_id"] == gid]
        per_T = g.groupby("T_round")["volume_fraction"].nunique()
        best_T, n_vf = per_T.idxmax(), int(per_T.max())
        if n_vf < 4:
            continue
        at_T = g[g["T_round"] == best_T]
        r = g.iloc[0]
        cands.append({
            "key": f"G{gid}", "group_id": int(gid),
            "nanoparticle": str(r["nanoparticle"]), "base_fluid": str(r["base_fluid"]),
            "particle_size": float(r["particle_size_nm"]), "temperature_K": float(best_T),
            "vf_points": n_vf, "test_points": int(at_T["is_test"].sum()),
            "vf_range_pct": f"{at_T['volume_fraction'].min()*100:.3g} to {at_T['volume_fraction'].max()*100:.3g}",
            "label": (f"G{gid}: {r['nanoparticle']} | {r['base_fluid']} | {r['particle_size_nm']} nm | "
                      f"{n_vf} φ at T={best_T:.0f} K | {int(at_T['is_test'].sum())} test pts"),
        })
    return cands, f"{msg}\nFound {len(cands)} available groups."

def scan_for_particle_size():
    df, msg = load_features_df()
    if df is None:
        return [], msg
    locked = load_locked()
    cands = []
    for (npart, bf), fam in df.groupby(["nanoparticle", "base_fluid"]):
        gids = [g for g in sorted(fam["group_id"].unique()) if g not in locked]
        size_of = fam.groupby("group_id")["particle_size_nm"].first()
        for a, b in itertools.combinations(gids, 2):
            if size_of[a] == size_of[b]:
                continue
            ga, gb = fam[fam["group_id"] == a], fam[fam["group_id"] == b]
            best_T, best_n = None, 0
            for T in sorted(set(ga["T_round"]) & set(gb["T_round"])):
                va = set(ga[ga["T_round"] == T]["volume_fraction"].round(8))
                vb = set(gb[gb["T_round"] == T]["volume_fraction"].round(8))
                if len(va & vb) > best_n:
                    best_T, best_n = T, len(va & vb)
            if best_n < 3:
                continue
            (g1, s1), (g2, s2) = sorted([(a, size_of[a]), (b, size_of[b])], key=lambda x: x[1])
            cands.append({
                "key": f"G{g1}-G{g2}", "group_ids": [int(g1), int(g2)],
                "sizes_nm": [float(s1), float(s2)], "size_ratio": round(float(s2 / s1), 2),
                "nanoparticle": str(npart), "base_fluid": str(bf),
                "temperature_K": float(best_T), "common_vf": int(best_n),
                "label": (f"{npart} / {bf}: {s1} nm (G{g1}) vs {s2} nm (G{g2}) | "
                          f"{best_n} common φ at T={best_T:.0f} K"),
            })
    return cands, f"{msg}\nFound {len(cands)} particle size pairs."

def scan_for_temperature():
    df, msg = load_features_df()
    if df is None:
        return [], msg
    locked = load_locked()
    cands = []
    for gid in sorted(df["group_id"].unique()):
        if gid in locked:
            continue
        g = df[df["group_id"] == gid]
        per_vf = g.groupby("volume_fraction")["T_round"].nunique()
        good = per_vf[per_vf >= 3]
        if g["T_round"].nunique() < 4 or good.empty:
            continue
        top_vfs = [float(v) for v in good.nlargest(3).index]
        r = g.iloc[0]
        cands.append({
            "key": f"G{gid}", "group_id": int(gid),
            "nanoparticle": str(r["nanoparticle"]), "base_fluid": str(r["base_fluid"]),
            "particle_size": float(r["particle_size_nm"]),
            "n_temps": int(g["T_round"].nunique()),
            "temp_range": f"{g['temperature_K'].min():.0f} to {g['temperature_K'].max():.0f} K",
            "volume_fractions": top_vfs, "test_points": int(g["is_test"].sum()),
            "label": (f"G{gid}: {r['nanoparticle']} | {r['base_fluid']} | {r['particle_size_nm']} nm | "
                      f"{g['T_round'].nunique()} temps ({g['temperature_K'].min():.0f} to "
                      f"{g['temperature_K'].max():.0f} K) | {len(top_vfs)} φ series"),
        })
    return cands, f"{msg}\nFound {len(cands)} available groups."

def ai_rank(cands, analysis_type):
    if not cands:
        return "No candidates available. Scan first."
    goals = {
        "model_comparison": "sigma_nf vs volume fraction model comparison. Prefer many volume "
                            "fractions, a wide range, held-out test points, and a scientifically "
                            "interesting nanofluid.",
        "particle_size": "particle size effect on sigma_nf. Prefer a large size ratio and many "
                         "common volume fractions at the same temperature.",
        "temperature": "temperature effect on sigma_nf. Prefer a wide temperature range and "
                       "several volume fraction series.",
    }
    slim = [{k: v for k, v in c.items() if k != "label"} for c in cands]
    prompt = f"""Rank these candidates for {goals[analysis_type]}

Candidates:
{json.dumps(slim, indent=2)}

Return ONLY a JSON list ranked best first, using each candidate's "key":
[{{"key": "...", "reason": "one sentence"}}]"""
    try:
        text = get_client().messages.create(
            model=CLAUDE_MODEL, max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        ).content[0].text.strip()
        m = re.search(r'\[.*\]', text, re.S)
        ranked = json.loads(m.group(0)) if m else []
        labels = {c["key"]: c["label"] for c in cands}
        return "\n\n".join(f"#{i+1}  {labels.get(r['key'], r['key'])}\n     {r['reason']}"
                           for i, r in enumerate(ranked)) or text
    except Exception as e:
        return f"❌ AI ranking error: {e}"

# ============================================================
# FIGURE GENERATORS (explicit parameters, so registry replays work)
# ============================================================
def make_model_comparison(group_id, temperature_K, fig_number, check_lock=True):
    models, scaler, err = load_models()
    if models is None:
        return None, f"❌ {err}"
    if check_lock and group_id in load_locked():
        return None, f"❌ Group {group_id} is locked. Choose another."
    df, msg = load_features_df()
    if df is None:
        return None, msg

    g = df[df["group_id"] == group_id]
    sub = g[g["T_round"] == round(temperature_K)].sort_values("volume_fraction")
    if len(sub) < 3:
        return None, f"❌ Not enough points for group {group_id} at T={temperature_K} K."
    r = g.iloc[0]
    preds = predict_for_rows(sub, models, scaler)
    vf_pct = sub["volume_fraction"].values * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_experimental(ax, vf_pct, sub)
    for name, yp in preds.items():
        ax.plot(vf_pct, yp, color=MODEL_COLORS.get(name, "#999"),
                linewidth=2.0, marker='o', markersize=5, label=name)
    style_ax(ax, "Volume Fraction (%)", "Electrical Conductivity (S/m)")
    ax.legend(fontsize=9, framealpha=0.9, edgecolor='#cccccc')
    plt.tight_layout()

    fname = (f"ec_{FS_TAG}_model_comparison_{fig_number}_{sanitize(r['nanoparticle'])}_"
             f"{sanitize(r['base_fluid'])}_{fmt_num(r['particle_size_nm'])}nm_"
             f"{fmt_num(temperature_K)}K_group{group_id}.png")
    path = save_fig(fig, fname)

    rows = summarize(sub, preds, "model_comparison", fig_number)
    save_summary(rows, "model_comparison", fig_number)
    add_record({"analysis": "model_comparison", "figure": fig_number,
                "group_ids": [int(group_id)], "temperature_K": float(temperature_K),
                "volume_fractions": None, "file": fname})
    return path, (f"✅ Figure {fig_number} saved, group {group_id} locked.\n"
                  f"{len(sub)} points ({int(sub['is_test'].sum())} held-out test)\n\n" + summary_text(rows))

def make_particle_size(group_ids, temperature_K, check_lock=True):
    models, scaler, err = load_models()
    if models is None:
        return None, f"❌ {err}"
    locked = load_locked()
    if check_lock and any(g in locked for g in group_ids):
        return None, f"❌ Groups {[g for g in group_ids if g in locked]} already locked."
    df, msg = load_features_df()
    if df is None:
        return None, msg

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    rows, sizes = [], []
    for ax, gid, panel in zip(axes, group_ids, ["a", "b"]):
        g = df[df["group_id"] == gid]
        sub = g[g["T_round"] == round(temperature_K)].sort_values("volume_fraction")
        if len(sub) < 2:
            plt.close(fig)
            return None, f"❌ Group {gid} has too few points at T={temperature_K} K."
        size_nm = float(g["particle_size_nm"].iloc[0])
        sizes.append(size_nm)
        preds = predict_for_rows(sub, models, scaler)
        vf_pct = sub["volume_fraction"].values * 100
        plot_experimental(ax, vf_pct, sub)
        for name, yp in preds.items():
            ax.plot(vf_pct, yp, color=MODEL_COLORS.get(name, "#999"),
                    linewidth=2.0, marker='o', markersize=4, label=name)
        style_ax(ax, "Volume Fraction (%)",
                 "Electrical Conductivity (S/m)" if panel == "a" else "",
                 title=f"({panel}) Particle size: {size_nm:g} nm")
        ax.legend(fontsize=8, framealpha=0.9, edgecolor='#cccccc')
        rows += summarize(sub, preds, "particle_size", 1, panel=f"{size_nm:g} nm")
    plt.tight_layout()

    r = df[df["group_id"] == group_ids[0]].iloc[0]
    fname = (f"ec_{FS_TAG}_particle_size_{sanitize(r['nanoparticle'])}_{sanitize(r['base_fluid'])}_"
             f"{fmt_num(sizes[0])}nm_vs_{fmt_num(sizes[1])}nm_{fmt_num(temperature_K)}K_"
             f"group{group_ids[0]}_{group_ids[1]}.png")
    path = save_fig(fig, fname)

    save_summary(rows, "particle_size", 1)
    add_record({"analysis": "particle_size", "figure": 1,
                "group_ids": [int(g) for g in group_ids], "temperature_K": float(temperature_K),
                "volume_fractions": None, "file": fname})
    return path, f"✅ Figure saved, groups {group_ids} locked.\n\n" + summary_text(rows)

def make_temperature(group_id, volume_fractions, check_lock=True):
    models, scaler, err = load_models()
    if models is None:
        return None, f"❌ {err}"
    if check_lock and group_id in load_locked():
        return None, f"❌ Group {group_id} is locked. Choose another."
    df, msg = load_features_df()
    if df is None:
        return None, msg

    g = df[df["group_id"] == group_id]
    r = g.iloc[0]
    vf_markers = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(9, 6))
    rows, used_vfs, any_test = [], [], False

    for i, vf in enumerate(volume_fractions[:3]):
        sub = g[np.isclose(g["volume_fraction"], vf, atol=1e-8)].sort_values("temperature_K")
        if len(sub) < 3:
            continue
        used_vfs.append(vf)
        mkr = vf_markers[i]
        any_test |= bool(sub["is_test"].any())
        plot_experimental(ax, sub["temperature_K"].values, sub, marker=mkr)
        preds = predict_for_rows(sub, models, scaler)
        for name, yp in preds.items():
            ax.plot(sub["temperature_K"], yp, color=MODEL_COLORS.get(name, "#999"),
                    linewidth=1.8, marker=mkr, markersize=3)
        rows += summarize(sub, preds, "temperature", 1, panel=f"φ={vf*100:.3g}%")

    if not used_vfs:
        plt.close(fig)
        return None, f"❌ No volume fraction in group {group_id} has 3 or more temperatures."

    handles = [mlines.Line2D([], [], color=EXP_COLOR, marker=vf_markers[i], linestyle='None',
                             markersize=7, label=f"φ = {vf*100:.3g}% experimental")
               for i, vf in enumerate(used_vfs)]
    if any_test:
        handles.append(mlines.Line2D([], [], markerfacecolor='none', markeredgecolor=EXP_COLOR,
                                     marker='o', linestyle='None', markersize=7,
                                     label="Open marker: held-out test"))
    handles += [mlines.Line2D([], [], color=c, linewidth=2, label=n)
                for n, c in MODEL_COLORS.items() if n in models]
    style_ax(ax, "Temperature (K)", "Electrical Conductivity (S/m)")
    ax.legend(handles=handles, fontsize=9, framealpha=0.9, edgecolor='#cccccc', ncol=2)
    plt.tight_layout()

    fname = (f"ec_{FS_TAG}_temperature_effect_{sanitize(r['nanoparticle'])}_{sanitize(r['base_fluid'])}_"
             f"{fmt_num(r['particle_size_nm'])}nm_{fmt_num(g['temperature_K'].min())}-"
             f"{fmt_num(g['temperature_K'].max())}K_group{group_id}.png")
    path = save_fig(fig, fname)

    save_summary(rows, "temperature", 1)
    add_record({"analysis": "temperature", "figure": 1, "group_ids": [int(group_id)],
                "temperature_K": None, "volume_fractions": [float(v) for v in used_vfs],
                "file": fname})
    return path, f"✅ Figure saved, group {group_id} locked.\n\n" + summary_text(rows)

# ============================================================
# REPLICATE SELECTIONS FROM ANOTHER FEATURE SET
# ============================================================
def replicate_from(tag):
    recs = load_registry(tag)
    if not recs:
        return (f"❌ No registry found at {registry_path(tag)}. That agent must use this "
                f"template (selection registry) for replication to work."), []
    out, paths = [f"Replicating {len(recs)} figure(s) from '{tag}' into '{FS_TAG}':"], []
    for r in recs:
        if r["analysis"] == "model_comparison":
            p, s = make_model_comparison(r["group_ids"][0], r["temperature_K"], r["figure"], check_lock=False)
        elif r["analysis"] == "particle_size":
            p, s = make_particle_size(r["group_ids"], r["temperature_K"], check_lock=False)
        else:
            p, s = make_temperature(r["group_ids"][0], r["volume_fractions"], check_lock=False)
        out.append(f"\n[{r['analysis']} fig {r['figure']}]\n{s}")
        if p:
            paths.append(p)
    return "\n".join(out), paths

# ============================================================
# DRIVE STATUS
# ============================================================
def check_drive():
    if not os.path.exists(DRIVE_DIR):
        return "❌ Drive not mounted. Click Mount Drive."
    lines = [f"✅ Drive mounted: {DRIVE_DIR}",
             f"Models found: {MODELS_PKL.exists()} | Test set found: {TEST_CSV.exists()}"]
    if VAL_DIR.exists():
        files = sorted(f for f in os.listdir(VAL_DIR) if (VAL_DIR / f).is_file())
        lines.append(f"\n{VAL_DIR.name}/ ({len(files)} files)")
        lines += [f"  {f}  ({(VAL_DIR / f).stat().st_size:,} bytes)" for f in files]
    lines.append(f"\nLocked groups: {sorted(load_locked()) or 'None'}")
    return "\n".join(lines)

def load_summary_table():
    return pd.read_csv(SUMMARY_CSV) if SUMMARY_CSV.exists() else pd.DataFrame()

# ============================================================
# GRADIO INTERFACE
# ============================================================
with gr.Blocks(title=f"MANIS EC Physical Validation Agent ({FS_NAME})",
               theme=gr.themes.Soft()) as val_app:

    gr.Markdown(f"# MANIS: EC Physical Validation Agent ({FS_NAME})")
    gr.Markdown("Validates the five models against experimental trends. Predictions are "
                "back-transformed to raw S/m. Filled markers are training points, open markers "
                "are held-out test points. **Each group can be used only once.**")

    mc_state, ps_state, tmp_state = gr.State([]), gr.State([]), gr.State([])

    def pick(label, cands):
        return next((c for c in (cands or []) if c["label"] == label), None)

    with gr.Tabs():

        with gr.Tab("1: Model Comparison"):
            with gr.Row():
                scan_mc_btn = gr.Button("Scan Candidates", variant="secondary")
                ai_mc_btn = gr.Button("AI Ranking", variant="secondary")
            scan_mc_status = gr.Textbox(label="Scan Status", interactive=False, lines=2)
            ai_mc_output = gr.Textbox(label="AI Recommendation", interactive=False, lines=10)

            gr.Markdown("#### Figure 1")
            dd_mc1 = gr.Dropdown(choices=[], label="Group for Figure 1")
            gen_mc1_btn = gr.Button("Generate Figure 1", variant="primary")
            st_mc1 = gr.Textbox(label="Status", interactive=False, lines=9)
            fig_mc1 = gr.Image(label="Model Comparison: Figure 1")

            gr.Markdown("#### Figure 2")
            dd_mc2 = gr.Dropdown(choices=[], label="Group for Figure 2")
            gen_mc2_btn = gr.Button("Generate Figure 2", variant="primary")
            st_mc2 = gr.Textbox(label="Status", interactive=False, lines=9)
            fig_mc2 = gr.Image(label="Model Comparison: Figure 2")

            def do_scan_mc():
                cands, msg = scan_for_model_comparison()
                ch = [c["label"] for c in cands]
                return msg, gr.update(choices=ch, value=None), gr.update(choices=ch, value=None), cands

            def do_gen_mc(label, cands, n):
                c = pick(label, cands)
                if c is None:
                    return None, "Scan and select a group first."
                return make_model_comparison(c["group_id"], c["temperature_K"], n)

            scan_mc_btn.click(do_scan_mc, outputs=[scan_mc_status, dd_mc1, dd_mc2, mc_state])
            ai_mc_btn.click(lambda c: ai_rank(c, "model_comparison"), inputs=[mc_state], outputs=[ai_mc_output])
            gen_mc1_btn.click(lambda l, c: do_gen_mc(l, c, 1), inputs=[dd_mc1, mc_state], outputs=[fig_mc1, st_mc1])
            gen_mc2_btn.click(lambda l, c: do_gen_mc(l, c, 2), inputs=[dd_mc2, mc_state], outputs=[fig_mc2, st_mc2])

        with gr.Tab("2: Particle Size Effect"):
            with gr.Row():
                scan_ps_btn = gr.Button("Scan Candidates", variant="secondary")
                ai_ps_btn = gr.Button("AI Ranking", variant="secondary")
            scan_ps_status = gr.Textbox(label="Scan Status", interactive=False, lines=2)
            ai_ps_output = gr.Textbox(label="AI Recommendation", interactive=False, lines=10)
            dd_ps = gr.Dropdown(choices=[], label="Particle size pair")
            gen_ps_btn = gr.Button("Generate Figure", variant="primary", size="lg")
            st_ps = gr.Textbox(label="Status", interactive=False, lines=14)
            fig_ps = gr.Image(label="Particle Size Effect")

            def do_scan_ps():
                cands, msg = scan_for_particle_size()
                return msg, gr.update(choices=[c["label"] for c in cands], value=None), cands

            def do_gen_ps(label, cands):
                c = pick(label, cands)
                if c is None:
                    return None, "Scan and select a pair first."
                return make_particle_size(c["group_ids"], c["temperature_K"])

            scan_ps_btn.click(do_scan_ps, outputs=[scan_ps_status, dd_ps, ps_state])
            ai_ps_btn.click(lambda c: ai_rank(c, "particle_size"), inputs=[ps_state], outputs=[ai_ps_output])
            gen_ps_btn.click(do_gen_ps, inputs=[dd_ps, ps_state], outputs=[fig_ps, st_ps])

        with gr.Tab("3: Temperature Effect"):
            with gr.Row():
                scan_tmp_btn = gr.Button("Scan Candidates", variant="secondary")
                ai_tmp_btn = gr.Button("AI Ranking", variant="secondary")
            scan_tmp_status = gr.Textbox(label="Scan Status", interactive=False, lines=2)
            ai_tmp_output = gr.Textbox(label="AI Recommendation", interactive=False, lines=10)
            dd_tmp = gr.Dropdown(choices=[], label="Group for temperature analysis")
            gen_tmp_btn = gr.Button("Generate Figure", variant="primary", size="lg")
            st_tmp = gr.Textbox(label="Status", interactive=False, lines=18)
            fig_tmp = gr.Image(label="Temperature Effect")

            def do_scan_tmp():
                cands, msg = scan_for_temperature()
                return msg, gr.update(choices=[c["label"] for c in cands], value=None), cands

            def do_gen_tmp(label, cands):
                c = pick(label, cands)
                if c is None:
                    return None, "Scan and select a group first."
                return make_temperature(c["group_id"], c["volume_fractions"])

            scan_tmp_btn.click(do_scan_tmp, outputs=[scan_tmp_status, dd_tmp, tmp_state])
            ai_tmp_btn.click(lambda c: ai_rank(c, "temperature"), inputs=[tmp_state], outputs=[ai_tmp_output])
            gen_tmp_btn.click(do_gen_tmp, inputs=[dd_tmp, tmp_state], outputs=[fig_tmp, st_tmp])

        with gr.Tab("4: Summary"):
            gr.Markdown(f"Per figure MAPE and trend agreement. Saved to `{SUMMARY_CSV.name}`.")
            sum_btn = gr.Button("Refresh Summary", variant="secondary")
            sum_tbl = gr.Dataframe(interactive=False, wrap=True)
            sum_btn.click(load_summary_table, outputs=[sum_tbl])

        with gr.Tab("5: Replicate Selections"):
            gr.Markdown("Regenerate the exact figures (same groups, temperatures, volume fractions) "
                        "chosen in another feature set's agent, for a like for like comparison.")
            rep_dd = gr.Dropdown(choices=REPLICATE_FROM_TAGS, value=REPLICATE_FROM_TAGS[0],
                                 label="Replicate from feature set")
            rep_btn = gr.Button("Replicate", variant="primary")
            rep_status = gr.Textbox(label="Status", interactive=False, lines=20)
            rep_gallery = gr.Gallery(label="Replicated figures", columns=2)
            rep_btn.click(replicate_from, inputs=[rep_dd], outputs=[rep_status, rep_gallery])

        with gr.Tab("6: Locked Groups"):
            with gr.Row():
                view_btn = gr.Button("View Registry", variant="secondary")
                reset_btn = gr.Button("Reset All Locks", variant="stop")
            with gr.Row():
                unlock_in = gr.Textbox(label="Unlock group IDs (e.g. 3, 7)")
                unlock_btn = gr.Button("Unlock", variant="secondary")
            locked_out = gr.Textbox(label="Registry", interactive=False, lines=15)
            view_btn.click(get_locked_display, outputs=[locked_out])
            reset_btn.click(reset_locked, outputs=[locked_out])
            unlock_btn.click(unlock_groups, inputs=[unlock_in], outputs=[locked_out])

        with gr.Tab("7: Google Drive"):
            gr.Markdown(f"Figures saved to: `{VAL_DIR}`")
            with gr.Row():
                mount_btn = gr.Button("Mount / Remount Drive", variant="secondary")
                check_btn = gr.Button("Check Drive Status", variant="secondary")
            drive_status = gr.Textbox(label="Status", interactive=False, lines=14)
            mount_btn.click(lambda: check_drive() if mount_drive() else "❌ Mount failed.",
                            outputs=[drive_status])
            check_btn.click(check_drive, outputs=[drive_status])

val_app.launch(share=True)
