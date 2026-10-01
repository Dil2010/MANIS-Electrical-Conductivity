# ============================================================
# MANIS — ELECTRICAL CONDUCTIVITY
# COLAB — CELL: FEATURE ENGINEERING AI AGENT
# ============================================================
# Reads  : nanofluid_ec_data_clean.csv
# Adds   : sigma_p, rho_p, sigma_bf, eps_r, mu_bf, rho_bf, upsilon
#          + three named feature-set column groups
# Writes : nanofluid_ec_features.xlsx           (ML training input)
#          ec_feature_engineering_ai_notes.txt  (AI justification)
#
# THREE FEATURE SETS benchmarked side by side:
#   1. Conventional (5)      — nanoparticle, base fluid, d_p, phi, T
#   2. Shen et al. (trimmed) — Conventional + sigma_bf, eps_r, rho_bf, upsilon
#                              (U0, lambda, T0 dropped — zero variance,
#                               fixed constants specific to one ZnO/oil fit,
#                               not usable as ML training columns; once
#                               dropped, the remaining Shen features are
#                               generic fluid properties, so this set is
#                               NOT restricted to ZnO rows)
#   3. New Correlation (trimmed) — Conventional + sigma_p, rho_p, sigma_bf,
#                              eps_r, mu_bf
#                              (U0, h, psi dropped for the same reason —
#                               U0 has some cross-material variance but was
#                               removed per final agreed parameter list)
#
# Target: sigma_nf (S/m), plus log10(sigma_nf) — the dataset spans ~10
# orders of magnitude, so log-scale target is provided for ML training
# alongside the raw value; the ML Development Agent should train primarily
# on log_sigma_nf given this range.
#
# AI step: justifies each feature set by tying columns back to the
# specific term in the corresponding model, and flags any feature with
# weak variance/correlation before training.
#
# REVISION NOTE: this code is unchanged from the version used to build
# the feature file. sigma_p, rho_p and eps_r take one fixed value per
# nanoparticle or base fluid, so they largely re-encode the category
# labels; sigma_bf equals the target on the base fluid (phi = 0) rows.
# The redundancy checks reviewed by the AI step are those listed in
# its prompt (section 3 of ai_analyze_features).
# ============================================================

import os
import re
import json
import numpy as np
import pandas as pd
import anthropic
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from google.colab import drive

# ── Google Drive ──────────────────────────────────────────────
DRIVE_DIR      = "/content/drive/MyDrive/MANIS_ELECTRICAL/"
CLEAN_CSV    = DRIVE_DIR + "nanofluid_ec_data_clean.csv"
FEATURE_XLS  = DRIVE_DIR + "nanofluid_ec_features.xlsx"
AI_NOTES_TXT = DRIVE_DIR + "ec_feature_engineering_ai_notes.txt"

def mount_drive():
    try:
        drive.mount('/content/drive', force_remount=False)
        os.makedirs(DRIVE_DIR, exist_ok=True)
        return True
    except Exception as e:
        print(f"Drive mount failed: {e}")
        return False

mount_drive()

# ── API KEY SETUP (fix for AuthenticationError) ─────────────────
# Preferred: store the key once in Colab's Secrets manager (key icon,
# left sidebar), named ANTHROPIC_API_KEY, and toggle "Notebook access" on.
# This survives across sessions, unlike os.environ set manually each time.
try:
    from google.colab import userdata
    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
    print("✅ API key loaded from Colab Secrets.")
except Exception as e:
    # Fallback: paste your key directly here for this session only.
    # Remove/blank this before sharing the notebook with anyone.
    manual_key = ""  # ← paste your key between the quotes if Secrets isn't set up
    if manual_key:
        os.environ["ANTHROPIC_API_KEY"] = manual_key
        print("✅ API key set manually for this session.")
    else:
        print("⚠️  No API key found via Secrets, and no manual key provided.")
        print("   Set ANTHROPIC_API_KEY in Colab Secrets, or paste it into")
        print("   the 'manual_key' variable above, before running this cell.")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Physical constants ─────────────────────────────────────────
EPS_0 = 8.8541878128e-12   # Vacuum permittivity, F/m (kept for reference;
                           # not needed directly in feature computation)

# ============================================================
# NANOPARTICLE ELECTRICAL PROPERTY LOOKUP
# sigma_p : bulk electrical conductivity, S/m  — LOW CONFIDENCE overall,
#           literature spread spans many orders of magnitude for these
#           oxides depending on crystallinity/doping/impurity; point
#           estimates only, not measurements.
# rho_p   : density, kg/m³ — HIGH CONFIDENCE, standard literature values
# ============================================================
NP_EC_PROPS = {
    # Source: Shen et al. (2012), Physics Letters A 376, 1053-1057 — DIRECT CITATION
    "ZNO":   {"sigma_p": 1.62e-6,  "rho_p": 5600},
    "AL2O3": {"sigma_p": 1e-11,    "rho_p": 3970},
    "TIO2":  {"sigma_p": 1e-7,     "rho_p": 4230},
    "CUO":   {"sigma_p": 1e-1,     "rho_p": 6310},
    "SIO2":  {"sigma_p": 1e-11,    "rho_p": 2200},
    "SIC":   {"sigma_p": 1e-3,     "rho_p": 3160},
    # Source: Dong et al. (2013), J. Nanomaterials 2013, 842963 — DIRECT CITATION
    "ALN":   {"sigma_p": 1e-12,    "rho_p": 3260},
    # sigma_p: CaCO3 (calcite) is a well-established electrical insulator;
    # no precise bulk conductivity figure found in literature, estimated
    # as class-consistent with other insulating oxides (Al2O3/SiO2) —
    # LOW CONFIDENCE. rho_p = 2700 kg/m3 is the standard calcite density,
    # confirmed directly in literature — HIGH CONFIDENCE.
    "CACO3": {"sigma_p": 1e-10,    "rho_p": 2700},
}

# ============================================================
# BASE FLUID PROPERTY FUNCTIONS
# Returns (eps_r, mu_bf [Pa·s], rho_bf [kg/m³])
# ============================================================

def props_water(T):
    eps_r = 80.0
    mu    = 2.414e-5 * 10**(247.8/(T - 140))
    rho   = 999.842594 + 6.793952e-2*(T-273.15) - 9.09529e-3*(T-273.15)**2
    return eps_r, mu, rho

def props_EG(T):
    eps_r = 37.0
    mu    = np.exp(-12.4936 + 1702.927/(T - 96.475))
    rho   = 1113.3 - 0.6457*(T - 293.15)
    return eps_r, mu, rho

def props_transformer_oil(T):
    """eps_r matches Shen et al./Dong et al.'s own insulated-oil value (2.2).
    mu, rho fitted from Patel et al. (2009) Table 1 — HIGH CONFIDENCE."""
    Tc    = T - 273.15
    eps_r = 2.2
    rho   = 879.0 - 0.6*(Tc - 20.0)
    mu    = 0.0211 * np.exp(-0.0431*(Tc - 20.0))
    return eps_r, mu, rho

def props_peg200(T):
    """PEG (MW~200). eps_r sourced from patent literature. mu, rho are
    LOW CONFIDENCE literature-range estimates."""
    Tc    = T - 273.15
    eps_r = 19.0
    mu    = 0.055 * np.exp(-0.02*(Tc - 20.0))
    rho   = 1125.0 - 0.6*(Tc - 20.0)
    return eps_r, mu, rho

def props_bio_glycol(T):
    """Treated as bio-based 1,2-propanediol equivalent — no distinct bulk
    constants exist for "bio glycol" specifically. LOW CONFIDENCE."""
    Tc    = T - 273.15
    eps_r = 28.0
    mu    = 0.048 * np.exp(-0.025*(Tc - 20.0))
    rho   = 1036.0 - 0.6*(Tc - 20.0)
    return eps_r, mu, rho

_PURE_FLUID_FN = {
    "water":            props_water,
    "eg":               props_EG,
    "ethylene glycol":  props_EG,
    "pg":               props_bio_glycol,
    "propylene glycol": props_bio_glycol,
}

def parse_mixture_ratio(name):
    """Parses 'Water:EG 90:10' or 'PG:Water 60:40' into
    (component_a, component_b, frac_a, frac_b)."""
    m = re.match(r"^\s*([A-Za-z0-9]+)\s*:\s*([A-Za-z0-9]+)\s+(\d+)\s*:\s*(\d+)\s*$", name)
    if not m:
        return None
    comp_a, comp_b, ratio_a, ratio_b = m.groups()
    frac_a = float(ratio_a) / (float(ratio_a) + float(ratio_b))
    frac_b = 1.0 - frac_a
    return comp_a.strip().lower(), comp_b.strip().lower(), frac_a, frac_b

def props_mixture(T, name):
    """Linear interpolation of (eps_r, mu_bf, rho_bf) between two pure
    fluids by volume ratio. MODERATE CONFIDENCE — standard first-order
    approximation, not perfectly accurate for real mixtures."""
    parsed = parse_mixture_ratio(name)
    if parsed is None:
        return None
    comp_a, comp_b, frac_a, frac_b = parsed
    fn_a = _PURE_FLUID_FN.get(comp_a)
    fn_b = _PURE_FLUID_FN.get(comp_b)
    if fn_a is None or fn_b is None:
        return None
    eps_a, mu_a, rho_a = fn_a(T)
    eps_b, mu_b, rho_b = fn_b(T)
    eps_r = frac_a*eps_a + frac_b*eps_b
    mu    = frac_a*mu_a  + frac_b*mu_b
    rho   = frac_a*rho_a + frac_b*rho_b
    return eps_r, mu, rho

BASE_FLUID_EC_FN = {
    "Water":            props_water,
    "Ethylene Glycol":  props_EG,
    "Transformer Oil":  props_transformer_oil,
    "Peg200":           props_peg200,
    "Bio Glycol":       props_bio_glycol,
}

def get_base_fluid_props(T, base_fluid_name):
    fn = BASE_FLUID_EC_FN.get(base_fluid_name)
    if fn is not None:
        return fn(T)
    mix = props_mixture(T, base_fluid_name)
    if mix is not None:
        return mix
    return np.nan, np.nan, np.nan

# ============================================================
# sigma_bf EXTRACTION FROM DATA
# ============================================================

def extract_sigma_bf_per_group(df):
    """For each (group_id, temperature_K), pulls sigma_bf from that
    group's own volume_fraction==0 row (uS/cm -> S/m). Falls back to
    the Dong et al. (2013) literature value for groups with NO phi=0
    row at ANY temperature (Group 13: ALN/Transformer Oil in this
    dataset), applied uniformly across whatever temperatures that
    group has — flagged as a literature estimate, not independently
    measured at each T."""
    UNIT_CONV = 1e-4
    sigma_bf_map = {}
    for gid, grp in df.groupby("group_id"):
        zero_rows = grp[grp["volume_fraction"] == 0.0]
        if len(zero_rows) > 0:
            for _, r in zero_rows.iterrows():
                sigma_bf_map[(gid, r["temperature_K"])] = r["electrical_conductivity_uScm"] * UNIT_CONV
        else:
            fallback_sigma_bf_Sm = 1.52e-12   # 1.52 pS/m, Dong et al. (2013)
            for T_val in grp["temperature_K"].unique():
                sigma_bf_map[(gid, T_val)] = fallback_sigma_bf_Sm
    return sigma_bf_map

# ============================================================
# FEATURE COMPUTATION
# ============================================================

def compute_features(df):
    """Adds sigma_p, rho_p, sigma_bf, eps_r, mu_bf, rho_bf, upsilon,
    log_sigma_nf + label encodings to dataframe."""
    df = df.copy()

    # sigma_bf per group/temperature, pulled from the data itself
    sigma_bf_map = extract_sigma_bf_per_group(df)
    df["sigma_bf"] = df.apply(
        lambda r: sigma_bf_map.get((r["group_id"], r["temperature_K"]), np.nan), axis=1
    )

    # Target: measured sigma_nf, converted to S/m, plus log10
    df["sigma_nf"] = df["electrical_conductivity_uScm"] * 1e-4
    with np.errstate(divide='ignore'):
        df["log_sigma_nf"] = np.where(df["sigma_nf"] > 0, np.log10(df["sigma_nf"]), np.nan)

    # Nanoparticle properties (case-insensitive)
    sigma_p_l, rho_p_l = [], []
    unknown_np = set()
    for _, row in df.iterrows():
        key = str(row["nanoparticle"]).strip().upper()
        props = NP_EC_PROPS.get(key, {})
        if not props:
            unknown_np.add(row["nanoparticle"])
        sigma_p_l.append(props.get("sigma_p", np.nan))
        rho_p_l.append(props.get("rho_p", np.nan))
    df["sigma_p"] = sigma_p_l
    df["rho_p"]   = rho_p_l

    # Base fluid properties
    eps_l, mu_l, rho_l = [], [], []
    unknown_bf = set()
    for _, row in df.iterrows():
        bf_name = str(row["base_fluid"]).strip()
        eps_r, mu_bf, rho_bf = get_base_fluid_props(row["temperature_K"], bf_name)
        if np.isnan(eps_r):
            unknown_bf.add(bf_name)
        eps_l.append(eps_r); mu_l.append(mu_bf); rho_l.append(rho_bf)
    df["eps_r"]  = eps_l
    df["mu_bf"]  = mu_l
    df["rho_bf"] = rho_l

    # Kinematic viscosity (Shen et al. feature set only)
    df["upsilon"] = df["mu_bf"] / df["rho_bf"]

    df["d_p_m"] = df["particle_size_nm"] * 1e-9

    # ── Label encoding ───────────────────────────────────────
    le_np = LabelEncoder()
    le_bf = LabelEncoder()
    df["nanoparticle_encoded"] = le_np.fit_transform(df["nanoparticle"].astype(str))
    df["base_fluid_encoded"]   = le_bf.fit_transform(df["base_fluid"].astype(str))

    print("\n📋 Nanoparticle encoding map:")
    for cls, idx in zip(le_np.classes_, le_np.transform(le_np.classes_)):
        print(f"   {idx:>3}  →  {cls}")
    print("\n📋 Base fluid encoding map:")
    for cls, idx in zip(le_bf.classes_, le_bf.transform(le_bf.classes_)):
        print(f"   {idx:>3}  →  {cls}")

    return df, unknown_np, unknown_bf, le_np, le_bf

# ============================================================
# FEATURE SET DEFINITIONS
# ============================================================
CONVENTIONAL_COLS = ["particle_size_nm", "volume_fraction", "temperature_K",
                     "nanoparticle_encoded", "base_fluid_encoded"]

SHEN_EXTRA_COLS   = ["sigma_bf", "eps_r", "rho_bf", "upsilon"]
SHEN_COLS         = CONVENTIONAL_COLS + SHEN_EXTRA_COLS

NEWCORR_EXTRA_COLS = ["sigma_p", "rho_p", "sigma_bf", "eps_r", "mu_bf"]
NEWCORR_COLS       = CONVENTIONAL_COLS + NEWCORR_EXTRA_COLS

TARGET_COLS = ["sigma_nf", "log_sigma_nf"]

# ============================================================
# AI RECOMMENDATION STEP
# ============================================================

def ai_analyze_features(df):
    def feature_stats(cols):
        stats = {}
        for col in cols:
            if col in df.columns:
                valid = df[[col, "log_sigma_nf"]].dropna()
                corr  = float(valid.corr().iloc[0, 1]) if len(valid) > 2 else None
                stats[col] = {
                    "variance":         float(df[col].var(skipna=True)),
                    "corr_with_target": corr,
                    "missing":          int(df[col].isna().sum()),
                }
        return stats

    stats_conventional = feature_stats(CONVENTIONAL_COLS)
    stats_shen         = feature_stats(SHEN_EXTRA_COLS)
    stats_newcorr       = feature_stats(NEWCORR_EXTRA_COLS)

    prompt = f"""You are an expert in nanofluid electrical transport and feature engineering for machine learning.

Three feature sets are being benchmarked for predicting electrical conductivity
(target: log10(sigma_nf), since sigma_nf spans about 10 orders of magnitude
across this dataset):

1. CONVENTIONAL (5 parameters): nanoparticle type, base fluid type, particle
   size (d_p), volume fraction (phi), temperature (T).

2. SHEN ET AL. (trimmed, 4 extra beyond conventional): base fluid electrical
   conductivity (sigma_bf), base fluid relative permittivity (eps_r), base
   fluid density (rho_bf), base fluid kinematic viscosity (upsilon). Note:
   the original Shen et al. (2012) model also used Zeta potential (U0),
   a viscosity decay index (lambda), and a reference temperature (T0), but
   these were fixed constants specific to one ZnO-in-oil experiment with
   ZERO variance across the dataset and were dropped as ML features for
   that reason.

3. NEW CORRELATION (trimmed, 5 extra beyond conventional): nanoparticle
   electrical conductivity (sigma_p), nanoparticle density (rho_p), base
   fluid electrical conductivity (sigma_bf), base fluid relative
   permittivity (eps_r), base fluid viscosity (mu_bf). Zeta potential (U0),
   nanolayer thickness (h), and sphericity (psi) were dropped from this
   feature set per the same reasoning or by explicit decision.

Conventional feature diagnostics (variance, correlation with log_sigma_nf, missing count):
{json.dumps(stats_conventional, indent=2)}

Shen et al. EXTRA feature diagnostics:
{json.dumps(stats_shen, indent=2)}

New Correlation EXTRA feature diagnostics:
{json.dumps(stats_newcorr, indent=2)}

Structure your response in exactly this order, using these section headers:

## 1. OVERALL FEATURE SET ASSESSMENT
Give a brief, quantitative comparison of how well each of the three feature
sets is supported by the data (variance and correlation ranges, missing-value
rate), and state which feature set looks best posed for training based on
diagnostics alone (before any model is even trained).

## 2. FEATURE-BY-FEATURE JUSTIFICATION
For each EXTRA feature in the Shen et al. and New Correlation sets
individually, justify its inclusion by tying it back to the physical role
it plays (e.g. base fluid conductivity sets the baseline being enhanced,
permittivity and viscosity control the electrophoretic/Brownian
contributions), and cite its variance and correlation with log_sigma_nf.

## 3. WEAK OR REDUNDANT FEATURES
Flag any feature with unexpectedly low variance or low correlation with
the target, and note if this is expected given the dataset's composition
(e.g. sigma_bf and eps_r may correlate heavily with base_fluid_encoded
since they are largely determined by which fluid was used, or with
temperature_K if only one base fluid is present per nanoparticle group).

## 4. DATA READINESS AND NEXT STEPS
Note any missing-value concerns that should be resolved before training,
and confirm whether all three feature sets are ready to be passed to the
ML Development Agent for the 5-algorithm benchmark (Random Forest,
Gradient Boosting, SVR, ANN, XGBoost).

Be specific, use the actual numbers, and keep each section concise (3-4 sentences)."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1400,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ============================================================
# EXCEL WRITER
# ============================================================

def write_excel(df, path):
    wb = Workbook()
    ws = wb.active
    ws.title = "EC Features"

    RAW_COLS_DISPLAY = [
        ("group_id",             "Group\nID",        "—"),
        ("subgroup_id",          "Sub\nGroup ID",    "—"),
        ("nanoparticle",         "Nanoparticle",     ""),
        ("base_fluid",           "Base Fluid",       ""),
        ("particle_size_nm",     "d_p",              "nm"),
        ("volume_fraction",      "φ",                "—"),
        ("temperature_K",        "T",                "K"),
    ]
    SHEN_EXTRA_DISPLAY = [
        ("sigma_bf",  "σ_bf",   "S/m"),
        ("eps_r",     "ε_r",    "—"),
        ("rho_bf",    "ρ_bf",   "kg/m³"),
        ("upsilon",   "υ",      "m²/s"),
    ]
    NEWCORR_EXTRA_DISPLAY = [
        ("sigma_p",   "σ_p",    "S/m"),
        ("rho_p",     "ρ_p",    "kg/m³"),
        ("mu_bf",     "μ_bf",   "Pa·s"),
    ]
    IDENTITY_DISPLAY = [
        ("nanoparticle_encoded", "NP\nEncoded", "int"),
        ("base_fluid_encoded",   "BF\nEncoded", "int"),
    ]
    TARGET_DISPLAY = [
        ("sigma_nf",      "σ_nf  ← TARGET",       "S/m"),
        ("log_sigma_nf",  "log10(σ_nf)  ← TARGET", "—"),
    ]

    seen = set()
    COLS = []
    for c, h, u in RAW_COLS_DISPLAY:
        COLS.append((c, h, u)); seen.add(c)
    for c, h, u in SHEN_EXTRA_DISPLAY:
        if c not in seen:
            COLS.append((c, h, u)); seen.add(c)
    for c, h, u in NEWCORR_EXTRA_DISPLAY:
        if c not in seen:
            COLS.append((c, h, u)); seen.add(c)
    for c, h, u in IDENTITY_DISPLAY:
        COLS.append((c, h, u))
    for c, h, u in TARGET_DISPLAY:
        COLS.append((c, h, u))

    COLS = [(c,h,u) for c,h,u in COLS if c in df.columns]
    n_cols = len(COLS)

    def fill(hex_):   return PatternFill("solid", fgColor=hex_)
    def font_(bold=False, color="000000", size=10, italic=False):
        return Font(name="Arial", bold=bold, color=color, size=size, italic=italic)
    thin   = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    FILLS = {
        "raw":      fill("2A6EBB"),
        "feature":  fill("1A6B3C"),
        "target":   fill("8B4513"),
        "row_a":    fill("FFFFFF"),
        "row_b":    fill("F5F5F5"),
        "feat_a":   fill("EBF7EE"),
        "feat_b":   fill("D6EFD9"),
        "tgt_a":    fill("FFF8E1"),
        "tgt_b":    fill("FFF3CD"),
    }

    RAW_COLS   = {c for c,_,_ in RAW_COLS_DISPLAY}
    NEW_COLS   = {"sigma_bf","eps_r","rho_bf","upsilon","sigma_p","rho_p","mu_bf",
                  "nanoparticle_encoded","base_fluid_encoded"}
    TARGET_SET = {"sigma_nf","log_sigma_nf"}

    raw_end  = sum(1 for c,_,_ in COLS if c in RAW_COLS)
    feat_end = raw_end + sum(1 for c,_,_ in COLS if c in NEW_COLS)
    last_col = n_cols

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=raw_end)
    ws.cell(1,1).value = "EXPERIMENTAL DATA"
    ws.cell(1,1).font = font_(bold=True, color="FFFFFF", size=9)
    ws.cell(1,1).fill = FILLS["raw"]
    ws.cell(1,1).alignment = center

    ws.merge_cells(start_row=1, start_column=raw_end+1, end_row=1, end_column=feat_end)
    ws.cell(1, raw_end+1).value = "PHYSICS-INFORMED FEATURES (Shen et al. + New Correlation, deduplicated)  |  Identity Encodings"
    ws.cell(1, raw_end+1).font = font_(bold=True, color="FFFFFF", size=9)
    ws.cell(1, raw_end+1).fill = FILLS["feature"]
    ws.cell(1, raw_end+1).alignment = center

    ws.merge_cells(start_row=1, start_column=feat_end+1, end_row=1, end_column=last_col)
    ws.cell(1, feat_end+1).value = "TARGETS"
    ws.cell(1, feat_end+1).font = font_(bold=True, color="FFFFFF", size=9)
    ws.cell(1, feat_end+1).fill = FILLS["target"]
    ws.cell(1, feat_end+1).alignment = center
    ws.row_dimensions[1].height = 16

    for ci, (col, hdr, _) in enumerate(COLS, 1):
        c = ws.cell(2, ci, hdr)
        c.font = font_(bold=True, color="FFFFFF", size=9)
        c.fill = (FILLS["raw"] if col in RAW_COLS else
                  FILLS["feature"] if col in NEW_COLS else
                  FILLS["target"])
        c.alignment = center
        c.border = border
    ws.row_dimensions[2].height = 30

    for ci, (col, _, unit) in enumerate(COLS, 1):
        c = ws.cell(3, ci, unit)
        c.font = font_(italic=True, color="FFFFFF", size=8)
        c.fill = fill("4A4A4A")
        c.alignment = center
        c.border = border
    ws.row_dimensions[3].height = 13

    NUM_FMT = {
        "volume_fraction": "0.000000",
        "temperature_K":   "0.00",
        "sigma_bf":        "0.00E+00",
        "eps_r":           "0.00",
        "rho_bf":          "0.00",
        "upsilon":         "0.00E+00",
        "sigma_p":         "0.00E+00",
        "rho_p":           "0.0",
        "mu_bf":           "0.00000000",
        "sigma_nf":        "0.00E+00",
        "log_sigma_nf":    "0.0000",
    }

    for ri, (_, row) in enumerate(df.iterrows(), start=4):
        even = (ri % 2 == 0)
        for ci, (col, _, _) in enumerate(COLS, 1):
            val = row.get(col, "")
            if pd.isna(val): val = ""
            c = ws.cell(ri, ci, val)
            c.border = border
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.font = font_(size=9)
            if col in NEW_COLS:
                c.fill = FILLS["feat_b"] if even else FILLS["feat_a"]
            elif col in TARGET_SET:
                c.fill = FILLS["tgt_b"] if even else FILLS["tgt_a"]
            else:
                c.fill = FILLS["row_b"] if even else FILLS["row_a"]
            if col in NUM_FMT:
                c.number_format = NUM_FMT[col]

    COL_W = {
        "group_id": 7, "subgroup_id": 7,
        "nanoparticle": 14, "base_fluid": 18,
        "particle_size_nm": 7, "volume_fraction": 11, "temperature_K": 8,
        "sigma_bf": 11, "eps_r": 8, "rho_bf": 9, "upsilon": 11,
        "sigma_p": 11, "rho_p": 9, "mu_bf": 13,
        "nanoparticle_encoded": 9, "base_fluid_encoded": 9,
        "sigma_nf": 12, "log_sigma_nf": 14,
    }
    for ci, (col, _, _) in enumerate(COLS, 1):
        ws.column_dimensions[get_column_letter(ci)].width = COL_W.get(col, 12)

    ws.freeze_panes = "A4"

    ws2 = wb.create_sheet("Feature Sets")
    ws2["A1"] = "MANIS — EC Feature Sets Reference"
    ws2["A1"].font = Font(name="Arial", bold=True, size=13, color="1E3A5F")
    ws2.row_dimensions[1].height = 20

    ws2["A3"] = "1. CONVENTIONAL (5 parameters)"
    ws2["A3"].font = Font(name="Arial", bold=True, size=11)
    for i, c in enumerate(CONVENTIONAL_COLS, 4):
        ws2.cell(i, 1, c).font = Font(name="Arial", size=10)

    r0 = 4 + len(CONVENTIONAL_COLS) + 1
    ws2.cell(r0, 1, "2. SHEN ET AL. (trimmed — 4 extra beyond conventional)").font = Font(name="Arial", bold=True, size=11)
    for i, c in enumerate(SHEN_COLS, r0+1):
        ws2.cell(i, 1, c).font = Font(name="Arial", size=10)

    r1 = r0 + 1 + len(SHEN_COLS) + 1
    ws2.cell(r1, 1, "3. NEW CORRELATION (trimmed — 5 extra beyond conventional)").font = Font(name="Arial", bold=True, size=11)
    for i, c in enumerate(NEWCORR_COLS, r1+1):
        ws2.cell(i, 1, c).font = Font(name="Arial", size=10)

    r2 = r1 + 1 + len(NEWCORR_COLS) + 1
    ws2.cell(r2, 1, "TARGET COLUMNS (train primarily on log_sigma_nf — 10 orders of magnitude span)").font = Font(name="Arial", bold=True, size=11)
    for i, c in enumerate(TARGET_COLS, r2+1):
        ws2.cell(i, 1, c).font = Font(name="Arial", size=10)

    ws2.column_dimensions["A"].width = 65

    ws3 = wb.create_sheet("Summary")
    ws3["A1"] = "MANIS — EC Feature Dataset Summary"
    ws3["A1"].font = Font(name="Arial", bold=True, size=13, color="1E3A5F")
    ws3.row_dimensions[1].height = 20

    info = [
        ("Total data points",     len(df)),
        ("Total groups",          df["group_id"].nunique()),
        ("Nanoparticles",         ", ".join(sorted(df["nanoparticle"].unique()))),
        ("Base fluids",           ", ".join(sorted(df["base_fluid"].unique()))),
        ("Temperature range",     f"{df['temperature_K'].min():.1f} – {df['temperature_K'].max():.1f} K"),
        ("Volume fraction range", f"{df['volume_fraction'].min():.6f} – {df['volume_fraction'].max():.4f}"),
        ("sigma_nf range",        f"{df['sigma_nf'].min():.3e} – {df['sigma_nf'].max():.3e} S/m"),
        ("log_sigma_nf range",    f"{df['log_sigma_nf'].min():.2f} – {df['log_sigma_nf'].max():.2f}"),
        ("", ""),
        ("NOTE", "U0, lambda, T0 (Shen et al.) and U0, h, psi (New Correlation) were"),
        ("",     "dropped as ML features — fixed constants with zero/near-zero"),
        ("",     "variance across rows, not usable for training. See prior"),
        ("",     "benchmarking discussion for full rationale."),
    ]
    for ri, (k, v) in enumerate(info, 3):
        ws3.cell(ri, 1, k).font = Font(name="Arial", bold=(k=="NOTE" or k=="Total data points"), size=10)
        ws3.cell(ri, 2, str(v)).font = Font(name="Arial", size=10)

    ws3.column_dimensions["A"].width = 25
    ws3.column_dimensions["B"].width = 60

    wb.save(path)

# ============================================================
# MAIN
# ============================================================

def run_feature_engineering():
    if not Path(CLEAN_CSV).exists():
        print(f"❌ {CLEAN_CSV} not found.")
        return

    print(f"📂 Loading: {CLEAN_CSV}")
    df = pd.read_csv(CLEAN_CSV)
    print(f"   {len(df)} rows  |  {df['group_id'].nunique()} groups")

    print("\n⚙️  Computing physics-informed features...")
    df, unknown_np, unknown_bf, le_np, le_bf = compute_features(df)

    if unknown_np:
        print(f"⚠️  Unknown nanoparticles (sigma_p/rho_p will be NaN): {sorted(unknown_np)}")
    if unknown_bf:
        print(f"⚠️  Unknown base fluids (eps_r/mu_bf/rho_bf will be NaN): {sorted(unknown_bf)}")

    all_extra_cols = list(dict.fromkeys(SHEN_EXTRA_COLS + NEWCORR_EXTRA_COLS +
                                         ["nanoparticle_encoded","base_fluid_encoded"]))
    missing = df[all_extra_cols].isna().sum()
    print("\n📊 Missing values per feature column:")
    for col, n in missing.items():
        status = "✅" if n == 0 else f"⚠️  {n} rows"
        print(f"   {col:<25} {status}")

    print(f"\n💾 Writing Excel: {FEATURE_XLS}")
    write_excel(df, FEATURE_XLS)

    print("\n🤖 Generating AI feature justification...")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Skipping AI justification step — no API key set.")
        print("   Excel file was still saved successfully above.")
        print(f"\n✅ Done (partial — Excel saved, AI notes skipped)!")
        return

    try:
        ai_notes = ai_analyze_features(df)
        header = "EC FEATURE ENGINEERING AI AGENT — AI Recommendation\n\n"
        print("\n" + "="*60)
        print(header + ai_notes)
        print("="*60)

        with open(AI_NOTES_TXT, "w") as f:
            f.write(header + ai_notes)
        print(f"\n✅ AI notes saved → {AI_NOTES_TXT}")
    except Exception as e:
        print(f"⚠️  AI justification step failed: {e}")
        print("   Excel file was still saved successfully above.")

    print(f"\n✅ Done!")
    print(f"   File : {FEATURE_XLS}")
    print(f"   Rows : {len(df)}")
    print(f"\n   Feature set 1 — Conventional (5): {CONVENTIONAL_COLS}")
    print(f"   Feature set 2 — Shen et al. (9):   {SHEN_COLS}")
    print(f"   Feature set 3 — New Correlation (10): {NEWCORR_COLS}")
    print(f"   Targets: {TARGET_COLS}  (train primarily on log_sigma_nf)")
    print(f"\n   → Upload nanofluid_ec_features.xlsx to Google Drive")
    print(f"   → Next: ML Development Agent (Random Forest, Gradient Boosting,")
    print(f"     SVR, ANN, XGBoost — 80/20 split — trained separately on all 3 sets)")

run_feature_engineering()
