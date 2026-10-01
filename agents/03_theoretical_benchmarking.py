# ============================================================
# MANIS: ELECTRICAL CONDUCTIVITY
# THEORETICAL MODEL BENCHMARKING AI AGENT
# ============================================================
# Reads nanofluid_ec_data_clean.csv from Google Drive
# Tests 5 electrical conductivity correlations against experimental data:
#   1. Maxwell (1873): full mixing rule, spherical (n=3)
#   2. Cruz et al.: modified Maxwell (insulating-particle limit, alpha->0)
#   3. Ganguly et al. (2009): empirical, Al2O3/water fit
#      (deliberately run on ALL rows as a negative control)
#   4. Shen et al. (2012): Maxwell + electrophoresis + Brownian,
#      restricted to ZnO rows only
#   5. New Correlation (this project): full Maxwell (shape factor +
#      nanolayer generalised) + electrophoresis + Brownian, with
#      Udawattha & Narayana (2018) viscosity
# Outputs: metrics table (R², MAE, RMSE, MAPE) per model
#          + AI-generated commentary on the benchmark
#
# Notes for readers:
#   - Predictions that are negative, NaN or infinite are treated as
#     invalid and excluded from that model's metrics. For Ganguly et al.
#     this removes the rows where its linear form returns negative
#     (physically impossible) conductivities, which is why its number
#     of valid points is below 509.
#   - The "MD %" column equals MAPE by construction and is not reported
#     separately in the manuscript.
#   - Language model: Claude Sonnet 4.5 (claude-sonnet-4-5). The API key
#     is read from the ANTHROPIC_API_KEY environment variable.
# ============================================================

# ── CELL 1: INSTALL ─────────────────────────────────────────
# !pip install gradio pandas numpy scipy scikit-learn anthropic -q

# ── CELL 2: IMPORTS & SETUP ─────────────────────────────────

import os
import json
import re
import numpy as np
import pandas as pd
import gradio as gr
import anthropic
from pathlib import Path
from google.colab import drive
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Google Drive ─────────────────────────────────────────────
DRIVE_DIR      = "/content/drive/MyDrive/MANIS_ELECTRICAL/"
EC_CSV         = DRIVE_DIR + "nanofluid_ec_data_clean.csv"
EC_METRICS_CSV = DRIVE_DIR + "ec_model_metrics.csv"
EC_PRED_CSV    = DRIVE_DIR + "ec_model_predictions.csv"

def mount_drive():
    try:
        drive.mount('/content/drive', force_remount=False)
        os.makedirs(DRIVE_DIR, exist_ok=True)
        return True
    except Exception as e:
        print(f"Drive mount failed: {e}")
        return False

mount_drive()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── Physical constants ────────────────────────────────────────
EPS_0 = 8.8541878128e-12   # Vacuum permittivity   F/m
R     = 8.314              # Gas constant          J/(mol·K)
N_A   = 6.02214076e23      # Avogadro constant     mol⁻¹

# ── Fixed model parameters (agreed defaults) ──────────────────
H_NM   = 1.0    # nanolayer thickness, nm (Udawattha default)
PSI    = 1.0    # sphericity: spherical for ALL materials, including SiC
N_SHAPE = 3.0 / PSI   # = 3 for all materials given PSI=1

# ============================================================
# NANOPARTICLE ELECTRICAL PROPERTIES LOOKUP
# sigma_p   : bulk electrical conductivity, S/m. LOW CONFIDENCE overall;
#             literature spread for these oxides spans many orders of
#             magnitude depending on crystallinity/doping/impurity. Point
#             values below are defensible single estimates, not measurements.
# rho_p     : density, kg/m³. HIGH CONFIDENCE.
# U0_mV     : |Zeta potential|, mV. MODERATE CONFIDENCE, literature
#             typical range for aqueous/EG dispersions; treated as fixed
#             default per material, ALWAYS USED AS ABSOLUTE VALUE.
# ============================================================
NP_EC_PROPS = {
    # Source: Shen et al. (2012), Physics Letters A 376, 1053-1057 (direct citation)
    "ZNO":   {"sigma_p": 1.62e-6,  "rho_p": 5600, "U0_mV": 10},
    # Bulk single-crystal alpha-Al2O3 is an excellent insulator; point estimate only
    "AL2O3": {"sigma_p": 1e-11,    "rho_p": 3970, "U0_mV": 40},
    # TiO2: wide-bandgap, oxygen-vacancy-sensitive semiconductor; point estimate only
    "TIO2":  {"sigma_p": 1e-7,     "rho_p": 4230, "U0_mV": 30},
    # CuO: narrow-bandgap p-type semiconductor, notably more conductive
    "CUO":   {"sigma_p": 1e-1,     "rho_p": 6310, "U0_mV": 25},
    # SiO2: well-established bulk insulator
    "SIO2":  {"sigma_p": 1e-11,    "rho_p": 2200, "U0_mV": 30},
    # SiC: semi-insulating to highly conductive depending on polytype/doping
    "SIC":   {"sigma_p": 1e-3,     "rho_p": 3160, "U0_mV": 20},
    # Source: Dong et al. (2013), J. Nanomaterials 2013, 842963 (direct citation)
    # AlN is not present in the final dataset; kept for completeness.
    "ALN":   {"sigma_p": 1e-12,    "rho_p": 3260, "U0_mV": 20},
    # CaCO3 (calcite): insulator, estimated as class-consistent with
    # Al2O3/SiO2; rho_p = 2700 kg/m3 standard calcite density;
    # U0 from food-grade/synthesised CaCO3 measurements (~12-16 mV)
    "CACO3": {"sigma_p": 1e-10,    "rho_p": 2700, "U0_mV": 15},
}

# ============================================================
# BASE FLUID ELECTRICAL / VISCOSITY PROPERTIES
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
    """Transformer (insulating) oil.
    eps_r matches Shen et al./Dong et al.'s own insulated-oil value (2.2).
    mu, rho fitted from Patel et al. (2009) Table 1."""
    Tc    = T - 273.15
    eps_r = 2.2
    rho   = 879.0 - 0.6*(Tc - 20.0)
    mu    = 0.0211 * np.exp(-0.0431*(Tc - 20.0))
    return eps_r, mu, rho

def props_peg200(T):
    """PEG (MW~200). eps_r=19 from literature; mu, rho are
    low-confidence literature-range estimates."""
    Tc    = T - 273.15
    eps_r = 19.0
    mu    = 0.055 * np.exp(-0.02*(Tc - 20.0))   # ~55 mPa·s at 20C
    rho   = 1125.0 - 0.6*(Tc - 20.0)
    return eps_r, mu, rho

def props_bio_glycol(T):
    """Bio Glycol, treated as bio-based 1,2-propanediol equivalent.
    Low confidence across all three properties."""
    Tc    = T - 273.15
    eps_r = 28.0
    mu    = 0.048 * np.exp(-0.025*(Tc - 20.0))  # ~48 mPa·s at 20C
    rho   = 1036.0 - 0.6*(Tc - 20.0)
    return eps_r, mu, rho

# Pure-fluid property functions, keyed for mixture interpolation
_PURE_FLUID_FN = {
    "water":            props_water,
    "eg":               props_EG,
    "ethylene glycol":  props_EG,
    "pg":               props_bio_glycol,   # propylene glycol stand-in
    "propylene glycol": props_bio_glycol,
}

def parse_mixture_ratio(name):
    """Parses 'Water:EG 90:10' or 'Water:Pg 40:60' into
    (component_a, component_b, frac_a, frac_b). Returns None if not a
    recognised 'X:Y AA:BB' mixture string."""
    m = re.match(r"^\s*([A-Za-z0-9]+)\s*:\s*([A-Za-z0-9]+)\s+(\d+)\s*:\s*(\d+)\s*$", name)
    if not m:
        return None
    comp_a, comp_b, ratio_a, ratio_b = m.groups()
    frac_a = float(ratio_a) / (float(ratio_a) + float(ratio_b))
    frac_b = 1.0 - frac_a
    return comp_a.strip().lower(), comp_b.strip().lower(), frac_a, frac_b

def props_mixture(T, name):
    """Linear interpolation of (eps_r, mu_bf, rho_bf) between two pure
    fluids by volume ratio (standard first-order approximation)."""
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
    """Looks up (eps_r, mu_bf, rho_bf) for a base fluid name. Falls back to
    mixture interpolation for 'X:Y AA:BB' style names."""
    fn = BASE_FLUID_EC_FN.get(base_fluid_name)
    if fn is not None:
        return fn(T)
    mix = props_mixture(T, base_fluid_name)
    if mix is not None:
        return mix
    return np.nan, np.nan, np.nan

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def brownian_velocity_ec(T, rho_p, d_p_m):
    """V_B = sqrt(18 R T / (pi N_A rho_p d_p^3)), Udawattha & Narayana form."""
    return np.sqrt(18 * R * T / (np.pi * N_A * rho_p * d_p_m**3))

def effective_phi(phi, h_m, r_m):
    """phi_e = phi * (1 + h/r)^3"""
    return phi * (1 + h_m/r_m)**3

def interparticle_spacing(d_p_m, phi):
    """delta = (pi d_p^3 / (6 phi))^(1/3)"""
    if phi <= 0:
        return np.nan
    return (np.pi * d_p_m**3 / (6*phi))**(1/3)

def udawattha_viscosity(mu_bf, T, phi, d_p_m, rho_p):
    """mu_nf = mu_bf * (1 + 2.5*phi_e + dynamic_term),
    Udawattha & Narayana (2018) full model."""
    if phi <= 0:
        return mu_bf
    r_m = d_p_m / 2.0
    h_m = H_NM * 1e-9
    phi_e = effective_phi(phi, h_m, r_m)
    V_B   = brownian_velocity_ec(T, rho_p, d_p_m)
    delta = interparticle_spacing(d_p_m, phi)
    correction = T * 1e-10 * phi**(-0.002*T - 0.284)
    if correction <= 0 or not np.isfinite(correction):
        return mu_bf * (1 + 2.5*phi_e)
    dynamic_term = (rho_p * V_B * d_p_m**2) / (72 * delta * correction)
    return mu_bf * (1 + 2.5*phi_e + dynamic_term)

# ============================================================
# 5 ELECTRICAL CONDUCTIVITY CORRELATIONS
# All return predicted sigma_nf in S/m
# ============================================================

def model_maxwell(sigma_bf, sigma_p, phi, d_p_m, **kw):
    """1. Maxwell (1873/1881): full mixing rule, generalised shape/nanolayer
    at n=3 (spherical, PSI=1 for all materials)."""
    r_m = d_p_m / 2.0
    h_m = H_NM * 1e-9
    phi_e = effective_phi(phi, h_m, r_m)
    n = N_SHAPE
    num = sigma_p + (n-1)*sigma_bf + (n-1)*phi_e*(sigma_p - sigma_bf)
    den = sigma_p + (n-1)*sigma_bf -       phi_e*(sigma_p - sigma_bf)
    return sigma_bf * num / den

def model_cruz(sigma_bf, phi, **kw):
    """2. Cruz et al.: modified Maxwell, insulating-particle limit (alpha->0)."""
    return sigma_bf * (1 - 1.5*phi)

def model_ganguly(sigma_bf, phi, T, **kw):
    """3. Ganguly et al. (2009): empirical, fit to Al2O3/water.
    Run on ALL rows deliberately as a negative control.
    T is converted to Celsius per the original fit; phi as a plain fraction."""
    T_C = T - 273.15
    return sigma_bf * (3679.049*phi + 1.085799*T_C - 42.6384)

def model_shen(sigma_bf, sigma_p, phi, T, d_p_m, eps_r, U0_mV,
               mu_bf, rho_bf, lam=0.04, T0=313.15, **kw):
    """4. Shen et al. (2012): Maxwell (linearised) + electrophoresis +
    Brownian. Restricted to ZnO rows at the calling level. Uses Shen et
    al.'s own viscosity submodel; T0 and lam are Shen et al.'s fitted
    constants for ZnO-in-oil and are not re-fitted here."""
    r_m  = d_p_m / 2.0
    U0   = abs(U0_mV) * 1e-3   # mV -> V, magnitude only
    upsilon = mu_bf / rho_bf   # kinematic viscosity, m^2/s

    sigma_M = sigma_bf * (1 + 3*phi)

    visc_corr = 1 + 25*phi + 625*phi**2
    exp_T = np.exp(lam*(T - T0))

    sigma_E = (2*phi*eps_r**2*EPS_0**2*U0**2 / (rho_bf*upsilon*visc_corr*r_m**2)) * exp_T

    sigma_B = (3*phi*eps_r*EPS_0*U0 / r_m**1.5) * np.sqrt(
        (R*T/N_A) * exp_T / (3*np.pi*rho_bf*upsilon*visc_corr)
    )

    return sigma_M + sigma_E + sigma_B

def model_new_correlation(sigma_bf, sigma_p, phi, T, d_p_m, eps_r, U0_mV,
                           mu_bf, rho_p, **kw):
    """5. New Correlation (this project): full generalised Maxwell
    (shape factor + nanolayer) + electrophoresis + Brownian, with
    Udawattha & Narayana (2018) viscosity substituted for mu_nf."""
    r_m = d_p_m / 2.0
    h_m = H_NM * 1e-9
    U0  = abs(U0_mV) * 1e-3

    phi_e = effective_phi(phi, h_m, r_m)
    n = N_SHAPE
    num = sigma_p + (n-1)*sigma_bf + (n-1)*phi_e*(sigma_p - sigma_bf)
    den = sigma_p + (n-1)*sigma_bf -       phi_e*(sigma_p - sigma_bf)
    sigma_M_ratio = num / den

    mu_nf = udawattha_viscosity(mu_bf, T, phi, d_p_m, rho_p)

    sigma_E = 8*phi*eps_r**2*EPS_0**2*U0**2 / (mu_nf * d_p_m**2)
    sigma_B = (6*np.sqrt(2)*phi*eps_r*EPS_0*U0 / d_p_m**1.5) * np.sqrt(
        R*T / (3*np.pi*mu_nf*N_A)
    )

    return sigma_bf * sigma_M_ratio + sigma_E + sigma_B

# ── Model registry ─────────────────────────────────────────────
# 'restrict' is an optional row filter: (row) -> bool. If provided, the
# model is scored ONLY on rows where restrict(row) is True.
MODELS = {
    "1. Maxwell (1873)": {
        "fn": model_maxwell, "restrict": None,
    },
    "2. Cruz et al., insulating limit": {
        "fn": model_cruz, "restrict": None,
    },
    "3. Ganguly et al. (2009), empirical": {
        "fn": model_ganguly, "restrict": None,   # deliberately unrestricted
    },
    "4. Shen et al. (2012)": {
        "fn": model_shen,
        "restrict": lambda row: str(row["nanoparticle"]).upper() == "ZNO",
    },
    "5. New Correlation (proposed)": {
        "fn": model_new_correlation, "restrict": None,
    },
}

# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    yt   = y_true[mask]
    yp   = y_pred[mask]
    n    = len(yt)
    if n < 2:
        return {"R²": np.nan, "MAE": np.nan, "RMSE": np.nan,
                "MAPE %": np.nan, "MD %": np.nan, "Valid pts": n}
    R2   = r2_score(yt, yp)
    MAE  = mean_absolute_error(yt, yp)
    RMSE = np.sqrt(mean_squared_error(yt, yp))
    MAPE = float(np.mean(np.abs((yt - yp) / (yt + 1e-30))) * 100)
    MD   = MAPE
    return {"R²": round(R2,4), "MAE": round(MAE,10),
            "RMSE": round(RMSE,10), "MAPE %": round(MAPE,2),
            "MD %": round(MD,2), "Valid pts": n}

# ============================================================
# DATA LOADING, sigma_bf EXTRACTION, & ENRICHMENT
# ============================================================

def extract_sigma_bf_per_group(df):
    """For each (group_id, temperature_K), takes sigma_bf from that
    group's own volume_fraction==0 row (uS/cm -> S/m). Falls back to a
    literature value for groups with NO phi=0 row at any temperature
    (not triggered by the final dataset, in which every group has
    phi=0 rows)."""

    UNIT_CONV = 1e-4   # uS/cm -> S/m

    sigma_bf_map = {}   # (group_id, temperature_K) -> sigma_bf in S/m

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

def load_and_enrich_ec():
    if not Path(EC_CSV).exists():
        return None, "❌ nanofluid_ec_data_clean.csv not found."

    df = pd.read_csv(EC_CSV)
    required = {"group_id","nanoparticle","base_fluid","particle_size_nm",
                "temperature_K","volume_fraction","electrical_conductivity_uScm"}
    missing = required - set(df.columns)
    if missing:
        return None, f"❌ Missing columns: {missing}"

    df = df.dropna(subset=list(required))
    if df.empty:
        return None, "❌ No valid rows after dropping NaNs."

    # sigma_bf per group/temperature, taken from the data itself
    sigma_bf_map = extract_sigma_bf_per_group(df)
    df["sigma_bf"] = df.apply(
        lambda r: sigma_bf_map.get((r["group_id"], r["temperature_K"]), np.nan), axis=1
    )

    # Measured sigma_nf, converted to S/m
    df["sigma_nf_measured"] = df["electrical_conductivity_uScm"] * 1e-4

    # Base fluid properties (eps_r, mu_bf, rho_bf)
    eps_l, mu_l, rho_l = [], [], []
    for _, row in df.iterrows():
        eps_r, mu_bf, rho_bf = get_base_fluid_props(row["temperature_K"], str(row["base_fluid"]).strip())
        eps_l.append(eps_r); mu_l.append(mu_bf); rho_l.append(rho_bf)
    df["eps_r"]  = eps_l
    df["mu_bf"]  = mu_l
    df["rho_bf"] = rho_l

    # Nanoparticle properties (case-insensitive)
    sigma_p_l, rho_p_l, U0_l = [], [], []
    for _, row in df.iterrows():
        props = NP_EC_PROPS.get(str(row["nanoparticle"]).strip().upper(), {})
        sigma_p_l.append(props.get("sigma_p", np.nan))
        rho_p_l.append(props.get("rho_p", np.nan))
        U0_l.append(props.get("U0_mV", np.nan))
    df["sigma_p"] = sigma_p_l
    df["rho_p"]   = rho_p_l
    df["U0_mV"]   = U0_l

    df["d_p_m"] = df["particle_size_nm"] * 1e-9

    unknown_bf = df[df["eps_r"].isna()]["base_fluid"].unique().tolist()
    unknown_np = df[df["sigma_p"].isna()]["nanoparticle"].unique().tolist()
    warnings = []
    if unknown_bf:
        warnings.append(f"⚠️ Unknown/unparsed base fluids: {unknown_bf}")
    if unknown_np:
        warnings.append(f"⚠️ Unknown nanoparticles: {unknown_np}")

    n_rows   = len(df)
    n_groups = df["group_id"].nunique()
    msg = (f"✅ Loaded {n_rows} rows from {n_groups} groups.\n"
           + ("\n".join(warnings) if warnings else "All base fluids and nanoparticles recognised."))
    return df, msg

# ============================================================
# RUN PREDICTIONS
# ============================================================

def run_all_models_ec(df, selected_models):
    results = df.copy()
    y_true  = df["sigma_nf_measured"].values.astype(float)

    metrics_rows = []

    for model_name in selected_models:
        spec     = MODELS[model_name]
        fn       = spec["fn"]
        restrict = spec["restrict"]
        preds = []
        for _, row in df.iterrows():
            if restrict is not None and not restrict(row):
                preds.append(np.nan)
                continue
            try:
                pred = fn(
                    sigma_bf = row["sigma_bf"],
                    sigma_p  = row["sigma_p"],
                    phi      = row["volume_fraction"],
                    T        = row["temperature_K"],
                    d_p_m    = row["d_p_m"],
                    eps_r    = row["eps_r"],
                    U0_mV    = row["U0_mV"],
                    mu_bf    = row["mu_bf"],
                    rho_bf   = row["rho_bf"],
                    rho_p    = row["rho_p"],
                )
                # Only reject negative/NaN/inf (no fixed magnitude bound,
                # since EC spans several orders of magnitude)
                if not np.isfinite(pred) or pred < 0:
                    pred = np.nan
            except Exception:
                pred = np.nan
            preds.append(pred)

        col = f"pred_{model_name.split('.')[0].strip()}"
        results[col] = preds

        m = compute_metrics(y_true, np.array(preds, dtype=float))
        m["Model"] = model_name
        metrics_rows.append(m)

    metrics_df = (pd.DataFrame(metrics_rows)
                  [["Model","R²","MAE","RMSE","MAPE %","MD %","Valid pts"]]
                  .sort_values("R²", ascending=False)
                  .reset_index(drop=True))

    metrics_df.to_csv(EC_METRICS_CSV, index=False)
    results.to_csv(EC_PRED_CSV, index=False)

    return results, metrics_df

# ============================================================
# PER-NANOPARTICLE METRICS
# ============================================================

def compute_all_per_nanoparticle_ec(results, model_names):
    out = {}
    for model_name in model_names:
        col = f"pred_{model_name.split('.')[0].strip()}"
        if col not in results.columns:
            continue
        per_np = {}
        for np_name, grp in results.groupby("nanoparticle"):
            y_t = grp["sigma_nf_measured"].values.astype(float)
            y_p = grp[col].values.astype(float)
            per_np[np_name] = compute_metrics(y_t, y_p)
        out[model_name] = per_np
    return out

# ============================================================
# AI COMMENTARY STEP
# ============================================================

def ai_analyze_benchmark_ec(metrics_df, per_np_metrics, n_rows, n_groups):
    overall_summary = metrics_df.to_dict(orient="records")

    condensed = {}
    for model_name, per_np in per_np_metrics.items():
        condensed[model_name] = {
            np_name: {"R2": m["R²"], "MAPE": m["MAPE %"], "n": m["Valid pts"]}
            for np_name, m in per_np.items()
        }

    prompt = f"""You are an expert in nanofluid electrical transport and theoretical model benchmarking.

A dataset of {n_rows} experimental electrical conductivity records across {n_groups}
nanofluid systems was used to benchmark 5 correlations: Maxwell (full mixing rule),
Cruz et al. (insulating-particle limit), Ganguly et al. (empirical, Al2O3/water fit,
run on all rows deliberately as a negative control), Shen et al. (mechanistic,
restricted to ZnO systems only), and a New Correlation (this project's proposed
model: generalized Maxwell + electrophoresis + Brownian motion, using the
Udawattha & Narayana viscosity model).

Overall benchmark results (R², MAE, RMSE, MAPE % across the full dataset):
{json.dumps(overall_summary, indent=2)}

Per-nanoparticle-type R², MAPE %, and sample count for each model:
{json.dumps(condensed, indent=2)}

Provide a concise, scientific commentary covering:

1. OVERALL BEST-FIT MODEL
   - Which model performs best overall, and by how much versus the next-best model?

2. MECHANISTIC VS EMPIRICAL
   - How does the Ganguly empirical model perform outside its Al2O3/water fitting
     domain, compared to the mechanistic models? Does this support the case that
     mechanistic models generalize better?

3. NEW CORRELATION PERFORMANCE
   - How does the New Correlation (proposed) compare to Shen et al. specifically
     on the ZnO subset where Shen et al. is scored, and to Maxwell/Cruz on the
     full dataset?

4. MODELS THAT UNDERPERFORM
   - Which model(s) show the weakest fit, and for which systems specifically?

5. IMPLICATIONS
   - Briefly state which physical parameters from the best-fitting model(s)
     seem most important, and note any data-quality caveats (e.g. uncertain
     sigma_p or Zeta potential lookup values) that may be limiting scores.

Be specific, cite the actual numbers, and keep each section to 3-4 sentences."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="MANIS: EC Model Benchmarking Agent", theme=gr.themes.Soft()) as app:

    gr.Markdown("# ⚡ MANIS: Electrical Conductivity Theoretical Model Benchmarking Agent")
    gr.Markdown(
        "Reads experimental data and tests 5 electrical conductivity correlations.  \n"
        "Model 3 (Ganguly) runs on all rows as a deliberate negative control.  \n"
        "Model 4 (Shen et al.) is restricted to ZnO systems only."
    )

    df_state      = gr.State(None)
    results_state = gr.State(None)
    metrics_state = gr.State(None)

    with gr.Tabs():

        with gr.Tab("1: Load Data"):
            gr.Markdown("### Load experimental data from Google Drive")
            load_btn    = gr.Button("📂 Load Dataset", variant="primary", size="lg")
            load_status = gr.Textbox(label="Status", interactive=False, lines=5)
            data_preview= gr.Dataframe(
                label="Dataset preview (first 20 rows)",
                interactive=False, wrap=True, max_height=300
            )

            def do_load():
                df, msg = load_and_enrich_ec()
                if df is None:
                    return msg, pd.DataFrame(), None
                preview_cols = ["group_id","nanoparticle","base_fluid","particle_size_nm",
                                "volume_fraction","temperature_K","sigma_nf_measured",
                                "sigma_bf","sigma_p","eps_r","mu_bf","U0_mV"]
                return msg, df[preview_cols].head(20), df

            load_btn.click(do_load, outputs=[load_status, data_preview, df_state])

        with gr.Tab("2: Run Models"):
            gr.Markdown("### Select models and run benchmark")
            model_selector = gr.CheckboxGroup(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys()),
                label="Select models to test"
            )
            run_btn     = gr.Button("▶ Run Benchmark", variant="primary", size="lg")
            run_status  = gr.Textbox(label="Status", interactive=False, lines=2)
            metrics_tbl = gr.Dataframe(
                label="📊 Model Metrics, sorted by R² (best first)",
                interactive=False, wrap=True
            )
            pred_preview= gr.Dataframe(
                label="Predictions preview (first 20 rows)",
                interactive=False, wrap=True, max_height=300
            )

            def do_run(df, selected):
                if df is None:
                    return ("❌ Load data first.", pd.DataFrame(), pd.DataFrame(), None, None)
                if not selected:
                    return ("⚠️ Select at least one model.", pd.DataFrame(), pd.DataFrame(), None, None)
                results, metrics_df = run_all_models_ec(df, selected)
                best = metrics_df.iloc[0]["Model"]
                r2   = metrics_df.iloc[0]["R²"]
                msg  = (f"✅ {len(selected)} models tested on {len(df)} data points.\n"
                        f"Best model: {best}  (R² = {r2})\n"
                        f"Go to Tab 4 for the AI-generated benchmarking commentary.")
                pred_cols = ["group_id","nanoparticle","base_fluid","particle_size_nm",
                             "volume_fraction","temperature_K","sigma_nf_measured"] + \
                            [c for c in results.columns if c.startswith("pred_")]
                return msg, metrics_df, results[pred_cols].head(20), results, metrics_df

            run_btn.click(do_run, inputs=[df_state, model_selector],
                          outputs=[run_status, metrics_tbl, pred_preview, results_state, metrics_state])

        with gr.Tab("3: Detailed Metrics"):
            gr.Markdown("### Per-nanoparticle metrics for a selected model")
            detail_model = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="5. New Correlation (proposed)",
                label="Select model"
            )
            detail_btn    = gr.Button("Compute Per-Group Metrics", variant="secondary")
            detail_status = gr.Textbox(label="Status", interactive=False)
            detail_tbl    = gr.Dataframe(label="Per-nanoparticle metrics", interactive=False, wrap=True)

            def do_detail(results, model_name):
                if results is None:
                    return "❌ Run benchmark first.", pd.DataFrame()
                col = f"pred_{model_name.split('.')[0].strip()}"
                if col not in results.columns:
                    return f"❌ Column {col} not found. Run benchmark first.", pd.DataFrame()

                rows = []
                for np_name, grp in results.groupby("nanoparticle"):
                    y_t = grp["sigma_nf_measured"].values.astype(float)
                    y_p = grp[col].values.astype(float)
                    m   = compute_metrics(y_t, y_p)
                    m["Nanoparticle"] = np_name
                    m["N data points"] = len(grp)
                    rows.append(m)

                detail_df = (pd.DataFrame(rows)
                             [["Nanoparticle","N data points","R²","MAE",
                               "RMSE","MAPE %","MD %","Valid pts"]]
                             .sort_values("R²", ascending=False)
                             .reset_index(drop=True))
                return f"✅ Per-group metrics for {model_name}", detail_df

            detail_btn.click(do_detail, inputs=[results_state, detail_model],
                             outputs=[detail_status, detail_tbl])

        with gr.Tab("4: AI Commentary"):
            gr.Markdown("### AI-Generated Benchmarking Commentary")
            ai_btn      = gr.Button("🤖 Generate AI Commentary", variant="primary", size="lg")
            ai_feedback = gr.Textbox(label="AI Commentary", interactive=False, lines=30)

            def do_ai_analysis(results, metrics_df):
                if results is None or metrics_df is None or len(metrics_df) == 0:
                    return "❌ Run the benchmark first (Tab 2)."
                model_names = metrics_df["Model"].tolist()
                per_np      = compute_all_per_nanoparticle_ec(results, model_names)
                n_rows      = len(results)
                n_groups    = results["group_id"].nunique()
                return ai_analyze_benchmark_ec(metrics_df, per_np, n_rows, n_groups)

            ai_btn.click(do_ai_analysis, inputs=[results_state, metrics_state], outputs=[ai_feedback])

        with gr.Tab("5: Google Drive"):
            gr.Markdown("### Google Drive Status")
            gr.Markdown(
                f"Reads from: `{EC_CSV}`  \n"
                f"Saves metrics to: `{EC_METRICS_CSV}`  \n"
                f"Saves predictions to: `{EC_PRED_CSV}`"
            )
            with gr.Row():
                mount_btn = gr.Button("Mount / Remount Drive", variant="secondary")
                check_btn = gr.Button("Check Drive Status", variant="secondary")
            drive_status = gr.Textbox(label="Status", interactive=False, lines=8)

            def check_drive():
                if not os.path.exists(DRIVE_DIR):
                    return "❌ Drive not mounted or project folder not found."
                files = os.listdir(DRIVE_DIR)
                if not files:
                    return "✅ Drive mounted. Project folder is empty."
                lines = ["✅ Drive mounted. Files in project folder:"]
                for f in sorted(files):
                    size = os.path.getsize(os.path.join(DRIVE_DIR, f))
                    lines.append(f"  {f}  ({size:,} bytes)")
                return "\n".join(lines)

            def do_mount():
                mount_drive()
                return check_drive()

            mount_btn.click(do_mount, outputs=[drive_status])
            check_btn.click(check_drive, outputs=[drive_status])

app.launch(share=True)
