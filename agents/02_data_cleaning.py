# ============================================================
# MANIS — ELECTRICAL CONDUCTIVITY
# AGENT 2: DATA CLEANING AI AGENT
# ============================================================
# Adapted from the thermal-conductivity Agent 2
# Target property : electrical_conductivity_uScm
# Reads  : nanofluid_ec_data.csv        (Agent 1 output)
# Writes : nanofluid_ec_data_clean.csv  (Colab 2 input)
#
# NOTE on filenames: the TC version of this agent reads a
# "_fixed" file (implying a separate volume-fraction-correction
# step runs between Agent 1 and Agent 2). The EC Agent 1 doesn't
# have that intermediate step yet, so this reads Agent 1's output
# directly. If you add an EC "fix" step later, just change
# DATA_CSV below to point at its output file.
#
# REVISION CHANGES (2026-09-29):
#   1. Frozen data guard: stops unless nanofluid_ec_data.csv is the
#      frozen revision dataset (509 rows, 22 groups, SHA256 below).
#   2. Duplicate stop: refuses to save if any rows would be removed
#      as duplicates, so no records can be dropped silently (the
#      duplicate check compares the physical fields, not group_id).
# ============================================================

import anthropic
import pandas as pd
import numpy as np
from pathlib import Path
import gradio as gr
import json
import os
import hashlib
import warnings
warnings.filterwarnings('ignore')
from google.colab import drive

# ============================================================
# GOOGLE DRIVE SETUP
# ============================================================

def mount_drive():
    try:
        drive.mount('/content/drive', force_remount=False)
        os.makedirs(DRIVE_DIR, exist_ok=True)
        return True
    except Exception as e:
        print(f"Drive mount failed: {e}")
        return False

DRIVE_DIR = "/content/drive/MyDrive/MANIS_ELECTRICAL/"
DATA_CSV  = DRIVE_DIR + "nanofluid_ec_data.csv"        # ← Agent 1 output
CLEAN_CSV = DRIVE_DIR + "nanofluid_ec_data_clean.csv"  # ← Colab 2 input

mount_drive()

# --- REVISION 1: Frozen data guard ---
_EXPECTED_SHA = "6ba8ece5f49791fd1dee6deb7599a527fc90c49541b7b4b8cd31a010788d2f9b"
assert hashlib.sha256(open(DATA_CSV, "rb").read()).hexdigest() == _EXPECTED_SHA, \
    "nanofluid_ec_data.csv is not the frozen version. STOP."
print("✅ Frozen data confirmed (509 rows, 22 groups)")

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ============================================================
# CLEANING PIPELINE
# ============================================================

def load_raw_data():
    if not Path(DATA_CSV).exists():
        return None, "❌ No raw data found! Run Agent 1 first."
    df = pd.read_csv(DATA_CSV)
    return df, f"✅ Loaded {len(df)} records from nanofluid_ec_data.csv"

def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(
        subset=["nanoparticle", "base_fluid", "particle_size_nm",
                "volume_fraction", "temperature_K"],
        keep="first"
    )
    return df, before - len(df)

def fix_volume_fraction(df):
    """
    Detect and fix volume fractions stored as percentages instead of decimals.

    Rule: volume_fraction should always be in range (0, 1).
    If a group has ANY volume_fraction > 1.0, it was stored as %
    and must be divided by 100.

    Applied per group_id so groups stored correctly are not affected.
    """
    fixed_groups = []
    if "group_id" not in df.columns:
        # No group info — apply globally
        if df["volume_fraction"].max() > 1.0:
            df["volume_fraction"] = df["volume_fraction"] / 100.0
            fixed_groups.append("ALL")
        return df, fixed_groups

    for gid in df["group_id"].unique():
        mask = df["group_id"] == gid
        vf_max = df.loc[mask, "volume_fraction"].max()
        if vf_max > 1.0:
            df.loc[mask, "volume_fraction"] = df.loc[mask, "volume_fraction"] / 100.0
            np_name = df.loc[mask, "nanoparticle"].iloc[0]
            bf_name = df.loc[mask, "base_fluid"].iloc[0]
            fixed_groups.append(f"Group {gid} ({np_name}/{bf_name})")

    return df, fixed_groups

def standardize_text(df):
    df["nanoparticle"] = df["nanoparticle"].str.strip().str.upper()
    # Use title() for base fluid but restore EG capitalisation.
    # e.g. str.title() turns "Water:EG 90:10" → "Water:Eg 90:10" and
    # "EG:Water 40:60" → "Eg:Water 40:60" — both break downstream lookups,
    # so fix "Eg" back to "EG" wherever it appears (start OR mid-string,
    # since base_fluid may be formatted "Water:EG x:y" or "EG:Water x:y").
    df["base_fluid"] = df["base_fluid"].str.strip().str.title()
    df["base_fluid"] = df["base_fluid"].str.replace(
        r"\bEg\b", "EG", regex=True
    )
    return df

def assign_subgroups(df):
    """Preserve group_id from Agent 1. Only add subgroup_id."""
    df["subgroup_id"] = df.groupby(["group_id", "temperature_K"]).ngroup() + 1
    return df

def sort_data(df):
    return df.sort_values(
        ["group_id", "temperature_K", "volume_fraction"]
    ).reset_index(drop=True)

def reorder_columns(df):
    cols = ["group_id", "subgroup_id", "nanoparticle", "base_fluid",
            "particle_size_nm", "temperature_K", "volume_fraction",
            "electrical_conductivity_uScm"]
    return df[[c for c in cols if c in df.columns]]

def generate_summary(df):
    summary = f"""
📊 Clean Dataset Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total records:    {len(df)}
Total groups:     {df['group_id'].nunique()}
Total subgroups:  {df['subgroup_id'].nunique()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Group Details:
"""
    for gid in sorted(df["group_id"].unique()):
        g    = df[df["group_id"] == gid]
        g0   = g.iloc[0]
        summary += (f"\nGroup {gid}: {g0['nanoparticle']} | {g0['base_fluid']} "
                    f"| {g0['particle_size_nm']}nm")
        summary += f"\n  → {g['temperature_K'].nunique()} temperatures | {len(g)} records"
        for temp in sorted(g["temperature_K"].unique()):
            sub = g[g["temperature_K"] == temp]
            summary += (f"\n     └── {temp}K → φ: {sub['volume_fraction'].min():.5f} "
                        f"to {sub['volume_fraction'].max():.5f} "
                        f"({len(sub)} points)")
    return summary

# ============================================================
# AI ANALYSIS
# ============================================================

def ai_analyze_dataset(df):
    group_stats = []
    for gid in sorted(df["group_id"].unique()):
        g  = df[df["group_id"] == gid]
        g0 = g.iloc[0]
        group_stats.append({
            "group_id":        int(gid),
            "nanoparticle":    g0["nanoparticle"],
            "base_fluid":      g0["base_fluid"],
            "particle_size_nm": float(g0["particle_size_nm"]) if pd.notna(g0["particle_size_nm"]) else None,
            "records":         len(g),
            "temperatures":    int(g["temperature_K"].nunique()),
            "temp_range":      f"{g['temperature_K'].min():.1f}K - {g['temperature_K'].max():.1f}K",
            "volume_fractions": int(g["volume_fraction"].nunique()),
            "vf_range":        f"{g['volume_fraction'].min():.5f} - {g['volume_fraction'].max():.5f}",
            "ec_range":        f"{g['electrical_conductivity_uScm'].min():.4f} - {g['electrical_conductivity_uScm'].max():.4f} uS/cm"
        })

    overall = {
        "total_records":  len(df),
        "total_groups":   int(df["group_id"].nunique()),
        "nanoparticles":  df["nanoparticle"].unique().tolist(),
        "base_fluids":    df["base_fluid"].unique().tolist(),
        "temp_range":     f"{df['temperature_K'].min():.1f}K - {df['temperature_K'].max():.1f}K",
        "vf_range":       f"{df['volume_fraction'].min():.5f} - {df['volume_fraction'].max():.5f}",
        "ec_range":       f"{df['electrical_conductivity_uScm'].min():.4f} - {df['electrical_conductivity_uScm'].max():.4f} uS/cm"
    }

    prompt = f"""You are an expert in nanofluid research and machine learning data analysis.

Analyze this cleaned nanofluid ELECTRICAL CONDUCTIVITY dataset and provide intelligent feedback.

Overall Dataset:
{json.dumps(overall, indent=2)}

Group Details:
{json.dumps(group_stats, indent=2)}

Provide feedback on:

1. DATA QUALITY ASSESSMENT
   - Are there groups with too few data points for ML?
   - Are temperature or volume fraction ranges adequate?
   - Any suspicious or unusual electrical conductivity values (e.g. non-monotonic
     trends vs. volume fraction, or enhancement ratios below 1)?

2. DATASET BALANCE
   - Is the dataset balanced across nanoparticle types and base fluids?
   - Note that different water:EG mixture ratios count as different base fluids/groups —
     which mixture ratios have the most/least data?
   - Any recommendations for data collection priorities?

3. ML READINESS
   - Is this dataset ready for machine learning?
   - What ML challenges might arise (e.g. base-fluid ionic strength/dielectric
     constant not being directly recorded, only inferred from composition)?
   - Recommended train/test split strategy?

4. RECOMMENDATIONS
   - Specific suggestions to improve data quality
   - Which nanofluid systems need more data points?
   - Any electrical conductivity values that seem physically unreasonable given
     Maxwell's model floor (recall: a non-conducting particle can only ever
     dilute conductivity slightly — large jumps are expected from EDL/percolation
     effects, not an error, but flag anything that looks like a reading mistake)?

Be specific, use the actual numbers, and write in a scientific style.
Keep each section concise — 3-4 sentences maximum."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ============================================================
# MAIN CLEANING FUNCTION
# ============================================================

def run_cleaning():
    df, msg = load_raw_data()
    if df is None:
        return msg, None, None

    if "group_id" not in df.columns:
        return "❌ group_id not found. Use Agent 1 v2+.", None, None

    original = len(df)
    df, removed      = remove_duplicates(df)

    # --- REVISION 2: Duplicate stop ---
    if removed > 0:
        return (f"❌ STOPPED: {removed} rows would be removed as duplicates. "
                f"Nothing saved. Check the data before continuing."), None, None

    df               = standardize_text(df)
    df, fixed_groups = fix_volume_fraction(df)   # ← fix % → decimal
    df               = assign_subgroups(df)
    df               = sort_data(df)
    df               = reorder_columns(df)
    df.to_csv(CLEAN_CSV, index=False)

    vf_fix_msg = ""
    if fixed_groups:
        vf_fix_msg = (f"\n⚠️  Volume fraction fixed (% → decimal) in "
                      f"{len(fixed_groups)} group(s):\n"
                      + "\n".join(f"   • {g}" for g in fixed_groups))
    else:
        vf_fix_msg = "\n✅ All volume fractions already in decimal form."

    status = f"""✅ Cleaning Complete!
━━━━━━━━━━━━━━━━━━━━━━━━
Original records:   {original}
Duplicates removed: {removed}
Final records:      {len(df)}
Groups preserved:   {df['group_id'].nunique()}
Subgroups created:  {df['subgroup_id'].nunique()}
━━━━━━━━━━━━━━━━━━━━━━━━
{vf_fix_msg}
━━━━━━━━━━━━━━━━━━━━━━━━
Saved to: nanofluid_ec_data_clean.csv

group_id values preserved from Agent 1."""

    return status, df, generate_summary(df)

def run_cleaning_and_show():
    status, df, summary = run_cleaning()
    preview = df.head(20) if df is not None else pd.DataFrame()
    return status, preview

def generate_ai_feedback():
    if not Path(CLEAN_CSV).exists():
        return "❌ Run cleaning first!"
    return ai_analyze_dataset(pd.read_csv(CLEAN_CSV))

def generate_summary_tab():
    if not Path(CLEAN_CSV).exists():
        return "Run cleaning first!"
    return generate_summary(pd.read_csv(CLEAN_CSV))

def get_group_detail(group_id):
    if not Path(CLEAN_CSV).exists():
        return "Run cleaning first!", pd.DataFrame()
    df = pd.read_csv(CLEAN_CSV)
    try:
        gid = int(group_id)
        g   = df[df["group_id"] == gid]
        if g.empty:
            return f"Group {gid} not found!", pd.DataFrame()
        g0  = g.iloc[0]
        info = (f"Group {gid}: {g0['nanoparticle']} | {g0['base_fluid']} | "
                f"{g0['particle_size_nm']}nm | {len(g)} records | "
                f"{g['subgroup_id'].nunique()} subgroups")
        return info, g
    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()

def check_group_id_consistency():
    if not Path(DATA_CSV).exists() or not Path(CLEAN_CSV).exists():
        return "❌ Run Agent 1 and cleaning first."
    raw   = pd.read_csv(DATA_CSV)
    clean = pd.read_csv(CLEAN_CSV)
    rg, cg = set(raw["group_id"].unique()), set(clean["group_id"].unique())
    if rg == cg:
        return f"✅ group_id consistent — {len(rg)} groups match."
    missing = rg - cg; extra = cg - rg
    msg = "⚠️ group_id mismatch!\n"
    if missing: msg += f"Missing in clean: {sorted(missing)}\n"
    if extra:   msg += f"Extra in clean: {sorted(extra)}"
    return msg

def view_raw_data():
    if Path(DATA_CSV).exists():
        df = pd.read_csv(DATA_CSV)
        return f"📊 Raw EC dataset: {len(df)} records.", df
    return "❌ No raw EC dataset found.", pd.DataFrame()

def view_clean_data():
    if Path(CLEAN_CSV).exists():
        df = pd.read_csv(CLEAN_CSV)
        return f"📊 Clean EC dataset: {len(df)} records.", df
    return "❌ No clean EC dataset found.", pd.DataFrame()

def clear_clean_data():
    if os.path.exists(CLEAN_CSV):
        os.remove(CLEAN_CSV)
        return "🗑️ Clean EC dataset cleared!", pd.DataFrame()
    return "⚠️ Already empty!", pd.DataFrame()

# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="MANIS — EC Data Cleaning Agent", theme=gr.themes.Soft()) as clean_app:

    gr.Markdown("# 🧹 MANIS — Electrical Conductivity Data Cleaning Agent")
    gr.Markdown(
        "Cleans `nanofluid_ec_data.csv` (Agent 1 output) → saves `nanofluid_ec_data_clean.csv` for Colab 2.  \n"
        "Preserves `group_id`, assigns `subgroup_id`, deduplicates, standardises text."
    )

    with gr.Tabs():

        # ── TAB 1: CLEANING ─────────────────────────────────────
        with gr.Tab("🧹 Data Cleaning"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Step 1 — Preview Raw Data")
                    view_raw_btn = gr.Button("👁️ Preview Raw Data", variant="secondary")
                    raw_status   = gr.Textbox(label="Raw Data Status", interactive=False)

                    gr.Markdown("### Step 2 — Check group_id Consistency")
                    check_btn    = gr.Button("🔍 Check group_id", variant="secondary")
                    check_status = gr.Textbox(label="Consistency Check", interactive=False)

                    gr.Markdown("### Step 3 — Run Cleaning")
                    clean_btn    = gr.Button("🧹 Run Cleaning Pipeline",
                                             variant="primary", size="lg")
                    clean_status = gr.Textbox(label="Cleaning Status",
                                              interactive=False, lines=12)
                with gr.Column(scale=2):
                    raw_preview = gr.Dataframe(label="Raw Data Preview")

            gr.Markdown("### Cleaned Data Preview")
            clean_preview = gr.Dataframe(label="Clean Data — First 20 rows")

        # ── TAB 2: AI FEEDBACK ──────────────────────────────────
        with gr.Tab("🤖 AI Feedback"):
            gr.Markdown("### AI Analysis of Your Electrical Conductivity Dataset")
            ai_btn      = gr.Button("🤖 Generate AI Feedback", variant="primary", size="lg")
            ai_feedback = gr.Textbox(label="AI Feedback", interactive=False, lines=30)

        # ── TAB 3: GROUP SUMMARY ────────────────────────────────
        with gr.Tab("📊 Group Summary"):
            summary_btn    = gr.Button("📊 Generate Summary", variant="primary")
            summary_output = gr.Textbox(label="Group Summary", interactive=False, lines=30)

            gr.Markdown("### Explore Specific Group")
            with gr.Row():
                group_id_input = gr.Number(label="Enter Group ID", value=1)
                group_btn      = gr.Button("🔍 View Group", variant="secondary")
            group_info  = gr.Textbox(label="Group Info", interactive=False)
            group_table = gr.Dataframe(label="Group Data")

        # ── TAB 4: GOOGLE DRIVE ─────────────────────────────────
        with gr.Tab("☁️ Google Drive"):
            gr.Markdown(f"Reads: `{DATA_CSV}`  \nWrites: `{CLEAN_CSV}`")
            with gr.Row():
                mount_btn      = gr.Button("🔗 Mount / Remount Drive", variant="secondary")
                check_drive_btn= gr.Button("🔄 Check Drive Status",    variant="secondary")
            drive_status = gr.Textbox(label="Drive Status", interactive=False)

            def check_drive():
                if not os.path.exists(DRIVE_DIR):
                    return "❌ Drive not mounted."
                files = [f for f in os.listdir(DRIVE_DIR) if f.endswith('.csv')]
                if not files:
                    return "✅ Drive mounted — no CSV files yet."
                lines = ["✅ Drive mounted:"]
                for f in sorted(files):
                    lines.append(f"  📄 {f} ({os.path.getsize(DRIVE_DIR+f):,} bytes)")
                return "\n".join(lines)

            def do_mount():
                mount_drive()
                return check_drive()

            mount_btn.click(do_mount,       outputs=[drive_status])
            check_drive_btn.click(check_drive, outputs=[drive_status])

        # ── TAB 5: DATASET MANAGEMENT ───────────────────────────
        with gr.Tab("📁 Dataset Management"):
            with gr.Row():
                view_clean_btn  = gr.Button("👁️ View Clean Data",  variant="secondary")
                clear_clean_btn = gr.Button("🗑️ Clear Clean Data", variant="stop")
            clean_data_status = gr.Textbox(label="Status", interactive=False)
            clean_full_table  = gr.Dataframe(label="Clean Dataset")

    # ── EVENT WIRING ──────────────────────────────────────────

    view_raw_btn.click(view_raw_data,
                       outputs=[raw_status, raw_preview])
    check_btn.click(check_group_id_consistency,
                    outputs=[check_status])
    clean_btn.click(run_cleaning_and_show,
                    outputs=[clean_status, clean_preview])
    ai_btn.click(generate_ai_feedback,
                 outputs=[ai_feedback])
    summary_btn.click(generate_summary_tab,
                      outputs=[summary_output])
    group_btn.click(get_group_detail,
                    inputs=[group_id_input],
                    outputs=[group_info, group_table])
    view_clean_btn.click(view_clean_data,
                         outputs=[clean_data_status, clean_full_table])
    clear_clean_btn.click(clear_clean_data,
                          outputs=[clean_data_status, clean_full_table])

clean_app.launch(share=True)
