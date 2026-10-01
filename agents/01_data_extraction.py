# ============================================================
# MANIS SYSTEM — ELECTRICAL CONDUCTIVITY
# AGENT 1: DATA EXTRACTION AI AGENT
# ============================================================
# Adapted from the thermal-conductivity Agent 1 (v4)
# Target property : electrical_conductivity_uScm
# New CSV         : nanofluid_ec_data.csv
# Columns         : nanoparticle, base_fluid, particle_size_nm,
#                   volume_fraction, temperature_K,
#                   electrical_conductivity_uScm
# ============================================================

import anthropic
import base64
import json
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import io
import gradio as gr
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
DATA_CSV  = DRIVE_DIR + "nanofluid_ec_data.csv"

mount_drive()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── Dropdown options ──────────────────────────────────────────
NANOPARTICLES = [
    "Select...", "Al2O3", "CuO", "TiO2", "Fe3O4", "ZnO",
    "SiO2", "SiC", "hBN", "CNT", "MWCNT", "SWCNT",
    "Graphene", "Nano-diamond", "Fe2O3", "MgO", "Ag",
    "Al", "ZrO2", "Fe","Cu","Mg(OH)2","AlN","CaCO3","Other"
]

BASE_FLUIDS = [
    "Select...", "Water", "Ethylene Glycol",
] + [
    f"Water:EG {100-eg}:{eg}" for eg in range(10, 100, 10)
] + [
    "Synthetic Oil", "Engine Oil", "Propylene Glycol",
    "Glycerol", "Acetone", "Diatheric Oil",
    "Heat Transfer Oil", "Transformer Oil","Bio Glycol", "PEG200", "Water:PG 40:60","Other"
]

# ── Column definitions ────────────────────────────────────────
DATA_COLS = [
    "nanoparticle", "base_fluid", "particle_size_nm",
    "volume_fraction", "temperature_K",
    "electrical_conductivity_uScm"
]

ALL_COLS = ["group_id"] + DATA_COLS

# ============================================================
# GROUP ID MANAGEMENT
# ============================================================

def get_existing_groups():
    if not Path(DATA_CSV).exists():
        return []
    df = pd.read_csv(DATA_CSV)
    if "group_id" not in df.columns or df.empty:
        return []
    return sorted(df["group_id"].unique().tolist())

def get_next_group_id():
    existing = get_existing_groups()
    if not existing:
        return 1
    for i in range(1, max(existing) + 2):
        if i not in existing:
            return i

def get_deleted_group_ids():
    existing = get_existing_groups()
    if not existing:
        return []
    return sorted(set(range(1, max(existing) + 1)) - set(existing))

def load_data():
    if not Path(DATA_CSV).exists():
        return pd.DataFrame(columns=ALL_COLS)
    df = pd.read_csv(DATA_CSV)
    return df.sort_values("group_id").reset_index(drop=True)

def save_data(df):
    df = df.sort_values("group_id").reset_index(drop=True)
    df.to_csv(DATA_CSV, index=False)

# ============================================================
# IMAGE ENCODING & JSON PARSING
# ============================================================

def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

def parse_response(text):
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1]
        if "```" in text:
            text = text.split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    last_complete = text.rfind('},')
    if last_complete != -1 and not text.rstrip().endswith(']}}'):
        text = text[:last_complete+1] + '\n    ]\n}'
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"data": [], "error": str(e)}

# ============================================================
# EXTRACTION FUNCTIONS
# ============================================================

EC_JSON_EXAMPLE = """{
    "nanoparticle": "ZnO",
    "base_fluid": "Water",
    "particle_size_nm": 30,
    "volume_fraction": 0.03,
    "temperature_K": 298.15,
    "electrical_conductivity_uScm": 42.7
}"""

def extract_from_table(image, nanoparticle=None, base_fluid=None):
    known = {}
    if nanoparticle and nanoparticle != "Select...":
        known["nanoparticle"] = nanoparticle
    if base_fluid and base_fluid != "Select...":
        known["base_fluid"] = base_fluid
    known_str = f"\nUse these known properties for ALL rows: {json.dumps(known)}" if known else ""

    prompt = f"""You are a nanofluid research expert.
Extract ALL electrical conductivity data from this table image.
{known_str}

Return ONLY this JSON format:
{{
    "data": [
        {EC_JSON_EXAMPLE}
    ]
}}

Rules:
- Extract EVERY row from the table
- Target property is ELECTRICAL CONDUCTIVITY in microsiemens per centimeter (uS/cm)
- If conductivity is reported in a different unit, convert to uS/cm:
    1 S/m = 10,000 uS/cm
    1 mS/cm = 1,000 uS/cm
    1 uS/m = 0.01 uS/cm
- If temperature is in Celsius, convert to Kelvin (+273.15)
- If volume fraction is given as a percentage (e.g. 3%), convert to decimal (0.03)
- Use null for missing values
- Return ONLY valid JSON, nothing else"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=16000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/png", "data": encode_image(image)}},
            {"type": "text", "text": prompt}
        ]}],
    )
    return parse_response(response.content[0].text)


def extract_from_figure(image, nanoparticle=None, base_fluid=None):
    known = {}
    if nanoparticle and nanoparticle != "Select...":
        known["nanoparticle"] = nanoparticle
    if base_fluid and base_fluid != "Select...":
        known["base_fluid"] = base_fluid
    known_str = f"\nKnown properties for ALL data points: {json.dumps(known)}" if known else ""

    prompt = f"""You are a nanofluid research expert and graph digitizer.
This is a scientific figure showing ELECTRICAL CONDUCTIVITY data.
{known_str}

Carefully read the graph axes, scale and legend, then extract every data point.

Return ONLY this JSON format:
{{
    "data": [
        {EC_JSON_EXAMPLE}
    ]
}}

Rules:
- Extract EVERY visible data point from ALL curves in the legend
- Y-axis is electrical conductivity, commonly in uS/cm (microsiemens/cm) — also watch
  for mS/cm or S/m and convert everything to uS/cm:
    1 S/m = 10,000 uS/cm
    1 mS/cm = 1,000 uS/cm
- X-axis may be volume fraction (%), concentration, or temperature — read carefully
- If volume fraction shown as a percentage (e.g. 0.5%), convert to decimal (0.005)
- If temperature is in Celsius, convert to Kelvin (+273.15)
- Each curve in the legend = different temperature or condition — extract ALL
- Use null for properties not visible in the figure
- Return ONLY valid JSON, nothing else"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=8000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/png", "data": encode_image(image)}},
            {"type": "text", "text": prompt}
        ]}],
    )
    return parse_response(response.content[0].text)


# ============================================================
# DATA PROCESSING
# ============================================================

def process_image(image, image_type, nanoparticle, base_fluid):
    if image is None:
        return "Please upload an image first!", None

    if image_type == "Figure/Graph":
        result = extract_from_figure(image, nanoparticle, base_fluid)
    else:
        result = extract_from_table(image, nanoparticle, base_fluid)

    data = result.get("data", [])
    if not data:
        return "No data extracted. Try a clearer image.", None

    df = pd.DataFrame(data)

    # Apply known values
    if nanoparticle and nanoparticle != "Select...":
        df["nanoparticle"] = nanoparticle
    if base_fluid and base_fluid != "Select...":
        df["base_fluid"] = base_fluid

    # Drop any extra columns not in our schema
    for col in list(df.columns):
        if col not in DATA_COLS:
            df = df.drop(columns=[col])

    # Ensure all columns exist
    for col in DATA_COLS:
        if col not in df.columns:
            df[col] = None

    df = df[DATA_COLS]

    return f"✅ Extracted {len(df)} rows. Review and edit below, then save.", df


def batch_fill(df, column, value):
    if df is None or column == "Select column..." or not value:
        return df, "Please select a column and enter a value"
    try:
        numeric_cols = ["particle_size_nm", "volume_fraction",
                        "temperature_K", "electrical_conductivity_uScm"]
        if column in numeric_cols:
            value = float(value)
        df[column] = df[column].where(df[column].notna(), value)
        return df, f"✅ Filled empty '{column}' cells with {value}"
    except Exception as e:
        return df, f"Error: {str(e)}"


def convert_units(df):
    """Convert any remaining Celsius temperatures to Kelvin."""
    if df is None:
        return df, "No data to convert"
    try:
        if "temperature_K" in df.columns:
            # If any value looks like Celsius (< 200), convert it
            mask = df["temperature_K"].notna() & (df["temperature_K"] < 200)
            df.loc[mask, "temperature_K"] = df.loc[mask, "temperature_K"] + 273.15
            converted = mask.sum()
            return df, f"✅ Converted {converted} temperature values from °C to K"
        return df, "No temperature column found"
    except Exception as e:
        return df, f"Error: {str(e)}"


def convert_conductivity_units(df, from_unit):
    """Convert a whole column of conductivity values into uS/cm."""
    if df is None:
        return df, "No data to convert"
    try:
        if "electrical_conductivity_uScm" not in df.columns:
            return df, "No electrical_conductivity_uScm column found"
        factor = {
            "S/m": 10000.0,
            "mS/cm": 1000.0,
            "uS/cm (no change)": 1.0,
        }.get(from_unit)
        if factor is None:
            return df, "Please select a source unit"
        mask = df["electrical_conductivity_uScm"].notna()
        df.loc[mask, "electrical_conductivity_uScm"] = (
            df.loc[mask, "electrical_conductivity_uScm"] * factor
        )
        return df, f"✅ Converted {mask.sum()} values from {from_unit} to uS/cm"
    except Exception as e:
        return df, f"Error: {str(e)}"


# ============================================================
# GROUP SAVE / DELETE
# ============================================================

def save_group(df, use_deleted_id, deleted_id_choice):
    if df is None or len(df) == 0:
        return "No data to save!", refresh_dataset_view()
    try:
        if use_deleted_id and deleted_id_choice and deleted_id_choice != "None available":
            group_id = int(deleted_id_choice.split(" ")[1])
        else:
            group_id = get_next_group_id()

        df = df.copy()
        df["group_id"] = group_id

        for col in ALL_COLS:
            if col not in df.columns:
                df[col] = None
        df = df[ALL_COLS]

        existing = load_data()
        existing = existing[existing["group_id"] != group_id]
        combined = pd.concat([existing, df], ignore_index=True)
        save_data(combined)
        return (f"✅ Saved as Group {group_id} — {len(df)} records. "
                f"Total: {len(combined)} records."), refresh_dataset_view()
    except Exception as e:
        return f"Error: {str(e)}", refresh_dataset_view()


def delete_group(group_id_str):
    if not group_id_str or group_id_str == "Select group...":
        return "Please select a group.", refresh_dataset_view(), refresh_group_dropdown()
    try:
        group_id = int(group_id_str.replace("Group ", ""))
        df = load_data()
        before = len(df)
        df = df[df["group_id"] != group_id]
        save_data(df)
        removed = before - len(df)
        return (f"✅ Deleted Group {group_id} — {removed} records removed. "
                f"Remaining: {len(df)}."), \
               refresh_dataset_view(), refresh_group_dropdown()
    except Exception as e:
        return f"Error: {str(e)}", refresh_dataset_view(), refresh_group_dropdown()


def refresh_dataset_view():
    return load_data()

def refresh_group_dropdown():
    groups = get_existing_groups()
    choices = [f"Group {g}" for g in groups] if groups else []
    return gr.Dropdown(choices=["Select group..."] + choices, value="Select group...")

def get_group_summary():
    df = load_data()
    if df.empty:
        return "No data yet.", pd.DataFrame()
    summary = df.groupby("group_id").agg(
        nanoparticle    =("nanoparticle",                   "first"),
        base_fluid      =("base_fluid",                     "first"),
        particle_size_nm=("particle_size_nm",                "first"),
        data_points     =("electrical_conductivity_uScm",    "count")
    ).reset_index()
    summary.columns = ["Group ID","Nanoparticle","Base Fluid",
                       "Particle Size (nm)","Data Points"]
    total = summary["Data Points"].sum()
    return f"{len(summary)} groups — {total} total data points", summary


# ============================================================
# GRADIO INTERFACE
# ============================================================

BATCH_COLS = ["Select column..."] + DATA_COLS
CONDUCTIVITY_UNITS = ["S/m", "mS/cm", "uS/cm (no change)"]

with gr.Blocks(title="MANIS — EC Data Extraction", theme=gr.themes.Soft()) as app:

    gr.Markdown("# ⚡ MANIS — Electrical Conductivity Data Extraction Agent")
    gr.Markdown(
        "Extract nanofluid **electrical conductivity** data from research paper images.  \n"
        f"Saves to `nanofluid_ec_data.csv` in Google Drive."
    )

    with gr.Tabs():

        # ── TAB 1: EXTRACT & SAVE ─────────────────────────────
        with gr.Tab("Extract Data"):

            with gr.Row():
                # LEFT
                with gr.Column(scale=1):
                    gr.Markdown("### Step 1: Upload Image")
                    image_input = gr.Image(type="pil", label="Upload Table or Figure")
                    image_type  = gr.Radio(
                        ["Table", "Figure/Graph"],
                        label="Image Type", value="Figure/Graph"
                    )
                    gr.Markdown("### Step 2: Nanofluid Info")
                    nanoparticle_dd = gr.Dropdown(
                        choices=NANOPARTICLES, value="Select...",
                        label="Nanoparticle Type"
                    )
                    base_fluid_dd = gr.Dropdown(
                        choices=BASE_FLUIDS, value="Select...",
                        label="Base Fluid"
                    )
                    extract_btn = gr.Button(
                        "🔍 Extract Data", variant="primary", size="lg"
                    )

                # RIGHT
                with gr.Column(scale=2):
                    status_msg = gr.Textbox(
                        label="Extraction Status", interactive=False
                    )
                    gr.Markdown("### Step 3: Review & Edit")
                    data_table = gr.Dataframe(
                        label="Extracted Data — Click any cell to edit",
                        interactive=True,
                        wrap=True,
                        headers=DATA_COLS,
                    )

                    with gr.Row():
                        batch_col = gr.Dropdown(
                            choices=BATCH_COLS,
                            value="Select column...",
                            label="Batch Fill — Column"
                        )
                        batch_val = gr.Textbox(label="Value to fill empty cells")
                        batch_btn = gr.Button("Fill Empty Cells", variant="secondary")

                    with gr.Row():
                        convert_btn = gr.Button("Convert °C → K", variant="secondary")
                        cond_unit_dd = gr.Dropdown(
                            choices=CONDUCTIVITY_UNITS,
                            value="uS/cm (no change)",
                            label="Convert conductivity FROM this unit"
                        )
                        convert_cond_btn = gr.Button(
                            "Convert conductivity → uS/cm", variant="secondary"
                        )

                    action_status = gr.Textbox(
                        label="Action Status", interactive=False
                    )

            gr.Markdown("### Step 4: Save as Group")
            with gr.Row():
                with gr.Column(scale=1):
                    use_deleted = gr.Checkbox(
                        label="Re-use a deleted group ID", value=False
                    )
                    deleted_dd = gr.Dropdown(
                        label="Select deleted group ID",
                        choices=["None available"],
                        value="None available",
                        visible=False
                    )
                with gr.Column(scale=2):
                    gr.Markdown(
                        "*Leave unchecked to auto-assign next available group ID*"
                    )

            save_btn    = gr.Button("💾 Save Group", variant="primary", size="lg")
            save_status = gr.Textbox(label="Save Status", interactive=False)

        # ── TAB 2: GROUP MANAGEMENT ───────────────────────────
        with gr.Tab("Group Management"):

            gr.Markdown("### All Groups Summary")
            refresh_btn    = gr.Button("🔄 Refresh", variant="secondary")
            summary_status = gr.Textbox(label="Summary", interactive=False)
            summary_table  = gr.Dataframe(label="Group Summary")

            gr.Markdown("---")
            gr.Markdown("### Delete a Group")
            with gr.Row():
                delete_dd  = gr.Dropdown(
                    choices=["Select group..."],
                    value="Select group...",
                    label="Select Group to Delete"
                )
                delete_btn = gr.Button("🗑️ Delete Group", variant="stop")
            delete_status = gr.Textbox(label="Delete Status", interactive=False)

        # ── TAB 3: GOOGLE DRIVE ───────────────────────────────
        with gr.Tab("Google Drive"):

            gr.Markdown("### Google Drive Status")
            gr.Markdown(f"Data saves to: `{DATA_CSV}`")
            with gr.Row():
                mount_btn         = gr.Button("Mount / Remount Drive", variant="secondary")
                refresh_drive_btn = gr.Button("Check Drive Status",    variant="secondary")
            drive_status = gr.Textbox(label="Drive Status", interactive=False)

            def check_drive():
                if os.path.exists(DRIVE_DIR):
                    files = [f for f in os.listdir(DRIVE_DIR) if f.endswith('.csv')]
                    if files:
                        msgs = ["✅ Drive mounted — project folder found"]
                        for f in files:
                            size = os.path.getsize(DRIVE_DIR + f)
                            msgs.append(f"  {f} ({size:,} bytes)")
                        return "\n".join(msgs)
                    return "✅ Drive mounted — folder exists but no CSV files yet."
                return "❌ Drive not mounted. Click Mount Drive."

            def do_mount():
                return check_drive() if mount_drive() else "❌ Mount failed."

            mount_btn.click(do_mount, outputs=[drive_status])
            refresh_drive_btn.click(check_drive, outputs=[drive_status])

        # ── TAB 4: FULL DATASET ───────────────────────────────
        with gr.Tab("Full Dataset"):

            gr.Markdown("### Complete Dataset — sorted by group ID")
            with gr.Row():
                view_btn      = gr.Button("🔄 Refresh Dataset", variant="secondary")
                clear_all_btn = gr.Button("🗑️ Clear ALL Data",  variant="stop")
            dataset_status = gr.Textbox(label="Status", interactive=False)
            full_dataset   = gr.Dataframe(label="All Records")

    # ── EVENT WIRING ──────────────────────────────────────────

    def toggle_deleted(checked):
        deleted = get_deleted_group_ids()
        choices = [f"Group {d}" for d in deleted] if deleted else ["None available"]
        return gr.Dropdown(visible=checked, choices=choices, value=choices[0])

    use_deleted.change(toggle_deleted, inputs=[use_deleted], outputs=[deleted_dd])

    extract_btn.click(
        process_image,
        inputs=[image_input, image_type, nanoparticle_dd, base_fluid_dd],
        outputs=[status_msg, data_table]
    )
    batch_btn.click(
        batch_fill,
        inputs=[data_table, batch_col, batch_val],
        outputs=[data_table, action_status]
    )
    convert_btn.click(
        convert_units,
        inputs=[data_table],
        outputs=[data_table, action_status]
    )
    convert_cond_btn.click(
        convert_conductivity_units,
        inputs=[data_table, cond_unit_dd],
        outputs=[data_table, action_status]
    )
    save_btn.click(
        save_group,
        inputs=[data_table, use_deleted, deleted_dd],
        outputs=[save_status, full_dataset]
    )
    refresh_btn.click(get_group_summary, outputs=[summary_status, summary_table])
    refresh_btn.click(
        lambda: gr.Dropdown(
            choices=["Select group..."] +
                    [f"Group {g}" for g in get_existing_groups()],
            value="Select group..."
        ),
        outputs=[delete_dd]
    )
    delete_btn.click(
        delete_group,
        inputs=[delete_dd],
        outputs=[delete_status, full_dataset, delete_dd]
    )
    view_btn.click(
        lambda: (f"{len(load_data())} total records", load_data()),
        outputs=[dataset_status, full_dataset]
    )

    def clear_all():
        if os.path.exists(DATA_CSV):
            os.remove(DATA_CSV)
        return "✅ All data cleared.", pd.DataFrame()

    clear_all_btn.click(clear_all, outputs=[dataset_status, full_dataset])

app.launch(share=True)
