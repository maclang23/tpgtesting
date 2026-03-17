import streamlit as st
import streamlit.components.v1 as components
import json
import os
import base64
import pandas as pd
import numpy as np

st.set_page_config(page_title="Gauntlet Timelapse", layout="wide")

CSV_PATH    = "roundlist.csv"
INDEX_PATH  = "static/player_index.json"
RANKS_DIR   = "static/ranks"
GRID_STEP   = 1.0

# ─────────────────────────────────────────────
# Loading Logic
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_meta():
    if not os.path.exists(CSV_PATH) or not os.path.exists(INDEX_PATH):
        return None
    
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    round_cols  = [c for c in df.columns if c.startswith("Round")]
    
    loc_row     = df[df["Name"] == "Location"].iloc[0]
    player_rows = df[df["Name"] != "Location"].copy()
    all_players = player_rows["Name"].str.strip().tolist()

    def is_active(val):
        return not pd.isna(val) and str(val).strip() != ""

    round_targets = {}
    for col in round_cols:
        val = loc_row[col]
        if not pd.isna(val) and str(val).strip():
            try:
                parts = str(val).split(",")
                round_targets[col] = [float(parts[0].strip()), float(parts[1].strip())]
            except: pass

    round_active = {}
    for col in round_cols:
        active = player_rows.loc[player_rows[col].apply(is_active), "Name"].str.strip().tolist()
        round_active[col] = active

    with open(INDEX_PATH) as f:
        name_to_file = json.load(f)

    return all_players, round_cols, round_targets, round_active, name_to_file

meta_data = load_meta()
if not meta_data:
    st.error("Missing files. Please ensure `roundlist.csv` and `static/player_index.json` exist.")
    st.stop()

all_players, round_cols, round_targets, round_active, name_to_file = meta_data

@st.cache_data(show_spinner=False)
def load_player_b64(player_name):
    safe = name_to_file.get(player_name)
    fpath = os.path.join(RANKS_DIR, f"{safe}.bin")
    if not os.path.exists(fpath): return None
    with open(fpath, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("Gauntlet Timelapse")

players_with_data = [p for p in all_players if name_to_file.get(p) and
                     os.path.exists(os.path.join(RANKS_DIR, name_to_file[p] + ".bin"))]

selected_player = st.selectbox("Select Player", sorted(players_with_data))
b64_data = load_player_b64(selected_player)

lats = np.arange(-90, 91, GRID_STEP)
lons = np.arange(-180, 181, GRID_STEP)
player_rounds = [col for col in round_cols if selected_player in round_active.get(col, [])]

meta = {
    "n_lat": len(lats), "n_lon": len(lons),
    "round_targets": {col: round_targets.get(col) for col in player_rounds},
}
meta_js = json.dumps(meta)

# ─────────────────────────────────────────────
# JS/HTML
# ─────────────────────────────────────────────
HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
    <style>
        body {{ margin:0; padding:0; background:#1a1a2e; color:white; font-family:sans-serif; }}
        #map {{ position:absolute; top:50px; bottom:0; width:100%; background:#000; }}
        #controls {{ height:50px; display:flex; align-items:center; padding:0 15px; gap:15px; background:#16213e; }}
        #overlay {{ position:absolute; inset:0; background:rgba(0,0,0,0.8); z-index:10; 
                    display:flex; justify-content:center; align-items:center; }}
        .hidden {{ display:none !important; }}
    </style>
</head>
<body>
    <div id="controls">
        <label>Round</label>
        <input type="range" id="round-sl" min="0" value="0" style="width:300px">
        <span id="round-val">0</span>
    </div>
    <div id="map-wrap">
        <div id="overlay">Loading Map Data...</div>
        <div id="map"></div>
    </div>

<script>
const META = {meta_js};
const B64 = "{b64_data}";

function decodeBinary(b64) {{
    try {{
        const bin = atob(b64);
        const buf = new ArrayBuffer(bin.length);
        const u8 = new Uint8Array(buf);
        for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
        const view = new DataView(buf);
        const n_rounds = view.getUint16(0, true);
        const ng = view.getUint32(2, true);
        const rounds = [];
        let offset = 6;
        for (let i = 0; i < n_rounds; i++) {{
            offset += 4; // skip rnum and n_active
            rounds.push(new Uint8Array(buf, offset, ng));
            offset += ng;
        }}
        return rounds;
    }} catch(e) {{ console.error("Decode Error:", e); return []; }}
}}

const roundData = decodeBinary(B64);
const map = new maplibregl.Map({{
    container: 'map',
    style: 'https://demotiles.maplibre.org/style.json', // Basic style
    center: [0, 20], zoom: 1.5
}});

let canvas, ctx, imgData, sourceAdded = false;

map.on('load', () => {{
    console.log("Map Loaded");
    document.getElementById('overlay').classList.add('hidden');
    
    canvas = document.createElement('canvas');
    canvas.width = META.n_lon; 
    canvas.height = META.n_lat;
    ctx = canvas.getContext('2d');
    imgData = ctx.createImageData(canvas.width, canvas.height);
    
    document.getElementById('round-sl').max = roundData.length - 1;
    update(0);
}});

function update(idx) {{
    if (!roundData[idx]) return;
    const ranks = roundData[idx];
    const d = imgData.data;
    
    // Simple heatmap paint
    for (let i = 0; i < ranks.length; i++) {{
        const p = i * 4;
        const r = ranks[i];
        d[p] = r === 0 ? 0 : 255;   // Red channel
        d[p+1] = r === 0 ? 255 : 0; // Green channel
        d[p+3] = r < 10 ? 200 : 0;  // Alpha
    }}
    
    ctx.putImageData(imgData, 0, 0);
    const url = canvas.toDataURL();
    const coords = [[-180, 90], [180, 90], [180, -90], [-180, -90]];

    if (!sourceAdded) {{
        map.addSource('rk', {{ type: 'image', url: url, coordinates: coords }});
        map.addLayer({{ id: 'rk-layer', type: 'raster', source: 'rk' }});
        sourceAdded = true;
    }} else {{
        map.getSource('rk').updateImage({{ url: url, coordinates: coords }});
    }}
    document.getElementById('round-val').textContent = idx;
}}

document.getElementById('round-sl').oninput = (e) => update(e.target.value);
</script>
</body>
</html>
"""

components.html(HTML, height=600)
