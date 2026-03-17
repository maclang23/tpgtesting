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

@st.cache_data(show_spinner=False)
def load_meta():
    if not os.path.exists(CSV_PATH) or not os.path.exists(INDEX_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    round_cols  = [c for c in df.columns if c.startswith("Round")]
    player_rows = df[df["Name"] != "Location"].copy()
    all_players = player_rows["Name"].str.strip().tolist()
    
    with open(INDEX_PATH) as f:
        name_to_file = json.load(f)

    # Simplified active player check for metadata
    round_active = {}
    for col in round_cols:
        active = player_rows.loc[~player_rows[col].isna(), "Name"].str.strip().tolist()
        round_active[col] = active

    return all_players, round_cols, round_active, name_to_file

meta_data = load_meta()
if not meta_data:
    st.error("Setup incomplete. Ensure `roundlist.csv` and `static/player_index.json` exist.")
    st.stop()

all_players, round_cols, round_active, name_to_file = meta_data

@st.cache_data(show_spinner=False)
def load_player_b64(player_name):
    safe = name_to_file.get(player_name)
    fpath = os.path.join(RANKS_DIR, f"{safe}.bin")
    if not os.path.exists(fpath): return ""
    with open(fpath, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

st.title("Gauntlet Timelapse")

players_with_data = [p for p in all_players if name_to_file.get(p) and 
                     os.path.exists(os.path.join(RANKS_DIR, name_to_file[p] + ".bin"))]

selected_player = st.selectbox("Select Player", sorted(players_with_data))
b64_data = load_player_b64(selected_player)

lats = np.arange(-90, 91, GRID_STEP)
lons = np.arange(-180, 181, GRID_STEP)

meta_js = json.dumps({
    "n_lat": len(lats),
    "n_lon": len(lons)
})

HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
    <style>
        body {{ margin:0; padding:0; background:#000; overflow:hidden; }}
        #map {{ position:absolute; top:40px; bottom:0; width:100%; height:calc(100vh - 40px); }}
        #controls {{ height:40px; background:#111; display:flex; align-items:center; padding:0 15px; color:white; gap:10px; }}
        #overlay {{ position:absolute; inset:0; background:#000; z-index:100; display:flex; justify-content:center; align-items:center; color:white; }}
        .hidden {{ display:none !important; }}
    </style>
</head>
<body>
    <div id="controls">
        <label>Round</label>
        <input type="range" id="round-sl" min="0" value="0" style="width:70%">
        <span id="round-val">0</span>
    </div>
    <div id="overlay">Initializing...</div>
    <div id="map"></div>

<script>
const META = {meta_js};
const B64 = "{b64_data}";

// Minimal local style to avoid projection issues
const style = {{
    "version": 8,
    "sources": {{}},
    "layers": [{{ "id": "background", "type": "background", "paint": {{ "background-color": "#111" }} }}]
}};

function decodeBinary(b64) {{
    if(!b64) return [];
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
        offset += 4; // skip rnum/n_active
        rounds.push(new Uint8Array(buf, offset, ng));
        offset += ng;
    }}
    return rounds;
}}

const roundData = decodeBinary(B64);
const map = new maplibregl.Map({{
    container: 'map',
    style: style,
    center: [0, 0],
    zoom: 1,
    fadeDuration: 0
}});

let canvas, ctx, imgData, sourceAdded = false;

map.on('load', () => {{
    console.log("Map Ready");
    map.resize(); // Fix for the "y=Infinity" error
    
    document.getElementById('overlay').classList.add('hidden');
    
    canvas = document.createElement('canvas');
    canvas.width = META.n_lon; 
    canvas.height = META.n_lat;
    ctx = canvas.getContext('2d');
    imgData = ctx.createImageData(canvas.width, canvas.height);
    
    const slider = document.getElementById('round-sl');
    slider.max = Math.max(0, roundData.length - 1);
    
    update(0);
}});

function update(idx) {{
    if (!roundData[idx]) return;
    const ranks = roundData[idx];
    const d = imgData.data;
    
    for (let i = 0; i < ranks.length; i++) {{
        const p = i * 4;
        const r = ranks[i];
        if (r === 0) {{ // 1st place
            d[p]=0; d[p+1]=255; d[p+2]=0; d[p+3]=200;
        }} else if (r < 5) {{ // Top 5
            d[p]=255; d[p+1]=255; d[p+2]=0; d[p+3]=150;
        }} else if (r > 200) {{ // High rank (example)
            d[p]=255; d[p+1]=0; d[p+2]=0; d[p+3]=100;
        }} else {{
            d[p+3]=0; // Transparent
        }}
    }}
    
    ctx.putImageData(imgData, 0, 0);
    const url = canvas.toDataURL();
    // Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    const coords = [[-180, 90], [180, 90], [180, -90], [-180, -90]];

    if (!sourceAdded) {{
        map.addSource('rk', {{ type: 'image', url: url, coordinates: coords }});
        map.addLayer({{ id: 'rk-layer', type: 'raster', source: 'rk', paint: {{ 'raster-fade-duration': 0 }} }});
        sourceAdded = true;
    }} else {{
        map.getSource('rk').updateImage({{ url: url, coordinates: coords }});
    }}
    document.getElementById('round-val').textContent = idx;
}}

document.getElementById('round-sl').oninput = (e) => update(parseInt(e.target.value));
</script>
</body>
</html>
"""

components.html(HTML, height=600)
