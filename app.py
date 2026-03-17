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
    loc_row     = df[df["Name"] == "Location"].iloc[0]
    player_rows = df[df["Name"] != "Location"].copy()
    all_players = player_rows["Name"].str.strip().tolist()

    round_targets = {}
    for col in round_cols:
        val = loc_row[col]
        if not pd.isna(val) and str(val).strip():
            try:
                parts = str(val).split(",")
                round_targets[col] = [float(parts[1].strip()), float(parts[0].strip())] # [lon, lat]
            except: pass

    round_active = {}
    for col in round_cols:
        active = player_rows.loc[~player_rows[col].isna(), "Name"].str.strip().tolist()
        round_active[col] = active

    with open(INDEX_PATH) as f:
        name_to_file = json.load(f)

    return all_players, round_cols, round_targets, round_active, name_to_file

meta_data = load_meta()
if not meta_data:
    st.error("Missing files. Check `roundlist.csv` and `static/` folder.")
    st.stop()

all_players, round_cols, round_targets, round_active, name_to_file = meta_data

@st.cache_data(show_spinner=False)
def load_player_b64(player_name):
    safe = name_to_file.get(player_name)
    fpath = os.path.join(RANKS_DIR, f"{safe}.bin")
    if not os.path.exists(fpath): return None
    with open(fpath, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

st.title("Gauntlet Timelapse")

players_with_data = [p for p in all_players if name_to_file.get(p) and 
                     os.path.exists(os.path.join(RANKS_DIR, name_to_file[p] + ".bin"))]

selected_player = st.selectbox("Select Player", sorted(players_with_data))
b64_data = load_player_b64(selected_player)

lats = np.arange(-90, 91, GRID_STEP)
lons = np.arange(-180, 181, GRID_STEP)

meta = {
    "n_lat": len(lats),
    "n_lon": len(lons),
    "round_targets": round_targets,
    "round_names": round_cols
}
meta_js = json.dumps(meta)

HTML = f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
    <style>
        body {{ margin:0; padding:0; background:#111; color:white; font-family:sans-serif; height:100vh; display:flex; flex-direction:column; }}
        #controls {{ height:50px; background:#222; display:flex; align-items:center; padding:0 15px; gap:15px; border-bottom:1px solid #444; }}
        #map {{ flex:1; width:100%; }}
        .stat {{ font-size:0.8rem; background:#333; padding:4px 8px; border-radius:4px; }}
        .val {{ color:#00ff00; font-weight:bold; }}
    </style>
</head>
<body>
    <div id="controls">
        <button id="play-btn" style="padding:5px 10px; cursor:pointer;">▶ Play</button>
        <input type="range" id="round-sl" min="0" value="0" style="width:300px">
        <div class="stat">Round: <span class="val" id="s-round">0</span></div>
        <div class="stat">Active Players: <span class="val" id="s-active">0</span></div>
    </div>
    <div id="map"></div>

<script>
const META = {meta_js};
const B64 = "{b64_data}";

const style = {{
    "version": 8,
    "sources": {{ "osm": {{ "type": "raster", "tiles": ["https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png"], "tileSize": 256 }} }},
    "layers": [{{ "id": "osm", "type": "raster", "source": "osm" }}]
}};

function decodeBinary(b64) {{
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
        const rnum = view.getUint16(offset, true); offset += 2;
        const n_active = view.getUint16(offset, true); offset += 2;
        const ranks = new Uint8Array(buf, offset, ng); offset += ng;
        rounds.push({{ rnum, n_active, ranks }});
    }}
    return rounds;
}}

const roundData = decodeBinary(B64);
const map = new maplibregl.Map({{
    container: 'map',
    style: style,
    center: [0, 20],
    zoom: 1.5,
    attributionControl: false
}});

let canvas, ctx, imgData, sourceAdded = false, playInterval = null, curIdx = 0;

map.on('load', () => {{
    map.resize();
    canvas = document.createElement('canvas');
    canvas.width = META.n_lon; canvas.height = META.n_lat;
    ctx = canvas.getContext('2d');
    imgData = ctx.createImageData(canvas.width, canvas.height);
    
    document.getElementById('round-sl').max = roundData.length - 1;
    update(0);
}});

function update(idx) {{
    if (!roundData[idx]) return;
    const {{ rnum, n_active, ranks }} = roundData[idx];
    const d = imgData.data;
    
    // Fill background with subtle transparency
    for (let i = 0; i < d.length; i += 4) d[i+3] = 0;

    for (let i = 0; i < ranks.length; i++) {{
        const r = ranks[i];
        const x = i % META.n_lon;
        const y = META.n_lat - 1 - Math.floor(i / META.n_lon);
        const cp = (y * META.n_lon + x) * 4;

        if (r === 0) {{ // Winner (Green)
            d[cp]=0; d[cp+1]=255; d[cp+2]=0; d[cp+3]=200;
        }} else if (r <= 5) {{ // Top 5 (Yellow)
            d[cp]=255; d[cp+1]=255; d[cp+2]=0; d[cp+3]=150;
        }} else if (r >= n_active - 1) {{ // Last Place (Red)
            d[cp]=255; d[cp+1]=0; d[cp+2]=0; d[cp+3]=180;
        }}
    }}
    
    ctx.putImageData(imgData, 0, 0);
    const url = canvas.toDataURL();
    const coords = [[-180, 90], [180, 90], [180, -90], [-180, -90]];

    if (!sourceAdded) {{
        map.addSource('rk', {{ type: 'image', url: url, coordinates: coords }});
        map.addLayer({{ id: 'rk-layer', type: 'raster', source: 'rk' }});
        
        // Add round target point
        map.addSource('tgt', {{ type: 'geojson', data: {{ type: 'FeatureCollection', features: [] }} }});
        map.addLayer({{ id: 'tgt-layer', type: 'circle', source: 'tgt', paint: {{ 'circle-radius': 8, 'circle-color': '#fff', 'circle-stroke-width': 2 }} }});
        sourceAdded = true;
    }} else {{
        map.getSource('rk').updateImage({{ url, coordinates: coords }});
    }}

    // Update target point
    const roundName = "Round " + rnum;
    const tgt = META.round_targets[roundName];
    if (tgt) {{
        map.getSource('tgt').setData({{
            type: 'Feature',
            geometry: {{ type: 'Point', coordinates: tgt }}
        }});
    }}

    document.getElementById('s-round').textContent = rnum;
    document.getElementById('s-active').textContent = n_active;
    curIdx = idx;
}}

document.getElementById('round-sl').oninput = (e) => update(parseInt(e.target.value));
document.getElementById('play-btn').onclick = function() {{
    if (playInterval) {{
        clearInterval(playInterval); playInterval = null;
        this.textContent = "▶ Play";
    }} else {{
        this.textContent = "⏸ Pause";
        playInterval = setInterval(() => {{
            curIdx = (curIdx + 1) % roundData.length;
            document.getElementById('round-sl').value = curIdx;
            update(curIdx);
        }}, 400);
    }}
}};
</script>
</body>
</html>
"""

components.html(HTML, height=750, scrolling=False)
