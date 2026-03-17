import streamlit as st
import streamlit.components.v1 as components
import json, os, base64
import pandas as pd
import numpy as np

st.set_page_config(page_title="Gauntlet Timelapse", layout="wide")

CSV_PATH    = "roundlist.csv"
INDEX_PATH  = "static/player_index.json"
RANKS_DIR   = "static/ranks"

@st.cache_data(show_spinner=False)
def load_meta():
    if not os.path.exists(CSV_PATH) or not os.path.exists(INDEX_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    round_cols = [c for c in df.columns if c.startswith("Round")]
    
    with open(INDEX_PATH) as f:
        name_to_file = json.load(f)
    return sorted(list(name_to_file.keys())), round_cols, name_to_file

meta = load_meta()
if not meta:
    st.error("Missing files in static/ directory.")
    st.stop()

players, round_cols, name_to_file = meta
selected_player = st.selectbox("Select Player", players)

@st.cache_data(show_spinner=False)
def get_player_b64(name):
    fname = name_to_file.get(name)
    fpath = os.path.join(RANKS_DIR, f"{fname}.bin")
    if os.path.exists(fpath):
        with open(fpath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

b64_data = get_player_b64(selected_player)

# JS Metadata (1.0 degree grid = 361x181)
meta_js = json.dumps({"n_lat": 181, "n_lon": 361})

HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
    <style>
        body {{ margin:0; padding:0; background:#000; font-family: sans-serif; height:100vh; overflow:hidden; }}
        #ui {{ position:absolute; top:0; z-index:10; width:100%; background:rgba(20,20,20,0.9); 
               height:60px; display:flex; align-items:center; padding:0 20px; gap:20px; color:white; border-bottom:1px solid #444; }}
        #map {{ position:absolute; top:60px; bottom:0; width:100%; }}
        input[type=range] {{ width: 300px; }}
        .btn {{ background:#e94560; color:white; border:none; padding:8px 16px; border-radius:4px; cursor:pointer; font-weight:bold; }}
    </style>
</head>
<body>
    <div id="ui">
        <button id="play" class="btn">▶ Play</button>
        <input type="range" id="slider" min="0" value="0">
        <div id="info">Loading binary data...</div>
    </div>
    <div id="map"></div>

<script>
const META = {meta_js};
const B64 = "{b64_data}";

const map = new maplibregl.Map({{
    container: 'map',
    style: {{
        "version": 8,
        "sources": {{ "osm": {{ "type": "raster", "tiles": ["https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png"], "tileSize": 256 }} }},
        "layers": [{{ "id": "osm", "type": "raster", "source": "osm" }}]
    }},
    center: [0, 20], zoom: 1.5,
    renderWorldCopies: false
}});

function decode(b64) {{
    const binStr = atob(b64);
    const len = binStr.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binStr.charCodeAt(i);
    
    const view = new DataView(bytes.buffer);
    const nRounds = view.getUint16(0, true);
    const gridSize = view.getUint32(2, true);
    
    console.log("Header -> Rounds:", nRounds, "GridSize:", gridSize);
    
    let rounds = [];
    let offset = 6;
    for(let i=0; i < nRounds; i++) {{
        if (offset + 4 + gridSize > len) break;
        const rnum = view.getUint16(offset, true); offset += 2;
        const nActive = view.getUint16(offset, true); offset += 2;
        const ranks = bytes.slice(offset, offset + gridSize);
        rounds.push({{ rnum, nActive, ranks }});
        offset += gridSize;
    }}
    return rounds;
}}

const roundData = decode(B64);
let canvas, ctx, sourceAdded = false;

map.on('load', () => {{
    map.resize();
    canvas = document.createElement('canvas');
    canvas.width = META.n_lon; canvas.height = META.n_lat;
    ctx = canvas.getContext('2d');
    
    document.getElementById('slider').max = roundData.length - 1;
    render(0);
}});

function render(idx) {{
    const data = roundData[idx];
    if(!data) return;
    
    const imgData = ctx.createImageData(canvas.width, canvas.height);
    const ranks = data.ranks;
    const nActive = data.nActive;

    for(let i=0; i < ranks.length; i++) {{
        const r = ranks[i];
        if (r === 255) continue; // Out of bounds/NaN

        const x = i % META.n_lon;
        const y = META.n_lat - 1 - Math.floor(i / META.n_lon); // Flip Y to match map
        const pos = (y * META.n_lon + x) * 4;
        
        // Heatmap Logic
        if (r === 0) {{ // 1st Place (Bright Green)
            imgData.data[pos]=0; imgData.data[pos+1]=255; imgData.data[pos+2]=0; imgData.data[pos+3]=220;
        }} else if (r < 5) {{ // Top 5 (Yellow)
            imgData.data[pos]=255; imgData.data[pos+1]=255; imgData.data[pos+2]=0; imgData.data[pos+3]=160;
        }} else if (r >= nActive - 1) {{ // Last Place (Red)
            imgData.data[pos]=255; imgData.data[pos+1]=0; imgData.data[pos+2]=0; imgData.data[pos+3]=180;
        }}
    }}
    
    ctx.putImageData(imgData, 0, 0);
    const url = canvas.toDataURL();
    const coords = [[-179.9, 89.9], [179.9, 89.9], [179.9, -89.9], [-179.9, -89.9]];

    if(!sourceAdded) {{
        map.addSource('ranks', {{ type: 'image', url: url, coordinates: coords }});
        map.addLayer({{ id: 'ranks-layer', type: 'raster', source: 'ranks', paint: {{ 'raster-fade-duration': 0 }} }});
        sourceAdded = true;
    }} else {{
        map.getSource('ranks').updateImage({{ url: url, coordinates: coords }});
    }}
    
    document.getElementById('info').textContent = "Round " + data.rnum + " | Active: " + nActive;
}}

document.getElementById('slider').oninput = (e) => render(parseInt(e.target.value));

let playInt;
document.getElementById('play').onclick = function() {{
    if(playInt) {{
        clearInterval(playInt); playInt = null;
        this.textContent = "▶ Play";
    }} else {{
        this.textContent = "⏸ Pause";
        playInt = setInterval(() => {{
            let s = document.getElementById('slider');
            let next = (parseInt(s.value) + 1) % roundData.length;
            s.value = next;
            render(next);
        }}, 400);
    }}
}};
</script>
</body>
</html>
"""

components.html(HTML, height=750)
