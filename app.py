import streamlit as st
import streamlit.components.v1 as components
import json, os, base64
import pandas as pd
import numpy as np

st.set_page_config(page_title="Gauntlet Timelapse", layout="wide")

# Paths
CSV_PATH    = "roundlist.csv"
INDEX_PATH  = "static/player_index.json"
RANKS_DIR   = "static/ranks"

@st.cache_data(show_spinner=False)
def load_everything():
    if not os.path.exists(CSV_PATH) or not os.path.exists(INDEX_PATH):
        return None
    
    # Load Round Meta
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    round_cols = [c for c in df.columns if c.startswith("Round")]
    
    # Get Locations
    loc_row = df[df["Name"] == "Location"].iloc[0]
    round_targets = {}
    for col in round_cols:
        val = loc_row[col]
        if not pd.isna(val):
            try:
                # Expecting "lat, lon" in CSV
                p = str(val).split(",")
                round_targets[col] = [float(p[1].strip()), float(p[0].strip())] # [lon, lat]
            except: pass

    # Player Mapping
    with open(INDEX_PATH) as f:
        name_to_file = json.load(f)
    
    return sorted(list(name_to_file.keys())), round_cols, round_targets, name_to_file

data = load_everything()
if not data:
    st.error("Data files missing in static/ folder.")
    st.stop()

players, round_cols, round_targets, name_to_file = data
selected_player = st.selectbox("Select Player to View Win/Loss Regions", players)

def get_player_data(name):
    fname = name_to_file.get(name)
    fpath = os.path.join(RANKS_DIR, f"{fname}.bin")
    if os.path.exists(fpath):
        with open(fpath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

b64_data = get_player_data(selected_player)

# JS Metadata
meta_js = json.dumps({
    "targets": round_targets,
    "rounds": round_cols,
    "n_lat": 181,
    "n_lon": 361
})

HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
    <style>
        body {{ margin:0; padding:0; background:#111; font-family: sans-serif; }}
        #map {{ position:absolute; top:50px; bottom:0; width:100%; }}
        #ui {{ position:absolute; top:0; height:50px; width:100%; background:#222; 
               display:flex; align-items:center; padding:0 20px; gap:20px; color:white; }}
        input[type=range] {{ width: 400px; }}
    </style>
</head>
<body>
    <div id="ui">
        <button id="play" style="padding:5px 15px;">▶ Play</button>
        <input type="range" id="slider" min="0" value="0">
        <div id="info">Loading data...</div>
    </div>
    <div id="map"></div>

<script>
const META = {meta_js};
const B64 = "{b64_data}";

// Initialize Map
const map = new maplibregl.Map({{
    container: 'map',
    style: {{
        "version": 8,
        "sources": {{ "osm": {{ "type": "raster", "tiles": ["https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png"], "tileSize": 256 }} }},
        "layers": [{{ "id": "osm", "type": "raster", "source": "osm" }}]
    }},
    center: [0, 20], zoom: 1.5
}});

function decode(b64) {{
    const str = atob(b64);
    const buf = new Uint8Array(str.length);
    for(let i=0; i<str.length; i++) buf[i] = str.charCodeAt(i);
    const view = new DataView(buf.buffer);
    const nRounds = view.getUint16(0, true);
    const gridSize = view.getUint32(2, true);
    
    let rounds = [];
    let offset = 6;
    for(let i=0; i<nRounds; i++) {{
        offset += 2; // skip rnum
        const nActive = view.getUint16(offset, true); offset += 2;
        const ranks = buf.slice(offset, offset + gridSize); offset += gridSize;
        rounds.push({{ nActive, ranks }});
    }}
    return rounds;
}}

const data = decode(B64);
let canvas, ctx, sourceAdded = false;

map.on('load', () => {{
    map.resize();
    canvas = document.createElement('canvas');
    canvas.width = META.n_lon; canvas.height = META.n_lat;
    ctx = canvas.getContext('2d');
    
    document.getElementById('slider').max = data.length - 1;
    document.getElementById('info').textContent = "Ready";
    render(0);
}});

function render(idx) {{
    const round = data[idx];
    if(!round) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const imgData = ctx.createImageData(canvas.width, canvas.height);
    
    for(let i=0; i<round.ranks.length; i++) {{
        const r = round.ranks[i];
        const x = i % META.n_lon;
        const y = META.n_lat - 1 - Math.floor(i / META.n_lon); // Flip Y
        const pos = (y * META.n_lon + x) * 4;
        
        if (r === 0) {{ // Win (Green)
            imgData.data[pos]=0; imgData.data[pos+1]=255; imgData.data[pos+2]=0; imgData.data[pos+3]=230;
        }} else if (r < 5) {{ // Top 5 (Yellow)
            imgData.data[pos]=255; imgData.data[pos+1]=255; imgData.data[pos+2]=0; imgData.data[pos+3]=180;
        }} else if (r > round.nActive - 3) {{ // Loser (Red)
            imgData.data[pos]=255; imgData.data[pos+1]=0; imgData.data[pos+2]=0; imgData.data[pos+3]=200;
        }}
    }}
    
    ctx.putImageData(imgData, 0, 0);
    const url = canvas.toDataURL();
    const coords = [[-180, 90], [180, 90], [180, -90], [-180, -90]];

    if(!sourceAdded) {{
        map.addSource('overlay', {{ type: 'image', url: url, coordinates: coords }});
        map.addLayer({{ id: 'overlay-layer', type: 'raster', source: 'overlay' }});
        sourceAdded = true;
    }} else {{
        map.getSource('overlay').updateImage({{ url: url, coordinates: coords }});
    }}
    
    document.getElementById('info').textContent = "Round " + (idx + 1) + " | Active: " + round.nActive;
}}

document.getElementById('slider').oninput = (e) => render(parseInt(e.target.value));

let playing = false;
let intv;
document.getElementById('play').onclick = () => {{
    playing = !playing;
    document.getElementById('play').textContent = playing ? "⏸ Pause" : "▶ Play";
    if(playing) {{
        intv = setInterval(() => {{
            let s = document.getElementById('slider');
            let v = (parseInt(s.value) + 1) % data.length;
            s.value = v;
            render(v);
        }}, 300);
    }} else clearInterval(intv);
}};
</script>
</body>
</html>
"""

components.html(HTML, height=750)
