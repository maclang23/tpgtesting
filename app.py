"""
app.py — Fixed for Black Screen & Initialization Issues
"""

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
# Sanity checks
# ─────────────────────────────────────────────
for path in [CSV_PATH, INDEX_PATH]:
    if not os.path.exists(path):
        st.error(f"`{path}` not found. Ensure `static/` folder is uploaded.")
        st.stop()

# ─────────────────────────────────────────────
# Load metadata (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_meta():
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

all_players, round_cols, round_targets, round_active, name_to_file = load_meta()

@st.cache_data(show_spinner=False)
def load_player_b64(player_name):
    safe = name_to_file.get(player_name)
    if not safe: return None
    fpath = os.path.join(RANKS_DIR, f"{safe}.bin")
    if not os.path.exists(fpath): return None
    with open(fpath, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

# ─────────────────────────────────────────────
# UI logic
# ─────────────────────────────────────────────
st.title("Gauntlet Timelapse")

players_with_data = [p for p in all_players if name_to_file.get(p) and
                     os.path.exists(os.path.join(RANKS_DIR, name_to_file[p] + ".bin"))]

selected_player = st.selectbox("Player", sorted(players_with_data))
b64_data = load_player_b64(selected_player)

if not b64_data:
    st.error(f"No rank data found for {selected_player}.")
    st.stop()

lats = np.arange(-90, 91, GRID_STEP)
lons = np.arange(-180, 181, GRID_STEP)

player_rounds = [col for col in round_cols if selected_player in round_active.get(col, [])]

meta = {
    "player":        selected_player,
    "n_lat":         len(lats),
    "n_lon":         len(lons),
    "round_targets": {col: round_targets.get(col) for col in player_rounds},
}
meta_js = json.dumps(meta, separators=(",", ":"))

# ─────────────────────────────────────────────
# HTML component
# ─────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html>
<head>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#1a1a2e; color:#e0e0e0; font-family:sans-serif; height:100vh; display:flex; flex-direction:column; overflow:hidden; }}
  #toolbar {{ background:#16213e; padding:10px; display:flex; align-items:center; gap:15px; border-bottom:1px solid #0f3460; }}
  #map-wrap {{ flex:1; position:relative; background:#000; }}
  #map {{ width:100%; height:100%; }}
  #overlay {{ position:absolute; inset:0; background:rgba(0,0,0,0.8); z-index:100; display:flex; justify-content:center; align-items:center; }}
  .hidden {{ display:none !important; }}
</style>
</head>
<body>
<div id="toolbar">
  <label style="font-size:0.8rem">Round</label>
  <input type="range" id="round-sl" min="0" value="0" style="width:200px">
  <span id="s-round" style="color:#e94560; font-weight:bold">0</span>
</div>

<div id="map-wrap">
  <div id="overlay">Initializing...</div>
  <div id="map"></div>
</div>

<script>
const META = {meta_js};
const B64  = "{b64_data}";

// Local Style: Avoids "Black Screen" if demotiles.maplibre.org is blocked 
const localStyle = {{
  "version": 8,
  "sources": {{}},
  "layers": [{{ "id": "background", "type": "background", "paint": {{ "background-color": "#0a0a1a" }} }}]
}};

function decodeBinary(b64) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8  = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const view     = new DataView(buf);
  const n_rounds = view.getUint16(0, true);
  const ng       = view.getUint32(2, true);
  const rounds   = [];
  let offset     = 6;
  for (let i = 0; i < n_rounds; i++) {{
    offset += 2; // skip rnum
    const n_active = view.getUint16(offset, true); offset += 2;
    const ranks    = new Uint8Array(buf, offset, ng); offset += ng;
    rounds.push({{ n_active, ranks }});
  }}
  return rounds;
}}

const roundData = decodeBinary(B64);
const map = new maplibregl.Map({{
  container: "map",
  style: localStyle,
  center: [0, 20], zoom: 1,
  attributionControl: false
}});

let canvas, ctx, imgData, sourceAdded = false;

map.on("load", () => {{
  console.log("Map Ready");
  map.resize(); // Fix for y=Infinity error 
  
  canvas = document.createElement("canvas");
  canvas.width = META.n_lon; canvas.height = META.n_lat;
  ctx = canvas.getContext("2d");
  imgData = ctx.createImageData(canvas.width, canvas.height);
  
  document.getElementById("round-sl").max = roundData.length - 1;
  document.getElementById("overlay").classList.add("hidden");
  update(0);
}});

function update(idx) {{
  if (!roundData[idx]) return;
  const {{ n_active, ranks }} = roundData[idx];
  const d = imgData.data;
  
  // Paint heatmap logic
  for (let i = 0; i < ranks.length; i++) {{
    const gi = i;
    const r  = ranks[gi];
    const p  = (META.n_lat - 1 - Math.floor(i / META.n_lon)) * META.n_lon + (i % META.n_lon);
    const cp = p * 4;

    if (r === 0) {{ // 1st Place (Green)
      d[cp]=0; d[cp+1]=255; d[cp+2]=0; d[cp+3]=220;
    }} else if (r <= 2) {{ // Top 3
      d[cp]=100; d[cp+1]=255; d[cp+2]=100; d[cp+3]=150;
    }} else if (r >= n_active - 1) {{ // Last Place (Red)
      d[cp]=255; d[cp+1]=0; d[cp+2]=0; d[cp+3]=200;
    }} else {{
      d[cp+3]=0; // Transparent
    }}
  }}
  
  ctx.putImageData(imgData, 0, 0);
  const url = canvas.toDataURL();
  const coords = [[-180, 90], [180, 90], [180, -90], [-180, -90]];

  if (!sourceAdded) {{
    map.addSource("rk", {{ type: "image", url, coordinates: coords }});
    map.addLayer({{ id: "rk-layer", type: "raster", source: "rk" }});
    sourceAdded = true;
  }} else {{
    map.getSource("rk").updateImage({{ url, coordinates: coords }});
  }}
  document.getElementById("s-round").textContent = idx;
}}

document.getElementById("round-sl").oninput = (e) => update(parseInt(e.target.value));
</script>
</body>
</html>"""

components.html(HTML, height=700, scrolling=False)
