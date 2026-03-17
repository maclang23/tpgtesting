"""
app.py — Streamlit Cloud deployment for Gauntlet Timelapse.

Run precompute.py locally first to generate static/ranks/*.bin and
static/player_index.json, then commit the static/ folder to your repo.
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import os
import struct
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
        st.error(f"`{path}` not found. Run `python precompute.py` locally and commit the output.")
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
            except Exception:
                pass

    round_active = {}
    for col in round_cols:
        active = player_rows.loc[player_rows[col].apply(is_active), "Name"].str.strip().tolist()
        round_active[col] = active

    with open(INDEX_PATH) as f:
        name_to_file = json.load(f)

    return all_players, round_cols, round_targets, round_active, name_to_file

all_players, round_cols, round_targets, round_active, name_to_file = load_meta()

# ─────────────────────────────────────────────
# Load per-player rank binary (cached per player)
# Returns base64-encoded bytes — small enough to embed directly
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_player_b64(player_name):
    safe = name_to_file.get(player_name)
    if not safe:
        return None
    fpath = os.path.join(RANKS_DIR, f"{safe}.bin")
    if not os.path.exists(fpath):
        return None
    with open(fpath, "rb") as f:
        raw = f.read()
    return base64.b64encode(raw).decode("ascii")

# ─────────────────────────────────────────────
# Player selector
# ─────────────────────────────────────────────
st.title("Gauntlet Timelapse")

players_with_data = [p for p in all_players if name_to_file.get(p) and
                     os.path.exists(os.path.join(RANKS_DIR, name_to_file[p] + ".bin"))]

selected_player = st.selectbox("Player", sorted(players_with_data))

b64_data = load_player_b64(selected_player)
if not b64_data:
    st.error(f"No rank data found for {selected_player}. Re-run precompute.py.")
    st.stop()

# ─────────────────────────────────────────────
# Build compact metadata for this player only
# ─────────────────────────────────────────────
lats = np.arange(-90, 91, GRID_STEP)
lons = np.arange(-180, 181, GRID_STEP)
N_GRID = len(lats) * len(lons)

player_rounds = [col for col in round_cols if selected_player in round_active.get(col, [])]

meta = {
    "player":        selected_player,
    "n_lat":         len(lats),
    "n_lon":         len(lons),
    "n_grid":        N_GRID,
    "round_cols":    player_rounds,
    "round_targets": {col: round_targets.get(col) for col in player_rounds},
    "round_active_counts": {col: len(round_active.get(col, [])) for col in player_rounds},
}
meta_js = json.dumps(meta, separators=(",", ":"))

fsize = len(base64.b64decode(b64_data))
st.caption(f"{selected_player} · {len(player_rounds)} rounds · {fsize/1e3:.0f} KB rank data")

# ─────────────────────────────────────────────
# HTML component
# The b64 payload is now ~50–200 KB (uint8 ranks) vs 35 MB (float32 distances)
# atob() on 200 KB is instant
# ─────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:system-ui,sans-serif; background:#1a1a2e; color:#e0e0e0;
          display:flex; flex-direction:column; height:100vh; overflow:hidden; }}

  #toolbar {{ background:#16213e; border-bottom:1px solid #0f3460;
    padding:10px 16px; display:flex; align-items:center; gap:18px; flex-shrink:0; flex-wrap:wrap; }}
  #toolbar h1 {{ font-size:1rem; font-weight:700; color:#e94560; white-space:nowrap; }}

  .ctrl {{ display:flex; align-items:center; gap:8px; }}
  label {{ font-size:0.75rem; color:#8892b0; white-space:nowrap; }}
  input[type=range] {{ accent-color:#e94560; cursor:pointer; width:200px; }}

  .stat {{ background:#0f3460; border-radius:6px; padding:6px 12px; text-align:center; }}
  .stat .val {{ font-size:1.1rem; font-weight:700; color:#e94560; }}
  .stat .key {{ font-size:0.65rem; color:#8892b0; }}

  .play-btn {{ background:#e94560; color:white; border:none; border-radius:6px;
    padding:7px 16px; font-size:0.85rem; font-weight:600; cursor:pointer; }}
  .play-btn.playing {{ background:#0f3460; border:1px solid #e94560; color:#e94560; }}

  .ctrl-s {{ display:flex; align-items:center; gap:6px; }}

  #map-wrap {{ flex:1; position:relative; }}
  #map {{ width:100%; height:100%; }}

  #legend {{ position:absolute; bottom:30px; left:10px; z-index:5;
    background:rgba(22,33,62,0.9); border-radius:8px; padding:8px 12px; }}
  .li {{ display:flex; align-items:center; gap:6px; font-size:0.75rem; color:#a8b2d8; margin-bottom:4px; }}
  .li:last-child {{ margin-bottom:0; }}
  .sw {{ width:14px; height:14px; border-radius:2px; flex-shrink:0; }}

  #overlay {{ position:absolute; inset:0; background:rgba(26,26,46,0.9);
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    z-index:10; gap:12px; }}
  #overlay.hidden {{ display:none; }}
  .spinner {{ width:40px; height:40px; border:3px solid #0f3460;
    border-top-color:#e94560; border-radius:50%; animation:spin 0.8s linear infinite; }}
  @keyframes spin {{ to{{ transform:rotate(360deg); }} }}
  #load-msg {{ font-size:0.9rem; color:#a8b2d8; }}
</style>
</head>
<body>
<div id="toolbar">
  <h1>Gauntlet Timelapse</h1>
  <div class="ctrl"><label>Round</label><input type="range" id="round-sl" min="0" max="0" value="0"></div>
  <div class="stat"><div class="val" id="s-round">—</div><div class="key">Round</div></div>
  <div class="stat"><div class="val" id="s-active">—</div><div class="key">Active</div></div>
  <div class="stat" id="tgt-wrap" style="display:none">
    <div class="val" style="font-size:0.75rem" id="s-target">—</div><div class="key">Target</div>
  </div>
  <div class="ctrl-s">
    <label>Speed</label>
    <input type="range" id="speed-sl" min="1" max="10" value="3" style="width:80px">
    <span id="speed-lbl" style="font-size:0.8rem;color:#e94560">3×</span>
  </div>
  <button class="play-btn" id="play-btn">▶ Play</button>
</div>

<div id="map-wrap">
  <div id="overlay"><div class="spinner"></div><div id="load-msg">Loading…</div></div>
  <div id="map"></div>
  <div id="legend">
    <div class="li"><div class="sw" style="background:#006600"></div>1st</div>
    <div class="li"><div class="sw" style="background:#00CC00"></div>2nd–3rd</div>
    <div class="li"><div class="sw" style="background:#FF6666"></div>3rd–2nd last</div>
    <div class="li"><div class="sw" style="background:#8B0000"></div>Last</div>
  </div>
</div>

<script>
// ── Embedded data ──────────────────────────
const META = {meta_js};
const B64  = "{b64_data}";

const {{ player, n_lat, n_lon, n_grid, round_cols, round_targets, round_active_counts }} = META;

// ── Decode binary (uint8 rank data) ───────
// Format: uint16 n_rounds, uint32 n_grid,
//   per round: uint16 rnum, uint16 n_active, n_grid×uint8 rank
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
    const rnum     = view.getUint16(offset,     true); offset += 2;
    const n_active = view.getUint16(offset,     true); offset += 2;
    const ranks    = new Uint8Array(buf, offset, ng);  offset += ng;
    rounds.push({{ rnum, n_active, ranks }});
  }}
  return rounds;
}}

const roundData = decodeBinary(B64);   // instant — only ~100–200 KB
const n_rounds  = roundData.length;

// ── MapLibre ──────────────────────────────
const map = new maplibregl.Map({{
  container: "map",
  style: "https://demotiles.maplibre.org/style.json",
  center: [0, 20], zoom: 1.2, minZoom: 0.5, maxZoom: 8,
}});

let canvas, ctx, imgData;
let curIdx = 0;
let playInterval = null, mapReady = false, sourceAdded = false;

// ── Canvas paint ──────────────────────────
const C = {{
  dg:[0,102,0,230], lg:[0,204,0,153],
  lr:[255,102,102,153], dr:[139,0,0,230], none:[0,0,0,0],
}};
function paint(ranks, nActive) {{
  const d = imgData.data;
  for (let li = 0; li < n_lat; li++) {{
    for (let oj = 0; oj < n_lon; oj++) {{
      const gi = li * n_lon + oj;
      const ci = (n_lat - 1 - li) * n_lon + oj;
      const r  = ranks[gi];
      const c  = r === 0              ? C.dg
               : r <= 2               ? C.lg
               : r >= nActive - 1     ? C.dr
               : (r >= nActive-3 && nActive > 4) ? C.lr
               : C.none;
      const p = ci * 4;
      d[p]=c[0]; d[p+1]=c[1]; d[p+2]=c[2]; d[p+3]=c[3];
    }}
  }}
  ctx.putImageData(imgData, 0, 0);
}}

const COORDS = [[-180,90],[180,90],[180,-90],[-180,-90]];
function pushMap() {{
  const url = canvas.toDataURL("image/png");
  if (!sourceAdded) {{
    map.addSource("rk", {{ type:"image", url, coordinates:COORDS }});
    map.addLayer({{ id:"rk-layer", type:"raster", source:"rk",
      paint:{{"raster-opacity":0.75,"raster-fade-duration":0}} }});
    sourceAdded = true;
  }} else {{
    map.getSource("rk").updateImage({{ url, coordinates:COORDS }});
  }}
}}

function update() {{
  if (!mapReady || !roundData.length) return;
  const {{ rnum, n_active, ranks }} = roundData[curIdx];
  document.getElementById("s-round").textContent  = rnum;
  document.getElementById("s-active").textContent = n_active;
  document.getElementById("round-sl").value       = curIdx;

  const col = `Round ${{rnum}}`;
  const tgt = round_targets[col];
  if (tgt) {{
    document.getElementById("s-target").textContent = tgt[0].toFixed(3)+", "+tgt[1].toFixed(3);
    document.getElementById("tgt-wrap").style.display = "";
  }}
  paint(ranks, n_active);
  pushMap();
}}

// ── Controls ──────────────────────────────
document.getElementById("round-sl").max = n_rounds - 1;
document.getElementById("round-sl").addEventListener("input", e => {{
  curIdx = +e.target.value; update();
}});

const playBtn = document.getElementById("play-btn");
playBtn.addEventListener("click", () => {{
  if (playInterval) {{
    clearInterval(playInterval); playInterval = null;
    playBtn.textContent = "▶ Play"; playBtn.classList.remove("playing");
  }} else {{
    playBtn.textContent = "⏸ Pause"; playBtn.classList.add("playing");
    const spd = +document.getElementById("speed-sl").value;
    const ms  = Math.max(80, 1100 - spd * 100);
    playInterval = setInterval(() => {{
      curIdx = (curIdx + 1) % n_rounds;
      document.getElementById("round-sl").value = curIdx;
      update();
    }}, ms);
  }}
}});
document.getElementById("speed-sl").addEventListener("input", e => {{
  document.getElementById("speed-lbl").textContent = e.target.value + "×";
  if (playInterval) {{ playBtn.click(); playBtn.click(); }}
}});

// ── Boot ──────────────────────────────────
map.on("load", () => {{ mapReady = true; }});

(async () => {{
  canvas        = document.createElement("canvas");
  canvas.width  = n_lon; canvas.height = n_lat;
  ctx           = canvas.getContext("2d");
  imgData       = ctx.createImageData(n_lon, n_lat);
  if (!mapReady) await new Promise(r => map.on("load", r));
  document.getElementById("overlay").classList.add("hidden");
  update();
}})();
</script>
</body>
</html>"""

components.html(HTML, height=680, scrolling=False)
