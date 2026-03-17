"""
app.py — Streamlit Cloud deployment wrapper for the Gauntlet Timelapse.

Requires gauntlet_data.bin and roundlist.csv in the same directory.
Run precompute.py locally first to generate gauntlet_data.bin, then
commit both files to your GitHub repo before deploying.
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import base64
import os
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Gauntlet Timelapse", layout="wide")

BIN_PATH   = "gauntlet_data.bin"
CSV_PATH   = "roundlist.csv"
API_PLAYERS = "https://tpg.marsmathis.com/api/players"
API_SUBS    = "https://tpg.marsmathis.com/api/submissions/{discord_id}"
GRID_STEP   = 1.0
R_EARTH     = 6371.0
MAX_WORKERS = 16

# ─────────────────────────────────────────────
# Check required files exist
# ─────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    st.error(f"`{CSV_PATH}` not found. Make sure it is committed to your repo.")
    st.stop()

if not os.path.exists(BIN_PATH):
    st.warning(
        f"`{BIN_PATH}` not found. "
        "Run `python precompute.py` locally and commit the output to your repo, "
        "or click the button below to precompute now (takes ~30s)."
    )

    if st.button("Precompute now (fetches from API)", type="primary"):
        # ── Load CSV ──────────────────────────────
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip() for c in df.columns]
        round_cols = [c for c in df.columns if c.startswith("Round")]
        player_rows = df[df["Name"] != "Location"].copy()
        all_players = player_rows["Name"].str.strip().tolist()

        def is_active(val):
            return not pd.isna(val) and str(val).strip() != ""

        # ── Resolve names ─────────────────────────
        with st.spinner("Fetching player list from API..."):
            resp = requests.get(API_PLAYERS, timeout=15)
            resp.raise_for_status()
            api_players = resp.json()

        name_to_id = {}
        for p in api_players:
            pid  = str(p.get("discord_id") or p.get("id") or "").strip()
            name = str(p.get("name") or p.get("username") or "").strip()
            if not pid:
                continue
            if name:
                name_to_id[name.lower()] = pid
            for val in [p.get("canonical_name")] + list(p.get("aliases") or []):
                v = str(val or "").strip()
                if v:
                    name_to_id[v.lower()] = pid

        resolved = {n: name_to_id[n.lower()] for n in all_players if n.lower() in name_to_id}

        # ── Fetch submissions ──────────────────────
        def fetch_subs(name, pid):
            r = requests.get(API_SUBS.format(discord_id=pid), timeout=15)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                data = (data.get("submissions") or data.get("data") or
                        data.get("results") or (list(data.values())[0] if data else []))
            if not isinstance(data, list):
                return name, None
            rows = [[float(s.get("lat") or s.get("latitude") or 0),
                     float(s.get("lon") or s.get("lng") or s.get("longitude") or 0)]
                    for s in data
                    if (s.get("lat") or s.get("latitude")) is not None]
            return name, (np.array(rows, dtype=np.float64) if rows else None)

        player_subs = {}
        prog = st.progress(0, text="Fetching submissions...")
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(fetch_subs, n, p): n for n, p in resolved.items()}
            for future in as_completed(futures):
                name, pts = future.result()
                if pts is not None and len(pts) > 0:
                    player_subs[name] = pts
                done += 1
                prog.progress(done / len(resolved), text=f"Fetched {done}/{len(resolved)}")
        prog.empty()

        # ── Build grids ───────────────────────────
        lats = np.arange(-90,  91,  GRID_STEP, dtype=np.float64)
        lons = np.arange(-180, 181, GRID_STEP, dtype=np.float64)
        N_GRID = len(lats) * len(lons)

        def build_grid(pts, chunk_lats=20):
            sub_lats = np.radians(pts[:, 0])
            sub_lons = np.radians(pts[:, 1])
            N_LAT_loc, N_LON_loc = len(lats), len(lons)
            out = np.empty((N_LAT_loc, N_LON_loc), dtype=np.float32)
            for i in range(0, N_LAT_loc, chunk_lats):
                lat_chunk = np.radians(lats[i:i + chunk_lats])
                g_lats = lat_chunk[:, None, None]
                g_lons = np.radians(lons)[None, :, None]
                s_lats = sub_lats[None, None, :]
                s_lons = sub_lons[None, None, :]
                dlat = s_lats - g_lats
                dlon = s_lons - g_lons
                a = np.clip(
                    np.sin(dlat/2)**2 + np.cos(g_lats)*np.cos(s_lats)*np.sin(dlon/2)**2,
                    0.0, 1.0
                )
                out[i:i + chunk_lats] = (R_EARTH * 2 * np.arcsin(np.sqrt(a))).min(axis=2).astype(np.float32)
            return out

        player_order = [p for p in all_players if p in player_subs]
        grids_dict = {}
        prog2 = st.progress(0, text="Building grids...")
        done2 = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(build_grid, player_subs[n]): n for n in player_order}
            for future in as_completed(futures):
                name = futures[future]
                grids_dict[name] = future.result()
                done2 += 1
                prog2.progress(done2 / len(player_order), text=f"Grid {done2}/{len(player_order)}")
        prog2.empty()

        grids_array = np.stack([grids_dict[p] for p in player_order], axis=0)
        grids_array.tofile(BIN_PATH)
        st.success(f"Saved {BIN_PATH} ({os.path.getsize(BIN_PATH)/1e6:.1f} MB). Commit it to your repo to skip this step next time.")
        st.rerun()

    st.stop()

# ─────────────────────────────────────────────
# Load binary and metadata
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading grid data...")
def load_data():
    # Load CSV
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

    # Load binary and base64-encode it
    with open(BIN_PATH, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")

    # Determine player order from binary size
    lats = np.arange(-90,  91,  GRID_STEP)
    lons = np.arange(-180, 181, GRID_STEP)
    N_GRID = len(lats) * len(lons)
    n_players_in_bin = len(raw) // (N_GRID * 4)

    # Players in the bin are those from all_players that had API data,
    # in original order — we can't know exactly without rerunning precompute,
    # but precompute.py uses all_players order filtered to those with data.
    # We expose all_players to JS and let it figure out indices.

    meta = {
        "players":              all_players,
        "n_lat":                len(lats),
        "n_lon":                len(lons),
        "n_players_in_bin":     n_players_in_bin,
        "round_cols":           round_cols,
        "round_targets":        round_targets,
        "round_active":         {col: active for col, active in round_active.items()},
    }
    return meta, b64

meta, b64_data = load_data()

# ─────────────────────────────────────────────
# Build round_active_indices in Python
# (maps player name → index in binary)
# ─────────────────────────────────────────────
# The binary was built by precompute.py using player_order = [p for p in all_players if p in player_subs]
# We can reconstruct this from what's in the binary vs all_players.
# Since we don't have the exact list, we expose all_players and compute indices in JS
# using the same logic: players.indexOf(name).

meta_js = json.dumps(meta, separators=(",", ":"))

# ─────────────────────────────────────────────
# Build the HTML with base64 data injected
# ─────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0;
          display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}

  #toolbar {{
    background: #16213e; border-bottom: 1px solid #0f3460;
    padding: 10px 16px; display: flex; align-items: center; gap: 20px; flex-shrink: 0;
    flex-wrap: wrap;
  }}
  #toolbar h1 {{ font-size: 1rem; font-weight: 700; color: #e94560; white-space: nowrap; }}

  .ctrl {{ display: flex; align-items: center; gap: 8px; }}
  label {{ font-size: 0.75rem; color: #8892b0; white-space: nowrap; }}
  select {{
    background: #0f3460; color: #e0e0e0; border: 1px solid #1a4a8a;
    border-radius: 6px; padding: 6px 8px; font-size: 0.85rem; cursor: pointer;
  }}
  input[type=range] {{ accent-color: #e94560; cursor: pointer; width: 180px; }}

  .stat {{ background: #0f3460; border-radius: 6px; padding: 6px 12px; text-align: center; }}
  .stat .val {{ font-size: 1.1rem; font-weight: 700; color: #e94560; }}
  .stat .key {{ font-size: 0.65rem; color: #8892b0; }}

  .play-btn {{
    background: #e94560; color: white; border: none; border-radius: 6px;
    padding: 7px 16px; font-size: 0.85rem; font-weight: 600; cursor: pointer;
  }}
  .play-btn.playing {{ background: #0f3460; border: 1px solid #e94560; color: #e94560; }}

  #map-wrap {{ flex: 1; position: relative; }}
  #map {{ width: 100%; height: 100%; }}

  #legend {{
    position: absolute; bottom: 30px; left: 10px; z-index: 5;
    background: rgba(22,33,62,0.9); border-radius: 8px; padding: 8px 12px;
  }}
  .li {{ display: flex; align-items: center; gap: 6px; font-size: 0.75rem; color: #a8b2d8; margin-bottom: 4px; }}
  .li:last-child {{ margin-bottom: 0; }}
  .sw {{ width: 14px; height: 14px; border-radius: 2px; flex-shrink: 0; }}

  #overlay {{
    position: absolute; inset: 0; background: rgba(26,26,46,0.9);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    z-index: 10; gap: 12px;
  }}
  #overlay.hidden {{ display: none; }}
  .spinner {{ width: 40px; height: 40px; border: 3px solid #0f3460; border-top-color: #e94560; border-radius: 50%; animation: spin 0.8s linear infinite; }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  #load-msg {{ font-size: 0.9rem; color: #a8b2d8; }}
  #load-bar-wrap {{ width: 200px; height: 5px; background: #0f3460; border-radius: 3px; }}
  #load-bar {{ height: 100%; background: #e94560; border-radius: 3px; width: 0%; transition: width 0.15s; }}
</style>
</head>
<body>

<div id="toolbar">
  <h1>Gauntlet Timelapse</h1>
  <div class="ctrl"><label>Player</label><select id="player-sel"></select></div>
  <div class="ctrl"><label>Round</label><input type="range" id="round-sl" min="0" max="0" value="0"></div>
  <div class="stat"><div class="val" id="s-round">—</div><div class="key">Round</div></div>
  <div class="stat"><div class="val" id="s-active">—</div><div class="key">Active</div></div>
  <div class="stat" id="s-target-wrap" style="display:none">
    <div class="val" style="font-size:0.75rem" id="s-target">—</div><div class="key">Target</div>
  </div>
  <div class="ctrl">
    <label>Speed</label>
    <input type="range" id="speed-sl" min="1" max="10" value="3" style="width:80px">
    <span id="speed-lbl" style="font-size:0.8rem;color:#e94560">3×</span>
  </div>
  <button class="play-btn" id="play-btn">▶ Play</button>
</div>

<div id="map-wrap">
  <div id="overlay">
    <div class="spinner"></div>
    <div id="load-msg">Decoding grid data…</div>
    <div id="load-bar-wrap"><div id="load-bar"></div></div>
  </div>
  <div id="map"></div>
  <div id="legend">
    <div class="li"><div class="sw" style="background:#006600"></div>1st place</div>
    <div class="li"><div class="sw" style="background:#00CC00"></div>2nd – 3rd</div>
    <div class="li"><div class="sw" style="background:#FF6666"></div>3rd – 2nd last</div>
    <div class="li"><div class="sw" style="background:#8B0000"></div>Last place</div>
  </div>
</div>

<script>
const META   = {meta_js};
const B64    = "{b64_data}";

const {{ players, n_lat, n_lon, round_cols, round_targets, round_active }} = META;
const N_GRID   = n_lat * n_lon;

const map = new maplibregl.Map({{
  container: "map",
  style: "https://demotiles.maplibre.org/style.json",
  center: [0, 20], zoom: 1.2, minZoom: 0.5, maxZoom: 8,
}});

let grids = null;
let canvas, ctx, imgData;
let curPlayerIdx = 0, curRoundIdx = 0;
let activeRoundCols = [];
let playInterval = null, mapReady = false, sourceAdded = false;

// ── Base64 → Float32Array ──────────────────
async function b64ToFloat32(b64) {{
  // fetch() on a data URL uses the browser's native C++ base64 decoder —
  // orders of magnitude faster than the JS atob() + charCodeAt loop.
  const resp   = await fetch("data:application/octet-stream;base64," + b64);
  const buffer = await resp.arrayBuffer();
  return new Float32Array(buffer);
}}

// ── Rank grid ─────────────────────────────
function computeRank(playerIdx, activeIndices) {{
  const myOff    = playerIdx * N_GRID;
  const rankGrid = new Uint8Array(N_GRID);
  for (const oi of activeIndices) {{
    if (oi === playerIdx) continue;
    const oOff = oi * N_GRID;
    for (let i = 0; i < N_GRID; i++) {{
      if (grids[oOff + i] < grids[myOff + i]) rankGrid[i]++;
    }}
  }}
  return rankGrid;
}}

// ── Canvas paint ──────────────────────────
const C = {{
  dg: [0,102,0,230], lg: [0,204,0,153],
  lr: [255,102,102,153], dr: [139,0,0,230], none: [0,0,0,0],
}};
function paintCanvas(rankGrid, nActive) {{
  const d = imgData.data;
  for (let li = 0; li < n_lat; li++) {{
    for (let oj = 0; oj < n_lon; oj++) {{
      const gi  = li * n_lon + oj;
      const ci  = (n_lat - 1 - li) * n_lon + oj;  // flip Y
      const r   = rankGrid[gi];
      const c   = r === 0         ? C.dg
                : r <= 2          ? C.lg
                : r >= nActive-1  ? C.dr
                : (r >= nActive-3 && nActive > 4) ? C.lr
                : C.none;
      const p = ci * 4;
      d[p]=c[0]; d[p+1]=c[1]; d[p+2]=c[2]; d[p+3]=c[3];
    }}
  }}
  ctx.putImageData(imgData, 0, 0);
}}

// ── MapLibre layer ─────────────────────────
const COORDS = [[-180,90],[180,90],[180,-90],[-180,-90]];
function pushToMap() {{
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

// ── Update ────────────────────────────────
function update() {{
  if (!grids || !mapReady || !activeRoundCols.length) return;
  const col     = activeRoundCols[curRoundIdx];
  const names   = round_active[col] || [];
  const indices = names.map(n => players.indexOf(n)).filter(i => i >= 0);
  const nActive = indices.length;
  const rnum    = col.replace("Round ","");

  document.getElementById("s-round").textContent  = rnum;
  document.getElementById("s-active").textContent = nActive;
  document.getElementById("round-sl").value       = curRoundIdx;

  const tgt = round_targets[col];
  if (tgt) {{
    document.getElementById("s-target").textContent =
      tgt[0].toFixed(3)+", "+tgt[1].toFixed(3);
    document.getElementById("s-target-wrap").style.display = "";
  }}

  paintCanvas(computeRank(curPlayerIdx, indices), nActive);
  pushToMap();
}}

// ── Player change ─────────────────────────
function onPlayerChange() {{
  const name    = document.getElementById("player-sel").value;
  curPlayerIdx  = players.indexOf(name);
  curRoundIdx   = 0;
  activeRoundCols = round_cols.filter(col =>
    (round_active[col] || []).includes(name)
  );
  const sl = document.getElementById("round-sl");
  sl.max   = activeRoundCols.length - 1;
  sl.value = 0;
  update();
}}

// ── Controls ──────────────────────────────
document.getElementById("round-sl").addEventListener("input", e => {{
  curRoundIdx = +e.target.value; update();
}});
document.getElementById("player-sel").addEventListener("change", onPlayerChange);

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
      curRoundIdx = (curRoundIdx + 1) % activeRoundCols.length;
      document.getElementById("round-sl").value = curRoundIdx;
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
  document.getElementById("load-msg").textContent = "Decoding grid data…";
  grids = await b64ToFloat32(B64);
  document.getElementById("load-bar").style.width = "100%";

  canvas       = document.createElement("canvas");
  canvas.width  = n_lon; canvas.height = n_lat;
  ctx           = canvas.getContext("2d");
  imgData       = ctx.createImageData(n_lon, n_lat);

  const sel = document.getElementById("player-sel");
  players.forEach(name => {{
    const o = document.createElement("option");
    o.value = o.text = name; sel.appendChild(o);
  }});

  if (!mapReady) await new Promise(r => map.on("load", r));
  document.getElementById("overlay").classList.add("hidden");
  onPlayerChange();
}})();
</script>
</body>
</html>"""

# ─────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────
st.title("Gauntlet Timelapse")
st.caption(
    f"Grid data loaded: {os.path.getsize(BIN_PATH)/1e6:.1f} MB · "
    f"{meta['n_players_in_bin']} players · {len(meta['round_cols'])} rounds"
)

components.html(HTML, height=700, scrolling=False)
