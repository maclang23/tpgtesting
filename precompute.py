"""
precompute.py — Run once to fetch all submissions, build distance grids,
and generate index.html + gauntlet_data.bin.

Usage:
    python precompute.py
    python -m http.server 8000    # then open http://localhost:8000
"""

import pandas as pd
import numpy as np
import requests
import json
import struct
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ROSTER_CSV  = "roundlist.csv"
API_PLAYERS = "https://tpg.marsmathis.com/api/players"
API_SUBS    = "https://tpg.marsmathis.com/api/submissions/{discord_id}"
GRID_STEP   = 1.0
R_EARTH     = 6371.0
MAX_WORKERS = 16
OUT_BIN     = "gauntlet_data.bin"
OUT_HTML    = "index.html"

# ─────────────────────────────────────────────
# 1. Load round data
# ─────────────────────────────────────────────
print("Loading round data...")
df = pd.read_csv(ROSTER_CSV)
df.columns = [c.strip() for c in df.columns]
round_cols = [c for c in df.columns if c.startswith("Round")]

loc_row    = df[df["Name"] == "Location"].iloc[0]
player_rows = df[df["Name"] != "Location"].copy()
all_players = player_rows["Name"].str.strip().tolist()

def is_active(val):
    return not pd.isna(val) and str(val).strip() != ""

# Round targets
round_targets = {}
for col in round_cols:
    val = loc_row[col]
    if not pd.isna(val) and str(val).strip():
        try:
            parts = str(val).split(",")
            round_targets[col] = [float(parts[0].strip()), float(parts[1].strip())]
        except Exception:
            pass

# Active players per round
round_active = {}
for col in round_cols:
    active = player_rows.loc[player_rows[col].apply(is_active), "Name"].str.strip().tolist()
    round_active[col] = active

print(f"  {len(all_players)} players, {len(round_cols)} rounds")

# ─────────────────────────────────────────────
# 2. Resolve player names → Discord IDs
# ─────────────────────────────────────────────
print("\nFetching player list from API...")
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
    for key in ["canonical_name"]:
        val = str(p.get(key) or "").strip()
        if val:
            name_to_id[val.lower()] = pid
    for alias in (p.get("aliases") or []):
        a = str(alias).strip()
        if a:
            name_to_id[a.lower()] = pid

resolved, unresolved = {}, []
for name in all_players:
    pid = name_to_id.get(name.lower())
    if pid:
        resolved[name] = pid
    else:
        unresolved.append(name)

if unresolved:
    print(f"  WARNING: {len(unresolved)} unresolved: {unresolved}")
print(f"  Resolved {len(resolved)}/{len(all_players)} players")

# ─────────────────────────────────────────────
# 3. Fetch submissions in parallel
# ─────────────────────────────────────────────
def fetch_submissions(name, pid):
    resp = requests.get(API_SUBS.format(discord_id=pid), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        data = (data.get("submissions") or data.get("data") or
                data.get("results") or (list(data.values())[0] if data else []))
    if not isinstance(data, list):
        return name, None
    rows = []
    for s in data:
        lat = s.get("lat") or s.get("latitude")
        lon = s.get("lon") or s.get("lng") or s.get("longitude")
        if lat is not None and lon is not None:
            rows.append([float(lat), float(lon)])
    return name, (np.array(rows, dtype=np.float64) if rows else None)

print(f"\nFetching submissions ({MAX_WORKERS} workers)...")
player_subs = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(fetch_submissions, n, p): n for n, p in resolved.items()}
    for future in tqdm(as_completed(futures), total=len(futures)):
        name, pts = future.result()
        if pts is not None and len(pts) > 0:
            player_subs[name] = pts
        else:
            print(f"  No data: {name}")

print(f"  Got data for {len(player_subs)} players")

# ─────────────────────────────────────────────
# 4. Build distance grids
# ─────────────────────────────────────────────
lats = np.arange(-90,  91,  GRID_STEP, dtype=np.float64)
lons = np.arange(-180, 181, GRID_STEP, dtype=np.float64)
N_LAT, N_LON = len(lats), len(lons)

def build_grid(pts, chunk_lats=20):
    """
    Build distance grid in latitude chunks to cap peak memory.
    chunk_lats=20 → max intermediate array ~(20, 361, N) float64.
    For N=608 that's ~35 MB per chunk instead of 303 MB all at once.
    """
    sub_lats = np.radians(pts[:, 0])   # (N,)
    sub_lons = np.radians(pts[:, 1])   # (N,)
    out = np.empty((N_LAT, N_LON), dtype=np.float32)

    for i in range(0, N_LAT, chunk_lats):
        lat_chunk = np.radians(lats[i:i + chunk_lats])  # (C,)
        g_lats = lat_chunk[:, None, None]                # (C, 1, 1)
        g_lons = np.radians(lons)[None, :, None]         # (1, O, 1)
        s_lats = sub_lats[None, None, :]                 # (1, 1, N)
        s_lons = sub_lons[None, None, :]                 # (1, 1, N)

        dlat = s_lats - g_lats
        dlon = s_lons - g_lons
        a = np.clip(
            np.sin(dlat / 2) ** 2 + np.cos(g_lats) * np.cos(s_lats) * np.sin(dlon / 2) ** 2,
            0.0, 1.0
        )
        out[i:i + chunk_lats] = (R_EARTH * 2 * np.arcsin(np.sqrt(a))).min(axis=2).astype(np.float32)

    return out

print(f"\nBuilding {len(player_subs)} distance grids...")
# Final ordered player list (only those with data)
players_with_data = list(player_subs.keys())
grids_dict = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(build_grid, pts): name for name, pts in player_subs.items()}
    for future in tqdm(as_completed(futures), total=len(futures)):
        name = futures[future]
        grids_dict[name] = future.result()

# Stack into (n_players, N_LAT, N_LON) float32
# Use a consistent ordering
player_order = [p for p in all_players if p in grids_dict]
grids_array  = np.stack([grids_dict[p] for p in player_order], axis=0)
player_index = {name: i for i, name in enumerate(player_order)}
print(f"  Grid array: {grids_array.shape}, {grids_array.nbytes/1e6:.1f} MB")

# ─────────────────────────────────────────────
# 5. Save distance grids binary (for standalone index.html)
# ─────────────────────────────────────────────
print(f"\nSaving {OUT_BIN}...")
grids_array.tofile(OUT_BIN)
print(f"  Saved {os.path.getsize(OUT_BIN)/1e6:.1f} MB")

# ─────────────────────────────────────────────
# 5b. Save per-player rank files for Streamlit app
#
# For each player, for each round they were active, compute their
# rank (0=closest) at every grid point vs other active players.
# Stored as uint8 → ~2–5 MB per player vs 26 MB for all distances.
#
# Binary format per player file:
#   [0]     n_rounds  uint16  — number of rounds in this file
#   [2]     N_GRID    uint32  — grid size (constant = N_LAT × N_LON)
#   per round block:
#     round_number  uint16
#     n_active      uint16
#     rank_grid     N_GRID × uint8   (rank among active players)
# ─────────────────────────────────────────────
RANKS_DIR = "static/ranks"
os.makedirs(RANKS_DIR, exist_ok=True)
N_GRID = N_LAT * N_LON

print(f"\nSaving per-player rank files to {RANKS_DIR}/...")
for p_name in tqdm(player_order):
    p_idx = player_index[p_name]
    rounds_data = []

    for col in round_cols:
        active_names   = [p for p in round_active.get(col, []) if p in player_index]
        active_indices = [player_index[p] for p in active_names]
        if p_idx not in active_indices:
            continue
        n_active = len(active_indices)
        rnum = int(col.replace("Round ", ""))

        # Rank = number of active players closer than this player at each point
        my_grid  = grids_array[p_idx].ravel()          # (N_GRID,)
        rank_grid = np.zeros(N_GRID, dtype=np.uint8)
        for oi in active_indices:
            if oi == p_idx:
                continue
            rank_grid += (grids_array[oi].ravel() < my_grid).astype(np.uint8)

        rounds_data.append((rnum, n_active, rank_grid))

    # Write binary
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in p_name)
    fpath = os.path.join(RANKS_DIR, f"{safe_name}.bin")
    with open(fpath, "wb") as f:
        n_rounds = len(rounds_data)
        f.write(struct.pack("<H", n_rounds))   # uint16
        f.write(struct.pack("<I", N_GRID))     # uint32
        for rnum, n_active, rank_grid in rounds_data:
            f.write(struct.pack("<H", rnum))     # uint16 round number
            f.write(struct.pack("<H", n_active)) # uint16 n_active
            f.write(rank_grid.tobytes())         # N_GRID uint8

# Save name → filename mapping
name_to_file = {
    p: "".join(c if c.isalnum() or c in "-_" else "_" for c in p)
    for p in player_order
}
with open("static/player_index.json", "w") as f:
    json.dump(name_to_file, f)

print(f"  Saved {len(player_order)} rank files")
print(f"  Avg size: {np.mean([os.path.getsize(os.path.join(RANKS_DIR, v+'.bin')) for v in name_to_file.values()])/1e3:.0f} KB")


# ─────────────────────────────────────────────
# 6. Build metadata for JS
# ─────────────────────────────────────────────

meta = {
    "players":       player_order,
    "n_lat":         N_LAT,
    "n_lon":         N_LON,
    "round_cols":    round_cols,
    "round_targets": round_targets,
    # active lists use only players that have grid data
    "round_active":  {
        col: [p for p in active if p in player_index]
        for col, active in round_active.items()
    },
    "round_active_indices": {
        col: [player_index[p] for p in active if p in player_index]
        for col, active in round_active.items()
    },
}

# ─────────────────────────────────────────────
# 7. Generate index.html
# ─────────────────────────────────────────────
print(f"Generating {OUT_HTML}...")

META_JS = json.dumps(meta, separators=(",", ":"))

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gauntlet Timelapse</title>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; display: flex; height: 100vh; overflow: hidden; }

  #sidebar {
    width: 280px; flex-shrink: 0;
    background: #16213e; border-right: 1px solid #0f3460;
    padding: 20px 16px; display: flex; flex-direction: column; gap: 16px;
    overflow-y: auto;
  }
  #sidebar h1 { font-size: 1.1rem; font-weight: 700; color: #e94560; letter-spacing: 0.5px; }

  label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; color: #8892b0; display: block; margin-bottom: 4px; }

  select, input[type=range] { width: 100%; }
  select {
    background: #0f3460; color: #e0e0e0; border: 1px solid #1a4a8a;
    border-radius: 6px; padding: 8px; font-size: 0.9rem; cursor: pointer;
  }
  select:focus { outline: 2px solid #e94560; }

  input[type=range] { accent-color: #e94560; cursor: pointer; }

  .stat-row { display: flex; gap: 8px; }
  .stat {
    flex: 1; background: #0f3460; border-radius: 8px; padding: 10px 8px;
    text-align: center;
  }
  .stat .val { font-size: 1.3rem; font-weight: 700; color: #e94560; }
  .stat .key { font-size: 0.65rem; text-transform: uppercase; color: #8892b0; margin-top: 2px; }

  .target-box {
    background: #0f3460; border-radius: 8px; padding: 10px;
    font-size: 0.8rem; color: #a8b2d8;
  }
  .target-box strong { color: #ccd6f6; display: block; margin-bottom: 2px; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; }

  .legend { background: #0f3460; border-radius: 8px; padding: 10px; }
  .legend-item { display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: #a8b2d8; margin-bottom: 6px; }
  .legend-item:last-child { margin-bottom: 0; }
  .swatch { width: 16px; height: 16px; border-radius: 3px; flex-shrink: 0; }

  .play-btn {
    background: #e94560; color: white; border: none; border-radius: 8px;
    padding: 10px; font-size: 0.9rem; font-weight: 600; cursor: pointer;
    width: 100%; transition: background 0.2s;
  }
  .play-btn:hover { background: #c73652; }
  .play-btn.playing { background: #0f3460; border: 1px solid #e94560; color: #e94560; }

  #speed-label { display: flex; justify-content: space-between; }
  #speed-label span { font-size: 0.75rem; color: #8892b0; }

  #map-container { flex: 1; position: relative; }
  #map { width: 100%; height: 100%; }

  #loading-overlay {
    position: absolute; inset: 0;
    background: rgba(26,26,46,0.85); backdrop-filter: blur(4px);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    z-index: 10; gap: 16px;
  }
  #loading-overlay.hidden { display: none; }
  .spinner {
    width: 48px; height: 48px; border: 4px solid #0f3460;
    border-top-color: #e94560; border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #loading-msg { font-size: 0.95rem; color: #a8b2d8; }
  #loading-bar-wrap { width: 240px; height: 6px; background: #0f3460; border-radius: 3px; }
  #loading-bar { height: 100%; background: #e94560; border-radius: 3px; width: 0%; transition: width 0.2s; }

  #round-display { font-size: 0.85rem; color: #8892b0; text-align: center; }
</style>
</head>
<body>

<div id="sidebar">
  <h1>Gauntlet Timelapse</h1>

  <div>
    <label>Player</label>
    <select id="player-select"></select>
  </div>

  <div>
    <label>Round <span id="round-display"></span></label>
    <input type="range" id="round-slider" min="0" max="0" value="0" step="1">
  </div>

  <div class="stat-row">
    <div class="stat"><div class="val" id="stat-round">—</div><div class="key">Round</div></div>
    <div class="stat"><div class="val" id="stat-active">—</div><div class="key">Active</div></div>
  </div>

  <div class="target-box">
    <strong>Round Target</strong>
    <span id="target-coords">—</span>
  </div>

  <div class="legend">
    <div class="legend-item"><div class="swatch" style="background:#006600"></div>1st place</div>
    <div class="legend-item"><div class="swatch" style="background:#00CC00"></div>2nd – 3rd</div>
    <div class="legend-item"><div class="swatch" style="background:#FF6666"></div>3rd – 2nd last</div>
    <div class="legend-item"><div class="swatch" style="background:#8B0000"></div>Last place</div>
  </div>

  <div>
    <div id="speed-label"><label>Playback speed</label><span id="speed-val">1×</span></div>
    <input type="range" id="speed-slider" min="1" max="10" value="3" step="1">
  </div>

  <button class="play-btn" id="play-btn">▶ Play</button>
</div>

<div id="map-container">
  <div id="loading-overlay">
    <div class="spinner"></div>
    <div id="loading-msg">Loading grid data…</div>
    <div id="loading-bar-wrap"><div id="loading-bar"></div></div>
  </div>
  <div id="map"></div>
</div>

<script>
// ── Embedded metadata ──────────────────────
const META = __META__;

// ── MapLibre init ──────────────────────────
const map = new maplibregl.Map({
  container: "map",
  style: "https://demotiles.maplibre.org/style.json",
  center: [0, 20],
  zoom: 1.2,
  minZoom: 0.5,
  maxZoom: 8,
});

// ── State ──────────────────────────────────
let grids = null;          // Float32Array, shape (n_players, N_LAT, N_LON)
let canvas, ctx, imageData;
const { players, n_lat, n_lon, round_cols, round_targets,
        round_active, round_active_indices } = META;

const N_PLAYERS = players.length;
const N_GRID    = n_lat * n_lon;

let currentPlayerIdx = 0;
let currentRoundIdx  = 0;
let activeRoundCols  = [];   // rounds the selected player participated in
let playInterval     = null;
let mapReady         = false;
let sourceAdded      = false;

// ── Load binary ────────────────────────────
async function loadBinary() {
  setLoading("Loading grid data…", 0);
  const response = await fetch("gauntlet_data.bin");
  const reader   = response.body.getReader();
  const total    = parseInt(response.headers.get("Content-Length") || "0");
  let received   = 0;
  const chunks   = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    if (total > 0) setLoading("Loading grid data…", received / total * 100);
  }

  const blob   = new Blob(chunks);
  const buffer = await blob.arrayBuffer();
  grids        = new Float32Array(buffer);
  setLoading("Initialising…", 100);
}

// ── Canvas setup ──────────────────────────
function initCanvas() {
  canvas    = document.createElement("canvas");
  canvas.width  = n_lon;
  canvas.height = n_lat;
  ctx       = canvas.getContext("2d");
  imageData = ctx.createImageData(n_lon, n_lat);
}

// ── Rank computation ──────────────────────
// For each grid point, count how many active players are closer than the
// selected player → that is their 0-based rank.
// Also track max rank so we can identify last / 2nd-last / 3rd-last.
function computeRankGrid(playerIdx, activeIndices) {
  const myStart   = playerIdx * N_GRID;
  const rankGrid  = new Uint8Array(N_GRID);
  const nActive   = activeIndices.length;

  for (const otherIdx of activeIndices) {
    if (otherIdx === playerIdx) continue;
    const oStart = otherIdx * N_GRID;
    for (let i = 0; i < N_GRID; i++) {
      if (grids[oStart + i] < grids[myStart + i]) rankGrid[i]++;
    }
  }
  return { rankGrid, nActive };
}

// ── Canvas render ─────────────────────────
const COLORS = {
  darkGreen:  [0,   102,   0, 230],
  lightGreen: [0,   204,   0, 153],
  lightRed:   [255, 102, 102, 153],
  darkRed:    [139,   0,   0, 230],
  none:       [0,     0,   0,   0],
};

function renderRankGrid(rankGrid, nActive) {
  const d = imageData.data;
  for (let lat_i = 0; lat_i < n_lat; lat_i++) {
    for (let lon_j = 0; lon_j < n_lon; lon_j++) {
      const gridIdx   = lat_i * n_lon + lon_j;
      // Canvas origin is top-left; our grid origin is south (-90) so flip vertically
      const canvasIdx = (n_lat - 1 - lat_i) * n_lon + lon_j;
      const rank      = rankGrid[gridIdx];
      let color;
      if      (rank === 0)               color = COLORS.darkGreen;
      else if (rank <= 2)                color = COLORS.lightGreen;
      else if (rank >= nActive - 1)      color = COLORS.darkRed;
      else if (rank >= nActive - 3 && nActive > 4) color = COLORS.lightRed;
      else                               color = COLORS.none;
      const p = canvasIdx * 4;
      d[p]     = color[0];
      d[p + 1] = color[1];
      d[p + 2] = color[2];
      d[p + 3] = color[3];
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

// ── MapLibre image source ──────────────────
const COORDS = [[-180, 90], [180, 90], [180, -90], [-180, -90]];

function updateMapLayer() {
  const url = canvas.toDataURL("image/png");
  if (!sourceAdded) {
    map.addSource("ranking", { type: "image", url, coordinates: COORDS });
    map.addLayer({
      id: "ranking-layer",
      type: "raster",
      source: "ranking",
      paint: { "raster-opacity": 0.75, "raster-fade-duration": 0 },
    });
    sourceAdded = true;
  } else {
    map.getSource("ranking").updateImage({ url, coordinates: COORDS });
  }
}

// ── Update everything for current selection ─
function update() {
  if (!grids || !mapReady || activeRoundCols.length === 0) return;

  const roundCol     = activeRoundCols[currentRoundIdx];
  const activeIdx    = round_active_indices[roundCol];
  const nActive      = activeIdx.length;
  const rnum         = roundCol.replace("Round ", "");

  // Stats
  document.getElementById("stat-round").textContent  = rnum;
  document.getElementById("stat-active").textContent = nActive;
  document.getElementById("round-display").textContent = `${currentRoundIdx + 1} / ${activeRoundCols.length}`;

  const tgt = round_targets[roundCol];
  document.getElementById("target-coords").textContent =
    tgt ? `${tgt[0].toFixed(4)}, ${tgt[1].toFixed(4)}` : "—";

  // Compute and render
  const { rankGrid } = computeRankGrid(currentPlayerIdx, activeIdx);
  renderRankGrid(rankGrid, nActive);
  updateMapLayer();
}

// ── Player change ─────────────────────────
function onPlayerChange() {
  const name = document.getElementById("player-select").value;
  currentPlayerIdx  = players.indexOf(name);
  currentRoundIdx   = 0;

  // Filter to rounds where this player has a grid entry
  activeRoundCols = round_cols.filter(col =>
    round_active_indices[col].includes(currentPlayerIdx)
  );

  const slider = document.getElementById("round-slider");
  slider.max   = activeRoundCols.length - 1;
  slider.value = 0;
  update();
}

// ── DOM wiring ────────────────────────────
function setLoading(msg, pct) {
  document.getElementById("loading-msg").textContent = msg;
  document.getElementById("loading-bar").style.width = pct + "%";
}

function hideLoading() {
  document.getElementById("loading-overlay").classList.add("hidden");
}

document.getElementById("round-slider").addEventListener("input", e => {
  currentRoundIdx = parseInt(e.target.value);
  update();
});

document.getElementById("player-select").addEventListener("change", onPlayerChange);

const playBtn = document.getElementById("play-btn");
playBtn.addEventListener("click", () => {
  if (playInterval) {
    clearInterval(playInterval);
    playInterval = null;
    playBtn.textContent = "▶ Play";
    playBtn.classList.remove("playing");
  } else {
    playBtn.textContent = "⏸ Pause";
    playBtn.classList.add("playing");
    const speedVal = parseInt(document.getElementById("speed-slider").value);
    const ms       = Math.max(100, 1100 - speedVal * 100);
    playInterval = setInterval(() => {
      currentRoundIdx++;
      if (currentRoundIdx >= activeRoundCols.length) {
        currentRoundIdx = 0;
      }
      document.getElementById("round-slider").value = currentRoundIdx;
      update();
    }, ms);
  }
});

document.getElementById("speed-slider").addEventListener("input", e => {
  const v = e.target.value;
  document.getElementById("speed-val").textContent = v + "×";
  // Restart interval with new speed if playing
  if (playInterval) {
    playBtn.click(); // stop
    playBtn.click(); // restart
  }
});

// ── Boot sequence ─────────────────────────
map.on("load", () => { mapReady = true; });

(async () => {
  await loadBinary();
  initCanvas();

  // Populate player select
  const sel = document.getElementById("player-select");
  players.forEach(name => {
    const opt  = document.createElement("option");
    opt.value  = name;
    opt.text   = name;
    sel.appendChild(opt);
  });

  // Wait for map if needed
  if (!mapReady) {
    await new Promise(resolve => map.on("load", resolve));
  }

  hideLoading();
  onPlayerChange();
})();
</script>
</body>
</html>
""".replace("__META__", META_JS)

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"  Saved {OUT_HTML} ({os.path.getsize(OUT_HTML)/1e3:.0f} KB)")
print(f"\nDone! Serve the directory:")
print(f"  python -m http.server 8000")
print(f"  Then open: http://localhost:8000")