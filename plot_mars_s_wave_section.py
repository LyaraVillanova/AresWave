"""
plot_mars_s_wave_section.py

Make a 2-D Mars cross-section figure of S-wave ray paths using ObsPy TauP,
with receiver epicentral distances fixed to the supplied table values.

How to use:
1) Put this script in the same folder as arrivals.csv and the TauP model files
   used by the Blender script, e.g. Geophysical_model225.nd/.npz, MD_model88.nd/.npz, etc.
2) Run it with the same Python environment that has obspy and matplotlib:
      python plot_mars_s_wave_section.py
3) Outputs:
      mars_s_wave_section.png
      mars_s_wave_section.pdf
      mars_s_wave_section.svg
"""

import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

try:
    from obspy.taup import TauPyModel
    from obspy.taup.taup_create import build_taup_model
except Exception as exc:
    raise RuntimeError(
        "ObsPy/TauP is required. Run this in the same environment used for the Blender TauP script."
    ) from exc

# --------------------------
# User settings
# --------------------------
BASE = Path(__file__).resolve().parent

MARS_RADIUS_KM = 3389.5
CORE_RADIUS_KM = 1830.0       # change if your model uses another core radius
NEAR_SURFACE_OFFSET_KM = 80.0 # just for the double surface line visual

TARGET_DISTANCES_DEG = [
    59.7, 59.8, 60.0, 60.3, 73.3, 84.8, 88.2, 88.5, 89.5,
    90.6, 91.5, 93.7, 94.1, 95.3, 98.0, 128.3, 146.3,
]
MATCH_TOLERANCE_DEG = 0.35

PHASE_TO_PLOT = "S"
OUTPUT_STEM = "mars_s_wave_section"

EVENTS_CSV_CANDIDATES = [
    BASE / "arrivals.csv",
    BASE / "events_selected_for_blender.csv",
    BASE / "events_gt50deg_realdata.csv",
    BASE / "events_gt50deg.csv",
]

ALLOWED_EVENTS = {
    "S0133a", "S0152a", "S0167a", "S0167b", "S0185a", "S0226b", "S0234c",
    "S0254b", "S0345a", "S0395a", "S0421b", "S0976a", "S1000a", "S1094b",
    "S1102a", "S1153a", "S1415a",
}

EVENT_MODEL_BASENAMES = {
    "S0133a": "Geophysical_model225",
    "S0152a": "MD_model88",
    "S0167a": "Geophysical_model64",
    "S0167b": "MD_model37",
    "S0185a": "MD_model78",
    "S0226b": "Geophysical_model776",
    "S0234c": "Geophysical_model979",
    "S0254b": "Geophysical_model277",
    "S0345a": "AK_model_95",
    "S0395a": "Geophysical_model573",
    "S0421b": "MD_model14",
    "S0976a": "Geophysical_model837",
    "S1000a": "AK_model_62",
    "S1094b": "Geophysical_model319",
    "S1102a": "Geophysical_model58",
    "S1153a": "Geophysical_model489",
    "S1415a": "MD_model34",
}

# --------------------------
# Loading and TauP helpers
# --------------------------
TAUP_MODEL_CACHE = {}
TAUP_PATH_CACHE = {}


def find_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def gc_distance_deg(lon1, lat1, lon2, lat2):
    lon1r, lat1r = math.radians(lon1), math.radians(lat1)
    lon2r, lat2r = math.radians(lon2), math.radians(lat2)
    arg = (
        math.sin(lat1r) * math.sin(lat2r)
        + math.cos(lat1r) * math.cos(lat2r) * math.cos(lon2r - lon1r)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, arg))))


def load_event_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = {name.strip(): name for name in reader.fieldnames or []}
        for raw in reader:
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
            if "event_id" in fieldnames or "event_id" in row:
                name = row.get("event_id", "").strip()
                if name not in ALLOWED_EVENTS:
                    continue
                rows.append({
                    "event": name,
                    "lat": float(row["latitude"]),
                    "lon_east": float(row["longitude"]),
                    "depth_km": float(row["depth"]),
                    "distance_deg": float(row["distance"]),
                })
            else:
                name = row["event"].strip()
                if name not in ALLOWED_EVENTS:
                    continue
                rows.append({
                    "event": name,
                    "lat": float(row["lat"]),
                    "lon_east": float(row["lon_east"]),
                    "depth_km": float(row.get("depth_km", row.get("depth", 30.0))),
                    "distance_deg": float(row.get("distance_deg", row.get("distance", np.nan))),
                })
    return rows


def select_rows_by_target_distances(rows):
    """Return rows ordered according to TARGET_DISTANCES_DEG."""
    selected = []
    used = set()
    for target in TARGET_DISTANCES_DEG:
        candidates = [
            (abs(row["distance_deg"] - target), idx, row)
            for idx, row in enumerate(rows)
            if idx not in used and not math.isnan(row["distance_deg"])
        ]
        if not candidates:
            print(f"[warning] No event row available for {target:.1f} deg")
            continue
        diff, idx, row = min(candidates, key=lambda x: x[0])
        if diff > MATCH_TOLERANCE_DEG:
            print(
                f"[warning] closest event to {target:.1f} deg is {row['event']} "
                f"at {row['distance_deg']:.2f} deg; diff={diff:.2f} deg; skipped"
            )
            continue
        row = dict(row)
        row["plot_distance_deg"] = target  # force receiver/star to exact table value
        selected.append(row)
        used.add(idx)
    return selected


def find_model_file(basename):
    npz = BASE / f"{basename}.npz"
    nd_candidates = [BASE / basename, BASE / f"{basename}.nd"]
    nd = find_existing(nd_candidates)
    return nd, npz


def get_taup_model(event_name):
    basename = EVENT_MODEL_BASENAMES.get(event_name)
    if basename is None:
        print(f"[TauP] no model basename for {event_name}")
        return None
    if basename in TAUP_MODEL_CACHE:
        return TAUP_MODEL_CACHE[basename]

    nd_path, npz_path = find_model_file(basename)
    if npz_path.exists():
        model = TauPyModel(model=str(npz_path))
        TAUP_MODEL_CACHE[basename] = model
        return model

    if nd_path is None:
        print(f"[TauP] missing .nd/.npz model for {event_name}: {basename}")
        return None

    print(f"[TauP] building {npz_path.name} from {nd_path.name}")
    build_taup_model(str(nd_path), output_folder=str(BASE))
    if not npz_path.exists():
        print(f"[TauP] failed to create {npz_path.name}")
        return None

    model = TauPyModel(model=str(npz_path))
    TAUP_MODEL_CACHE[basename] = model
    return model


def path_field(path, names):
    if path is None:
        return None
    dtype = getattr(path, "dtype", None)
    if dtype is None or dtype.names is None:
        return None
    for name in names:
        if name in dtype.names:
            return path[name]
    return None


def get_arrival_path(row, phase="S"):
    key = (row["event"], phase, row["plot_distance_deg"])
    if key in TAUP_PATH_CACHE:
        return TAUP_PATH_CACHE[key]

    model = get_taup_model(row["event"])
    if model is None:
        TAUP_PATH_CACHE[key] = None
        return None

    requested = phase.upper()
    if requested == "S":
        phase_trials = [["S"], ["Sdiff"], ["s"], ["sS"], ["pS"], ["SS"], ["SSS"], ["tts"], ["ttbasic"], ["ttall"]]
    elif requested == "P":
        phase_trials = [["P"], ["Pdiff"], ["p"], ["pP"], ["sP"], ["PP"], ["PPP"], ["ttp"], ["ttbasic"], ["ttall"]]
    else:
        phase_trials = [[phase], ["ttbasic"], ["ttall"]]

    def usable(arr):
        path = getattr(arr, "path", None)
        if path is None or len(path) < 2:
            return False
        names = getattr(path.dtype, "names", None)
        return names is not None and "dist" in names and "depth" in names

    def arrival_name(arr):
        return str(getattr(arr, "name", "") or getattr(arr, "phase", "") or "")

    def family_ok(arr):
        nm = arrival_name(arr).upper()
        if not nm:
            return True
        if requested == "S":
            return nm.startswith(("S", "SS", "SSS", "SDIFF"))
        if requested == "P":
            return nm.startswith(("P", "PP", "PPP", "PDIFF"))
        return True

    last_error = None
    for phase_list in phase_trials:
        try:
            arrivals = model.get_ray_paths(
                source_depth_in_km=row["depth_km"],
                distance_in_degree=row["plot_distance_deg"],
                phase_list=phase_list,
            )
        except Exception as exc:
            last_error = exc
            continue

        broad = phase_list in (["ttp"], ["tts"], ["ttbasic"], ["ttall"])
        arrs = [a for a in arrivals if usable(a) and (not broad or family_ok(a))]
        if arrs:
            arrs.sort(key=lambda a: float(getattr(a, "time", float("inf"))))
            chosen = arrs[0]
            print(
                f"[TauP] {row['event']:7s} {row['plot_distance_deg']:5.1f}°: "
                f"phase {arrival_name(chosen)} from {phase_list}"
            )
            TAUP_PATH_CACHE[key] = chosen.path
            return chosen.path

    print(f"[TauP] no usable path for {row['event']} at {row['plot_distance_deg']:.1f}°; last_error={last_error}")
    TAUP_PATH_CACHE[key] = None
    return None


# --------------------------
# Cross-section geometry
# --------------------------
def polar_xy(distance_deg, depth_km):
    """0 deg is source at left; positive distance moves clockwise along top arc."""
    theta = math.radians(180.0 - distance_deg)
    r = MARS_RADIUS_KM - depth_km
    return r * math.cos(theta), r * math.sin(theta)


def taup_path_to_xy(row, path):
    dist_vals = path_field(path, ["dist"])
    depth_vals = path_field(path, ["depth"])
    if dist_vals is None or depth_vals is None or len(depth_vals) < 2:
        return None

    max_dist = max(float(d) for d in dist_vals)
    dist_is_radians = max_dist <= 2.0 * math.pi + 1e-6
    xy = []
    for dist_val, depth_val in zip(dist_vals, depth_vals):
        ddeg = math.degrees(float(dist_val)) if dist_is_radians else float(dist_val)
        # Scale any tiny numeric endpoint mismatch so the path ends at the exact receiver distance.
        ddeg *= row["plot_distance_deg"] / max(row["distance_deg"], 1e-9)
        ddeg = max(0.0, min(row["plot_distance_deg"], ddeg))
        xy.append(polar_xy(ddeg, float(depth_val)))

    # Force exact surface endpoints for visual cleanliness.
    xy[0] = polar_xy(0.0, row["depth_km"])
    xy[-1] = polar_xy(row["plot_distance_deg"], 0.0)
    return np.array(xy)


def draw_arc(ax, radius, start_deg=0, end_deg=150, **kwargs):
    deg = np.linspace(start_deg, end_deg, 600)
    theta = np.deg2rad(180.0 - deg)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), **kwargs)


def add_surface_ticks(ax, start=0, end=150):
    for deg in range(start, end + 1, 10):
        theta = math.radians(180 - deg)
        length = 70 if deg % 20 == 0 else 42
        x0, y0 = MARS_RADIUS_KM * math.cos(theta), MARS_RADIUS_KM * math.sin(theta)
        x1, y1 = (MARS_RADIUS_KM - length) * math.cos(theta), (MARS_RADIUS_KM - length) * math.sin(theta)
        ax.plot([x0, x1], [y0, y1], color="black", lw=1.1, zorder=10)
        if deg % 20 == 0:
            xt, yt = (MARS_RADIUS_KM + 190) * math.cos(theta), (MARS_RADIUS_KM + 190) * math.sin(theta)
            ax.text(xt, yt, f"{deg}°", ha="center", va="center", fontsize=13)


def plot_figure(rows):
    fig, ax = plt.subplots(figsize=(12.5, 6.6), dpi=300)
    ax.set_aspect("equal")
    ax.axis("off")

    # Boundaries
    draw_arc(ax, MARS_RADIUS_KM, color="black", lw=2.0, zorder=1)
    draw_arc(ax, MARS_RADIUS_KM - NEAR_SURFACE_OFFSET_KM, color="black", lw=1.15, zorder=1)
    draw_arc(ax, MARS_RADIUS_KM - 800.0, color="black", lw=1.45, ls=(0, (7, 6)), zorder=1)
    draw_arc(ax, MARS_RADIUS_KM - 1000.0, color="black", lw=1.45, ls=(0, (7, 6)), zorder=1)
    draw_arc(ax, CORE_RADIUS_KM, color="black", lw=1.6, zorder=1)
    add_surface_ticks(ax)

    # Rays
    colors = plt.cm.plasma_r(np.linspace(0.08, 0.92, len(rows)))
    plotted = 0
    for row, color in zip(rows, colors):
        path = get_arrival_path(row, PHASE_TO_PLOT)
        if path is None:
            continue
        xy = taup_path_to_xy(row, path)
        if xy is None:
            continue
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.35, zorder=3)
        sx, sy = polar_xy(row["plot_distance_deg"], 0.0)
        ax.plot(sx, sy, marker="*", ms=11, color="#ff00e6", mec="#ff00e6", zorder=5)
        plotted += 1

    # Source marker: triangle at surface, plus true event-depth ray starts are just below surface.
    src = np.array(polar_xy(0.0, 0.0))
    tri = np.array([[0, 0], [110, 60], [110, -60]])
    ang = math.radians(145)
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    tri2 = tri @ rot.T + src + np.array([-10, 40])
    ax.add_patch(Polygon(tri2, closed=True, facecolor="#1734ff", edgecolor="#1734ff", zorder=6))

    # Labels
    ax.text(src[0] - 120, src[1] + 620, "S Wave", fontsize=18, ha="left", va="center")
    ax.text(260, 2560, "Upper Mantle", fontsize=15, ha="center", va="center")
    ax.text(180, 1470, "Mantle\nTransition Zone", fontsize=15, ha="center", va="center", linespacing=1.2)
    ax.text(170, 330, "Core", fontsize=20, ha="center", va="center")

    for radius, label, shift in [
        (MARS_RADIUS_KM - 800.0, "800 km", 70),
        (MARS_RADIUS_KM - 1000.0, "1000 km", 70),
    ]:
        theta = math.radians(47)
        x, y = radius * math.cos(theta), radius * math.sin(theta)
        ax.text(x + shift, y - 20, label, fontsize=11.5, rotation=-55, ha="left", va="center")

    ax.set_xlim(-3600, 3350)
    ax.set_ylim(-80, 3700)

    for ext in ["png", "pdf", "svg"]:
        out = BASE / f"{OUTPUT_STEM}.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
        print(f"[saved] {out}")
    print(f"[done] plotted {plotted}/{len(rows)} ray paths")


if __name__ == "__main__":
    csv_path = find_existing(EVENTS_CSV_CANDIDATES)
    if csv_path is None:
        raise FileNotFoundError("No arrivals/events CSV found in the script folder.")
    print(f"[info] using CSV: {csv_path}")

    event_rows = load_event_rows(csv_path)
    selected_rows = select_rows_by_target_distances(event_rows)
    if len(selected_rows) != len(TARGET_DISTANCES_DEG):
        print(f"[warning] matched {len(selected_rows)} rows for {len(TARGET_DISTANCES_DEG)} target distances")
    print("[info] selected rows:")
    for row in selected_rows:
        print(f"  {row['event']:7s} target={row['plot_distance_deg']:5.1f}°, csv={row['distance_deg']:7.3f}°, depth={row['depth_km']:6.1f} km")

    plot_figure(selected_rows)
