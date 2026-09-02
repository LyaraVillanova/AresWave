#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_s0185a_fixed_rows_local_filtered_aligned.py

Purpose
-------
Use the exact fixed rows from the PSO CSVs and plot the corresponding waveforms
for ONLY:
    - X
    - TAYAK

This version fixes the problems from the previous attempts:
    1) NO IRIS download. Uses the local denoised ZRT SAC files.
    2) Uses the fixed CSV rows directly for depth/strike/dip/rake/Cost.
    3) Generates a UNIQUE event_id for each snapshot to avoid DSMpy/SPC reuse.
    4) Applies bandpass 0.3--0.9 Hz to real and synthetic.
    5) Applies polarization_filter to real and synthetic.
    6) Aligns synthetic to observed by component using align_by_correlation
       with ±2 s, matching the PSO visual/comparison logic.
    7) Normalizes each plotted window separately, so the panels are readable.
    8) Runs a sanity check: initial / iter15 / iter30 synthetics must not be
       identical. iter30 and final are allowed to be identical when the final best is
       the same CSV row.

Run
---
cd /home/lyara/areswave
/home/lyara/areswave/areswave-venv/bin/python /mnt/data/plot_s0185a_fixed_rows_local_filtered_aligned.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_MAXIMUM_THREADS", "1")

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from obspy import Stream, read, UTCDateTime
from dsmpy import seismicmodel_Mars
from dsmpy.event_Mars import Event, MomentTensor
from dsmpy.station_Mars import Station

from areswave.synthetics_function import (
    generate_synthetics,
    calculate_moment_tensor,
    normalize,
    align_by_correlation,
)
from areswave.denoising import polarization_filter


# =============================================================================
# PATHS / SETTINGS
# =============================================================================
PROJECT_DIR = Path("/home/lyara/areswave")
OUT_DIR = PROJECT_DIR / "figs_s0185a_fixed_rows_local_filtered_aligned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_X = PROJECT_DIR / "figs" / "all_tested_parameters_S0185a_mqs2019kxjd.csv"
CSV_TAYAK = PROJECT_DIR / "figs" / "all_tested_parameters_S0185a_mqs2019kxjd_tayak.csv"

SAC_DIR = PROJECT_DIR / "SAC"
LOCAL_SAC_FILES = [
    SAC_DIR / "S0185a_trlq_denois03.Z.sac",
    SAC_DIR / "S0185a_trlq_denois04.T.sac",
    SAC_DIR / "S0185a_trlq_denois05.R.sac",
]

PARAM_CSV_OUT = OUT_DIR / "S0185a_fixed_rows_local_filtered_aligned_v2_parameters.csv"

SAMPLING_HZ = 20.0
FREQMIN = 0.3
FREQMAX = 0.9
FILTER_ORDER = 4

NSPC = 256
TLEN = 400.0

LATITUDE = 41.59816
LONGITUDE = 90.13083
DISTANCE = 59.8
MAGNITUDE = 3.1
TIME_P = UTCDateTime("2019-06-05T02:13:48")
TIME_S = UTCDateTime("2019-06-05T02:19:47")
CENTROID_TIME = UTCDateTime("2019-06-05 02:06:37")
INITIAL_MT = MomentTensor(
    Mrr=-2.8e20,
    Mrt=-1.9e20,
    Mrp=-1.3e20,
    Mtt=-1.4e20,
    Mtp=-5.3e20,
    Mpp=1.8e20,
)

STATION = Station(name="ELYSE", network="XB", latitude=4.502384, longitude=135.623447)

P_PRE, P_POST = 5.0, 5.0
S_PRE, S_POST = 5.0, 10.0
MAX_SHIFT_SECONDS = 2.0

RUNS = [
    {
        "label": "X",
        "base_event_id": "S0185a_mqs2019kxjd",
        "csv": CSV_X,
        "model_builder": seismicmodel_Mars.SeismicModel.test2,
        "rows": {
            "initial": 41,
            "iter15": 437,
            "iter30": 984,
            "final": "best",
        },
    },
    {
        "label": "TAYAK",
        "base_event_id": "S0185a_mqs2019kxjd_tayak",
        "csv": CSV_TAYAK,
        "model_builder": seismicmodel_Mars.SeismicModel.tayak,
        "rows": {
            "initial": 39,
            "iter15": 634,
            "iter30": 1296,
            "final": "best",
        },
    },
]

ROW_LABELS = {
    "initial": "Initial\nbest of iter. 1",
    "iter15": "Best by\niter. 15",
    "iter30": "Best before\nrestart/iter. 30",
    "final": "Final best",
}


def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.makedirs(str(OUT_DIR), exist_ok=True)


def safe_demean_normalize(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = x - np.nanmean(x)
    amp = np.nanmax(np.abs(x))
    if not np.isfinite(amp) or amp == 0:
        return x
    return x / amp


def bandpass_array(x, fs=SAMPLING_HZ, freqmin=FREQMIN, freqmax=FREQMAX, order=FILTER_ORDER):
    x = np.asarray(x, dtype=float)
    if x.size < 10:
        return x
    nyq = 0.5 * fs
    b, a = butter(order, [freqmin / nyq, freqmax / nyq], btype="band")
    return filtfilt(b, a, x)


def select_trace(st, suffix):
    suffix = suffix.upper()
    matches = [tr for tr in st if tr.stats.channel.upper().endswith(suffix)]
    if not matches:
        raise ValueError(f"Could not find channel ending with {suffix}. Channels: {[tr.stats.channel for tr in st]}")
    if len(matches) > 1:
        print(f"WARNING: multiple {suffix} traces found; using first: {[tr.id for tr in matches]}")
    return matches[0]


def trim_or_pad(x, n):
    x = np.asarray(x, dtype=float)
    if len(x) >= n:
        return x[:n]
    out = np.zeros(n, dtype=float)
    out[:len(x)] = x
    return out


def stage_process_zrt(z, r, t):
    """Bandpass 0.3--0.9 Hz + polarization_filter."""
    z_bp = bandpass_array(z)
    r_bp = bandpass_array(r)
    t_bp = bandpass_array(t)
    pol = polarization_filter([z_bp, r_bp, t_bp], SAMPLING_HZ)
    return {
        "Z": np.asarray(pol[0], dtype=float),
        "R": np.asarray(pol[1], dtype=float),
        "T": np.asarray(pol[2], dtype=float),
    }


def load_real_local():
    missing = [str(p) for p in LOCAL_SAC_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing local SAC files: " + ", ".join(missing))

    st = Stream()
    for sac_file in LOCAL_SAC_FILES:
        tr = read(str(sac_file))[0]
        tr.detrend("linear")
        tr.taper(max_percentage=0.05)
        tr.resample(SAMPLING_HZ)
        tr.trim(starttime=TIME_P - 10, endtime=TIME_P + 380)
        st += tr

    z = np.asarray(select_trace(st, "Z").data, dtype=float)
    r = np.asarray(select_trace(st, "R").data, dtype=float)
    t = np.asarray(select_trace(st, "T").data, dtype=float)
    n = min(len(z), len(r), len(t))
    z, r, t = z[:n], r[:n], t[:n]

    real_proc = stage_process_zrt(z, r, t)
    n2 = min(len(real_proc["Z"]), len(real_proc["R"]), len(real_proc["T"]))
    real_proc = {k: normalize(v[:n2]) for k, v in real_proc.items()}

    start_time = select_trace(st, "Z").stats.starttime
    p_idx = int((TIME_P - start_time) * SAMPLING_HZ)
    s_idx = int((TIME_S - start_time) * SAMPLING_HZ)

    print("Real data: local denoised ZRT SAC")
    print(f"  files: {[str(p) for p in LOCAL_SAC_FILES]}")
    print(f"  n={n2}, p_idx={p_idx}, s_idx={s_idx}")
    print("  processing: bandpass 0.3-0.9 Hz + polarization_filter")

    return {
        "data": real_proc,
        "p_idx": p_idx,
        "s_idx": s_idx,
        "n": n2,
    }


def mt_dict_from_sdr(depth, strike, dip, rake):
    mts = calculate_moment_tensor(
        MAGNITUDE,
        strike,
        dip,
        rake,
        depth,
        DISTANCE,
        frequency_range=(0.1, 1.0),
        interval=0.1,
    )
    if mts is None or len(mts) == 0:
        raise RuntimeError("calculate_moment_tensor returned empty/None.")

    return {
        "Mrr": float(np.mean([m["moment_tensor"].Mrr for m in mts])),
        "Mrt": float(np.mean([m["moment_tensor"].Mrt for m in mts])),
        "Mrp": float(np.mean([m["moment_tensor"].Mrp for m in mts])),
        "Mtt": float(np.mean([m["moment_tensor"].Mtt for m in mts])),
        "Mtp": float(np.mean([m["moment_tensor"].Mtp for m in mts])),
        "Mpp": float(np.mean([m["moment_tensor"].Mpp for m in mts])),
    }


def build_event(snapshot):
    mt = snapshot["mt"]
    tensor = MomentTensor(
        mt["Mrr"],
        mt["Mrt"],
        mt["Mrp"],
        mt["Mtt"],
        mt["Mtp"],
        mt["Mpp"],
    )

    return Event(
        event_id=snapshot["unique_event_id"],
        latitude=LATITUDE,
        longitude=LONGITUDE,
        depth=snapshot["depth"],
        mt=tensor,
        centroid_time=CENTROID_TIME,
        source_time_function=None,
    )


def snapshot_from_csv(run_cfg, snap_key, df):
    idx_cfg = run_cfg["rows"][snap_key]
    if idx_cfg == "best":
        idx = int(df["Cost"].astype(float).idxmin())
    else:
        idx = int(idx_cfg)
    row = df.loc[idx]
    depth = float(row["Depth (km)"])
    strike = float(row["Strike (°)"])
    dip = float(row["Dip (°)"])
    rake = float(row["Rake (°)"])

    unique_event_id = f"S0185a_{run_cfg['label']}_{snap_key}_row{idx}"

    return {
        "run_label": run_cfg["label"],
        "base_event_id": run_cfg["base_event_id"],
        "unique_event_id": unique_event_id,
        "snapshot_key": snap_key,
        "snapshot_label": ROW_LABELS[snap_key],
        "row_index": int(idx),
        "csv_cost": float(row["Cost"]),
        "depth": depth,
        "strike": strike,
        "dip": dip,
        "rake": rake,
        "mt": mt_dict_from_sdr(depth, strike, dip, rake),
    }


def generate_snapshot_synthetic(snapshot, seismic_model, real):
    event = build_event(snapshot)
    output = generate_synthetics(event, [STATION], seismic_model, TLEN, NSPC, SAMPLING_HZ)

    ts = output.ts
    max_idx = np.searchsorted(ts, TLEN)

    z = np.asarray(output["Z", "ELYSE_XB"][:max_idx], dtype=float)
    r = np.asarray(output["R", "ELYSE_XB"][:max_idx], dtype=float)
    t = np.asarray(output["T", "ELYSE_XB"][:max_idx], dtype=float)

    syn_proc = stage_process_zrt(z, r, t)

    n = real["n"]
    syn_proc = {comp: normalize(trim_or_pad(syn_proc[comp], n)) for comp in ["Z", "R", "T"]}

    max_shift_samples = int(MAX_SHIFT_SECONDS * SAMPLING_HZ)
    aligned = {}
    for comp in ["Z", "R", "T"]:
        aligned[comp] = align_by_correlation(
            real["data"][comp],
            syn_proc[comp],
            max_shift_samples,
        )

    snapshot["syn"] = aligned
    return snapshot


def rms_diff(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    return float(np.sqrt(np.mean((np.asarray(a[:n]) - np.asarray(b[:n])) ** 2)))


def sanity_check_not_identical(run_label, snapshots):
    pairs = [("initial", "iter15"), ("iter15", "iter30"), ("initial", "iter30")]
    diffs = {}
    for a, b in pairs:
        vals = []
        for comp in ["Z", "R", "T"]:
            vals.append(rms_diff(snapshots[a]["syn"][comp], snapshots[b]["syn"][comp]))
        diffs[f"{a}_vs_{b}"] = float(np.nanmax(vals))

    print(f"Synthetic non-identity check for {run_label}: {diffs}")

    if all(v < 1e-8 for v in diffs.values()):
        raise RuntimeError(
            f"All synthetic snapshots for {run_label} are numerically identical. "
            "That means the parameters are not being propagated to the synthetics."
        )


def window_slice(center_idx, pre_s, post_s, n):
    i0 = max(0, int(center_idx - pre_s * SAMPLING_HZ))
    i1 = min(n, int(center_idx + post_s * SAMPLING_HZ))
    return i0, i1


def get_panel_data(real, snap, comp, phase):
    if phase == "P":
        i0, i1 = window_slice(real["p_idx"], P_PRE, P_POST, real["n"])
        center = real["p_idx"]
    else:
        i0, i1 = window_slice(real["s_idx"], S_PRE, S_POST, real["n"])
        center = real["s_idx"]

    time_axis = (np.arange(i0, i1) - center) / SAMPLING_HZ
    obs = safe_demean_normalize(real["data"][comp][i0:i1])
    syn = safe_demean_normalize(snap["syn"][comp][i0:i1])
    return time_axis, obs, syn


def plot_panel(ax, real, snap, comp, phase, title):
    t, obs, syn = get_panel_data(real, snap, comp, phase)

    ax.plot(t, obs, color="black", lw=1.2, label="Observed")
    ax.plot(t, syn, color="red", lw=1.05, alpha=0.85, label="Synthetic")
    ax.axvline(0.0, color="0.45", lw=0.8, ls=":")
    ax.set_ylim(-1.08, 1.08)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_title(title, fontsize=8.8)
    ax.set_xlabel(f"Time from {phase} pick (s)", fontsize=8)
    ax.set_ylabel("Norm. amp.", fontsize=8)
    ax.tick_params(labelsize=8)


def side_text(snap):
    return (
        f"{snap['snapshot_label']}\n"
        f"row {snap['row_index']}\n"
        f"CSV cost={snap['csv_cost']:.4f}\n"
        f"h={snap['depth']:.1f} km\n"
        f"str={snap['strike']:.1f}°\n"
        f"dip={snap['dip']:.1f}°\n"
        f"rake={snap['rake']:.1f}°"
    )


def ordered_snapshots(snapshots):
    return [
        snapshots["initial"],
        snapshots["iter15"],
        snapshots["iter30"],
        snapshots["final"],
    ]


def plot_4components(run_label, real, snapshots):
    fig, axes = plt.subplots(
        4,
        5,
        figsize=(18.0, 10.5),
        gridspec_kw={"width_ratios": [1.20, 2.6, 2.6, 2.6, 2.6]},
        squeeze=False,
    )

    columns = [
        ("Z", "P", "PZ"),
        ("R", "P", "PR"),
        ("Z", "S", "SZ"),
        ("T", "S", "ST"),
    ]

    for i, snap in enumerate(ordered_snapshots(snapshots)):
        axm = axes[i, 0]
        axm.axis("off")
        axm.text(0.02, 0.52, side_text(snap), ha="left", va="center", fontsize=9.0)

        for j, (comp, phase, title) in enumerate(columns, start=1):
            plot_panel(axes[i, j], real, snap, comp, phase, title)

    axes[0, 1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(
        f"S0185a: fixed-row PSO snapshots for {run_label} "
        f"(local SAC, 0.3-0.9 Hz + polarization, ±2 s alignment)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])

    ensure_out_dir()
    out_png = OUT_DIR / f"S0185a_{run_label}_4components_local_filtered_aligned.png"
    out_pdf = OUT_DIR / f"S0185a_{run_label}_4components_local_filtered_aligned.pdf"
    fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_pdf), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def plot_compact(run_label, real, snapshots):
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(14.5, 10.5),
        gridspec_kw={"width_ratios": [1.20, 3.3, 3.3]},
        squeeze=False,
    )

    columns = [
        ("Z", "P", "PZ"),
        ("T", "S", "ST"),
    ]

    for i, snap in enumerate(ordered_snapshots(snapshots)):
        axm = axes[i, 0]
        axm.axis("off")
        axm.text(0.02, 0.52, side_text(snap), ha="left", va="center", fontsize=9.0)

        for j, (comp, phase, title) in enumerate(columns, start=1):
            plot_panel(axes[i, j], real, snap, comp, phase, title)

    axes[0, 1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(
        f"S0185a: fixed-row PSO snapshots for {run_label} "
        f"(local SAC, 0.3-0.9 Hz + polarization, ±2 s alignment)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])

    ensure_out_dir()
    out_png = OUT_DIR / f"S0185a_{run_label}_compact_local_filtered_aligned.png"
    out_pdf = OUT_DIR / f"S0185a_{run_label}_compact_local_filtered_aligned.pdf"
    fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_pdf), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main():
    ensure_out_dir()

    print(f"Output directory: {OUT_DIR}")
    real = load_real_local()

    param_rows = []

    for run_cfg in RUNS:
        print("\n" + "=" * 80)
        print(f"RUN {run_cfg['label']}")
        print("=" * 80)

        if not run_cfg["csv"].exists():
            raise FileNotFoundError(f"Missing CSV: {run_cfg['csv']}")

        df = pd.read_csv(run_cfg["csv"])
        best_idx = int(df["Cost"].astype(float).idxmin())
        print(f"CSV global best row for {run_cfg['label']}: {best_idx}, cost={float(df.loc[best_idx, 'Cost']):.6f}")
        seismic_model = run_cfg["model_builder"]()

        snapshots = {}
        for snap_key in ["initial", "iter15", "iter30", "final"]:
            snap = snapshot_from_csv(run_cfg, snap_key, df)

            print(
                f"{run_cfg['label']} {snap_key:7s} | "
                f"event_id={snap['unique_event_id']} | "
                f"row={snap['row_index']} cost={snap['csv_cost']:.6f} "
                f"h={snap['depth']:.2f} str={snap['strike']:.2f} "
                f"dip={snap['dip']:.2f} rake={snap['rake']:.2f}"
            )

            snap = generate_snapshot_synthetic(snap, seismic_model, real)
            snapshots[snap_key] = snap

            param_rows.append({
                "run_label": run_cfg["label"],
                "base_event_id": run_cfg["base_event_id"],
                "unique_event_id": snap["unique_event_id"],
                "snapshot_key": snap_key,
                "snapshot_label": snap["snapshot_label"].replace("\n", " "),
                "row_index": snap["row_index"],
                "csv_cost": snap["csv_cost"],
                "depth_km": snap["depth"],
                "strike_deg": snap["strike"],
                "dip_deg": snap["dip"],
                "rake_deg": snap["rake"],
                **{f"mt_{k}": v for k, v in snap["mt"].items()},
            })

        sanity_check_not_identical(run_cfg["label"], snapshots)

        plot_4components(run_cfg["label"], real, snapshots)
        plot_compact(run_cfg["label"], real, snapshots)

    pd.DataFrame(param_rows).to_csv(str(PARAM_CSV_OUT), index=False)
    print(f"Saved: {PARAM_CSV_OUT}")
    print("DONE.")


if __name__ == "__main__":
    main()
