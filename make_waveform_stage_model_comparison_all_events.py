#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_waveform_stage_model_comparison_all_events.py

Goal
----
For ALL events, compare observed and synthetic waveform windows across:
  - all preprocessing stages:
      1) raw
      2) bandpass 0.1--0.5 Hz
      3) bandpass 0.1--0.8 Hz
      4) bandpass 0.3--0.9 Hz
      5) bandpass 0.3--0.9 Hz + polarization
  - all structural choices:
      A) Model X
      B) TAYAK
      C) best 17-event MTZ model
      D) best event-by-event model

For each event, the script creates a multi-row page:
  rows   = model × stage
  cols   = PZ, SZ, PR, ST

Observed = black
Synthetic = red

The synthetic trace in each panel is aligned to the observed trace using the
best sign (+/-) and best lag within the selected panel window. This panel-level
CC is intended for VISUAL/STAGE comparison only. It is NOT the official
inversion misfit.

Outputs
-------
/home/lyara/areswave/figs_waveform_stage_model_comparison_all_events/
    waveform_stage_model_comparison_all_events.pdf
    per_event_png/*.png
    waveform_stage_model_diagnostics.csv
    waveform_stage_model_summary_by_event.csv
    waveform_stage_model_summary_by_model_stage.csv
    waveform_stage_model_notes.txt

How to run
----------
cd /home/lyara/areswave
/home/lyara/areswave/areswave-venv/bin/python make_waveform_stage_model_comparison_all_events.py
"""

import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, filtfilt

# Optional, but expected in the user's environment.
from areswave.denoising import polarization_filter


# =============================================================================
# USER SETTINGS
# =============================================================================
PROJECT_DIR = Path("/home/lyara/areswave")
ARES_SRC_DIR = PROJECT_DIR / "areswave"
INV_SCRIPT_PATH = ARES_SRC_DIR / "inversion_17_models_1400.py"

OUT_DIR = PROJECT_DIR / "figs_waveform_stage_model_comparison_all_events"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_EVENT_PNG_DIR = OUT_DIR / "per_event_png"
PER_EVENT_PNG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "waveform_stage_model_comparison_all_events.pdf"
OUT_CSV = OUT_DIR / "waveform_stage_model_diagnostics.csv"
OUT_SUMMARY_EVENT_CSV = OUT_DIR / "waveform_stage_model_summary_by_event.csv"
OUT_SUMMARY_MODEL_STAGE_CSV = OUT_DIR / "waveform_stage_model_summary_by_model_stage.csv"
OUT_NOTES = OUT_DIR / "waveform_stage_model_notes.txt"

TAYAK_MODEL_PATH = PROJECT_DIR / "models" / "TAYAK.nd"
BEST17_MODEL_ID = "Geophysical_model298"
BEST17_MODEL_PATH = PROJECT_DIR / "models" / f"{BEST17_MODEL_ID}.nd"

SAMPLING_HZ = 20.0

STAGE_ORDER = [
    "raw",
    "bp_01_05",
    "bp_01_08",
    "bp_03_09",
    "bp_03_09_pol",
]
STAGE_LABELS = {
    "raw": "raw / no bandpass",
    "bp_01_05": "bandpass 0.1-0.5",
    "bp_01_08": "bandpass 0.1-0.8",
    "bp_03_09": "bandpass 0.3-0.9",
    "bp_03_09_pol": "0.3-0.9 + pol.",
}

# Event-by-event best models.
EVENT_BEST_MODELS = {
    "S0133a": {"model_id": "Geophysical_model225", "source_path": "/home/lyara/areswave/models/Geophysical_model225.nd"},
    "S0152a": {"model_id": "MD_model88", "source_path": "/home/lyara/areswave/models/MD_model88.nd"},
    "S0167a": {"model_id": "Geophysical_model64", "source_path": "/home/lyara/areswave/models/Geophysical_model64.nd"},
    "S0167b": {"model_id": "MD_model37", "source_path": "/home/lyara/areswave/models/MD_model37.nd"},
    "S0185a": {"model_id": "MD_model78", "source_path": "/home/lyara/areswave/models/MD_model78.nd"},
    "S0226b": {"model_id": "Geophysical_model776", "source_path": "/home/lyara/areswave/models/Geophysical_model776.nd"},
    "S0234c": {"model_id": "Geophysical_model979", "source_path": "/home/lyara/areswave/models/Geophysical_model979.nd"},
    "S0254b": {"model_id": "Geophysical_model277", "source_path": "/home/lyara/areswave/models/Geophysical_model277.nd"},
    "S0345a": {"model_id": "CD_model68", "source_path": "/home/lyara/areswave/models/CD_model68.nd"},
    "S0395a": {"model_id": "Geophysical_model573", "source_path": "/home/lyara/areswave/models/CD_model44.nd"},
    "S0421b": {"model_id": "MD_model14", "source_path": "/home/lyara/areswave/models/MD_model14.nd"},
    "S0976a": {"model_id": "Geophysical_model837", "source_path": "/home/lyara/areswave/models/Geophysical_model837.nd"},
    "S1000a": {"model_id": "AK_model_62", "source_path": "/home/lyara/areswave/models/AK_model_62.nd"},
    "S1094b": {"model_id": "Geophysical_model319", "source_path": "/home/lyara/areswave/models/Geophysical_model319.nd"},
    "S1102a": {"model_id": "Geophysical_model58", "source_path": "/home/lyara/areswave/models/Geophysical_model58.nd"},
    "S1153a": {"model_id": "Geophysical_model489", "source_path": "/home/lyara/areswave/models/Geophysical_model489.nd"},
    "S1415a": {"model_id": "MD_model34", "source_path": "/home/lyara/areswave/models/MD_model34.nd"},
}

MODEL_SPECS = [
    {"key": "X", "kind": "base_x", "label": "Model X", "model_id": "X"},
    {"key": "TAYAK", "kind": "nd", "label": "TAYAK", "model_id": "TAYAK", "path": TAYAK_MODEL_PATH},
    {"key": "Best17", "kind": "nd", "label": f"17-event best ({BEST17_MODEL_ID})", "model_id": BEST17_MODEL_ID, "path": BEST17_MODEL_PATH},
    {"key": "EventBest", "kind": "event_best", "label": "Event best", "model_id": "event-specific"},
]


# =============================================================================
# IMPORT INVERSION SCRIPT
# =============================================================================
def import_inversion_script():
    if not INV_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Cannot find inversion script: {INV_SCRIPT_PATH}")

    os.chdir(PROJECT_DIR)
    for p in (PROJECT_DIR, ARES_SRC_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    print(f"[INFO] Importing inversion script from: {INV_SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("inv17models_stage_model_compare", str(INV_SCRIPT_PATH))
    inv = importlib.util.module_from_spec(spec)
    sys.modules["inv17models_stage_model_compare"] = inv
    spec.loader.exec_module(inv)
    return inv


inv = import_inversion_script()


# =============================================================================
# SMALL HELPERS
# =============================================================================
def safe_demean_normalize(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = x - np.nanmean(x)
    amp = np.nanmax(np.abs(x))
    if (not np.isfinite(amp)) or amp == 0:
        return x
    return x / amp


def bandpass_array(x, fs, freqmin, freqmax, order=4):
    x = np.asarray(x, dtype=float)
    if x.size < 12:
        return x.copy()
    nyq = 0.5 * fs
    low = freqmin / nyq
    high = freqmax / nyq
    b, a = butter(order, [low, high], btype="band")
    try:
        return filtfilt(b, a, x)
    except Exception:
        return x.copy()


def extract_zrt_triplet(st):
    out = {}
    for comp in ["Z", "R", "T"]:
        arr = inv._extract_single_component(st, comp)
        if arr is None:
            raise ValueError(f"Could not extract component {comp} from stream.")
        out[comp] = np.asarray(arr, dtype=float)
    n = min(len(out["Z"]), len(out["R"]), len(out["T"]))
    return {k: np.asarray(v[:n], dtype=float) for k, v in out.items()}


def preprocess_stages(triplet, fs=SAMPLING_HZ):
    raw = {c: np.asarray(triplet[c], dtype=float) for c in ["Z", "R", "T"]}

    bp_01_05 = {c: bandpass_array(raw[c], fs, 0.1, 0.5) for c in ["Z", "R", "T"]}
    bp_01_08 = {c: bandpass_array(raw[c], fs, 0.1, 0.8) for c in ["Z", "R", "T"]}
    bp_03_09 = {c: bandpass_array(raw[c], fs, 0.3, 0.9) for c in ["Z", "R", "T"]}

    try:
        pol = polarization_filter([bp_03_09["Z"], bp_03_09["R"], bp_03_09["T"]], fs)
        bp_03_09_pol = {
            "Z": np.asarray(pol[0], dtype=float),
            "R": np.asarray(pol[1], dtype=float),
            "T": np.asarray(pol[2], dtype=float),
        }
    except Exception:
        bp_03_09_pol = {c: np.asarray(bp_03_09[c], dtype=float) for c in ["Z", "R", "T"]}

    # Trim all to same length.
    n = min(
        len(raw["Z"]), len(raw["R"]), len(raw["T"]),
        len(bp_01_05["Z"]), len(bp_01_08["Z"]), len(bp_03_09["Z"]), len(bp_03_09_pol["Z"]),
    )
    stages = {
        "raw": {c: raw[c][:n] for c in ["Z", "R", "T"]},
        "bp_01_05": {c: bp_01_05[c][:n] for c in ["Z", "R", "T"]},
        "bp_01_08": {c: bp_01_08[c][:n] for c in ["Z", "R", "T"]},
        "bp_03_09": {c: bp_03_09[c][:n] for c in ["Z", "R", "T"]},
        "bp_03_09_pol": {c: bp_03_09_pol[c][:n] for c in ["Z", "R", "T"]},
    }
    return stages


def build_model_from_nd(nd_path):
    nd_path = Path(nd_path)
    if not nd_path.exists():
        raise FileNotFoundError(f"Cannot find model .nd file: {nd_path}")
    base_model = inv.SeismicModel.test2()
    suite_model = inv.load_suite_model_from_nd(str(nd_path), base_model)
    full_model = inv.build_full_model_from_suite(base_model, suite_model)
    return suite_model, full_model


def build_model_from_spec(spec, event_id=None):
    kind = spec["kind"]
    if kind == "base_x":
        suite_model = None
        full_model = inv.SeismicModel.test2()
        return suite_model, full_model, spec["label"], spec["model_id"], ""
    if kind == "nd":
        suite_model, full_model = build_model_from_nd(spec["path"])
        return suite_model, full_model, spec["label"], spec["model_id"], str(spec["path"])
    if kind == "event_best":
        if event_id not in EVENT_BEST_MODELS:
            raise KeyError(f"No event-best model defined for {event_id}")
        meta = EVENT_BEST_MODELS[event_id]
        suite_model, full_model = build_model_from_nd(meta["source_path"])
        return suite_model, full_model, f"Event best ({meta['model_id']})", meta["model_id"], meta["source_path"]
    raise ValueError(f"Unknown model spec kind: {kind}")


def source_depth_for_event(evt, suite_model):
    if "depth" in evt and np.isfinite(float(evt.get("depth", np.nan))):
        return float(evt["depth"])
    if suite_model is not None and hasattr(suite_model, "source_depth_km"):
        return float(suite_model.source_depth_km)
    return 30.0


def cut_window_pair(obs_arr, syn_arr, pick_seconds, window):
    """
    Same branch logic as earlier plotting scripts:
    - if pick exists: cut both around that same pick
    - else: cut obs around obs peak and syn around syn peak
    """
    if pick_seconds is not None and np.isfinite(pick_seconds):
        ow = inv._cut_window_from_pick(obs_arr, float(pick_seconds), window)
        sw = inv._cut_window_from_pick(syn_arr, float(pick_seconds), window)
        window_type = "pick-based"
    else:
        iobs = inv._pick_peak_index(obs_arr)
        isyn = inv._pick_peak_index(syn_arr)
        ow = inv._cut_around_peak(obs_arr, iobs, window)
        sw = inv._cut_around_peak(syn_arr, isyn, window)
        window_type = "peak-based"

    if ow is None or sw is None:
        return None, None, window_type
    n = min(len(ow), len(sw))
    if n < 8:
        return None, None, window_type
    return np.asarray(ow[:n], dtype=float), np.asarray(sw[:n], dtype=float), window_type


def best_align_cc(obs_w, syn_w):
    """
    Visualization metric only:
    choose best sign (+/-) and best lag using inv._crosscorr_best_shift,
    then return aligned/normalized traces and panel CC.
    """
    best = None
    obs_n_base = safe_demean_normalize(obs_w)

    for sign in (+1.0, -1.0):
        syn_signed = sign * np.asarray(syn_w, dtype=float)
        syn_n_base = safe_demean_normalize(syn_signed)
        try:
            shift, cc = inv._crosscorr_best_shift(obs_n_base, syn_n_base, inv.MAX_SHIFT_SAMPLES)
            syn_aligned = inv._apply_shift(syn_n_base, shift)
        except Exception:
            shift = 0
            syn_aligned = syn_n_base
            if np.nanstd(obs_n_base) == 0 or np.nanstd(syn_n_base) == 0:
                cc = np.nan
            else:
                cc = float(np.corrcoef(obs_n_base, syn_n_base)[0, 1])

        if best is None or (np.isfinite(cc) and (not np.isfinite(best["cc"]) or cc > best["cc"])):
            best = {
                "obs": obs_n_base,
                "syn": syn_aligned,
                "cc": float(cc) if np.isfinite(cc) else np.nan,
                "shift_samples": int(shift),
                "shift_s": float(shift) * float(inv.DT),
                "sign": "+" if sign > 0 else "-",
            }
    return best


def phase_windows_for_event(event_id):
    p_window = inv.P_WINDOW
    s_window = inv.S_WINDOW_EXTENDED if event_id in inv.S_WINDOW_EXTENDED_EVENTS else inv.S_WINDOW
    return p_window, s_window


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_event_model(evt, event_id, model_spec, model_cache, obs_cache):
    cache_key = (model_spec["key"], event_id) if model_spec["kind"] == "event_best" else (model_spec["key"], "common")
    if cache_key not in model_cache:
        model_cache[cache_key] = build_model_from_spec(model_spec, event_id=event_id)
    suite_model, full_model, model_label, model_id, model_path = model_cache[cache_key]

    if event_id not in obs_cache:
        obs_st = inv.load_observed_waveforms(event_id)
        if obs_st is None:
            raise RuntimeError(f"No observed waveforms for {event_id}")
        obs_cache[event_id] = obs_st
    else:
        obs_st = obs_cache[event_id]

    event_depth_km = source_depth_for_event(evt, suite_model)
    syn_st = inv.synthesize_event_traces(evt, full_model, source_depth_km=event_depth_km)
    if syn_st is None:
        raise RuntimeError(f"No synthetic waveforms produced for {event_id}, model {model_spec['key']}")

    obs_triplet = extract_zrt_triplet(obs_st)
    syn_triplet = extract_zrt_triplet(syn_st)

    obs_stages = preprocess_stages(obs_triplet, SAMPLING_HZ)
    syn_stages = preprocess_stages(syn_triplet, SAMPLING_HZ)

    p_pick = inv._event_phase_pick_seconds(evt, "p", obs_st)
    s_pick = inv._event_phase_pick_seconds(evt, "s", obs_st)
    p_window, s_window = phase_windows_for_event(event_id)

    panel_defs = [
        ("PZ", "Z", p_pick, p_window),
        ("SZ", "Z", s_pick, s_window),
        ("PR", "R", p_pick, p_window),
        ("ST", "T", s_pick, s_window),
    ]

    panel_results = {}
    panel_rows = []

    for stage in STAGE_ORDER:
        panel_results[stage] = {}
        for panel_name, comp, pick_seconds, window in panel_defs:
            obs_arr = obs_stages[stage][comp]
            syn_arr = syn_stages[stage][comp]
            ow, sw, window_type = cut_window_pair(obs_arr, syn_arr, pick_seconds, window)

            if ow is None or sw is None:
                panel_results[stage][panel_name] = None
                panel_rows.append({
                    "event_id": event_id,
                    "model_key": model_spec["key"],
                    "model_label": model_label,
                    "model_id": model_id,
                    "stage": stage,
                    "panel": panel_name,
                    "component": comp,
                    "window_type": window_type,
                    "cc": np.nan,
                    "shift_samples": np.nan,
                    "shift_s": np.nan,
                    "sign": "",
                    "npts": 0,
                })
                continue

            align = best_align_cc(ow, sw)
            align["window_type"] = window_type
            align["window"] = window
            align["component"] = comp
            align["npts"] = min(len(align["obs"]), len(align["syn"]))
            panel_results[stage][panel_name] = align

            panel_rows.append({
                "event_id": event_id,
                "model_key": model_spec["key"],
                "model_label": model_label,
                "model_id": model_id,
                "stage": stage,
                "panel": panel_name,
                "component": comp,
                "window_type": window_type,
                "cc": align["cc"],
                "shift_samples": align["shift_samples"],
                "shift_s": align["shift_s"],
                "sign": align["sign"],
                "npts": align["npts"],
            })

    return {
        "event_id": event_id,
        "model_key": model_spec["key"],
        "model_label": model_label,
        "model_id": model_id,
        "model_path": model_path,
        "source_depth_km": event_depth_km,
        "panel_results": panel_results,
        "panel_rows": panel_rows,
    }


# =============================================================================
# PLOTTING
# =============================================================================
def plot_event_page(event_id, results_for_event):
    rows = []
    by_key = {r["model_key"]: r for r in results_for_event if r is not None}
    for spec in MODEL_SPECS:
        result = by_key.get(spec["key"])
        if result is None:
            continue
        for stage in STAGE_ORDER:
            rows.append((result, stage))

    nrows = len(rows)
    ncols = 4
    fig_h = max(10.0, 1.22 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, fig_h), squeeze=False)

    panel_order = ["PZ", "SZ", "PR", "ST"]

    for i, (result, stage) in enumerate(rows):
        for j, panel_name in enumerate(panel_order):
            ax = axes[i, j]
            item = result["panel_results"][stage].get(panel_name)

            if item is None:
                ax.text(0.5, 0.5, "no valid window", ha="center", va="center", transform=ax.transAxes, fontsize=7)
                ax.set_axis_off()
                continue

            n = min(len(item["obs"]), len(item["syn"]))
            window = item["window"]
            t = np.arange(n, dtype=float) * float(inv.DT) + float(window[0])

            ax.plot(t, item["obs"][:n], color="black", lw=0.95, label="Observed")
            ax.plot(t, item["syn"][:n], color="red", lw=0.90, alpha=0.85, label="Synthetic")
            ax.axvline(0.0, color="0.35", lw=0.7, ls=":")
            ax.set_xlim(float(window[0]), float(window[1]))
            ax.set_ylim(-1.08, 1.08)
            ax.grid(True, alpha=0.22, lw=0.45)

            if i == 0:
                ax.set_title(panel_name, fontsize=10)

            if j == 0:
                ax.set_ylabel(f"{result['model_key']} | {STAGE_LABELS[stage]}\nnormalized amp.", fontsize=7)

            if i == nrows - 1:
                ax.set_xlabel("Time from pick/window center (s)", fontsize=7)

            ax.tick_params(labelsize=7)
            txt = f"CC={item['cc']:.2f} | sign={item['sign']} | shift={item['shift_s']:+.2f}s"
            ax.text(
                0.02, 0.95, txt,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=6.7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.80),
            )

    axes[0, 0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle(
        f"{event_id}: waveform stages for Model X, TAYAK, 17-event best, and event-best model",
        fontsize=12,
        y=0.997,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.992])
    return fig


# =============================================================================
# SUMMARIES
# =============================================================================
def write_notes(diag, summary_event, summary_model_stage, out_path):
    lines = []
    lines.append("Waveform stage/model comparison notes")
    lines.append("=" * 46)
    lines.append("")
    lines.append("Important:")
    lines.append("- The panel CC values are visualization-stage diagnostics, not the official inversion misfit.")
    lines.append("- Each panel is aligned independently using the best sign (+/-) and best lag.")
    lines.append("- Therefore, this file is best used to discuss how waveform agreement evolves")
    lines.append("  across preprocessing stages and across model choices.")
    lines.append("")

    if len(summary_model_stage):
        lines.append("Mean panel CC by model and stage:")
        for _, row in summary_model_stage.iterrows():
            lines.append(
                f"- {row['model_key']:9s} | {row['stage']:11s} | "
                f"mean CC={row['mean_cc']:.3f} | median CC={row['median_cc']:.3f} | n={int(row['n_panels'])}"
            )
        lines.append("")

    if len(summary_event):
        lines.append("Examples of strongest EventBest improvement over Best17 in final stage:")
        tmp = summary_event.copy()
        final_stage = tmp[tmp["stage"] == "bp_03_09_pol"].copy()
        wide = final_stage.pivot_table(index=["event_id", "panel"], columns="model_key", values="mean_cc", aggfunc="first")
        if "EventBest" in wide.columns and "Best17" in wide.columns:
            wide["EventBest_minus_Best17"] = wide["EventBest"] - wide["Best17"]
            wide = wide.sort_values("EventBest_minus_Best17", ascending=False)
            for idx, row in wide.head(12).iterrows():
                event_id, panel = idx
                lines.append(
                    f"- {event_id} {panel}: EventBest={row['EventBest']:.3f}, "
                    f"Best17={row['Best17']:.3f}, improvement={row['EventBest_minus_Best17']:.3f}"
                )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================
def main():
    events = inv.load_event_catalog(str(PROJECT_DIR / inv.CSV_FILE))
    events = events.dropna(subset=["event_id", "latitude", "longitude", "depth"]).reset_index(drop=True)

    obs_cache = {}
    model_cache = {}
    all_results = {}
    diag_rows = []

    with PdfPages(OUT_PDF) as pdf:
        for _, evt in events.iterrows():
            event_id = str(evt["event_id"]).strip()
            print(f"\n=== {event_id} ===")
            results_for_event = []

            for spec in MODEL_SPECS:
                if spec["kind"] == "event_best" and event_id not in EVENT_BEST_MODELS:
                    print(f"[{event_id}] skipping EventBest; not defined.")
                    continue
                try:
                    result = evaluate_event_model(evt, event_id, spec, model_cache, obs_cache)
                    results_for_event.append(result)
                    diag_rows.extend(result["panel_rows"])
                    print(f"[{event_id}] done {spec['key']}")
                except Exception as exc:
                    print(f"[ERROR] {event_id} | {spec['key']}: {exc}")

            if not results_for_event:
                continue

            all_results[event_id] = results_for_event
            fig = plot_event_page(event_id, results_for_event)
            pdf.savefig(fig)
            fig.savefig(PER_EVENT_PNG_DIR / f"{event_id}_waveform_stage_model_comparison.png", dpi=250, bbox_inches="tight")
            plt.close(fig)

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(OUT_CSV, index=False)

    if len(diag):
        # summary by event/model/stage/panel
        summary_event = (
            diag.groupby(["event_id", "model_key", "stage", "panel"], dropna=False)
            .agg(
                mean_cc=("cc", "mean"),
                median_cc=("cc", "median"),
                mean_abs_shift_s=("shift_s", lambda x: float(np.nanmean(np.abs(x)))),
                n_panels=("cc", "count"),
            )
            .reset_index()
            .sort_values(["event_id", "model_key", "stage", "panel"])
        )
        summary_event.to_csv(OUT_SUMMARY_EVENT_CSV, index=False)

        summary_model_stage = (
            diag.groupby(["model_key", "stage"], dropna=False)
            .agg(
                mean_cc=("cc", "mean"),
                median_cc=("cc", "median"),
                mean_abs_shift_s=("shift_s", lambda x: float(np.nanmean(np.abs(x)))),
                n_panels=("cc", "count"),
            )
            .reset_index()
            .sort_values(["model_key", "stage"])
        )
        summary_model_stage.to_csv(OUT_SUMMARY_MODEL_STAGE_CSV, index=False)

        write_notes(diag, summary_event, summary_model_stage, OUT_NOTES)
    else:
        pd.DataFrame().to_csv(OUT_SUMMARY_EVENT_CSV, index=False)
        pd.DataFrame().to_csv(OUT_SUMMARY_MODEL_STAGE_CSV, index=False)
        OUT_NOTES.write_text("No diagnostics rows were generated.\n", encoding="utf-8")

    print("\nSaved:")
    print(f"  {OUT_PDF}")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_SUMMARY_EVENT_CSV}")
    print(f"  {OUT_SUMMARY_MODEL_STAGE_CSV}")
    print(f"  {OUT_NOTES}")
    print(f"  {PER_EVENT_PNG_DIR}")


if __name__ == "__main__":
    main()
