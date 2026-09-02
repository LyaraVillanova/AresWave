#!/usr/bin/env python3
"""
Plot observed vs synthetic waveform windows for the best-fitting 17-event model,
with an explicit per-event/per-phase consistency check against the official
evaluate_single_event() result from inversion_17_models_1400.py.

This version is stricter than the earlier plotting script:
- It calls inv.evaluate_single_event(obs_st, syn_st) for each event.
- It uses the official selected group/component stored in p_group/s_group and
  p_used/s_used.
- It reconstructs the aligned window only for that official selected component.
- It writes official_misfit, plotted_misfit, and their difference to CSV.
- It prints a warning if any plotted window does not match the official inversion
  result within tolerance.

Run:
  cd /home/lyara/areswave
  /home/lyara/areswave/areswave-venv/bin/python plot_best_waveform_misfit_17_exact.py
"""

import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------
PROJECT_DIR = Path("/home/lyara/areswave")
ARES_SRC_DIR = PROJECT_DIR / "areswave"
INV_SCRIPT_PATH = ARES_SRC_DIR / "inversion_17_models_1400.py"

BEST_MODEL_ID = "Geophysical_model298"
BEST_MODEL_PATH = PROJECT_DIR / "models" / f"{BEST_MODEL_ID}.nd"

OUT_DIR = PROJECT_DIR / "figs_best_waveforms17_exact"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERVIEW_PNG = OUT_DIR / "best_waveform_misfit_overview_exact.png"
PER_EVENT_PDF = OUT_DIR / "best_waveform_misfit_by_event_exact.pdf"
DIAG_CSV = OUT_DIR / "best_waveform_misfit_diagnostics_exact.csv"


# ---------------------------------------------------------------------
# IMPORT INVERSION SCRIPT BY FILE PATH
# ---------------------------------------------------------------------
def import_inversion_script():
    if not INV_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Cannot find inversion script: {INV_SCRIPT_PATH}")

    os.chdir(PROJECT_DIR)
    for p in (PROJECT_DIR, ARES_SRC_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    print(f"[INFO] Importing inversion script from: {INV_SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("inv17models_runtime_exact", str(INV_SCRIPT_PATH))
    inv = importlib.util.module_from_spec(spec)
    sys.modules["inv17models_runtime_exact"] = inv
    spec.loader.exec_module(inv)
    return inv


inv = import_inversion_script()


# ---------------------------------------------------------------------
# EXACT RECONSTRUCTION HELPERS
# ---------------------------------------------------------------------
def _safe_norm(x):
    return inv._normalize(np.asarray(x, dtype=float))


def _extract_array(st, comp):
    # inversion_17_models_1400.py's _extract_single_component returns an array.
    arr = inv._extract_single_component(st, comp)
    if arr is None:
        return None
    return np.asarray(arr, dtype=float)


def _official_window(obs_arr, syn_arr, pick_seconds, window):
    """
    Reproduce the windowing branch in _evaluate_component_group():
      if pick exists: cut both observed and synthetic around that same pick
      else: cut observed around observed peak and synthetic around synthetic peak
    """
    if pick_seconds is not None and np.isfinite(pick_seconds):
        ow = inv._cut_window_from_pick(obs_arr, float(pick_seconds), window)
        sw = inv._cut_window_from_pick(syn_arr, float(pick_seconds), window)
    else:
        iobs = inv._pick_peak_index(obs_arr)
        isyn = inv._pick_peak_index(syn_arr)
        ow = inv._cut_around_peak(obs_arr, iobs, window)
        sw = inv._cut_around_peak(syn_arr, isyn, window)

    if ow is None or sw is None:
        return None, None

    n = min(len(ow), len(sw))
    if n < 8:
        return None, None

    return np.asarray(ow[:n], dtype=float), np.asarray(sw[:n], dtype=float)


def _parse_group(group_str):
    if group_str is None:
        return tuple()
    s = str(group_str).strip()
    if not s:
        return tuple()
    return tuple(x.strip() for x in s.split("+") if x.strip())


def _component_from_used(used_str, fallback_group):
    """
    In paper mode the official group contains one component.
    In zr_zt mode used may contain multiple components; for plotting we choose
    the first official used component but keep the official group mean in the CSV.
    """
    used = _parse_group(used_str)
    if used:
        return used[0]
    group = _parse_group(fallback_group)
    if group:
        return group[0]
    return ""


def _reconstruct_for_official_group(obs_st, syn_st, group, window, pick_seconds, official_misfit):
    """
    Reconstruct best sign and shifts for the official selected group.
    Returns a dict for one component to plot, plus group-level plotted misfit.
    """
    group = tuple(group)
    if not group:
        return None

    per_sign = []
    per_sign_items = []

    for sign in (+1.0, -1.0):
        misfits = []
        ccs = []
        items = []

        for comp in group:
            obs_arr = _extract_array(obs_st, comp)
            syn_arr = _extract_array(syn_st, comp)
            if obs_arr is None or syn_arr is None:
                continue

            ow, sw = _official_window(obs_arr, syn_arr, pick_seconds, window)
            if ow is None or sw is None:
                continue

            # Use official phase function for the numerical misfit.
            out = inv._phase_misfit_cc(ow, sign * sw)
            if out is None:
                continue
            m, cc = out

            obs_n = _safe_norm(ow)
            syn_n = _safe_norm(sign * sw)
            shift, cc2 = inv._crosscorr_best_shift(obs_n, syn_n, inv.MAX_SHIFT_SAMPLES)
            syn_aligned = inv._apply_shift(syn_n, shift)

            items.append({
                "component": comp,
                "obs": obs_n,
                "syn": syn_aligned,
                "shift_samples": int(shift),
                "shift_s": float(shift) * float(inv.DT),
                "sign": "+" if sign > 0 else "-",
                "cc": float(cc),
                "misfit": float(m),
            })
            misfits.append(float(m))
            ccs.append(float(cc))

        if misfits:
            group_misfit = float(np.mean(misfits))
            group_cc = float(np.mean(ccs))
            per_sign.append((group_misfit, group_cc, "+" if sign > 0 else "-"))
            per_sign_items.append(items)

    if not per_sign:
        return None

    # Official _evaluate_component_group picks the sign with minimum group mean.
    best_idx = int(np.argmin([x[0] for x in per_sign]))

    # If an official_misfit was provided, choose the sign that matches it most closely.
    # This protects against rare exact ties.
    if official_misfit is not None and np.isfinite(official_misfit):
        diffs = [abs(x[0] - float(official_misfit)) for x in per_sign]
        best_idx = int(np.argmin(diffs))

    group_misfit, group_cc, sign_symbol = per_sign[best_idx]
    items = per_sign_items[best_idx]

    # Choose plotted component:
    # - in paper mode this is the only component;
    # - in multi-component mode choose the component with largest observed amplitude.
    if len(items) == 1:
        item = items[0]
    else:
        item = max(items, key=lambda x: float(np.nanmax(np.abs(x["obs"]))) if x["obs"].size else -np.inf)

    item = dict(item)
    item.update({
        "group_misfit": group_misfit,
        "group_cc": group_cc,
        "group_sign": sign_symbol,
        "group": "+".join(group),
        "n_components_in_group": len(items),
    })
    return item


def _plot_one_phase(ax, event_id, phase, item, window, official_misfit, official_cc, official_group, official_used):
    if item is None:
        ax.text(0.5, 0.5, f"{event_id} {phase}: no valid official window", ha="center", va="center")
        ax.set_axis_off()
        return

    n = min(len(item["obs"]), len(item["syn"]))
    t = np.arange(n, dtype=float) * float(inv.DT) + float(window[0])

    ax.plot(t, item["obs"][:n], color="black", lw=1.3, label="Observed")
    ax.plot(t, item["syn"][:n], color="red", lw=1.1, alpha=0.85, label="Synthetic aligned")
    ax.axvline(0.0, color="0.4", lw=0.8, ls=":")

    diff = abs(float(item["group_misfit"]) - float(official_misfit)) if np.isfinite(official_misfit) else np.nan
    title = (
        f"{event_id} {phase} | official {official_group}/{official_used} | "
        f"plot comp={item['component']} | sign={item['sign']} | "
        f"shift={item['shift_s']:+.2f}s | "
        f"misfit={item['group_misfit']:.6f} | Δ={diff:.1e}"
    )
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel("Time from pick (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.25, lw=0.5)


def _phase_window(event_id, phase):
    if phase.upper() == "P":
        return inv.P_WINDOW
    return inv.S_WINDOW_EXTENDED if event_id in inv.S_WINDOW_EXTENDED_EVENTS else inv.S_WINDOW


def build_best_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Cannot find best model file: {BEST_MODEL_PATH}")

    base_model = inv.SeismicModel.test2()
    suite_model = inv.load_suite_model_from_nd(str(BEST_MODEL_PATH), base_model)
    full_model = inv.build_full_model_from_suite(base_model, suite_model)
    return suite_model, full_model


def main():
    events = inv.load_event_catalog(str(PROJECT_DIR / inv.CSV_FILE))
    events = events.dropna(subset=["event_id", "latitude", "longitude", "depth"]).reset_index(drop=True)

    suite_model, full_model = build_best_model()

    diagnostics = []
    plot_records = []

    tol = 1e-10
    mismatch_count = 0

    for _, evt in events.iterrows():
        event_id = str(evt["event_id"]).strip()

        obs_st = inv.load_observed_waveforms(event_id)
        if obs_st is None:
            print(f"[{event_id}] no observed waveforms")
            continue

        event_depth_km = float(evt.get("depth", suite_model.source_depth_km))
        syn_st = inv.synthesize_event_traces(evt, full_model, source_depth_km=event_depth_km)
        if syn_st is None:
            print(f"[{event_id}] no synthetic waveforms")
            continue

        # Official inversion result for this event.
        official = inv.evaluate_single_event(evt, obs_st, syn_st)

        p_pick = inv._event_phase_pick_seconds(evt, "p", obs_st)
        s_pick = inv._event_phase_pick_seconds(evt, "s", obs_st)

        p_window = _phase_window(event_id, "P")
        s_window = _phase_window(event_id, "S")

        p_group = _parse_group(official.get("p_group", ""))
        s_group = _parse_group(official.get("s_group", ""))

        p_item = _reconstruct_for_official_group(
            obs_st, syn_st, p_group, p_window, p_pick, official.get("p_misfit", np.nan)
        )
        s_item = _reconstruct_for_official_group(
            obs_st, syn_st, s_group, s_window, s_pick, official.get("s_misfit", np.nan)
        )

        for phase, item, window, misfit_key, cc_key, group_key, used_key in [
            ("P", p_item, p_window, "p_misfit", "p_cc", "p_group", "p_used"),
            ("S", s_item, s_window, "s_misfit", "s_cc", "s_group", "s_used"),
        ]:
            official_m = float(official.get(misfit_key, np.nan))
            official_cc = float(official.get(cc_key, np.nan))
            official_group = str(official.get(group_key, ""))
            official_used = str(official.get(used_key, ""))

            plotted_m = np.nan if item is None else float(item["group_misfit"])
            plotted_cc = np.nan if item is None else float(item["group_cc"])
            diff = np.nan
            match = False
            if np.isfinite(official_m) and np.isfinite(plotted_m):
                diff = abs(official_m - plotted_m)
                match = bool(diff <= tol)
                if not match:
                    mismatch_count += 1

            diagnostics.append({
                "event_id": event_id,
                "phase": phase,
                "official_group": official_group,
                "official_used": official_used,
                "plotted_component": "" if item is None else item["component"],
                "sign": "" if item is None else item["sign"],
                "shift_samples": np.nan if item is None else item["shift_samples"],
                "shift_s": np.nan if item is None else item["shift_s"],
                "official_cc": official_cc,
                "plotted_group_cc": plotted_cc,
                "official_misfit": official_m,
                "plotted_group_misfit": plotted_m,
                "abs_difference": diff,
                "match_official": match,
                "window_start_s": window[0],
                "window_end_s": window[1],
                "component_mode": inv.COMPONENT_MODE,
                "best_model_id": BEST_MODEL_ID,
            })

        print(
            f"[{event_id}] "
            f"P official={official.get('p_misfit', np.nan):.6f} plot={np.nan if p_item is None else p_item['group_misfit']:.6f} | "
            f"S official={official.get('s_misfit', np.nan):.6f} plot={np.nan if s_item is None else s_item['group_misfit']:.6f}"
        )

        plot_records.append((event_id, p_item, p_window, s_item, s_window, official))

    diag = pd.DataFrame(diagnostics)
    diag.to_csv(DIAG_CSV, index=False)

    if mismatch_count:
        print(f"\n[WARNING] {mismatch_count} phase windows did not match official inversion misfit within {tol:g}.")
        print(f"Check: {DIAG_CSV}")
    else:
        print(f"\n[OK] All plotted phase windows match official inversion misfits within {tol:g}.")

    # Overview.
    n_events = len(plot_records)
    fig_h = max(8, 1.65 * n_events)
    fig, axes = plt.subplots(n_events, 2, figsize=(13, fig_h), squeeze=False)

    for i, (event_id, p_item, p_window, s_item, s_window, official) in enumerate(plot_records):
        _plot_one_phase(
            axes[i, 0], event_id, "P", p_item, p_window,
            official.get("p_misfit", np.nan), official.get("p_cc", np.nan),
            official.get("p_group", ""), official.get("p_used", "")
        )
        _plot_one_phase(
            axes[i, 1], event_id, "S", s_item, s_window,
            official.get("s_misfit", np.nan), official.get("s_cc", np.nan),
            official.get("s_group", ""), official.get("s_used", "")
        )
        if i == 0:
            axes[i, 0].legend(loc="upper right", fontsize=7)

    fig.suptitle(
        f"Observed vs synthetic windows — exact official inversion selections — {BEST_MODEL_ID}",
        y=0.996,
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.992])
    fig.savefig(OVERVIEW_PNG, dpi=300)
    plt.close(fig)

    # Per-event PDF.
    with PdfPages(PER_EVENT_PDF) as pdf:
        for event_id, p_item, p_window, s_item, s_window, official in plot_records:
            fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.2), sharex=False)
            _plot_one_phase(
                axes[0], event_id, "P", p_item, p_window,
                official.get("p_misfit", np.nan), official.get("p_cc", np.nan),
                official.get("p_group", ""), official.get("p_used", "")
            )
            _plot_one_phase(
                axes[1], event_id, "S", s_item, s_window,
                official.get("s_misfit", np.nan), official.get("s_cc", np.nan),
                official.get("s_group", ""), official.get("s_used", "")
            )
            axes[0].legend(loc="upper right")
            fig.suptitle(f"{event_id} — exact official inversion selection — {BEST_MODEL_ID}", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

    # Official aggregate recomputed from exact per-event official rows.
    pmean = diag.loc[diag["phase"] == "P", "official_misfit"].mean()
    smean = diag.loc[diag["phase"] == "S", "official_misfit"].mean()
    total = np.average([pmean, smean], weights=[1.0, float(inv.S_WEIGHT)])

    print("\nSaved:")
    print(f"  {OVERVIEW_PNG}")
    print(f"  {PER_EVENT_PDF}")
    print(f"  {DIAG_CSV}")

    print(f"\nOfficial mean P misfit: {pmean:.15f}")
    print(f"Official mean S misfit: {smean:.15f}")
    print(f"Official weighted total: {total:.15f}")


if __name__ == "__main__":
    main()
