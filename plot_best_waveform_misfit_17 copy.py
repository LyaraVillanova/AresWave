#!/usr/bin/env python3
"""
Compute and plot waveform fits for the best 17-event model using an
OVERLAP-ONLY metric.

Important:
- The official inversion misfit/CC are still reported.
- An alternative "overlap-only" CC/misfit is also computed, using only the
  sample region where observed and shifted synthetic both exist.
- This usually looks visually fairer because it excludes the padded/truncated
  edges introduced by large shifts.
- HOWEVER, overlap-only misfit is NOT the same metric used in the inversion.

Run:
  cd /home/lyara/areswave
  /home/lyara/areswave/areswave-venv/bin/python plot_best_waveform_misfit_17_overlap_only_fixed_axes.py
"""

import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_DIR = Path("/home/lyara/areswave")
ARES_SRC_DIR = PROJECT_DIR / "areswave"
INV_SCRIPT_PATH = ARES_SRC_DIR / "inversion_17_models_1400.py"

BEST_MODEL_ID = "Geophysical_model298"
BEST_MODEL_PATH = PROJECT_DIR / "models" / f"{BEST_MODEL_ID}.nd"

OUT_DIR = PROJECT_DIR / "figs_best_waveforms17_overlap_only_fixed_axes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERVIEW_PNG = OUT_DIR / "best_waveform_misfit_overview_overlap_only_fixed_axes.png"
PER_EVENT_PDF = OUT_DIR / "best_waveform_misfit_by_event_overlap_only_fixed_axes.pdf"
DIAG_CSV = OUT_DIR / "best_waveform_misfit_diagnostics_overlap_only_fixed_axes.csv"


def import_inversion_script():
    if not INV_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Cannot find inversion script: {INV_SCRIPT_PATH}")

    os.chdir(PROJECT_DIR)
    for p in (PROJECT_DIR, ARES_SRC_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    print(f"[INFO] Importing inversion script from: {INV_SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("inv17models_runtime_overlap_metric", str(INV_SCRIPT_PATH))
    inv = importlib.util.module_from_spec(spec)
    sys.modules["inv17models_runtime_overlap_metric"] = inv
    spec.loader.exec_module(inv)
    return inv


inv = import_inversion_script()


def _safe_norm(x):
    return inv._normalize(np.asarray(x, dtype=float))


def _extract_array(st, comp):
    arr = inv._extract_single_component(st, comp)
    if arr is None:
        return None
    return np.asarray(arr, dtype=float)


def _official_window(obs_arr, syn_arr, pick_seconds, window):
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


def _parse_group(group_str):
    if group_str is None:
        return tuple()
    s = str(group_str).strip()
    if not s:
        return tuple()
    return tuple(x.strip() for x in s.split("+") if x.strip())


def _overlap_segments(obs_arr, syn_arr, shift_samples):
    """
    Compare only where both observed and shifted synthetic exist.

    If shifted synthetic is out = apply_shift(syn, k):
      k > 0  => out[k:] = syn[:-k]     -> compare obs[k:] with syn[:-k]
      k < 0  => out[:k] = syn[-k:]     -> compare obs[:k] with syn[-k:]
      k = 0  => compare full arrays
    """
    n = min(len(obs_arr), len(syn_arr))
    obs_arr = np.asarray(obs_arr[:n], dtype=float)
    syn_arr = np.asarray(syn_arr[:n], dtype=float)

    k = int(shift_samples)
    if k > 0:
        obs_ov = obs_arr[k:]
        syn_ov = syn_arr[:-k]
    elif k < 0:
        m = -k
        obs_ov = obs_arr[:-m]
        syn_ov = syn_arr[m:]
    else:
        obs_ov = obs_arr
        syn_ov = syn_arr

    if len(obs_ov) < 8 or len(syn_ov) < 8:
        return None, None
    return obs_ov, syn_ov


def _overlap_only_cc_misfit(obs_arr, syn_arr, shift_samples):
    """
    Alternative metric:
    - extract only overlap portion after applying the official shift
    - normalize on the overlap only
    - compute zero-lag CC and misfit = 1 - CC
    """
    obs_ov, syn_ov = _overlap_segments(obs_arr, syn_arr, shift_samples)
    if obs_ov is None or syn_ov is None:
        return np.nan, np.nan, 0

    obs_n = _safe_norm(obs_ov)
    syn_n = _safe_norm(syn_ov)

    if len(obs_n) < 8 or len(syn_n) < 8:
        return np.nan, np.nan, len(obs_n)

    # same-length guaranteed
    cc = float(np.corrcoef(obs_n, syn_n)[0, 1])
    if not np.isfinite(cc):
        return np.nan, np.nan, len(obs_n)
    cc = float(np.clip(cc, -1.0, 1.0))
    misfit = float(1.0 - cc)
    return cc, misfit, len(obs_n)


def _reconstruct_for_official_group(obs_st, syn_st, group, window, pick_seconds, official_misfit):
    """
    Reconstruct official selected sign/shift for the chosen group, then also
    compute overlap-only CC/misfit using ONLY the overlap area.
    """
    group = tuple(group)
    if not group:
        return None

    per_sign = []
    per_sign_items = []

    for sign in (+1.0, -1.0):
        misfits = []
        ccs = []
        ov_misfits = []
        ov_ccs = []
        items = []

        for comp in group:
            obs_arr = _extract_array(obs_st, comp)
            syn_arr = _extract_array(syn_st, comp)
            if obs_arr is None or syn_arr is None:
                continue

            ow, sw, window_type = _official_window(obs_arr, syn_arr, pick_seconds, window)
            if ow is None or sw is None:
                continue

            out = inv._phase_misfit_cc(ow, sign * sw)
            if out is None:
                continue
            m, cc = out

            obs_n = _safe_norm(ow)
            syn_signed_norm = _safe_norm(sign * sw)

            shift_samples, cc_shift = inv._crosscorr_best_shift(obs_n, syn_signed_norm, inv.MAX_SHIFT_SAMPLES)
            shift_s = float(shift_samples) * float(inv.DT)

            # overlap-only alternative metric
            ov_cc, ov_misfit, ov_n = _overlap_only_cc_misfit(obs_n, syn_signed_norm, shift_samples)

            items.append({
                "component": comp,
                "obs": obs_n,
                "syn_unshifted": syn_signed_norm,
                "shift_samples": int(shift_samples),
                "shift_s": shift_s,
                "sign": "+" if sign > 0 else "-",
                "official_cc_component": float(cc),
                "official_misfit_component": float(m),
                "overlap_cc_component": float(ov_cc) if np.isfinite(ov_cc) else np.nan,
                "overlap_misfit_component": float(ov_misfit) if np.isfinite(ov_misfit) else np.nan,
                "overlap_npts_component": int(ov_n),
                "window_type": window_type,
            })
            misfits.append(float(m))
            ccs.append(float(cc))
            if np.isfinite(ov_misfit):
                ov_misfits.append(float(ov_misfit))
                ov_ccs.append(float(ov_cc))

        if misfits:
            official_group_misfit = float(np.mean(misfits))
            official_group_cc = float(np.mean(ccs))
            overlap_group_misfit = float(np.mean(ov_misfits)) if ov_misfits else np.nan
            overlap_group_cc = float(np.mean(ov_ccs)) if ov_ccs else np.nan
            per_sign.append((
                official_group_misfit,
                official_group_cc,
                overlap_group_misfit,
                overlap_group_cc,
                "+" if sign > 0 else "-"
            ))
            per_sign_items.append(items)

    if not per_sign:
        return None

    # choose the sign that matches the official inversion result
    best_idx = int(np.argmin([abs(x[0] - float(official_misfit)) for x in per_sign]))

    official_group_misfit, official_group_cc, overlap_group_misfit, overlap_group_cc, sign_symbol = per_sign[best_idx]
    items = per_sign_items[best_idx]

    if len(items) == 1:
        item = items[0]
    else:
        item = max(items, key=lambda x: float(np.nanmax(np.abs(x["obs"]))) if x["obs"].size else -np.inf)

    item = dict(item)
    item.update({
        "group": "+".join(group),
        "official_group_misfit": official_group_misfit,
        "official_group_cc": official_group_cc,
        "overlap_group_misfit": overlap_group_misfit,
        "overlap_group_cc": overlap_group_cc,
        "group_sign": sign_symbol,
        "n_components_in_group": len(items),
    })
    return item


def _phase_window(event_id, phase):
    if phase.upper() == "P":
        return inv.P_WINDOW
    return inv.S_WINDOW_EXTENDED if event_id in inv.S_WINDOW_EXTENDED_EVENTS else inv.S_WINDOW


def _plot_one_phase(ax, event_id, phase, item, window, official_group, official_used):
    """
    Uniform-axis overlap-only plot.

    The overlap-only CC/misfit is still computed only on the common overlap.
    For the figure, however, every P panel keeps the full P window and every S
    panel keeps the full S window. This makes all subplots visually comparable
    and avoids variable x-axis sizes.
    """
    if item is None:
        ax.text(0.5, 0.5, f"{event_id} {phase}: no valid official window", ha="center", va="center")
        ax.set_axis_off()
        return

    n = min(len(item["obs"]), len(item["syn_unshifted"]))
    t = np.arange(n, dtype=float) * float(inv.DT) + float(window[0])
    t_shifted = t + float(item["shift_s"])

    w0 = float(window[0])
    w1 = float(window[1])
    s = float(item["shift_s"])

    # Common overlap used for the overlap-only metric.
    overlap_start = max(w0, w0 + s)
    overlap_end = min(w1, w1 + s)

    obs_mask = (t >= overlap_start) & (t <= overlap_end)
    syn_mask = (t_shifted >= overlap_start) & (t_shifted <= overlap_end)

    # If something unexpected happens, fall back to visible part in fixed window.
    if obs_mask.sum() < 8 or syn_mask.sum() < 8:
        obs_mask = (t >= w0) & (t <= w1)
        syn_mask = (t_shifted >= w0) & (t_shifted <= w1)

    # Plot only the overlap portions used by ovCC/ovMis.
    ax.plot(t[obs_mask], item["obs"][:n][obs_mask], color="black", lw=1.35, label="Observed overlap")
    ax.plot(t_shifted[syn_mask], item["syn_unshifted"][:n][syn_mask],
            color="red", lw=1.15, alpha=0.95, label="Synthetic shifted overlap")

    # Optional context: full observed and unshifted synthetic, very faint.
    ax.plot(t, item["obs"][:n], color="black", lw=0.55, alpha=0.15)
    ax.plot(t, item["syn_unshifted"][:n], color="tab:blue", lw=0.70, alpha=0.22, label="Synthetic before shift")

    ax.axvline(0.0, color="0.4", lw=0.8, ls=":")
    ax.set_xlim(w0, w1)
    ax.set_ylim(-1.08, 1.08)

    title = (
        f"{event_id} {phase} | overlap-only, fixed axes | {item['window_type']} | "
        f"official={official_group}/{official_used} | plot={item['component']} | "
        f"sign={item['sign']} | shift={item['shift_s']:+.2f}s | "
        f"ovCC={item['overlap_group_cc']:.3f} | ovMis={item['overlap_group_misfit']:.3f}"
    )
    ax.set_title(title, fontsize=7.0)
    ax.set_xlabel("Time from pick (s)" if item["window_type"] == "pick-based" else "Time from window center (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.25, lw=0.5)

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
    mismatch_count = 0
    tol = 1e-10

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

        official = inv.evaluate_single_event(evt, obs_st, syn_st)

        p_pick = inv._event_phase_pick_seconds(evt, "p", obs_st)
        s_pick = inv._event_phase_pick_seconds(evt, "s", obs_st)

        p_window = _phase_window(event_id, "P")
        s_window = _phase_window(event_id, "S")

        p_item = _reconstruct_for_official_group(
            obs_st, syn_st, _parse_group(official.get("p_group", "")),
            p_window, p_pick, official.get("p_misfit", np.nan)
        )
        s_item = _reconstruct_for_official_group(
            obs_st, syn_st, _parse_group(official.get("s_group", "")),
            s_window, s_pick, official.get("s_misfit", np.nan)
        )

        for phase, item, window, misfit_key, cc_key, group_key, used_key in [
            ("P", p_item, p_window, "p_misfit", "p_cc", "p_group", "p_used"),
            ("S", s_item, s_window, "s_misfit", "s_cc", "s_group", "s_used"),
        ]:
            official_m = float(official.get(misfit_key, np.nan))
            plotted_m = np.nan if item is None else float(item["official_group_misfit"])
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
                "official_group": str(official.get(group_key, "")),
                "official_used": str(official.get(used_key, "")),
                "plotted_component": "" if item is None else item["component"],
                "window_type": "" if item is None else item["window_type"],
                "sign": "" if item is None else item["sign"],
                "shift_samples": np.nan if item is None else item["shift_samples"],
                "shift_s": np.nan if item is None else item["shift_s"],
                "official_cc": float(official.get(cc_key, np.nan)),
                "official_group_cc_reconstructed": np.nan if item is None else float(item["official_group_cc"]),
                "official_misfit": official_m,
                "official_group_misfit_reconstructed": plotted_m,
                "official_abs_difference": diff,
                "match_official": match,
                "overlap_group_cc": np.nan if item is None else float(item["overlap_group_cc"]),
                "overlap_group_misfit": np.nan if item is None else float(item["overlap_group_misfit"]),
                "overlap_component_cc": np.nan if item is None else float(item["overlap_cc_component"]),
                "overlap_component_misfit": np.nan if item is None else float(item["overlap_misfit_component"]),
                "overlap_npts_component": np.nan if item is None else int(item["overlap_npts_component"]),
                "window_start_s": window[0],
                "window_end_s": window[1],
                "best_model_id": BEST_MODEL_ID,
                "note": "Overlap-only metric uses only the common overlap after applying the official sign+shift. It is not the same metric used by the inversion.",
            })

        print(
            f"[{event_id}] "
            f"P off={official.get('p_misfit', np.nan):.3f} ov={np.nan if p_item is None else p_item['overlap_group_misfit']:.3f} | "
            f"S off={official.get('s_misfit', np.nan):.3f} ov={np.nan if s_item is None else s_item['overlap_group_misfit']:.3f}"
        )

        plot_records.append((event_id, p_item, p_window, s_item, s_window, official))

    diag = pd.DataFrame(diagnostics)
    diag.to_csv(DIAG_CSV, index=False)

    if mismatch_count:
        print(f"\n[WARNING] {mismatch_count} official phase values did not reconstruct within {tol:g}.")
    else:
        print(f"\n[OK] All official phase values reconstructed within {tol:g}.")

    # Official and overlap summary means.
    p_off = diag.loc[diag["phase"] == "P", "official_misfit"].mean()
    s_off = diag.loc[diag["phase"] == "S", "official_misfit"].mean()
    total_off = np.average([p_off, s_off], weights=[1.0, float(inv.S_WEIGHT)])

    p_ov = diag.loc[diag["phase"] == "P", "overlap_group_misfit"].mean()
    s_ov = diag.loc[diag["phase"] == "S", "overlap_group_misfit"].mean()
    total_ov = np.average([p_ov, s_ov], weights=[1.0, float(inv.S_WEIGHT)])

    # Overview figure.
    n_events = len(plot_records)
    fig_h = max(8, 1.8 * n_events)
    fig, axes = plt.subplots(n_events, 2, figsize=(14, fig_h), squeeze=False)

    for i, (event_id, p_item, p_window, s_item, s_window, official) in enumerate(plot_records):
        _plot_one_phase(
            axes[i, 0], event_id, "P", p_item, p_window,
            official.get("p_group", ""), official.get("p_used", "")
        )
        _plot_one_phase(
            axes[i, 1], event_id, "S", s_item, s_window,
            official.get("s_group", ""), official.get("s_used", "")
        )
        if i == 0:
            axes[i, 0].legend(loc="upper right", fontsize=7)

    fig.suptitle(
        f"Observed vs synthetic windows — overlap-only CC/misfit, fixed axes — {BEST_MODEL_ID}",
        y=0.996,
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.992])
    fig.savefig(OVERVIEW_PNG, dpi=300)
    plt.close(fig)

    # Per-event PDF.
    with PdfPages(PER_EVENT_PDF) as pdf:
        for event_id, p_item, p_window, s_item, s_window, official in plot_records:
            fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), sharex=False)
            _plot_one_phase(
                axes[0], event_id, "P", p_item, p_window,
                official.get("p_group", ""), official.get("p_used", "")
            )
            _plot_one_phase(
                axes[1], event_id, "S", s_item, s_window,
                official.get("s_group", ""), official.get("s_used", "")
            )
            axes[0].legend(loc="upper right")
            fig.suptitle(f"{event_id} — overlap-only metric with fixed axes; CSV saves official and overlap metrics — {BEST_MODEL_ID}", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

    print("\nSaved:")
    print(f"  {OVERVIEW_PNG}")
    print(f"  {PER_EVENT_PDF}")
    print(f"  {DIAG_CSV}")

    print(f"\nOfficial mean P misfit: {p_off:.15f}")
    print(f"Official mean S misfit: {s_off:.15f}")
    print(f"Official weighted total: {total_off:.15f}")
    print(f"\nOverlap-only mean P misfit: {p_ov:.15f}")
    print(f"Overlap-only mean S misfit: {s_ov:.15f}")
    print(f"Overlap-only weighted total: {total_ov:.15f}")


if __name__ == "__main__":
    main()
