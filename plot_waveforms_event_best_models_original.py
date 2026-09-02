#!/usr/bin/env python3
"""
Plot observed vs synthetic waveform windows using the single-event best model
for each event.

This is based on the original/exact waveform plotting workflow:
- one row per event
- P and S columns
- black = observed
- red = aligned synthetic
- same official evaluate_single_event() consistency check
- but each event uses its own best-fitting model from best_model_summary_S*.txt.

Run:
  cd /home/lyara/areswave
  /home/lyara/areswave/areswave-venv/bin/python plot_waveforms_event_best_models_original.py

Outputs:
  figs_best_waveforms17_event_bests_original/
    event_best_waveform_overview_original.png
    event_best_waveform_by_event_original.pdf
    event_best_waveform_diagnostics_original.csv
    event_best_model_mapping.csv
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

OUT_DIR = PROJECT_DIR / "figs_best_waveforms17_event_bests_original"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERVIEW_PNG = OUT_DIR / "event_best_waveform_overview_original.png"
PER_EVENT_PDF = OUT_DIR / "event_best_waveform_by_event_original.pdf"
DIAG_CSV = OUT_DIR / "event_best_waveform_diagnostics_original.csv"
MODEL_MAP_CSV = OUT_DIR / "event_best_model_mapping.csv"

# Embedded from the uploaded best_model_summary_S*.txt files.
# The script uses source_path to load the .nd file, because for S0395a the
# uploaded summary has model_id != basename(source_path). Keeping source_path is
# the safer/faithful choice.
EVENT_BEST_MODELS = {
    "S0133a": {
        "model_id": "Geophysical_model225",
        "source_path": "/home/lyara/areswave/models/Geophysical_model225.nd",
        "summary_p_misfit": 0.0313857764891482,
        "summary_s_misfit": 0.3303838830330056,
        "summary_total_misfit": 0.23071784751838645
    },
    "S0152a": {
        "model_id": "MD_model88",
        "source_path": "/home/lyara/areswave/models/MD_model88.nd",
        "summary_p_misfit": 0.2768501526914632,
        "summary_s_misfit": 0.19776688446013568,
        "summary_total_misfit": 0.22412797387057817
    },
    "S0167a": {
        "model_id": "Geophysical_model64",
        "source_path": "/home/lyara/areswave/models/Geophysical_model64.nd",
        "summary_p_misfit": 0.34554557713580025,
        "summary_s_misfit": 0.30662822833327197,
        "summary_total_misfit": 0.31960067793411473
    },
    "S0167b": {
        "model_id": "MD_model37",
        "source_path": "/home/lyara/areswave/models/MD_model37.nd",
        "summary_p_misfit": 0.425276764455167,
        "summary_s_misfit": 0.210178367710393,
        "summary_total_misfit": 0.281877833291984
    },
    "S0185a": {
        "model_id": "MD_model78",
        "source_path": "/home/lyara/areswave/models/MD_model78.nd",
        "summary_p_misfit": 0.32800858732286886,
        "summary_s_misfit": 0.44726924298042126,
        "summary_total_misfit": 0.4075156910945705
    },
    "S0226b": {
        "model_id": "Geophysical_model776",
        "source_path": "/home/lyara/areswave/models/Geophysical_model776.nd",
        "summary_p_misfit": 0.400045766409881,
        "summary_s_misfit": 0.235009474925241,
        "summary_total_misfit": 0.290021572086788
    },
    "S0234c": {
        "model_id": "Geophysical_model979",
        "source_path": "/home/lyara/areswave/models/Geophysical_model979.nd",
        "summary_p_misfit": 0.501908068783437,
        "summary_s_misfit": 0.145749482445461,
        "summary_total_misfit": 0.264469011224786
    },
    "S0254b": {
        "model_id": "Geophysical_model277",
        "source_path": "/home/lyara/areswave/models/Geophysical_model277.nd",
        "summary_p_misfit": 0.3906100232206333,
        "summary_s_misfit": 0.31694724318358325,
        "summary_total_misfit": 0.34150150319593325
    },
    "S0345a": {
        "model_id": "CD_model68",
        "source_path": "/home/lyara/areswave/models/CD_model68.nd",
        "summary_p_misfit": 0.301898132723157,
        "summary_s_misfit": 0.258903773752344,
        "summary_total_misfit": 0.273235226742615
    },
    "S0395a": {
        "model_id": "Geophysical_model573",
        "source_path": "/home/lyara/areswave/models/CD_model44.nd",
        "summary_p_misfit": 0.331355114085345,
        "summary_s_misfit": 0.579222405953596,
        "summary_total_misfit": 0.496599975330846
    },
    "S0421b": {
        "model_id": "MD_model14",
        "source_path": "/home/lyara/areswave/models/MD_model14.nd",
        "summary_p_misfit": 0.6177392804949529,
        "summary_s_misfit": 0.18582118928482927,
        "summary_total_misfit": 0.32979388635487045
    },
    "S0976a": {
        "model_id": "Geophysical_model837",
        "source_path": "/home/lyara/areswave/models/Geophysical_model837.nd",
        "summary_p_misfit": 0.16731047658573683,
        "summary_s_misfit": 0.09366334051393232,
        "summary_total_misfit": 0.11821238587120049
    },
    "S1000a": {
        "model_id": "AK_model_62",
        "source_path": "/home/lyara/areswave/models/AK_model_62.nd",
        "summary_p_misfit": 0.2180104907018412,
        "summary_s_misfit": 0.3390404758615161,
        "summary_total_misfit": 0.2986971474749578
    },
    "S1094b": {
        "model_id": "Geophysical_model319",
        "source_path": "/home/lyara/areswave/models/Geophysical_model319.nd",
        "summary_p_misfit": 0.16182545288353245,
        "summary_s_misfit": 0.3068130704377694,
        "summary_total_misfit": 0.2584838645863571
    },
    "S1102a": {
        "model_id": "Geophysical_model58",
        "source_path": "/home/lyara/areswave/models/Geophysical_model58.nd",
        "summary_p_misfit": 0.21364046599052922,
        "summary_s_misfit": 0.437157981432166,
        "summary_total_misfit": 0.36265214295162035
    },
    "S1153a": {
        "model_id": "Geophysical_model489",
        "source_path": "/home/lyara/areswave/models/Geophysical_model489.nd",
        "summary_p_misfit": 0.5096684910973643,
        "summary_s_misfit": 0.32494329101954433,
        "summary_total_misfit": 0.386518357712151
    },
    "S1415a": {
        "model_id": "MD_model34",
        "source_path": "/home/lyara/areswave/models/MD_model34.nd",
        "summary_p_misfit": 0.433122857829756,
        "summary_s_misfit": 0.425225382357435,
        "summary_total_misfit": 0.427857874181542
    }
}


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
    spec = importlib.util.spec_from_file_location("inv17models_runtime_event_bests", str(INV_SCRIPT_PATH))
    inv = importlib.util.module_from_spec(spec)
    sys.modules["inv17models_runtime_event_bests"] = inv
    spec.loader.exec_module(inv)
    return inv


inv = import_inversion_script()


# ---------------------------------------------------------------------
# EXACT RECONSTRUCTION HELPERS
# ---------------------------------------------------------------------
def _safe_norm(x):
    return inv._normalize(np.asarray(x, dtype=float))


def _extract_array(st, comp):
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

            ow, sw, window_type = _official_window(obs_arr, syn_arr, pick_seconds, window)
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
                "window_type": window_type,
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

    best_idx = int(np.argmin([x[0] for x in per_sign]))
    if official_misfit is not None and np.isfinite(official_misfit):
        diffs = [abs(x[0] - float(official_misfit)) for x in per_sign]
        best_idx = int(np.argmin(diffs))

    group_misfit, group_cc, sign_symbol = per_sign[best_idx]
    items = per_sign_items[best_idx]

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


def _phase_window(event_id, phase):
    if phase.upper() == "P":
        return inv.P_WINDOW
    return inv.S_WINDOW_EXTENDED if event_id in inv.S_WINDOW_EXTENDED_EVENTS else inv.S_WINDOW


def _plot_one_phase(ax, event_id, phase, item, window, official_misfit, official_cc, official_group, official_used, model_label):
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
        f"{event_id} {phase} | best={model_label} | official {official_group}/{official_used} | "
        f"comp={item['component']} | sign={item['sign']} | shift={item['shift_s']:+.2f}s | "
        f"CC={item['group_cc']:.3f} | misfit={item['group_misfit']:.3f} | Δ={diff:.1e}"
    )
    ax.set_title(title, fontsize=7.1)
    ax.set_xlabel("Time from pick (s)" if item.get("window_type") == "pick-based" else "Time from window center (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_ylim(-1.08, 1.08)
    ax.grid(True, alpha=0.25, lw=0.5)


def build_model_from_nd(nd_path):
    nd_path = Path(nd_path)
    if not nd_path.exists():
        raise FileNotFoundError(f"Cannot find model .nd file: {nd_path}")

    base_model = inv.SeismicModel.test2()
    suite_model = inv.load_suite_model_from_nd(str(nd_path), base_model)
    full_model = inv.build_full_model_from_suite(base_model, suite_model)
    return suite_model, full_model


def main():
    events = inv.load_event_catalog(str(PROJECT_DIR / inv.CSV_FILE))
    events = events.dropna(subset=["event_id", "latitude", "longitude", "depth"]).reset_index(drop=True)

    # Save the model map for record.
    pd.DataFrame([
        {"event_id": event_id, **meta}
        for event_id, meta in EVENT_BEST_MODELS.items()
    ]).sort_values("event_id").to_csv(MODEL_MAP_CSV, index=False)

    model_cache = {}
    diagnostics = []
    plot_records = []

    tol = 1e-10
    mismatch_count = 0

    for _, evt in events.iterrows():
        event_id = str(evt["event_id"]).strip()

        if event_id not in EVENT_BEST_MODELS:
            print(f"[{event_id}] no event-specific best model in EVENT_BEST_MODELS; skipping")
            continue

        meta = EVENT_BEST_MODELS[event_id]
        model_id = str(meta["model_id"])
        source_path = str(meta["source_path"])
        model_label = model_id

        if source_path not in model_cache:
            suite_model, full_model = build_model_from_nd(source_path)
            model_cache[source_path] = (suite_model, full_model)
        else:
            suite_model, full_model = model_cache[source_path]

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

        p_group = _parse_group(official.get("p_group", ""))
        s_group = _parse_group(official.get("s_group", ""))

        p_item = _reconstruct_for_official_group(
            obs_st, syn_st, p_group, p_window, p_pick, official.get("p_misfit", np.nan)
        )
        s_item = _reconstruct_for_official_group(
            obs_st, syn_st, s_group, s_window, s_pick, official.get("s_misfit", np.nan)
        )

        for phase, item, window, misfit_key, cc_key, group_key, used_key, summary_key in [
            ("P", p_item, p_window, "p_misfit", "p_cc", "p_group", "p_used", "summary_p_misfit"),
            ("S", s_item, s_window, "s_misfit", "s_cc", "s_group", "s_used", "summary_s_misfit"),
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

            summary_m = meta.get(summary_key, np.nan)
            summary_diff = np.nan
            if summary_m is not None and np.isfinite(float(summary_m)) and np.isfinite(official_m):
                summary_diff = abs(float(summary_m) - official_m)

            diagnostics.append({
                "event_id": event_id,
                "phase": phase,
                "event_best_model_id": model_id,
                "event_best_source_path": source_path,
                "source_basename": Path(source_path).name,
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
                "abs_difference_plot_vs_official": diff,
                "match_official": match,
                "summary_misfit": summary_m,
                "abs_difference_summary_vs_recomputed": summary_diff,
                "summary_total_misfit": meta.get("summary_total_misfit", np.nan),
                "window_start_s": window[0],
                "window_end_s": window[1],
                "component_mode": inv.COMPONENT_MODE,
            })

        print(
            f"[{event_id} | {model_id}] "
            f"P official={official.get('p_misfit', np.nan):.6f} summary={meta.get('summary_p_misfit', np.nan):.6f} | "
            f"S official={official.get('s_misfit', np.nan):.6f} summary={meta.get('summary_s_misfit', np.nan):.6f}"
        )

        plot_records.append((event_id, p_item, p_window, s_item, s_window, official, model_label, meta))

    diag = pd.DataFrame(diagnostics)
    diag.to_csv(DIAG_CSV, index=False)

    if mismatch_count:
        print(f"\n[WARNING] {mismatch_count} phase windows did not match recomputed official misfit within {tol:g}.")
        print(f"Check: {DIAG_CSV}")
    else:
        print(f"\n[OK] All plotted phase windows match recomputed official misfits within {tol:g}.")

    # Check against uploaded single-event summaries.
    max_summary_diff = diag["abs_difference_summary_vs_recomputed"].max(skipna=True)
    print(f"[INFO] Max |summary misfit - recomputed misfit| = {max_summary_diff:.3e}")

    # Overview.
    n_events = len(plot_records)
    fig_h = max(8, 1.70 * n_events)
    fig, axes = plt.subplots(n_events, 2, figsize=(14, fig_h), squeeze=False)

    for i, (event_id, p_item, p_window, s_item, s_window, official, model_label, meta) in enumerate(plot_records):
        _plot_one_phase(
            axes[i, 0], event_id, "P", p_item, p_window,
            official.get("p_misfit", np.nan), official.get("p_cc", np.nan),
            official.get("p_group", ""), official.get("p_used", ""), model_label
        )
        _plot_one_phase(
            axes[i, 1], event_id, "S", s_item, s_window,
            official.get("s_misfit", np.nan), official.get("s_cc", np.nan),
            official.get("s_group", ""), official.get("s_used", ""), model_label
        )
        if i == 0:
            axes[i, 0].legend(loc="upper right", fontsize=7)

    fig.suptitle(
        "Observed vs synthetic windows — event-specific best model for each event",
        y=0.996,
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.992])
    fig.savefig(OVERVIEW_PNG, dpi=300)
    plt.close(fig)

    # Per-event PDF.
    with PdfPages(PER_EVENT_PDF) as pdf:
        for event_id, p_item, p_window, s_item, s_window, official, model_label, meta in plot_records:
            fig, axes = plt.subplots(2, 1, figsize=(12, 7.4), sharex=False)
            _plot_one_phase(
                axes[0], event_id, "P", p_item, p_window,
                official.get("p_misfit", np.nan), official.get("p_cc", np.nan),
                official.get("p_group", ""), official.get("p_used", ""), model_label
            )
            _plot_one_phase(
                axes[1], event_id, "S", s_item, s_window,
                official.get("s_misfit", np.nan), official.get("s_cc", np.nan),
                official.get("s_group", ""), official.get("s_used", ""), model_label
            )
            axes[0].legend(loc="upper right")
            fig.suptitle(f"{event_id} — event-specific best model: {model_label}", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

    # Aggregate just as an informative summary over the event-specific bests.
    pmean = diag.loc[diag["phase"] == "P", "official_misfit"].mean()
    smean = diag.loc[diag["phase"] == "S", "official_misfit"].mean()
    total = np.average([pmean, smean], weights=[1.0, float(inv.S_WEIGHT)])

    print("\nSaved:")
    print(f"  {OVERVIEW_PNG}")
    print(f"  {PER_EVENT_PDF}")
    print(f"  {DIAG_CSV}")
    print(f"  {MODEL_MAP_CSV}")

    print(f"\nMean P misfit across event-specific bests: {pmean:.15f}")
    print(f"Mean S misfit across event-specific bests: {smean:.15f}")
    print(f"Weighted total across event-specific bests: {total:.15f}")


if __name__ == "__main__":
    main()
