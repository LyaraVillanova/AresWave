#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import contextlib
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict
import numpy as np
from obspy import UTCDateTime
from dsmpy.event_Mars import MomentTensor
from dsmpy.station_Mars import Station
import areswave.PSO as PSO
import null_test_fast as ntf

RUN_MODELS = "TAYAK,X"
RUN_MAX_EVENTS = None
RUN_EVENTS = "S1000a,S1094b"
RUN_QUIET = False
RUN_BATCH = 1
RUN_OUT_DIR = "depth_results"

RUN_DEPTH_MIN_KM = None
RUN_DEPTH_MAX_KM = None
RUN_DEPTH_PAD_KM = 20.0
RUN_DEPTH_STEP_KM = 1.0


EVENT_DEPTH_HDI_KM = {
    "S1000a": (10.0, 100.0),
    "S1094b": (10.0, 100.0),
}

FALLBACK_EVENT_DECLARED = {
    "S1000a": {
        "event_id": "mqs2021sddf",
        "latitude": 32.9,
        "longitude": -89.0,
        "magnitude": 4.1,
        "distance": 128.3,
        "baz": 41.4,
        "depth_hint": 0.0,
        "centroid_time": "2021-09-18T17:46:20",
        "time_p": "2021-09-18T18:01:47",
        "time_s": "2021-09-18T18:14:08",
    },
    "S1094b": {
        "event_id": "mqs2021ysph",
        "latitude": 42.6,
        "longitude": -173.9,
        "magnitude": 4.0,
        "distance": 59.7,
        "baz": 45.2,
        "depth_hint": 0.0,
        "centroid_time": "2021-12-24T22:38:02",
        "time_p": "2021-12-24T22:45:13",
        "time_s": "2021-12-24T22:50:47",
    },
}

TABLE2_DEPTH_BY_MODEL: Dict[str, Dict[str, float]] = {
    "S1000a": {
        "TAYAK": 0.0,
        "X": 0.0,
    },
    "S1094b": {
        "TAYAK": 0.0,
        "X": 0.0,
    },
}

MT_BY_MODEL: Dict[str, Dict[str, Dict[str, float]]] = {
    "S1000a": {
        "TAYAK": {
            "Mpp": 5.923e+21,
            "Mrp": 0.0,
            "Mrr": 5.923e+21,
            "Mrt": 0.0,
            "Mtp": 0.0,
            "Mtt": 5.923e+21,
        },
        "X": {
            "Mpp": 5.923e+21,
            "Mrp": 0.0,
            "Mrr": 5.923e+21,
            "Mrt": 0.0,
            "Mtp": 0.0,
            "Mtt": 5.923e+21,
        },
    },
    "S1094b": {
        "TAYAK": {
            "Mpp": 4.196e+21,
            "Mrp": 0.0,
            "Mrr": 4.196e+21,
            "Mrt": 0.0,
            "Mtp": 0.0,
            "Mtt": 4.196e+21,
        },
        "X": {
            "Mpp": 4.196e+21,
            "Mrp": 0.0,
            "Mrr": 4.196e+21,
            "Mrt": 0.0,
            "Mtp": 0.0,
            "Mtt": 4.196e+21,
        },
    },
}

def _depth_grid(
    ev_id: str,
    best_depth_km: float,
    depth_min_km: float | None,
    depth_max_km: float | None,
    depth_pad_km: float,
    depth_step_km: float,
) -> tuple[np.ndarray, float, float, str]:
    if depth_step_km <= 0:
        raise ValueError("depth_step_km must be > 0")

    if depth_min_km is not None or depth_max_km is not None:
        dmin = float(depth_min_km if depth_min_km is not None else best_depth_km - depth_pad_km)
        dmax = float(depth_max_km if depth_max_km is not None else best_depth_km + depth_pad_km)
        source = "manual_override"
    elif ev_id in EVENT_DEPTH_HDI_KM:
        dmin, dmax = map(float, EVENT_DEPTH_HDI_KM[ev_id])
        source = "event_hdi"
    else:
        dmin = float(best_depth_km) - float(depth_pad_km)
        dmax = float(best_depth_km) + float(depth_pad_km)
        source = "best_plusminus_pad"

    if dmax < dmin:
        raise ValueError(f"Invalid depth range: [{dmin}, {dmax}]")

    n = int(np.floor((dmax - dmin) / float(depth_step_km))) + 1
    depths = dmin + np.arange(n, dtype=float) * float(depth_step_km)

    if depths.size == 0 or depths[-1] < dmax - 1e-9:
        depths = np.append(depths, dmax)

    depths = np.unique(np.round(depths.astype(float), 6))
    return depths, float(dmin), float(dmax), source


@contextlib.contextmanager
def _maybe_quiet(enabled: bool):
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _moment_tensor_from_table(ev_id: str, model_key: str) -> MomentTensor:
    vals = MT_BY_MODEL.get(ev_id, {}).get(model_key)

    if vals is None:
        raise KeyError(f"No Table-2 MT found for {ev_id}/{model_key}")

    return MomentTensor(
        float(vals["Mrr"]),
        float(vals["Mrt"]),
        float(vals["Mrp"]),
        float(vals["Mtt"]),
        float(vals["Mtp"]),
        float(vals["Mpp"]),
    )


def _guess_sac_paths_for_event(sac_dir: Path, ev_id: str, decl: dict):
    """
    First tries the normal event name, e.g. S1000a.
    If that fails, tries the MQS event code, e.g. mqs2021sddf.
    This helps when SAC files are named by MQS code instead of S-event name.
    """
    sac_paths = ntf._guess_sac_paths(sac_dir, ev_id)

    if sac_paths:
        return sac_paths

    alt_id = decl.get("event_id")
    if alt_id:
        sac_paths = ntf._guess_sac_paths(sac_dir, str(alt_id))
        if sac_paths:
            print(f"[info] {ev_id}: SAC files found using alternate id {alt_id}", flush=True)
            return sac_paths

    return sac_paths


def _make_fixed_mt_cost_function(
    base_event,
    stations,
    seismic_model,
    tlen: float,
    nspc: int,
    sampling_hz: float,
    real_data_list,
    magnitude: float,
    time_p,
    time_s,
    fixed_mt: MomentTensor,
):
    real_data_list = PSO.reorder_traces(real_data_list)

    n_samples = int(len(real_data_list[0].data))
    if n_samples <= 1:
        raise ValueError("Real data window is empty or too short.")

    max_shift_samples = int(2.0 * float(sampling_hz))
    start_time = real_data_list[0].stats.starttime

    p_idx = int((time_p - start_time) * float(sampling_hz))
    s_idx = int((time_s - start_time) * float(sampling_hz))

    syn_times = np.arange(n_samples, dtype=float) / float(sampling_hz)

    real_Z = PSO.normalize(real_data_list[0].data[:n_samples])
    real_R = PSO.normalize(real_data_list[1].data[:n_samples])
    real_T = PSO.normalize(real_data_list[2].data[:n_samples])

    def cost_func(depth_values):
        costs = []

        for dpt in np.ravel(np.asarray(depth_values, dtype=float)):
            if dpt < 0.0 or dpt > 560.0:
                costs.append(1e6)
                continue

            try:
                base_event.depth = float(dpt)
                base_event.mt = fixed_mt

                output = PSO.generate_synthetics(
                    base_event,
                    stations,
                    seismic_model,
                    float(tlen),
                    int(nspc),
                    float(sampling_hz),
                )

            except Exception as e:
                print(f"Error to generate the synthetics at depth={dpt:.2f} km: {e}", flush=True)
                costs.append(1e6)
                continue

            try:
                ts = output.ts
                max_idx = int(np.searchsorted(ts, float(tlen)))

                u_Z = output["Z", "ELYSE_XB"][:max_idx]
                u_R = output["R", "ELYSE_XB"][:max_idx]
                u_T = output["T", "ELYSE_XB"][:max_idx]

                if u_Z.size == 0 or u_R.size == 0 or u_T.size == 0:
                    costs.append(1e6)
                    continue

                u_Z_f = PSO.apply_filter(u_Z, float(sampling_hz))
                u_R_f = PSO.apply_filter(u_R, float(sampling_hz))
                u_T_f = PSO.apply_filter(u_T, float(sampling_hz))

                filtered = PSO.polarization_filter(
                    [u_Z_f, u_R_f, u_T_f],
                    float(sampling_hz),
                )

                syn_Z_raw = PSO.normalize(filtered[0][:n_samples])
                syn_R_raw = PSO.normalize(filtered[1][:n_samples])
                syn_T_raw = PSO.normalize(filtered[2][:n_samples])

                syn_Z = PSO.align_by_correlation(real_Z, syn_Z_raw, max_shift_samples)
                syn_R = PSO.align_by_correlation(real_R, syn_R_raw, max_shift_samples)
                syn_T = PSO.align_by_correlation(real_T, syn_T_raw, max_shift_samples)

                var = PSO.calculate_variation(
                    real_Z,
                    syn_Z,
                    real_R,
                    syn_R,
                    real_Z,
                    syn_Z,
                    real_T,
                    syn_T,
                    syn_times,
                    float(magnitude),
                    float(sampling_hz),
                    int(p_idx),
                    int(s_idx),
                )

            except Exception as e:
                print(f"Error in waveform misfit at depth={dpt:.2f} km: {e}", flush=True)
                var = 1e6

            costs.append(float(var))

        return np.ravel(np.asarray(costs, dtype=float))

    return cost_func, p_idx, s_idx, n_samples


def run_one(
    ev_id: str,
    model_key: str,
    sac_dir: Path,
    out_dir: Path,
    quiet: bool,
    batch: int,
    depth_min_km: float | None,
    depth_max_km: float | None,
    depth_pad_km: float,
    depth_step_km: float,
):
    best = ntf.BEST_SOLUTIONS.get(ev_id, {}).get(model_key)
    best_source = "BEST_SOLUTIONS"

    if best is None:
        table_depth = TABLE2_DEPTH_BY_MODEL.get(ev_id, {}).get(model_key)

        if table_depth is None:
            print(
                f"[skip] {ev_id}/{model_key}: no BEST_SOLUTIONS entry and no Table-2 depth fallback.",
                flush=True,
            )
            return

        best = {
            "depth_km": float(table_depth),
            "cost": np.nan,
            "strike_deg": np.nan,
            "dip_deg": np.nan,
            "rake_deg": np.nan,
        }
        best_source = "Table2_depth_fallback"

    if MT_BY_MODEL.get(ev_id, {}).get(model_key) is None:
        print(f"[skip] {ev_id}/{model_key}: no MT entry from Table 2.", flush=True)
        return

    decl = ntf.EVENT_DECLARED.get(ev_id, {}) or FALLBACK_EVENT_DECLARED.get(ev_id, {})

    if not decl:
        print(f"[skip] {ev_id}: missing EVENT_DECLARED.", flush=True)
        return

    lat = decl.get("latitude")
    lon = decl.get("longitude")
    mag = decl.get("magnitude") or decl.get("Mw")
    tp_s = decl.get("time_p")
    ts_s = decl.get("time_s")
    tc_s = decl.get("centroid_time")

    if lat is None or lon is None or mag is None or not tp_s or not ts_s or not tc_s:
        print(f"[skip] {ev_id}: missing lat/lon/mag or picks in EVENT_DECLARED.", flush=True)
        return

    time_p = UTCDateTime(str(tp_s))
    time_s = UTCDateTime(str(ts_s))
    centroid_time = UTCDateTime(str(tc_s))

    sac_paths = _guess_sac_paths_for_event(sac_dir, ev_id, decl)

    if not sac_paths:
        alt_id = decl.get("event_id")
        print(
            f"[skip] {ev_id}: SAC files not found in {sac_dir}. Tried event id '{ev_id}'"
            + (f" and alternate id '{alt_id}'." if alt_id else "."),
            flush=True,
        )
        return

    real_full = ntf._read_and_prepare_traces(sac_paths, ntf.SAMPLING_HZ)

    real_trim, tlen = ntf._trim_traces_to_picks(
        real_full,
        time_p,
        time_s,
        ntf.RUN_PRE_P_SEC,
        ntf.RUN_POST_S_SEC,
        ntf.RUN_TLEN_MIN,
    )

    best_depth = float(best["depth_km"])

    depths, dmin, dmax, depth_range_source = _depth_grid(
        ev_id,
        best_depth,
        depth_min_km,
        depth_max_km,
        depth_pad_km,
        depth_step_km,
    )

    fixed_mt = _moment_tensor_from_table(ev_id, model_key)
    mt_vals = MT_BY_MODEL[ev_id][model_key]

    print(
        f"\n[start] {ev_id}/{model_key}  fixed MT from Table 2"
        f"  depth0={best_depth:.2f} km ({best_source})"
        f"  depth_range={dmin:.2f}-{dmax:.2f} km ({depth_range_source})"
        f"  ndepth={len(depths)}  tlen={tlen:.1f}s",
        flush=True,
    )

    print(
        f"[info] fixed MT = "
        f"(Mrr={mt_vals['Mrr']:.3e}, "
        f"Mtt={mt_vals['Mtt']:.3e}, "
        f"Mpp={mt_vals['Mpp']:.3e}, "
        f"Mrt={mt_vals['Mrt']:.3e}, "
        f"Mrp={mt_vals['Mrp']:.3e}, "
        f"Mtp={mt_vals['Mtp']:.3e})",
        flush=True,
    )

    event_tag = f"{ev_id}_{model_key}_fixedMT"

    event = ntf._build_event(
        float(lat),
        float(lon),
        best_depth,
        centroid_time,
        event_tag,
    )
    event.mt = fixed_mt

    stations = [
        Station(
            name="ELYSE",
            network="XB",
            latitude=4.502384,
            longitude=135.623447,
        )
    ]

    seismic_model = ntf._get_seismic_model(model_key)

    cost_func, p_idx, s_idx, n_samples = _make_fixed_mt_cost_function(
        event,
        stations,
        seismic_model,
        float(tlen),
        int(ntf.NSPC),
        float(ntf.SAMPLING_HZ),
        real_trim,
        float(mag),
        time_p,
        time_s,
        fixed_mt,
    )

    print(f"[check] {ev_id}: p_idx={p_idx} s_idx={s_idx} n={n_samples}", flush=True)

    depths_arr = np.asarray(depths, dtype=float)
    costs = np.empty(len(depths_arr), dtype=float)

    batch = max(1, int(batch))
    t0 = time.time()

    for j in range(0, len(depths_arr), batch):
        sl = slice(j, min(j + batch, len(depths_arr)))

        with _maybe_quiet(quiet):
            c = cost_func(depths_arr[sl])

        costs[sl] = np.ravel(c)

        dt = time.time() - t0
        print(
            f"[run] {ev_id}/{model_key} depths {sl.stop}/{len(depths_arr)}  elapsed={dt:.1f}s",
            flush=True,
        )

    finite = costs[np.isfinite(costs)]

    if finite.size:
        i_best = int(np.nanargmin(costs))
        z_best_scan = float(depths_arr[i_best])
        c_best_scan = float(costs[i_best])
        q05, q50, q95 = np.quantile(finite, [0.05, 0.5, 0.95])
    else:
        z_best_scan = float("nan")
        c_best_scan = float("nan")
        q05 = q50 = q95 = float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{ev_id}_{model_key}_fixedMT_depthscan_explo.npz"

    np.savez_compressed(
        out_path,
        event=ev_id,
        model=model_key,
        mt_source="Table2_event_model_explo",
        fixed_Mrr=float(mt_vals["Mrr"]),
        fixed_Mtt=float(mt_vals["Mtt"]),
        fixed_Mpp=float(mt_vals["Mpp"]),
        fixed_Mrt=float(mt_vals["Mrt"]),
        fixed_Mrp=float(mt_vals["Mrp"]),
        fixed_Mtp=float(mt_vals["Mtp"]),
        best_depth=best_depth,
        best_cost=float(best.get("cost", np.nan)),
        best_strike=float(best.get("strike_deg", np.nan)),
        best_dip=float(best.get("dip_deg", np.nan)),
        best_rake=float(best.get("rake_deg", np.nan)),
        depths=depths_arr,
        costs=costs,
        depth_min=dmin,
        depth_max=dmax,
        depth_range_source=depth_range_source,
        time_p=str(time_p),
        time_s=str(time_s),
        tlen=float(tlen),
        p_idx=int(p_idx),
        s_idx=int(s_idx),
    )

    print(
        f"[done] {ev_id}/{model_key}  "
        f"scan_best_depth={z_best_scan:.2f} km  "
        f"scan_best_cost={c_best_scan:.6g}  "
        f"median={q50:.6g}  "
        f"p05={q05:.6g}  "
        f"p95={q95:.6g}  "
        f"saved={out_path}",
        flush=True,
    )


def _out_path_for(out_dir: Path, ev_id: str, model_key: str) -> Path:
    return out_dir / f"{ev_id}_{model_key}_fixedMT_depthscan_explo.npz"


def _worker(kwargs):
    return run_one(**kwargs)


def main():
    ap = argparse.ArgumentParser(
        description="Depth scan with fixed event/model moment tensor from Table 2."
    )

    ap.add_argument(
        "--sac_dir",
        default=str(Path.cwd() / "SAC"),
        help="Directory containing SAC files",
    )
    ap.add_argument(
        "--out_dir",
        default=str(Path.cwd() / RUN_OUT_DIR),
        help="Output directory for .npz results",
    )
    ap.add_argument(
        "--events",
        default=RUN_EVENTS,
        help="Comma-separated event IDs. Default: S1000a,S1094b",
    )
    ap.add_argument(
        "--models",
        default=RUN_MODELS,
        help="Comma-separated models among TAYAK,X",
    )
    ap.add_argument(
        "--max_events",
        type=int,
        default=RUN_MAX_EVENTS,
        help="Limit number of events for debugging",
    )
    ap.add_argument(
        "--depth_min",
        type=float,
        default=RUN_DEPTH_MIN_KM,
        help="Override minimum depth in km",
    )
    ap.add_argument(
        "--depth_max",
        type=float,
        default=RUN_DEPTH_MAX_KM,
        help="Override maximum depth in km",
    )
    ap.add_argument(
        "--depth_pad",
        type=float,
        default=RUN_DEPTH_PAD_KM,
        help="Fallback pad for best±pad when no HDI or override is available",
    )
    ap.add_argument(
        "--depth_step",
        type=float,
        default=RUN_DEPTH_STEP_KM,
        help="Depth grid step in km",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=RUN_BATCH,
        help="Depths per loop chunk; compatibility only, no vectorized forward",
    )
    ap.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel event/model jobs",
    )
    ap.add_argument(
        "--skip_existing",
        dest="skip_existing",
        action="store_true",
        help="Skip existing .npz outputs",
    )
    ap.add_argument(
        "--no-skip_existing",
        dest="skip_existing",
        action="store_false",
    )
    ap.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
    )
    ap.add_argument(
        "--no-quiet",
        dest="quiet",
        action="store_false",
    )

    ap.set_defaults(quiet=RUN_QUIET, skip_existing=True)

    args = ap.parse_args()

    model_list = [m.strip().upper() for m in str(args.models).split(",") if m.strip()]
    model_list = [m for m in model_list if m in ntf.MODEL_NAME]

    if not model_list:
        raise SystemExit("No valid models selected. Use TAYAK and/or X.")

    if args.events:
        event_ids = [e.strip() for e in str(args.events).split(",") if e.strip()]
    else:
        event_ids = sorted(set(ntf.BEST_SOLUTIONS.keys()) | set(MT_BY_MODEL.keys()))

    if args.max_events is not None:
        event_ids = event_ids[: int(args.max_events)]

    sac_dir = Path(args.sac_dir)
    out_dir = Path(args.out_dir)

    print(f"[info] SAC dir: {sac_dir}", flush=True)
    print(f"[info] Output dir: {out_dir}", flush=True)
    print(f"[info] Models: {model_list}", flush=True)
    print(f"[info] Events: {event_ids}", flush=True)
    print(f"[info] n_jobs: {int(args.n_jobs)}", flush=True)
    print(f"[info] skip_existing: {bool(args.skip_existing)}", flush=True)
    print("[info] Depth scan uses fixed Table-2 MT per event/model and varies only depth.", flush=True)

    jobs = []

    for ev_id in event_ids:
        for model_key in model_list:
            out_path = _out_path_for(out_dir, ev_id, model_key)

            if bool(args.skip_existing) and out_path.exists():
                print(
                    f"[skip] {ev_id}/{model_key}: existing output found at {out_path}",
                    flush=True,
                )
                continue

            jobs.append(
                dict(
                    ev_id=ev_id,
                    model_key=model_key,
                    sac_dir=sac_dir,
                    out_dir=out_dir,
                    quiet=bool(args.quiet),
                    batch=int(args.batch),
                    depth_min_km=args.depth_min,
                    depth_max_km=args.depth_max,
                    depth_pad_km=float(args.depth_pad),
                    depth_step_km=float(args.depth_step),
                )
            )

    if not jobs:
        print("[info] Nothing to run.", flush=True)
        return

    if int(args.n_jobs) <= 1:
        for kw in jobs:
            run_one(**kw)
    else:
        with ProcessPoolExecutor(max_workers=int(args.n_jobs)) as ex:
            futs = [ex.submit(_worker, kw) for kw in jobs]

            for fut in as_completed(futs):
                fut.result()


if __name__ == "__main__":
    main()