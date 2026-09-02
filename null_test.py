#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Null test = benchmark for PSO, using EXACTLY the same objective as areswave.PSO.create_cost_function,
with NO monkeypatching and NO DSM "basis cache" tricks.

What this script does:
- For each (event, model) in BEST_SOLUTIONS:
    - fixes depth = best-fit depth
    - draws N random (strike, dip, rake) inside the SAME bounds used in your PSO scripts
      (strike [0,360], dip [0,90], rake [-180,180])
    - evaluates the areswave.PSO cost function on those draws (batched)
    - writes results to: null_results/<event>_<model>_null.npz
This is a drop-in standalone script; it does NOT modify any library code.
"""

import argparse
import contextlib
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from obspy import read, Stream, UTCDateTime
from dsmpy import seismicmodel_Mars
from dsmpy.event_Mars import Event, MomentTensor
from dsmpy.station_Mars import Station
import areswave.PSO as PSO

# =============================================================================
# VS CODE RUN DEFAULTS
# =============================================================================
RUN_N = 500                      # number of random DC draws per (event, model)
RUN_BATCH = 10                   # batch size for cost eval
RUN_MODELS = "TAYAK,X"           # "TAYAK" or "X" or "TAYAK,X"
RUN_MAX_EVENTS = None            # set to 1 while debugging; None = all
RUN_EVENTS = None                # e.g., "S0133a,S0152a" or None
RUN_QUIET = True                 # suppress noisy per-particle prints inside PSO objective during sampling

# Windowing so p_idx/s_idx are ALWAYS inside the trace length:
RUN_PRE_P_SEC = 5.0              # seconds before P pick (match PSO: P-5s)
RUN_POST_S_SEC = 10.0            # seconds after S pick (match PSO: S+10s)
RUN_TLEN_MIN = 0.0               # no minimum; window is defined by (P-5s) to (S+10s)

# DSM numeric settings
SAMPLING_HZ = float(20.0)
NSPC = int(256)

# Same parameter bounds used in your PSO scripts (see Mars_PSO_*.py):
STRIKE_RANGE = (0.0, 360.0)
DIP_RANGE    = (0.0, 90.0)
RAKE_RANGE   = (-180.0, 180.0)

# Seismic model mapping (matches your original null_test.py)
MODEL_NAME = {"TAYAK": "tayak", "X": "x2"}

# -----------------------------------------------------------------------------
# Metadata copied from your original null_test.py (Table1/Table2 + best results)
# -----------------------------------------------------------------------------
BEST_SOLUTIONS = {'S0133a': {'TAYAK': {'depth_km': 24.7, 'strike_deg': 301.75, 'dip_deg': 34.76, 'rake_deg': 55.95, 'cost': 0.0407},
            'X': {'depth_km': 31.7240944880194,
                  'strike_deg': 247.606378592696,
                  'dip_deg': 62.0849739467053,
                  'rake_deg': -123.786030757008,
                  'cost': 0.0434459238743402}},
 'S0152a': {'TAYAK': {'depth_km': 26.0424936590257,
                      'strike_deg': 299.01838898959,
                      'dip_deg': 45.0821614385013,
                      'rake_deg': -47.8244162220692,
                      'cost': 0.151439},
            'X': {'depth_km': 20.74781937,
                  'strike_deg': 243.05519429,
                  'dip_deg': 21.62365895,
                  'rake_deg': -24.59139415,
                  'cost': 0.1376784707508}},
 'S0167a': {'TAYAK': {'depth_km': 16.73695269,
                      'strike_deg': 80.58265868,
                      'dip_deg': 15.13783198,
                      'rake_deg': 64.98420735,
                      'cost': 0.0938},
            'X': {'depth_km': 33.95206837,
                  'strike_deg': 237.15757689,
                  'dip_deg': 19.32843642,
                  'rake_deg': 177.72405249,
                  'cost': 0.0935}},
 'S0167b': {'TAYAK': {'depth_km': 22.9050139494313,
                      'strike_deg': 119.90764279225,
                      'dip_deg': 21.4856200009075,
                      'rake_deg': 0.561129871287224,
                      'cost': 0.0623341030263063},
            'X': {'depth_km': 16.8389933,
                  'strike_deg': 231.2073213,
                  'dip_deg': 76.62591412,
                  'rake_deg': -5.061598633,
                  'cost': 0.048336382}},
 'S0185a': {'TAYAK': {'depth_km': 19.4104880917327,
                      'strike_deg': 185.953056641419,
                      'dip_deg': 30.6455190607108,
                      'rake_deg': 124.274160342353,
                      'cost': 0.0824626130746303},
            'X': {'depth_km': 30.372131900686,
                  'strike_deg': 305.893166685142,
                  'dip_deg': 80.43828945419,
                  'rake_deg': 16.272557204186,
                  'cost': 0.0650162333410065}},
 'S0226b': {'TAYAK': {'depth_km': 11.4815412094259,
                      'strike_deg': 125.044593431172,
                      'dip_deg': 78.0483048206085,
                      'rake_deg': -36.4889458021609,
                      'cost': 0.120938638217726},
            'X': None},
 'S0234c': {'TAYAK': {'depth_km': 30.3556007742448,
                      'strike_deg': 179.333942284437,
                      'dip_deg': 61.1961901367909,
                      'rake_deg': -68.5496754301641,
                      'cost': 0.0601909653680253},
            'X': {'depth_km': 38.4560065621096,
                  'strike_deg': 40.7302600036638,
                  'dip_deg': 52.4571791790738,
                  'rake_deg': 0.58165177285858,
                  'cost': 0.0545832762373182}},
 'S0254b': {'TAYAK': {'depth_km': 11.69044012,
                      'strike_deg': 198.27446976,
                      'dip_deg': 33.26500327,
                      'rake_deg': -14.82131059,
                      'cost': 0.101158166983786},
            'X': {'depth_km': 29.11642832,
                  'strike_deg': 290.46939977,
                  'dip_deg': 52.88151013,
                  'rake_deg': 121.7423355,
                  'cost': 0.101165}},
 'S0345a': {'TAYAK': {'depth_km': 13.45095944,
                      'strike_deg': 179.94249393,
                      'dip_deg': 64.92672214,
                      'rake_deg': 172.06410958,
                      'cost': 0.0672974629107085},
            'X': {'depth_km': 14.82499424,
                  'strike_deg': 48.05022326,
                  'dip_deg': 62.12071928,
                  'rake_deg': 9.68990321,
                  'cost': 0.0669}},
 'S0395a': {'TAYAK': {'depth_km': 12.87317637,
                      'strike_deg': 51.31234728,
                      'dip_deg': 56.21567772,
                      'rake_deg': 37.67207166,
                      'cost': 0.0336},
            'X': {'depth_km': 14.9520744,
                  'strike_deg': 99.87598525,
                  'dip_deg': 37.60132772,
                  'rake_deg': -161.71517831,
                  'cost': 0.0300404407698696}},
 'S0421b': {'TAYAK': {'depth_km': 31.29616013,
                      'strike_deg': 165.56353496,
                      'dip_deg': 61.96787503,
                      'rake_deg': 0.29939255,
                      'cost': 0.0704},
            'X': {'depth_km': 30.27869612,
                  'strike_deg': 264.45311963,
                  'dip_deg': 11.64512236,
                  'rake_deg': -26.02294645,
                  'cost': 0.0738}},
 #'S0976a': {'TAYAK': None, 'X': None},
 'S1102a': {'TAYAK': {'depth_km': 22.4374561275734,
                      'strike_deg': 140.120878966849,
                      'dip_deg': 25.9506083541553,
                      'rake_deg': -149.638787386462,
                      'cost': 0.026472468690851},
            'X': {'depth_km': 32.1479306911118,
                  'strike_deg': 218.194419812979,
                  'dip_deg': 9.0269378118486,
                  'rake_deg': 10.8170851615928,
                  'cost': 0.026378802188692}},
 'S1153a': {'TAYAK': {'depth_km': 29.1531448123843,
                      'strike_deg': 243.959975587441,
                      'dip_deg': 26.9990204598176,
                      'rake_deg': -138.1429532,
                      'cost': 0.186525201911062},
            'X': {'depth_km': 45.6616710580597,
                  'strike_deg': 48.7155999466433,
                  'dip_deg': 57.0531923987105,
                  'rake_deg': 157.231303521435,
                  'cost': 0.189405451929929}},
 'S1415a': {'TAYAK': {'depth_km': 52.0605135506289,
                      'strike_deg': 252.674762423985,
                      'dip_deg': 73.8295432943338,
                      'rake_deg': 161.455553333874,
                      'cost': 0.156062417518557},
            'X': {'depth_km': 59.8527112195147,
                  'strike_deg': 256.429298029711,
                  'dip_deg': 47.5775711621902,
                  'rake_deg': 72.2103431866216,
                  'cost': 0.14691652741483}}}

EVENT_META = {'S0133a': {'Mw': 3.1, 'distance_deg': 88.5},
 'S0152a': {'Mw': 2.9, 'distance_deg': 98.0},
 'S0167a': {'Mw': 3.8, 'distance_deg': 95.3},
 'S0167b': {'Mw': 3.0, 'distance_deg': 60.3},
 'S0185a': {'Mw': 3.1, 'distance_deg': 59.8},
 'S0226b': {'Mw': 3.2, 'distance_deg': 94.1},
 'S0234c': {'Mw': 2.9, 'distance_deg': 60.0},
 'S0254b': {'Mw': 2.7, 'distance_deg': 89.5},
 'S0345a': {'Mw': 3.2, 'distance_deg': 91.5},
 'S0395a': {'Mw': 3.1, 'distance_deg': 90.6},
 'S0421b': {'Mw': 3.3, 'distance_deg': 93.7},
 #'S0976a': {'Mw': 4.3, 'distance_deg': 146.3},
 'S1102a': {'Mw': 3.2, 'distance_deg': 73.3},
 'S1153a': {'Mw': 3.0, 'distance_deg': 84.8},
 'S1415a': {'Mw': 3.3, 'distance_deg': 88.2}}

EVENT_DECLARED = {'S0133a': {'event_id': 'mqs2019hdxw',
            'latitude': -34.0,
            'longitude': -140.7,
            'magnitude': 3.1,
            'distance': 88.5,
            'baz': 101.0,
            'depth_hint': None,
            'centroid_time': '2019-04-12T18:10:36',
            'time_p': '2019-04-12T18:12:25',
            'time_s': '2019-04-12T18:18:05'},
 'S0152a': {'event_id': 'mqs2019inqk',
            'latitude': -43.7,
            'longitude': -112.6,
            'magnitude': 2.9,
            'distance': 98.0,
            'baz': 101.0,
            'depth_hint': None,
            'centroid_time': '2019-05-02T07:19:49',
            'time_p': '2019-05-02T07:29:25',
            'time_s': '2019-05-02T07:34:39'},
 'S0167a': {'event_id': 'mqs2019jptm',
            'latitude': -13.0,
            'longitude': -143.7,
            'magnitude': 3.8,
            'distance': 95.3,
            'baz': 101.0,
            'depth_hint': None,
            'centroid_time': '2019-05-17T16:43:26',
            'time_p': '2019-05-17T16:46:43',
            'time_s': '2019-05-17T16:52:08'},
 'S0167b': {'event_id': 'mqs2019jpyu',
            'latitude': -23.5,
            'longitude': -136.3,
            'magnitude': 3.0,
            'distance': 60.3,
            'baz': 107.0,
            'depth_hint': None,
            'centroid_time': '2019-05-17T19:27:04',
            'time_p': '2019-05-17T19:31:38',
            'time_s': '2019-05-17T19:37:29'},
 'S0185a': {'event_id': 'mqs2019kxjd',
            'latitude': 41.59816,
            'longitude': 90.13083,
            'magnitude': 3.1,
            'distance': 59.8,
            'baz': 322.7,
            'depth_hint': 24.1,
            'centroid_time': '2019-06-05T02:06:37',
            'time_p': '2019-06-05T02:13:48',
            'time_s': '2019-06-05T02:19:47'},
 'S0226b': {'event_id': 'mqs2019nwjf',
            'latitude': -38.1,
            'longitude': -122.9,
            'magnitude': 3.2,
            'distance': 94.1,
            'baz': 101.0,
            'depth_hint': None,
            'centroid_time': '2019-07-17T05:40:01',
            'time_p': '2019-07-17T05:44:00',
            'time_s': '2019-07-17T05:49:17'},
 'S0234c': {'event_id': 'mqs2019olnq',
            'latitude': 17.7,
            'longitude': -165.2,
            'magnitude': 2.9,
            'distance': 60.0,
            'baz': 152.8,
            'depth_hint': None,
            'centroid_time': '2019-07-25T12:51:18',
            'time_p': '2019-07-25T12:54:01',
            'time_s': '2019-07-25T12:59:59'},
 'S0254b': {'event_id': 'mqs2019pxdv',
            'latitude': -42.3,
            'longitude': -114.1,
            'magnitude': 2.7,
            'distance': 89.5,
            'baz': None,
            'depth_hint': None,
            'centroid_time': '2019-08-15T03:06:01',
            'time_p': '2019-08-15T03:02:12',
            'time_s': '2019-08-15T03:06:34'},
 'S0345a': {'event_id': 'mqs2019wltj',
            'latitude': -66.0,
            'longitude': 57.8,
            'magnitude': 3.2,
            'distance': 91.5,
            'baz': 179.0,
            'depth_hint': None,
            'centroid_time': '2019-11-16T12:03:41',
            'time_p': '2019-11-16T12:07:01',
            'time_s': '2019-11-16T12:10:40'},
 'S0395a': {'event_id': 'mqs2020akwb',
            'latitude': -50.7,
            'longitude': -124.4,
            'magnitude': 3.1,
            'distance': 90.6,
            'baz': 34.2,
            'depth_hint': None,
            'centroid_time': '2020-01-06T22:28:45',
            'time_p': '2020-01-06T22:32:46',
            'time_s': '2020-01-06T22:36:07'},
 'S0421b': {'event_id': 'mqs2020chtn',
            'latitude': -7.4,
            'longitude': -142.2,
            'magnitude': 3.3,
            'distance': 93.7,
            'baz': 34.2,
            'depth_hint': None,
            'centroid_time': '2020-02-02T16:53:32',
            'time_p': '2020-02-02T16:55:38',
            'time_s': '2020-02-02T17:01:12'},
# 'S0976a': {'event_id': 'mqs2021qpls',
#            'latitude': -5.96400035,
#            'longitude': -78.23190891,
#            'magnitude': 4.3,
#            'distance': 146.3,
#            'baz': 97.6,
#            'depth_hint': None,
#            'centroid_time': '2021-08-25T03:32:20',
#            'time_p': '2021-08-25T03:48:20',
#            'time_s': '2021-08-25T04:00:47'},
 'S1102a': {'event_id': 'mqs2022aceh',
            'latitude': -20.0,
            'longitude': 65.2,
            'magnitude': 3.2,
            'distance': 73.3,
            'baz': 267.0,
            'depth_hint': None,
            'centroid_time': '2022-01-02T04:27:10',
            'time_p': '2022-01-02T04:35:32',
            'time_s': '2022-01-02T04:42:09'},
 'S1153a': {'event_id': 'mqs2022dulj',
            'latitude': -64.7,
            'longitude': -142.2,
            'magnitude': 3.0,
            'distance': 84.8,
            'baz': 87.2,
            'depth_hint': None,
            'centroid_time': '2022-02-23T21:00:32',
            'time_p': '2022-02-23T21:09:50',
            'time_s': '2022-02-23T21:17:40'},
 'S1415a': {'event_id': 'mqs2022wrzi',
            'latitude': -56.3,
            'longitude': -134.8,
            'magnitude': 3.3,
            'distance': 88.2,
            'baz': 115.4,
            'depth_hint': None,
            'centroid_time': '2022-11-19T21:53:34',
            'time_p': '2022-11-19T21:56:03',
            'time_s': '2022-11-19T22:03:29'}}


def _get_seismic_model(model_key: str):
    """Robust constructor across dsmpy versions."""
    name = MODEL_NAME[model_key]

    # 1) Newer helper
    if hasattr(seismicmodel_Mars.SeismicModel, "model_from_name"):
        try:
            return seismicmodel_Mars.SeismicModel.model_from_name(name)
        except Exception:
            pass

    # 2) Direct named constructors (common in dsmpy forks)
    for cand in [name, name.lower(), name.upper()]:
        if hasattr(seismicmodel_Mars.SeismicModel, cand):
            fn = getattr(seismicmodel_Mars.SeismicModel, cand)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass

    # 3) Fallback (older examples)
    if hasattr(seismicmodel_Mars.SeismicModel, "test2"):
        print(f"[warn] Falling back to SeismicModel.test2() for model '{name}'. Check dsmpy model constructors.", flush=True)
        return seismicmodel_Mars.SeismicModel.test2()

    raise RuntimeError(f"Cannot construct seismic model '{name}' (key={model_key})")


def _guess_sac_paths(sac_dir: Path, event_id: str) -> Dict[str, Path]:
    """Find Z/R/T (or Z/N/E) SAC files for an event under sac_dir."""
    if not sac_dir.exists():
        raise FileNotFoundError(f"SAC dir not found: {sac_dir}")

    z_pats = [
        f"*{event_id}*BHZ*.sac*",
        f"*{event_id}*BHZ*",
        f"*{event_id}*HHZ*.sac*",
        f"*{event_id}*HHZ*",
        f"*{event_id}*Z*.sac*",
        f"*{event_id}*Z*",
    ]
    z_cands = []
    for pat in z_pats:
        z_cands.extend(sac_dir.glob(pat))
    z_cands = [p for p in z_cands if p.is_file()]
    if not z_cands:
        return {}

    z_path = sorted(z_cands, key=lambda p: (len(p.name), p.name))[0]
    name = z_path.name

    def swap_chan(nm: str, old: str, new: str) -> str:
        return nm.replace(old, new) if old in nm else nm

    pairs = []
    if "BHZ" in name:
        pairs.append((swap_chan(name, "BHZ", "BHR"), swap_chan(name, "BHZ", "BHT")))
        pairs.append((swap_chan(name, "BHZ", "BHN"), swap_chan(name, "BHZ", "BHE")))
    if "HHZ" in name:
        pairs.append((swap_chan(name, "HHZ", "HHR"), swap_chan(name, "HHZ", "HHT")))
        pairs.append((swap_chan(name, "HHZ", "HHN"), swap_chan(name, "HHZ", "HHE")))
    if "Z" in name and "BHZ" not in name and "HHZ" not in name:
        pairs.append((swap_chan(name, "Z", "R"), swap_chan(name, "Z", "T")))
        pairs.append((swap_chan(name, "Z", "N"), swap_chan(name, "Z", "E")))

    fallback = [("BHR", "BHT"), ("BHN", "BHE"), ("HHR", "HHT"), ("HHN", "HHE"), ("R", "T"), ("N", "E")]
    for ch1, ch2 in fallback:
        r = sorted(sac_dir.glob(f"*{event_id}*{ch1}*"))
        t = sorted(sac_dir.glob(f"*{event_id}*{ch2}*"))
        if r and t:
            pairs.append((r[0].name, t[0].name))

    r_path = t_path = None
    for r_name, t_name in pairs:
        rp = sac_dir / r_name
        tp = sac_dir / t_name
        if rp.exists() and tp.exists():
            r_path, t_path = rp, tp
            break

    if r_path is None or t_path is None:
        return {}

    return {"Z": z_path, "R": r_path, "T": t_path}


def _read_and_prepare_traces(sac_paths: Dict[str, Path], sampling_hz: float) -> List:
    """Return [Z, R, T] ObsPy Traces, detrended/tapered/resampled (same as your PSO scripts)."""
    stream = Stream()
    real_data_dict: Dict[str, object] = {}
    for comp in ["R", "T", "Z"]:
        tr = read(str(sac_paths[comp]))[0]
        tr.detrend("linear")
        tr.taper(max_percentage=0.05)
        tr.resample(sampling_hz)
        stream += tr
        real_data_dict[comp] = tr
    return [real_data_dict["Z"], real_data_dict["R"], real_data_dict["T"]]


def _trim_traces_to_picks(real_data_list: List, time_p: UTCDateTime, time_s: UTCDateTime,
                          pre_p: float, post_s: float, tlen_min: float) -> Tuple[List, float]:
    """Trim real traces to the same effective window used by the PSO misfit.

    PSO's misfit windows (see calculate_variation) are:
      - P window: (P-5s, P+5s)
      - S window: (S-5s, S+10s)

    To keep the *same behavior* in preprocessing steps that act on the full trace
    (corr-align, polarization filter, normalization), we use a single contiguous
    window that contains both: (P-pre_p) .. (S+post_s).
    """
    # Window is defined by picks (no forced minimum length here)
    win_start = time_p - float(pre_p)
    win_end = time_s + float(post_s)

    out = []
    for tr in real_data_list:
        tr2 = tr.copy()
        # Pad with zeros if the requested window exceeds available data
        tr2.trim(win_start, win_end, pad=True, fill_value=0.0)
        out.append(tr2)

    # Compute tlen from the trimmed sample count so that synthetic arrays are >= real arrays
    # DSMpy convention in our pipeline is: n = tlen*fs + 1  -> tlen = (n-1)/fs
    n = int(len(out[0].data))
    tlen = float(max(0.0, n - 1) / SAMPLING_HZ)
    return out, tlen


def _sample_random_dc(n: int, rng: np.random.Generator) -> np.ndarray:
    strike = rng.uniform(STRIKE_RANGE[0], STRIKE_RANGE[1], size=n)
    dip = rng.uniform(DIP_RANGE[0], DIP_RANGE[1], size=n)
    rake = rng.uniform(RAKE_RANGE[0], RAKE_RANGE[1], size=n)
    return np.column_stack([strike, dip, rake])


def _build_event(lat: float, lon: float, depth_km: float, centroid_time: UTCDateTime, ev_id: str) -> Event:
    mt0 = MomentTensor(0, 0, 0, 0, 0, 0)
    return Event(
        event_id=ev_id,
        latitude=float(lat),
        longitude=float(lon),
        depth=float(depth_km),
        mt=mt0,
        centroid_time=centroid_time,
        source_time_function=None,
    )


@contextlib.contextmanager
def _maybe_quiet(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def run_one(ev_id: str, model_key: str, sac_dir: Path, out_dir: Path, rng: np.random.Generator,
            N: int, batch: int, quiet: bool):
    best = BEST_SOLUTIONS.get(ev_id, {}).get(model_key)
    if best is None:
        print(f"[skip] {ev_id}/{model_key}: no BEST_SOLUTIONS entry.", flush=True)
        return

    decl = EVENT_DECLARED.get(ev_id, {})
    if not decl:
        print(f"[skip] {ev_id}: missing EVENT_DECLARED.", flush=True)
        return

    lat = decl.get("latitude")
    lon = decl.get("longitude")
    mag = decl.get("magnitude") or decl.get("Mw")
    dist = decl.get("distance") or decl.get("distance_deg")
    tp_s = decl.get("time_p")
    ts_s = decl.get("time_s")
    tc_s = decl.get("centroid_time")

    if lat is None or lon is None or mag is None or dist is None or not tp_s or not ts_s or not tc_s:
        print(f"[skip] {ev_id}: missing lat/lon/mag/dist or picks in EVENT_DECLARED.", flush=True)
        return

    time_p = UTCDateTime(str(tp_s))
    time_s = UTCDateTime(str(ts_s))
    centroid_time = UTCDateTime(str(tc_s))

    sac_paths = _guess_sac_paths(sac_dir, ev_id)
    if not sac_paths:
        print(f"[skip] {ev_id}: SAC files not found in {sac_dir}", flush=True)
        return

    real_full = _read_and_prepare_traces(sac_paths, SAMPLING_HZ)
    real_trim, tlen = _trim_traces_to_picks(real_full, time_p, time_s, RUN_PRE_P_SEC, RUN_POST_S_SEC, RUN_TLEN_MIN)

    p_idx = int((time_p - real_trim[0].stats.starttime) * SAMPLING_HZ)
    s_idx = int((time_s - real_trim[0].stats.starttime) * SAMPLING_HZ)
    n = int(len(real_trim[0].data))
    print(f"\n[start] {ev_id}/{model_key}  N={N}  depth={best['depth_km']:.2f}  tlen={tlen:.1f}s", flush=True)
    print(f"[check] {ev_id}: p_idx={p_idx} s_idx={s_idx} n={n}", flush=True)

    event_tag = f"{ev_id}_{model_key}"
    event = _build_event(float(lat), float(lon), float(best["depth_km"]), centroid_time, event_tag)
    stations = [Station(name="ELYSE", network="XB", latitude=4.502384, longitude=135.623447)]
    seismic_model = _get_seismic_model(model_key)

    cost_func, _ = PSO.create_cost_function(
        event, stations, seismic_model, float(tlen), int(NSPC), float(SAMPLING_HZ),
        real_trim, float(mag), float(dist), time_p, time_s
    )

    # Warmup (same best-fit SDR used by PSO). This first evaluation can be slow (DSM cache build).
    warm = np.array([[best["depth_km"], best["strike_deg"], best["dip_deg"], best["rake_deg"]]], dtype=float)
    print("[warmup] One evaluation to build/load DSM cache (can be slow the first time for this event/model/depth)...", flush=True)
    tw = time.time()
    with _maybe_quiet(quiet):
        _ = cost_func(warm)
    print(f"[warmup] Done in {time.time() - tw:.1f}s", flush=True)

    # Null sampling
    sdr = _sample_random_dc(int(N), rng)
    X = np.column_stack([np.full(int(N), float(best["depth_km"])), sdr])  # (N,4)

    costs = np.empty(int(N), dtype=float)
    t0 = time.time()
    for j in range(0, int(N), int(batch)):
        sl = slice(j, min(j + int(batch), int(N)))
        with _maybe_quiet(quiet):
            c = cost_func(X[sl])
        costs[sl] = np.ravel(c)
        dt = time.time() - t0
        print(f"[run] {ev_id}/{model_key} draws {sl.stop}/{N}  elapsed={dt:.1f}s", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ev_id}_{model_key}_null.npz"
    np.savez_compressed(
        out_path,
        event=ev_id,
        model=model_key,
        depth=float(best["depth_km"]),
        X=X,
        costs=costs,
        time_p=str(time_p),
        time_s=str(time_s),
        sampling_hz=float(SAMPLING_HZ),
        tlen=float(tlen),
    )

    finite = costs[np.isfinite(costs)]
    if finite.size:
        q05, q50, q95 = np.quantile(finite, [0.05, 0.5, 0.95])
    else:
        q05 = q50 = q95 = float("nan")
    print(f"[done] {ev_id}/{model_key}  mean={np.nanmean(costs):.4g}  median={q50:.4g}  p05={q05:.4g}  p95={q95:.4g}", flush=True)
    print(f"[saved] {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sac_dir", default=str(Path.cwd() / "SAC"))
    ap.add_argument("--out_dir", default=str(Path.cwd() / "null_results"))
    ap.add_argument("--events", default=RUN_EVENTS)
    ap.add_argument("--models", default=RUN_MODELS)
    ap.add_argument("--max_events", type=int, default=RUN_MAX_EVENTS)
    ap.add_argument("--N", type=int, default=RUN_N)
    ap.add_argument("--batch", type=int, default=RUN_BATCH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", dest="quiet", action="store_true")
    ap.add_argument("--no-quiet", dest="quiet", action="store_false")
    ap.set_defaults(quiet=RUN_QUIET)
    args = ap.parse_args()

    sac_dir = Path(args.sac_dir)
    out_dir = Path(args.out_dir)

    event_ids = sorted(BEST_SOLUTIONS.keys())
    if args.events:
        wanted = [e.strip() for e in str(args.events).split(",") if e.strip()]
        event_ids = [e for e in event_ids if e in wanted]
    if args.max_events is not None:
        event_ids = event_ids[: int(args.max_events)]

    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    rng = np.random.default_rng(int(args.seed))

    print(f"[info] SAC dir: {sac_dir}", flush=True)
    print(f"[info] Models: {models}", flush=True)
    print(f"[info] Events: {event_ids}", flush=True)
    print("[info] Null test uses the unmodified areswave.PSO objective (same as PSO).", flush=True)
    # If running under WSL, CPU/RAM usage shows up as VmmemWSL in Windows Task Manager.
    try:
        import platform
        rel = platform.release().lower()
        if "microsoft" in rel or "wsl" in rel:
            print("[note] Detected WSL: in Windows Task Manager, look at VmmemWSL for CPU/RAM while this runs.", flush=True)
    except Exception:
        pass

    for ev_id in event_ids:
        for model_key in models:
            if model_key not in MODEL_NAME:
                print(f"[skip] unknown model key: {model_key} (expected {list(MODEL_NAME.keys())})", flush=True)
                continue
            run_one(ev_id, model_key, sac_dir, out_dir, rng, args.N, args.batch, args.quiet)


if __name__ == "__main__":
    main()

