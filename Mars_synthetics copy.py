# -*- coding: utf-8 -*-
"""
Compare real and synthetic Marsquake waveforms for three processing stages:
1) raw ZRT (no bandpass, only detrend/taper/resample and UVW->ZRT rotation for real)
2) ZRT + bandpass 0.1--0.5 Hz
3) ZRT + bandpass 0.1--0.8 Hz
4) ZRT + bandpass 0.3--0.9 Hz
5) ZRT + bandpass 0.3--0.9 Hz + polarization filter

For each event, the script plots Z/R/T components in two columns:
- P window:  [-5, +5] s
- S window:  [-5, +10] s
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from obspy import Stream, Trace, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from dsmpy import seismicmodel_Mars
from dsmpy.event_Mars import Event, MomentTensor
from dsmpy.station_Mars import Station
from areswave.synthetics_function import generate_synthetics
from areswave.denoising import polarization_filter

# -----------------------------------------------------------------------------
# PATHS / GLOBAL SETTINGS
# -----------------------------------------------------------------------------
OUT_FIG_DIR = "/home/lyara/areswave/figs"
SYNTH_OUT_DIR = "/home/lyara/areswave/synthetics"
TAUP_MODEL_PATH = "/home/lyara/areswave/models/TAYAK.npz"

SAMPLING_HZ = 20.0
FREQMIN = 0.3
FREQMAX = 0.9

NSEC_BEFORE_P = 100.0
NSEC_AFTER_S = 200.0
EXTRA_DOWNLOAD = 100.0

NSPC = 1152
TLEN = 1276.8
MAX_TIME = 1500.0

STATION = Station(
    name="ELYSE",
    network="XB",
    latitude=4.502384,
    longitude=135.623447,
)
CLIENT = Client("IRIS")

# -----------------------------------------------------------------------------
# EVENT INPUTS
# -----------------------------------------------------------------------------
EVENT_CONFIGS = {
    "S0167b": dict(
        event_id="S0167b",
        name="S0167b",
        latitude=4.5024,
        longitude=-135.6234,
        distance=60.3,
        baz=107.0,
        depth=30.0,
        magnitude=2.9,
        time_p=UTCDateTime("2019-05-17T19:31:38"),
        time_s=UTCDateTime("2019-05-17T19:37:29"),
        centroid_time=UTCDateTime("2019-05-17 19:27:04"),
        mt=dict(Mrr=-2.8e20, Mrt=-1.9e20, Mrp=-1.3e20, Mtt=-1.4e20, Mtp=-5.3e20, Mpp=1.8e20),
    ),
    "S1102a": dict(
        event_id="S1102a",
        name="S1102a",
        latitude=25.22091318,
        longitude=61.96497917,
        distance=73.3,
        baz=267.0,
        depth=30.0,
        magnitude=3.2,
        time_p=UTCDateTime("2022-01-02T04:35:32"),
        time_s=UTCDateTime("2022-01-02T04:42:09"),
        centroid_time=UTCDateTime("2022-01-02T04:27:10"),
        mt=dict(Mrr=-2.8e20, Mrt=-1.9e20, Mrp=-1.3e20, Mtt=-1.4e20, Mtp=-5.3e20, Mpp=1.8e20),
    ),
}

# -----------------------------------------------------------------------------
# SMALL UTILITIES
# -----------------------------------------------------------------------------
def check_event_config(cfg):
    required = [
        "event_id", "name", "latitude", "longitude", "distance",
        "depth", "centroid_time", "time_p", "time_s", "mt",
    ]
    missing = [key for key in required if cfg.get(key) is None]
    if missing:
        raise ValueError(
            f"{cfg.get('name', 'EVENT')}: missing required fields: {missing}."
        )
    for key in ["Mrr", "Mrt", "Mrp", "Mtt", "Mtp", "Mpp"]:
        if key not in cfg["mt"] or cfg["mt"][key] is None:
            raise ValueError(f"{cfg['name']}: missing mt['{key}'].")
    if cfg.get("baz") is None:
        raise ValueError(
            f"{cfg['name']}: missing baz. I need it to rotate real UVW to ZRT before filtering."
        )


def as_utc(value):
    if isinstance(value, UTCDateTime):
        return value
    return UTCDateTime(value)


def safe_demean_normalize(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = x - np.nanmean(x)
    amp = np.nanmax(np.abs(x))
    if not np.isfinite(amp) or amp == 0:
        return x
    return x / amp


def bandpass_array(x, fs, freqmin=FREQMIN, freqmax=FREQMAX, order=4):
    x = np.asarray(x, dtype=float)
    nyq = 0.5 * fs
    b, a = butter(order, [freqmin / nyq, freqmax / nyq], btype="band")
    return filtfilt(b, a, x)


def select_trace(st, suffix):
    suffix = suffix.upper()
    for tr in st:
        if tr.stats.channel.upper().endswith(suffix):
            return tr
    channels = [tr.stats.channel for tr in st]
    raise ValueError(f"Could not find channel ending with {suffix}. Channels: {channels}")


def make_trace(data, template, channel):
    stats = template.stats.copy()
    stats.channel = channel
    return Trace(data=np.asarray(data, dtype=float), header=stats)


def trim_same_length_dict(data_dict, time_axis=None):
    n = min(len(data_dict[comp]) for comp in ["Z", "R", "T"])
    out = {comp: np.asarray(data_dict[comp][:n], dtype=float) for comp in ["Z", "R", "T"]}
    if time_axis is not None:
        return out, np.asarray(time_axis[:n], dtype=float)
    return out


# -----------------------------------------------------------------------------
# REAL DATA: IRIS UVW -> ZRT -> bandpass -> polarization
# -----------------------------------------------------------------------------
def rotate_xy(c1, c2, angle_deg):
    a = np.radians(angle_deg)
    o1 = np.cos(a) * c1 - np.sin(a) * c2
    o2 = np.sin(a) * c1 + np.cos(a) * c2
    return o1, o2


def uvw_to_zrt(st_uvw, baz):
    tr_u = select_trace(st_uvw, "U")
    tr_v = select_trace(st_uvw, "V")
    tr_w = select_trace(st_uvw, "W")

    n = min(len(tr_u.data), len(tr_v.data), len(tr_w.data))
    u = np.asarray(tr_u.data[:n], dtype=float)
    v = np.asarray(tr_v.data[:n], dtype=float)
    w = np.asarray(tr_w.data[:n], dtype=float)

    d = np.radians(-30.0)
    a_u = np.radians(135.0)
    a_v = np.radians(15.0)
    a_w = np.radians(255.0)

    A = np.array([
        [np.cos(d) * np.sin(a_u), np.cos(d) * np.cos(a_u), -np.sin(d)],
        [np.cos(d) * np.sin(a_v), np.cos(d) * np.cos(a_v), -np.sin(d)],
        [np.cos(d) * np.sin(a_w), np.cos(d) * np.cos(a_w), -np.sin(d)],
    ])

    e, n_, z = np.dot(np.linalg.inv(A), np.vstack([u, v, w]))
    t, r = rotate_xy(e, n_, baz)

    return Stream(traces=[
        make_trace(z, tr_u, "BHZ"),
        make_trace(r, tr_u, "BHR"),
        make_trace(t, tr_u, "BHT"),
    ])


def download_and_prepare_real(cfg):
    time_p = as_utc(cfg["time_p"])
    time_s = as_utc(cfg["time_s"])

    begin = time_p - NSEC_BEFORE_P
    end = time_s + NSEC_AFTER_S

    st = CLIENT.get_waveforms(
        "XB", "ELYSE", "02", "BH*",
        begin - EXTRA_DOWNLOAD / 2.0,
        end + EXTRA_DOWNLOAD,
        attach_response=True,
    )

    st = st.copy()
    st.detrend("linear")
    st.taper(max_percentage=0.05)
    st.resample(SAMPLING_HZ)

    # IMPORTANT: rotate first, then apply all filter stages in ZRT.
    st_zrt = uvw_to_zrt(st, cfg["baz"])
    st_zrt.detrend("demean")

    real_raw = {
        "Z": select_trace(st_zrt, "Z").data.copy(),
        "R": select_trace(st_zrt, "R").data.copy(),
        "T": select_trace(st_zrt, "T").data.copy(),
    }
    n = min(len(real_raw["Z"]), len(real_raw["R"]), len(real_raw["T"]))
    real_raw = {comp: real_raw[comp][:n] for comp in ["Z", "R", "T"]}
    t_real_abs = st_zrt[0].times()[:n]
    real_start = st_zrt[0].stats.starttime

    real_bp_01_05 = {
        comp: bandpass_array(real_raw[comp], SAMPLING_HZ, 0.1, 0.5)
        for comp in ["Z", "R", "T"]
    }
    real_bp_01_08 = {
        comp: bandpass_array(real_raw[comp], SAMPLING_HZ, 0.1, 0.8)
        for comp in ["Z", "R", "T"]
    }
    real_bp_03_09 = {
        comp: bandpass_array(real_raw[comp], SAMPLING_HZ, 0.3, 0.9)
        for comp in ["Z", "R", "T"]
    }

    real_pol_list = polarization_filter(
        [real_bp_03_09["Z"], real_bp_03_09["R"], real_bp_03_09["T"]],
        SAMPLING_HZ,
    )
    real_pol = {
        "Z": np.asarray(real_pol_list[0], dtype=float),
        "R": np.asarray(real_pol_list[1], dtype=float),
        "T": np.asarray(real_pol_list[2], dtype=float),
    }

    real_raw, t_real_abs = trim_same_length_dict(real_raw, t_real_abs)
    real_bp_01_05, _ = trim_same_length_dict(real_bp_01_05, t_real_abs)
    real_bp_01_08, _ = trim_same_length_dict(real_bp_01_08, t_real_abs)
    real_bp_03_09, _ = trim_same_length_dict(real_bp_03_09, t_real_abs)
    real_pol, t_real_abs = trim_same_length_dict(real_pol, t_real_abs)

    t_real_p = t_real_abs - (time_p - real_start)
    t_real_s = t_real_abs - (time_s - real_start)

    return {
        "raw": real_raw,
        "bp_01_05": real_bp_01_05,
        "bp_01_08": real_bp_01_08,
        "bp_03_09": real_bp_03_09,
        "bp_03_09_pol": real_pol,
        "t_p": t_real_p,
        "t_s": t_real_s,
    }

# -----------------------------------------------------------------------------
# SYNTHETIC DATA: DSMpy ZRT -> bandpass -> polarization
# -----------------------------------------------------------------------------
def build_dsmpy_event(cfg):
    mt = cfg["mt"]
    tensor = MomentTensor(
        mt["Mrr"], mt["Mrt"], mt["Mrp"],
        mt["Mtt"], mt["Mtp"], mt["Mpp"],
    )

    return Event(
        event_id=cfg["event_id"],
        latitude=cfg["latitude"],
        longitude=cfg["longitude"],
        depth=cfg["depth"],
        mt=tensor,
        centroid_time=as_utc(cfg["centroid_time"]).timestamp,
        source_time_function=None,
    )


def get_taup_arrivals(cfg):
    taup_model = TauPyModel(model=TAUP_MODEL_PATH)
    arrivals = taup_model.get_travel_times(
        source_depth_in_km=cfg["depth"],
        distance_in_degree=cfg["distance"],
        phase_list=["P", "S"],
    )
    p_arr = next((arr for arr in arrivals if arr.name.upper().startswith("P")), None)
    s_arr = next((arr for arr in arrivals if arr.name.upper().startswith("S")), None)
    if p_arr is None or s_arr is None:
        raise ValueError(f"{cfg['name']}: TauP did not return both P and S arrivals: {arrivals}")
    return p_arr.time, s_arr.time


def calculate_and_prepare_synthetic(cfg):
    event = build_dsmpy_event(cfg)
    seismic_model = seismicmodel_Mars.SeismicModel.test2()
    output = generate_synthetics(
        event,
        [STATION],
        seismic_model,
        TLEN,
        NSPC,
        SAMPLING_HZ,
    )

    os.makedirs(SYNTH_OUT_DIR, exist_ok=True)
    output.write(root_path=SYNTH_OUT_DIR, format="sac")
    ts = np.asarray(output.ts, dtype=float)
    max_time = min(MAX_TIME, ts[-1])
    max_idx = np.searchsorted(ts, max_time)
    ts = ts[:max_idx]

    syn_raw = {
        "Z": np.asarray(output["Z", "ELYSE_XB"][:max_idx], dtype=float),
        "R": np.asarray(output["R", "ELYSE_XB"][:max_idx], dtype=float),
        "T": np.asarray(output["T", "ELYSE_XB"][:max_idx], dtype=float),
    }

    syn_bp_01_05 = {
        comp: bandpass_array(syn_raw[comp], SAMPLING_HZ, 0.1, 0.5)
        for comp in ["Z", "R", "T"]
    }
    syn_bp_01_08 = {
        comp: bandpass_array(syn_raw[comp], SAMPLING_HZ, 0.1, 0.8)
        for comp in ["Z", "R", "T"]
    }
    syn_bp_03_09 = {
        comp: bandpass_array(syn_raw[comp], SAMPLING_HZ, 0.3, 0.9)
        for comp in ["Z", "R", "T"]
    }

    syn_pol_list = polarization_filter(
        [syn_bp_03_09["Z"], syn_bp_03_09["R"], syn_bp_03_09["T"]],
        SAMPLING_HZ,
    )
    syn_pol = {
        "Z": np.asarray(syn_pol_list[0], dtype=float),
        "R": np.asarray(syn_pol_list[1], dtype=float),
        "T": np.asarray(syn_pol_list[2], dtype=float),
    }

    syn_raw, ts = trim_same_length_dict(syn_raw, ts)
    syn_bp_01_05, _ = trim_same_length_dict(syn_bp_01_05, ts)
    syn_bp_01_08, _ = trim_same_length_dict(syn_bp_01_08, ts)
    syn_bp_03_09, _ = trim_same_length_dict(syn_bp_03_09, ts)
    syn_pol, ts = trim_same_length_dict(syn_pol, ts)

    travel_time_p, travel_time_s = get_taup_arrivals(cfg)
    t_syn_p = ts - travel_time_p
    t_syn_s = ts - travel_time_s

    return {
        "raw": syn_raw,
        "bp_01_05": syn_bp_01_05,
        "bp_01_08": syn_bp_01_08,
        "bp_03_09": syn_bp_03_09,
        "bp_03_09_pol": syn_pol,
        "t_p": t_syn_p,
        "t_s": t_syn_s,
        "travel_time_p": travel_time_p,
        "travel_time_s": travel_time_s,
    }

# -----------------------------------------------------------------------------
# CROSS CORRELATION AND PLOTTING
# -----------------------------------------------------------------------------
def window_to_grid(t, x, window, fs=SAMPLING_HZ):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    t0, t1 = window
    grid = np.arange(t0, t1 + 0.5 / fs, 1.0 / fs)

    order = np.argsort(t)
    t_sorted = t[order]
    x_sorted = x[order]

    y = np.interp(grid, t_sorted, x_sorted, left=np.nan, right=np.nan)
    return grid, y


def corr_in_window(t_real, x_real, t_syn, x_syn, window):
    grid, real_w = window_to_grid(t_real, x_real, window)
    _, syn_w = window_to_grid(t_syn, x_syn, window)

    mask = np.isfinite(real_w) & np.isfinite(syn_w)
    if np.count_nonzero(mask) < 5:
        return np.nan, grid, safe_demean_normalize(real_w), safe_demean_normalize(syn_w)

    real_n = safe_demean_normalize(real_w[mask])
    syn_n = safe_demean_normalize(syn_w[mask])

    if np.nanstd(real_n) == 0 or np.nanstd(syn_n) == 0:
        cc = np.nan
    else:
        cc = float(np.corrcoef(real_n, syn_n)[0, 1])

    return cc, grid, safe_demean_normalize(real_w), safe_demean_normalize(syn_w)


def plot_event_comparison(name, real, syn):
    stage_order = [
        "raw",
        "bp_01_05",
        "bp_01_08",
        "bp_03_09",
        "bp_03_09_pol",
    ]
    stage_labels = {
        "raw": "sem filtro",
        "bp_01_05": "bandpass 0.1-0.5",
        "bp_01_08": "bandpass 0.1-0.8",
        "bp_03_09": "bandpass 0.3-0.9",
        "bp_03_09_pol": "bandpass 0.3-0.9 + polarization",
    }

    panel_defs = [
        ("PZ", "Z", "P", "t_p", (-1.0, 10.0)),
        ("SZ", "Z", "S", "t_s", (-1.0, 10.0)),
        ("PR", "R", "P", "t_p", (-1.0, 10.0)),
        ("ST", "T", "S", "t_s", (-1.0, 10.0)),
    ]

    fig, axs = plt.subplots(
        nrows=len(stage_order),
        ncols=len(panel_defs),
        figsize=(18, 14),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )

    for i, stage in enumerate(stage_order):
        for j, (panel_name, comp, phase, time_key, window) in enumerate(panel_defs):
            ax = axs[i, j]

            cc, grid, real_plot, syn_plot = corr_in_window(
                real[time_key], real[stage][comp],
                syn[time_key], syn[stage][comp],
                window,
            )

            ax.plot(grid, real_plot, color="firebrick", lw=1.5, label="Real")
            ax.plot(grid, syn_plot, color="royalblue", lw=1.5, ls="--", label="Synthetic")
            ax.axvline(0.0, color="black", lw=0.8, ls="--", alpha=0.6)
            ax.set_xlim(window)
            ax.set_ylim(-1.15, 1.15)
            ax.grid(alpha=0.2)

            if i == 0:
                ax.set_title(panel_name, fontsize=12)

            if j == 0:
                ax.set_ylabel(f"{stage_labels[stage]}\nnormalized amp.")

            if i == len(stage_order) - 1:
                ax.set_xlabel(f"Time from {phase} arrival (s)")

            cc_text = "CC = NaN" if not np.isfinite(cc) else f"CC = {cc:.2f}"
            ax.text(
                0.03, 0.95, cc_text,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.80),
            )

            if i == 0 and j == 0:
                ax.legend(loc="lower right", fontsize=9, frameon=False)

    fig.suptitle(
        f"{name}: real vs synthetic in ZRT | rows = filter stages, cols = PZ SZ PR ST",
        fontsize=15,
    )

    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    out_png = os.path.join(OUT_FIG_DIR, f"{name}_real_synthetic_stage_grid.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def run_event(name):
    cfg = EVENT_CONFIGS[name]
    check_event_config(cfg)

    print(f"\n=== {name}: preparing real data ===")
    real = download_and_prepare_real(cfg)

    print(f"=== {name}: calculating synthetic data ===")
    syn = calculate_and_prepare_synthetic(cfg)

    print(f"=== {name}: plotting ===")
    plot_event_comparison(name, real, syn)


def main():
    for name in ["S0167b", "S1102a"]:
        run_event(name)


if __name__ == "__main__":
    main()