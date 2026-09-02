# -*- coding: utf-8 -*-
"""
Compare real and synthetic EARTHQUAKE waveforms for the terrestrial benchmark.

Adapted from the original Marsquake waveform-comparison script.

For each source configuration, the script plots Z/R/T components in two columns:
- P window: [-1, +10] s
- S window: [-1, +10] s

Two configurations are run:
1) IU_MAJO_original_reference
2) IU_MAJO_best_pso
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
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy.signal.rotate import rotate_ne_rt

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

SAMPLING_HZ = 20.0
FREQMIN = 0.3
FREQMAX = 0.9

NSEC_BEFORE_P = 100.0
NSEC_AFTER_S = 200.0
EXTRA_DOWNLOAD = 100.0

NSPC = 1152
TLEN = 1276.8
MAX_TIME = 1500.0

CLIENT = Client("IRIS")
TAUP_MODEL = TauPyModel(model="prem")


# -----------------------------------------------------------------------------
# EARTH STATION
# -----------------------------------------------------------------------------
STATION = Station(
    name="MAJO",
    network="IU",
    latitude=41.9486,
    longitude=142.5864,
)


# -----------------------------------------------------------------------------
# MOMENT TENSOR FROM Mw + STRIKE / DIP / RAKE
# -----------------------------------------------------------------------------
def mw_to_m0(mw):
    """
    Hanks & Kanamori relation, returning M0 in dyne cm.
    """
    return 10.0 ** (1.5 * mw + 16.1)


def sdr_to_mt_areswave(strike, dip, rake, m0):
    """
    Convert strike/dip/rake to moment tensor components.

    Output order:
    Mrr, Mrt, Mrp, Mtt, Mtp, Mpp
    """
    phi = np.deg2rad(strike)
    delta = np.deg2rad(dip)
    lamb = np.deg2rad(rake)

    sd = np.sin(delta)
    cd = np.cos(delta)
    s2d = np.sin(2.0 * delta)
    c2d = np.cos(2.0 * delta)

    sl = np.sin(lamb)
    cl = np.cos(lamb)

    sp = np.sin(phi)
    cp = np.cos(phi)
    s2p = np.sin(2.0 * phi)
    c2p = np.cos(2.0 * phi)

    Mrr =  m0 * s2d * sl
    Mtt = -m0 * (sd * cl * s2p + s2d * sl * sp * sp)
    Mpp =  m0 * (sd * cl * s2p - s2d * sl * cp * cp)
    Mrt = -m0 * (cd * cl * cp + c2d * sl * sp)
    Mrp =  m0 * (cd * cl * sp - c2d * sl * cp)
    Mtp = -m0 * (sd * cl * c2p + 0.5 * s2d * sl * s2p)

    return dict(
        Mrr=Mrr,
        Mrt=Mrt,
        Mrp=Mrp,
        Mtt=Mtt,
        Mtp=Mtp,
        Mpp=Mpp,
    )


MW_EARTH = 5.17
M0_EARTH = mw_to_m0(MW_EARTH)


# -----------------------------------------------------------------------------
# EVENT INPUTS
# -----------------------------------------------------------------------------
# IMPORTANT:
# Vanuatu is in the southern hemisphere, so latitude is negative.
# If you intentionally want to test your original typed value, change -18.512 to +18.512.
EARTH_LAT = -18.512
EARTH_LON = 168.062

DISTANCE_DEG = locations2degrees(
    EARTH_LAT,
    EARTH_LON,
    STATION.latitude,
    STATION.longitude,
)

_, AZIMUTH, BACK_AZIMUTH = gps2dist_azimuth(
    EARTH_LAT,
    EARTH_LON,
    STATION.latitude,
    STATION.longitude,
)

EVENT_CONFIGS = {
    "IU_MAJO_original_reference": dict(
        event_id="IU_MAJO_original_reference",
        name="IU_MAJO_original_reference",
        latitude=EARTH_LAT,
        longitude=EARTH_LON,
        distance=DISTANCE_DEG,
        baz=BACK_AZIMUTH,
        depth=31.0,
        magnitude=MW_EARTH,
        time_p=UTCDateTime("2006-11-29T08:42:15"),
        time_s=UTCDateTime("2006-11-29T08:49:16"),
        centroid_time=UTCDateTime(
            (
                UTCDateTime("2006-11-29T08:42:15").timestamp
                + UTCDateTime("2006-11-29T08:49:16").timestamp
            )
            / 2.0
        ),
        strike=187.0,
        dip=67.0,
        rake=104.0,
        mt=sdr_to_mt_areswave(
            strike=187.0,
            dip=67.0,
            rake=104.0,
            m0=M0_EARTH,
        ),
    ),

    "IU_MAJO_best_pso": dict(
        event_id="IU_MAJO_best_pso",
        name="IU_MAJO_best_pso",
        latitude=EARTH_LAT,
        longitude=EARTH_LON,
        distance=DISTANCE_DEG,
        baz=BACK_AZIMUTH,
        depth=32.3,
        magnitude=MW_EARTH,
        time_p=UTCDateTime("2006-11-29T08:42:15"),
        time_s=UTCDateTime("2006-11-29T08:49:16"),
        centroid_time=UTCDateTime(
            (
                UTCDateTime("2006-11-29T08:42:15").timestamp
                + UTCDateTime("2006-11-29T08:49:16").timestamp
            )
            / 2.0
        ),
        strike=190.6,
        dip=7.6,
        rake=54.9,
        mt=sdr_to_mt_areswave(
            strike=190.6,
            dip=7.6,
            rake=54.9,
            m0=M0_EARTH,
        ),
    ),
}


# -----------------------------------------------------------------------------
# SMALL UTILITIES
# -----------------------------------------------------------------------------
def check_event_config(cfg):
    required = [
        "event_id", "name", "latitude", "longitude", "distance",
        "depth", "centroid_time", "time_p", "time_s", "mt", "baz",
    ]
    missing = [key for key in required if cfg.get(key) is None]
    if missing:
        raise ValueError(
            f"{cfg.get('name', 'EVENT')}: missing required fields: {missing}."
        )

    for key in ["Mrr", "Mrt", "Mrp", "Mtt", "Mtp", "Mpp"]:
        if key not in cfg["mt"] or cfg["mt"][key] is None:
            raise ValueError(f"{cfg['name']}: missing mt['{key}'].")


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

    out = {
        comp: np.asarray(data_dict[comp][:n], dtype=float)
        for comp in ["Z", "R", "T"]
    }

    if time_axis is not None:
        return out, np.asarray(time_axis[:n], dtype=float)

    return out


# -----------------------------------------------------------------------------
# REAL DATA: EARTH ZNE/BH1/BH2 -> ZRT
# -----------------------------------------------------------------------------
def rotate_earth_to_zrt(st, inv, baz):
    """
    Rotate ordinary Earth station data to ZRT.

    This replaces the Mars/InSight UVW -> ZRT rotation.
    """
    st = st.copy()

    try:
        st.rotate(method="->ZNE", inventory=inv)
    except Exception as e:
        print(f"Warning: ->ZNE rotation failed or was unnecessary: {e}")

    try:
        st.rotate(method="NE->RT", back_azimuth=baz)

        z = select_trace(st, "Z").data
        r = select_trace(st, "R").data
        t = select_trace(st, "T").data
        template = select_trace(st, "Z")

        return Stream(traces=[
            make_trace(z, template, "BHZ"),
            make_trace(r, template, "BHR"),
            make_trace(t, template, "BHT"),
        ])

    except Exception as e:
        print(f"Warning: ObsPy NE->RT rotation failed. Trying manual N/E rotation: {e}")

        tr_z = select_trace(st, "Z")
        tr_n = select_trace(st, "N")
        tr_e = select_trace(st, "E")

        n = min(len(tr_z.data), len(tr_n.data), len(tr_e.data))

        z = np.asarray(tr_z.data[:n], dtype=float)
        north = np.asarray(tr_n.data[:n], dtype=float)
        east = np.asarray(tr_e.data[:n], dtype=float)

        r, t = rotate_ne_rt(north, east, baz)

        return Stream(traces=[
            make_trace(z, tr_z, "BHZ"),
            make_trace(r, tr_z, "BHR"),
            make_trace(t, tr_z, "BHT"),
        ])


def download_and_prepare_real(cfg):
    time_p = as_utc(cfg["time_p"])
    time_s = as_utc(cfg["time_s"])

    begin = time_p - NSEC_BEFORE_P
    end = time_s + NSEC_AFTER_S

    st = CLIENT.get_waveforms(
        "IU", "MAJO", "*", "BH*",
        begin - EXTRA_DOWNLOAD / 2.0,
        end + EXTRA_DOWNLOAD,
        attach_response=True,
    )

    inv = CLIENT.get_stations(
        network="IU",
        station="MAJO",
        location="*",
        channel="BH*",
        starttime=begin - EXTRA_DOWNLOAD / 2.0,
        endtime=end + EXTRA_DOWNLOAD,
        level="response",
    )

    st = st.copy()
    st.merge(method=1, fill_value="interpolate")
    st.detrend("linear")
    st.taper(max_percentage=0.05)
    st.resample(SAMPLING_HZ)

    st_zrt = rotate_earth_to_zrt(st, inv, cfg["baz"])
    st_zrt.detrend("demean")

    real_raw = {
        "Z": select_trace(st_zrt, "Z").data.copy(),
        "R": select_trace(st_zrt, "R").data.copy(),
        "T": select_trace(st_zrt, "T").data.copy(),
    }

    n = min(len(real_raw["Z"]), len(real_raw["R"]), len(real_raw["T"]))

    real_raw = {
        comp: real_raw[comp][:n]
        for comp in ["Z", "R", "T"]
    }

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
# SYNTHETIC DATA: DSMpy PREM -> bandpass -> polarization
# -----------------------------------------------------------------------------
def build_dsmpy_event(cfg):
    mt = cfg["mt"]

    tensor = MomentTensor(
        mt["Mrr"],
        mt["Mrt"],
        mt["Mrp"],
        mt["Mtt"],
        mt["Mtp"],
        mt["Mpp"],
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
    arrivals = TAUP_MODEL.get_travel_times(
        source_depth_in_km=cfg["depth"],
        distance_in_degree=cfg["distance"],
        phase_list=["P", "S"],
    )

    p_arr = next((arr for arr in arrivals if arr.name.upper().startswith("P")), None)
    s_arr = next((arr for arr in arrivals if arr.name.upper().startswith("S")), None)

    if p_arr is None or s_arr is None:
        raise ValueError(
            f"{cfg['name']}: TauP did not return both P and S arrivals: {arrivals}"
        )

    return p_arr.time, s_arr.time


def calculate_and_prepare_synthetic(cfg):
    event = build_dsmpy_event(cfg)

    seismic_model = seismicmodel_Mars.SeismicModel.prem()

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

    station_key = "MAJO_IU"

    syn_raw = {
        "Z": np.asarray(output["Z", station_key][:max_idx], dtype=float),
        "R": np.asarray(output["R", station_key][:max_idx], dtype=float),
        "T": np.asarray(output["T", station_key][:max_idx], dtype=float),
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
                real[time_key],
                real[stage][comp],
                syn[time_key],
                syn[stage][comp],
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
                0.03,
                0.95,
                cc_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
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

    out_png = os.path.join(
        OUT_FIG_DIR,
        f"{name}_real_synthetic_stage_grid.png",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")


def run_event(name):
    cfg = EVENT_CONFIGS[name]
    check_event_config(cfg)

    print("\n============================================================")
    print(f"Running: {name}")
    print("============================================================")
    print(f"Latitude / longitude = {cfg['latitude']}, {cfg['longitude']}")
    print(f"Distance = {cfg['distance']:.3f} deg")
    print(f"Back-azimuth = {cfg['baz']:.3f} deg")
    print(f"Depth = {cfg['depth']} km")
    print(f"Strike / dip / rake = {cfg['strike']} / {cfg['dip']} / {cfg['rake']}")
    print("Moment tensor:")
    for key in ["Mrr", "Mrt", "Mrp", "Mtt", "Mtp", "Mpp"]:
        print(f"  {key} = {cfg['mt'][key]:.6e}")

    print(f"\n=== {name}: preparing real data ===")
    real = download_and_prepare_real(cfg)

    print(f"=== {name}: calculating synthetic data ===")
    syn = calculate_and_prepare_synthetic(cfg)

    print(f"=== {name}: plotting ===")
    plot_event_comparison(name, real, syn)


def main():
    for name in [
        "IU_MAJO_original_reference",
        "IU_MAJO_best_pso",
    ]:
        run_event(name)


if __name__ == "__main__":
    main()