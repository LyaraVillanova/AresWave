from __future__ import annotations
import os, sys, glob, time, argparse, logging, inspect, json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read as obspy_read
from obspy import Trace, UTCDateTime
from obspy.taup import TauPyModel
from obspy.core.trace import Stats
from scipy.optimize import differential_evolution

# ==============================================================================
# Paths & constants
# ==============================================================================
PROJ_ROOT = Path(__file__).resolve().parents[0]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

OBS_DIR = "data/obs"
ARRIVALS_FILE = "data/arrivals.csv"
FIGS_DIR = "figs"
OUTPUTS_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUTS_DIR, "result.csv")
OUTPUT_ENSEMBLE = os.path.join(OUTPUTS_DIR, "ensemble_results.csv")
OUTPUT_EVENTCOSTS = os.path.join(OUTPUTS_DIR, "event_costs.csv")
OUTPUT_MODEL = os.path.join(OUTPUTS_DIR, "best_model.json")

# Search depth window for misfit (km)
DEPTH_MIN = 500.0
DEPTH_MAX = 1800.0
DEPTH_RANGE = (700.0, 1100.0)
DEPTH_STEP  = 100.0
REF_TOPO    = 1000.0  # km
REF_BASE    = 1200.0  # km
APPARENT_VP = 8.0     # km/s
APPARENT_VS = 4.5     # km/s

# Time windows (seconds)
P_BEFORE, P_AFTER = 5.0, 15.0
S_BEFORE, S_AFTER = 10.0, 25.0

# Plotting
CMAP_NAME = "plasma"

# DE defaults (override via CLI)
MAX_ITER = 3 #10
POPSIZE  = 5 #10
SEED     = 2025

# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("invert")
logger.setLevel(logging.DEBUG)   # força debug por padrão para facilitar diagnósticos
logger.debug("Debug mode FORÇADO no código.")

# ======================================================================
# Bloco auxiliar local — geração de sintéticos DSMpy dentro do inversion.py
# ======================================================================

import numpy as np
from dsmpy import dsm, event_Mars as evm

def _build_local_dsmpy_model(z_grid, vp_prof, vs_prof):
    """
    Constrói um modelo DSMpy compatível com a versão que não aceita parâmetros nomeados.
    """
    import numpy as np
    from dsmpy.seismicmodel import SeismicModel

    vpv = np.asarray(vp_prof, dtype=float)
    vsv = np.asarray(vs_prof, dtype=float)
    vph = vpv.copy()
    vsh = vsv.copy()
    eta = np.ones_like(vpv)
    rho = 2.6 + 0.3 * (vpv / np.max(vpv))
    qkappa = np.full_like(vpv, 600.0)
    qmu = np.full_like(vpv, 300.0)
    vrmin = np.zeros_like(vpv)
    vrmax = np.ones_like(vpv)

    # Construtor posicional
    model = SeismicModel(
        "local_model",   # model_id
        vrmin, vrmax,
        rho, vpv, vph, vsv, vsh,
        eta, qmu, qkappa,
        np.asarray(z_grid, dtype=float)
    )

    return model


def _local_synthesize_event_traces(z_grid, vp_prof, vs_prof, arr_event):
    """
    Gera traços Z, R, T usando o modelo local DSMpy.
    Não depende dos outros módulos.
    """
    seis_model = _build_local_dsmpy_model(z_grid, vp_prof, vs_prof)

    ev = evm.EventMars(
        lat=float(arr_event.get('lat', 0.0)),
        lon=float(arr_event.get('lon', 0.0)),
        depth=float(arr_event.get('depth', 30.0)),
        mt=np.asarray(arr_event.get('mt', np.zeros((3, 3))), dtype=float),
        t0=float(arr_event.get('t0', 0.0)),
    )

    solver = dsm.DSM()
    solver.set_seismic_model(seis_model)
    solver.set_event(ev)
    solver.compute_psv()
    solver.compute_sh()

    out = {}
    try:
        out['Z'] = solver.get_trace('Z')
        out['R'] = solver.get_trace('R')
        out['T'] = solver.get_trace('T')
    except Exception as e:
        print(f"[DSM local synth] Falha ao extrair traços: {e}")
    return out

# ==============================================================================
# Utilities
# ==============================================================================

def ensure_dirs():
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUTS_DIR, "synthetics_debug"), exist_ok=True)

def norm_by_rms(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x*x) + 1e-12)
    return x / rms

def pad_trim_to_n(x: np.ndarray, n: int) -> np.ndarray:
    """Pad/trim para comprimento exato n."""
    x = np.asarray(x, dtype=float)
    if x.size == n:
        return x
    if x.size > n:
        return x[:n]
    return np.pad(x, (0, n - x.size), mode='constant', constant_values=0.0)

def fixed_window_len(dt: float, t_before: float, t_after: float) -> int:
    """Define comprimento de janela de forma consistente (sem incluir endpoint)."""
    if not np.isfinite(dt) or dt <= 0:
        return 0
    return int(np.round((t_before + t_after) / dt))

def extract_window_time(arr: np.ndarray, dt: float, center_t: float|None,
                        t_before: float, t_after: float) -> np.ndarray:
    """Recorte simples (sem padding)."""
    if center_t is None or not np.isfinite(center_t):
        return np.array([], dtype=float)
    t0 = max(0.0, center_t - t_before)
    t1 = center_t + t_after
    if t1 <= 0.0:
        return np.array([], dtype=float)
    i0 = int(max(0, round(t0 / dt)))
    i1 = int(min(len(arr), round(t1 / dt)))
    if i1 <= i0:
        return np.array([], dtype=float)
    return np.asarray(arr[i0:i1], dtype=float)

def extract_window_time_with_padding(arr: np.ndarray, dt: float,
                                     center_t: float|None, t_before: float, t_after: float) -> np.ndarray:
    """
    Recorte com zero-padding quando a janela extrapola o traço.
    Retorna exatamente round((t_before+t_after)/dt) amostras após pad/trim externo.
    """
    if not np.isfinite(dt) or dt <= 0:
        return np.array([], dtype=float)
    n_desired = fixed_window_len(dt, t_before, t_after)
    if center_t is None or not np.isfinite(center_t):
        return np.zeros(n_desired, dtype=float)

    n = len(arr)
    if n == 0:
        return np.zeros(n_desired, dtype=float)

    t0d = center_t - t_before
    t1d = center_t + t_after
    i0d = int(np.floor(t0d / dt))
    i1d = int(np.ceil (t1d / dt))
    i0 = max(0, i0d)
    i1 = min(n, i1d)
    core = arr[i0:i1].astype(float, copy=False)

    pad_left = max(0, -i0d)
    pad_right = max(0, i1d - n)
    if pad_left or pad_right:
        core = np.pad(core, (pad_left, pad_right), mode='constant', constant_values=0.0)

    return pad_trim_to_n(core, n_desired)

def resample_to_dt(x: np.ndarray, dt_src: float, dt_tgt: float) -> np.ndarray:
    """
    Reamostra preservando a DURAÇÃO: N_tgt = round(len(x)*dt_src/dt_tgt)
    Limpa NaN/Inf e garante tamanho alvo.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    if not (np.isfinite(dt_src) and np.isfinite(dt_tgt)) or dt_src <= 0 or dt_tgt <= 0:
        return np.zeros_like(x)
    if np.isclose(dt_src, dt_tgt):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    dur = x.size * dt_src
    n_tgt = int(np.round(dur / dt_tgt))
    n_tgt = max(n_tgt, 1)

    # Interpolação linear estável
    xi = np.arange(x.size) * dt_src
    xo = np.arange(n_tgt) * dt_tgt
    xi = xi - xi[0]; xo = xo - xo[0]
    if xo[-1] > xi[-1] and x.size > 1:
        xo[-1] = xi[-1]  # evita extrapolação no último ponto
    y = np.interp(xo, xi, x).astype(float)

    # Ajuste fino
    return pad_trim_to_n(y, n_tgt)

def equalize_lengths(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    if n <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return a[:n], b[:n]

def _log_if_flat(tag: str, arr: np.ndarray):
    if arr.size == 0 or np.allclose(arr, 0.0):
        logger.debug(f"[{tag}] janela toda zero (size={arr.size})")

import numpy as np
import os
SAVE_WAVEFORMS_IMG = True     # salva imagem além do npy
IMG_FMT = "png"               # "png" ou "jpg"
MAX_LAG_FRAC = 0.10           # procurar melhor lag em ±10% do traço
ALPHA_CC = 0.7                # peso de 1-CC^2
ALPHA_MSE = 0.3               # peso de MSE

def _best_lag_cc(obs_z, syn_z, max_lag):
    """Return (best_cc, best_lag amostras, syn_shifted)."""
    n = len(obs_z)
    if max_lag <= 0:
        # equivalente à CC sem lag
        cc = float(np.clip(np.dot(obs_z, syn_z) / n, -1.0, 1.0))
        return cc, 0, syn_z.copy()

    # correlação cruzada via FFT
    fft_len = int(1 << (len(obs_z) * 2 - 1).bit_length())
    O = np.fft.rfft(obs_z, fft_len)
    S = np.fft.rfft(syn_z, fft_len)
    xcorr = np.fft.irfft(O * np.conj(S), fft_len)
    # rearranja para lags negativos/positivos ao redor de 0
    xcorr = np.concatenate((xcorr[-(n-1):], xcorr[:n]))

    # janela de busca: [-max_lag, +max_lag]
    center = n - 1
    i0 = center - max_lag
    i1 = center + max_lag + 1
    i0 = max(i0, 0); i1 = min(i1, len(xcorr))
    seg = xcorr[i0:i1]

    # normalização por energia para CC adequada
    denom = np.linalg.norm(obs_z) * np.linalg.norm(syn_z)
    cc_seg = seg / (denom + 1e-12)
    k = int(np.argmax(cc_seg))
    best_cc = float(np.clip(cc_seg[k], -1.0, 1.0))
    best_lag = (i0 + k) - center

    # aplica o atraso no sintético (shift por padding)
    if best_lag > 0:
        syn_shifted = np.pad(syn_z, (best_lag, 0), mode='constant')[:n]
    elif best_lag < 0:
        syn_shifted = np.pad(syn_z, (0, -best_lag), mode='constant')[-n:]
    else:
        syn_shifted = syn_z.copy()

    return best_cc, best_lag, syn_shifted

def _save_waveform_image(tag, obs, syn, lag_samp, outdir="outputs/synthetics_debug"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(outdir, exist_ok=True)
        fpath = os.path.join(outdir, f"{tag}.{IMG_FMT}")

        plt.figure(figsize=(9, 4))
        plt.plot(obs, label="Obs")
        plt.plot(syn, label=f"Syn (lag={lag_samp} samp)")
        plt.title(tag)
        plt.xlabel("amostras")
        plt.ylabel("amplitude (normalizada)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fpath, dpi=150)
        plt.close()
    except Exception as e:
        # não falhar a inversão por causa do plot
        print(f"[WARN] falha ao salvar imagem {tag}: {e}")

def misfit_metric(obs, syn, tag=""):
    """Calcula custo e salva debug (npy + opcionalmente png/jpg)."""
    # cópias float64
    obs = np.asarray(obs, dtype=float).copy()
    syn = np.asarray(syn, dtype=float).copy()
    n = min(len(obs), len(syn))
    if n == 0:
        return 1.0

    obs = obs[:n]; syn = syn[:n]

    # normalização z-score simples (evita ganho dominar)
    o_mu, o_std = float(np.mean(obs)), float(np.std(obs) + 1e-12)
    s_mu, s_std = float(np.mean(syn)), float(np.std(syn) + 1e-12)
    obs_z = (obs - o_mu) / o_std
    syn_z = (syn - s_mu) / s_std

    # busca melhor lag
    max_lag = int(MAX_LAG_FRAC * n)
    cc, lag_samp, syn_best = _best_lag_cc(obs_z, syn_z, max_lag)

    # MSE entre séries alinhadas
    mse = float(np.mean((obs_z - syn_best) ** 2))

    # custo híbrido
    cost = ALPHA_CC * (1.0 - cc ** 2) + ALPHA_MSE * mse

    # debug: salvar arrays
    if tag:
        outdir = "outputs/synthetics_debug"
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, f"{tag}_obs.npy"), obs)
        np.save(os.path.join(outdir, f"{tag}_syn.npy"), syn)
        np.save(os.path.join(outdir, f"{tag}_syn_aligned.npy"), syn_best)
        if SAVE_WAVEFORMS_IMG:
            _save_waveform_image(tag, obs_z, syn_best, lag_samp, outdir=outdir)

    # log simpático
    if tag:
        print(f"{time.strftime('%H:%M:%S')} | DEBUG | [misfit_metric] {tag}: "
              f"CC={cc:.4f} lag={lag_samp:+d} -> MSE={mse:.4f} cost={cost:.4f}")

    return cost

def estimate_lag_seconds(obs: np.ndarray, dt_obs: float,
                         syn: np.ndarray, dt_syn: float,
                         search_halfwin_s: float = 120.0) -> float:
    """
    Estima lag (syn -> obs) maximizando correlação normalizada.
    Retorna lag em segundos: somar esse valor ao tempo-centro do SYN alinha com o OBS.
    """
    # resample SYN para dt_obs (evita bias)
    syn_rs = resample_to_dt(np.asarray(syn, float), dt_syn, dt_obs)
    obs    = np.asarray(obs, float)

    # normalização robusta
    def _n(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x -= x.mean() if x.size else 0.0
        v = (x*x).mean() if x.size else 0.0
        return x/np.sqrt(v+1e-12)
    o = _n(obs); s = _n(syn_rs)
    if o.size < 8 or s.size < 8:
        return 0.0

    # limita auto-janela de busca
    max_lag_samps = int(round(search_halfwin_s / dt_obs))
    # correlação cheia e recorte de lags
    cc_full = np.correlate(o, s, mode='full')
    lags = np.arange(-len(s)+1, len(o))
    keep = (lags >= -max_lag_samps) & (lags <= max_lag_samps)
    if not np.any(keep):
        return 0.0
    lag_samps = lags[keep][np.argmax(cc_full[keep])]
    return float(lag_samps * dt_obs)

# ==============================================================================
# I/O: arrivals and observed data
# ==============================================================================

def _parse_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default

def _parse_time(v):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)) or (isinstance(v, str) and v.strip()==""):
            return None
        return UTCDateTime(v)
    except Exception:
        return None

def load_arrivals(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"arrivals.csv not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    out: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        ev = str(r.get('event_id'))
        meta: Dict[str, Any] = {
            'event_id': ev,
            'p_sec': _parse_float(r.get('p_sec', np.nan)),
            's_sec': _parse_float(r.get('s_sec', np.nan)),
            'latitude': _parse_float(r.get('latitude', np.nan)),
            'longitude': _parse_float(r.get('longitude', np.nan)),
            'depth': _parse_float(r.get('depth', np.nan)),
            'distance': _parse_float(r.get('distance', np.nan)),
            'time_p': _parse_time(r.get('time_p', None)),
            'time_s': _parse_time(r.get('time_s', None)),
        }
        # Moment tensor (optional)
        mrr = _parse_float(r.get('mrr', np.nan)); mrt = _parse_float(r.get('mrt', np.nan))
        mrp = _parse_float(r.get('mrp', np.nan)); mtt = _parse_float(r.get('mtt', np.nan))
        mtp = _parse_float(r.get('mtp', np.nan)); mpp = _parse_float(r.get('mpp', np.nan))
        M = np.array([mrr, mrt, mrp, mtt, mtp, mpp], dtype=float)
        if np.isfinite(M).any():
            vmax = np.nanmax(np.abs(M))
            scale = 1.0
            # autoscale toward ~1e25 dyne*cm expected by DSMpy
            if vmax < 1e23:
                scale = 1e5
            M *= scale
            meta['mt'] = {'mrr': M[0], 'mrt': M[1], 'mrp': M[2], 'mtt': M[3], 'mtp': M[4], 'mpp': M[5]}
        out[ev] = meta
    return out

def group_traces_by_event(obs_dir: str) -> Dict[str, Dict[str, Trace]]:
    if not os.path.isdir(obs_dir):
        raise FileNotFoundError(f"Observed data folder not found: {obs_dir}")
    events: Dict[str, Dict[str, Trace]] = {}
    for ev_dir in sorted(glob.glob(os.path.join(obs_dir, '*'))):
        if not os.path.isdir(ev_dir):
            continue
        ev = os.path.basename(ev_dir)
        traces: Dict[str, Trace] = {}
        for fn in sorted(glob.glob(os.path.join(ev_dir, '*'))):
            try:
                st = obspy_read(fn)
            except Exception:
                continue
            for tr in st:
                ch = (tr.stats.channel or '').upper()
                if 'Z' in ch: traces['Z'] = tr.copy()
                elif 'R' in ch or 'N' in ch: traces['R'] = tr.copy()
                elif 'T' in ch or 'E' in ch: traces['T'] = tr.copy()
        if traces:
            # garante dt consistente nos canais presentes
            for k, tr in traces.items():
                if not hasattr(tr.stats, "delta") or not np.isfinite(tr.stats.delta):
                    raise RuntimeError(f"{ev}:{k} sem delta válido")
            events[ev] = traces
    return events

# ==============================================================================
# Forward synthesis
# ==============================================================================

def _build_dsmpy_model_from_profiles(depths: np.ndarray, vp: np.ndarray, vs: np.ndarray):
    """
    Constrói um modelo DSMpy isotrópico a partir de perfis Vp/Vs de uma iteração.
    Compatível com a estrutura de atributos usada em dsmpy.seismicmodel_Mars.
    """
    try:
        from dsmpy import seismicmodel_Mars as seismicmodel_mod
    except Exception:
        from dsmpy import seismicmodel as seismicmodel_mod

    sm = seismicmodel_mod.SeismicModel.test2()

    depths = np.asarray(depths, float)
    vp = np.asarray(vp, float)
    vs = np.asarray(vs, float)

    # Atributos principais — note que DSMpy usa 'z' e não 'depths'
    sm.z = depths
    sm.vpv = vp.copy()
    sm.vph = vp.copy()
    sm.vsv = vs.copy()
    sm.vsh = vs.copy()
    sm.eta = np.ones_like(vp)
    sm.rho = np.clip(1.6612 * vp - 0.4721, 2.0, None)
    sm.qmu = np.full_like(vp, 10000.0)
    sm.qkappa = np.full_like(vp, 10000.0)

    # Atualiza número de camadas e recompila
    sm.nlayer = len(depths)
    if hasattr(sm, "build"):
        try:
            sm.build()
        except Exception as e:
            logger.debug(f"DSM model build() failed: {e}")

    logger.debug(f"DSMpy model built: nlayer={sm.nlayer}, z[0]={sm.z[0]:.1f}, vpv[0]={sm.vpv[0]:.2f}")
    return sm

def _wrap_out_to_traces(out, fs: float) -> Dict[str, Trace]:
    def _as_arr(x):
        arr = np.asarray(x).squeeze()
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        uZ = _as_arr(out['Z']); uR = _as_arr(out['R']); uT = _as_arr(out['T'])
    except Exception:
        # dsmpy object with us[3, nr(=1), nt]
        uZ = _as_arr(out.us[0, 0, :])
        uR = _as_arr(out.us[1, 0, :])
        uT = _as_arr(out.us[2, 0, :])
    stats = Stats(); stats.delta = 1.0 / fs; stats.starttime = UTCDateTime(0)
    trZ = Trace(uZ.astype(np.float32), header=stats.copy()); trZ.stats.channel = 'Z'
    trR = Trace(uR.astype(np.float32), header=stats.copy()); trR.stats.channel = 'R'
    trT = Trace(uT.astype(np.float32), header=stats.copy()); trT.stats.channel = 'T'
    return {"Z": trZ, "R": trR, "T": trT}

def _dsmpy_generate(ev_meta: Dict[str, Any], sm, fs: float, tlen: float) -> Dict[str, Trace]:
    """
    Generate synthetics using DSMpy with proper PyDSMInput construction.
    """
    from dsmpy.event_Mars import Event, MomentTensor
    from dsmpy.station_Mars import Station
    from dsmpy.dsm_Mars import PyDSMInput
    from dsmpy import dsm

    ev_id = str(ev_meta.get('event_id', 'ev'))
    logger.info(f"[DSMpy] Generating synthetics for event {ev_id}")

    # --- Event & MT
    mt_meta = ev_meta.get('mt', None)
    Mrr = float((mt_meta or {}).get('mrr', -2.9e25))
    Mrt = float((mt_meta or {}).get('mrt', -1.1e25))
    Mrp = float((mt_meta or {}).get('mrp', -1.6e25))
    Mtt = float((mt_meta or {}).get('mtt', -1.8e25))
    Mtp = float((mt_meta or {}).get('mtp', -4.6e25))
    Mpp = float((mt_meta or {}).get('mpp',  2.5e25))
    mt = MomentTensor(Mrr, Mrt, Mrp, Mtt, Mtp, Mpp)

    ev = Event(
        event_id=ev_id,
        latitude=float(ev_meta.get('latitude', 10.0)),
        longitude=float(ev_meta.get('longitude', 30.0)),
        depth=float(ev_meta.get('depth', 30.0)),
        mt=mt,
        centroid_time=0.0,
        source_time_function=None,
    )

    stations = [Station(name='ELYSE', network='XB', latitude=4.502384, longitude=135.623447)]

    # --- PyDSMInput
    nspc = 256  # P+S
    _ = inspect.signature(dsm.compute)  # se falhar, cai no except do caller
    pydsm_input = PyDSMInput.input_from_arrays(ev, stations, sm, tlen, nspc, fs)
    synthetics = dsm.compute(pydsm_input)
    synthetics.to_time_domain()
    return _wrap_out_to_traces(synthetics, fs)

def _apply_optional_filter(uZ: np.ndarray, uR: np.ndarray, uT: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from areswave.synthetics_function import apply_filter as _af
        uZ = _af(uZ, fs); uR = _af(uR, fs); uT = _af(uT, fs)
    except Exception:
        pass
    return uZ, uR, uT

@dataclass
class ModelParam:
    """
    Modelo 1D com múltiplas interfaces de transição:
    d_shallow (~800 km), d1000 (~1000 km), d_deep (~1200 km), d_layer (~1600 km)
    """

    def __init__(self, vp_top, vp_bot, vs_top, vs_bot, d1000, w1000, d_layer,
                 d_shallow=800.0, d_deep=1200.0):
        self.vp_top = vp_top
        self.vp_bot = vp_bot
        self.vs_top = vs_top
        self.vs_bot = vs_bot
        self.d_shallow = d_shallow
        self.d1000 = d1000
        self.d_deep = d_deep
        self.d_layer = d_layer
        self.w1000 = w1000

    def sigmoid(self, z, center, width):
        return 1.0 / (1.0 + np.exp(-(z - center) / width))

    def to_profiles(self, z):
        """
        Retorna perfis contínuos de Vp e Vs ao longo de z (km),
        combinando 4 interfaces sigmoides.
        """
        s_shallow = self.sigmoid(z, self.d_shallow, self.w1000)
        s_1000 = self.sigmoid(z, self.d1000, self.w1000)
        s_deep = self.sigmoid(z, self.d_deep, self.w1000)
        s_layer = self.sigmoid(z, self.d_layer, self.w1000)

        # combinação ponderada de transições
        weight = 0.2*s_shallow + 0.3*s_1000 + 0.3*s_deep + 0.2*s_layer

        vp_prof = self.vp_top + (self.vp_bot - self.vp_top) * weight
        vs_prof = self.vs_top + (self.vs_bot - self.vs_top) * weight

        return vp_prof, vs_prof


def _kinematic_warp_forward(model_param: ModelParam, fs: float) -> Dict[str, Trace]:
    """Fallback sintético simples e suave em θ (mantém aprendizado da DE)."""
    dt = 1.0 / fs
    n = max(1, int(1200 * fs))
    t = np.arange(n) * dt
    phi = (model_param.vp_top + model_param.vp_bot + model_param.vs_top + model_param.vs_bot) / 40.0
    z = np.exp(-t/200.0) * np.sin(2*np.pi*(0.02 + 0.001*phi)*t)
    r = np.exp(-t/220.0) * np.sin(2*np.pi*(0.018 + 0.0012*phi)*t + 0.3)
    tr = Stats(); tr.delta = dt; tr.starttime = UTCDateTime(0)
    return {
        'Z': Trace(z.astype(np.float32), header=tr.copy()),
        'R': Trace(r.astype(np.float32), header=tr.copy()),
        'T': Trace((z*0.8 - r*0.2).astype(np.float32), header=tr.copy())
    }

from obspy.taup import TauPyModel
def synthesize_event_traces(z_grid, vp_prof, vs_prof, ev_meta):
    fs = 20.0
    tlen = 1276.8
    sm = _build_dsmpy_model_from_profiles(z_grid, vp_prof, vs_prof)
    try:
        syn = _dsmpy_generate(ev_meta, sm, fs, tlen)
    except Exception as e:
        logger.warning("DSMpy failed for event %s: %s -> using kinematic warp", ev_meta.get('event_id', 'ev'), e)
        mp = ModelParam(vp_top=float(vp_prof[0]), vp_bot=float(vp_prof[-1]),
                        vs_top=float(vs_prof[0]), vs_bot=float(vs_prof[-1]), d1000=1000.0, w1000=100.0)
        syn = _kinematic_warp_forward(mp, fs)

    # filtro opcional protegido contra RMS zerado
    fZ, fR, fT = _apply_optional_filter(syn['Z'].data.copy(), syn['R'].data.copy(), syn['T'].data.copy(), fs)
    def _safe_apply(orig, fil):
        o = np.asarray(orig, float); f = np.asarray(fil, float)
        return f if np.sqrt((f*f).mean()+1e-12) > 1e-6*np.sqrt((o*o).mean()+1e-12) else o
    syn['Z'].data = _safe_apply(syn['Z'].data, fZ)
    syn['R'].data = _safe_apply(syn['R'].data, fR)
    syn['T'].data = _safe_apply(syn['T'].data, fT)
    return syn

# ==============================================================================
# Bounds & grid
# ==============================================================================
@dataclass
class ParamBounds:
    vp_min: float = 6.0; vp_max: float = 10.0
    vs_min: float = 3.2; vs_max: float = 5.5
    d1000_min: float = 900.0; d1000_max: float = 1100.0
    grad_min: float = 20.0;   grad_max: float = 150.0

PB = ParamBounds()
Z_GRID = np.arange(PB.d1000_min-200, PB.d1000_max+800 + 1.0, 1.0, dtype=np.float32)

# ==============================================================================
# Misfits vs depth
# ==============================================================================

import copy

def compute_misfits_vs_depth(
    ev_id: str,
    obs: Dict[str, Trace],
    syn: Dict[str, Trace],
    arr: Dict[str, Any],
    depth_range: Tuple[float, float],
    step: float,
    regen_synthetics: bool = False,
    vp_prof=None,
    vs_prof=None
):
    # --- TauP: crie SEMPRE fora do if/raise ---
    custom_model_path = "/home/lyara/areswave/models/tayak.npz"
    if not os.path.exists(custom_model_path):
        raise FileNotFoundError(f"Modelo TauP não encontrado: {custom_model_path}")
    model = TauPyModel(model=custom_model_path)

    # --- helper para checar "syn" válido ---
    def _valid_syn_dict(s) -> bool:
        try:
            return (
                isinstance(s, dict)
                and ('Z' in s) and ('R' in s) and ('T' in s)
                and hasattr(s['Z'].stats, 'delta') and np.isfinite(s['Z'].stats.delta) and s['Z'].stats.delta > 0
                and hasattr(s['R'].stats, 'delta') and np.isfinite(s['R'].stats.delta) and s['R'].stats.delta > 0
                and hasattr(s['T'].stats, 'delta') and np.isfinite(s['T'].stats.delta) and s['T'].stats.delta > 0
            )
        except Exception:
            return False

    # dist, dt, janelas observadas
    dist_deg = float(arr.get('distance', np.nan))
    start_obs = obs['Z'].stats.starttime
    dt_obs = float(obs['Z'].stats.delta)

    nP = fixed_window_len(dt_obs, P_BEFORE, P_AFTER)
    nS = fixed_window_len(dt_obs, S_BEFORE, S_AFTER)

    # profundidades a testar
    depths = np.arange(depth_range[0], depth_range[1] + step, step, dtype=float)

    # se NÃO vamos regerar a cada profundidade, precisamos validar "syn" já na entrada
    if not regen_synthetics and not _valid_syn_dict(syn):
        logger.warning("Synthetic generation failed or returned incomplete data (faltando Z/R/T ou delta inválido).")
        bad = np.full_like(depths, 999.0, dtype=float)
        return depths, bad, bad, bad, bad

    # tempos teóricos observados (start_obs é o relógio dos dados)
    tP_obs = float(arr['time_p'] - start_obs) if arr.get('time_p') else None
    tS_obs = float(arr['time_s'] - start_obs) if arr.get('time_s') else None

    # janelas fixas observadas
    zP_obs = pad_trim_to_n(extract_window_time(obs['Z'].data, dt_obs, tP_obs, P_BEFORE, P_AFTER), nP)
    rP_obs = pad_trim_to_n(extract_window_time(obs['R'].data, dt_obs, tP_obs, P_BEFORE, P_AFTER), nP)
    zS_obs = pad_trim_to_n(extract_window_time(obs['Z'].data, dt_obs, tS_obs, S_BEFORE, S_AFTER), nS)
    tS_obs_arr = pad_trim_to_n(extract_window_time(obs['T'].data, dt_obs, tS_obs, S_BEFORE, S_AFTER), nS)

    misP_top, misS_top, misP_base, misS_base = [], [], [], []

    # dt_syn (se não for regerado) — quando regera, recalculamos dentro do loop
    dt_syn_fixed = None
    if not regen_synthetics:
        dt_syn_fixed = float(syn['Z'].stats.delta)

    for d in depths:
        # teóricos P/S (TauP) para a profundidade "d"
        try:
            p_theo = model.get_travel_times(distance_in_degree=dist_deg,
                                            source_depth_in_km=d,
                                            phase_list=["P"])[0].time
        except Exception:
            p_theo = np.nan
        try:
            s_theo = model.get_travel_times(distance_in_degree=dist_deg,
                                            source_depth_in_km=d,
                                            phase_list=["S"])[0].time
        except Exception:
            s_theo = np.nan

        # (opcional) regera sintéticos a cada profundidade
        if regen_synthetics:
            if (vp_prof is None) or (vs_prof is None):
                logger.warning("regen_synthetics=True mas vp_prof/vs_prof são None.")
                bad = np.full_like(depths, 999.0, dtype=float)
                return depths, bad, bad, bad, bad
            arr_d = copy.deepcopy(arr)
            arr_d['depth'] = float(d)
            try:
                syn_d = _local_synthesize_event_traces(Z_GRID, vp_prof, vs_prof, arr_d)
            except Exception as e:
                logger.warning("Falha ao regerar sintéticos em d=%.1f km: %s", d, e)
                syn_d = None
            if not _valid_syn_dict(syn_d):
                # se falhou nessa profundidade, penaliza alto nessa amostra
                misP_top.append(999.0); misS_top.append(999.0)
                misP_base.append(999.0); misS_base.append(999.0)
                continue
            syn_used = syn_d
            dt_syn = float(syn_d['Z'].stats.delta)
        else:
            syn_used = syn
            dt_syn = dt_syn_fixed

        # deslocamentos aparentes (topo/base) para alinhar janelas
        shiftP_top  = (d - REF_TOPO) / APPARENT_VP
        shiftS_top  = (d - REF_TOPO) / APPARENT_VS
        shiftP_base = (d - REF_BASE) / APPARENT_VP
        shiftS_base = (d - REF_BASE) / APPARENT_VS

        cP_top  = (p_theo if np.isfinite(p_theo) else 0.0) - shiftP_top
        cS_top  = (s_theo if np.isfinite(s_theo) else 0.0) - shiftS_top
        cP_base = (p_theo if np.isfinite(p_theo) else 0.0) - shiftP_base
        cS_base = (s_theo if np.isfinite(s_theo) else 0.0) - shiftS_base

        # ---------------- P Top ----------------
        zP_syn_top = extract_window_time_with_padding(syn_used['Z'].data, dt_syn, cP_top, P_BEFORE, P_AFTER)
        rP_syn_top = extract_window_time_with_padding(syn_used['R'].data, dt_syn, cP_top, P_BEFORE, P_AFTER)
        p_obs_top  = np.concatenate([zP_obs, 0.5 * rP_obs])
        p_syn_top  = np.concatenate([
            resample_to_dt(zP_syn_top, dt_syn, dt_obs),
            0.5 * resample_to_dt(rP_syn_top, dt_syn, dt_obs)
        ])
        misP_top.append(misfit_metric(p_obs_top, p_syn_top, f"{ev_id}_P_d{int(d)}"))

        # ---------------- S Top ----------------
        zS_syn_top = extract_window_time_with_padding(syn_used['Z'].data, dt_syn, cS_top, S_BEFORE, S_AFTER)
        tS_syn_top = extract_window_time_with_padding(syn_used['T'].data, dt_syn, cS_top, S_BEFORE, S_AFTER)
        s_obs_top  = np.concatenate([zS_obs, tS_obs_arr])
        s_syn_top  = np.concatenate([
            resample_to_dt(zS_syn_top, dt_syn, dt_obs),
            resample_to_dt(tS_syn_top, dt_syn, dt_obs)
        ])
        # fliphack pra ambiguidade de polaridade
        cost_s_norm = misfit_metric(s_obs_top,  s_syn_top,  f"{ev_id}_S_d{int(d)}")
        cost_s_flip = misfit_metric(s_obs_top, -s_syn_top, f"{ev_id}_S_d{int(d)}_flip")
        misS_top.append(min(cost_s_norm, cost_s_flip))

        # ---------------- P Base ----------------
        zP_syn_base = extract_window_time_with_padding(syn_used['Z'].data, dt_syn, cP_base, P_BEFORE, P_AFTER)
        rP_syn_base = extract_window_time_with_padding(syn_used['R'].data, dt_syn, cP_base, P_BEFORE, P_AFTER)
        p_syn_base  = np.concatenate([
            resample_to_dt(zP_syn_base, dt_syn, dt_obs),
            0.5 * resample_to_dt(rP_syn_base, dt_syn, dt_obs)
        ])
        misP_base.append(misfit_metric(p_obs_top, p_syn_base, f"{ev_id}_Pbase_d{int(d)}"))

        # ---------------- S Base ----------------
        zS_syn_base = extract_window_time_with_padding(syn_used['Z'].data, dt_syn, cS_base, S_BEFORE, S_AFTER)
        tS_syn_base = extract_window_time_with_padding(syn_used['T'].data, dt_syn, cS_base, S_BEFORE, S_AFTER)
        s_syn_base  = np.concatenate([
            resample_to_dt(zS_syn_base, dt_syn, dt_obs),
            resample_to_dt(tS_syn_base, dt_syn, dt_obs)
        ])
        cost_s_norm_b = misfit_metric(s_obs_top,  s_syn_base,  f"{ev_id}_Sbase_d{int(d)}")
        cost_s_flip_b = misfit_metric(s_obs_top, -s_syn_base, f"{ev_id}_Sbase_d{int(d)}_flip")
        misS_base.append(min(cost_s_norm_b, cost_s_flip_b))

    return (
        depths,
        np.asarray(misP_top),  np.asarray(misS_top),
        np.asarray(misP_base), np.asarray(misS_base)
    )


# ==============================================================================
# Objective and inversion
# ==============================================================================
_cost_history: List[float] = []
_eval_history: List[int] = []
_EVENTCOST_LOG: List[Dict[str, Any]] = []
_eval_counter: int = 0

def evaluate_theta(theta: np.ndarray,
                   obs_events: Dict[str, Dict[str, Trace]],
                   arrivals: Dict[str, Dict[str, Any]],
                   events: List[str]) -> Tuple[float, List[Tuple[str, float]]]:

    vp_top, vp_bot, vs_top, vs_bot, d1000, w1000, d_layer = map(float, theta)

    mp = ModelParam(vp_top, vp_bot, vs_top, vs_bot, d1000, w1000, d_layer)
    vp_prof, vs_prof = mp.to_profiles(Z_GRID)

    per_event: List[Tuple[str, float]] = []

    for ev_id in events:
        obs_trs = obs_events[ev_id]
        arr = arrivals[ev_id]

        # profundidade da fonte a partir da tabela
        src_depth = float(arr.get("depth", np.nan))
        if not np.isfinite(src_depth):
            logger.warning("Evento %s sem 'depth' válido em arrivals; pulando.", ev_id)
            continue

        # janela de varredura em profundidade ao redor da fonte (ex.: ±15 km)
        scan_half = 15.0  # ajuste se quiser varrer mais/menos
        depth_min = max(0.0, src_depth - scan_half)
        depth_max = src_depth + scan_half
        depth_step = 1.0

        # forward (usar os sintéticos gerados para ESTE theta e ESTE evento)
        try:
            syn_trs = synthesize_event_traces(Z_GRID, vp_prof, vs_prof, arr)
        except Exception as e:
            logger.warning("Forward failed for event %s: %s", ev_id, e)
            return 999.0, []

        # misfit vs depth (sem regerar sintéticos internamente)
        try:
            depths, mP_top, mS_top, mP_base, mS_base = compute_misfits_vs_depth(
                ev_id,
                obs_trs,
                syn_trs,
                arr,
                depth_range=(depth_min, depth_max),
                step=depth_step,
                regen_synthetics=False,
                vp_prof=vp_prof,
                vs_prof=vs_prof
            )
        except Exception as e:
            logger.warning("Misfit computation failed for %s: %s", ev_id, e)
            return 999.0, []

        # custo do evento = média entre (P,S) e (top,base)
        mean_top = 0.5 * (np.nanmean(mP_top) + np.nanmean(mS_top))
        mean_base = 0.5 * (np.nanmean(mP_base) + np.nanmean(mS_base))
        c_ev = float(np.nanmean([mean_top, mean_base]))
        per_event.append((ev_id, c_ev))

    val = float(np.nanmean([c for _, c in per_event])) if per_event else 999.0

    import csv, os
    csv_path = "/home/lyara/areswave/outputs/event_costs.csv"
    header = ["theta", "event", "cost"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        for ev, c_ev in per_event:
            writer.writerow([theta.tolist(), ev, c_ev])

    return val, per_event


def run_inversion(max_iter: int = MAX_ITER, popsize: int = POPSIZE, seed: int | None = SEED) -> Dict[str, Any]:
    """
    Versão única e final do run_inversion.
    Executa a inversão completa multi-evento, usa evaluate_theta() e
    devolve os perfis, custos e parâmetros compatíveis com os plots.
    """
    ensure_dirs()

    arrivals = load_arrivals(ARRIVALS_FILE)
    obs_events = group_traces_by_event(OBS_DIR)
    common = sorted(set(obs_events.keys()) & set(arrivals.keys()))
    if not common:
        raise RuntimeError("No common events between obs and arrivals.csv")

    # limites de busca (inclui camada profunda)
    bounds = [
        (PB.vp_min, PB.vp_max),
        (PB.vp_min, PB.vp_max),
        (PB.vs_min, PB.vs_max),
        (PB.vs_min, PB.vs_max),
        (PB.d1000_min, PB.d1000_max),
        (PB.grad_min, PB.grad_max),
        (1500.0, 1700.0)
    ]

    history: List[Tuple[np.ndarray, float]] = []

    def obj(theta):
        theta = np.array(theta, dtype=float)
        logger.debug("θ = " + ", ".join(f"{v:.3f}" for v in theta))
        val, _ = evaluate_theta(theta, obs_events, arrivals, common)
        history.append((theta.copy(), float(val)))
        return val

    logger.info("Events used: %s", ", ".join(common))
    logger.info("DE config: max_iter=%s, popsize=%s, seed=%s", max_iter, popsize, seed)

    t0 = time.time()
    res = differential_evolution(
        obj, bounds=bounds, maxiter=max_iter, popsize=popsize,
        strategy='best1bin', updating='immediate', workers=1,
        polish=True, seed=seed
    )
    elapsed = time.time() - t0
    logger.info("Total wall time: %.1fs", elapsed)
    logger.info("Best cost: %.6f", float(res.fun))
    logger.info("Best θ: %s", ", ".join(f"{v:.4f}" for v in res.x))

    # histórico
    thetas = np.array([t for t, _ in history], dtype=float)
    costs  = np.array([c for _, c in history], dtype=float)

    vp_ens, vs_ens = [], []
    for th in thetas:
        vp_top, vp_bot, vs_top, vs_bot, d1000, w1000, d_layer = th
        mp = ModelParam(vp_top, vp_bot, vs_top, vs_bot, d1000, w1000, d_layer)
        vp_prof, vs_prof = mp.to_profiles(Z_GRID)
        vp_ens.append(vp_prof)
        vs_ens.append(vs_prof)
    vp_ens = np.asarray(vp_ens)
    vs_ens = np.asarray(vs_ens)

    # melhor modelo
    theta_best = list(map(float, res.x))
    mp_best = ModelParam(vp_top=theta_best[0], vp_bot=theta_best[1],
                     vs_top=theta_best[2], vs_bot=theta_best[3],
                     d1000=theta_best[4], w1000=theta_best[5], d_layer=theta_best[6])
    vp_best, vs_best = mp_best.to_profiles(Z_GRID)
    logger.debug(f"vp_prof[min,max]={vp_prof.min():.3f},{vp_prof.max():.3f}  "f"vs_prof[min,max]={vs_prof.min():.3f},{vs_prof.max():.3f}")

    # salva CSV do ensemble
    try:
        df = pd.DataFrame(thetas, columns=[
            "vp_top", "vp_bot", "vs_top", "vs_bot", "d1000", "w1000", "d_layer"
        ])
        df["cost"] = costs
        df["model_id"] = np.arange(1, len(costs) + 1)
        df["elapsed_s"] = elapsed
        df["run_seed"] = seed if seed is not None else np.nan
        df.to_csv(OUTPUT_ENSEMBLE, index=False)
        logger.info(f"Saved ensemble results to: {OUTPUT_ENSEMBLE}")
    except Exception as e:
        logger.warning(f"Could not export ensemble CSV: {e}")

    return {
        "z_grid": Z_GRID,
        "vp_best": vp_best,
        "vs_best": vs_best,
        "vp_ens": vp_ens,
        "vs_ens": vs_ens,
        "ens_costs": costs,
        "theta_best": theta_best,
        "theta_ens": thetas,
        "bounds": bounds,
        "elapsed_s": elapsed,
    }


# ==============================================================================
# Plotting
# ==============================================================================

def plot_fig3_style(inv_out: Dict[str, Any], out_png: str):
    """
    Plota perfis de Vp e Vs em estilo Fig.3 (PNAS),
    com média ±σ e percentis ponderados pelo custo,
    incluindo linha da média e colorbar do misfit normalizado.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    z = inv_out['z_grid']
    vp_ens = inv_out['vp_ens']
    vs_ens = inv_out['vs_ens']
    costs = inv_out['ens_costs']

    # ----- Ponderação pelo custo -----
    weights = 1.0 / (costs + 1e-8)
    weights /= np.sum(weights)

    # Média ponderada
    vp_mean = np.average(vp_ens, axis=0, weights=weights)
    vs_mean = np.average(vs_ens, axis=0, weights=weights)

    # Desvio padrão ponderado
    vp_var = np.average((vp_ens - vp_mean) ** 2, axis=0, weights=weights)
    vs_var = np.average((vs_ens - vs_mean) ** 2, axis=0, weights=weights)
    vp_std = np.sqrt(vp_var)
    vs_std = np.sqrt(vs_var)

    # Função auxiliar para percentis ponderados
    def weighted_percentile(data, weights, percent):
        sorter = np.argsort(data)
        data_sorted = data[sorter]
        weights_sorted = weights[sorter]
        cdf = np.cumsum(weights_sorted)
        cdf /= cdf[-1]
        return np.interp(percent / 100.0, cdf, data_sorted)

    vp_p10 = np.array([weighted_percentile(vp_ens[:, i], weights, 10) for i in range(vp_ens.shape[1])])
    vp_p90 = np.array([weighted_percentile(vp_ens[:, i], weights, 90) for i in range(vp_ens.shape[1])])
    vs_p10 = np.array([weighted_percentile(vs_ens[:, i], weights, 10) for i in range(vs_ens.shape[1])])
    vs_p90 = np.array([weighted_percentile(vs_ens[:, i], weights, 90) for i in range(vs_ens.shape[1])])

    # ----- Plots -----
    fig, axs = plt.subplots(1, 4, figsize=(28, 10), sharey=True)
    cmap = plt.get_cmap("Blues")

    # Normaliza custos para barra de cor
    norm_costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs) + 1e-12)

    # Criar colorbar apenas para legenda
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(costs), vmax=np.max(costs)))
    sm.set_array([])

    # Painel 1: Vp média ± σ ponderada
    axs[0].fill_betweenx(z, vp_mean - vp_std, vp_mean + vp_std,
                         color=cmap(0.4), alpha=0.5, label="±1σ")
    axs[0].plot(vp_mean, z, 'k-', lw=2, label="Average")
    axs[0].plot(inv_out['vp_best'], z, 'r--', lw=2, label="Best solution")
    axs[0].invert_yaxis()
    axs[0].set_xlabel('Vp (km/s)')
    axs[0].set_ylabel('Depth (km)')
    axs[0].set_title('Vp: average ± σ')
    axs[0].legend()

    # Painel 2: Vs média ± σ ponderada
    axs[1].fill_betweenx(z, vs_mean - vs_std, vs_mean + vs_std,
                         color=cmap(0.4), alpha=0.5)
    axs[1].plot(vs_mean, z, 'k-', lw=2)
    axs[1].plot(inv_out['vs_best'], z, 'r--', lw=2)
    axs[1].invert_yaxis()
    axs[1].set_xlabel('Vs (km/s)')
    axs[1].set_title('Vs: average ± σ')

    # Painel 3: Vp percentis 10–90% ponderados
    axs[2].fill_betweenx(z, vp_p10, vp_p90,
                         color=cmap(0.6), alpha=0.5, label="P10–P90")
    axs[2].plot(vp_mean, z, 'k-', lw=2, label="Average")
    axs[2].plot(inv_out['vp_best'], z, 'r--', lw=2, label="Best solution")
    axs[2].invert_yaxis()
    axs[2].set_xlabel('Vp (km/s)')
    axs[2].set_title('Vp: P10–P90')
    axs[2].legend()

    # Painel 4: Vs percentis 10–90% ponderados
    axs[3].fill_betweenx(z, vs_p10, vs_p90,
                         color=cmap(0.6), alpha=0.5)
    axs[3].plot(vs_mean, z, 'k-', lw=2)
    axs[3].plot(inv_out['vs_best'], z, 'r--', lw=2)
    axs[3].invert_yaxis()
    axs[3].set_xlabel('Vs (km/s)')
    axs[3].set_title('Vs: P10–P90')

    # Adiciona barra de cor representando custo
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Cost (misfit normalized)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    logger.info("Saved plot with cost-weighted envelopes + colorbar: %s", out_png)


def plot_cost_evolution(out_png: str):
    if not _cost_history:
        return
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(_eval_history, _cost_history, 'o-', ms=3)
    ax.set_xlabel("Number of Evaluations")
    ax.set_ylabel("Cost (1−CC²)")
    ax.set_title("Cost Evolution")
    ax.grid(True, which='both', ls='--', alpha=0.5)
    #ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    logger.info("Saved cost evolution: %s", out_png)

def plot_misfit_subplots(inv_out: Dict[str, Any], out_png: str):
    """
    Ensemble-only diagnostics (no forward recomputation):
    - Vp and Vs envelopes (mean ± 1σ) with best solution
    - Cost vs d1000
    - Cost vs d_layer
    """
    import numpy as np
    import matplotlib.pyplot as plt

    z = inv_out['z_grid']
    vp_ens = inv_out['vp_ens']
    vs_ens = inv_out['vs_ens']
    costs  = inv_out['ens_costs']
    vp_best = inv_out['vp_best']
    vs_best = inv_out['vs_best']
    thetas = inv_out.get('theta_ens', None)

    vp_mean, vp_std = np.nanmean(vp_ens, axis=0), np.nanstd(vp_ens, axis=0)
    vs_mean, vs_std = np.nanmean(vs_ens, axis=0), np.nanstd(vs_ens, axis=0)

    cmin, cmax = np.nanmin(costs), np.nanmax(costs)
    c_norm = (costs - cmin) / (cmax - cmin + 1e-12)

    d1000 = d_layer = None
    if thetas is not None and len(thetas) == len(costs):
        thetas = np.asarray(thetas)
        if thetas.ndim == 2 and thetas.shape[1] >= 7:
            d1000 = thetas[:, 4]
            d_layer = thetas[:, 6]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Vp
    axs[0, 0].fill_betweenx(z, vp_mean - vp_std, vp_mean + vp_std,
                            color="#6aaed6", alpha=0.4)
    axs[0, 0].plot(vp_mean, z, 'k-', lw=2)
    axs[0, 0].plot(vp_best, z, 'r--', lw=2)
    axs[0, 0].invert_yaxis()
    axs[0, 0].set_xlabel('Vp (km/s)'); axs[0, 0].set_ylabel('Depth (km)')
    axs[0, 0].set_title('Vp: average ± σ (ensemble)')

    # Vs
    axs[0, 1].fill_betweenx(z, vs_mean - vs_std, vs_mean + vs_std,
                            color="#98df8a", alpha=0.4)
    axs[0, 1].plot(vs_mean, z, 'k-', lw=2)
    axs[0, 1].plot(vs_best, z, 'r--', lw=2)
    axs[0, 1].invert_yaxis()
    axs[0, 1].set_xlabel('Vs (km/s)')
    axs[0, 1].set_title('Vs: average ± σ (ensemble)')

    # Cost vs d1000
    ax = axs[1, 0]
    if d1000 is not None:
        sc = ax.scatter(d1000, costs, c=c_norm, cmap="magma_r", s=18)
        cb = plt.colorbar(sc, ax=ax); cb.set_label("Normalized cost")
        ax.set_xlabel("d1000 (km)"); ax.set_ylabel("Cost")
        ax.set_title("Cost vs d1000")
    else:
        ax.text(0.5, 0.5, "No d1000 data", ha='center', va='center')
        ax.axis('off')

    # Cost vs d_layer
    ax = axs[1, 1]
    if d_layer is not None:
        sc = ax.scatter(d_layer, costs, c=c_norm, cmap="magma_r", s=18)
        cb = plt.colorbar(sc, ax=ax); cb.set_label("Normalized cost")
        ax.set_xlabel("d_layer (km)"); ax.set_ylabel("Cost")
        ax.set_title("Cost vs d_layer")
    else:
        ax.text(0.5, 0.5, "No d_layer data", ha='center', va='center')
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(out_png, dpi=250)
    plt.close(fig)
    logger.info("Saved ensemble-only misfit subplots: %s", out_png)


def plot_vp_vs_heatmap(inv_out, out_png_prefix="figs/inversion_heatmap"):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    z = inv_out["z_grid"]
    vp_ens, vs_ens = inv_out["vp_ens"], inv_out["vs_ens"]
    costs = inv_out["ens_costs"]

    w = 1.0 / (costs + 1e-8)
    w /= np.sum(w)

    vp_grid = np.linspace(PB.vp_min, PB.vp_max, 100)
    vs_grid = np.linspace(PB.vs_min, PB.vs_max, 100)

    misfit_vp = np.zeros((len(z), len(vp_grid)))
    misfit_vs = np.zeros((len(z), len(vs_grid)))

    for i in range(len(z)):
        misfit_vp[i, :] = griddata(vp_ens[:, i], costs, vp_grid, method='linear', fill_value=np.nan)
        misfit_vs[i, :] = griddata(vs_ens[:, i], costs, vs_grid, method='linear', fill_value=np.nan)

    def _norm(x):
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)
    misfit_vp, misfit_vs = _norm(misfit_vp), _norm(misfit_vs)

    fig, axs = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    cmap = "magma_r"
    levels = np.linspace(0, 1, 11)

    im1 = axs[0].imshow(misfit_vp, origin="upper", aspect="auto", cmap=cmap,
                        extent=[vp_grid.min(), vp_grid.max(), z.max(), z.min()])
    axs[0].contour(misfit_vp, levels, colors='white', linewidths=0.5,
                   extent=[vp_grid.min(), vp_grid.max(), z.max(), z.min()])
    axs[0].plot(inv_out["vp_best"], z, "r--", lw=2)
    axs[0].set_xlabel("Vp (km/s)"); axs[0].set_ylabel("Depth (km)")
    axs[0].set_title("Normalized misfit map — Vp")
    fig.colorbar(im1, ax=axs[0], label="Normalized misfit")

    im2 = axs[1].imshow(misfit_vs, origin="upper", aspect="auto", cmap=cmap,
                        extent=[vs_grid.min(), vs_grid.max(), z.max(), z.min()])
    axs[1].contour(misfit_vs, levels, colors='white', linewidths=0.5,
                   extent=[vs_grid.min(), vs_grid.max(), z.max(), z.min()])
    axs[1].plot(inv_out["vs_best"], z, "r--", lw=2)
    axs[1].set_xlabel("Vs (km/s)")
    axs[1].set_title("Normalized misfit map — Vs")
    fig.colorbar(im2, ax=axs[1], label="Normalized misfit")

    plt.tight_layout()
    fpath = f"{out_png_prefix}_vp_vs_heatmap.png"
    fig.savefig(fpath, dpi=400, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved styled misfit heatmaps: {fpath}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Multi-event Vp/Vs inversion on Mars (Fig.3-style)")
    p.add_argument("--max-iter", type=int, default=MAX_ITER, help="DE iterations")
    p.add_argument("--popsize",  type=int, default=POPSIZE,  help="DE population size")
    p.add_argument("--seed",     type=int, default=SEED,     help="Random seed")
    p.add_argument("--no-plots", action="store_true", help="Skip figures (only CSVs)")
    p.add_argument("--debug",    action="store_true", help="Verbose logs (θ and per-event costs)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    ensure_dirs()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")

    try:
        inv = run_inversion(max_iter=args.max_iter, popsize=args.popsize, seed=args.seed)
        theta_best = inv["theta_best"]  # array de 7 parâmetros
        mp_best = ModelParam(
            vp_top=theta_best[0], vp_bot=theta_best[1],
            vs_top=theta_best[2], vs_bot=theta_best[3],
            d1000=theta_best[4], w1000=theta_best[5], d_layer=theta_best[6]
        )

        with open(OUTPUT_MODEL, "w") as f:
            json.dump({
                "theta_best": list(theta_best),
                "model": {
                    "vp_top": mp_best.vp_top, "vp_bot": mp_best.vp_bot,
                    "vs_top": mp_best.vs_top, "vs_bot": mp_best.vs_bot,
                    "d1000": mp_best.d1000, "w1000": mp_best.w1000,
                    "d_layer": mp_best.d_layer
                }
            }, f, indent=2)
        logger.info(f"Saved best model to: {OUTPUT_MODEL}")
        print("[i] Best model JSON:", OUTPUT_MODEL)

        if not args.no_plots:
            plot_fig3_style(inv, os.path.join(FIGS_DIR, "inversion_fig3_style.png"))
            plot_misfit_subplots(inv, os.path.join(FIGS_DIR, "inversion_misfit_subplots.png"))
            plot_cost_evolution(os.path.join(FIGS_DIR, "inversion_cost_evolution.png"))
            plot_vp_vs_heatmap(inv, os.path.join(FIGS_DIR, "inversion"))

        print("[✓] Inversion finished.")
        print("[i] Ensemble CSV:", OUTPUT_ENSEMBLE)
        print("[i] Per-event costs CSV:", OUTPUT_EVENTCOSTS)
        print("[i] Best cost CSV:", OUTPUT_CSV)

        # --- build best model after inversion ---
        theta_best = inv.get("best_theta") if isinstance(inv, dict) and "best_theta" in inv else None
        if theta_best is not None:
            mp_best = ModelParam(vp_top=theta_best[0],
                                 vp_bot=theta_best[1],
                                 vs_top=theta_best[2],
                                 vs_bot=theta_best[3],
                                 d1000=theta_best[4],
                                 w1000=theta_best[5],
                                 d_layer=theta_best[6])

            vp_prof_best, vs_prof_best = mp_best.to_profiles(Z_GRID)
            np.savez(OUTPUT_MODEL, z=Z_GRID, vp=vp_prof_best, vs=vs_prof_best)
            print("[✓] Saved best model to:", OUTPUT_MODEL)

    except Exception as e:
        logger.exception("Inversion error: %s", e)
        print("[!] Inversion error:", e)
