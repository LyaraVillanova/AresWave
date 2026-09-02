import os
import re
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from obspy import Trace, Stream, UTCDateTime, read
from scipy.interpolate import interp1d
from synthetics_function import generate_synthetics, apply_filter
from denoising import polarization_filter
from dsmpy.seismicmodel_Mars import SeismicModel
from dsmpy.station_Mars import Station
from dsmpy.event_Mars import Event, MomentTensor

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# PATHS / CONFIG
# ============================================================
DATA_DIR = "/home/lyara/areswave/SAC5/"
CSV_FILE = "data/arrivals_S1000a.csv"
MODELS_DIR = "/home/lyara/areswave/models"
OUTPUT_DIR = "/home/lyara/areswave/outputs_modelsdir/"
FIG_DIR = "/home/lyara/areswave/figs_modelsdir/"
FS = 20.0
DT = 1.0 / FS
MAX_SHIFT_S = 5.0
MAX_SHIFT_SAMPLES = max(1, int(round(MAX_SHIFT_S / DT)))

DEPTH_START_KM = 700.0
DEPTH_END_KM = 1800.0
CORE_RADIUS_KM = 1830.0
ALLOW_BELOW_CMB = True
SURFACE_RADIUS_FALLBACK_KM = 3389.5
SUITE_SELECTION_MODE = "numeric_families"   # "numeric_families", "all_models", "reference_only"
LOAD_BM_FILES = False

DEFAULT_MINOR_DISC_KM = 800.0
DEFAULT_MINOR_WIDTH_KM = 35.0
DEFAULT_MAIN_WIDTH_KM = 60.0
MTZ_TARGET_DEPTHS_KM = (800.0, 1000.0, 1200.0, 1400.0, 1600.0)
MTZ_HALF_WINDOW_KM = 100.0

P_WINDOW = (-5.0, 8.0)
S_WINDOW = (-5.0, 20.0)
S_WINDOW_EXTENDED_EVENTS = {"S0185a"}
S_WINDOW_EXTENDED = (-5.0, 30.0)

COMPONENT_MODE = "paper"   # opções: "paper", "zr_zt"
S_WEIGHT = 2.0

COMP_MARKERS = {
    "DW85": "o",
    "EH45": "s",
    "KC08": "^",
    "LF97": "D",
    "TAY13": "P",
    "YM20": "X",
    "UNK": "o",
}
ADIABAT_SIZE = {
    0.125: 35,
    0.150: 65,
    0.175: 95,
}

# ============================================================
# MONKEY PATCHES / PROFILE TO MODEL
# ============================================================
def _update_mantle(
    self,
    vpv, vph, vsv, vsh,
    depth_min=700.0, depth_max=1800.0,
    core_radius_km=CORE_RADIUS_KM,
    allow_below_cmb=ALLOW_BELOW_CMB,
):
    """
    Atualiza apenas as camadas que interceptam a janela [depth_min, depth_max],
    preservando a estrutura rasa do modelo base.

    Esta rotina segue a mesma lógica do script original do usuário, que já vinha
    funcionando com DSMpy.
    """
    try:
        vrmin = self._vrmin
        vrmax = self._vrmax
        R = float(vrmax[-1])

        if (core_radius_km is not None) and (not allow_below_cmb):
            try:
                depth_cmb = float(R - float(core_radius_km))
            except Exception:
                depth_cmb = None
        else:
            depth_cmb = None

        dmin = float(min(depth_min, depth_max))
        dmax = float(max(depth_min, depth_max))

        if (depth_cmb is not None) and (not allow_below_cmb):
            if dmin >= depth_cmb:
                return
            dmax = min(dmax, depth_cmb)

        r_shallow = R - dmin
        r_deep = R - dmax
        r_low = min(r_shallow, r_deep)
        r_high = max(r_shallow, r_deep)

        for i, (zmin, zmax) in enumerate(zip(vrmin, vrmax)):
            if (zmax >= r_low) and (zmin <= r_high):
                if np.ndim(vpv) == 0:
                    self._vpv[:, i] = np.full_like(self._vpv[:, i], float(vpv))
                    self._vph[:, i] = np.full_like(self._vph[:, i], float(vph))
                    self._vsv[:, i] = np.full_like(self._vsv[:, i], float(vsv))
                    self._vsh[:, i] = np.full_like(self._vsh[:, i], float(vsh))
                else:
                    self._vpv[:, i] = vpv
                    self._vph[:, i] = vph
                    self._vsv[:, i] = vsv
                    self._vsh[:, i] = vsh
    except Exception as e:
        logger.error(f"Erro em _update_mantle (depth {depth_min}-{depth_max} km): {e}")
        raise


setattr(SeismicModel, "update_mantle", _update_mantle)


def _surface_radius_from_model(model: SeismicModel) -> float:
    try:
        return float(model._vrmax[-1])
    except Exception:
        return float(SURFACE_RADIUS_FALLBACK_KM)


def _depth_centers_from_vr(vrmin: np.ndarray, vrmax: np.ndarray) -> np.ndarray:
    R = float(vrmax[-1])
    r_ctr = 0.5 * (np.asarray(vrmin, dtype=float) + np.asarray(vrmax, dtype=float))
    return (R - r_ctr).astype(float)


def _coerce_profile_1d(a) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(a, dtype=float)
    except Exception:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size >= 2:
        return arr.astype(float)
    if arr.ndim == 2 and min(arr.shape) >= 2:
        # arrays DSMpy-style: (4, n_layers) -> média radial/anisotrópica por camada
        axis = 0 if arr.shape[0] <= arr.shape[1] else 1
        out = np.nanmean(arr, axis=axis)
        out = np.asarray(out, dtype=float).squeeze()
        if out.ndim == 1 and out.size >= 2:
            return out.astype(float)
    return None


# ============================================================
# DATA CLASS
# ============================================================
@dataclass
class SuiteModel:
    model_id: str
    source_path: str
    source_format: str
    composition: str
    tpot_K: float
    adiabat_K_km: float
    d_minor_km: float
    w_minor_km: float
    d_main_km: float
    w_main_km: float
    z_km: np.ndarray
    vp_kms: np.ndarray
    vs_kms: np.ndarray
    source_depth_km: float = 30.0
    liquid_start_km: float = np.nan
    d_800_km: float = np.nan
    w_800_km: float = np.nan
    d_1000_km: float = np.nan
    w_1000_km: float = np.nan
    d_1200_km: float = np.nan
    w_1200_km: float = np.nan
    d_1400_km: float = np.nan
    w_1400_km: float = np.nan
    d_1600_km: float = np.nan
    w_1600_km: float = np.nan


# ============================================================
# IO / EVENTS
# ============================================================
def load_event_catalog(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    logger.info(f"{len(df)} eventos carregados de {csv_path}")
    return df



def load_observed_waveforms(event_id: str) -> Optional[Stream]:
    base_event_path = os.path.join(DATA_DIR, event_id)
    if not os.path.exists(base_event_path):
        base_event_path = DATA_DIR

    st = Stream()
    for fname in os.listdir(base_event_path):
        if fname.lower().endswith(".sac"):
            st += read(os.path.join(base_event_path, fname))

    if len(st) == 0:
        logger.warning(f"[{event_id}] Nenhum SAC encontrado em {base_event_path}")
        return None

    _sanitize_stream_inplace(st, f"{event_id}/raw")
    st.detrend("demean")
    st.filter("bandpass", freqmin=0.3, freqmax=0.9, zerophase=True)
    _sanitize_stream_inplace(st, f"{event_id}/filtered")
    return st




def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _sanitize_trace_data(data: np.ndarray, *, fill_value: float = 0.0) -> np.ndarray:
    arr = np.asarray(data, dtype=float).copy()
    if arr.ndim != 1:
        arr = np.ravel(arr).astype(float)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        return np.full(arr.shape, fill_value, dtype=float)
    if idx.size == 1:
        return np.full(arr.shape, arr[idx[0]], dtype=float)
    arr[~finite] = np.interp(np.flatnonzero(~finite), idx, arr[finite])
    return arr

def _sanitize_stream_inplace(st: Stream, label: str = "") -> int:
    fixed = 0
    for tr in st:
        bad = int(np.size(tr.data) - np.count_nonzero(np.isfinite(tr.data)))
        if bad > 0:
            tr.data = _sanitize_trace_data(tr.data)
            fixed += bad
    if fixed > 0:
        who = f"[{label}] " if label else ""
        logger.warning(f"{who}{fixed} amostras não finitas foram interpoladas/substituídas antes do misfit.")
    return fixed

def _surface_radius_from_base_model(base_model: SeismicModel) -> float:
    """Obtém o raio de superfície (km) do modelo-base de forma robusta."""
    for attr in ("radius", "_vrmax", "vrmax", "_vrmin", "vrmin"):
        vals = getattr(base_model, attr, None)
        if vals is None:
            continue
        arr = np.asarray(vals, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return float(np.nanmax(arr))
    logger.warning(
        "Raio de superfície não encontrado explicitamente no modelo-base; usando fallback Mars 3389.5 km."
    )
    return 3389.5

def classify_model_family(stem: str) -> str:
    s = str(stem).strip()
    sl = s.lower()
    if re.match(r"^geophysical_model\d+$", sl):
        return "Geophysical"
    if re.match(r"^md_model\d+$", sl):
        return "MD"
    if re.match(r"^ak_model_\d+$", sl):
        return "AK"
    if re.match(r"^cd_model\d+$", sl):
        return "CD"
    if sl.startswith("kks21gp"):
        return "KKS21GP"
    if sl.startswith("gudkova"):
        return "Gudkova"
    if sl.startswith("ceylan"):
        return "ceylan"
    if sl.startswith("dwak"):
        return "DWAK"
    if sl.startswith("maak"):
        return "MAAK"
    if sl in {"prem", "ak135", "x", "tayak", "sanak", "zgdw", "lfak"}:
        return "reference"
    return "other"

def model_is_selected(stem: str, mode: str = SUITE_SELECTION_MODE) -> bool:
    fam = classify_model_family(stem)
    mode = str(mode).lower()
    if mode == "all_models":
        return True
    if mode == "reference_only":
        return fam not in {"Geophysical", "MD", "AK", "CD", "other"}
    # default: suítes numeradas + modelos-base úteis do paper
    return fam in {"Geophysical", "MD", "AK", "CD", "KKS21GP", "Gudkova", "ceylan", "DWAK", "MAAK"}

def _parse_filename_metadata(path: str) -> Dict[str, float | str]:
    stem = Path(path).stem
    upper = stem.upper()
    comp = "UNK"
    for token in ("DW85", "EH45", "KC08", "LF97", "TAY13", "YM20"):
        if token in upper:
            comp = token
            break

    def _grab(patterns, default=np.nan):
        for pat in patterns:
            m = re.search(pat, stem, flags=re.IGNORECASE)
            if m:
                return _safe_float(m.group(1), default)
        return float(default)

    tpot = _grab([r"(?:^|[_-])T(?:POT)?[_-]?(\d+(?:\.\d+)?)", r"(?:^|[_-])TP[_-]?(\d+(?:\.\d+)?)"])
    adiabat = _grab([r"(?:^|[_-])G[_-]?(\d+(?:\.\d+)?)", r"adiabat[_-]?(\d+(?:\.\d+)?)", r"grad[_-]?(\d+(?:\.\d+)?)"])
    if np.isfinite(adiabat) and adiabat > 2.0:
        adiabat = adiabat / 1000.0
    dmain = _grab([r"(?:^|[_-])D(?:MAIN)?[_-]?(\d+(?:\.\d+)?)", r"disc[_-]?(\d+(?:\.\d+)?)"])
    wmain = _grab([r"(?:^|[_-])W(?:MAIN)?[_-]?(\d+(?:\.\d+)?)", r"width[_-]?(\d+(?:\.\d+)?)"], default=DEFAULT_MAIN_WIDTH_KM)

    return {
        "composition": comp,
        "tpot_K": tpot,
        "adiabat_K_km": adiabat,
        "d_main_km": dmain,
        "w_main_km": wmain,
    }

def _coerce_1d(a) -> Optional[np.ndarray]:
    return _coerce_profile_1d(a)

def _find_matching_array(mapping: Dict[str, np.ndarray], aliases: Sequence[str]) -> Optional[np.ndarray]:
    lower_map = {str(k).lower(): v for k, v in mapping.items()}
    for alias in aliases:
        if alias.lower() in lower_map:
            arr = _coerce_1d(lower_map[alias.lower()])
            if arr is not None:
                return arr
    for alias in aliases:
        for k, v in lower_map.items():
            if alias.lower() in k:
                arr = _coerce_1d(v)
                if arr is not None:
                    return arr
    return None

def _infer_depth_axis(x: np.ndarray, surface_radius_km: float) -> Tuple[np.ndarray, str]:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("Eixo de profundidade/radius insuficiente.")
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    first = float(x[0])
    last = float(x[-1])
    diffs = np.diff(x)
    inc = bool(np.all(diffs >= -1e-6))
    dec = bool(np.all(diffs <= 1e-6))
    near0_first = abs(first) <= 10.0
    near0_last = abs(last) <= 10.0
    nearR_first = abs(first - surface_radius_km) <= 25.0
    nearR_last = abs(last - surface_radius_km) <= 25.0

    # Convenção dominante dos .nd marcianos desta suíte: profundidade explícita 0 -> R.
    if inc and near0_first and nearR_last:
        return x.astype(float), "depth"
    # Convenção clássica em raio: superfície -> centro (R -> 0).
    if dec and nearR_first and near0_last:
        depth = surface_radius_km - x
        return depth.astype(float), "radius"

    # Alguns arquivos .nd usam raio em megâmetros (0.0 ... 3.3895) em vez de km.
    if xmax <= 10.0:
        x_mm = x * 1000.0
        nearR_first_mm = abs(float(x[0]) * 1000.0 - surface_radius_km) <= max(5.0, 0.02 * surface_radius_km)
        nearR_last_mm = abs(float(x[-1]) * 1000.0 - surface_radius_km) <= max(5.0, 0.02 * surface_radius_km)
        if (dec and nearR_first_mm) or ((not dec) and nearR_last_mm):
            depth = surface_radius_km - x_mm
            return depth.astype(float), "radius_Mm"

    is_radius_like = (xmax > 2500.0) and (xmin >= 0.0) and nearR_first
    if is_radius_like:
        depth = surface_radius_km - x
        return depth.astype(float), "radius"

    return x.astype(float), "depth"

def _guess_velocity_columns(arr: np.ndarray) -> Tuple[int, int]:
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Arquivo tabular sem colunas suficientes para Vp/Vs.")
    best = None
    for i in range(1, arr.shape[1]):
        for j in range(1, arr.shape[1]):
            if i == j:
                continue
            vp = arr[:, i]
            vs = arr[:, j]
            finite = np.isfinite(vp) & np.isfinite(vs)
            if finite.sum() < 5:
                continue
            vpv = vp[finite]
            vsv = vs[finite]
            score = 0.0
            score += np.mean((vpv > 3.0) & (vpv < 12.5))
            score += np.mean((vsv > 1.0) & (vsv < 8.5))
            score += np.mean(vpv > vsv)
            score += 0.1 * np.std(vpv)
            score += 0.1 * np.std(vsv)
            item = (score, i, j)
            if (best is None) or (item[0] > best[0]):
                best = item

    if best is None:
        raise ValueError("Não consegui inferir colunas de Vp e Vs.")
    return int(best[1]), int(best[2])

def _clean_and_clip_profile(z_km: np.ndarray, vp_kms: np.ndarray, vs_kms: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(z_km, dtype=float)
    vp = np.asarray(vp_kms, dtype=float)
    vs = np.asarray(vs_kms, dtype=float)

    ok = np.isfinite(z) & np.isfinite(vp) & np.isfinite(vs)
    z, vp, vs = z[ok], vp[ok], vs[ok]

    if z.size < 2:
        raise ValueError("Perfil insuficiente após limpeza.")

    order = np.argsort(z)
    z, vp, vs = z[order], vp[order], vs[order]

    z_unique, idx = np.unique(z, return_index=True)
    z, vp, vs = z_unique, vp[idx], vs[idx]

    mask = (z >= DEPTH_START_KM) & (z <= DEPTH_END_KM)
    z, vp, vs = z[mask], vp[mask], vs[mask]
    if z.size < 2:
        raise ValueError(f"Perfil não cobre {DEPTH_START_KM:.0f}–{DEPTH_END_KM:.0f} km.")
    return z.astype(float), vp.astype(float), vs.astype(float)

def _infer_liquid_depth_km(z_km: np.ndarray, vs_kms: np.ndarray, thresh: float = 0.05) -> float:
    z = np.asarray(z_km, dtype=float)
    vs = np.asarray(vs_kms, dtype=float)
    mask = np.isfinite(z) & np.isfinite(vs) & (vs <= float(thresh))
    if not np.any(mask):
        return float('nan')
    return float(np.nanmin(z[mask]))

def _estimate_discontinuity_near_target(
    z_km: np.ndarray,
    vs_kms: np.ndarray,
    target_depth_km: float,
    half_window_km: float = MTZ_HALF_WINDOW_KM,
) -> Tuple[float, float]:
    z = np.asarray(z_km, dtype=float)
    vs = np.asarray(vs_kms, dtype=float)
    if z.size < 5:
        return np.nan, float(DEFAULT_MAIN_WIDTH_KM)

    target = float(target_depth_km)
    hw = float(half_window_km)
    mask = np.isfinite(z) & np.isfinite(vs) & (z >= target - hw) & (z <= target + hw)
    if mask.sum() < 5:
        idx_near = int(np.nanargmin(np.abs(z - target)))
        i0 = max(0, idx_near - 2)
        i1 = min(z.size, idx_near + 3)
        mask = np.zeros_like(z, dtype=bool)
        mask[i0:i1] = True
    if mask.sum() < 2:
        return float(target), float(DEFAULT_MAIN_WIDTH_KM)

    z2 = z[mask]
    vs2 = vs[mask]
    order = np.argsort(z2)
    z2 = z2[order]
    vs2 = vs2[order]
    if z2.size < 2:
        return float(target), float(DEFAULT_MAIN_WIDTH_KM)

    g = np.gradient(vs2, z2)
    idx = int(np.argmax(np.abs(g)))
    d_est = float(z2[idx])

    gabs = np.abs(g)
    gmax = float(np.nanmax(gabs)) if gabs.size else 0.0
    thr = 0.5 * gmax if np.isfinite(gmax) else 0.0
    above = np.where(gabs >= thr)[0]
    if above.size >= 2:
        w_est = float(max(10.0, z2[above[-1]] - z2[above[0]]))
    else:
        w_est = float(DEFAULT_MAIN_WIDTH_KM)
    return d_est, w_est

def _estimate_main_discontinuity(z_km: np.ndarray, vs_kms: np.ndarray) -> Tuple[float, float]:
    return _estimate_discontinuity_near_target(z_km, vs_kms, 1000.0)

def _suite_model_from_profile(
    path: str,
    source_format: str,
    z_km: np.ndarray,
    vp_kms: np.ndarray,
    vs_kms: np.ndarray,
    meta_override: Optional[Dict[str, float | str]] = None,
    expected_cmb_depth_km: Optional[float] = None,
) -> SuiteModel:
    z_km, vp_kms, vs_kms = _clean_and_clip_profile(z_km, vp_kms, vs_kms)
    liquid_start_km = _infer_liquid_depth_km(z_km, vs_kms)
    if np.isfinite(liquid_start_km) and expected_cmb_depth_km is not None:
        # Mantém apenas como diagnóstico; não rejeita nesta etapa.
        if liquid_start_km < (float(expected_cmb_depth_km) - 250.0):
            logger.warning(
                f"[{Path(path).stem}] liquid_start={liquid_start_km:.1f} km acima do CMB do modelo-base (~{expected_cmb_depth_km:.1f} km); mantendo para avaliação."
            )

    meta = _parse_filename_metadata(path)
    if meta_override:
        meta.update({k: v for k, v in meta_override.items() if v is not None})

    d_800_est, w_800_est = _estimate_discontinuity_near_target(z_km, vs_kms, 800.0)
    d_1000_est, w_1000_est = _estimate_discontinuity_near_target(z_km, vs_kms, 1000.0)
    d_1200_est, w_1200_est = _estimate_discontinuity_near_target(z_km, vs_kms, 1200.0)
    d_1400_est, w_1400_est = _estimate_discontinuity_near_target(z_km, vs_kms, 1400.0)
    d_1600_est, w_1600_est = _estimate_discontinuity_near_target(z_km, vs_kms, 1600.0)

    d_main = _safe_float(meta.get("d_main_km", np.nan), d_1000_est)
    if not np.isfinite(d_main):
        d_main = d_1000_est
    w_main = _safe_float(meta.get("w_main_km", np.nan), w_1000_est)
    if not np.isfinite(w_main):
        w_main = w_1000_est

    source_depth = _safe_float(meta.get("source_depth_km", 30.0), 30.0)

    return SuiteModel(
        model_id=Path(path).stem,
        source_path=str(path),
        source_format=source_format,
        composition=str(meta.get("composition", "UNK") or "UNK"),
        tpot_K=_safe_float(meta.get("tpot_K", np.nan)),
        adiabat_K_km=_safe_float(meta.get("adiabat_K_km", np.nan)),
        d_minor_km=float(DEFAULT_MINOR_DISC_KM),
        w_minor_km=float(DEFAULT_MINOR_WIDTH_KM),
        d_main_km=float(d_main),
        w_main_km=float(w_main),
        z_km=z_km,
        vp_kms=vp_kms,
        vs_kms=vs_kms,
        source_depth_km=float(source_depth),
        liquid_start_km=float(liquid_start_km) if np.isfinite(liquid_start_km) else np.nan,
        d_800_km=float(d_800_est),
        w_800_km=float(w_800_est),
        d_1000_km=float(d_main),
        w_1000_km=float(w_main),
        d_1200_km=float(d_1200_est),
        w_1200_km=float(w_1200_est),
        d_1400_km=float(d_1400_est),
        w_1400_km=float(w_1400_est),
        d_1600_km=float(d_1600_est),
        w_1600_km=float(w_1600_est),
    )

def load_suite_model_from_npz(path: str, base_model: Optional[SeismicModel]) -> SuiteModel:
    surface_radius_km = _surface_radius_from_base_model(base_model)
    expected_cmb_depth_km = surface_radius_km - float(CORE_RADIUS_KM)
    raw = np.load(path, allow_pickle=True)

    mapping: Dict[str, np.ndarray] = {}
    meta_override: Dict[str, float | str] = {}
    for key in raw.files:
        value = raw[key]
        if np.isscalar(value) or (np.asarray(value).ndim == 0):
            meta_override[key] = np.asarray(value).item()
        else:
            mapping[key] = value

    # suporte a npz com dict serializado
    for key, value in list(mapping.items()):
        arr = np.asarray(value)
        if arr.ndim == 0 and arr.dtype == object:
            try:
                obj = arr.item()
                if isinstance(obj, dict):
                    for k2, v2 in obj.items():
                        if np.isscalar(v2) or (np.asarray(v2).ndim == 0):
                            meta_override[k2] = np.asarray(v2).item()
                        else:
                            mapping[k2] = v2
            except Exception:
                pass

    lower_keys = {str(k).lower(): k for k in mapping.keys()}

    # Caso DSMpy-style: vrmin, vrmax, vpv/vph, vsv/vsh, etc.
    if (('vrmin' in lower_keys) and ('vrmax' in lower_keys) and
        (('vpv' in lower_keys) or ('vph' in lower_keys)) and
        (('vsv' in lower_keys) or ('vsh' in lower_keys))):

        vrmin = np.asarray(mapping[lower_keys['vrmin']], dtype=float).squeeze()
        vrmax = np.asarray(mapping[lower_keys['vrmax']], dtype=float).squeeze()
        if vrmin.ndim != 1 or vrmax.ndim != 1 or vrmin.size != vrmax.size:
            raise ValueError(f"{path}: vrmin/vrmax inválidos no npz.")

        r_ctr = 0.5 * (vrmin + vrmax)
        z_raw = surface_radius_km - r_ctr

        vp_raw = None
        for key in ('vpv', 'vph', 'vp', 'vp_kms'):
            if key in lower_keys:
                vp_raw = _coerce_profile_1d(mapping[lower_keys[key]])
                if vp_raw is not None:
                    break
        vs_raw = None
        for key in ('vsv', 'vsh', 'vs', 'vs_kms'):
            if key in lower_keys:
                vs_raw = _coerce_profile_1d(mapping[lower_keys[key]])
                if vs_raw is not None:
                    break

        if z_raw is None or vp_raw is None or vs_raw is None:
            raise ValueError(f"{path}: npz DSMpy-style incompleto para z/vp/vs.")

        return _suite_model_from_profile(path, "npz", z_raw, vp_raw, vs_raw, meta_override=meta_override)

    z_raw = _find_matching_array(mapping, ["z_km", "depth_km", "depth", "z", "radius_km", "r_km", "radius", "r"])
    vp_raw = _find_matching_array(mapping, ["vp_kms", "vp", "vpv", "vph"])
    vs_raw = _find_matching_array(mapping, ["vs_kms", "vs", "vsv", "vsh"])

    if z_raw is None:
        candidates = []
        for key, value in mapping.items():
            arr = _coerce_profile_1d(value)
            if arr is None:
                continue
            if np.all(np.diff(arr) >= 0) or np.all(np.diff(arr) <= 0):
                candidates.append((key, arr))
        if candidates:
            z_raw = candidates[0][1]

    if z_raw is None or vp_raw is None or vs_raw is None:
        if "v_mod" in raw.files or "v_mod.layers" in raw.files:
            try:
                layers = None
                if "v_mod" in raw.files:
                    try:
                        v_mod = raw["v_mod"].item()
                        layers = getattr(v_mod, "layers", None)
                    except Exception:
                        layers = None
                if layers is None and "v_mod.layers" in raw.files:
                    layers = raw["v_mod.layers"]
                if layers is None:
                    raise ValueError("TauP npz sem layers utilizáveis")
                z_raw = np.concatenate([
                    np.asarray(layers["top_depth"], dtype=float),
                    np.asarray(layers["bot_depth"], dtype=float)[-1:]
                ])
                vp_raw = np.concatenate([
                    np.asarray(layers["top_p_velocity"], dtype=float),
                    np.asarray(layers["bot_p_velocity"], dtype=float)[-1:]
                ])
                vs_raw = np.concatenate([
                    np.asarray(layers["top_s_velocity"], dtype=float),
                    np.asarray(layers["bot_s_velocity"], dtype=float)[-1:]
                ])
            except Exception as exc:
                raise ValueError(f"{path}: falha ao extrair z/vp/vs do TauP npz: {exc}") from exc

    if z_raw is None or vp_raw is None or vs_raw is None:
        raise ValueError(f"{path}: não consegui identificar z/depth, vp e vs no npz.")

    z_km, axis_kind = _infer_depth_axis(z_raw, surface_radius_km)
    if axis_kind == "radius":
        logger.info(f"[{Path(path).name}] eixo interpretado como radius -> depth.")

    return _suite_model_from_profile(
        path, "npz", z_km, vp_raw, vs_raw,
        meta_override=meta_override,
        expected_cmb_depth_km=expected_cmb_depth_km,
    )

def load_suite_model_from_bm(path: str, base_model: Optional[SeismicModel]) -> SuiteModel:
    surface_radius_km = _surface_radius_from_base_model(base_model)
    expected_cmb_depth_km = surface_radius_km - float(CORE_RADIUS_KM)
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            vals = []
            ok = True
            for tok in toks:
                tok2 = tok.replace("D", "E").replace("d", "e")
                try:
                    vals.append(float(tok2))
                except Exception:
                    ok = False
                    break
            if ok and len(vals) >= 4:
                rows.append(vals[:4])

    if len(rows) < 3:
        raise ValueError(f"{path}: poucas linhas numéricas legíveis em .bm")

    arr = np.asarray(rows, dtype=float)
    radius_raw = arr[:, 0]
    rho_raw = arr[:, 1]
    vp_raw = arr[:, 2]
    vs_raw = arr[:, 3]

    # arquivos .bm costumam estar em metros e m/s; converte para km e km/s
    radius_km = radius_raw.copy()
    if np.nanmax(np.abs(radius_km)) > 1.0e4:
        radius_km = radius_km / 1000.0
    if np.nanmax(np.abs(vp_raw)) > 50.0:
        vp_raw = vp_raw / 1000.0
    if np.nanmax(np.abs(vs_raw)) > 50.0:
        vs_raw = vs_raw / 1000.0

    z_km, axis_kind = _infer_depth_axis(radius_km, surface_radius_km)
    if axis_kind == "radius":
        logger.info(f"[{Path(path).name}] primeira coluna interpretada como radius -> depth (.bm).")

    return _suite_model_from_profile(
        path, "bm", z_km, vp_raw, vs_raw,
        expected_cmb_depth_km=expected_cmb_depth_km,
    )

def load_suite_model_from_nd(path: str, base_model: Optional[SeismicModel]) -> SuiteModel:
    surface_radius_km = _surface_radius_from_base_model(base_model)
    expected_cmb_depth_km = surface_radius_km - float(CORE_RADIUS_KM)
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tokens = s.split()
            vals = []
            ok = True
            for tok in tokens:
                tok2 = tok.replace("D", "E").replace("d", "e")
                try:
                    vals.append(float(tok2))
                except Exception:
                    ok = False
                    break
            if ok and len(vals) >= 3:
                rows.append(vals)

    if len(rows) < 3:
        raise ValueError(f"{path}: poucas linhas numéricas legíveis em .nd")

    arr = np.asarray(rows, dtype=float)

    # Suporte a formato por camadas com raio inferior/superior em Mm (ex.: DWAK.nd)
    if arr.shape[1] >= 4 and np.nanmax(arr[:, :2]) <= 10.0:
        r1 = arr[:, 0] * 1000.0
        r2 = arr[:, 1] * 1000.0
        vp_raw = arr[:, 2]
        vs_raw = arr[:, 3]
        z_vals, vp_vals, vs_vals = [], [], []
        for a, b, vp_i, vs_i in zip(r1, r2, vp_raw, vs_raw):
            za = surface_radius_km - float(a)
            zb = surface_radius_km - float(b)
            z_lo, z_hi = sorted((za, zb))
            z_vals.extend([z_lo, z_hi])
            vp_vals.extend([vp_i, vp_i])
            vs_vals.extend([vs_i, vs_i])
        z_km = np.asarray(z_vals, dtype=float)
        vp_raw = np.asarray(vp_vals, dtype=float)
        vs_raw = np.asarray(vs_vals, dtype=float)
        logger.info(f"[{Path(path).name}] formato por camadas/radius interval interpretado como radius -> depth.")
        return _suite_model_from_profile(
            path, "nd", z_km, vp_raw, vs_raw,
            expected_cmb_depth_km=expected_cmb_depth_km,
        )

    vp_idx, vs_idx = _guess_velocity_columns(arr)
    z_raw = arr[:, 0]
    vp_raw = arr[:, vp_idx]
    vs_raw = arr[:, vs_idx]

    # Prioriza profundidade direta para .nd que sobem de ~0 km até ~R.
    if (
        z_raw.size > 1
        and np.all(np.diff(z_raw) >= -1e-6)
        and abs(float(z_raw[0])) <= 10.0
        and abs(float(z_raw[-1]) - surface_radius_km) <= 25.0
    ):
        z_km = z_raw.copy()
        axis_kind = "depth"
    else:
        z_km, axis_kind = _infer_depth_axis(z_raw, surface_radius_km)
    if axis_kind == "radius":
        logger.info(f"[{Path(path).name}] primeira coluna interpretada como radius -> depth.")

    return _suite_model_from_profile(
        path, "nd", z_km, vp_raw, vs_raw,
        expected_cmb_depth_km=expected_cmb_depth_km,
    )

def load_model_suite_from_models_dir(models_dir: str, base_model: Optional[SeismicModel]) -> List[SuiteModel]:
    model_dir = Path(models_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Pasta de modelos não encontrada: {models_dir}")

    nd_files = sorted(model_dir.rglob("*.nd"))
    npz_files = sorted(model_dir.rglob("*.npz"))
    if not nd_files and not npz_files:
        raise FileNotFoundError(f"Nenhum .nd ou .npz encontrado em {models_dir}")

    # Seleciona apenas as famílias desejadas e evita contar o mesmo modelo duas vezes
    # quando existem .nd e .npz com o mesmo stem. Por padrão, .nd tem prioridade.
    selected: Dict[str, Path] = {}
    for fp in npz_files:
        if model_is_selected(fp.stem):
            selected[fp.stem.lower()] = fp
    for fp in nd_files:
        if model_is_selected(fp.stem):
            selected[fp.stem.lower()] = fp  # .nd tem prioridade

    files = [selected[k] for k in sorted(selected.keys())]

    suite: List[SuiteModel] = []
    failures = []
    for fp in files:
        try:
            if fp.suffix.lower() == ".nd":
                suite.append(load_suite_model_from_nd(str(fp), base_model))
            elif fp.suffix.lower() == ".npz":
                suite.append(load_suite_model_from_npz(str(fp), base_model))
        except Exception as e:
            failures.append((str(fp), str(e)))
            logger.warning(f"Falha ao ler {fp}: {e}")

    if not suite:
        msg = "Não consegui ler nenhum modelo da pasta models/."
        if failures:
            msg += " Primeiro erro: " + failures[0][1]
        raise RuntimeError(msg)

    logger.info(f"Suíte carregada de models/: {len(suite)} modelos válidos; {len(failures)} falhas.")
    return suite

# ============================================================
# SYNTHETICS / PROCESSING
# ============================================================
def synthesize_event_traces(event_row: pd.Series, model: SeismicModel, source_depth_km: float = 30.0) -> Optional[Stream]:
    try:
        mt = MomentTensor(
            Mrr=event_row["mrr"],
            Mtt=event_row["mtt"],
            Mpp=event_row["mpp"],
            Mrt=event_row["mrt"],
            Mrp=event_row["mrp"],
            Mtp=event_row["mtp"],
        )

        try:
            centroid_time = UTCDateTime(event_row.get("time_p", UTCDateTime()))
        except Exception:
            centroid_time = UTCDateTime()

        event = Event(
            event_id=str(event_row["event_id"]),
            latitude=event_row["latitude"],
            longitude=event_row["longitude"],
            depth=float(source_depth_km),
            mt=mt,
            centroid_time=centroid_time,
            source_time_function=None,
        )

        stations = [Station(name="ELYSE", network="XB", latitude=4.502384, longitude=135.623447)]

        output = generate_synthetics(
            event,
            stations,
            model,
            tlen=2000,
            nspc=256,
            sampling_hz=FS,
        )

        us = output.us
        st_syn = Stream()
        for i, comp in enumerate(("Z", "R", "T")):
            tr = Trace(data=us[i, 0, :].astype(float))
            tr.stats.delta = 1.0 / FS
            tr.stats.channel = comp
            st_syn += tr

        for tr in st_syn:
            tr.data = apply_filter(tr.data, FS)

        z, r, t = [tr.data for tr in st_syn]
        zf, rf, tf = polarization_filter([z, r, t], FS)
        for tr, arr in zip(st_syn, [zf, rf, tf]):
            tr.data = np.asarray(arr, dtype=float)

        return st_syn

    except Exception as e:
        logger.error(f"[{event_row['event_id']}] Erro em synthesize_event_traces(): {e}")
        return None

def _extract_single_component(st: Stream, component: str) -> Optional[np.ndarray]:
    sel = st.select(channel=f"*{component}")
    if len(sel) == 0:
        return None
    return np.asarray(sel[0].data, dtype=float)

def _stream_duration_seconds(obs_st: Stream) -> float:
    durations = []
    for tr in obs_st:
        try:
            dt = float(getattr(tr.stats, "delta", DT))
            npts = int(getattr(tr.stats, "npts", len(getattr(tr, "data", []))))
            durations.append(max(0.0, (max(npts, 1) - 1) * dt))
        except Exception:
            continue
    return max(durations) if durations else 0.0

def _estimate_phase_pick_from_waveform(obs_st: Stream, phase: str, p_ref_s: Optional[float] = None) -> Optional[float]:
    phase = str(phase).upper()
    priority = ["Z", "R"] if phase == "P" else ["T", "R", "Z"]
    for comp in priority:
        sel = obs_st.select(channel=f"*{comp}")
        if len(sel) == 0:
            continue
        tr = sel[0]
        data = np.asarray(tr.data, dtype=float)
        if data.size < 8:
            continue
        dt = float(getattr(tr.stats, "delta", DT)) if float(getattr(tr.stats, "delta", DT)) > 0 else DT
        start_idx = 0
        end_idx = data.size
        if phase == "P":
            end_idx = max(8, int(round(0.65 * data.size)))
        else:
            start_idx = int(round((p_ref_s + 20.0) / dt)) if (p_ref_s is not None and np.isfinite(p_ref_s)) else int(round(0.25 * data.size))
            start_idx = max(0, min(start_idx, data.size - 8))
            end_idx = max(start_idx + 8, int(round(0.98 * data.size)))
        work = np.abs(np.nan_to_num(data[start_idx:end_idx], nan=0.0, posinf=0.0, neginf=0.0))
        if work.size == 0:
            continue
        smooth_n = max(1, int(round(5.0 / dt)))
        if smooth_n > 1 and work.size >= smooth_n:
            kern = np.ones(smooth_n, dtype=float) / smooth_n
            work = np.convolve(work, kern, mode="same")
        rel_idx = int(np.argmax(work))
        return float((start_idx + rel_idx) * dt)
    return None

def _coerce_pick_seconds(value, duration_s: float, trace_start=None) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        sec = float(value)
        if np.isfinite(sec) and 0.0 <= sec <= duration_s:
            return sec

        # Alguns CSVs guardam pick como segundo-do-dia (UTC), e nao relativo ao traço.
        if np.isfinite(sec) and trace_start is not None:
            try:
                start_sod = (
                    float(getattr(trace_start, "hour", 0)) * 3600.0
                    + float(getattr(trace_start, "minute", 0)) * 60.0
                    + float(getattr(trace_start, "second", 0))
                    + float(getattr(trace_start, "microsecond", 0)) / 1.0e6
                )
                rel_sec = sec - start_sod
                if np.isfinite(rel_sec) and 0.0 <= rel_sec <= duration_s:
                    return rel_sec
            except Exception:
                pass
    except Exception:
        pass
    if trace_start is not None:
        try:
            sec = float(UTCDateTime(value) - trace_start)
            if np.isfinite(sec) and 0.0 <= sec <= duration_s:
                return sec
        except Exception:
            pass
    return None

def _event_phase_pick_seconds(event_row: pd.Series, phase: str, obs_st) -> Optional[float]:
    if obs_st is None or len(obs_st) == 0:
        return None
    trace_start = obs_st[0].stats.starttime
    duration_s = float(obs_st[0].stats.npts) * float(obs_st[0].stats.delta)

    if phase.upper() == "P":
        for key in ("P_sec", "p_sec", "time_p"):
            pick = _coerce_pick_seconds(event_row.get(key, None), duration_s, trace_start=trace_start)
            if pick is not None:
                return pick
        pick = _estimate_phase_pick_from_waveform(obs_st, "P")
        if pick is not None:
            logger.warning("[%s] pick P fora do traço; usando fallback por waveform em %.1fs.", event_row.get("event_id", "event"), pick)
        return pick

    for key in ("S_sec", "s_sec", "time_s"):
        pick = _coerce_pick_seconds(event_row.get(key, None), duration_s, trace_start=trace_start)
        if pick is not None:
            return pick
    p_pick = _event_phase_pick_seconds(event_row, "P", obs_st)
    pick = _estimate_phase_pick_from_waveform(obs_st, "S", p_ref_s=p_pick)
    if pick is not None:
        logger.warning("[%s] pick S fora do traço; usando fallback por waveform em %.1fs.", event_row.get("event_id", "event"), pick)
    return pick

def _cut_around_peak(tr: Any, peak_idx: int, window_s: Any) -> Optional[np.ndarray]:
    if peak_idx is None:
        return None
    if hasattr(tr, "stats") and hasattr(tr, "data"):
        dt = float(getattr(tr.stats, "delta", 0.0) or 0.0)
        data = tr.data
    else:
        dt = float(DT)
        data = tr
    if dt <= 0.0:
        return None
    data = np.asarray(data, dtype=np.float64)

    if isinstance(window_s, (tuple, list, np.ndarray)) and len(window_s) == 2:
        # Janela no formato (antes, depois), com "antes" podendo vir negativo
        # como nas janelas P/S do script: (-5, 8), (-5, 20), etc.
        before_s = abs(float(window_s[0]))
        after_s = abs(float(window_s[1]))
        n_before = max(1, int(round(before_s / dt)))
        n_after = max(1, int(round(after_s / dt)))
        i0 = max(0, int(peak_idx) - n_before)
        i1 = min(len(data), int(peak_idx) + n_after)
        min_len = max(8, (n_before + n_after) // 2)
    else:
        nwin = max(8, int(round(abs(float(window_s)) / dt)))
        i0 = max(0, int(peak_idx) - nwin // 2)
        i1 = min(len(data), i0 + nwin)
        min_len = max(8, nwin // 2)

    if i1 - i0 < min_len:
        return None
    cut = _sanitize_trace_data(data[i0:i1])
    if cut.size < min_len:
        return None
    if np.allclose(cut, 0.0):
        return None
    return cut.astype(float)

def _cut_window_from_pick(tr: Any, pick_seconds: float, window_s: Tuple[float, float]) -> Optional[np.ndarray]:
    if hasattr(tr, "stats") and hasattr(tr, "data"):
        dt_local = float(getattr(tr.stats, "delta", DT))
        x = np.asarray(tr.data, dtype=float)
    else:
        dt_local = DT
        x = np.asarray(tr, dtype=float)
    if not np.isfinite(pick_seconds):
        return None
    i0 = int(round((float(pick_seconds) + float(window_s[0])) / dt_local))
    i1 = int(round((float(pick_seconds) + float(window_s[1])) / dt_local))
    i0 = max(i0, 0)
    i1 = min(i1, x.size)
    cut = x[i0:i1]
    if cut.size < 8 or np.all(~np.isfinite(cut)):
        return None
    return cut.astype(float)

def _best_windowed_trace_in_group(st: Stream, group: Tuple[str, ...], pick_seconds: float, window_s: Tuple[float, float]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    best_arr, best_comp, best_amp = None, None, -np.inf
    for comp in group:
        tr = _extract_single_component(st, comp)
        if tr is None:
            continue
        arr = _cut_window_from_pick(tr, pick_seconds, window_s)
        if arr is None:
            continue
        amp = float(np.nanmax(np.abs(arr)))
        if np.isfinite(amp) and amp > best_amp:
            best_amp = amp
            best_arr = arr
            best_comp = comp
    return best_arr, best_comp

def _pick_peak_index(tr: Any) -> Optional[int]:
    data = tr.data if hasattr(tr, "data") else tr
    data = _sanitize_trace_data(np.asarray(data, dtype=np.float64))
    if data.size == 0:
        return None
    amp = np.abs(data)
    if not np.isfinite(amp).any():
        return None
    if np.allclose(amp, 0.0):
        return int(len(amp) // 2)
    return int(np.nanargmax(amp))

def _normalize(x: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    arr = _sanitize_trace_data(np.asarray(x, dtype=float))
    if arr.size == 0:
        return arr.astype(float)
    if np.isfinite(arr).any():
        arr = arr - float(np.nanmean(arr))
    else:
        return np.zeros_like(arr, dtype=float)
    scale = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 0.0
    if (not np.isfinite(scale)) or (scale <= eps):
        return np.zeros_like(arr, dtype=float)
    out = arr / scale
    out[~np.isfinite(out)] = 0.0
    return out.astype(float)

def _apply_shift(x: np.ndarray, shift: int) -> np.ndarray:
    arr = _sanitize_trace_data(np.asarray(x, dtype=float))
    if arr.size == 0:
        return arr.astype(float)
    ishift = int(round(shift))
    if ishift == 0:
        return arr.copy()
    out = np.zeros_like(arr, dtype=float)
    n = arr.size
    if abs(ishift) >= n:
        return out
    if ishift > 0:
        out[ishift:] = arr[: n - ishift]
    else:
        k = -ishift
        out[: n - k] = arr[k:]
    return out

def _crosscorr_best_shift(obs_n: np.ndarray, syn_n: np.ndarray, max_shift_samples: int) -> Tuple[int, float]:
    obs = _normalize(obs_n)
    syn = _normalize(syn_n)
    if obs.size == 0 or syn.size == 0:
        return 0, np.nan
    n = min(obs.size, syn.size)
    obs = obs[:n]
    syn = syn[:n]
    if np.allclose(obs, 0.0) or np.allclose(syn, 0.0):
        return 0, np.nan

    max_shift = int(max(0, max_shift_samples))
    obs_norm = float(np.linalg.norm(obs))
    best_shift = 0
    best_cc = -np.inf

    for shift in range(-max_shift, max_shift + 1):
        syn_a = _apply_shift(syn, shift)
        syn_norm = float(np.linalg.norm(syn_a))
        if syn_norm <= 0.0:
            continue
        cc = float(np.dot(obs, syn_a) / (obs_norm * syn_norm))
        if np.isfinite(cc) and cc > best_cc:
            best_cc = cc
            best_shift = shift

    if not np.isfinite(best_cc):
        return 0, np.nan
    return int(best_shift), float(best_cc)

def _phase_misfit_cc(obs_win: np.ndarray, syn_win: np.ndarray) -> Optional[Tuple[float, float]]:
    obs_n = _normalize(obs_win)
    syn_n = _normalize(syn_win)
    if (not np.isfinite(obs_n).all()) or (not np.isfinite(syn_n).all()):
        return None
    if np.allclose(obs_n, 0.0) or np.allclose(syn_n, 0.0):
        return None
    shift, cc = _crosscorr_best_shift(obs_n, syn_n, MAX_SHIFT_SAMPLES)
    syn_a = _apply_shift(syn_n, shift)
    if syn_a is None or syn_a.size != obs_n.size:
        return None
    if (not np.isfinite(syn_a).all()) or np.allclose(syn_a, 0.0):
        return None
    #misfit = float(np.mean((obs_n - syn_a) ** 2))
    cc = float(np.clip(cc, -1.0, 1.0))
    misfit = float(1.0 - cc)
    if not np.isfinite(misfit):
        return None
    return misfit, float(cc)

def _evaluate_component_group(
    obs_st: Stream,
    syn_st: Stream,
    components: Sequence[str],
    window: Tuple[float, float],
    *,
    pick_seconds: Optional[float] = None,
) -> Tuple[float, float, str]:
    per_sign = []
    for sign in (+1.0, -1.0):
        misfits = []
        ccs = []
        used = []
        for comp in components:
            obs = _extract_single_component(obs_st, comp)
            syn = _extract_single_component(syn_st, comp)
            if obs is None or syn is None:
                continue

            if pick_seconds is not None and np.isfinite(pick_seconds):
                ow = _cut_window_from_pick(obs, float(pick_seconds), window)
                sw = _cut_window_from_pick(syn, float(pick_seconds), window)
            else:
                iobs = _pick_peak_index(obs)
                isyn = _pick_peak_index(syn)
                ow = _cut_around_peak(obs, iobs, window)
                sw = _cut_around_peak(syn, isyn, window)

            if ow is None or sw is None:
                continue
            n = min(len(ow), len(sw))
            if n < 8:
                continue
            out = _phase_misfit_cc(ow[:n], sign * sw[:n])
            if out is None:
                continue
            m, cc = out
            if np.isfinite(m):
                misfits.append(m)
                ccs.append(cc)
                used.append(comp)

        if misfits:
            per_sign.append((float(np.mean(misfits)), float(np.mean(ccs)), "+" if sign > 0 else "-", "+".join(used)))

    if not per_sign:
        return np.nan, np.nan, ""

    best = min(per_sign, key=lambda x: x[0])
    return float(best[0]), float(best[1]), best[3]

def _phase_component_groups(mode: str) -> Tuple[List[Tuple[str, ...]], List[Tuple[str, ...]]]:
    if str(mode).lower() == "zr_zt":
        return [("Z", "R")], [("Z", "T")]
    # strict paper-equivalent default
    return [("Z",)], [("T",), ("R",)]

def evaluate_single_event(event_row: pd.Series, obs_st: Stream, syn_st: Stream) -> Dict[str, float | str]:
    event_id = str(event_row["event_id"]).strip()
    p_groups, s_groups = _phase_component_groups(COMPONENT_MODE)

    p_pick = _event_phase_pick_seconds(event_row, "p", obs_st)
    s_pick = _event_phase_pick_seconds(event_row, "s", obs_st)

    p_candidates = []
    for group in p_groups:
        m, cc, used = _evaluate_component_group(obs_st, syn_st, group, P_WINDOW, pick_seconds=p_pick)
        if np.isfinite(m):
            p_candidates.append((m, cc, "+".join(group), used))

    sw = S_WINDOW_EXTENDED if event_id in S_WINDOW_EXTENDED_EVENTS else S_WINDOW
    s_candidates = []
    for group in s_groups:
        m, cc, used = _evaluate_component_group(obs_st, syn_st, group, sw, pick_seconds=s_pick)
        if np.isfinite(m):
            s_candidates.append((m, cc, "+".join(group), used))

    if p_candidates:
        p_misfit, p_cc, p_group, p_used = min(p_candidates, key=lambda x: x[0])
    else:
        logger.warning(f"[{event_id}] nenhuma combinação P produziu janela/misfit válido.")
        p_misfit, p_cc, p_group, p_used = np.nan, np.nan, "", ""

    if s_candidates:
        s_misfit, s_cc, s_group, s_used = min(s_candidates, key=lambda x: x[0])
    else:
        logger.warning(f"[{event_id}] nenhuma combinação S produziu janela/misfit válido.")
        s_misfit, s_cc, s_group, s_used = np.nan, np.nan, "", ""

    valid = int(np.isfinite(p_misfit) or np.isfinite(s_misfit))
    return {
        "valid": valid,
        "p_misfit": float(p_misfit) if np.isfinite(p_misfit) else np.nan,
        "s_misfit": float(s_misfit) if np.isfinite(s_misfit) else np.nan,
        "p_cc": float(p_cc) if np.isfinite(p_cc) else np.nan,
        "s_cc": float(s_cc) if np.isfinite(s_cc) else np.nan,
        "p_group": p_group,
        "s_group": s_group,
        "p_used": p_used,
        "s_used": s_used,
    }

def build_full_model_from_suite(base_model: SeismicModel, suite_model: SuiteModel) -> SeismicModel:
    model = SeismicModel.test2()

    z = np.asarray(suite_model.z_km, dtype=float)
    vp = np.asarray(suite_model.vp_kms, dtype=float)
    vs = np.asarray(suite_model.vs_kms, dtype=float)

    if z.size < 2:
        raise ValueError(f"[{suite_model.model_id}] perfil insuficiente para construir modelo.")

    liquid_start = float(getattr(suite_model, 'liquid_start_km', np.nan))
    if np.isfinite(liquid_start) and liquid_start < (DEPTH_START_KM + 50.0):
        raise ValueError(f"[{suite_model.model_id}] líquido começa cedo demais ({liquid_start:.1f} km).")

    # Atualização por janelas, preservando a estrutura rasa do modelo base.
    # Se o perfil entra em líquido perto do CMB esperado, paramos de atualizar e
    # deixamos o núcleo do modelo-base intacto.
    for i in range(len(z) - 1):
        d0 = float(z[i])
        d1 = float(z[i + 1])
        if d1 <= DEPTH_START_KM or d0 >= DEPTH_END_KM:
            continue
        depth_min = max(d0, DEPTH_START_KM)
        depth_max = min(d1, DEPTH_END_KM)
        if depth_max <= depth_min:
            continue

        vp_mid = float(0.5 * (vp[i] + vp[i + 1]))
        vs_mid = float(0.5 * (vs[i] + vs[i + 1]))

        # Se chegou na parte líquida do perfil, não atualiza mais abaixo.
        if (vs_mid <= 0.05) or (vp_mid <= 0.5):
            break

        # Guardas conservadoras contra modelos mal lidos.
        if not (0.5 < vp_mid < 15.0 and 0.05 < vs_mid < 10.0 and vp_mid > vs_mid):
            raise ValueError(
                f"[{suite_model.model_id}] Vp/Vs inválidos em {depth_min:.1f}-{depth_max:.1f} km: "
                f"Vp={vp_mid:.3f}, Vs={vs_mid:.3f}"
            )

        model.update_mantle(
            vpv=vp_mid, vph=vp_mid, vsv=vs_mid, vsh=vs_mid,
            depth_min=depth_min, depth_max=depth_max,
            core_radius_km=CORE_RADIUS_KM, allow_below_cmb=ALLOW_BELOW_CMB,
        )
    return model

def _weighted_total_misfit(mean_p_misfit: float, mean_s_misfit: float) -> float:
    values = []
    weights = []
    if np.isfinite(mean_p_misfit):
        values.append(float(mean_p_misfit))
        weights.append(1.0)
    if np.isfinite(mean_s_misfit):
        values.append(float(mean_s_misfit))
        weights.append(float(S_WEIGHT))
    if not values:
        return np.nan
    return float(np.average(values, weights=weights))

def evaluate_suite_model(base_model: SeismicModel, events: pd.DataFrame, suite_model: SuiteModel) -> Dict[str, float | str]:
    model = build_full_model_from_suite(base_model, suite_model)

    p_misfits = []
    s_misfits = []
    p_ccs = []
    s_ccs = []
    used_events = 0
    for _, evt in events.iterrows():
        event_id = str(evt["event_id"]).strip()
        obs_st = load_observed_waveforms(event_id)
        if obs_st is None:
            continue
        event_depth_km = float(evt.get("depth", suite_model.source_depth_km))
        syn_st = synthesize_event_traces(evt, model, source_depth_km=event_depth_km)
        if syn_st is None:
            continue

        out = evaluate_single_event(evt, obs_st, syn_st)
        if out["valid"] == 0:
            continue
        used_events += 1

        if np.isfinite(out.get("p_misfit", np.nan)):
            p_misfits.append(out["p_misfit"])
            p_ccs.append(out["p_cc"])
        if np.isfinite(out.get("s_misfit", np.nan)):
            s_misfits.append(out["s_misfit"])
            s_ccs.append(out["s_cc"])

    if used_events == 0:
        return {
            "model_id": suite_model.model_id,
            "source_path": suite_model.source_path,
            "source_format": suite_model.source_format,
            "composition": suite_model.composition,
            "tpot_K": suite_model.tpot_K,
            "adiabat_K_km": suite_model.adiabat_K_km,
            "d_main_km": suite_model.d_main_km,
            "w_main_km": suite_model.w_main_km,
            "d_800_km": suite_model.d_800_km,
            "w_800_km": suite_model.w_800_km,
            "d_1000_km": suite_model.d_1000_km,
            "w_1000_km": suite_model.w_1000_km,
            "d_1200_km": suite_model.d_1200_km,
            "w_1200_km": suite_model.w_1200_km,
            "d_1400_km": suite_model.d_1400_km,
            "w_1400_km": suite_model.w_1400_km,
            "d_1600_km": suite_model.d_1600_km,
            "w_1600_km": suite_model.w_1600_km,
            "mean_p_misfit": np.nan,
            "mean_s_misfit": np.nan,
            "total_misfit": np.nan,
            "mean_p_cc": np.nan,
            "mean_s_cc": np.nan,
            "used_events": 0,
        }

    mean_p_misfit = float(np.nanmean(p_misfits)) if len(p_misfits) else np.nan
    mean_s_misfit = float(np.nanmean(s_misfits)) if len(s_misfits) else np.nan
    mean_p_cc = float(np.nanmean(p_ccs)) if len(p_ccs) else np.nan
    mean_s_cc = float(np.nanmean(s_ccs)) if len(s_ccs) else np.nan
    total = _weighted_total_misfit(mean_p_misfit, mean_s_misfit)

    return {
        "model_id": suite_model.model_id,
        "source_path": suite_model.source_path,
        "source_format": suite_model.source_format,
        "composition": suite_model.composition,
        "tpot_K": suite_model.tpot_K,
        "adiabat_K_km": suite_model.adiabat_K_km,
        "d_main_km": suite_model.d_main_km,
        "w_main_km": suite_model.w_main_km,
        "d_800_km": suite_model.d_800_km,
        "w_800_km": suite_model.w_800_km,
        "d_1000_km": suite_model.d_1000_km,
        "w_1000_km": suite_model.w_1000_km,
        "d_1200_km": suite_model.d_1200_km,
        "w_1200_km": suite_model.w_1200_km,
        "d_1400_km": suite_model.d_1400_km,
        "w_1400_km": suite_model.w_1400_km,
        "d_1600_km": suite_model.d_1600_km,
        "w_1600_km": suite_model.w_1600_km,
        "mean_p_misfit": mean_p_misfit,
        "mean_s_misfit": mean_s_misfit,
        "total_misfit": total,
        "mean_p_cc": mean_p_cc,
        "mean_s_cc": mean_s_cc,
        "used_events": int(used_events),
    }

# ============================================================
# FIGURES
# ============================================================
def _norm_for_colors(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    y = np.full_like(x, np.nan, dtype=float)
    if not np.any(ok):
        return y
    mn, mx = np.nanmin(x[ok]), np.nanmax(x[ok])
    den = mx - mn if mx > mn else 1.0
    y[ok] = (x[ok] - mn) / den
    return y

def plot_fig3_ab(suite: Sequence[SuiteModel], results_df: pd.DataFrame, output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    cmap = plt.cm.Blues_r

    by_id = {m.model_id: m for m in suite}
    df = results_df.copy()
    pnorm = _norm_for_colors(df["mean_p_misfit"].values)
    snorm = _norm_for_colors(df["mean_s_misfit"].values)

    best_p = df.loc[df["mean_p_misfit"].idxmin()] if df["mean_p_misfit"].notna().any() else None
    best_s = df.loc[df["mean_s_misfit"].idxmin()] if df["mean_s_misfit"].notna().any() else None

    fig, ax = plt.subplots(figsize=(4.8, 7.2))
    for i, row in df.iterrows():
        m = by_id[row["model_id"]]
        color = cmap(pnorm[i]) if np.isfinite(pnorm[i]) else (0.9, 0.9, 0.9, 1.0)
        ax.plot(m.vp_kms, m.z_km, color=color, lw=0.9, alpha=0.65)
    if best_p is not None:
        m = by_id[best_p["model_id"]]
        ax.plot(m.vp_kms, m.z_km, "r--", lw=2.7)
    ax.invert_yaxis()
    ax.set_xlabel("Vp (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.text(0.03, 0.05, "A", transform=ax.transAxes, fontweight="bold", fontsize=16)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    plt.colorbar(sm, ax=ax, label="P misfit")
    out_a = os.path.join(output_dir, "fig3A_vp_profiles_pmisfit_S0395a.png")
    plt.tight_layout()
    plt.savefig(out_a, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 7.2))
    for i, row in df.iterrows():
        m = by_id[row["model_id"]]
        color = cmap(snorm[i]) if np.isfinite(snorm[i]) else (0.9, 0.9, 0.9, 1.0)
        ax.plot(m.vs_kms, m.z_km, color=color, lw=0.9, alpha=0.65)
    if best_s is not None:
        m = by_id[best_s["model_id"]]
        ax.plot(m.vs_kms, m.z_km, "r--", lw=2.7)
    ax.invert_yaxis()
    ax.set_xlabel("Vs (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.text(0.03, 0.05, "B", transform=ax.transAxes, fontweight="bold", fontsize=16)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    plt.colorbar(sm, ax=ax, label="S misfit")
    out_b = os.path.join(output_dir, "fig3B_vs_profiles_smisfit_S0395a.png")
    plt.tight_layout()
    plt.savefig(out_b, dpi=300)
    plt.close(fig)

    return out_a, out_b

def _scatter_depth_misfit(ax, df: pd.DataFrame, ycol: str, letter: str, ylabel: str):
    finite_t = df["tpot_K"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_t) == 0:
        tmin, tmax = 1500.0, 1700.0
    else:
        tmin = float(finite_t.min())
        tmax = float(finite_t.max())
    den = (tmax - tmin) if (tmax > tmin) else 1.0

    for _, row in df.iterrows():
        marker = COMP_MARKERS.get(str(row["composition"]), "o")
        adiabat_val = row["adiabat_K_km"]
        size = ADIABAT_SIZE.get(round(float(adiabat_val), 3), 60) if np.isfinite(adiabat_val) else 60
        tval = float(row["tpot_K"]) if np.isfinite(row["tpot_K"]) else 0.5 * (tmin + tmax)
        color = plt.cm.viridis((tval - tmin) / den)
        ax.scatter(
            row["d_main_km"],
            row[ycol],
            s=size,
            marker=marker,
            color=color,
            edgecolor="k",
            linewidth=0.3,
            alpha=0.85,
        )

    ax.set_xlabel("Main discontinuity depth (km)")
    ax.set_ylabel(ylabel)
    ax.text(0.03, 0.90, letter, transform=ax.transAxes, fontweight="bold", fontsize=15)

def plot_fig3_cde(results_df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    df = results_df.dropna(subset=["d_main_km"]).copy()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
    _scatter_depth_misfit(axes[0], df, "mean_p_misfit", "C", "P misfit")
    _scatter_depth_misfit(axes[1], df, "mean_s_misfit", "D", "S misfit")
    _scatter_depth_misfit(axes[2], df, "total_misfit", "E", "Total misfit")

    finite_t = df["tpot_K"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_t) == 0:
        tmin, tmax = 1500.0, 1700.0
    else:
        tmin = float(finite_t.min())
        tmax = float(finite_t.max())
    norm = plt.Normalize(vmin=tmin, vmax=tmax)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Mantle potential temperature (K)")

    out = os.path.join(output_dir, "fig3CDE_misfit_vs_depth_S0395a.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_mtz4_scatter(results_df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    depth_cols = [
        ("d_800_km", "700-900 km"),
        ("d_1000_km", "900-1100 km"),
        ("d_1200_km", "1100-1300 km"),
        ("d_1400_km", "1300-1500 km"),
        ("d_1600_km", "1500-1800 km"),
    ]
    misfit_cols = [
        ("mean_p_misfit", "P misfit"),
        ("mean_s_misfit", "S misfit"),
        ("total_misfit", "Total misfit"),
    ]
    needed = [c for c, _ in depth_cols] + [c for c, _ in misfit_cols]
    df = results_df.dropna(subset=needed, how="all").copy()
    if df.empty:
        return ""

    finite_t = df["tpot_K"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_t) == 0:
        tmin, tmax = 1500.0, 1700.0
    else:
        tmin = float(finite_t.min())
        tmax = float(finite_t.max())
    den = (tmax - tmin) if (tmax > tmin) else 1.0

    fig, axes = plt.subplots(5, 3, figsize=(13.5, 16.5), sharex=False, sharey=False)
    letters = list("CDEFGHIJKLMNOPQ")
    ilet = 0
    for i, (dcol, dlabel) in enumerate(depth_cols):
        for j, (mcol, mlabel) in enumerate(misfit_cols):
            ax = axes[i, j]
            sub = df.dropna(subset=[dcol, mcol]).copy()
            for _, row in sub.iterrows():
                marker = COMP_MARKERS.get(str(row.get("composition", "UNK")), "o")
                adiabat_val = row.get("adiabat_K_km", np.nan)
                size = ADIABAT_SIZE.get(round(float(adiabat_val), 3), 60) if np.isfinite(adiabat_val) else 60
                tval = float(row["tpot_K"]) if np.isfinite(row.get("tpot_K", np.nan)) else 0.5 * (tmin + tmax)
                color = plt.cm.viridis((tval - tmin) / den)
                ax.scatter(
                    row[dcol], row[mcol], s=size, marker=marker, color=color,
                    edgecolor="k", linewidth=0.3, alpha=0.85,
                )
            ax.set_xlabel(f"Depth near {dlabel} (km)")
            ax.set_ylabel(mlabel)
            ax.text(0.03, 0.90, letters[ilet], transform=ax.transAxes, fontweight="bold", fontsize=12)
            ilet += 1
            if i == 0:
                ax.set_title(mlabel)

    norm = plt.Normalize(vmin=tmin, vmax=tmax)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Mantle potential temperature (K)")

    out = os.path.join(output_dir, "fig_mtz4_scatter_misfit_vs_depth_S0395a.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_mtz4_paperlike(suite: Sequence[SuiteModel], results_df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    if results_df is None or results_df.empty:
        return ""
    best = results_df.sort_values("total_misfit", ascending=True, na_position="last").iloc[0]
    by_id = {m.model_id: m for m in suite}
    model_id = str(best["model_id"])
    if model_id not in by_id:
        return ""
    m = by_id[model_id]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 7.2), sharey=True)
    ax_vp, ax_vs = axes

    ax_vp.plot(m.vp_kms, m.z_km, color="tab:blue", lw=2.0)
    ax_vs.plot(m.vs_kms, m.z_km, color="tab:orange", lw=2.0)

    layer_info = [
        ("d_800_km", "800 km", "C"),
        ("d_1000_km", "1000 km", "D"),
        ("d_1200_km", "1200 km", "E"),
        ("d_1400_km", "1400 km", "F"),
        ("d_1600_km", "1600 km", "G"),
    ]
    colors = ["0.35", "0.45", "0.55", "0.60", "0.65"]
    for (col, label, _), c in zip(layer_info, colors):
        dval = best.get(col, np.nan)
        if np.isfinite(dval):
            for ax in axes:
                ax.axhline(float(dval), color=c, ls="--", lw=1.2)
            ax_vs.text(
                0.98,
                float(dval),
                f" {label}: {float(dval):.1f} km",
                transform=ax_vs.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=9,
                color=c,
            )

    ax_vp.invert_yaxis()
    ax_vp.set_xlabel("Vp (km/s)")
    ax_vs.set_xlabel("Vs (km/s)")
    ax_vp.set_ylabel("Depth (km)")
    ax_vp.text(0.03, 0.05, "A", transform=ax_vp.transAxes, fontweight="bold", fontsize=16)
    ax_vs.text(0.03, 0.05, "B", transform=ax_vs.transAxes, fontweight="bold", fontsize=16)
    fig.suptitle(f"Best-fitting full model: {model_id}", y=0.98)

    out = os.path.join(output_dir, "vp_vs_paperlike_mtz4_bestmodel_S0395a.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    return out

# ============================================================
# MAIN
# ============================================================
def main(models_dir: str = MODELS_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    events = load_event_catalog(CSV_FILE)
    events = events.dropna(subset=["event_id", "latitude", "longitude", "depth"])
    events = events.reset_index(drop=True)
    logger.info(f"{len(events)} eventos válidos após limpeza.")

    base_model = SeismicModel.test2()
    logger.info("Modelo base carregado.")

    suite = load_model_suite_from_models_dir(models_dir, base_model)

    rows = []
    for suite_model in tqdm(suite, desc="Evaluating full-model suite"):
        try:
            row = evaluate_suite_model(base_model, events, suite_model)
            rows.append(row)
        except Exception as e:
            logger.warning(f"[{suite_model.model_id}] modelo ignorado durante a avaliação: {e}")
            continue
        logger.info(
            f"{suite_model.model_id} | P={row['mean_p_misfit']:.4f} | "
            f"S={row['mean_s_misfit']:.4f} | TOT={row['total_misfit']:.4f} | "
            f"d_800={suite_model.d_800_km:.1f} | d_1000={suite_model.d_1000_km:.1f} | "
            f"d_1200={suite_model.d_1200_km:.1f} | d_1400={suite_model.d_1400_km:.1f} | "
            f"d_1600={suite_model.d_1600_km:.1f}"
        )

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        raise RuntimeError("Nenhum modelo produziu misfit válido. Verifique os logs de picks/janelas.")
    if "total_misfit" not in results_df.columns:
        results_df["total_misfit"] = np.nan
    results_df = results_df.sort_values("total_misfit", ascending=True, na_position="last").reset_index(drop=True)

    out_csv = os.path.join(OUTPUT_DIR, "pnas_equivalent_suite_results_S0395a.csv")
    results_df.to_csv(out_csv, index=False)
    logger.info(f"Resultados salvos em {out_csv}")

    plot_fig3_ab(suite, results_df, FIG_DIR)
    plot_fig3_cde(results_df, FIG_DIR)
    plot_mtz4_scatter(results_df, FIG_DIR)
    plot_mtz4_paperlike(suite, results_df, FIG_DIR)

    best = results_df.iloc[0]
    summary_txt = os.path.join(OUTPUT_DIR, "best_model_summary_S0395a.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Best-fitting full model (models/ directory workflow)\n")
        f.write(f"component_mode = {COMPONENT_MODE}\n")
        f.write(f"model_id = {best['model_id']}\n")
        f.write(f"source_path = {best['source_path']}\n")
        f.write(f"source_format = {best['source_format']}\n")
        f.write(f"composition = {best['composition']}\n")
        f.write(f"tpot_K = {best['tpot_K']}\n")
        f.write(f"adiabat_K_km = {best['adiabat_K_km']}\n")
        f.write(f"d_main_km = {best['d_main_km']}\n")
        f.write(f"w_main_km = {best['w_main_km']}\n")
        f.write(f"d_800_km = {best['d_800_km']}\n")
        f.write(f"w_800_km = {best['w_800_km']}\n")
        f.write(f"d_1000_km = {best['d_1000_km']}\n")
        f.write(f"w_1000_km = {best['w_1000_km']}\n")
        f.write(f"d_1200_km = {best['d_1200_km']}\n")
        f.write(f"w_1200_km = {best['w_1200_km']}\n")
        f.write(f"d_1400_km = {best['d_1400_km']}\n")
        f.write(f"w_1400_km = {best['w_1400_km']}\n")
        f.write(f"d_1600_km = {best['d_1600_km']}\n")
        f.write(f"w_1600_km = {best['w_1600_km']}\n")
        f.write(f"mean_p_misfit = {best['mean_p_misfit']}\n")
        f.write(f"mean_s_misfit = {best['mean_s_misfit']}\n")
        f.write(f"total_misfit = {best['total_misfit']}\n")
        f.write(f"used_events = {best['used_events']}\n")

    logger.info(f"Resumo do melhor modelo salvo em {summary_txt}")
    logger.info("=== End ===")

if __name__ == "__main__":
    main(models_dir=MODELS_DIR)
