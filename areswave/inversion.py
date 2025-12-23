import os
import numpy as np
import pandas as pd
import logging
from obspy import read
from obspy.signal.filter import bandpass
from scipy.signal import correlate
from scipy.signal.windows import tukey

# --- módulos corretos do seu ambiente ---
from dsmpy import dsm_Mars as dsm
from dsmpy import seismicmodel_Mars as seismicmodel_mod

# ================================================================
# CONFIGURAÇÃO DE LOGGING E DIRETÓRIOS
# ================================================================
BASE_DIR = os.getcwd()
DATA_DIR = "/home/lyara/areswave/SAC"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGS_DIR = os.path.join(BASE_DIR, "figs")

for d in [OUTPUTS_DIR, FIGS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MarsInversion")

# ================================================================
# UTILITÁRIOS DE LEITURA DE DADOS
# ================================================================
def load_arrivals(csv_path: str):
    """
    Carrega tabela de tempos de chegada e momento sísmico de cada evento.
    Espera colunas: event_id, p_sec, s_sec, Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, depth, lat, lon
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["event_id"])
    arrivals = {}
    for _, row in df.iterrows():
        arrivals[row["event_id"]] = {
            "p_sec": row.get("p_sec", np.nan),
            "s_sec": row.get("s_sec", np.nan),
            "Mrr": row.get("Mrr", 0.0),
            "Mtt": row.get("Mtt", 0.0),
            "Mpp": row.get("Mpp", 0.0),
            "Mrt": row.get("Mrt", 0.0),
            "Mrp": row.get("Mrp", 0.0),
            "Mtp": row.get("Mtp", 0.0),
            "source_depth_km": row.get("depth", 50.0),
            "lat": row.get("lat", 0.0),
            "lon": row.get("lon", 0.0),
            "distance_deg": row.get("distance_deg", 20.0)
        }
    logger.info(f"{len(arrivals)} eventos carregados de {csv_path}")
    return arrivals


def group_traces_by_event(data_dir: str):
    """Lê waveforms .SAC (Z,R,T) e agrupa por evento"""
    events = {}
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(".sac"):
            continue
        parts = fname.split("_")
        if len(parts) < 2:
            continue
        event_id = parts[0]
        comp = parts[-1].split(".")[0].upper()  # Z, R, T
        tr = read(os.path.join(data_dir, fname))[0]
        tr.detrend("demean")
        tr.filter("bandpass", freqmin=0.05, freqmax=1.0, corners=4, zerophase=True)
        events.setdefault(event_id, {})[comp] = tr
    logger.info(f"{len(events)} eventos com waveforms agrupados de {data_dir}")
    return events

# ================================================================
# MODELO FÍSICO: Vp / Vs / DENSIDADE 1D DE MARTE
# ================================================================
class ModelParam:
    """
    Representa um modelo 1D paramétrico de Marte.
    Vp, Vs variam linearmente com profundidade + descontinuidades gaussianas.
    """
    def __init__(self, vp_top, vp_bot, vs_top, vs_bot,
                 d_shallow=800, d1000=1000, d_deep=1200, d_layer=1600,
                 w1000=80):
        self.vp_top = vp_top
        self.vp_bot = vp_bot
        self.vs_top = vs_top
        self.vs_bot = vs_bot
        self.d_shallow = d_shallow
        self.d1000 = d1000
        self.d_deep = d_deep
        self.d_layer = d_layer
        self.w1000 = w1000

    def get_velocity_profiles(self, depths_km):
        """Gera perfis Vp e Vs com gradiente e descontinuidades gaussianas."""
        z = np.asarray(depths_km)
        frac = (z - z.min()) / (z.max() - z.min())
        vp = self.vp_top + frac * (self.vp_bot - self.vp_top)
        vs = self.vs_top + frac * (self.vs_bot - self.vs_top)
        for d in [self.d_shallow, self.d1000, self.d_deep, self.d_layer]:
            amp_vp = 0.2 * np.sin(d / 300.0)
            amp_vs = 0.15 * np.cos(d / 400.0)
            vp += amp_vp * np.exp(-0.5 * ((z - d) / self.w1000) ** 2)
            vs += amp_vs * np.exp(-0.5 * ((z - d) / self.w1000) ** 2)
        return vp, vs

# ================================================================
# CONSTRUÇÃO DE MODELO DSMpy (compatível com dsmpy)
# ================================================================
def build_dsm_model(depths_km, vp, vs):
    """
    Cria um modelo compatível com dsmpy.seismicmodel_Mars.SeismicModel.
    """
    sm = seismicmodel_mod.SeismicModel.tayak()  # modelo base de Marte
    z = np.asarray(depths_km, dtype=float)
    sm.z = z
    sm.vpv = vp.copy()
    sm.vph = vp.copy()
    sm.vsv = vs.copy()
    sm.vsh = vs.copy()
    sm.eta = np.ones_like(vp)
    sm.rho = np.clip(1.6612 * vp - 0.4721, 2.0, None)
    sm.qmu = np.full_like(vp, 10000.0)
    sm.qkappa = np.full_like(vp, 10000.0)
    if hasattr(sm, "build"):
        sm.build()
    return sm

# ================================================================
# FUNÇÃO DE SÍNTESE SISMOGRÁFICA (dsmpy + fallback Ricker)
# ================================================================
def synthesize_event_traces(depths, vp, vs, arr, obs_trs,
                            duration=1200.0, fs=20.0):
    """
    Gera traços sintéticos (Z, R, T) via DSMpy (versão Mars compatível).
    Se falhar, usa fallback Ricker.
    """
    try:
        import numpy as np
        from obspy import UTCDateTime
        from dsmpy import seismicmodel_Mars
        from dsmpy.event_Mars import Event, MomentTensor
        from dsmpy.station_Mars import Station
        from synthetics_function import generate_synthetics, apply_filter

        # Modelo sísmico marciano
        sm = seismicmodel_Mars.SeismicModel.tayak()

        # Tensor de momento do arrivals.csv
        mt = MomentTensor(
            arr.get("Mrr", 0.0),
            arr.get("Mrt", 0.0),
            arr.get("Mrp", 0.0),
            arr.get("Mtt", 0.0),
            arr.get("Mtp", 0.0),
            arr.get("Mpp", 0.0)
        )

        # Cria evento DSM
        event = Event(
            event_id=str(arr.get("event_id", "EVT")),
            latitude=arr.get("lat", 0.0),
            longitude=arr.get("lon", 0.0),
            depth=arr.get("source_depth_km", 30.0),
            mt=mt,
            centroid_time=UTCDateTime().timestamp,
            source_time_function=None
        )

        # Define estação
        station = Station(
            name="ELYSE",
            network="XB",
            latitude=4.502384,
            longitude=135.623447
        )

        # Gera sintéticos usando função dareswave (DSMpy integrado)
        output = generate_synthetics(
            event, [station], sm,
            tlen=duration,
            nspc=1256,
            sampling_hz=fs
        )

        ts = output.ts
        syn_dict = {
            "Z": apply_filter(output["Z", "ELYSE_XB"], fs),
            "R": apply_filter(output["R", "ELYSE_XB"], fs),
            "T": apply_filter(output["T", "ELYSE_XB"], fs)
        }

        logger.info("✅ DSMpy (areswave) executado com sucesso com MomentTensor real.")
        return syn_dict

    except Exception as e:
        logger.warning(f"⚠️ DSMpy (areswave) falhou, usando fallback Ricker ({e})")
        t = np.arange(0, duration, 1 / fs)

        def ricker(t, f):
            return (1 - 2 * (np.pi * f * (t - 1/f))**2) * np.exp(-(np.pi * f * (t - 1/f))**2)

        wave = ricker(t, 0.2)
        syn = {c: obs_trs[c].copy() for c in ["Z", "R", "T"]}
        for c in ["Z", "R", "T"]:
            syn[c].data = np.convolve(syn[c].data, wave, mode="same")
        return syn

# ================================================================
# CONTINUAÇÃO — INVERSÃO E AVALIAÇÃO (COMPATÍVEL COM dsmpy)
# ================================================================
from scipy.optimize import differential_evolution
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================================================================
# MÉTRICAS DE AJUSTE
# ================================================================
def misfit_metric(obs, syn, normalize=True):
    """
    Calcula o misfit entre traços observados e sintéticos.
    Usa correlação cruzada normalizada como métrica de semelhança.
    """
    obs = np.nan_to_num(obs)
    syn = np.nan_to_num(syn)
    if normalize:
        if np.max(np.abs(obs)) > 0:
            obs /= np.max(np.abs(obs))
        if np.max(np.abs(syn)) > 0:
            syn /= np.max(np.abs(syn))

    corr = correlate(obs, syn, mode="valid")
    corr_val = np.max(np.abs(corr)) / len(obs)
    misfit = 1.0 - corr_val
    return misfit


def window_trace(trace, center, width, fs):
    """Aplica janela Tukey ao redor do tempo central 'center' (s)."""
    npts = int(width * fs)
    start = int(center * fs - npts // 2)
    end = start + npts
    if start < 0 or end > len(trace):
        return np.zeros(npts)
    w = tukey(npts, 0.2)
    return trace[start:end] * w

# ================================================================
# GRID SEARCH — FASE 1 (EXPLORATÓRIA)
# ================================================================
def grid_search_inversion(events, arrivals, vp_range, vs_range,
                          depths=np.linspace(700, 1800, 56),
                          fs=20.0, duration=1200.0):
    """
    Varredura inicial para achar regiões promissoras de Vp/Vs.
    Retorna DataFrame com misfit médio.
    """
    results = []
    for vp_t, vp_b, vs_t, vs_b in tqdm(product(vp_range, vp_range, vs_range, vs_range),
                                       desc="Grid search"):
        model = ModelParam(vp_t, vp_b, vs_t, vs_b)
        vp_prof, vs_prof = model.get_velocity_profiles(depths)
        costs = []
        for ev_id, obs_trs in events.items():
            arr = arrivals.get(ev_id, {})
            try:
                syn = synthesize_event_traces(depths, vp_prof, vs_prof, arr, obs_trs, duration, fs)
                tP = arr.get("p_sec", 300.0)
                tS = arr.get("s_sec", 600.0)
                obsP = window_trace(obs_trs["Z"].data, tP, 20, fs)
                synP = window_trace(syn["Z"].data, tP, 20, fs)
                obsS = window_trace(obs_trs["R"].data, tS, 30, fs)
                synS = window_trace(syn["R"].data, tS, 30, fs)
                cost = misfit_metric(obsP, synP) + misfit_metric(obsS, synS)
                costs.append(cost)
            except Exception as e:
                logger.debug(f"[{ev_id}] Grid falhou: {e}")
        mean_cost = np.mean(costs) if costs else 999
        results.append(dict(vp_top=vp_t, vp_bot=vp_b, vs_top=vs_t, vs_bot=vs_b, misfit=mean_cost))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUTS_DIR, "grid_results.csv"), index=False)
    logger.info("Grid search concluída.")
    return df


# ================================================================
# DIFFERENTIAL EVOLUTION — FASE 2 (REFINAMENTO)
# ================================================================
def differential_refinement(events, arrivals,
                            depths=np.linspace(700, 1800, 56),
                            fs=20.0, duration=1200.0):
    """
    Refina o modelo inicial usando differential evolution.
    """
    def cost_function(params):
        vp_t, vp_b, vs_t, vs_b = params
        model = ModelParam(vp_t, vp_b, vs_t, vs_b)
        vp_prof, vs_prof = model.get_velocity_profiles(depths)
        total_costs = []
        for ev_id, obs_trs in events.items():
            arr = arrivals.get(ev_id, {})
            try:
                syn = synthesize_event_traces(depths, vp_prof, vs_prof, arr, obs_trs, duration, fs)
                tP = arr.get("p_sec", 300.0)
                tS = arr.get("s_sec", 600.0)
                obsP = window_trace(obs_trs["Z"].data, tP, 20, fs)
                synP = window_trace(syn["Z"].data, tP, 20, fs)
                obsS = window_trace(obs_trs["R"].data, tS, 30, fs)
                synS = window_trace(syn["R"].data, tS, 30, fs)
                c = misfit_metric(obsP, synP) + misfit_metric(obsS, synS)
                total_costs.append(c)
            except Exception as e:
                logger.debug(f"Falha DE {ev_id}: {e}")
                total_costs.append(999)
        return np.mean(total_costs)

    bounds = [(6.5, 9.5), (8.0, 10.0), (3.5, 5.0), (4.0, 5.5)]
    result = differential_evolution(
        cost_function,
        bounds,
        strategy="best1bin",
        maxiter=20,
        popsize=8,
        seed=42,
        tol=0.02,
        disp=True
    )
    logger.info(f"Melhor resultado DE: {result.x}, custo={result.fun:.4f}")
    return result


# ================================================================
# UTILITÁRIOS E EXPORT
# ================================================================
def ensure_dirs():
    for d in [OUTPUTS_DIR, FIGS_DIR]:
        os.makedirs(d, exist_ok=True)


def save_best_model(result, depths, filename="best_model.csv"):
    vp_t, vp_b, vs_t, vs_b = result.x
    model = ModelParam(vp_t, vp_b, vs_t, vs_b)
    vp_prof, vs_prof = model.get_velocity_profiles(depths)
    df = pd.DataFrame({"depth_km": depths, "Vp": vp_prof, "Vs": vs_prof})
    df.to_csv(os.path.join(OUTPUTS_DIR, filename), index=False)
    logger.info(f"Modelo salvo em {filename}.")
    return df



# ================================================================
# VISUALIZAÇÃO (RÁPIDA)
# ================================================================
def plot_model(df_model):
    plt.figure(figsize=(6, 6))
    plt.plot(df_model["Vp"], df_model["depth_km"], label="Vp", lw=2)
    plt.plot(df_model["Vs"], df_model["depth_km"], label="Vs", lw=2)
    plt.gca().invert_yaxis()
    plt.xlabel("Velocity (km/s)")
    plt.ylabel("Depth (km)")
    plt.title("Best-fit Vp/Vs Model — Mars Interior")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "vp_vs_profile.png"), dpi=250)
    plt.close()


# ================================================================
# BLOCO DE TESTE LOCAL
# ================================================================
if __name__ == "__main__":
    arrivals = load_arrivals("/home/lyara/areswave/data/arrivals.csv")
    events = group_traces_by_event("/home/lyara/areswave/SAC")
    ensure_dirs()

    # Etapa 1 — Grid Search
    vp_range = np.linspace(7.0, 9.0, 3)
    vs_range = np.linspace(4.0, 5.0, 3)
    df_grid = grid_search_inversion(events, arrivals, vp_range, vs_range)

    # Etapa 2 — Refinamento Differential Evolution
    result = differential_refinement(events, arrivals)
    df_model = save_best_model(result, np.linspace(700, 1800, 56))
    plot_model(df_model)
