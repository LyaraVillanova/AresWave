import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read, Stream
from scipy.optimize import differential_evolution, curve_fit
from tqdm import tqdm
import logging
from synthetics_function import generate_synthetics, apply_filter
from denoising import polarization_filter
from dsmpy.seismicmodel_Mars import SeismicModel
from dsmpy.station_Mars import Station
from dsmpy.event_Mars import Event, MomentTensor
from scipy.interpolate import griddata, RegularGridInterpolator
from obspy import Trace, Stream, read, UTCDateTime

# === LOGGING ===
COST_HISTORY = []
PARAM_HISTORY = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

DATA_DIR = "/home/lyara/areswave/SACS0185a/"
CSV_FILE = "data/arrivals_S0185a.csv"
OUTPUT_DIR = "/home/lyara/areswave/outputs/"
FIG_DIR = "/home/lyara/areswave/figs/"
FS = 20.0  # Hz

# ---------------------------------------------------------------------------
# MONKEY PATCH
# ---------------------------------------------------------------------------
def _update_mantle(self, vpv, vph, vsv, vsh, depth_min=700, depth_max=1800):
    try:
        vrmin = self._vrmin
        vrmax = self._vrmax

        R = float(vrmax[-1])

        dmin = float(min(depth_min, depth_max))
        dmax = float(max(depth_min, depth_max))

        r_shallow = R - dmin
        r_deep    = R - dmax

        r_low  = min(r_shallow, r_deep)
        r_high = max(r_shallow, r_deep)

        for i, (zmin, zmax) in enumerate(zip(vrmin, vrmax)):
            if (zmax >= r_low) and (zmin <= r_high):
                if np.ndim(vpv) == 0:
                    self._vpv[:, i] = np.full_like(self._vpv[:, i], vpv)
                    self._vph[:, i] = np.full_like(self._vph[:, i], vph)
                    self._vsv[:, i] = np.full_like(self._vsv[:, i], vsv)
                    self._vsh[:, i] = np.full_like(self._vsh[:, i], vsh)
                else:
                    self._vpv[:, i] = vpv
                    self._vph[:, i] = vph
                    self._vsv[:, i] = vsv
                    self._vsh[:, i] = vsh

    except Exception as e:
        logger.error(f"Erro em _update_mantle (depth {depth_min}-{depth_max} km): {e}")

setattr(SeismicModel, "update_mantle", _update_mantle)

# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------------------------
def load_event_catalog(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    logger.info(f"{len(df)} eventos carregados de {csv_path}")
    return df

def load_observed_waveforms(event_id):
    base_event_path = os.path.join(DATA_DIR, event_id)
    if not os.path.exists(base_event_path):
        base_event_path = DATA_DIR
    st = Stream()
    for fname in os.listdir(base_event_path):
        if fname.endswith(".sac"):
            st += read(os.path.join(base_event_path, fname))
    if len(st) == 0:
        logger.warning(f"[{event_id}] No data found in {base_event_path}")
        return None
    st.detrend("demean")
    st.filter("bandpass", freqmin=0.3, freqmax=0.9, zerophase=True)
    logger.info(f"[{event_id}] {len(st)} traces loaded from {base_event_path}")
    return st

def synthesize_event_traces(event_row, model, duration=2000, components=("Z", "R", "T")):
    try:
        mt = MomentTensor(
            Mrr=event_row["mrr"],
            Mtt=event_row["mtt"],
            Mpp=event_row["mpp"],
            Mrt=event_row["mrt"],
            Mrp=event_row["mrp"],
            Mtp=event_row["mtp"]
        )

        try:
            centroid_time = UTCDateTime(event_row.get("time_p", UTCDateTime()))
        except Exception:
            centroid_time = UTCDateTime()

        event = Event(
            event_id=str(event_row["event_id"]),
            latitude=event_row["latitude"],
            longitude=event_row["longitude"],
            depth=event_row["depth"],
            mt=mt,
            centroid_time=centroid_time,
            source_time_function=None
        )

        stations = [Station(name='ELYSE', network='XB', latitude=4.502384, longitude=135.623447)]

        output = generate_synthetics(
            event,
            stations,
            model,
            tlen=duration,
            nspc=2,
            sampling_hz=FS
        )

        us = output.us  # (3, nr, tlen)
        st_syn = Stream()
        for i, comp in enumerate(components):
            tr = Trace(data=us[i, 0, :])
            tr.stats.delta = 1.0 / FS
            tr.stats.channel = comp
            st_syn += tr

        for tr in st_syn:
            tr.data = apply_filter(tr.data, FS)
        z, r, t = [tr.data for tr in st_syn]
        zf, rf, tf = polarization_filter([z, r, t], FS)
        for tr, data in zip(st_syn, [zf, rf, tf]):
            tr.data = data

        logger.info(f"[{event_row['event_id']}] Sintéticos gerados com DSMpy.")
        return st_syn

    except Exception as e:
        logger.error(f"[{event_row['event_id']}] Erro em synthesize_event_traces(): {e}")
        return None

def compute_misfit(obs_st, syn_st):
    try:
        common_channels = set(tr.stats.channel[-1] for tr in obs_st) & \
                          set(tr.stats.channel[-1] for tr in syn_st)
        if not common_channels:
            return np.inf

        misfits = []
        for ch in common_channels:
            o = obs_st.select(channel=f"*{ch}")
            s = syn_st.select(channel=f"*{ch}")
            if len(o) == 0 or len(s) == 0:
                continue

            obs = o[0].data.astype(float)
            syn = s[0].data.astype(float)
            n = min(len(obs), len(syn))
            obs, syn = obs[:n], syn[:n]

            if np.max(np.abs(obs)) > 0:
                obs = obs / np.max(np.abs(obs))
            if np.max(np.abs(syn)) > 0:
                syn = syn / np.max(np.abs(syn))

            obs -= np.mean(obs)
            syn -= np.mean(syn)

            diff = obs - syn
            misfit = np.sqrt(np.mean(diff ** 2))  # entre 0 e 2
            misfit = min(misfit / 2, 1.0)  # reescala 0–1

            misfits.append(misfit)

        mean_misfit = np.mean(misfits) if misfits else np.inf
        logger.debug(f"[DEBUG] Misfit normalizado (Mars-style): {mean_misfit:.4f}")
        return mean_misfit

    except Exception as e:
        logger.error(f"Erro em compute_misfit(): {e}")
        return np.nan

def total_cost(params, events, depth_min, depth_max):
    try:
        vpv, vph, vsv, vsh = params
        p = np.array(params, dtype=float)
        if np.any(p <= 0.0) or np.any(p > 12.0):
            return 999.0, 999.0, 999.0

        model = SeismicModel.test2()
        model.update_mantle(
            vpv=vpv,
            vph=vph,
            vsv=vsv,
            vsh=vsh,
            depth_min=depth_min,
            depth_max=depth_max,
        )

        total_misfit_vp, total_misfit_vs, valid = 0.0, 0.0, 0

        for _, evt in events.iterrows():
            event_id = str(evt["event_id"]).strip()
            obs_st = load_observed_waveforms(event_id)
            syn_st = synthesize_event_traces(evt, model)
            if syn_st is None or obs_st is None:
                continue

            # P in Z + R
            obs_vp = obs_st.select(channel="*Z") + obs_st.select(channel="*R")
            syn_vp = syn_st.select(channel="*Z") + syn_st.select(channel="*R")

            # S in Z + T
            obs_vs = obs_st.select(channel="*Z") + obs_st.select(channel="*T")
            syn_vs = syn_st.select(channel="*Z") + syn_st.select(channel="*T")

            m_vp = compute_misfit(obs_vp, syn_vp)
            m_vs = compute_misfit(obs_vs, syn_vs)

            if np.isfinite(m_vp) and np.isfinite(m_vs):
                total_misfit_vp += m_vp
                total_misfit_vs += m_vs
                valid += 1

        if valid == 0:
            return 999.0, 999.0, 999.0

        mean_vp = total_misfit_vp / valid
        mean_vs = total_misfit_vs / valid
        mean_total = 0.5 * (mean_vp + mean_vs)

        global COST_HISTORY, PARAM_HISTORY
        COST_HISTORY.append(mean_total)
        PARAM_HISTORY.append(params)

        logger.info(
            f"[Iteração camada {depth_min:.0f}-{depth_max:.0f} km] "
            f"vpv={vpv:.2f}, vsv={vsv:.2f} | "
            f"Misfit Vp={mean_vp:.6f}, Vs={mean_vs:.6f}, Total={mean_total:.6f}"
        )

        return mean_total, mean_vp, mean_vs

    except Exception as e:
        logger.error(f"Erro em total_cost() para camada {depth_min}-{depth_max}: {e}")
        return 999.0, 999.0, 999.0

def total_cost_scalar(params, events, depth_min, depth_max):
    mean_total, mean_vp, mean_vs = total_cost(params, events, depth_min, depth_max)
    return float(mean_total)

# ---------------------------------------------------------------------------
# HEATMAP OF MISFIT
# ---------------------------------------------------------------------------
def misfit_vs_depth_heatmap(events, model):
    vp_values = np.linspace(7.0, 10.0, 20)
    vs_values = np.linspace(3.5, 5.5, 20)
    misfit_grid = np.zeros((len(vs_values), len(vp_values)))

    logger.info("Calculando mapa de misfit vs profundidade...")
    for i, vs in enumerate(tqdm(vs_values)):
        for j, vp in enumerate(vp_values):
            model.update_mantle(vpv=vp, vph=vp, vsv=vs, vsh=vs)
            m = total_cost_scalar((vp, vp, vs, vs), events, model)
            misfit_grid[i, j] = m

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(
        misfit_grid,
        extent=[vp_values.min(), vp_values.max(), vs_values.max(), vs_values.min()],
        aspect="auto",
        cmap="viridis_r"
    )
    plt.colorbar(c, ax=ax, label="Misfit médio")
    ax.set_xlabel("Vp (km/s)")
    ax.set_ylabel("Vs (km/s)")
    #ax.set_title("Mapa de Misfit vs. Profundidade (700–1800 km)")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "misfit_heatmap.png"), dpi=300)
    logger.info(f"Mapa salvo em {FIG_DIR}misfit_heatmap.png")

def compute_layer_misfit(events, vp, vs, depth_min, depth_max):
    """
    Retorna (mean_total, mean_vp, mean_vs) para a camada depth_min-depth_max,
    onde:
      - mean_vp usa componentes Z+R (proxy de P)
      - mean_vs usa componentes Z+T (proxy de S)
      - mean_total = 0.5*(mean_vp + mean_vs)

    Obs: Mantém a mesma lógica do seu pipeline; a diferença é só expor
    os misfits separados, para que possamos:
      (i) fazer Fig. 3A com misfit de P (Vp),
      (ii) Fig. 3B com misfit de S (Vs),
      (iii) ajustar um modelo "completo" com descontinuidades.
    """
    try:
        model_layer = SeismicModel.test2()
        model_layer.update_mantle(
            vpv=vp,
            vph=vp,
            vsv=vs,
            vsh=vs,
            depth_min=depth_min,
            depth_max=depth_max,
        )

        total_misfit_vp, total_misfit_vs, valid = 0.0, 0.0, 0

        for _, evt in events.iterrows():
            event_id = str(evt["event_id"]).strip()
            obs_st = load_observed_waveforms(event_id)
            syn_st = synthesize_event_traces(evt, model_layer)
            if syn_st is None or obs_st is None:
                continue

            # P in Z + R
            obs_vp = obs_st.select(channel="*Z") + obs_st.select(channel="*R")
            syn_vp = syn_st.select(channel="*Z") + syn_st.select(channel="*R")

            # S in Z + T
            obs_vs = obs_st.select(channel="*Z") + obs_st.select(channel="*T")
            syn_vs = syn_st.select(channel="*Z") + syn_st.select(channel="*T")

            m_vp = compute_misfit(obs_vp, syn_vp)
            m_vs = compute_misfit(obs_vs, syn_vs)

            if np.isfinite(m_vp) and np.isfinite(m_vs):
                total_misfit_vp += m_vp
                total_misfit_vs += m_vs
                valid += 1

        if valid == 0:
            return 999.0, 999.0, 999.0

        mean_vp = total_misfit_vp / valid
        mean_vs = total_misfit_vs / valid
        mean_total = 0.5 * (mean_vp + mean_vs)

        return float(mean_total), float(mean_vp), float(mean_vs)

    except Exception as e:
        logger.error(
            f"Erro em compute_layer_misfit(Vp={vp:.3f}, Vs={vs:.3f}, "
            f"depth={depth_min}-{depth_max}): {e}"
        )
        return 999.0, 999.0, 999.0

        mean_vp = total_misfit_vp / valid
        mean_vs = total_misfit_vs / valid
        mean_total = 0.5 * (mean_vp + mean_vs)

        return float(mean_total)

    except Exception as e:
        logger.error(
            f"Erro em compute_layer_misfit(Vp={vp:.3f}, Vs={vs:.3f}, "
            f"depth={depth_min}-{depth_max}): {e}"
        )
        return 999.0

def pnas_like_normalize_misfit(misfit_grid, q_clip=0.70, gamma=0.60, eps=1e-12):
    """
    Normaliza misfit para ficar estilo PNAS:
      - valores acima do quantil q_clip ficam ~1 (branco, com Blues_r)
      - valores próximos ao mínimo ficam escuros
      - gamma < 1 deixa o fundo ainda mais branco (puxa valores pra cima)
    """
    import numpy as np

    g = np.asarray(misfit_grid, dtype=float)
    g = np.where(np.isfinite(g), g, np.nan)

    mn = np.nanmin(g)
    qv = np.nanquantile(g, q_clip)

    denom = (qv - mn) if (qv - mn) > eps else eps
    x = (g - mn) / denom
    x = np.clip(x, 0.0, 1.0)

    # gamma < 1 => “branqueia” a maior parte do mapa
    if gamma is not None and gamma > 0:
        x = np.power(x, gamma)

    # substitui NaNs por 1 (branco)
    x = np.where(np.isfinite(x), x, 1.0)
    return x

def plot_layered_fig3_heatmaps(
    depth_windows,
    step,
    vp_values,
    vs_values,
    misfit_grid_vp,   # esperado: (n_depth, n_vp) já marginalizado (min em Vs)
    misfit_grid_vs,   # esperado: (n_depth, n_vs) já marginalizado (min em Vp)
    fig_dir,
    q_clip=0.70,
    gamma=0.60,
    vp_curve=None,    # arrays (zfine, vp(zfine)) ou None
    vs_curve=None,    # arrays (zfine, vs(zfine)) ou None
):
    """
    Fig. 3A/3B estilo PNAS (heatmap com fundo branco + linha vermelha tracejada).

    Importante:
      - O heatmap deve representar um "vale" de misfit, então usamos grids já
        marginalizados (minimizando o parâmetro "não plotado").
      - A curva vermelha deve ser um MODELO COMPLETO (perfil suave) quando fornecido
        via vp_curve / vs_curve.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)

    depth_windows = np.asarray(depth_windows, dtype=float)
    vp_values = np.asarray(vp_values, dtype=float)
    vs_values = np.asarray(vs_values, dtype=float)

    # ---- EDGES (camadas são [dmin, dmin+step]) ----
    depth_edges = np.append(depth_windows, depth_windows[-1] + step)

    dvp = vp_values[1] - vp_values[0]
    vp_edges = np.concatenate(([vp_values[0] - 0.5 * dvp], vp_values + 0.5 * dvp))

    dvs = vs_values[1] - vs_values[0]
    vs_edges = np.concatenate(([vs_values[0] - 0.5 * dvs], vs_values + 0.5 * dvs))

    # ---- Normalização estilo PNAS (fundo branco) ----
    misfit_vp_norm = pnas_like_normalize_misfit(misfit_grid_vp, q_clip=q_clip, gamma=gamma)
    misfit_vs_norm = pnas_like_normalize_misfit(misfit_grid_vs, q_clip=q_clip, gamma=gamma)

    # =======================
    # Painel A: Vp (P)
    # =======================
    plt.figure(figsize=(4.5, 7))
    plt.pcolormesh(
        vp_edges,
        depth_edges,
        misfit_vp_norm,
        cmap="Blues_r",
        shading="auto",
    )

    if vp_curve is not None:
        zfine, vpfine = vp_curve
        plt.plot(vpfine, zfine, "r--", lw=2.8)

    plt.gca().invert_yaxis()
    plt.xlabel("Vp (km/s)")
    plt.ylabel("Profundidade (km)")
    plt.text(0.04, 0.06, "A", transform=plt.gca().transAxes, fontweight="bold", fontsize=16)

    cb = plt.colorbar()
    cb.set_label("Misfit (P) normalizado")

    outA = os.path.join(fig_dir, "vp_layered_fig3A.png")
    plt.tight_layout()
    plt.savefig(outA, dpi=300)
    plt.close()

    # =======================
    # Painel B: Vs (S)
    # =======================
    plt.figure(figsize=(4.5, 7))
    plt.pcolormesh(
        vs_edges,
        depth_edges,
        misfit_vs_norm,
        cmap="Blues_r",
        shading="auto",
    )

    if vs_curve is not None:
        zfine, vsfine = vs_curve
        plt.plot(vsfine, zfine, "r--", lw=2.8)

    plt.gca().invert_yaxis()
    plt.xlabel("Vs (km/s)")
    plt.ylabel("Profundidade (km)")
    plt.text(0.04, 0.06, "B", transform=plt.gca().transAxes, fontweight="bold", fontsize=16)

    cb = plt.colorbar()
    cb.set_label("Misfit (S) normalizado")

    outB = os.path.join(fig_dir, "vs_layered_fig3B.png")
    plt.tight_layout()
    plt.savefig(outB, dpi=300)
    plt.close()

    return outA, outB


def plot_pnas_style_transition(
    depths_center,
    vp_best_list,
    vs_best_list,
    output_dir,
    w_bounds=(40.0, 400.0),
    d0_bounds=None,
):
    """
    Figura adicional (2 painéis) estilo PNAS:
      - pontos: resultado por camada (layered inversion)
      - linha: modelo logístico suave (Vp e Vs juntos) com mesmo d0 e mesma largura w

    Ajuste conjunto: [vp_low, vp_high, vs_low, vs_high, d0, w]
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    os.makedirs(output_dir, exist_ok=True)

    z = np.asarray(depths_center, float)
    vp_obs = np.asarray(vp_best_list, float)
    vs_obs = np.asarray(vs_best_list, float)

    # --- logístico ---
    def logistic(z, v_low, v_high, d0, w):
        return v_low + (v_high - v_low) / (1.0 + np.exp(-(z - d0) / w))

    # --- modelo conjunto (mesmo d0 e w) ---
    def joint_model(z_concat, vp_low, vp_high, vs_low, vs_high, d0, w):
        z1 = z_concat[: len(z)]
        z2 = z_concat[len(z) :]
        return np.concatenate(
            [logistic(z1, vp_low, vp_high, d0, w),
             logistic(z2, vs_low, vs_high, d0, w)]
        )

    z_concat = np.concatenate([z, z])
    v_concat = np.concatenate([vp_obs, vs_obs])

    # chute inicial mais estável: transição onde a derivada do "médio" é maior
    vmean = 0.5 * (vp_obs / (np.nanmax(vp_obs) + 1e-12) + vs_obs / (np.nanmax(vs_obs) + 1e-12))
    dv = np.gradient(vmean)
    d0_guess = z[int(np.nanargmax(np.abs(dv)))]
    w_guess = 120.0

    p0 = [
        float(np.nanmin(vp_obs)), float(np.nanmax(vp_obs)),
        float(np.nanmin(vs_obs)), float(np.nanmax(vs_obs)),
        float(d0_guess), float(w_guess),
    ]

    # bounds (evita w ~ 0 -> degrau vertical)
    if d0_bounds is None:
        d0_bounds = (float(np.nanmin(z)) - 50.0, float(np.nanmax(z)) + 50.0)

    lower = [
        float(np.nanmin(vp_obs) - 2.0), float(np.nanmin(vp_obs) - 2.0),
        float(np.nanmin(vs_obs) - 2.0), float(np.nanmin(vs_obs) - 2.0),
        float(d0_bounds[0]), float(w_bounds[0]),
    ]
    upper = [
        float(np.nanmax(vp_obs) + 2.0), float(np.nanmax(vp_obs) + 2.0),
        float(np.nanmax(vs_obs) + 2.0), float(np.nanmax(vs_obs) + 2.0),
        float(d0_bounds[1]), float(w_bounds[1]),
    ]

    popt, _ = curve_fit(
        joint_model,
        z_concat,
        v_concat,
        p0=p0,
        bounds=(lower, upper),
        maxfev=40000,
    )

    vp_low, vp_high, vs_low, vs_high, d0, w = popt

    zfine = np.linspace(z.min(), z.max(), 600)
    vp_smooth = logistic(zfine, vp_low, vp_high, d0, w)
    vs_smooth = logistic(zfine, vs_low, vs_high, d0, w)

    # --- figura ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 7), sharey=True)

    ax = axes[0]
    ax.scatter(vp_obs, z, c="k", s=28, label="Layered inversion")
    ax.plot(vp_smooth, zfine, "r--", lw=2.5, label="PNAS-style logistic model")
    ax.invert_yaxis()
    ax.set_xlabel("Vp (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.text(0.02, 0.05, "A", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    ax.scatter(vs_obs, z, c="k", s=28, label="Layered inversion")
    ax.plot(vs_smooth, zfine, "r--", lw=2.5, label="PNAS-style logistic model")
    ax.invert_yaxis()
    ax.set_xlabel("Vs (km/s)")
    ax.text(0.02, 0.05, "B", transform=ax.transAxes, fontweight="bold")

    for ax in axes:
        ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    out = os.path.join(output_dir, "vp_vs_pnas_style_transition.png")
    plt.savefig(out, dpi=300)
    plt.close()

    return out, {"vp_low": vp_low, "vp_high": vp_high, "vs_low": vs_low, "vs_high": vs_high, "d0": d0, "w": w}


# ---------------------------------------------------------------------------
# AJUSTE "MODELO COMPLETO" COM DESCONTINUIDADES (MTZ)
# ---------------------------------------------------------------------------
def _sigmoid(z, d0, w):
    z = np.asarray(z, float)
    return 1.0 / (1.0 + np.exp(-(z - d0) / w))

def multistep_profile(z, v0, deltas, d_list, w_list):
    """
    Perfil do tipo soma de sigmoides (multi-step):
        v(z) = v0 + Σ Δ_k * sigmoid(z, d_k, w_k)

    d_k: profundidades das descontinuidades
    w_k: larguras (km) das transições
    """
    z = np.asarray(z, float)
    d_list = np.asarray(d_list, float)
    w_list = np.asarray(w_list, float)
    deltas = np.asarray(deltas, float)

    v = np.full_like(z, float(v0), dtype=float)
    for dk, wk, dv in zip(d_list, w_list, deltas):
        v += dv * _sigmoid(z, dk, wk)
    return v

def build_layer_interpolators(vp_values, vs_values, misfit_cube, fill_value=999.0):
    """
    Constrói um interpolador (RegularGridInterpolator) por camada.
    misfit_cube: (n_depth, n_vp, n_vs)
    """
    vp_values = np.asarray(vp_values, float)
    vs_values = np.asarray(vs_values, float)
    interps = []
    for i in range(misfit_cube.shape[0]):
        g = np.asarray(misfit_cube[i], float)
        interps.append(
            RegularGridInterpolator(
                (vp_values, vs_values),
                g,
                bounds_error=False,
                fill_value=float(fill_value),
            )
        )
    return interps

def fit_discontinuities_from_cube(
    depths_center,
    vp_values,
    vs_values,
    cube_target,
    n_disc=3,
    depth_bounds=None,
    w_bounds=(5.0, 120.0),
    vpad=1.0,
    maxiter=250,
    popsize=25,
    seed=0,
):
    """
    Ajusta um modelo global (perfil completo) com n_disc descontinuidades
    diretamente sobre um cubo de misfit por camada (Vp×Vs).

    Retorna:
      - dict com parâmetros (vp0, vs0, deltas_vp, deltas_vs, d, w),
      - curvas (zfine, vp(zfine), vs(zfine)),
      - custo final.
    """
    z = np.asarray(depths_center, float)
    zmin, zmax = float(np.min(z)), float(np.max(z))

    if depth_bounds is None:
        depth_bounds = (zmin, zmax)
    d_lo, d_hi = float(depth_bounds[0]), float(depth_bounds[1])

    vp_values = np.asarray(vp_values, float)
    vs_values = np.asarray(vs_values, float)
    vp_lo, vp_hi = float(vp_values.min() - vpad), float(vp_values.max() + vpad)
    vs_lo, vs_hi = float(vs_values.min() - vpad), float(vs_values.max() + vpad)

    interps = build_layer_interpolators(vp_values, vs_values, cube_target)

    # parâmetros: [vp0, vs0, dvp_1..K, dvs_1..K, d_1..K, w_1..K]
    K = int(n_disc)
    bounds = []
    bounds.append((vp_lo, vp_hi))  # vp0
    bounds.append((vs_lo, vs_hi))  # vs0

    # deltas
    for _ in range(K):
        bounds.append((-2.5, 2.5))  # dvp
    for _ in range(K):
        bounds.append((-2.0, 2.0))  # dvs

    # d_k
    for _ in range(K):
        bounds.append((d_lo, d_hi))

    # w_k
    for _ in range(K):
        bounds.append((float(w_bounds[0]), float(w_bounds[1])))

    def unpack(p):
        vp0 = float(p[0]); vs0 = float(p[1])
        dvp = np.asarray(p[2:2+K], float)
        dvs = np.asarray(p[2+K:2+2*K], float)
        d = np.asarray(p[2+2*K:2+3*K], float)
        w = np.asarray(p[2+3*K:2+4*K], float)

        # ordena por profundidade (mantém o modelo identificável)
        order = np.argsort(d)
        d = d[order]
        w = w[order]
        dvp = dvp[order]
        dvs = dvs[order]
        return vp0, vs0, dvp, dvs, d, w

    def objective(p):
        vp0, vs0, dvp, dvs, d, w = unpack(p)

        vp_pred = multistep_profile(z, vp0, dvp, d, w)
        vs_pred = multistep_profile(z, vs0, dvs, d, w)

        # penaliza sair do grid (evita soluções não físicas/fora do domínio)
        pen = 0.0
        if np.any(vp_pred < vp_values.min()) or np.any(vp_pred > vp_values.max()):
            pen += 50.0
        if np.any(vs_pred < vs_values.min()) or np.any(vs_pred > vs_values.max()):
            pen += 50.0

        pts = np.column_stack([vp_pred, vs_pred])
        cost = 0.0
        for i, interp in enumerate(interps):
            cost += float(interp(pts[i]))
        return cost + pen

    res = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=int(maxiter),
        popsize=int(popsize),
        tol=1e-6,
        seed=int(seed),
        polish=True,
        updating="deferred",
        workers=1,
    )

    vp0, vs0, dvp, dvs, d, w = unpack(res.x)

    zfine = np.linspace(zmin, zmax, 800)
    vp_fine = multistep_profile(zfine, vp0, dvp, d, w)
    vs_fine = multistep_profile(zfine, vs0, dvs, d, w)

    out = {
        "vp0": vp0, "vs0": vs0,
        "dvp": dvp, "dvs": dvs,
        "d": d, "w": w,
        "fun": float(res.fun),
        "success": bool(res.success),
        "message": str(res.message),
    }
    curves = (zfine, vp_fine, vs_fine)
    return out, curves

def plot_transition_with_discontinuities(
    depths_center,
    vp_best_list,
    vs_best_list,
    model_curves,   # (zfine, vp_fine, vs_fine)
    disc_depths,    # array d_k
    output_dir,
    filename="vp_vs_pnas_style_transition.png",
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    z = np.asarray(depths_center, float)
    vp_obs = np.asarray(vp_best_list, float)
    vs_obs = np.asarray(vs_best_list, float)

    zfine, vp_fine, vs_fine = model_curves
    d = np.asarray(disc_depths, float)

    fig, axes = plt.subplots(1, 2, figsize=(8, 7), sharey=True)

    ax = axes[0]
    ax.scatter(vp_obs, z, c="k", s=28, label="Layered inversion")
    ax.plot(vp_fine, zfine, "r--", lw=2.5, label="Best-fitting full model")
    for dk in d:
        ax.axhline(dk, color="r", lw=1.2, alpha=0.55)
    ax.invert_yaxis()
    ax.set_xlabel("Vp (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.text(0.02, 0.05, "A", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    ax.scatter(vs_obs, z, c="k", s=28, label="Layered inversion")
    ax.plot(vs_fine, zfine, "r--", lw=2.5, label="Best-fitting full model")
    for dk in d:
        ax.axhline(dk, color="r", lw=1.2, alpha=0.55)
    ax.invert_yaxis()
    ax.set_xlabel("Vs (km/s)")
    ax.text(0.02, 0.05, "B", transform=ax.transAxes, fontweight="bold")

    for ax in axes:
        ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=300)
    plt.close()
    return out

# ---------------------------------------------------------------------------
# PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    logger.info("=== Iniciando inversão camada a camada (step = 20 km) ===")

    # ------------------------------------------------------------------
    # 1) Carrega eventos
    # ------------------------------------------------------------------
    events = load_event_catalog(CSV_FILE)
    events = events.dropna(subset=["event_id", "latitude", "longitude", "depth"])
    events.reset_index(drop=True, inplace=True)
    logger.info(f"{len(events)} eventos válidos após limpeza.")

    # ------------------------------------------------------------------
    # 2) Modelo base (apenas referência)
    # ------------------------------------------------------------------
    base_model = SeismicModel.test2()
    logger.info("Modelo sísmico base carregado (SeismicModel.test2).")

    # ------------------------------------------------------------------
    # 3) Profundidades (camadas) + grid de busca
    # ------------------------------------------------------------------
    depth_windows = np.arange(800, 1300, 20)
    step = 20.0
    depths_center = depth_windows + 0.5 * step

    n_vp = 21
    n_vs = 21
    vp_values = np.linspace(6.5, 9.5, n_vp)
    vs_values = np.linspace(3.5, 5.0, n_vs)

    # ------------------------------------------------------------------
    # 4) Saídas da inversão por camada
    # ------------------------------------------------------------------
    vp_best_list = []
    vs_best_list = []
    misfit_best_list = []

    # Mapas estilo Fig. 3A/3B (já marginalizados)
    #   - Fig 3A (Vp): misfit_P marginalizado em Vs  -> shape (n_depth, n_vp)
    #   - Fig 3B (Vs): misfit_S marginalizado em Vp  -> shape (n_depth, n_vs)
    misfit_grid_vp = np.zeros((len(depth_windows), n_vp))
    misfit_grid_vs = np.zeros((len(depth_windows), n_vs))

    # Cubos completos por camada (Vp×Vs), necessários para ajustar "modelo completo"
    misfit_cube_total = np.zeros((len(depth_windows), n_vp, n_vs))
    misfit_cube_vp = np.zeros((len(depth_windows), n_vp, n_vs))  # P (Z+R)
    misfit_cube_vs = np.zeros((len(depth_windows), n_vp, n_vs))  # S (Z+T)

    # ------------------------------------------------------------------
    # 5) Loop camada a camada
    # ------------------------------------------------------------------
    for i, dmin in enumerate(depth_windows):
        dmax = dmin + step
        logger.info(f"--- Invertendo camada {dmin:.0f}-{dmax:.0f} km (grid search) ---")

        misfit_layer_total = np.zeros((n_vp, n_vs))
        misfit_layer_vp = np.zeros((n_vp, n_vs))
        misfit_layer_vs = np.zeros((n_vp, n_vs))

        for i_vp, vp in enumerate(vp_values):
            for i_vs, vs in enumerate(vs_values):
                m_tot, m_vp, m_vs = compute_layer_misfit(events, vp, vs, dmin, dmax)
                misfit_layer_total[i_vp, i_vs] = m_tot
                misfit_layer_vp[i_vp, i_vs] = m_vp
                misfit_layer_vs[i_vp, i_vs] = m_vs

        # guarda cubos completos (para o "modelo completo")
        misfit_cube_total[i, :, :] = misfit_layer_total
        misfit_cube_vp[i, :, :] = misfit_layer_vp
        misfit_cube_vs[i, :, :] = misfit_layer_vs

        # Fig. 3A/3B: "vale" de misfit (marginalização)
        # Vp x depth (P): min em Vs
        misfit_grid_vp[i, :] = np.min(misfit_layer_vp, axis=1)

        # Vs x depth (S): min em Vp
        misfit_grid_vs[i, :] = np.min(misfit_layer_vs, axis=0)

        # pontos pretos: melhor Vp e Vs (a partir dos vetores marginalizados)
        vp_best = vp_values[int(np.argmin(misfit_grid_vp[i, :]))]
        vs_best = vs_values[int(np.argmin(misfit_grid_vs[i, :]))]

        # misfit_total_min da camada (global 2D)
        idx_min = np.unravel_index(np.argmin(misfit_layer_total), misfit_layer_total.shape)
        misfit_best = float(misfit_layer_total[idx_min])

        vp_best_list.append(vp_best)
        vs_best_list.append(vs_best)
        misfit_best_list.append(misfit_best)

        logger.info(
            f"Camada {dmin:.0f}-{dmax:.0f} km | "
            f"Vp*={vp_best:.2f} km/s, Vs*={vs_best:.2f} km/s, misfit_total_min={misfit_best:.4f}"
        )

    # Converte listas para arrays numpy
    vp_best_list = np.asarray(vp_best_list, float)
    vs_best_list = np.asarray(vs_best_list, float)
    misfit_best_list = np.asarray(misfit_best_list, float)

    # ------------------------------------------------------------------
    # 6) Ajuste "modelo completo" com descontinuidades (multi-step)
    # ------------------------------------------------------------------
    # Se você quer X + top + intermediate + bottom => 4 descontinuidades:
    N_DISC = 4

    logger.info("Ajustando modelo completo com descontinuidades (multi-step)...")
    depth_bounds = (float(depth_windows.min()), float(depth_windows.max() + step))

    # Best-fitting para P e S (para Fig. 3A e Fig. 3B separadamente, como no paper)
    best_P, curves_P = fit_discontinuities_from_cube(
        depths_center=depths_center,
        vp_values=vp_values,
        vs_values=vs_values,
        cube_target=misfit_cube_vp,
        n_disc=N_DISC,
        depth_bounds=depth_bounds,
        w_bounds=(5.0, 120.0),
        maxiter=260,
        popsize=24,
        seed=1,
    )

    best_S, curves_S = fit_discontinuities_from_cube(
        depths_center=depths_center,
        vp_values=vp_values,
        vs_values=vs_values,
        cube_target=misfit_cube_vs,
        n_disc=N_DISC,
        depth_bounds=depth_bounds,
        w_bounds=(5.0, 120.0),
        maxiter=260,
        popsize=24,
        seed=2,
    )

    # Best-fitting global (TOTAL) para estimar as descontinuidades principais
    best_TOT, curves_TOT = fit_discontinuities_from_cube(
        depths_center=depths_center,
        vp_values=vp_values,
        vs_values=vs_values,
        cube_target=misfit_cube_total,
        n_disc=N_DISC,
        depth_bounds=depth_bounds,
        w_bounds=(5.0, 120.0),
        maxiter=320,
        popsize=28,
        seed=3,
    )

    logger.info(f"Best P fun={best_P['fun']:.4f} | d={np.round(best_P['d'],1)}")
    logger.info(f"Best S fun={best_S['fun']:.4f} | d={np.round(best_S['d'],1)}")
    logger.info(f"Best TOT fun={best_TOT['fun']:.4f} | d={np.round(best_TOT['d'],1)}")

    # ------------------------------------------------------------------
    # 7) Figuras Fig. 3A/3B estilo PNAS (heatmaps) + transição com linhas
    # ------------------------------------------------------------------
    os.makedirs(FIG_DIR, exist_ok=True)

    # Fig. 3A: Vp heatmap (misfit_P) + curva do best_P (Vp)
    vp_curve = (curves_P[0], curves_P[1])
    # Fig. 3B: Vs heatmap (misfit_S) + curva do best_S (Vs)
    vs_curve = (curves_S[0], curves_S[2])

    plot_layered_fig3_heatmaps(
        depth_windows=depth_windows,
        step=step,
        vp_values=vp_values,
        vs_values=vs_values,
        misfit_grid_vp=misfit_grid_vp,
        misfit_grid_vs=misfit_grid_vs,
        fig_dir=FIG_DIR,
        q_clip=0.70,
        gamma=0.60,
        vp_curve=vp_curve,
        vs_curve=vs_curve,
    )

    # Figura adicional: pontos pretos + best_TOT + linhas nas descontinuidades
    plot_transition_with_discontinuities(
        depths_center=depths_center,
        vp_best_list=vp_best_list,
        vs_best_list=vs_best_list,
        model_curves=curves_TOT,
        disc_depths=best_TOT["d"],
        output_dir=FIG_DIR,
        filename="vp_vs_pnas_style_transition.png",
    )

    # ------------------------------------------------------------------
    # 8) Salva resultados (CSV + TXT de descontinuidades)
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    disc_txt = os.path.join(OUTPUT_DIR, "mtz_discontinuities_bestTOT.txt")
    with open(disc_txt, "w") as f:
        f.write("Best-fitting MTZ discontinuities (from TOTAL misfit)\n")
        f.write(f"N_DISC = {N_DISC}\n")
        f.write(f"fun = {best_TOT['fun']:.6f}\n\n")
        for k, (dk, wk) in enumerate(zip(best_TOT["d"], best_TOT["w"]), start=1):
            f.write(f"disc_{k}: depth_km = {dk:.3f}   width_km = {wk:.3f}\n")

    df = pd.DataFrame(
        {
            "depth_min_km": depth_windows,
            "depth_max_km": depth_windows + step,
            "depth_center_km": depths_center,
            "vp_best_kms": vp_best_list,
            "vs_best_kms": vs_best_list,
            "misfit_total_min": misfit_best_list,
        }
    )
    out_csv = os.path.join(OUTPUT_DIR, "layered_inversion_results.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"Resultados por camada salvos em {out_csv}")
    logger.info(f"Descontinuidades (TOTAL) salvas em {disc_txt}")

    logger.info("=== Fim da inversão ===")


if __name__ == "__main__":
    main()

