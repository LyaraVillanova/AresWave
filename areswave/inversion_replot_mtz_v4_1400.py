from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import PowerNorm

DEPTH_MIN_KM = 700.0
DEPTH_MAX_KM = 1800.0
SURFACE_RADIUS_KM = 3389.5

RESULTS_CSV = "/home/lyara/areswave/outputs_modelsdir/pnas_equivalent_suite_results_S0345a.csv"
MODELS_DIR = "/home/lyara/areswave/models/"
OUTDIR = "/home/lyara/areswave/figs_modelsdir/"
ORIGINAL_SCRIPT = "/home/lyara/areswave/areswave/"

# ============================================================
# IO helpers
# ============================================================
def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _coerce_profile_1d(a) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(a, dtype=float)
    except Exception:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size >= 2:
        return arr.astype(float)
    if arr.ndim == 2 and min(arr.shape) >= 2:
        axis = 0 if arr.shape[0] <= arr.shape[1] else 1
        out = np.nanmean(arr, axis=axis)
        out = np.asarray(out, dtype=float).squeeze()
        if out.ndim == 1 and out.size >= 2:
            return out.astype(float)
    return None


def _find_matching_array(mapping: Dict[str, np.ndarray], aliases) -> Optional[np.ndarray]:
    lower_map = {str(k).lower(): v for k, v in mapping.items()}
    for alias in aliases:
        if alias.lower() in lower_map:
            arr = _coerce_profile_1d(lower_map[alias.lower()])
            if arr is not None:
                return arr
    for alias in aliases:
        for k, v in lower_map.items():
            if alias.lower() in k:
                arr = _coerce_profile_1d(v)
                if arr is not None:
                    return arr
    return None


def _infer_depth_axis(x: np.ndarray, surface_radius_km: float) -> Tuple[np.ndarray, str]:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("Eixo de profundidade/radius insuficiente.")

    first = float(x[0])
    last = float(x[-1])
    xmax = float(np.nanmax(x))
    diffs = np.diff(x)
    inc = bool(np.all(diffs >= -1e-6))
    dec = bool(np.all(diffs <= 1e-6))
    near0_first = abs(first) <= 10.0
    near0_last = abs(last) <= 10.0
    nearR_first = abs(first - surface_radius_km) <= 25.0
    nearR_last = abs(last - surface_radius_km) <= 25.0

    if inc and near0_first and nearR_last:
        return x.astype(float), "depth"
    if dec and nearR_first and near0_last:
        return (surface_radius_km - x).astype(float), "radius"

    if xmax <= 10.0:
        x_mm = x * 1000.0
        nearR_first_mm = abs(float(x[0]) * 1000.0 - surface_radius_km) <= max(5.0, 0.02 * surface_radius_km)
        nearR_last_mm = abs(float(x[-1]) * 1000.0 - surface_radius_km) <= max(5.0, 0.02 * surface_radius_km)
        if (dec and nearR_first_mm) or ((not dec) and nearR_last_mm):
            return (surface_radius_km - x_mm).astype(float), "radius_Mm"

    is_radius_like = (xmax > 2500.0) and nearR_first
    if is_radius_like:
        return (surface_radius_km - x).astype(float), "radius"

    return x.astype(float), "depth"


def _guess_velocity_columns(arr: np.ndarray) -> Tuple[int, int]:
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Arquivo sem colunas suficientes para Vp/Vs.")

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
            score += np.mean((vsv > 0.0) & (vsv < 8.5))
            score += np.mean(vpv > vsv)
            item = (score, i, j)
            if (best is None) or (item[0] > best[0]):
                best = item

    if best is None:
        raise ValueError("Nao consegui inferir as colunas de Vp e Vs.")
    return int(best[1]), int(best[2])


def _clean_profile(z_km: np.ndarray, v_kms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z_km, dtype=float)
    v = np.asarray(v_kms, dtype=float)
    ok = np.isfinite(z) & np.isfinite(v)
    z = z[ok]
    v = v[ok]
    if z.size < 2:
        raise ValueError("Perfil insuficiente apos limpeza.")
    order = np.argsort(z)
    z = z[order]
    v = v[order]
    z, idx = np.unique(z, return_index=True)
    v = v[idx]
    mask = (z >= DEPTH_MIN_KM) & (z <= DEPTH_MAX_KM)
    z = z[mask]
    v = v[mask]
    if z.size < 2:
        raise ValueError("Perfil nao cobre 700-1800 km.")
    return z, v


def _resolve_model_path(source_path: str, models_dir: Optional[str] = None) -> Path:
    src = Path(source_path)
    if src.exists():
        return src
    if models_dir is not None:
        alt = Path(models_dir) / src.name
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Modelo nao encontrado: {source_path}")


def load_profile_from_model(source_path: str, models_dir: Optional[str] = None, surface_radius_km: float = SURFACE_RADIUS_KM):
    path = _resolve_model_path(source_path, models_dir=models_dir)
    suffix = path.suffix.lower()

    if suffix == ".nd":
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
                if ok and len(vals) >= 3:
                    rows.append(vals)
        if len(rows) < 3:
            raise ValueError(f"{path}: poucas linhas numericas legiveis.")

        arr = np.asarray(rows, dtype=float)

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
            vp_kms = np.asarray(vp_vals, dtype=float)
            vs_kms = np.asarray(vs_vals, dtype=float)
        else:
            vp_idx, vs_idx = _guess_velocity_columns(arr)
            z_raw = arr[:, 0]
            vp_kms = arr[:, vp_idx]
            vs_kms = arr[:, vs_idx]
            if (
                z_raw.size > 1
                and np.all(np.diff(z_raw) >= -1e-6)
                and abs(float(z_raw[0])) <= 10.0
                and abs(float(z_raw[-1]) - surface_radius_km) <= 25.0
            ):
                z_km = z_raw.copy()
            else:
                z_km, _ = _infer_depth_axis(z_raw, surface_radius_km)

    elif suffix == ".npz":
        raw = np.load(path, allow_pickle=True)
        mapping: Dict[str, np.ndarray] = {k: raw[k] for k in raw.files}
        z_raw = _find_matching_array(mapping, ["z_km", "depth_km", "depth", "z", "radius_km", "r_km", "radius", "r"])
        vp_kms = _find_matching_array(mapping, ["vp_kms", "vp", "vpv", "vph"])
        vs_kms = _find_matching_array(mapping, ["vs_kms", "vs", "vsv", "vsh"])
        if z_raw is None or vp_kms is None or vs_kms is None:
            raise ValueError(f"{path}: npz sem arrays reconheciveis de z/vp/vs.")
        z_km, _ = _infer_depth_axis(z_raw, surface_radius_km)
    else:
        raise ValueError(f"Formato de modelo nao suportado aqui: {suffix}")

    z_vp, vp_kms = _clean_profile(z_km, vp_kms)
    z_vs, vs_kms = _clean_profile(z_km, vs_kms)

    z_common = np.union1d(z_vp, z_vs)
    vp_common = np.interp(z_common, z_vp, vp_kms)
    vs_common = np.interp(z_common, z_vs, vs_kms)

    return {
        "z_km": z_common,
        "vp_kms": vp_common,
        "vs_kms": vs_common,
        "path": str(path),
    }


# ============================================================
# Figure 1: scatter categorizado por misfit
# ============================================================
def _misfit_category_color(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    colors = np.empty(values.shape, dtype=object)
    colors[values < 0.25] = "deepskyblue"
    colors[(values >= 0.25) & (values <= 0.5)] = "springgreen"
    colors[(values >= 0.5) & (values <= 0.75)] = "khaki"
    colors[values > 0.75] = "indianred"
    colors[~np.isfinite(values)] = "lightgray"
    return colors


def plot_scatter_categorized(results_csv: str, outpath: str):
    df = pd.read_csv(results_csv)
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

    fig, axes = plt.subplots(5, 3, figsize=(13.5, 16.5), sharex=False, sharey=False)
    for i, (dcol, dlabel) in enumerate(depth_cols):
        for j, (mcol, mlabel) in enumerate(misfit_cols):
            ax = axes[i, j]
            sub = df.dropna(subset=[dcol, mcol]).copy()
            colors = _misfit_category_color(sub[mcol].values)
            ax.scatter(
                sub[dcol].values,
                sub[mcol].values,
                s=48,
                c=colors,
                alpha=0.55,
                edgecolors="k",
                linewidths=0.35,
            )
            ax.set_xlabel(f"Depth near {dlabel} (km)")
            ax.set_ylabel(mlabel)
            if i == 0:
                ax.set_title(mlabel)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="deepskyblue", markeredgecolor="k", markersize=8, alpha=0.75, label="misfit < 0.25"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="springgreen", markeredgecolor="k", markersize=8, alpha=0.75, label="0.25 ≤ misfit ≤ 0.5"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="khaki", markeredgecolor="k", markersize=8, alpha=0.75, label="0.5 ≤ misfit ≤ 0.75"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="indianred", markeredgecolor="k", markersize=8, alpha=0.75, label="misfit > 0.5"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.995))
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Figure 2/3: heatmaps de perfis
# ============================================================
def _misfit_to_weights(misfit: np.ndarray) -> np.ndarray:
    m = np.asarray(misfit, dtype=float)
    ok = np.isfinite(m)
    w = np.zeros_like(m, dtype=float)
    if not np.any(ok):
        return w
    m_clip = np.clip(m[ok], 0.0, None)
    w_ok = 1.0 / (m_clip + 0.03)
    w_ok /= np.nanmax(w_ok)
    w[ok] = w_ok
    return w


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return np.nan
    v = values[ok]
    w = weights[ok]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / np.sum(w)
    idx = int(np.searchsorted(cdf, 0.5, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def _collect_profiles(df: pd.DataFrame, value_col: str, models_dir: Optional[str]):
    profiles = []
    bad = []
    for _, row in df.iterrows():
        try:
            prof = load_profile_from_model(row["source_path"], models_dir=models_dir)
            profiles.append(
                {
                    "model_id": row["model_id"],
                    "z_km": prof["z_km"],
                    "v_kms": prof[value_col],
                    "misfit": _safe_float(row["mean_p_misfit"] if value_col == "vp_kms" else row["mean_s_misfit"]),
                }
            )
        except Exception as exc:
            bad.append((row["model_id"], str(exc)))
    return profiles, bad


def _truncated_cmap(name: str = "Greys_r", minval: float = 0.02, maxval: float = 0.82, n: int = 256):
    base = plt.get_cmap(name)
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        f"trunc_{name}_{minval:.2f}_{maxval:.2f}",
        base(np.linspace(minval, maxval, n)),
    )


def plot_profile_heatmap(
    results_csv: str,
    outpath: str,
    value_col: str,
    misfit_col: str,
    xlabel: str,
    cbar_label: str,
    models_dir: Optional[str] = None,
    depth_min_km: float = DEPTH_MIN_KM,
    depth_max_km: float = DEPTH_MAX_KM,
):
    df = pd.read_csv(results_csv)
    df = df.dropna(subset=[misfit_col, "source_path", "model_id"]).copy()
    df = df.sort_values(misfit_col, ascending=True)

    profiles = []
    failed = []
    for _, row in df.iterrows():
        try:
            prof = load_profile_from_model(row["source_path"], models_dir=models_dir)
            profiles.append(
                {
                    "model_id": row["model_id"],
                    "z_km": prof["z_km"],
                    "v_kms": prof[value_col],
                    "misfit": float(row[misfit_col]),
                }
            )
        except Exception as exc:
            failed.append((row["model_id"], str(exc)))

    if not profiles:
        msg = "Nao consegui carregar nenhum perfil."
        if failed:
            msg += f" Primeiro erro: {failed[0][0]} -> {failed[0][1]}"
        raise RuntimeError(msg)

    # grade comum em profundidade para calcular a curva representativa e um fundo
    # de densidade bem suave, sem apagar os perfis individuais.
    depth_grid = np.arange(depth_min_km, depth_max_km + 1.0, 2.0)
    interpolated = []
    misfits = []
    for prof in profiles:
        z = np.asarray(prof["z_km"], dtype=float)
        v = np.asarray(prof["v_kms"], dtype=float)
        ok = np.isfinite(z) & np.isfinite(v)
        z = z[ok]
        v = v[ok]
        if z.size < 2:
            continue
        order = np.argsort(z)
        z = z[order]
        v = v[order]
        z, idx = np.unique(z, return_index=True)
        v = v[idx]

        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))
        overlap_min = max(depth_min_km, zmin)
        overlap_max = min(depth_max_km, zmax)
        if overlap_max <= overlap_min:
            continue

        v_on_grid = np.full(depth_grid.shape, np.nan, dtype=float)
        inside = (depth_grid >= overlap_min) & (depth_grid <= overlap_max)
        if np.count_nonzero(inside) < 2:
            continue
        v_on_grid[inside] = np.interp(depth_grid[inside], z, v)
        interpolated.append(v_on_grid)
        misfits.append(prof["misfit"])

    V = np.asarray(interpolated, dtype=float)
    misfits = np.asarray(misfits, dtype=float)
    weights = _misfit_to_weights(misfits)
    if V.size == 0:
        raise RuntimeError("Nenhum perfil tem dados validos entre 700 e 1800 km.")

    finite_m = misfits[np.isfinite(misfits)]
    if finite_m.size == 0:
        raise RuntimeError("Nao ha misfits finitos para colorir os perfis.")

    # Curva vermelha = mediana ponderada pelos melhores misfits em cada profundidade.
    rep_curve = np.full(depth_grid.size, np.nan, dtype=float)
    for iz in range(depth_grid.size):
        vals = V[:, iz]
        ok = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
        if np.any(ok):
            rep_curve[iz] = _weighted_median(vals[ok], weights[ok])

    # Fundo de densidade muito leve, só para sugerir a região mais povoada sem
    # sumir com as linhas individuais. Isso fica mais próximo do visual do paper.
    vmin = float(np.nanmin(V))
    vmax = float(np.nanmax(V))
    pad = 0.035 * (vmax - vmin if vmax > vmin else 1.0)
    xbins = np.linspace(vmin - pad, vmax + pad, 220)
    ybins = np.linspace(depth_min_km, depth_max_km, depth_grid.size)

    xs, ys, ws = [], [], []
    for row_vals, w in zip(V, weights):
        ok = np.isfinite(row_vals)
        if not np.any(ok):
            continue
        xs.append(row_vals[ok])
        ys.append(depth_grid[ok])
        ws.append(np.full(np.count_nonzero(ok), w, dtype=float))

    H = None
    if xs:
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        ws = np.concatenate(ws)
        H, xe, ye = np.histogram2d(xs, ys, bins=[xbins, ybins], weights=ws)
        # suavizacao leve com kernels separaveis, sem apagar a estrutura principal
        kx = np.array([1, 2, 3, 2, 1], dtype=float)
        ky = np.array([1, 2, 1], dtype=float)
        kx /= kx.sum()
        ky /= ky.sum()
        H = np.apply_along_axis(lambda a: np.convolve(a, kx, mode="same"), 0, H)
        H = np.apply_along_axis(lambda a: np.convolve(a, ky, mode="same"), 1, H)
        pos = H[H > 0]
        if pos.size:
            lo = float(np.nanpercentile(pos, 15))
            hi = float(np.nanpercentile(pos, 99.5))
            if hi > lo:
                H = np.clip((H - lo) / (hi - lo), 0.0, 1.0)
            else:
                H = np.clip(H / max(np.nanmax(H), 1e-12), 0.0, 1.0)

    cmap = _truncated_cmap("Greys_r", 0.02, 0.82)
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=float(np.nanmin(finite_m)), vmax=float(np.nanmax(finite_m)))

    fig, ax = plt.subplots(figsize=(5.2, 7.4))
    ax.set_facecolor((0.96, 0.96, 0.96))

    if H is not None:
        ax.imshow(
            H.T,
            origin="lower",
            aspect="auto",
            extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            alpha=0.22,
            interpolation="bilinear",
            zorder=0,
        )

    # Desenha os piores primeiro e os melhores por cima.
    order = np.argsort(misfits)[::-1]
    for idx in order:
        prof = profiles[idx]
        z = np.asarray(prof["z_km"], dtype=float)
        v = np.asarray(prof["v_kms"], dtype=float)
        ok = np.isfinite(z) & np.isfinite(v) & (z >= depth_min_km) & (z <= depth_max_km)
        z = z[ok]
        v = v[ok]
        if z.size < 2:
            continue
        ord2 = np.argsort(z)
        z = z[ord2]
        v = v[ord2]
        alpha = 0.10 + 0.45 * float(weights[idx])
        lw = 0.65 + 0.45 * float(weights[idx])
        ax.plot(v, z, color=cmap(norm(misfits[idx])), lw=lw, alpha=alpha, zorder=2)

    ax.plot(rep_curve, depth_grid, color="red", lw=3.0, zorder=4)
    ax.set_ylim(depth_max_km, depth_min_km)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth (km)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close(fig)

    if failed:
        print(f"[aviso] {len(failed)} perfis nao puderam ser lidos para {outpath}. Exemplo: {failed[0]}")


# ============================================================
# Figure 4: best model 700-1800 km only
# ============================================================
def plot_best_model_paperlike(results_csv: str, outpath: str, models_dir: Optional[str] = None):
    df = pd.read_csv(results_csv)
    df = df.dropna(subset=["total_misfit"]).copy()
    best = df.sort_values("total_misfit", ascending=True).iloc[0]
    prof = load_profile_from_model(best["source_path"], models_dir=models_dir)

    z = prof["z_km"]
    vp = prof["vp_kms"]
    vs = prof["vs_kms"]
    mask = (z >= DEPTH_MIN_KM) & (z <= DEPTH_MAX_KM)
    z = z[mask]
    vp = vp[mask]
    vs = vs[mask]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 7.2), sharey=True)
    ax_vp, ax_vs, ax_both = axes

    ax_vp.plot(vp, z, color="tab:red", lw=2.0)
    ax_vs.plot(vs, z, color="tab:red", lw=2.0)

    ax_both.plot(vp, z, color="tab:red", lw=2.0, label="Vp")
    ax_both.plot(vs, z, color="tab:blue", lw=2.0, label="Vs")

    layer_info = [
        ("d_800_km", "   800 km"),
        ("d_1000_km", "   1000 km"),
        ("d_1200_km", "   1200 km"),
        ("d_1400_km", "   1400 km"),
        ("d_1600_km", "   1600 km"),
    ]
    colors = ["0.35", "0.45", "0.55", "0.60", "0.65"]
    for (col, label), c in zip(layer_info, colors):
        dval = _safe_float(best.get(col, np.nan))
        if np.isfinite(dval) and DEPTH_MIN_KM <= dval <= DEPTH_MAX_KM:
            for ax in axes:
                ax.axhline(dval, color=c, ls="--", lw=1.2)
            ax_vs.text(
                0.98,
                dval,
                f" {label}: {dval:.1f} km",
                transform=ax_vs.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=9,
                color=c,
            )

    ax_vp.set_ylim(DEPTH_MAX_KM, DEPTH_MIN_KM)
    ax_vp.set_xlabel("Vp (km/s)")
    ax_vs.set_xlabel("Vs (km/s)")
    ax_both.set_xlabel("Velocity (km/s)")
    ax_vp.set_ylabel("Depth (km)")
    ax_both.legend(loc="best", fontsize=9)

    fig.suptitle(f"Best-fitting full model: {best['model_id']}", y=0.98)

    plt.tight_layout()
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close(fig)



def _extract_fig_dir_from_original_script(path: Path) -> Optional[Path]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    import re
    m = re.search(r"^\s*FIG_DIR\s*=\s*[\"\']([^\"\']+)[\"\']", text, flags=re.MULTILINE)
    if not m:
        return None
    return Path(m.group(1)).expanduser().resolve()


# ============================================================
# Main
# ============================================================
def _resolve_runtime_paths(args):
    script_dir = Path(__file__).resolve().parent

    if args.results_csv is not None:
        results_csv = Path(args.results_csv).resolve()
    elif RESULTS_CSV is not None:
        results_csv = Path(RESULTS_CSV).expanduser().resolve()
    else:
        results_csv = (script_dir / "pnas_equivalent_suite_results_S0345a.csv").resolve()

    if not results_csv.exists():
        raise FileNotFoundError(
            f"CSV de resultados nao encontrado: {results_csv}\n"
            "No VS Code, preencha RESULTS_CSV no topo do script ou deixe o CSV na mesma pasta do .py."
        )

    original_script = None
    if ORIGINAL_SCRIPT is not None:
        original_script = Path(ORIGINAL_SCRIPT).expanduser().resolve()
    else:
        guess_original = script_dir / "inversion_models.py"
        if guess_original.exists():
            original_script = guess_original

    if args.outdir is not None:
        outdir = Path(args.outdir).expanduser().resolve()
    elif OUTDIR is not None:
        outdir = Path(OUTDIR).expanduser().resolve()
    elif original_script is not None:
        fig_dir = _extract_fig_dir_from_original_script(original_script)
        outdir = fig_dir if fig_dir is not None else results_csv.parent
    else:
        outdir = results_csv.parent

    if args.models_dir is not None:
        models_dir = str(Path(args.models_dir).expanduser().resolve())
    elif MODELS_DIR is not None:
        models_dir = str(Path(MODELS_DIR).expanduser().resolve())
    else:
        models_dir = None

    return results_csv, outdir, models_dir


def main():
    parser = argparse.ArgumentParser(description="Replota figuras MTZ a partir do CSV de resultados, sem rerodar a inversao.")
    parser.add_argument("--results-csv", default=None, help="CSV de resultados final.")
    parser.add_argument("--outdir", default=None, help="Pasta de saida para as novas figuras. Se omitido, usa o FIG_DIR do inversion_models.py original quando existir.")
    parser.add_argument("--models-dir", default=None, help="Pasta onde estao os modelos .nd/.npz, caso source_path do CSV nao exista diretamente.")
    args = parser.parse_args()

    results_csv, outdir, models_dir = _resolve_runtime_paths(args)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_scatter_categorized(
        str(results_csv),
        str(outdir / "fig_mtz4_scatter_misfit_vs_depth_S0345a_custom.png"),
    )

    plot_profile_heatmap(
        str(results_csv),
        str(outdir / "fig3A_vp_profiles_pmisfit_S0345a_heatmap.png"),
        value_col="vp_kms",
        misfit_col="mean_p_misfit",
        xlabel="Vp (km/s)",
        cbar_label="Smoothed weighted density (lower P misfit = darker)",
        models_dir=models_dir,
    )

    plot_profile_heatmap(
        str(results_csv),
        str(outdir / "fig3B_vs_profiles_smisfit_S0345a_heatmap.png"),
        value_col="vs_kms",
        misfit_col="mean_s_misfit",
        xlabel="Vs (km/s)",
        cbar_label="Smoothed weighted density (lower S misfit = darker)",
        models_dir=models_dir,
    )

    plot_best_model_paperlike(
        str(results_csv),
        str(outdir / "vp_vs_paperlike_mtz4_bestmodel_S0345a_700_1800km.png"),
        models_dir=models_dir,
    )

    print(f"Figuras salvas em: {outdir.resolve()}")

if __name__ == "__main__":
    main()
