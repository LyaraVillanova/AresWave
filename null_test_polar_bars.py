#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import io
import os
import glob
import zipfile
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
})

COLOR_LOW = "#029743"   # green
COLOR_MID = "#f8e700"   # yellow
COLOR_HIGH = '#d73027'  # red
COLOR_BEST_EDGE = 'black'

NULL_RESULTS_DIR = 'null_results'
NULL_RESULTS_ZIP = None
OUT_PNG = 'null_results/Figure_3_polar_bar_summary.png'
OUT_PDF = None

EVENT_ORDER = [
    'S0185a', 'S0234c', 'S0167b', 'S1102a', 'S1153a', 'S1415a',
    'S0133a', 'S0254b', 'S0395a', 'S0345a', 'S0421b', 'S0226b',
    'S0167a', 'S0152a', 'S0976a',
]
MODELS = ['TAYAK', 'X']

BEST_SOLUTIONS: Dict[str, Dict[str, Dict[str, float]]] = {
    'S0185a': {'TAYAK': {'strike_deg': 185.953056641419, 'cost': 0.0824626130746303},
               'X':     {'strike_deg': 305.893166685142, 'cost': 0.0650162333410065}},
    'S0234c': {'TAYAK': {'strike_deg': 179.333942284437, 'cost': 0.0601909653680253},
               'X':     {'strike_deg': 40.7302600036638,  'cost': 0.0545832762373182}},
    'S0167b': {'TAYAK': {'strike_deg': 119.90764279225,  'cost': 0.0623341030263063},
               'X':     {'strike_deg': 231.2073213,      'cost': 0.048336382}},
    'S1102a': {'TAYAK': {'strike_deg': 140.120878966849, 'cost': 0.026472468690851},
               'X':     {'strike_deg': 218.194419812979, 'cost': 0.026378802188692}},
    'S1153a': {'TAYAK': {'strike_deg': 243.959975587441, 'cost': 0.186525201911062},
               'X':     {'strike_deg': 48.7155999466433, 'cost': 0.189405451929929}},
    'S1415a': {'TAYAK': {'strike_deg': 252.674762423985, 'cost': 0.156062417518557},
               'X':     {'strike_deg': 256.429298029711, 'cost': 0.14691652741483}},
    'S0133a': {'TAYAK': {'strike_deg': 301.75,           'cost': 0.0407},
               'X':     {'strike_deg': 247.606378592696, 'cost': 0.0434459238743402}},
    'S0254b': {'TAYAK': {'strike_deg': 198.27446976,     'cost': 0.101158166983786},
               'X':     {'strike_deg': 290.46939977,     'cost': 0.101165}},
    'S0395a': {'TAYAK': {'strike_deg': 51.31234728,      'cost': 0.0336},
               'X':     {'strike_deg': 99.87598525,      'cost': 0.0300404407698696}},
    'S0345a': {'TAYAK': {'strike_deg': 179.94249393,     'cost': 0.0672974629107085},
               'X':     {'strike_deg': 48.05022326,      'cost': 0.0669}},
    'S0421b': {'TAYAK': {'strike_deg': 165.56353496,     'cost': 0.0704},
               'X':     {'strike_deg': 264.45311963,     'cost': 0.0738}},
    'S0226b': {'TAYAK': {'strike_deg': 125.044593431172, 'cost': 0.120938638217726},
               'X':     {'strike_deg': 300.77483763,     'cost': 0.0965}},
    'S0167a': {'TAYAK': {'strike_deg': 80.58265868,      'cost': 0.0938},
               'X':     {'strike_deg': 237.15757689,     'cost': 0.0935}},
    'S0152a': {'TAYAK': {'strike_deg': 299.01838898959,  'cost': 0.151439},
               'X':     {'strike_deg': 243.05519429,     'cost': 0.1376784707508}},
    'S0976a': {'TAYAK': {'strike_deg': 234.999014181556, 'cost': 0.146384626376218},
               'X':     {'strike_deg': 272.360733224122, 'cost': 0.14647802118839}},
}


def _iter_npz_payloads(null_results_dir=None, null_results_zip=None):
    if null_results_zip:
        with zipfile.ZipFile(null_results_zip, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('_null.npz'):
                    yield name, io.BytesIO(zf.read(name))
    else:
        if not null_results_dir:
            raise ValueError('Provide NULL_RESULTS_DIR or NULL_RESULTS_ZIP.')
        for fp in sorted(glob.glob(os.path.join(null_results_dir, '*_null.npz'))):
            yield os.path.basename(fp), fp


def _to_scalar(x):
    if isinstance(x, np.ndarray):
        try:
            x = x.item()
        except Exception:
            pass
    if isinstance(x, bytes):
        x = x.decode()
    return x


def _best_percentile(costs: np.ndarray, best_cost: float) -> Tuple[int, int, float, str]:
    r = int(np.sum(costs <= best_cost))
    n = int(costs.size)
    pct = 100.0 * r / n if n else np.nan
    if r == 0 and n > 0:
        pct_label = f'<{100.0/n:.1f}%'
    else:
        pct_label = f'{pct:.1f}%'
    return r, n, pct, pct_label


def _load_null_results(null_results_dir=None, null_results_zip=None):
    out = {}
    all_costs = []
    for _, source in _iter_npz_payloads(null_results_dir, null_results_zip):
        z = np.load(source, allow_pickle=True)

        event = str(_to_scalar(z['event'])).strip()
        model = str(_to_scalar(z['model'])).strip().upper()
        costs = np.asarray(z['costs'], dtype=float)

        if 'X' not in z.files:
            out[(event, model)] = {'missing_x': True}
            continue
        X = np.asarray(z['X'], dtype=float)
        if X.ndim != 2 or X.shape[1] < 2:
            out[(event, model)] = {'missing_x': True}
            continue

        strikes = np.mod(X[:, 1], 360.0)
        p05 = float(np.percentile(costs, 5))
        p95 = float(np.percentile(costs, 95))

        out[(event, model)] = {
            'strikes_deg': strikes,
            'costs': costs,
            'p05': p05,
            'p95': p95,
            'missing_x': False,
        }
        all_costs.append(costs[np.isfinite(costs)])

    global_rmax = float(np.nanmax(np.concatenate(all_costs))) if all_costs else 1.0
    return out, global_rmax


def _cost_to_color(cost: float, p05: float, p95: float) -> str:
    if cost <= p05:
        return COLOR_LOW
    if cost >= p95:
        return COLOR_HIGH
    return COLOR_MID


def _plot_panel(ax, ev, model, data, global_rmax):
    key = (ev, model)
    if key not in data:
        ax.text(0.5, 0.5, f'{ev}/{model}\nmissing', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    d = data[key]
    if d.get('missing_x', False):
        ax.text(0.5, 0.5, f'{ev}/{model}\nmissing X', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    strikes = np.asarray(d['strikes_deg'], dtype=float)
    costs = np.asarray(d['costs'], dtype=float)
    p05 = float(d['p05'])
    p95 = float(d['p95'])

    width = np.deg2rad(1.0)
    theta = np.deg2rad(strikes)

    # Draw each sample in sequence, no transparency.
    for th, cst in zip(theta, costs):
        color = _cost_to_color(float(cst), p05, p95)
        ax.bar([th], [cst], width=width, bottom=0.0,
               color=color, edgecolor=color, linewidth=0.0,
               align='center', zorder=3)

    best = BEST_SOLUTIONS.get(ev, {}).get(model)
    title = f'{ev} / {model}'
    if best is not None:
        best_theta = np.deg2rad(best['strike_deg'] % 360.0)
        best_cost = float(best['cost'])
        ax.bar([best_theta], [best_cost], width=width * 2.0, bottom=0.0,
               facecolor='none', edgecolor=COLOR_BEST_EDGE, linewidth=1.4,
               align='center', zorder=6)
        _, _, _, pct_label = _best_percentile(costs, best_cost)
        title = f'{ev} / {model} | best-cost pct {pct_label}'

    ax.set_title(title, pad=10)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, global_rmax * 1.03)

    # Remove radial numbers completely.
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.35)


def make_figure(out_png, null_results_dir=None, null_results_zip=None, out_pdf=None):
    data, global_rmax = _load_null_results(null_results_dir, null_results_zip)

    events = list(EVENT_ORDER)
    n_rows = len(events)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.0, 2.8 * n_rows),
                             subplot_kw={'projection': 'polar'}, squeeze=False)

    for r, ev in enumerate(events):
        for c, model in enumerate(MODELS):
            _plot_panel(axes[r, c], ev, model, data, global_rmax)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLOR_HIGH, edgecolor=COLOR_HIGH, label='cost ≥ p95'),
        Patch(facecolor=COLOR_MID, edgecolor=COLOR_MID, label='p05 < cost < p95'),
        Patch(facecolor=COLOR_LOW, edgecolor=COLOR_LOW, label='cost ≤ p05'),
        Patch(facecolor='none', edgecolor='black', linewidth=1.4, label='PSO best solution'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.995))

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.975])
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    if out_pdf:
        os.makedirs(os.path.dirname(out_pdf) or '.', exist_ok=True)
        fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description='Polar bar summary of null-test random draws.')
    p.add_argument('--null-dir', default=NULL_RESULTS_DIR)
    p.add_argument('--null-zip', default=NULL_RESULTS_ZIP)
    p.add_argument('--out-png', default=OUT_PNG)
    p.add_argument('--out-pdf', default=OUT_PDF)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    make_figure(out_png=args.out_png,
                null_results_dir=args.null_dir,
                null_results_zip=args.null_zip,
                out_pdf=args.out_pdf)
