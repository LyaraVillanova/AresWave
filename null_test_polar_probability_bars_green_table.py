#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polar probability bars (green/gray/black only) + per-angle probability table.
- keeps only three visual elements:
    * light-gray occupancy background = tested strike sectors
    * green probability bars = empirical P(success | strike bin)
    * black outlined bar = bin containing the PSO best strike
- uses success defined by default as: cost <= p05
- exports a CSV table with one row per angular bin, including probabilities,
  counts, colors, and classifications.
"""
from __future__ import annotations
import argparse
import csv
import io
import os
import glob
import zipfile
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
})

NULL_RESULTS_DIR = 'null_results'
NULL_RESULTS_ZIP = None
OUT_PNG = 'null_results/Figure_3_polar_probability_bars_green.png'
OUT_PDF = None
OUT_TABLE = 'null_results/Figure_3_polar_probability_bins_p05.csv'

EVENT_ORDER = [
    'S0185a', 'S0234c', 'S0167b', 'S1102a', 'S1153a', 'S1415a',
    'S0133a', 'S0254b', 'S0395a', 'S0345a', 'S0421b', 'S0226b',
    'S0167a', 'S0152a', 'S0976a',
]
MODELS = ['TAYAK', 'X']

COLOR_SUCCESS = "#06ed6a"   # green
COLOR_OCC = '0.88'          # light gray
COLOR_BEST_EDGE = 'black'

SHOW_OCCUPANCY_BACKGROUND = True
SHOW_EMPTY_BINS = False
NBINS = 72
SUCCESS_MODE = 'p05'

# Visual scaling inside the polar panel
RING_BASE = 0.18
RING_SPAN = 0.72
OCC_SPAN = 0.16
BEST_MIN_VISIBLE = 0.10     # keep black outline visible even if P(best bin)=0
BAR_WIDTH_SCALE = 0.72

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
    max_occ_fraction = 0.0

    for _, source in _iter_npz_payloads(null_results_dir, null_results_zip):
        z = np.load(source, allow_pickle=True)
        event = str(_to_scalar(z['event'])).strip()
        model = str(_to_scalar(z['model'])).strip().upper()

        if 'costs' not in z.files or 'X' not in z.files:
            out[(event, model)] = {'missing_x': True}
            continue

        costs = np.asarray(z['costs'], dtype=float)
        X = np.asarray(z['X'], dtype=float)
        if X.ndim != 2 or X.shape[1] < 2:
            out[(event, model)] = {'missing_x': True}
            continue

        # X columns are [depth, strike, dip, rake]
        strikes = np.mod(X[:, 1], 360.0)
        p05 = float(np.percentile(costs, 5))
        p95 = float(np.percentile(costs, 95))

        edges_deg = np.linspace(0.0, 360.0, NBINS + 1)
        centers_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
        bin_index = np.digitize(strikes, edges_deg, right=False) - 1
        bin_index = np.clip(bin_index, 0, NBINS - 1)

        best = BEST_SOLUTIONS.get(event, {}).get(model)
        if best is None:
            out[(event, model)] = {'missing_x': True}
            continue
        best_cost = float(best['cost'])
        best_strike = float(best['strike_deg']) % 360.0

        if SUCCESS_MODE == 'best':
            success = costs <= best_cost
            success_threshold = best_cost
        elif SUCCESS_MODE == 'p05':
            success = costs <= p05
            success_threshold = p05
        else:
            raise ValueError("SUCCESS_MODE must be 'best' or 'p05'.")

        total_per_bin = np.bincount(bin_index, minlength=NBINS).astype(int)
        success_per_bin = np.bincount(bin_index, weights=success.astype(float), minlength=NBINS).astype(int)
        prob_per_bin = np.divide(success_per_bin, total_per_bin,
                                 out=np.zeros_like(success_per_bin, dtype=float),
                                 where=total_per_bin > 0)

        occ_fraction = total_per_bin / max(total_per_bin.sum(), 1.0)
        if np.any(np.isfinite(occ_fraction)):
            max_occ_fraction = max(max_occ_fraction, float(np.nanmax(occ_fraction)))

        best_bin = int(np.digitize([best_strike], edges_deg, right=False)[0] - 1)
        best_bin = int(np.clip(best_bin, 0, NBINS - 1))

        out[(event, model)] = {
            'missing_x': False,
            'costs': costs,
            'p05': p05,
            'p95': p95,
            'success_threshold': success_threshold,
            'edges_deg': edges_deg,
            'centers_deg': centers_deg,
            'total_per_bin': total_per_bin,
            'success_per_bin': success_per_bin,
            'prob_per_bin': prob_per_bin,
            'occ_fraction': occ_fraction,
            'best_bin': best_bin,
            'best_cost': best_cost,
            'best_strike': best_strike,
        }

    return out, max(max_occ_fraction, 1e-12)


def _plot_panel(ax, ev, model, data, global_occ_max):
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

    centers_deg = np.asarray(d['centers_deg'], dtype=float)
    edges_deg = np.asarray(d['edges_deg'], dtype=float)
    probs = np.asarray(d['prob_per_bin'], dtype=float)
    occ_fraction = np.asarray(d['occ_fraction'], dtype=float)
    total_per_bin = np.asarray(d['total_per_bin'], dtype=float)
    best_bin = int(d['best_bin'])
    best_cost = float(d['best_cost'])
    costs = np.asarray(d['costs'], dtype=float)

    theta = np.deg2rad(centers_deg)
    width = np.deg2rad(edges_deg[1] - edges_deg[0]) * BAR_WIDTH_SCALE

    # Gray occupancy background
    if SHOW_OCCUPANCY_BACKGROUND:
        occ_scaled = OCC_SPAN * np.divide(occ_fraction, global_occ_max,
                                          out=np.zeros_like(occ_fraction),
                                          where=global_occ_max > 0)
        for th, occ_h in zip(theta, occ_scaled):
            if occ_h <= 0 and not SHOW_EMPTY_BINS:
                continue
            ax.bar([th], [occ_h], width=width, bottom=RING_BASE,
                   color=COLOR_OCC, edgecolor=COLOR_OCC, linewidth=0.0,
                   align='center', zorder=1)

    # Green probability bars only
    for th, pk, nk in zip(theta, probs, total_per_bin):
        if nk <= 0 and not SHOW_EMPTY_BINS:
            continue
        if pk <= 0.0:
            continue
        height = RING_SPAN * float(pk)
        ax.bar([th], [height], width=width, bottom=RING_BASE,
               color=COLOR_SUCCESS, edgecolor=COLOR_SUCCESS, linewidth=0.0,
               align='center', zorder=3)

    # Black outlined best-solution bin with same thin-bar shape
    best_theta = theta[best_bin]
    best_height = max(RING_SPAN * float(probs[best_bin]), BEST_MIN_VISIBLE)
    ax.bar([best_theta], [best_height], width=width, bottom=RING_BASE,
           facecolor='none', edgecolor=COLOR_BEST_EDGE, linewidth=1.2,
           align='center', zorder=6)

    _, _, _, pct_label = _best_percentile(costs, best_cost)
    if SUCCESS_MODE == 'best':
        title = f'{ev} / {model} | best-cost pct {pct_label}'
    else:
        title = f'{ev} / {model} | success = cost ≤ p05'
    ax.set_title(title, pad=10)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, RING_BASE + max(OCC_SPAN, RING_SPAN) + 0.05)

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.35)


def _build_table_rows(data):
    rows = []
    for ev in EVENT_ORDER:
        for model in MODELS:
            key = (ev, model)
            if key not in data:
                continue
            d = data[key]
            if d.get('missing_x', False):
                continue

            edges_deg = np.asarray(d['edges_deg'], dtype=float)
            centers_deg = np.asarray(d['centers_deg'], dtype=float)
            totals = np.asarray(d['total_per_bin'], dtype=int)
            successes = np.asarray(d['success_per_bin'], dtype=int)
            probs = np.asarray(d['prob_per_bin'], dtype=float)
            occ_fraction = np.asarray(d['occ_fraction'], dtype=float)
            best_bin = int(d['best_bin'])
            best_strike = float(d['best_strike'])
            best_cost = float(d['best_cost'])
            p05 = float(d['p05'])
            p95 = float(d['p95'])
            success_threshold = float(d['success_threshold'])

            for i in range(len(centers_deg)):
                n_total = int(totals[i])
                n_success = int(successes[i])
                p_success = float(probs[i])
                has_gray = bool(n_total > 0)
                has_green = bool(p_success > 0.0)
                is_best = bool(i == best_bin)

                if is_best and has_green:
                    classification = 'best_bin_with_success_probability'
                elif is_best and not has_green:
                    classification = 'best_bin_no_success_probability'
                elif has_green:
                    classification = 'success_probability_bin'
                elif has_gray:
                    classification = 'tested_bin_no_success'
                else:
                    classification = 'empty_bin'

                if has_green:
                    color_fill = 'green'
                elif has_gray:
                    color_fill = 'gray'
                else:
                    color_fill = 'none'

                rows.append({
                    'event': ev,
                    'model': model,
                    'success_mode': SUCCESS_MODE,
                    'p05_cost_threshold': p05,
                    'p95_cost_threshold': p95,
                    'success_threshold_used': success_threshold,
                    'best_cost': best_cost,
                    'best_strike_deg': best_strike,
                    'bin_index': i,
                    'theta_start_deg': float(edges_deg[i]),
                    'theta_end_deg': float(edges_deg[i + 1]),
                    'theta_center_deg': float(centers_deg[i]),
                    'n_total_in_bin': n_total,
                    'n_success_in_bin': n_success,
                    'p_success_given_bin': p_success,
                    'occupancy_fraction': float(occ_fraction[i]),
                    'has_gray_background': has_gray,
                    'has_green_probability_bar': has_green,
                    'has_black_best_outline': is_best,
                    'color_fill': color_fill,
                    'classification': classification,
                })
    return rows


def _write_table_csv(rows, out_table):
    os.makedirs(os.path.dirname(out_table) or '.', exist_ok=True)
    fieldnames = [
        'event', 'model', 'success_mode',
        'p05_cost_threshold', 'p95_cost_threshold', 'success_threshold_used',
        'best_cost', 'best_strike_deg',
        'bin_index', 'theta_start_deg', 'theta_end_deg', 'theta_center_deg',
        'n_total_in_bin', 'n_success_in_bin', 'p_success_given_bin', 'occupancy_fraction',
        'has_gray_background', 'has_green_probability_bar', 'has_black_best_outline',
        'color_fill', 'classification',
    ]
    with open(out_table, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(out_png, out_table, null_results_dir=None, null_results_zip=None, out_pdf=None):
    data, global_occ_max = _load_null_results(null_results_dir, null_results_zip)
    rows = _build_table_rows(data)
    _write_table_csv(rows, out_table)

    events = list(EVENT_ORDER)
    n_rows = len(events)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.0, 2.8 * n_rows),
                             subplot_kw={'projection': 'polar'}, squeeze=False)

    for r, ev in enumerate(events):
        for c, model in enumerate(MODELS):
            _plot_panel(axes[r, c], ev, model, data, global_occ_max)

    legend_handles = []
    if SHOW_OCCUPANCY_BACKGROUND:
        legend_handles.append(Patch(facecolor=COLOR_OCC, edgecolor=COLOR_OCC, label='tested strike sectors'))
    legend_handles.extend([
        Patch(facecolor=COLOR_SUCCESS, edgecolor=COLOR_SUCCESS, label='P(success | strike bin) > 0'),
        Patch(facecolor='none', edgecolor='black', linewidth=1.2, label='bin containing PSO best strike'),
    ])
    fig.legend(handles=legend_handles, loc='upper center', ncol=min(3, len(legend_handles)),
               frameon=False, bbox_to_anchor=(0.5, 0.995))

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.975])
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    if out_pdf:
        os.makedirs(os.path.dirname(out_pdf) or '.', exist_ok=True)
        fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description='Polar probability bars from null-test random draws (green/gray/black only) + CSV table.')
    p.add_argument('--null-dir', default=NULL_RESULTS_DIR)
    p.add_argument('--null-zip', default=NULL_RESULTS_ZIP)
    p.add_argument('--out-png', default=OUT_PNG)
    p.add_argument('--out-pdf', default=OUT_PDF)
    p.add_argument('--out-table', default=OUT_TABLE)
    p.add_argument('--nbins', type=int, default=NBINS)
    p.add_argument('--success-mode', choices=['best', 'p05'], default=SUCCESS_MODE)
    p.add_argument('--hide-occupancy', action='store_true', help='Do not draw the light-gray occupancy bars.')
    p.add_argument('--ring-base', type=float, default=RING_BASE)
    p.add_argument('--ring-span', type=float, default=RING_SPAN)
    p.add_argument('--occ-span', type=float, default=OCC_SPAN)
    p.add_argument('--bar-width-scale', type=float, default=BAR_WIDTH_SCALE)
    p.add_argument('--best-min-visible', type=float, default=BEST_MIN_VISIBLE)
    return p.parse_args()


def main():
    global NBINS, SUCCESS_MODE, SHOW_OCCUPANCY_BACKGROUND, RING_BASE, RING_SPAN, OCC_SPAN, BAR_WIDTH_SCALE, BEST_MIN_VISIBLE
    args = parse_args()
    NBINS = int(args.nbins)
    SUCCESS_MODE = str(args.success_mode)
    SHOW_OCCUPANCY_BACKGROUND = not bool(args.hide_occupancy)
    RING_BASE = float(args.ring_base)
    RING_SPAN = float(args.ring_span)
    OCC_SPAN = float(args.occ_span)
    BAR_WIDTH_SCALE = float(args.bar_width_scale)
    BEST_MIN_VISIBLE = float(args.best_min_visible)
    make_figure(args.out_png, args.out_table, null_results_dir=args.null_dir, null_results_zip=args.null_zip, out_pdf=args.out_pdf)


if __name__ == '__main__':
    main()
