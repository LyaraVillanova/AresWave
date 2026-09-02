#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import io
import glob
import zipfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -------------------------------------------------------------------------
# Local imports
# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DSMPY_DIR = SCRIPT_DIR / "dsmpy"
if str(DSMPY_DIR) not in sys.path:
    sys.path.insert(0, str(DSMPY_DIR))

import null_test_fast as ntf


# -------------------------------------------------------------------------
# Matplotlib style
# -------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
})


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
DEPTH_RESULTS_DIR = str(SCRIPT_DIR / "depth_results")
DEPTH_RESULTS_ZIP = None

OUT_PNG = str(
    SCRIPT_DIR
    / "depth_results"
    / "Figure_fixedMT_depthscan_explosion.png"
)


# -------------------------------------------------------------------------
# Plot setup
# -------------------------------------------------------------------------
EVENT_ORDER = [
    "S1000a",
    "S1094b",
]

SCAN_DOT_COLOR = "0.75"
SCAN_DOT_SIZE = 10

SCAN_MIN_COLOR = "k"
SCAN_MIN_SIZE = 28

PSO_COLOR = "m"
PSO_LINE_WIDTH = 1.6
PSO_DOT_SIZE = 28


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def _iter_npz_payloads(
    depth_results_dir: str | None = None,
    depth_results_zip: str | None = None,
):
    """
    Iterate over depth-scan .npz files.

    Accepts both:
        *_fixedMT_depthscan.npz
        *_fixedMT_depthscan_explo.npz

    This is necessary because the scan script currently saves the explosion
    test as *_fixedMT_depthscan_explo.npz.
    """
    if depth_results_zip:
        with zipfile.ZipFile(depth_results_zip, "r") as zf:
            for name in zf.namelist():
                base = os.path.basename(name)

                if (
                    base.endswith("_fixedMT_depthscan.npz")
                    or base.endswith("_fixedMT_depthscan_explo.npz")
                ):
                    yield name, io.BytesIO(zf.read(name))

    else:
        if not depth_results_dir:
            raise ValueError("Provide DEPTH_RESULTS_DIR or DEPTH_RESULTS_ZIP.")

        patterns = [
            os.path.join(depth_results_dir, "*_fixedMT_depthscan.npz"),
            os.path.join(depth_results_dir, "*_fixedMT_depthscan_explo.npz"),
        ]

        seen = set()

        for pat in patterns:
            for fp in sorted(glob.glob(pat)):
                if fp in seen:
                    continue

                seen.add(fp)
                yield os.path.basename(fp), fp


def _as_scalar(x):
    """
    Convert numpy scalar/array or bytes into a normal Python scalar/string.
    """
    if isinstance(x, np.ndarray):
        try:
            x = x.item()
        except Exception:
            pass

    if isinstance(x, bytes):
        x = x.decode()

    return x


def _declared_best_lookup():
    """
    Load PSO reference depth/cost from null_test_fast.BEST_SOLUTIONS.

    Some event/model pairs may be absent from BEST_SOLUTIONS. In that case,
    the plot will use best_depth/best_cost stored inside each .npz file.
    """
    out = {}

    for ev, mdict in ntf.BEST_SOLUTIONS.items():
        for model, vals in mdict.items():
            if not vals:
                continue

            out[(str(ev).strip(), str(model).strip().upper())] = {
                "depth_km": float(vals["depth_km"]),
                "cost": float(vals["cost"]),
            }

    return out


def _load_depth_results(
    depth_results_dir: str | None = None,
    depth_results_zip: str | None = None,
):
    """
    Load all depth-scan results into a dictionary keyed by:

        (event, model)

    The function also stores best_depth and best_cost from the .npz file.
    This avoids falsely marking panels as "missing" when BEST_SOLUTIONS does
    not contain the event/model pair.
    """
    out = {}

    for _, source in _iter_npz_payloads(depth_results_dir, depth_results_zip):
        z = np.load(source, allow_pickle=True)

        event = str(_as_scalar(z["event"])).strip()
        model = str(_as_scalar(z["model"])).strip().upper()

        depths = np.asarray(z["depths"], dtype=float)
        costs = np.asarray(z["costs"], dtype=float)

        finite = np.isfinite(costs)

        if np.any(finite):
            imin = int(np.nanargmin(costs))
            scan_best_depth = float(depths[imin])
            scan_best_cost = float(costs[imin])
        else:
            imin = None
            scan_best_depth = np.nan
            scan_best_cost = np.nan

        best_depth_npz = z["best_depth"] if "best_depth" in z.files else np.nan
        best_cost_npz = z["best_cost"] if "best_cost" in z.files else np.nan

        try:
            best_depth_npz = float(_as_scalar(best_depth_npz))
        except Exception:
            best_depth_npz = np.nan

        try:
            best_cost_npz = float(_as_scalar(best_cost_npz))
        except Exception:
            best_cost_npz = np.nan

        out[(event, model)] = {
            "depths": depths,
            "costs": costs,
            "imin": imin,
            "scan_best_depth": scan_best_depth,
            "scan_best_cost": scan_best_cost,
            "best_depth_npz": best_depth_npz,
            "best_cost_npz": best_cost_npz,
        }

    print("[debug] loaded depth result keys:", sorted(out.keys()))

    return out


# -------------------------------------------------------------------------
# Figure
# -------------------------------------------------------------------------
def make_figure(best_lookup, depth_stats, out_png, dpi=250):
    events = list(EVENT_ORDER)

    mid = int(np.ceil(len(events) / 2))
    left = events[:mid]
    right = events[mid:]

    n_rows = max(len(left), len(right))
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18.5, 2.15 * n_rows),
        squeeze=False,
    )

    def _plot_panel(ax, ev, model):
        key = (ev, model)

        if key not in depth_stats:
            ax.text(
                0.5,
                0.5,
                f"{ev}/{model}\nmissing npz",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return

        d = depth_stats[key]

        # Prefer BEST_SOLUTIONS when available.
        # Otherwise, use the reference depth/cost saved in the .npz file.
        if key in best_lookup:
            b = best_lookup[key]
        else:
            b = {
                "depth_km": float(d.get("best_depth_npz", np.nan)),
                "cost": float(d.get("best_cost_npz", np.nan)),
            }

        x = np.asarray(d["depths"], dtype=float)
        y = np.asarray(d["costs"], dtype=float)

        finite = np.isfinite(y)

        if not np.any(finite):
            ax.text(
                0.5,
                0.5,
                f"{ev}/{model}\nall costs NaN",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return

        xf = x[finite]
        yf = y[finite]

        # Grey dots: all scan costs
        ax.scatter(
            xf,
            yf,
            s=SCAN_DOT_SIZE,
            c=SCAN_DOT_COLOR,
            edgecolors="none",
        )

        # Black dot: scan minimum
        imin = d["imin"]

        if imin is not None and np.isfinite(y[imin]):
            ax.scatter(
                [x[imin]],
                [y[imin]],
                s=SCAN_MIN_SIZE,
                c=SCAN_MIN_COLOR,
                edgecolors="none",
            )

        # Magenta vertical: reference/PSO depth, if available
        zref = b.get("depth_km", np.nan)

        if np.isfinite(zref):
            ax.axvline(
                zref,
                linestyle="--",
                linewidth=PSO_LINE_WIDTH,
                color=PSO_COLOR,
            )

        # Magenta horizontal/dot: reference/PSO cost, if finite
        bc = d.get("best_cost_npz", np.nan)

        if not np.isfinite(bc):
            bc = b.get("cost", np.nan)

        if np.isfinite(bc):
            ax.axhline(
                bc,
                linestyle="--",
                linewidth=PSO_LINE_WIDTH,
                color=PSO_COLOR,
            )

            if np.isfinite(zref):
                ax.scatter(
                    [zref],
                    [bc],
                    s=PSO_DOT_SIZE,
                    c=PSO_COLOR,
                    edgecolors="none",
                )

        ax.set_title(f"{ev} / {model}", pad=3)

        if np.isfinite(zref):
            dz = d["scan_best_depth"] - zref

            txt = (
                f"ref depth={zref:.1f}  "
                f"best test={d['scan_best_depth']:.1f}  "
                f"Δ={dz:+.1f}\n"
                f"min={d['scan_best_cost']:.3f}"
            )

        else:
            txt = (
                f"best test={d['scan_best_depth']:.1f}\n"
                f"min={d['scan_best_cost']:.3f}"
            )

        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

        xmin = np.nanmin(xf)
        xmax = np.nanmax(xf)

        if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
            pad = 0.03 * (xmax - xmin)
            ax.set_xlim(xmin - pad, xmax + pad)

    # ---------------------------------------------------------------------
    # Fill panels
    # ---------------------------------------------------------------------
    for r in range(n_rows):
        if r < len(left):
            ev = left[r]
            _plot_panel(axes[r, 0], ev, "TAYAK")
            _plot_panel(axes[r, 1], ev, "X")
        else:
            axes[r, 0].axis("off")
            axes[r, 1].axis("off")

        if r < len(right):
            ev = right[r]
            _plot_panel(axes[r, 2], ev, "TAYAK")
            _plot_panel(axes[r, 3], ev, "X")
        else:
            axes[r, 2].axis("off")
            axes[r, 3].axis("off")

        for c in range(n_cols):
            if r == n_rows - 1 and axes[r, c].has_data():
                axes[r, c].set_xlabel("Depth (km)")
            else:
                axes[r, c].set_xlabel("")

            if c in (0, 2) and axes[r, c].has_data():
                axes[r, c].set_ylabel("Cost")
            else:
                axes[r, c].set_ylabel("")

    # ---------------------------------------------------------------------
    # Group labels
    # ---------------------------------------------------------------------
    axes[0, 0].annotate(
        "First half of events",
        xy=(0, 1.20),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=11,
        weight="bold",
    )

    axes[0, 2].annotate(
        "Second half of events",
        xy=(0, 1.20),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=11,
        weight="bold",
    )

    # ---------------------------------------------------------------------
    # Legend
    # ---------------------------------------------------------------------
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SCAN_DOT_COLOR,
            markersize=6,
            label="scan costs",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SCAN_MIN_COLOR,
            markersize=7,
            label="scan minimum",
        ),
        Line2D(
            [0],
            [0],
            color=PSO_COLOR,
            linestyle="--",
            linewidth=PSO_LINE_WIDTH,
            label="reference depth / reference cost",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PSO_COLOR,
            markersize=7,
            label="(z_ref, reference cost)",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.tight_layout(rect=(0.02, 0.01, 0.98, 0.97))

    out_png = str(out_png)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=dpi)
    print("[ok] saved:", out_png)

    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    best_lookup = _declared_best_lookup()

    depth_stats = _load_depth_results(
        DEPTH_RESULTS_DIR,
        DEPTH_RESULTS_ZIP,
    )

    make_figure(
        best_lookup,
        depth_stats,
        OUT_PNG,
    )


if __name__ == "__main__":
    main()