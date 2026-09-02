import os
import io
import glob
import zipfile
import argparse

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Plot style (bigger fonts, light gray bars, magenta PSO markers)
# =========================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})

HIST_COLOR = '0.85'  # light gray
PSO_COLOR = 'magenta'
MED_COLOR = '0.35'
P05_COLOR = '0.55'

# =========================
# User-configurable paths
# =========================
NULL_RESULTS_DIR = "null_results"   # folder with *_null.npz files (set to None if using zip)
NULL_RESULTS_ZIP = None             # e.g., "all_null_results.zip"
OUT_PNG = "null_results/Figure_S31_null_summary_hist.png"

# Preferred row order (matches Table S4)
EVENT_ORDER = [
    "S0185a", "S0234c", "S0167b", "S1102a", "S1153a", "S1415a",
    "S0133a", "S0254b", "S0395a", "S0345a", "S0421b", "S0226b",
    "S0167a", "S0152a", "S0976a",
]
MODELS = ["TAYAK", "X"]

# ==========================================================
# Declared PSO best-fit results (same style idea as null_test)
# Only "cost" is needed for Figure S31, but nested dict keeps it
# easy to mirror your null_test / manuscript tables.
# ==========================================================
BEST_SOLUTIONS = {
    "S0185a": {
        "TAYAK": {"cost": 0.0825},
        "X": {"cost": 0.0650},
    },
    "S0234c": {
        "TAYAK": {"cost": 0.0602},
        "X": {"cost": 0.0546},
    },
    "S0167b": {
        "TAYAK": {"cost": 0.0623},
        "X": {"cost": 0.0483},
    },
    "S1102a": {
        "TAYAK": {"cost": 0.0265},
        "X": {"cost": 0.0264},
    },
    "S1153a": {
        "TAYAK": {"cost": 0.1865},
        "X": {"cost": 0.1894},
    },
    "S1415a": {
        "TAYAK": {"cost": 0.1561},
        "X": {"cost": 0.1469},
    },
    "S0133a": {
        "TAYAK": {"cost": 0.0407},
        "X": {"cost": 0.0434},
    },
    "S0254b": {
        "TAYAK": {"cost": 0.1012},
        "X": {"cost": 0.1012},
    },
    "S0395a": {
        "TAYAK": {"cost": 0.0336},
        "X": {"cost": 0.0300},
    },
    "S0345a": {
        "TAYAK": {"cost": 0.0673},
        "X": {"cost": 0.0669},
    },
    "S0421b": {
        "TAYAK": {"cost": 0.0704},
        "X": {"cost": 0.0738},
    },
    "S0226b": {
        "TAYAK": {"cost": 0.1209},
        "X": {"cost": 0.0965},
    },
    "S0167a": {
        "TAYAK": {"cost": 0.0938},
        "X": {"cost": 0.0935},
    },
    "S0152a": {
        "TAYAK": {"cost": 0.1514},
        "X": {"cost": 0.1377},
    },
    "S0976a": {
        "TAYAK": {"cost": 0.1464},
        "X": {"cost": 0.1464},
    },
}


def _load_best_costs_from_declared():
    out = {}
    for ev, mdict in BEST_SOLUTIONS.items():
        for model, vals in mdict.items():
            if vals is None:
                continue
            if "cost" not in vals:
                continue
            out[(str(ev).strip(), str(model).strip().upper())] = float(vals["cost"])
    return out


def _iter_npz_payloads(null_results_dir=None, null_results_zip=None):
    if null_results_zip:
        with zipfile.ZipFile(null_results_zip, "r") as zf:
            for name in zf.namelist():
                if not name.endswith("_null.npz"):
                    continue
                yield name, io.BytesIO(zf.read(name))
    else:
        if not null_results_dir:
            raise ValueError("Provide NULL_RESULTS_DIR or NULL_RESULTS_ZIP.")
        for fp in sorted(glob.glob(os.path.join(null_results_dir, "*_null.npz"))):
            yield os.path.basename(fp), fp


def _load_null_results(null_results_dir=None, null_results_zip=None):
    out = {}
    for _, source in _iter_npz_payloads(null_results_dir, null_results_zip):
        z = np.load(source, allow_pickle=True)

        event = z["event"]
        model = z["model"]
        if isinstance(event, np.ndarray):
            event = event.item()
        if isinstance(model, np.ndarray):
            model = model.item()
        if isinstance(event, bytes):
            event = event.decode()
        if isinstance(model, bytes):
            model = model.decode()

        event = str(event).strip()
        model = str(model).strip().upper()
        costs = np.asarray(z["costs"], dtype=float)

        out[(event, model)] = {
            "costs": costs,
            "n": int(costs.size),
            "median": float(np.median(costs)),
            "p05": float(np.percentile(costs, 5)),
            "p95": float(np.percentile(costs, 95)),
        }
    return out


def _best_percentile(costs, best_cost):
    r = int(np.sum(costs <= best_cost))
    n = int(costs.size)
    pct = 100.0 * r / n if n else np.nan
    if r == 0 and n > 0:
        pct_label = f"<{100.0/n:.1f}%"
    else:
        pct_label = f"{pct:.1f}%"
    p_emp = (r + 1) / (n + 1) if n else np.nan
    return r, n, pct, pct_label, p_emp



def make_figure_s31(best_costs, null_stats, out_png, out_pdf=None):
    """Figure S31: 4 columns.
    Layout: left block (cols 0-1) = first half of events (TAYAK, X),
            right block (cols 2-3) = second half of events (TAYAK, X).
    """
    events = list(EVENT_ORDER)
    mid = int(np.ceil(len(events) / 2))
    left_events = events[:mid]
    right_events = events[mid:]

    n_rows = max(len(left_events), len(right_events))
    n_cols = 4

    # Wider figure for 4 columns; height per row tuned for readability
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18.5, 2.2 * n_rows), squeeze=False)

    def _plot_panel(ax, ev, model):
        key = (ev, model)
        if key not in null_stats or key not in best_costs:
            ax.text(0.5, 0.5, f"{ev}/{model}\nmissing", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return

        costs = null_stats[key]["costs"]
        best = best_costs[key]
        med = null_stats[key]["median"]
        p05 = null_stats[key]["p05"]
        _, _, _, pct_label, _ = _best_percentile(costs, best)

        ax.hist(costs, bins=30, color=HIST_COLOR, edgecolor=HIST_COLOR, alpha=1.0)

        # Markers: PSO best (magenta) + null summary lines
        ax.axvline(best, linestyle="--", linewidth=2.4, color=PSO_COLOR, zorder=5)
        ax.axvline(med, linestyle=":", linewidth=1.6, color=MED_COLOR, zorder=4)
        ax.axvline(p05, linestyle="-.", linewidth=1.6, color=P05_COLOR, zorder=4)

        ax.set_title(f"{ev} / {model}", pad=4)
        ax.text(
            0.98,
            0.95,
            f"best={best:.4f}\nmed={med:.4f}\np05={p05:.4f}\n{pct_label}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
        )

    # Fill panels
    for r in range(n_rows):
        # left event block
        if r < len(left_events):
            evL = left_events[r]
            _plot_panel(axes[r, 0], evL, "TAYAK")
            _plot_panel(axes[r, 1], evL, "X")
        else:
            for c in (0, 1):
                axes[r, c].axis("off")

        # right event block
        if r < len(right_events):
            evR = right_events[r]
            _plot_panel(axes[r, 2], evR, "TAYAK")
            _plot_panel(axes[r, 3], evR, "X")
        else:
            for c in (2, 3):
                axes[r, c].axis("off")

        # Labels (only on bottom row for x; leftmost of each block for y)
        for c in range(n_cols):
            if r == n_rows - 1 and axes[r, c].has_data():
                axes[r, c].set_xlabel("Cost")
            else:
                axes[r, c].set_xlabel("")
            if c in (0, 2) and axes[r, c].has_data():
                axes[r, c].set_ylabel("Count")
            else:
                axes[r, c].set_ylabel("")

    # Column headers
    axes[0, 0].annotate("First half of events", xy=(0, 1.22), xycoords="axes fraction",
                        ha="left", va="bottom", fontsize=14, weight="bold")
    axes[0, 2].annotate("Second half of events", xy=(0, 1.22), xycoords="axes fraction",
                        ha="left", va="bottom", fontsize=14, weight="bold")

    fig.text(
        0.5,
        0.995,
        "Bars = null costs (light gray); dashed magenta = PSO best; dotted = null median; dash-dot = null p05",
        ha="center",
        va="top",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.02, 0.01, 0.98, 0.975))

    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Build Figure S31 from null-test .npz outputs using declared PSO best costs (BEST_SOLUTIONS)."
    )
    parser.add_argument("--null-dir", default=NULL_RESULTS_DIR, help="Directory with *_null.npz files")
    parser.add_argument("--null-zip", default=NULL_RESULTS_ZIP, help="Zip file with *_null.npz files")
    parser.add_argument("--out-png", default=OUT_PNG, help="Output PNG path")
    args = parser.parse_args()

    null_dir = args.null_dir
    null_zip = args.null_zip

    # Auto-detect a zip if no folder is provided/found
    if (not null_zip) and (not null_dir or not os.path.isdir(null_dir)):
        zips = sorted(glob.glob("*null*.zip"))
        if len(zips) == 1:
            null_zip = zips[0]
            print(f"[info] Auto-detected zip: {null_zip}")

    best = _load_best_costs_from_declared()
    nulls = _load_null_results(null_dir, null_zip)

    missing = []
    for ev in EVENT_ORDER:
        for m in MODELS:
            if (ev, m) not in best or (ev, m) not in nulls:
                missing.append((ev, m))
    if missing:
        print("[warn] Missing entries:", missing)

    make_figure_s31(best, nulls, args.out_png)
    print(f"[saved] {args.out_png}")


if __name__ == "__main__":
    main()