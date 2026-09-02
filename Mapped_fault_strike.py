from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROTATIONS_PPT_DEG = {
    "S0133a": 236,
    "S0152a": 228,
    "S0167a": 249,
    "S0167b": 238,
    "S0185a": 286,
    "S0226b": 308,
    "S0234c": 26,
    "S0254b": 225,
    "S0345a": 55,
    "S0395a": 98,
    "S0421b": 177,
    "S0976a": 282,
    "S1102a": 254,
    "S1153a": 243,
    "S1415a": 241,
}

BEST_STRIKES = {
    "S0133a": {"fig2_model": "X",     "fig2_best_strike_deg": 247.606, "tayak_best_strike_deg": 301.750, "x_best_strike_deg": 247.606},
    "S0152a": {"fig2_model": "X",     "fig2_best_strike_deg": 243.055, "tayak_best_strike_deg": 299.018, "x_best_strike_deg": 243.055},
    "S0167a": {"fig2_model": "X",     "fig2_best_strike_deg": 237.158, "tayak_best_strike_deg": 80.583,  "x_best_strike_deg": 237.158},
    "S0167b": {"fig2_model": "X",     "fig2_best_strike_deg": 231.207, "tayak_best_strike_deg": 119.908, "x_best_strike_deg": 231.207},
    "S0185a": {"fig2_model": "X",     "fig2_best_strike_deg": 305.893, "tayak_best_strike_deg": 185.953, "x_best_strike_deg": 305.893},
    "S0226b": {"fig2_model": "X",     "fig2_best_strike_deg": 120.769, "tayak_best_strike_deg": 84.125,  "x_best_strike_deg": 120.769},
    "S0234c": {"fig2_model": "X",     "fig2_best_strike_deg": 220.727, "tayak_best_strike_deg": 322.989, "x_best_strike_deg": 220.727},
    "S0254b": {"fig2_model": "TAYAK", "fig2_best_strike_deg": 251.728, "tayak_best_strike_deg": 251.728, "x_best_strike_deg": 239.980},
    "S0345a": {"fig2_model": "X",     "fig2_best_strike_deg": 241.950, "tayak_best_strike_deg": 234.872, "x_best_strike_deg": 241.950},
    "S0395a": {"fig2_model": "X",     "fig2_best_strike_deg": 279.882, "tayak_best_strike_deg": 236.875, "x_best_strike_deg": 279.882},
    "S0421b": {"fig2_model": "TAYAK", "fig2_best_strike_deg": 8.556,   "tayak_best_strike_deg": 8.556,   "x_best_strike_deg": 203.700},
    "S0976a": {"fig2_model": "X",     "fig2_best_strike_deg": 111.640, "tayak_best_strike_deg": 130.447, "x_best_strike_deg": 111.640},
    "S1102a": {"fig2_model": "X",     "fig2_best_strike_deg": 39.813,  "tayak_best_strike_deg": 119.194, "x_best_strike_deg": 39.813},
    "S1153a": {"fig2_model": "TAYAK", "fig2_best_strike_deg": 62.040,  "tayak_best_strike_deg": 62.040,  "x_best_strike_deg": 234.896},
    "S1415a": {"fig2_model": "X",     "fig2_best_strike_deg": 45.571,  "tayak_best_strike_deg": 45.574,  "x_best_strike_deg": 45.571},
}


def axial_deg(angle_deg: float) -> float:
    return angle_deg % 180.0


def axial_delta(a_deg: float, b_deg: float) -> float:
    d = abs(axial_deg(a_deg) - axial_deg(b_deg))
    return min(d, 180.0 - d)


def build_dataframe() -> pd.DataFrame:
    rows = []
    for ev, rot in ROTATIONS_PPT_DEG.items():
        info = BEST_STRIKES[ev]
        mapped = axial_deg(rot)
        fig2_best = info["fig2_best_strike_deg"]
        tayak_best = info["tayak_best_strike_deg"]
        x_best = info["x_best_strike_deg"]

        rows.append({
            "event": ev,
            "ppt_rotation_deg": rot,
            "mapped_fault_strike_axial_deg": mapped,
            "fig2_model": info["fig2_model"],
            "fig2_best_strike_deg": fig2_best,
            "fig2_best_strike_axial_deg": axial_deg(fig2_best),
            "delta_fig2_deg": axial_delta(mapped, fig2_best),
            "tayak_best_strike_deg": tayak_best,
            "tayak_best_strike_axial_deg": axial_deg(tayak_best),
            "delta_tayak_deg": axial_delta(mapped, tayak_best),
            "x_best_strike_deg": x_best,
            "x_best_strike_axial_deg": axial_deg(x_best),
            "delta_x_deg": axial_delta(mapped, x_best),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("event").reset_index(drop=True)


def plot_histogram(df: pd.DataFrame, metric: str, bin_width: float, out_png: Path) -> None:
    bins = np.arange(0.0, 90.0 + bin_width, bin_width)
    plt.figure(figsize=(7, 4.5))
    plt.hist(df[metric], bins=bins)
    plt.xlabel("Angular deviation (degrees)")
    plt.ylabel("Number of faults")
    plt.title("Mapped local fault vs. best strike deviation")
    plt.xlim(0, 45)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare PowerPoint-measured mapped local fault strikes with best-fit strikes.")
    ap.add_argument("--out_csv", default="figs/mapped_fault_strike_comparison.csv", help="Output CSV path.")
    ap.add_argument("--out_png", default="figs/mapped_fault_strike_histogram.png", help="Output histogram PNG path.")
    ap.add_argument("--metric", default="delta_fig2_deg", choices=["delta_fig2_deg", "delta_tayak_deg", "delta_x_deg"], help="Which delta column to histogram.")
    ap.add_argument("--bin_width", type=float, default=10.0, help="Histogram bin width in degrees.")
    args = ap.parse_args()

    df = build_dataframe()
    df.to_csv(args.out_csv, index=False)
    plot_histogram(df, args.metric, args.bin_width, Path(args.out_png))
    print(f"[done] wrote {args.out_csv}")
    print(f"[done] wrote {args.out_png}")


if __name__ == "__main__":
    main()
