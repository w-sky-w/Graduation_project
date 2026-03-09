from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.fingerflex_baseline import run_baseline


def run_experiment(
    mat_path: str,
    dropouts: list[float],
    repeats: int,
    out_csv: str,
    out_plot: str,
):
    rows = []
    for d in dropouts:
        for rep in range(repeats):
            seed = 1000 + rep
            metrics = run_baseline(mat_path=mat_path, dropout_ratio=d, seed=seed)
            metrics["repeat"] = rep
            rows.append(metrics)
            print(f"dropout={d:.2f}, repeat={rep}, corr_mean={metrics['corr_mean']:.4f}, r2_mean={metrics['r2_mean']:.4f}")

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby("dropout_ratio")[ ["corr_mean", "r2_mean"] ]
        .agg(["mean", "std"])
        .reset_index()
    )

    x = summary["dropout_ratio"].values
    corr_mean = summary[("corr_mean", "mean")].values
    corr_std = summary[("corr_mean", "std")].fillna(0).values
    r2_mean = summary[("r2_mean", "mean")].values
    r2_std = summary[("r2_mean", "std")].fillna(0).values

    plt.figure(figsize=(8, 5))
    plt.errorbar(x, corr_mean, yerr=corr_std, marker="o", capsize=4, label="Corr mean")
    plt.errorbar(x, r2_mean, yerr=r2_std, marker="s", capsize=4, label="R2 mean")
    plt.xlabel("Electrode dropout ratio")
    plt.ylabel("Performance")
    plt.title("Robustness under electrode dropout")
    plt.grid(alpha=0.3)
    plt.legend()
    Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)

    print(f"Saved metrics to: {out_csv}")
    print(f"Saved plot to: {out_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dropout robustness experiment")
    parser.add_argument("--mat", required=True, help="Path to ##_fingerflex.mat")
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0, 0.05, 0.1, 0.15, 0.2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-csv", default="results/dropout_metrics.csv")
    parser.add_argument("--out-plot", default="results/dropout_curve.png")
    args = parser.parse_args()

    run_experiment(
        mat_path=args.mat,
        dropouts=args.dropouts,
        repeats=args.repeats,
        out_csv=args.out_csv,
        out_plot=args.out_plot,
    )
