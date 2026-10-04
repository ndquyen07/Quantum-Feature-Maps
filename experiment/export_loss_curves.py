"""Plot the 3 training losses across the 3 datasets and export TikZ-ready CSV.

Produces, for a given depth:
  * a 1x3 matplotlib figure (one panel per dataset, 3 mean loss trajectories each)
  * one CSV per dataset in ``tikz_data/`` for pgfplots ``\addplot table``

Run with the ``svqsvm`` env (the one that has qiskit_algorithms):

    python experiment/export_loss_curves.py --depth 4
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# The pickled models were written when feature_map.py was named feature_map_1.py.
import src.feature_map as _fm  # noqa: E402

sys.modules.setdefault("src.feature_map_1", _fm)

PRETTY = {"circle": "Circle", "moon": "Moon", "gauss_3class": "Gaussian (3-class)"}

# Noiseless runs have all 3 losses; the *_loglik_from_Eff_noisy runs have only
# log_likelihood_macro (the loglik entry scripts override loss_functions).
NOISELESS_DIRS = {k: PROJECT_ROOT / f"results_losses_{k}_Eff" for k in PRETTY}
NOISY_DIRS = {k: PROJECT_ROOT / f"results_losses_{k}_loglik_from_Eff_noisy" for k in PRETTY}

LOSSES = {
    "hilbert_schmidt": ("Hilbert-Schmidt", "hs", "tab:blue"),
    "pre_determine_inner": ("Inner Product", "ip", "tab:green"),
    "log_likelihood_macro": ("Log Likelihood", "ll", "tab:orange"),
}


def load_histories(results_dir: Path, loss_type: str, depth: int) -> np.ndarray:
    """Return (n_runs, n_iter) array of loss trajectories, truncated to the shortest run."""
    histories = []
    for pkl_file in sorted(results_dir.glob(f"tqfm_{loss_type}_depth{depth}_*.pkl")):
        with open(pkl_file, "rb") as f:
            tqfm = pickle.load(f)
        if tqfm.loss_history:
            histories.append(np.asarray(tqfm.loss_history, dtype=float))

    if not histories:
        return np.empty((0, 0))

    min_len = min(len(h) for h in histories)
    return np.stack([h[:min_len] for h in histories])


DATASET_COLORS = {"circle": "tab:blue", "moon": "tab:green", "gauss_3class": "tab:orange"}


def combined_mode(args, dirs, tag) -> None:
    """One axes, one loss, all three datasets overlaid."""
    label, short, _ = LOSSES[args.loss]
    fig, ax = plt.subplots(figsize=(7, 5))
    columns: dict[str, np.ndarray] = {}

    for key, results_dir in dirs.items():
        runs = load_histories(results_dir, args.loss, args.depth)
        if runs.size == 0:
            print(f"[warn] no runs: {key} / {args.loss} / depth {args.depth}")
            continue

        mean = runs.mean(axis=0)
        std = runs.std(axis=0)
        iters = np.arange(len(mean))
        color = DATASET_COLORS[key]

        ax.plot(iters, mean, color=color, lw=1.8, label=PRETTY[key])

        columns.setdefault("iter", iters)
        # datasets have different lengths → pad short columns with NaN to a common grid
        columns[f"mean_{key}"] = mean
        columns[f"std_{key}"] = std
        columns[f"lo_{key}"] = mean - std
        columns[f"hi_{key}"] = mean + std

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"{label} loss across datasets (depth {args.depth}, {tag})")
    ax.grid(alpha=0.3)
    ax.legend()
    if args.logy:
        ax.set_yscale("log")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig_path = args.out_dir / f"loss_curves_combined_{tag}_{short}_depth{args.depth}.{ext}"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"wrote {fig_path}")

    # Pad every column to the longest dataset so pgfplots gets a rectangular table.
    n = max(len(v) for v in columns.values())
    names = ["iter"] + [c for c in columns if c != "iter"]
    grid = np.arange(n)
    keep = np.unique(np.r_[np.arange(0, n, args.stride), n - 1])

    padded = {"iter": grid}
    for c in names[1:]:
        col = np.full(n, np.nan)
        col[: len(columns[c])] = columns[c]
        padded[c] = col

    table = np.column_stack([padded[c][keep] for c in names])
    csv_path = args.out_dir / f"loss_curves_combined_{tag}_{short}_depth{args.depth}.csv"
    np.savetxt(csv_path, table, delimiter=",", header=",".join(names), comments="",
               fmt="%.6g")
    print(f"wrote {csv_path}  ({table.shape[0]} rows x {table.shape[1]} cols; NaN = past run end)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--noisy", action="store_true", help="use the *_loglik_from_Eff_noisy dirs")
    parser.add_argument("--stride", type=int, default=10, help="row decimation for the CSV")
    parser.add_argument("--logy", action="store_true", help="log-scale the loss axis")
    parser.add_argument("--loss", default="log_likelihood_macro", choices=list(LOSSES),
                        help="which loss to plot in --combined mode")
    parser.add_argument("--combined", action="store_true",
                        help="overlay the 3 datasets (one loss) on a single axes")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).parent / "tikz_data")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dirs = NOISY_DIRS if args.noisy else NOISELESS_DIRS
    tag = "noisy" if args.noisy else "noiseless"

    if args.combined:
        combined_mode(args, dirs, tag)
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

    for ax, (key, results_dir) in zip(axes, dirs.items()):
        pretty = PRETTY[key]
        columns: dict[str, np.ndarray] = {}

        for loss_type, (label, short, color) in LOSSES.items():
            runs = load_histories(results_dir, loss_type, args.depth)
            if runs.size == 0:
                print(f"[warn] no runs: {key} / {loss_type} / depth {args.depth}")
                continue

            mean = runs.mean(axis=0)
            std = runs.std(axis=0)
            iters = np.arange(len(mean))

            ax.plot(iters, mean, color=color, lw=1.8, label=f"{label} (n={len(runs)})")
            ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.18, lw=0)

            columns.setdefault("iter", iters)
            columns[f"mean_{short}"] = mean
            columns[f"std_{short}"] = std
            columns[f"lo_{short}"] = mean - std
            columns[f"hi_{short}"] = mean + std

        ax.set_title(f"{pretty} (depth {args.depth}, {tag})")
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if args.logy:
            ax.set_yscale("log")

        if not columns:
            continue

        # Decimate, always keeping the final iteration.
        n = len(columns["iter"])
        keep = np.unique(np.r_[np.arange(0, n, args.stride), n - 1])
        names = list(columns)
        table = np.column_stack([columns[c][keep] for c in names])

        csv_path = args.out_dir / f"loss_curves_{tag}_depth{args.depth}_{key}.csv"
        np.savetxt(csv_path, table, delimiter=",", header=",".join(names), comments="", fmt="%.6g")
        print(f"wrote {csv_path}  ({table.shape[0]} rows x {table.shape[1]} cols)")

    axes[0].set_ylabel("Loss")
    fig.suptitle(f"Training loss trajectories, {tag} (mean $\\pm$ 1 std over runs)")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig_path = args.out_dir / f"loss_curves_{tag}_depth{args.depth}.{ext}"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
