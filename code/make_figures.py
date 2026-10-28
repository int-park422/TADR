#!/usr/bin/env python3
"""Draw the paper's result figures from the aggregated CSVs.

    Fig. 3  dataset_generality.png    exp2   four pooled datasets, 40% block missing
    Fig. 4  refiner_comparison.png    exp1   refiner comparison, SMD machine-1-1
    Fig. 5  detector_generality.png   exp3   three detectors, pooled SMD
    Fig. 6  lambda_sensitivity.png    exp7   lambda_feat x lambda_enhance grid

Reads only results/exp<N>.csv and exp<N>_paired.csv, so it runs on the published
results as well as on a fresh run:

    python3 make_figures.py --results ../results --out ../figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"Raw": "#636EFA", "iTransformer": "#EF553B", "TIMBA": "#00CC96",
          "CSDI": "#AB63FA", "TADR": "#FF6692"}
METRIC_COLORS = {"P2P F1": "#636EFA", "Precision": "#EF553B", "Recall": "#00CC96"}
POS, NEG, MUTED, INK = "#636EFA", "#EF553B", "#6B7280", "#1F2937"

# Continuous channels per dataset, as counted by continuous_channels() in
# tadr_experiments.py. The gain tracks this number, so the axis labels carry it.
CHANNELS = {"SMD": "33 / 38", "PSM": "25 / 25", "SMAP": "1 / 25", "MSL": "1 / 55"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#D1D5DB",
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E5E7EB",
    "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight",
})
ERR = dict(ecolor="#374151", elinewidth=1.2, capsize=3)


def _read(results, name):
    return pd.read_csv(os.path.join(results, name))


def _short(name):
    """'TADR (Ours)' / 'Raw (No Refiner)' -> the key used for colours."""
    return name.split(" (")[0]


def _signed(x):
    return f"{x:+.3f}"


def dataset_generality(results, out):
    df = _read(results, "exp2.csv")
    df = df[np.isclose(df.Ratio, 0.4)]
    paired = _read(results, "exp2_paired.csv")
    datasets = [d for d in ["SMD", "PSM", "SMAP", "MSL"] if d in set(df.Dataset)]
    methods = ["Raw", "iTransformer", "TIMBA", "CSDI", "TADR"]
    width = 0.16

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, ds in enumerate(datasets):
        if ds == "SMD":
            ax.axvspan(i - 0.5, i + 0.5, color="#EEF0FB", zorder=0)
        sub = df[df.Dataset == ds].set_index("Method")
        for j, m in enumerate(methods):
            x = i + (j - 2) * width
            ax.bar(x, sub.loc[m, "F1_mean"], width * 0.92, color=COLORS[m],
                   yerr=sub.loc[m, "F1_ci95"], error_kw=ERR,
                   label=("TADR (ours)" if m == "TADR" else m) if i == 0 else None)
        d = paired[(paired.Dataset == ds) & np.isclose(paired.Ratio, 0.4)
                   & (paired.method == "TADR")].mean_diff.iloc[0]
        top = (sub.F1_mean + sub.F1_ci95).max()
        ax.text(i, top + 0.04, _signed(d), ha="center", fontsize=13,
                color=POS if d > 0 else MUTED, fontweight="bold" if d > 0 else None)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([f"{d}\n{CHANNELS[d]} continuous ch." for d in datasets])
    for lbl in ax.get_xticklabels():
        if lbl.get_text().startswith("SMD"):
            lbl.set_fontweight("bold")
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Global P2P F1")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    fig.savefig(os.path.join(out, "dataset_generality.png"))
    plt.close(fig)


def refiner_comparison(results, out):
    df = _read(results, "exp1.csv")
    paired = _read(results, "exp1_paired.csv")
    delta = {_short(m): d for m, d in zip(paired.method, paired.mean_diff)}
    df["key"] = df.Refiner.map(_short)
    df = df.sort_values("F1_mean")          # best at the top of the chart
    bars = [("Recall", "Recall"), ("Precision", "Precision"), ("P2P F1", "F1")]
    h = 0.26

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, (_, row) in enumerate(df.iterrows()):
        ours = row.key == "TADR"
        if ours:
            ax.axhspan(i - 0.48, i + 0.48, color="#EEF0FB", zorder=0)
        for k, (label, col) in enumerate(bars):
            y = i + (k - 1) * h
            v = row[f"{col}_mean"]
            err = row["F1_ci95"] if col == "F1" else None
            ax.barh(y, v, h * 0.9, color=METRIC_COLORS[label], xerr=err,
                    error_kw=ERR, label=label if i == 0 else None)
            ax.text(v + (err or 0) + 0.01, y, f"{v:.3f}", va="center", fontsize=9.5,
                    color=INK if ours else MUTED, fontweight="bold" if ours else None)
        d = delta.get(row.key)
        ax.text(1.32, i, "ref" if d is None else _signed(d), ha="right", va="center",
                fontsize=12, color=MUTED if d is None else (POS if d > 0 else NEG),
                fontweight="bold" if ours else None)

    ax.text(1.32, len(df) - 0.45, "ΔF1 vs. Raw", ha="right", fontsize=9,
            style="italic", color=MUTED)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(["TADR (ours)" if k == "TADR" else k for k in df.key], fontsize=12)
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith("TADR"):
            lbl.set_fontweight("bold")
    ax.set_xlim(0, 1.33)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="x", color="#E5E7EB")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Score")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.45, 1.1),
              frameon=False)
    fig.savefig(os.path.join(out, "refiner_comparison.png"))
    plt.close(fig)


def detector_generality(results, out):
    df = _read(results, "exp3.csv")
    detectors = ["KAN-AD", "TimesNet", "Anomaly Transformer"]
    panels = [("Block Missing", "40% block missing"), ("Gaussian", "Gaussian noise")]
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        sub = df[df.Condition.str.startswith(key)].set_index("Detector")
        for i, det in enumerate(detectors):
            r = sub.loc[det]
            ax.bar(i - width / 2, r.Raw_F1_mean, width * 0.92, color=NEG,
                   label="Detector alone" if i == 0 else None)
            ax.bar(i + width / 2, r.TADR_F1_mean, width * 0.92, color=POS,
                   yerr=r.TADR_F1_ci95, error_kw=ERR, label="+ TADR" if i == 0 else None)
            top = max(r.Raw_F1_mean, r.TADR_F1_mean + r.TADR_F1_ci95)
            gain = r.Delta_mean > 0
            ax.text(i, top + 0.012, _signed(r.Delta_mean), ha="center", fontsize=12,
                    color=INK if gain else MUTED, fontweight="bold" if gain else None)
        ax.set_title(title, color=INK)
        ax.set_xticks(range(len(detectors)))
        ax.set_xticklabels([d.replace(" ", "\n") for d in detectors])
    axes[0].set_ylabel("Global P2P F1")
    axes[0].set_ylim(0, 0.31)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               frameon=False)
    fig.savefig(os.path.join(out, "detector_generality.png"))
    plt.close(fig)


def lambda_sensitivity(results, out, chosen=(0.1, 0.5)):
    df = _read(results, "exp7.csv")
    grid = df.pivot(index="lambda_feat", columns="lambda_enhance", values="F1_mean")
    feats, enhs = list(grid.index), list(grid.columns)

    fig, ax = plt.subplots(figsize=(5.2, 4))
    im = ax.imshow(grid.values, cmap="Blues", vmin=0.28, vmax=0.8, aspect="auto")
    for i in range(len(feats)):
        for j in range(len(enhs)):
            v = grid.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if v > 0.55 else INK)
    i, j = feats.index(chosen[0]), enhs.index(chosen[1])
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=NEG, linewidth=3))
    ax.set_xticks(range(len(enhs)))
    ax.set_xticklabels([f"{v:g}" for v in enhs])
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([f"{v:g}" for v in feats])
    ax.set_xlabel(r"$\lambda_{\mathrm{enhance}}$")
    ax.set_ylabel(r"$\lambda_{\mathrm{feat}}$")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, label="P2P F1")
    fig.savefig(os.path.join(out, "lambda_sensitivity.png"))
    plt.close(fig)


FIGURES = {
    "dataset_generality": dataset_generality,
    "refiner_comparison": refiner_comparison,
    "detector_generality": detector_generality,
    "lambda_sensitivity": lambda_sensitivity,
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "..", "results"))
    ap.add_argument("--out", default=os.path.join(here, "..", "figures"))
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.only or FIGURES:
        FIGURES[name](args.results, args.out)
        print(f"wrote {os.path.join(args.out, name + '.png')}")


if __name__ == "__main__":
    main()
