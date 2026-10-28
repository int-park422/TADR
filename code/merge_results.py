#!/usr/bin/env python3
"""Merge per-(experiment, seed) result shards into the final tables.

The suite is run as one process per (experiment, seed) so that jobs are
independent, land incrementally, and can be spread across GPUs. This collects
the shards, aggregates over seeds, and runs the paired tests.
"""
import argparse, glob, os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tadr_experiments import EXPERIMENTS, aggregate, paired_tests
from rebuttal_experiments import EXPERIMENTS as REB


def spec(exp_id):
    if exp_id in EXPERIMENTS:
        _, _, keys, paired = EXPERIMENTS[exp_id]
        return keys, paired
    _, _, keys = REB[exp_id]
    ref = {"11": ("Method", "Raw"), "9": ("Method", "Raw"), "10": ("Method", "Raw")}
    return keys, ref.get(exp_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_final")
    args = ap.parse_args()

    shards = {}
    for f in sorted(glob.glob(os.path.join(args.root, "*", "exp*_runs.csv"))):
        m = re.match(r"exp(\w+)_runs\.csv", os.path.basename(f))
        shards.setdefault(m.group(1), []).append(f)

    if not shards:
        print(f"no shards under {args.root}")
        return

    for exp_id, files in sorted(shards.items()):
        raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        raw = raw.drop_duplicates()
        keys, paired = spec(exp_id)
        keys = [k for k in keys if k in raw.columns]

        raw.to_csv(os.path.join(args.root, f"exp{exp_id}_runs.csv"), index=False)
        agg = aggregate(raw, keys)
        agg.to_csv(os.path.join(args.root, f"exp{exp_id}.csv"), index=False)

        n_seeds = raw["seed"].nunique()
        note = ""
        if paired and n_seeds > 1 and paired[0] in raw.columns and "F1" in raw.columns:
            pt = paired_tests(raw, keys, paired[0], "F1", paired[1])
            if len(pt):
                pt.to_csv(os.path.join(args.root, f"exp{exp_id}_paired.csv"), index=False)
                note = f", {len(pt)} paired tests"
        print(f"exp{exp_id}: {len(files)} shards, {n_seeds} seeds, {len(agg)} rows{note}")


if __name__ == "__main__":
    main()
