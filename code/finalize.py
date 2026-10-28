#!/usr/bin/env python3
"""Turn the finished run into deliverables: a readable report and one archive.

Produces, under <root>/:
    REPORT.md          every table, with CI and paired tests, plus a health check
    RESULTS.xlsx       one sheet per experiment (skipped if openpyxl is absent)
    run_summary.txt    per-job status and wall time
    tadr_results_<stamp>.tar.gz   report + all CSVs + all logs
"""
from __future__ import annotations

import argparse, datetime, glob, os, re, subprocess, sys, tarfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TITLES = {
    "1": "Refiner comparison — SMD machine-1-1, 40% block missing",
    "2": "Global P2P F1 — four datasets, 40% block missing",
    "3": "Detector generality with and without TADR — pooled SMD",
    "4": "Global P2P F1 — varying block-missing ratios",
    "5": "Ablation on loss components — smoothed P2P",
    "6": "System inference overhead — batch size 256",
    "7": "Hyperparameter sensitivity — lambda_feat x lambda_enhance",
    "8": "Edge deployment profiling",
    "9": "Redundancy ablation — channels and window overlap",
    "10": "Anomaly-coverage decomposition",
    "11": "Sampling-step budget — DDIM stride vs F1",
}
PAPER_EXPS = {"1", "2", "3", "4", "5", "6"}

# A refiner is only credible if it still reconstructs the signal, and these
# catch the failure mode where a degenerate output scores well on F1 anyway.
#
# Saturation is scale-free, so it is an absolute bound. Reconstruction error is
# NOT: after per-entity standardisation, 38 of 110 entities contain channels
# that are near-constant in train and vary in test, which sends scaled test
# values to 1e4-1e5 and the MSE with them. An absolute MSE bound therefore
# flags whole datasets rather than bad refiners. The error is compared within
# its own (experiment, dataset, ratio, condition) group instead: every method
# there saw identical data, so a refiner whose error is far above the best of
# its peers is genuinely the outlier.
SAT_MAX_PCT = 1.0
OBS_MSE_RATIO_MAX = 10.0


def fmt(df, floats=4):
    """Markdown table without pulling in tabulate."""
    if df is None or not len(df):
        return "_(no rows)_"

    def cell(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.{floats}f}"
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    cols = list(df.columns)
    body = [[cell(v) for v in row] for row in df.itertuples(index=False, name=None)]
    width = [max(len(str(c)), *(len(r[i]) for r in body)) if body else len(str(c))
             for i, c in enumerate(cols)]
    numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in cols]

    def line(vals):
        return "| " + " | ".join(
            v.rjust(width[i]) if numeric[i] else v.ljust(width[i])
            for i, v in enumerate(vals)) + " |"

    return "\n".join([
        line([str(c) for c in cols]),
        "|" + "|".join(("-" * (width[i] + 1)) + (":" if numeric[i] else "-")
                       for i in range(len(cols))) + "|",
        *(line(r) for r in body),
    ])


CONTEXT_COLS = ("Dataset", "Ratio", "Condition", "steps", "knob", "setting")
LABEL_COLS = ("Refiner", "Method", "Loss", "Detector")


def health_flags(root):
    """Rows whose reconstruction diagnostics say the F1 cannot be trusted."""
    bad = []
    for f in sorted(glob.glob(os.path.join(root, "exp*_runs.csv"))):
        exp = re.match(r"exp(\w+)_runs\.csv", os.path.basename(f)).group(1)
        d = pd.read_csv(f)
        if "obs_mse" not in d.columns or not d["obs_mse"].notna().any():
            continue

        label = next((c for c in LABEL_COLS if c in d.columns), None)
        ctx = [c for c in CONTEXT_COLS if c in d.columns]
        d = d[d["obs_mse"].notna()].copy()
        # Peer floor: the best reconstruction achieved under identical conditions.
        d["_floor"] = (d.groupby(ctx + ["seed"])["obs_mse"].transform("min")
                       if ctx else d.groupby("seed")["obs_mse"].transform("min"))
        d["_ratio"] = d["obs_mse"] / d["_floor"].clip(lower=1e-12)

        for _, r in d.iterrows():
            why = []
            if pd.notna(r.get("sat_pct")) and r["sat_pct"] > SAT_MAX_PCT:
                why.append(f"output saturated on {r['sat_pct']:.1f}% of samples")
            if r["_ratio"] > OBS_MSE_RATIO_MAX:
                why.append(f"reconstruction {r['_ratio']:.0f}x worse than the best "
                           f"method on the same data")
            if why:
                row = {"exp": exp, "seed": r.get("seed"),
                       "row": str(r[label]) if label else "?"}
                row.update({c: r[c] for c in ctx})
                row.update({"F1": r.get("F1"), "obs_mse": r["obs_mse"],
                            "issue": "; ".join(why)})
                bad.append(row)
    return pd.DataFrame(bad)


def job_summary(root, logdir):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "exp*_seed*"))):
        m = re.search(r"exp(\w+)_seed(\d+)$", d)
        if not m:
            continue
        exp, seed = m.groups()
        done = bool(glob.glob(os.path.join(d, "exp*_runs.csv")))
        log = os.path.join(logdir, f"exp{exp}_seed{seed}.log")
        rows.append({"exp": exp, "seed": int(seed),
                     "status": "ok" if done else "INCOMPLETE",
                     "log_kb": round(os.path.getsize(log) / 1024, 1)
                     if os.path.exists(log) else 0})
    return pd.DataFrame(rows).sort_values(["exp", "seed"]) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "results_final"))
    ap.add_argument("--logs", default=os.path.join(HERE, "logs"))
    args = ap.parse_args()
    root, logs = args.root, args.logs

    subprocess.run([sys.executable, os.path.join(HERE, "merge_results.py"),
                    "--root", root], check=False)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    jobs = job_summary(root, logs)
    flags = health_flags(root)

    out = [f"# TADR — experiment results\n",
           f"Generated {stamp}\n",
           "All figures are means over independent seeds with 95% confidence "
           "intervals. `_ci95` columns are half-widths: the interval is "
           "`mean ± ci95`. `*_paired` tables test each method against a "
           "reference on per-seed differences, which is the comparison the "
           "paper's claims actually rest on.\n",
           "Every table carries reconstruction diagnostics next to F1 "
           "(`out_std`, `sat_pct`, `obs_mse`, `miss_mse`). Best-threshold P2P "
           "F1 alone cannot distinguish a genuine refinement from a degenerate "
           "one — a refiner saturating at its clipping bound, with "
           "reconstruction 30x worse than doing nothing, still scored F1 0.69. "
           "Read the F1 columns together with these.\n"]

    if len(jobs):
        n_ok = (jobs.status == "ok").sum()
        out += [f"## Run status\n\n{n_ok} / {len(jobs)} jobs complete.\n"]
        if n_ok < len(jobs):
            out += ["\n**Incomplete jobs — these tables are partial:**\n\n",
                    fmt(jobs[jobs.status != "ok"], 0), "\n"]

    out += ["\n## Reconstruction health check\n"]
    if len(flags):
        out += [f"\n**{len(flags)} row(s) failed the fidelity check.** Their F1 "
                "should not be quoted without checking what the refiner actually "
                "produced.\n\n", fmt(flags), "\n"]
    else:
        out += ["\nAll rows pass: no output saturation, reconstruction error in range.\n"]

    for group, label in ((PAPER_EXPS, "Paper tables"), (None, "Additional experiments")):
        ids = sorted([e for e in TITLES
                      if os.path.exists(os.path.join(root, f"exp{e}.csv"))
                      and ((e in PAPER_EXPS) if group else (e not in PAPER_EXPS))],
                     key=lambda x: int(x))
        if not ids:
            continue
        out += [f"\n---\n\n# {label}\n"]
        for e in ids:
            out += [f"\n## exp{e} — {TITLES[e]}\n\n",
                    fmt(pd.read_csv(os.path.join(root, f"exp{e}.csv"))), "\n"]
            pf = os.path.join(root, f"exp{e}_paired.csv")
            if os.path.exists(pf):
                out += ["\n**Paired comparison against reference**\n\n",
                        fmt(pd.read_csv(pf)), "\n"]

    report = os.path.join(root, "REPORT.md")
    with open(report, "w") as f:
        f.write("".join(out))

    if len(jobs):
        jobs.to_csv(os.path.join(root, "run_summary.txt"), sep="\t", index=False)

    try:
        xl = os.path.join(root, "RESULTS.xlsx")
        with pd.ExcelWriter(xl) as w:
            for e in sorted(TITLES, key=lambda x: int(x)):
                p = os.path.join(root, f"exp{e}.csv")
                if os.path.exists(p):
                    pd.read_csv(p).to_excel(w, sheet_name=f"exp{e}", index=False)
            if len(flags):
                flags.to_excel(w, sheet_name="health_flags", index=False)
    except Exception as exc:
        print(f"xlsx skipped: {exc}")

    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    bundle = os.path.join(root, f"tadr_results_{tag}.tar.gz")
    with tarfile.open(bundle, "w:gz") as t:
        for p in sorted(glob.glob(os.path.join(root, "*.csv"))
                        + glob.glob(os.path.join(root, "*.md"))
                        + glob.glob(os.path.join(root, "*.xlsx"))
                        + glob.glob(os.path.join(root, "*.txt"))):
            t.add(p, arcname=f"results/{os.path.basename(p)}")
        for p in sorted(glob.glob(os.path.join(logs, "*.log"))):
            t.add(p, arcname=f"logs/{os.path.basename(p)}")
        for p in ("tadr_experiments.py", "rebuttal_experiments.py", "diffusion.py",
                  "merge_results.py", "run_all.sh", "finalize.py"):
            fp = os.path.join(HERE, p)
            if os.path.exists(fp):
                t.add(fp, arcname=f"code/{p}")

    print(f"\nREPORT   {report}")
    print(f"BUNDLE   {bundle}  ({os.path.getsize(bundle) / 1024**2:.1f} MB)")
    if len(flags):
        print(f"WARNING  {len(flags)} row(s) failed the reconstruction health check")


if __name__ == "__main__":
    main()
