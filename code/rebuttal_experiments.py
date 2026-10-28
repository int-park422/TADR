#!/usr/bin/env python3
"""
Additional experiments answering the HiPC 2026 reviews.

  exp7   Hyperparameter sensitivity: lambda_feat x lambda_enhance grid
         -> Reviewer 2: "values are set based on some analysis which is not shown"

  exp8   Edge deployment profiling: CPU / single-thread / batch-1 / FP16 / INT8
         -> Reviewers 1 & 3: high-end server does not proxy a constrained device

  exp9   Redundancy ablation: fewer channels, less window overlap
         -> Reviewer 1: "relies on spatial/temporal redundancy and overlapping
            windows; low-correlation or sparse settings are not established"

  exp11  Sampling-step budget: does DDIM stride buy latency without costing F1?
         -> the evidence for (or against) the step-distillation direction the
            paper proposes in its future-work section, and the knob that decides
            whether the whole evaluation suite is affordable at 5 seeds

  exp10  Anomaly-coverage decomposition: where does the gain actually come from?
         -> Reviewer 2: "the learners can only succeed if there is information in
            the signal immediately preceding the anomaly about the anomaly"

Usage
-----
    python rebuttal_experiments.py --exp 8              # no dataset needed
    python rebuttal_experiments.py --exp 7 --runs 3
    python rebuttal_experiments.py --exp all --runs 5
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tadr_experiments import (
    CFG, CSDI_Lite, FrozenDetector, KAN_AD_Lite, NoiseInjector, TaskAwareLoss, build_refiner,
    aggregate, best_f1_with_pr, create_sliding_windows, entity_scores,
    load_and_scale_dataset, make_diffusion_refiner, refinement_diagnostics,
    set_seed, train_baseline_refiner, train_tadr, train_detector,
)
from diffusion import DiffusionRefiner


# =============================================================================
# Shared fixture: SMD machine-1-1 with a frozen KAN-AD detector
# =============================================================================

def build_smd_fixture(seed, entity="machine-1-1", keep_channels=None):
    """Returns (train_dataset, test_windows, labels, frozen_detector, feat_dim).

    `keep_channels` selects a random channel subset, used by exp9 to reduce the
    cross-sensor redundancy the method is accused of depending on.
    """
    set_seed(seed)
    tr, te, labels = load_and_scale_dataset("SMD", entity)

    if keep_channels is not None and keep_channels < tr.shape[1]:
        rng = np.random.default_rng(seed)
        cols = np.sort(rng.choice(tr.shape[1], size=keep_channels, replace=False))
        tr, te = tr[:, cols], te[:, cols]

    tr_win = create_sliding_windows(tr, CFG.window_size, CFG.train_stride)
    te_win = create_sliding_windows(te, CFG.window_size, CFG.test_stride)
    feat_dim = tr_win.shape[-1]
    train_ds = TensorDataset(tr_win)

    det = FrozenDetector(
        train_detector(KAN_AD_Lite(CFG.window_size, feat_dim).to(CFG.device),
                       train_ds, CFG.detector_epochs)
    ).to(CFG.device)
    return train_ds, te_win, labels, det, feat_dim


def mean_abs_channel_corr(x: np.ndarray) -> float:
    """Mean |pairwise Pearson r| across channels -- the redundancy TADR exploits."""
    c = np.corrcoef(np.nan_to_num(x), rowvar=False)
    iu = np.triu_indices_from(c, k=1)
    return float(np.nanmean(np.abs(c[iu])))


# =============================================================================
# exp7 -- hyperparameter sensitivity
# =============================================================================

LAMBDA_FEAT_GRID = [0.0, 0.05, 0.1, 0.2, 0.5]
LAMBDA_ENHANCE_GRID = [0.0, 0.1, 0.5, 1.0, 2.0]


def exp7_lambda_sensitivity(seed):
    """P2P F1/P/R over the (lambda_feat, lambda_enhance) grid, SMD 1-1, 40% missing.

    The paper reports (0.1, 0.5) without showing how it was chosen. This is the
    missing selection evidence, and it also shows how sharp the optimum is.
    """
    train_ds, te_win, labels, det, feat_dim = build_smd_fixture(seed)
    train_inj = NoiseInjector("mixed", missing_ratio=0.2)
    test_inj = NoiseInjector("block_missing", missing_ratio=0.4)

    rows = []
    for lf in LAMBDA_FEAT_GRID:
        for le in LAMBDA_ENHANCE_GRID:
            loss_fn = TaskAwareLoss(use_feat=lf > 0, use_enhance=le > 0,
                                    lambda_feat=lf, lambda_enhance=le)
            refiner = train_tadr(build_refiner(None, feat_dim, "SMD"), det, train_ds,
                                 train_inj, CFG.refiner_epochs, loss_fn=loss_fn)
            scores = entity_scores(refiner, det, te_win, len(labels), test_inj, seed)
            f1, p, r, _ = best_f1_with_pr(scores, labels)
            rows.append({"seed": seed, "lambda_feat": lf, "lambda_enhance": le,
                         "F1": f1, "Precision": p, "Recall": r})
            print(f"    lf={lf:<5} le={le:<5} F1 {f1:.4f}  P {p:.4f}  R {r:.4f}", flush=True)
            del refiner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# =============================================================================
# exp8 -- edge deployment profiling
# =============================================================================

EDGE_BATCH_SIZES = [1, 32, 256]
# DDIM stride. T=1 is the single-pass limit; the gap between T=50 and T=5 is the
# evidence for (or against) the step-distillation direction the paper proposes.
SAMPLE_STEP_GRID = [50, 10, 5, 1]


def _bench(fn, iters, warmup=10, cuda=False):
    """Median + p95 latency in ms. Median resists the occasional scheduler spike."""
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    s = np.asarray(samples)
    return float(np.median(s)), float(np.percentile(s, 95))


def _model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2


def _int8_dynamic(model: nn.Module):
    """Dynamic INT8. Only quantises Linear/LSTM, so a Conv1d U-Net gains little --
    reported honestly rather than presented as the acceleration story."""
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def exp8_edge_profiling(seed):
    """Latency / memory / model size across deployment targets.

    Targets, weakest first:
      cpu-1thread   -- proxy for a low-power ARM gateway or a PLC co-processor
      cpu-Nthread   -- a modest multi-core industrial PC
      cuda-fp32/16  -- the server used in the paper

    batch=1 is the number that matters for a streaming edge deployment; the
    paper's batch=256 figure is a throughput measurement and must not be read as
    per-sample latency.
    """
    set_seed(seed)
    feat_dim = 38  # SMD; latency depends on tensor shape, not on data content
    n_cpu = os.cpu_count() or 1

    det_cpu = FrozenDetector(KAN_AD_Lite(CFG.window_size, feat_dim)).eval()
    ref_cpu = DiffusionRefiner(feat_dim, num_train_steps=CFG.diffusion_train_steps,
                               schedule=CFG.diffusion_schedule).eval()

    targets = [("cpu-1thread", "cpu", torch.float32, 1),
               (f"cpu-{n_cpu}thread", "cpu", torch.float32, n_cpu)]
    if torch.cuda.is_available():
        targets += [("cuda-fp32", "cuda", torch.float32, None),
                    ("cuda-fp16", "cuda", torch.float16, None)]

    rows = []
    for tname, device, dtype, threads in targets:
        if threads is not None:
            torch.set_num_threads(threads)
        det = FrozenDetector(KAN_AD_Lite(CFG.window_size, feat_dim)).to(device, dtype).eval()
        ref = DiffusionRefiner(feat_dim, num_train_steps=CFG.diffusion_train_steps,
                               num_sample_steps=CFG.diffusion_sample_steps,
                               schedule=CFG.diffusion_schedule).to(device, dtype).eval()

        for bs in EDGE_BATCH_SIZES:
            x = torch.randn(bs, CFG.window_size, feat_dim, device=device, dtype=dtype)
            mask = torch.ones_like(x)
            mask[:, 30:70, :] = 0.0        # a 40% outage, the paper's headline case
            cuda = device == "cuda"

            with torch.no_grad():
                t_det, _ = _bench(lambda: det(x), 50 if bs == 1 else 20, cuda=cuda)
                for steps in SAMPLE_STEP_GRID:
                    # Sampling cost is linear in the step count -- this is the
                    # O(T) term, and the only knob that moves the edge budget.
                    iters = max(3, int(60 / steps)) if device == "cpu" else 20
                    if device == "cpu" and bs == 256:
                        iters = 2
                    t_full, p95 = _bench(
                        lambda: det(ref(x, mask, num_steps=steps)), iters, warmup=2, cuda=cuda)
                    rows.append({
                        "seed": seed, "target": tname, "batch": bs, "sample_steps": steps,
                        "detector_only_ms": t_det, "tadr_pipeline_ms": t_full,
                        "refiner_overhead_ms": t_full - t_det,
                        "ms_per_window": t_full / bs, "p95_ms": p95,
                        "windows_per_sec": bs / (t_full / 1000),
                        "streams_at_1hz": bs / (t_full / 1000),
                    })
                    print(f"    {tname:<14} bs={bs:<4} T={steps:<3} "
                          f"pipeline {t_full:9.2f} ms ({t_full / bs:8.3f} ms/window)", flush=True)

        del det, ref
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.set_num_threads(n_cpu)

    # Footprint: what has to fit in the device's flash and RAM.
    ref_int8 = _int8_dynamic(ref_cpu)
    x1 = torch.randn(1, CFG.window_size, feat_dim)
    m1 = torch.ones_like(x1); m1[:, 30:70, :] = 0.0
    torch.set_num_threads(1)
    with torch.no_grad():
        t_fp32, _ = _bench(lambda: det_cpu(ref_cpu(x1, m1, num_steps=10)), 20, warmup=2)
        t_int8, _ = _bench(lambda: det_cpu(ref_int8(x1, m1, num_steps=10)), 20, warmup=2)
    torch.set_num_threads(n_cpu)

    rows.append({
        "seed": seed, "target": "cpu-1thread-int8dynamic", "batch": 1, "sample_steps": 10,
        "detector_only_ms": float("nan"), "tadr_pipeline_ms": t_int8,
        "refiner_overhead_ms": float("nan"), "ms_per_window": t_int8,
        "p95_ms": float("nan"), "windows_per_sec": 1000 / t_int8,
        "streams_at_1hz": 1000 / t_int8,
    })
    print(f"\n    refiner params        : {sum(p.numel() for p in ref_cpu.parameters()):,}")
    print(f"    refiner size (FP32)   : {_model_size_mb(ref_cpu):.2f} MB")
    print(f"    detector size (FP32)  : {_model_size_mb(det_cpu.detector):.2f} MB")
    print(f"    cpu-1thread bs=1 T=10 : FP32 {t_fp32:.2f} ms   INT8-dynamic {t_int8:.2f} ms")
    print("    (INT8 dynamic covers Linear only; a Conv1d U-Net needs static or "
          "QAT quantisation to benefit -- state this rather than claiming the win.)",
          flush=True)
    return pd.DataFrame(rows)


# =============================================================================
# exp9 -- redundancy ablation
# =============================================================================

CHANNEL_GRID = [38, 24, 12, 6, 3, 1]
STRIDE_GRID = [1, 5, 25, 50, 100]


def _train_trio(train_ds, det, feat_dim, train_inj, gate=True):
    """Raw / CSDI (MSE-only, same capacity) / TADR -- the three-way contrast.

    `gate=False` is required by exp9, which manipulates the channel set on
    purpose: the gating indices are defined against the full 38-channel SMD
    layout, so applying them to a subsampled window would silently gate some
    settings and not others and confound the very axis being measured.
    """
    ds_name = "SMD" if gate else None
    csdi = train_baseline_refiner(
        build_refiner(CSDI_Lite, feat_dim, ds_name), train_ds, train_inj, CFG.refiner_epochs)
    tadr = train_tadr(build_refiner(None, feat_dim, ds_name), det, train_ds,
                      train_inj, CFG.refiner_epochs)
    return {"Raw": None, "CSDI": csdi, "TADR": tadr}


def exp9_redundancy(seed):
    """Two independent redundancy knobs, 40% block missing, SMD 1-1.

    (a) channels : fewer sensors -> less spatial redundancy. 1 channel is the
        degenerate case Reviewer 1 asks about -- no cross-sensor information
        exists at all, so any gain there must come from temporal structure.
    (b) stride   : test stride 100 means non-overlapping windows, i.e. every
        timestep is scored exactly once. This removes the sliding-window
        averaging the paper credits for robustness.
    """
    rows = []
    train_inj = NoiseInjector("mixed", missing_ratio=0.2)
    test_inj = NoiseInjector("block_missing", missing_ratio=0.4)
    orig_stride = CFG.test_stride

    # ---- (a) spatial redundancy -------------------------------------------
    for n_ch in CHANNEL_GRID:
        CFG.test_stride = 1
        train_ds, te_win, labels, det, feat_dim = build_smd_fixture(seed, keep_channels=n_ch)
        _, te_raw, _ = load_and_scale_dataset("SMD", "machine-1-1")
        rng = np.random.default_rng(seed)
        cols = np.sort(rng.choice(te_raw.shape[1], size=min(n_ch, te_raw.shape[1]), replace=False))
        corr = mean_abs_channel_corr(te_raw[:, cols]) if n_ch > 1 else float("nan")

        for name, refiner in _train_trio(train_ds, det, feat_dim, train_inj,
                                         gate=False).items():
            scores = entity_scores(refiner, det, te_win, len(labels), test_inj, seed)
            f1, p, r = best_f1_with_pr(scores, labels)[:3]
            rows.append({"seed": seed, "knob": "channels", "setting": n_ch,
                         "mean_abs_corr": corr, "Method": name,
                         "F1": f1, "Precision": p, "Recall": r})
        got = {d["Method"]: d["F1"] for d in rows if d["knob"] == "channels" and d["setting"] == n_ch}
        print(f"    channels={n_ch:<3} corr={corr:.3f}  " +
              "  ".join(f"{k}:{v:.4f}" for k, v in got.items()), flush=True)
        del train_ds, te_win, det
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- (b) temporal redundancy ------------------------------------------
    CFG.test_stride = 1
    train_ds, _, labels, det, feat_dim = build_smd_fixture(seed)
    refiners = _train_trio(train_ds, det, feat_dim, train_inj, gate=False)
    _, te_sc, labels = load_and_scale_dataset("SMD", "machine-1-1")

    for stride in STRIDE_GRID:
        CFG.test_stride = stride
        te_win = create_sliding_windows(te_sc, CFG.window_size, stride)
        for name, refiner in refiners.items():
            scores = entity_scores(refiner, det, te_win, len(labels), test_inj, seed)
            f1, p, r = best_f1_with_pr(scores, labels)[:3]
            rows.append({"seed": seed, "knob": "test_stride", "setting": stride,
                         "mean_abs_corr": float("nan"), "Method": name,
                         "F1": f1, "Precision": p, "Recall": r})
        got = {d["Method"]: d["F1"] for d in rows
               if d["knob"] == "test_stride" and d["setting"] == stride}
        print(f"    stride={stride:<4} " + "  ".join(f"{k}:{v:.4f}" for k, v in got.items()),
              flush=True)
        del te_win

    CFG.test_stride = orig_stride
    return pd.DataFrame(rows)


# =============================================================================
# exp10 -- anomaly-coverage decomposition
# =============================================================================

COVERAGE_BINS = [0.0, 0.01, 0.25, 0.50, 0.75, 0.999, 1.01]
COVERAGE_NAMES = ["0% (visible)", "0-25%", "25-50%", "50-75%", "75-100%", "100% (hidden)"]


@torch.no_grad()
def _scores_and_coverage(refiner, det, test_windows, n_labels, injector, seed):
    """Per-timestep anomaly score AND the fraction of covering windows that
    masked that timestep. Coverage 1.0 means the timestep was never observed by
    any window -- the case Reviewer 2 argues is unrecoverable in principle."""
    if refiner is not None:
        refiner.eval()
    det.eval()

    score_sum = np.zeros(n_labels)
    masked_sum = np.zeros(n_labels)
    count = np.zeros(n_labels)
    idx = 0
    W = CFG.window_size

    for (x,) in DataLoader(TensorDataset(test_windows), batch_size=CFG.eval_batch_size):
        x = x.to(CFG.device)
        x_noisy, mask = injector(x, seed=seed + idx)
        x_ref = x_noisy if refiner is None else refiner(x_noisy, mask)
        score = det(x_ref)[1].cpu().numpy()
        # a timestep is "dropped" in this window if every channel was masked out
        dropped = (mask.mean(dim=-1) < 0.5).float().cpu().numpy()
        for i in range(x.shape[0]):
            s = idx * CFG.test_stride
            e = min(s + W, n_labels)
            score_sum[s:e] += score[i, :e - s]
            masked_sum[s:e] += dropped[i, :e - s]
            count[s:e] += 1
            idx += 1

    count = np.maximum(count, 1.0)
    return score_sum / count, masked_sum / count


def exp10_coverage_decomposition(seed):
    """Where the gain comes from, bucketed by how hidden each anomaly was.

    Reviewer 2's hypothesis predicts TADR's advantage lives in the fully-hidden
    bucket (it would have to be inferring the anomaly from precursors). The
    alternative -- that TADR simply avoids erasing partially-visible anomalies
    and avoids fabricating artefacts on normal data -- predicts the advantage
    lives in the low-coverage buckets and in precision on normal timesteps.
    """
    train_ds, te_win, labels, det, feat_dim = build_smd_fixture(seed)
    train_inj = NoiseInjector("mixed", missing_ratio=0.2)
    test_inj = NoiseInjector("block_missing", missing_ratio=0.4)
    refiners = _train_trio(train_ds, det, feat_dim, train_inj)

    labels = np.asarray(labels).astype(int)
    rows = []
    per_method = {}

    for name, refiner in refiners.items():
        scores, coverage = _scores_and_coverage(refiner, det, te_win, len(labels),
                                                test_inj, seed)
        f1, p, r, thr = best_f1_with_pr(scores, labels)
        per_method[name] = (scores, coverage, thr)
        rows.append({"seed": seed, "Method": name, "bucket": "ALL",
                     "n_points": int(labels.sum()), "detect_rate": r,
                     "mean_score": float(scores[labels == 1].mean()),
                     "F1": f1, "Precision": p, "Recall": r})

    # Bucket anomalous timesteps by how much of them the outage hid.
    # Coverage is computed from the Raw pass; the corruption is seeded, so every
    # method saw exactly the same mask and the buckets are directly comparable.
    _, ref_coverage, _ = per_method["Raw"]
    anom = labels == 1
    bins = np.digitize(ref_coverage[anom], COVERAGE_BINS) - 1
    bins = np.clip(bins, 0, len(COVERAGE_NAMES) - 1)

    for name, (scores, _, thr) in per_method.items():
        s_anom = scores[anom]
        for b, bname in enumerate(COVERAGE_NAMES):
            sel = bins == b
            n = int(sel.sum())
            if n == 0:
                continue
            rows.append({"seed": seed, "Method": name, "bucket": bname,
                         "n_points": n,
                         "detect_rate": float((s_anom[sel] > thr).mean()),
                         "mean_score": float(s_anom[sel].mean()),
                         "F1": float("nan"), "Precision": float("nan"),
                         "Recall": float("nan")})
        # False-alarm rate on normal timesteps: the other half of the story.
        rows.append({"seed": seed, "Method": name, "bucket": "NORMAL (false alarms)",
                     "n_points": int((~anom).sum()),
                     "detect_rate": float((scores[~anom] > thr).mean()),
                     "mean_score": float(scores[~anom].mean()),
                     "F1": float("nan"), "Precision": float("nan"),
                     "Recall": float("nan")})

    df = pd.DataFrame(rows)
    print(df[df.bucket != "ALL"].pivot_table(index="bucket", columns="Method",
                                             values="detect_rate")
          .to_string(float_format=lambda v: f"{v:.4f}"), flush=True)
    return df


# T <= 5 is retained deliberately: one-step DDIM from pure noise divides by
# sqrt(abar) ~ 0.006, so it cannot recover x0 in principle. Keeping those rows,
# with the fidelity columns beside them, is what shows the failure rather than
# hiding it -- an earlier version of this experiment reported T=1 as a headline
# result on F1 alone, when 63% of its output was pinned to the clipping bound.
SAMPLING_STEPS = [50, 25, 10, 5, 2, 1]


def exp11_sampling_steps(seed):
    """F1 and wall-clock as a function of the DDIM step count.

    One model per objective is trained at the full schedule (T_train = 50) and
    then sampled at progressively coarser strides. Nothing is retrained, so any
    change is attributable to the sampling budget alone.

    The paper proposes consistency/step-distillation to cut O(T) to O(1). This
    measures how much of that gap plain DDIM striding already closes -- and
    whether the O(1) limit (T = 1) is actually usable or merely fast.
    """
    train_ds, te_win, labels, det, feat_dim = build_smd_fixture(seed)
    train_inj = NoiseInjector("mixed", missing_ratio=0.2)
    test_inj = NoiseInjector("block_missing", missing_ratio=0.4)

    rows = []
    scores = entity_scores(None, det, te_win, len(labels), test_inj, seed)
    f1, p, r, _ = best_f1_with_pr(scores, labels)
    rows.append({"seed": seed, "Method": "Raw", "steps": 0,
                 "F1": f1, "Precision": p, "Recall": r, "eval_sec": 0.0})

    models = {
        "CSDI": train_baseline_refiner(
            make_diffusion_refiner(feat_dim), train_ds, train_inj, CFG.refiner_epochs),
        "TADR": train_tadr(
            make_diffusion_refiner(feat_dim), det, train_ds, train_inj, CFG.refiner_epochs),
    }

    for name, refiner in models.items():
        refiner.num_sample_steps = CFG.diffusion_train_steps
        for steps in SAMPLING_STEPS:
            refiner.num_sample_steps = steps
            t0 = time.time()
            scores = entity_scores(refiner, det, te_win, len(labels), test_inj, seed)
            elapsed = time.time() - t0
            f1, p, r, _ = best_f1_with_pr(scores, labels)
            row = {"seed": seed, "Method": name, "steps": steps,
                   "F1": f1, "Precision": p, "Recall": r, "eval_sec": elapsed}
            row.update(refinement_diagnostics(refiner, det, te_win, test_inj, seed))
            rows.append(row)
            print(f"    {name:<5} T={steps:<3} F1 {f1:.4f} P {p:.4f} R {r:.4f} | "
                  f"std {row['out_std']:6.3f} sat {row['sat_pct']:5.2f}% "
                  f"obs_mse {row['obs_mse']:7.3f}  ({elapsed:5.1f}s)", flush=True)
        del refiner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


EXPERIMENTS = {
    "11": ("Sampling-step budget (DDIM stride vs F1)", exp11_sampling_steps,
           ["Method", "steps"]),
    "7": ("Hyperparameter sensitivity (lambda_feat x lambda_enhance)",
          exp7_lambda_sensitivity, ["lambda_feat", "lambda_enhance"]),
    "8": ("Edge deployment profiling", exp8_edge_profiling,
          ["target", "batch", "sample_steps"]),
    "9": ("Redundancy ablation (channels, window overlap)",
          exp9_redundancy, ["knob", "setting", "Method"]),
    "10": ("Anomaly-coverage decomposition", exp10_coverage_decomposition,
           ["Method", "bucket"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="all")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-list", default=None,
                    help="explicit comma-separated seeds; overrides --seed/--runs")
    ap.add_argument("--data-dir", default=CFG.base_dir)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=CFG.device)
    args = ap.parse_args()

    CFG.base_dir = args.data_dir
    CFG.device = args.device

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "results_rebuttal")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(CFG), f, indent=2)

    ids = list(EXPERIMENTS) if args.exp == "all" else [e.strip() for e in args.exp.split(",")]
    unknown = [e for e in ids if e not in EXPERIMENTS]
    if unknown:
        ap.error(f"unknown experiment id(s): {unknown}; choose from {list(EXPERIMENTS)}")

    seeds = ([int(x) for x in args.seed_list.split(",")] if args.seed_list
             else list(range(args.seed, args.seed + args.runs)))
    print(f"device={CFG.device}  seeds={seeds}  experiments={ids}  out={out_dir}")

    for e in ids:
        title, fn, keys = EXPERIMENTS[e]
        print(f"\n{'=' * 78}\n[exp{e}] {title}\n{'=' * 78}", flush=True)
        raw = pd.concat([fn(s) for s in seeds], ignore_index=True)
        raw.to_csv(os.path.join(out_dir, f"exp{e}_runs.csv"), index=False)
        agg = aggregate(raw, keys)
        agg.to_csv(os.path.join(out_dir, f"exp{e}.csv"), index=False)
        print(f"\n[exp{e}] aggregated over {len(seeds)} run(s)")
        print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nDone. CSVs written to {out_dir}")


if __name__ == "__main__":
    main()
