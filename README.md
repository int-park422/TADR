# TADR: Task-Aware Diffusion Refiner

Code and per-seed results for

> **Don't Just Impute, Detect: Task-Aware Diffusion Pipelines for Edge Anomaly Detection**
> Jeongsu Park, Taehyeon Yun, Geonwoo Kim, Yuji Lim, Moohong Min.
> *IEEE International Conference on High Performance Computing, Data, and Analytics (HiPC)*, 2026.

When IIoT sensor streams lose contiguous blocks of data, the usual pipeline imputes
the gap with an MSE-trained model and then runs an anomaly detector on the result.
The imputer fills the gap with the most probable *normal* pattern, which smooths
partially observed anomalies away and leaves boundary artifacts that raise false
alarms. TADR is a conditional diffusion refiner trained against the loss of the
frozen downstream detector instead. The task-aware terms act on the one-step
estimate of the clean window that the noise prediction already yields, so training
costs no more than plain diffusion training.

## Repository layout

```
code/
  diffusion.py              conditional DDPM refiner: noise schedule, time-conditioned
                            1D U-Net, DDIM sampling
  tadr_experiments.py       datasets, corruption model, detectors, losses, P2P
                            evaluation, experiments 1-6
  rebuttal_experiments.py   experiments 7-11
  run_all.sh                full 5-seed suite, one process per (experiment, seed)
  merge_results.py          merges per-seed shards, computes CIs and paired t-tests
  finalize.py               writes REPORT.md / RESULTS.xlsx, runs the fidelity check
  make_figures.py           draws Figs. 3-6 from the aggregated CSVs
figures/                    Figs. 3-6, as drawn from results/
results/
  REPORT.md                 every table with 95% CIs, paired tests, fidelity check
  RESULTS.xlsx              one sheet per experiment
  exp<N>.csv                aggregated over seeds (mean, std, ci95)
  exp<N>_paired.csv         paired t-tests on per-seed differences
  exp<N>_runs.csv           one row per seed
  per_seed/                 the raw shards, so every interval can be recomputed
  logs/                     stdout of every run
  config.json               the configuration every number was produced under
```

## Setup

Python 3.12 with a CUDA build of PyTorch (we used PyTorch 2.5.1, CUDA 12.1):

```bash
pip install -r requirements.txt
```

## Data

The benchmarks are not redistributed here. Download them from their original
sources and arrange them as follows:

```
$TADR_DATA_DIR/
  ServerMachineDataset/          SMD   https://github.com/NetManAIOps/OmniAnomaly
    train/  test/  test_label/   machine-*.txt
  telemanom/                     SMAP, MSL   https://github.com/khundman/telemanom
    labeled_anomalies.csv
    train/  test/                *.npy   (a nested data/train, data/test also works)
  PSM/                           PSM   https://github.com/eBay/RANSynCoders
    train.csv  test.csv  test_label.csv
```

Point the code at that directory with the `TADR_DATA_DIR` environment variable or
`--data-dir` (default: `./data`).

## Reproducing the paper

Everything, five seeds, spread across all visible GPUs:

```bash
cd code
TADR_DATA_DIR=/path/to/data ./run_all.sh 3     # 3 concurrent jobs per GPU
```

Re-running skips shards that already finished, so the same command resumes an
interrupted run. On two RTX 3090s the full suite takes roughly a day, almost all
of it in the pooled experiments (2-4). Output goes to `code/results_final/`, in
the same format as `results/`.

A single experiment and seed:

```bash
python3 tadr_experiments.py     --exp 1 --seed-list 0 --data-dir /path/to/data --out out/exp1_seed0
python3 rebuttal_experiments.py --exp 9 --seed-list 0 --data-dir /path/to/data --out out/exp9_seed0
```

`--fast` restricts the pooled experiments to a few entities for a quick check.

The figures are drawn from the aggregated CSVs, so they can be regenerated from
the published results without rerunning anything:

```bash
python3 make_figures.py --results ../results --out ../figures
```

Point `--results` at `code/results_final` to draw them from your own run.

### Where each result in the paper comes from

| Paper | Experiment | Content |
|---|---|---|
| Fig. 4 | exp1 | Refiner comparison, SMD machine-1-1, 40% block missing |
| Table I | exp10 | Detection rate by masked fraction |
| Fig. 5 | exp3 | Detector generality (KAN-AD, TimesNet, Anomaly Transformer), pooled SMD |
| Table II | exp9 | Redundancy ablation: channels retained, test stride |
| Fig. 3 | exp2 | Four pooled datasets, 40% block missing |
| Sec. V-E | exp4 | Pooled datasets, 10-50% block missing |
| Table III | exp5 | Loss-component ablation, single-pass refiner |
| Fig. 6 | exp7 | λ<sub>feat</sub> × λ<sub>enhance</sub> grid |
| Table IV | exp6, exp8 | Batch latency on GPU; streaming latency on one CPU thread |
| Table V | exp11 | Sampling-step budget with reconstruction fidelity |

## Reading the results

- **Metric.** Point-to-Point (P2P) F1, precision and recall at the best threshold,
  with no point adjustment. Every number is a mean over five seeds with a 95%
  confidence interval (`_ci95` columns are half-widths); comparisons use paired
  t-tests on per-seed differences.
- **Read F1 together with fidelity.** Best-threshold F1 alone cannot tell genuine
  refinement from a degenerate refiner: in the sampling-step sweep, a refiner whose
  output was pinned to the clipping bound on half of all samples still scored 0.69.
  Every table therefore carries `out_std`, `sat_pct` (share of output at the
  clipping bound), `obs_mse` and `miss_mse`, and `REPORT.md` lists every row that
  fails this check.
- **Latency.** Run the latency experiments on an otherwise idle GPU. The published
  `exp6` was measured that way; an earlier measurement taken while other jobs
  shared the GPU came out about 2.5 times slower.

## Scope

The gain tracks the continuous, mutually redundant sensor signal available. It is
large on SMD (33 of 38 continuous channels), and absent on SMAP (1 of 25) and MSL
(1 of 55), whose remaining channels are one-hot command signals, and on PSM, where
the unrefined detector already reaches 0.88 recall. Across detectors it helps
reconstruction-error detectors (KAN-AD, TimesNet) and not Anomaly Transformer.

## Citation

```bibtex
@inproceedings{park2026tadr,
  title     = {Don't Just Impute, Detect: Task-Aware Diffusion Pipelines for Edge Anomaly Detection},
  author    = {Park, Jeongsu and Yun, Taehyeon and Kim, Geonwoo and Lim, Yuji and Min, Moohong},
  booktitle = {IEEE International Conference on High Performance Computing, Data, and Analytics (HiPC)},
  year      = {2026}
}
```

## Acknowledgment

This research was supported by the Development of Field Technology for Responding
to Illegal Drugs Program through the Korea Institutes of Police Technology (KIPoT),
funded by the Ministry of Science and ICT and the Korean National Police Agency
(RS-2026-25535833), and by the Korea Institute of Science and Technology
Information (KISTI) (P26035, K26L8M1C1).

## License

MIT; see [LICENSE](LICENSE).
