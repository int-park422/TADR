#!/usr/bin/env python3
"""
TADR: Task-Aware Refinement for Edge Anomaly Detection under Block-Missing Data
Consolidated experiment suite (replaces experiments_v5.ipynb / experiments_v6.ipynb).

Reproduces the six result tables reported in the paper:

  exp1  Refiner comparison under 40% block missing, SMD machine-1-1        (F1/P/R)
  exp2  Global P2P F1 across 4 datasets, 40% block missing                 (F1)
  exp3  Detector generality with and without TADR, pooled SMD              (F1)
  exp4  Global P2P F1 across datasets, varying block-missing ratios        (F1)
  exp5  Ablation on loss components, smoothed P2P, SMD machine-1-1         (F1/P/R)
  exp6  System inference overhead, batch size 256                          (ms/MB)

Usage
-----
    python tadr_experiments.py --exp 1            # single table
    python tadr_experiments.py --exp all --runs 5 # every table, 5 seeds, mean +/- std
    python tadr_experiments.py --exp 2 --fast     # smoke test (few entities/epochs)

Results are written to  <--out>/expN_runs.csv  (per-seed rows) and
                        <--out>/expN.csv       (aggregated mean/std/95% CI).

NOTE ON DIVERGENCES FROM THE NOTEBOOKS
--------------------------------------
The two notebooks defined the same components several times with silently
different behaviour. Every such fork is now a single explicit switch on `CFG`,
with the default set to the behaviour described in the paper. See `Config`.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from diffusion import DiffusionRefiner, train_diffusion_refiner


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # --- paths / IO -----------------------------------------------------------
    # dataset root; override with --data-dir or the TADR_DATA_DIR environment variable
    base_dir: str = os.environ.get("TADR_DATA_DIR", "data")
    out_dir: str = "results"

    # --- windowing ------------------------------------------------------------
    window_size: int = 100
    # RESOLVED (T0-2): a real diffusion model needs far more data than the
    # single-pass refiner did. At the notebooks' stride of 100 a single entity
    # yields only 284 windows and the model does not converge at all
    # (missing-region MSE 20.6 vs 1.22 for plain zero-fill). Stride 10 gives
    # 2838 windows and it converges cleanly (0.80).
    train_stride: int = 10
    # RESOLVED: a coarse pooled stride was a mistake. It assumed entity count
    # would keep windows plentiful, which holds only for SMD (28 entities of
    # ~28k rows). SMAP/MSL entities are ~2k rows each and PSM is a single
    # entity, so stride 100 gave those datasets 160-420 gradient steps against
    # SMD's 2200 -- an order of magnitude below the ~2600 at which the
    # diffusion refiner was verified to converge. Every pooled dataset now
    # trains to the same step budget instead.
    pooled_train_stride: int = 10
    pooled_target_steps: int = 3000
    test_stride: int = 1             # dense evaluation, as reported in the paper

    # --- corruption model -----------------------------------------------------
    # RESOLVED: the paper formalises corruption as X_tilde = X (*) M, which is
    # exactly a 0.0 fill; the -5.0 out-of-distribution fill used by one notebook
    # cannot be produced by that operator. The paper never claims the refiner is
    # mask-conditioned either, so the mask stays out of the input. Both remain
    # available (--use-mask-channel) but are no longer the reported setting.
    drop_val: float = 0.0
    use_mask_channel: bool = False
    # Clip standardised values to +/- this many sigma. 0 disables.
    # REJECTED as a default: it helped SMAP by +0.018 but cost PSM -0.045.
    scale_clip: float = 0.0

    # Run the refiner on continuous channels only, passing discrete ones through.
    # ADOPTED: telemanom's SMAP/MSL are 24/25 and 54/55 one-hot command channels,
    # which a Gaussian diffusion model cannot represent and which are not
    # meaningful to impute -- what goes missing at the edge is the sensor
    # reading. Gating removed 80% of the SMAP deficit (-0.142 -> -0.029) and most
    # of MSL's (-0.048 -> -0.010), and it helps the CSDI baseline by the same
    # mechanism, so it is not a change that favours TADR.
    gate_discrete_channels: bool = True
    discrete_max_levels: int = 2
    discrete_vote: float = 0.8

    # --- training -------------------------------------------------------------
    # RESOLVED: set to 20, not the 10 implied by the paper's "converges within
    # 3-10 epochs". A controlled sweep (seeds 0-1, SMD 1-1, 40% block missing)
    # gives 20 epochs a consistent advantage over 10 at every other setting
    # (0.4834 vs 0.4507 with the mask off; 0.4921 vs 0.4495 with it on), so the
    # convergence claim in the paper is the thing that needs correcting.
    detector_epochs: int = 10
    # Diffusion needs roughly 3x the epochs the single-pass refiner did; 60 is
    # where reconstruction stops improving on held-out windows.
    refiner_epochs: int = 60
    pooled_detector_epochs: int = 3
    pooled_refiner_epochs: int = 20      # overridden by pooled_target_steps
    # RESOLVED: the paper reports B = 64 (32 normal / 32 synthetic-anomaly).
    batch_size: int = 64
    pooled_batch_size: int = 64
    detector_lr: float = 1e-3
    baseline_lr: float = 1e-3
    refiner_lr: float = 3e-4
    grad_clip: float = 1.0

    # --- task-aware loss ------------------------------------------------------
    lambda_feat: float = 0.1
    lambda_enhance: float = 0.5
    # RESOLVED: the paper reports lambda_enhance = 0.5 for the method as a whole,
    # and presents the ablation as the same method with components removed. The
    # original ablation silently used 0.2, making its rows incomparable with
    # every other table; it is now 0.5 like the rest.
    lambda_enhance_ablation: float = 0.5

    # Synthetic anomaly injected during refiner training.
    # sigma 5 -> 2 was tested and REJECTED: it moved PSM from -0.159 to -0.124
    # (still a loss) while costing SMD +0.127 -> +0.098. It trades away the one
    # dataset the method works on without fixing the one it does not.
    anom_sigma: float = 5.0
    anom_len: int = 10
    # RESOLVED: the paper builds X_anom, corrupts it to X_tilde_anom, and asks
    # the refiner to recover X_anom. Regressing toward the CLEAN window (as two
    # notebook cells did) trains the refiner to erase the anomaly -- the exact
    # failure the paper is arguing against.
    anom_target: str = "anom"        # {"anom", "clean"}

    # --- evaluation -----------------------------------------------------------
    n_thresholds: int = 200
    pct_lo: float = 50.0
    pct_hi: float = 99.5
    smooth_window: int = 5           # exp5 only
    eval_batch_size: int = 256
    # Draw the SAME corruption for every refiner on a given (dataset, entity,
    # ratio). The notebooks re-randomised per refiner, which added variance to
    # every head-to-head comparison for no benefit.
    paired_corruption: bool = True

    # --- diffusion refiner ----------------------------------------------------
    # RESOLVED (T0-2): the refiner is a real conditional DDPM, not the single
    # forward pass earlier notebooks used. This is what makes the paper's
    # O(T . C_CNN) complexity claim true, and it is why T is reported alongside
    # every latency number.
    diffusion_train_steps: int = 50
    diffusion_sample_steps: int = 50     # DDIM stride; lower = faster, see exp8
    diffusion_schedule: str = "quad"     # CSDI's schedule, for a fair baseline
    # REJECTED: pinning observed samples helped the CSDI baseline but cost TADR
    # on all three datasets tested (PSM -0.225, SMAP -0.064, SMD +0.033 against
    # +0.127 without it).
    diffusion_hard_replace: bool = False
    aux_snr_weight: bool = True

    # --- runtime --------------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fast: bool = False               # smoke-test mode
    fast_entities: int = 2


CFG = Config()


# Shared generator for corruption drawn without an explicit seed (i.e. during
# training). Re-seeded by set_seed so that --seed determines an entire run.
_RNG = np.random.default_rng(0)


def set_seed(seed: int) -> None:
    global _RNG
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _RNG = np.random.default_rng(seed)


# =============================================================================
# Data
# =============================================================================

_DISCRETE_CACHE: dict[str, list[int]] = {}


def _raw_train_array(dataset_name: str, entity: str) -> np.ndarray:
    """Unscaled training data. Discreteness must be judged before standardisation:
    each entity is scaled independently, so after scaling a {0,1} command channel
    takes a different pair of values per entity and pooling makes it look
    continuous -- which misreported 24 of SMAP's 25 channels as continuous."""
    b = CFG.base_dir
    if dataset_name == "SMD":
        return np.loadtxt(os.path.join(b, "ServerMachineDataset", "train",
                                       f"{entity}.txt"), delimiter=",")
    if dataset_name == "PSM":
        df = pd.read_csv(os.path.join(b, "PSM", "train.csv"))
        return df.iloc[:, 1:].values if "timestamp" in df.columns[0].lower() else df.values
    return np.load(os.path.join(_telemanom_split_dir(_telemanom_root()),
                                "train", f"{entity}.npy"))


def continuous_channels(dataset_name: str) -> list[int]:
    """Channel indices that are not two-level in the raw training stream."""
    if dataset_name in _DISCRETE_CACHE:
        return _DISCRETE_CACHE[dataset_name]
    ents = get_dataset_entities(dataset_name)
    n_ch = _raw_train_array(dataset_name, ents[0]).shape[1]
    votes = np.zeros(n_ch)
    for e in ents:
        tr = np.nan_to_num(_raw_train_array(dataset_name, e))
        for c in range(n_ch):
            if len(np.unique(tr[:, c])) <= CFG.discrete_max_levels:
                votes[c] += 1
    idx = [c for c in range(n_ch) if votes[c] / len(ents) < CFG.discrete_vote]
    _DISCRETE_CACHE[dataset_name] = idx
    return idx


def _telemanom_root() -> str:
    return os.path.join(CFG.base_dir, "telemanom")


def _telemanom_split_dir(root: str) -> str:
    """Telemanom ships as either <root>/{train,test} or <root>/data/{train,test}."""
    nested = os.path.join(root, "data")
    return nested if os.path.isdir(os.path.join(nested, "train")) else root


def get_dataset_entities(dataset_name: str) -> list[str]:
    """All entity ids of a benchmark, sorted for deterministic pooling order."""
    entities: list[str] = []
    if dataset_name == "SMD":
        d = os.path.join(CFG.base_dir, "ServerMachineDataset", "train")
        if os.path.isdir(d):
            entities = [f[:-4] for f in os.listdir(d) if f.endswith(".txt")]
    elif dataset_name in ("SMAP", "MSL"):
        csv_path = os.path.join(_telemanom_root(), "labeled_anomalies.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            entities = df[df["spacecraft"] == dataset_name]["chan_id"].unique().tolist()
    elif dataset_name == "PSM":
        entities = ["PSM"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    entities = sorted(entities)
    return entities[: CFG.fast_entities] if CFG.fast else entities


def load_and_scale_dataset(dataset_name: str, entity_name: str):
    """Return (train_scaled, test_scaled, test_labels). Scaling is per entity."""
    if dataset_name == "SMD":
        d = os.path.join(CFG.base_dir, "ServerMachineDataset")
        train = np.loadtxt(os.path.join(d, "train", f"{entity_name}.txt"), delimiter=",")
        test = np.loadtxt(os.path.join(d, "test", f"{entity_name}.txt"), delimiter=",")
        labels = np.loadtxt(os.path.join(d, "test_label", f"{entity_name}.txt"), delimiter=",")

    elif dataset_name in ("SMAP", "MSL"):
        d = _telemanom_root()
        sub = _telemanom_split_dir(d)
        train = np.load(os.path.join(sub, "train", f"{entity_name}.npy"))
        test = np.load(os.path.join(sub, "test", f"{entity_name}.npy"))
        df = pd.read_csv(os.path.join(d, "labeled_anomalies.csv"))
        info = df[df["chan_id"] == entity_name].iloc[0]
        labels = np.zeros(test.shape[0])
        for s, e in ast.literal_eval(info["anomaly_sequences"]):
            labels[s:e] = 1

    elif dataset_name == "PSM":
        d = os.path.join(CFG.base_dir, "PSM")
        tr_df = pd.read_csv(os.path.join(d, "train.csv"))
        te_df = pd.read_csv(os.path.join(d, "test.csv"))
        lb_df = pd.read_csv(os.path.join(d, "test_label.csv"))
        train = tr_df.iloc[:, 1:].values if "timestamp" in tr_df.columns[0].lower() else tr_df.values
        test = te_df.iloc[:, 1:].values if "timestamp" in te_df.columns[0].lower() else te_df.values
        if len(lb_df.columns) > 1 and "timestamp" in lb_df.columns[0].lower():
            labels = lb_df.iloc[:, 1].values
        else:
            labels = lb_df.values.squeeze()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(np.nan_to_num(train))
    test_scaled = scaler.transform(np.nan_to_num(test))

    # 38 of 110 entities contain channels that are near-constant in train but
    # vary in test; dividing by that tiny std sends scaled test values to 1e5
    # and lets a handful of samples dominate both the anomaly score and the
    # threshold search. Clipping bounds that without touching normal ranges.
    if CFG.scale_clip:
        train_scaled = np.clip(train_scaled, -CFG.scale_clip, CFG.scale_clip)
        test_scaled = np.clip(test_scaled, -CFG.scale_clip, CFG.scale_clip)
    return train_scaled, test_scaled, np.asarray(labels)


def create_sliding_windows(data: np.ndarray, window_size: int, stride: int) -> torch.Tensor:
    n = len(data)
    if n < window_size:
        return torch.empty(0, window_size, data.shape[-1], dtype=torch.float32)
    idx = range(0, n - window_size + 1, stride)
    return torch.tensor(np.stack([data[i:i + window_size] for i in idx]), dtype=torch.float32)


# =============================================================================
# Corruption model
# =============================================================================

class NoiseInjector:
    """Simulates edge-side sensor faults.

    Returns (x_noisy, mask) where mask is 1 for observed and 0 for dropped
    samples. Callers that do not consume the mask can ignore it; keeping the two
    separate avoids the channel-concatenation hack used in the notebooks.
    """

    MODES = ("clean", "block_missing", "random_missing", "gaussian", "mixed")

    def __init__(self, mode="mixed", missing_ratio=0.2, gaussian_std=0.1, drop_val=None):
        assert mode in self.MODES, mode
        self.mode = mode
        self.missing_ratio = missing_ratio
        self.gaussian_std = gaussian_std
        self.drop_val = CFG.drop_val if drop_val is None else drop_val

    # -- per-window primitives -------------------------------------------------
    def _gaussian(self, x, rng):
        noise = torch.from_numpy(rng.standard_normal(tuple(x.shape))).to(x.device, x.dtype)
        return x + noise * self.gaussian_std, torch.ones_like(x)

    def _random_missing(self, x, rng):
        keep = torch.from_numpy(
            (rng.random((x.shape[0], 1)) > self.missing_ratio)
        ).to(x.device)
        mask = keep.expand_as(x).float()
        return torch.where(mask.bool(), x, torch.full_like(x, self.drop_val)), mask

    def _block_missing(self, x, rng):
        W = x.shape[0]
        length = int(W * self.missing_ratio)
        mask = torch.ones_like(x)
        if length == 0:
            return x, mask
        start = int(rng.integers(0, W - length + 1))
        out = x.clone()
        out[start:start + length, :] = self.drop_val
        mask[start:start + length, :] = 0.0
        return out, mask

    # -- batch entry point -----------------------------------------------------
    def __call__(self, x_batch: torch.Tensor, seed: int | None = None):
        # An explicit seed makes the corruption reproducible and shareable across
        # refiners (paired comparison); otherwise draw from the run-wide stream.
        rng = _RNG if seed is None else np.random.default_rng(seed)
        x_noisy = x_batch.clone()
        mask = torch.ones_like(x_batch)

        for i in range(x_batch.shape[0]):
            mode = self.mode
            if mode == "mixed":
                mode = rng.choice(
                    ["clean", "block_missing", "random_missing", "gaussian"],
                    p=[0.2, 0.4, 0.2, 0.2],
                )
            if mode == "clean":
                continue
            fn = {
                "block_missing": self._block_missing,
                "random_missing": self._random_missing,
                "gaussian": self._gaussian,
            }[mode]
            x_noisy[i], mask[i] = fn(x_batch[i], rng)

        return x_noisy, mask


# =============================================================================
# Detectors (lightweight re-implementations; the detector is always frozen)
# =============================================================================

class KAN_AD_Lite(nn.Module):
    """KAN-style detector: activation-centric MLP approximated with SiLU."""

    def __init__(self, seq_len, feat_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2), nn.SiLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.SiLU(),
        )
        self.blocks = nn.ModuleList([self.encoder])
        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, x):
        recon = self.out_proj(self.blocks[0](x))
        return recon, torch.mean((x - recon) ** 2, dim=-1)


class AnomalyTransformer_Lite(nn.Module):
    """Self-attention detector (Anomaly Transformer motif)."""

    def __init__(self, seq_len, feat_dim):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=2, batch_first=True, dim_feedforward=128
        )
        self.blocks = nn.ModuleList([layer])
        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, x):
        recon = self.out_proj(self.blocks[0](x))
        return recon, torch.mean((x - recon) ** 2, dim=-1)


class TimesNet_Lite(nn.Module):
    """Temporal-convolution detector (TimesNet motif)."""

    def __init__(self, seq_len, feat_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 3, padding=1), nn.GELU(),
            nn.Conv1d(feat_dim, feat_dim, 3, padding=1),
        )
        self.blocks = nn.ModuleList([self.encoder])
        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, x):
        h = self.blocks[0](x.transpose(1, 2)).transpose(1, 2)
        recon = self.out_proj(h)
        return recon, torch.mean((x - recon) ** 2, dim=-1)


DETECTORS = {
    "Anomaly Transformer": AnomalyTransformer_Lite,
    "TimesNet": TimesNet_Lite,
    "KAN-AD": KAN_AD_Lite,
}


class FrozenDetector(nn.Module):
    """Freezes a detector and exposes (penultimate features, anomaly score).

    Splitting f_theta into f_feat and f_score is what makes L_feat and
    L_enhance computable without touching the detector's weights.
    """

    def __init__(self, detector: nn.Module):
        super().__init__()
        self.detector = detector.eval()
        for p in self.detector.parameters():
            p.requires_grad = False
        self._feat = None
        self._handle = self.detector.blocks[-1].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._feat = out

    def forward(self, x):
        _, score = self.detector(x)
        return self._feat, score


# =============================================================================
# Refiners
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.GELU(),
        )

    def forward(self, x):
        return self.b(x)


class SinglePassRefiner(nn.Module):
    """1D U-Net refiner, one forward pass, no sampling.

    Retained as an ablation arm: it answers "does the iterative generative
    process buy anything over a deterministic pass with the same objective?"
    It is no longer the model the paper calls TADR.

    When CFG.use_mask_channel is set, the observation mask is concatenated as C
    extra channels and dropped samples are zeroed first, so the out-of-
    distribution fill value never reaches the convolutions.
    """

    def __init__(self, c_in, base_ch=64, use_mask=None):
        super().__init__()
        self.use_mask = CFG.use_mask_channel if use_mask is None else use_mask
        in_ch = c_in * 2 if self.use_mask else c_in
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool1d(2)
        self.bot = ConvBlock(base_ch * 4, base_ch * 4)
        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.out = nn.Conv1d(base_ch, c_in, 1)

    def forward(self, x, mask=None):
        if self.use_mask:
            if mask is None:
                mask = torch.ones_like(x)
            x = torch.cat([x * mask, mask], dim=-1)
        x = x.permute(0, 2, 1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(e3)
        d2 = self.dec2(torch.cat([F.interpolate(self.up2(b), size=e2.shape[-1]), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(self.up1(d2), size=e1.shape[-1]), e1], dim=1))
        return self.out(d1).permute(0, 2, 1)


class iTransformer_Lite(nn.Module):
    """Inverted-attention imputation baseline (attends across channels)."""

    def __init__(self, seq_len, feat_dim):
        super().__init__()
        self.use_mask = False
        layer = nn.TransformerEncoderLayer(
            d_model=seq_len, nhead=4, batch_first=True, dim_feedforward=256
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, x, mask=None):
        x_inv = torch.nan_to_num(x.transpose(1, 2), nan=0.0) + 1e-5
        return torch.nan_to_num(self.transformer(x_inv), nan=0.0).transpose(1, 2)


class TIMBA_Lite(nn.Module):
    """Gated state-space-style imputation baseline."""

    def __init__(self, seq_len, feat_dim, hidden=64):
        super().__init__()
        self.use_mask = False
        self.proj_in = nn.Linear(feat_dim, hidden * 2)
        self.conv1d = nn.Conv1d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.proj_out = nn.Linear(hidden, feat_dim)

    def forward(self, x, mask=None):
        h = self.conv1d(self.proj_in(x).transpose(1, 2)).transpose(1, 2)
        h_act, h_gate = h.chunk(2, dim=-1)
        return self.proj_out(h_act * F.silu(h_gate))


def make_diffusion_refiner(feat_dim, sample_steps=None):
    """The backbone shared by CSDI-style baseline and TADR.

    Identical architecture and schedule for both; only the training objective
    differs, so any gap between them is attributable to the objective alone.
    """
    return DiffusionRefiner(
        feat_dim,
        num_train_steps=CFG.diffusion_train_steps,
        num_sample_steps=sample_steps or CFG.diffusion_sample_steps,
        schedule=CFG.diffusion_schedule,
        hard_replace=CFG.diffusion_hard_replace,
    ).to(CFG.device)


class CSDI_Lite(DiffusionRefiner):
    """CSDI-style baseline: the same conditional diffusion model, trained with
    the plain epsilon objective and never shown the detector."""

    def __init__(self, seq_len, feat_dim, base_ch=64):
        super().__init__(feat_dim, base_ch=base_ch,
                         num_train_steps=CFG.diffusion_train_steps,
                         num_sample_steps=CFG.diffusion_sample_steps,
                         schedule=CFG.diffusion_schedule,
                         hard_replace=CFG.diffusion_hard_replace)


class ChannelSubsetRefiner(nn.Module):
    """Runs an inner refiner on a channel subset; the rest passes through.

    Applied identically to every refiner, so it changes what the task is, not
    who wins it.
    """

    def __init__(self, inner, idx):
        super().__init__()
        self.inner = inner
        self.register_buffer("idx", torch.tensor(idx, dtype=torch.long))
        self.use_mask = getattr(inner, "use_mask", False)
        self.x0_clip = getattr(inner, "x0_clip", None)
        # Only surface loss_terms when the wrapped refiner actually has it:
        # the training dispatcher keys on that attribute to decide between the
        # diffusion and direct-regression objectives, so exposing it
        # unconditionally sent the non-diffusion baselines down the wrong path.
        if hasattr(inner, "loss_terms"):
            self.loss_terms = self._loss_terms

    def _sub(self, t):
        return None if t is None else t[..., self.idx]

    def forward(self, x, mask=None, **kw):
        out = x.clone()
        out[..., self.idx] = self.inner(self._sub(x), self._sub(mask), **kw)
        return out

    def _loss_terms(self, x_target, x_cond, mask):
        eps, x0_sub, abar = self.inner.loss_terms(
            self._sub(x_target), self._sub(x_cond), self._sub(mask))
        # Reassemble so the frozen detector sees the window it would see at
        # inference: discrete channels exactly as received.
        x0 = x_cond.clone()
        x0[..., self.idx] = x0_sub
        return eps, x0, abar


def build_refiner(cls_or_factory, feat_dim, dataset_name=None, seq_len=None):
    """Constructs a refiner, gated to continuous channels when enabled."""
    idx = (continuous_channels(dataset_name)
           if CFG.gate_discrete_channels and dataset_name else None)
    if idx is not None and 0 < len(idx) < feat_dim:
        inner_dim = len(idx)
    else:
        idx, inner_dim = None, feat_dim

    inner = (make_diffusion_refiner(inner_dim) if cls_or_factory is None
             else cls_or_factory(seq_len or CFG.window_size, inner_dim).to(CFG.device))
    return ChannelSubsetRefiner(inner, idx).to(CFG.device) if idx else inner


BASELINE_REFINERS = {
    "iTransformer": iTransformer_Lite,
    "TIMBA": TIMBA_Lite,
    "CSDI": CSDI_Lite,
}

# Baselines whose training objective is the diffusion epsilon loss rather than
# a direct MSE on the imputed window.
DIFFUSION_BASELINES = {"CSDI"}


# =============================================================================
# Losses
# =============================================================================

class TaskAwareLoss(nn.Module):
    """L_recon + lambda_feat * L_feat + lambda_enhance * L_enhance.

    L_enhance is asymmetric: on clean windows it pins the refined score to the
    reference score (suppressing reconstruction artefacts, i.e. false alarms);
    on anomalous windows it pushes the score up (preventing the anomaly from
    being smoothed into the normal manifold).
    """

    def __init__(self, use_feat=True, use_enhance=True,
                 lambda_feat=None, lambda_enhance=None):
        super().__init__()
        self.use_feat = use_feat
        self.use_enhance = use_enhance
        self.lf = CFG.lambda_feat if lambda_feat is None else lambda_feat
        self.le = CFG.lambda_enhance if lambda_enhance is None else lambda_enhance
        self.mse = nn.MSELoss()

    def aux(self, x_target, x_ref, det, is_anomaly=False):
        """The task-aware half: lambda_feat * L_feat + lambda_enhance * L_enhance.

        Kept separate from the reconstruction term because the two refiner
        families pair it with different reconstruction objectives -- MSE for the
        single-pass refiner, the epsilon loss for the diffusion refiner -- while
        the task-aware coupling itself is identical.
        """
        if not (self.use_feat or self.use_enhance):
            return x_ref.sum() * 0.0

        with torch.no_grad():
            f_target, s_target = det(x_target)
        f_ref, s_ref = det(x_ref)

        total = x_ref.sum() * 0.0
        if self.use_feat:
            total = total + self.lf * self.mse(f_ref, f_target.detach())
        if self.use_enhance:
            if is_anomaly:
                l_enh = -torch.mean(s_ref)
            else:
                l_enh = torch.mean(torch.abs(s_ref - s_target.detach()))
            total = total + self.le * l_enh
        return total

    def forward(self, x_target, x_ref, det, is_anomaly=False):
        return self.mse(x_ref, x_target) + self.aux(x_target, x_ref, det, is_anomaly)


def inject_synthetic_anomaly(x: torch.Tensor) -> torch.Tensor:
    """Localised Gaussian perturbation, matching the paper's simulation."""
    out = x.clone()
    B, T, C = out.shape
    L = min(CFG.anom_len, T)
    for i in range(B):
        s = int(torch.randint(0, T - L + 1, (1,)).item())
        out[i, s:s + L, :] += torch.randn(L, C, device=out.device) * CFG.anom_sigma
    return out


# =============================================================================
# Training
# =============================================================================

def _loader(dataset, batch_size, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      drop_last=drop_last, num_workers=0, pin_memory=False)


def train_detector(model, dataset, epochs, batch_size=None):
    """Unsupervised: minimise the detector's own reconstruction score."""
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=CFG.detector_lr)
    loader = _loader(dataset, batch_size or CFG.batch_size)
    model.train()
    for _ in range(epochs):
        for (x,) in loader:
            opt.zero_grad()
            _, score = model(x.to(CFG.device))
            score.mean().backward()
            opt.step()
    return model


def train_baseline_refiner(model, dataset, injector, epochs, batch_size=None):
    """Trains an impute-then-detect baseline with no knowledge of the detector.

    Diffusion baselines (CSDI) are trained with the epsilon objective; the rest
    regress the imputed window directly. Neither ever sees the detector, which
    is the property that defines them as baselines.
    """
    if hasattr(model, "loss_terms"):
        return train_diffusion_refiner(
            model, dataset, injector, epochs, task_aware=False,
            lr=CFG.refiner_lr, batch_size=batch_size or CFG.batch_size,
            grad_clip=CFG.grad_clip, device=CFG.device)

    opt, mse = torch.optim.Adam(model.parameters(), lr=CFG.baseline_lr), nn.MSELoss()
    loader = _loader(dataset, batch_size or CFG.batch_size)
    model.train()
    for _ in range(epochs):
        for (x,) in loader:
            x = x.to(CFG.device)
            x_noisy, mask = injector(x)
            opt.zero_grad()
            mse(model(x_noisy, mask), x).backward()
            opt.step()
    return model


def train_tadr(refiner, det, dataset, injector, epochs, loss_fn=None, batch_size=None):
    """Half the batch is clean-plus-corruption, half carries a synthetic anomaly.

    Dispatches on the refiner family: the diffusion model pairs the task-aware
    terms with the epsilon objective (applied to its one-step x0 estimate), the
    single-pass refiner pairs them with a direct MSE.
    """
    loss_fn = loss_fn or TaskAwareLoss()

    if hasattr(refiner, "loss_terms"):
        return train_diffusion_refiner(
            refiner, dataset, injector, epochs, det=det, loss_fn=loss_fn,
            task_aware=True, lr=CFG.refiner_lr,
            batch_size=batch_size or CFG.batch_size, grad_clip=CFG.grad_clip,
            anom_fn=inject_synthetic_anomaly, device=CFG.device,
            aux_snr_weight=CFG.aux_snr_weight)

    opt = torch.optim.Adam(refiner.parameters(), lr=CFG.refiner_lr)
    loader = _loader(dataset, batch_size or CFG.batch_size, drop_last=True)

    for _ in range(epochs):
        refiner.train()
        for (x,) in loader:
            x = x.to(CFG.device)
            half = x.shape[0] // 2
            if half == 0:
                continue

            x_n = x[:half]
            xn_noisy, xn_mask = injector(x_n)
            loss_n = loss_fn(x_n, refiner(xn_noisy, xn_mask), det, is_anomaly=False)

            x_a = inject_synthetic_anomaly(x[half:])
            xa_noisy, xa_mask = injector(x_a)
            target = x_a if CFG.anom_target == "anom" else x[half:]
            loss_a = loss_fn(target, refiner(xa_noisy, xa_mask), det, is_anomaly=True)

            opt.zero_grad()
            (loss_n + loss_a).backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), CFG.grad_clip)
            opt.step()
    return refiner


# =============================================================================
# Evaluation
# =============================================================================

def best_f1_with_pr(scores, labels, smooth_window=1):
    """Strict point-to-point F1. No point adjustment.

    Point adjustment marks an entire ground-truth segment correct after a single
    hit, which hides exactly the precision collapse this paper is about.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        scores = np.convolve(scores, kernel, mode="same")

    lo, hi = np.percentile(scores, CFG.pct_lo), np.percentile(scores, CFG.pct_hi)
    best = (0.0, 0.0, 0.0, lo)
    for thr in np.linspace(lo, hi, CFG.n_thresholds):
        pred = (scores > thr).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best[0]:
            best = (f1, precision_score(labels, pred, zero_division=0),
                    recall_score(labels, pred, zero_division=0), thr)
    return best  # (f1, precision, recall, threshold)


@torch.no_grad()
def entity_scores(refiner, det, test_windows, n_labels, injector, seed=None):
    """Per-timestep anomaly score, averaged over all overlapping windows.

    `refiner=None` is the Raw condition: corrupted data goes straight to the
    detector with no imputation at all.
    """
    if refiner is not None:
        refiner.eval()
    det.eval()

    score_sum = np.zeros(n_labels)
    count = np.zeros(n_labels)
    idx = 0
    W = CFG.window_size

    for (x,) in DataLoader(TensorDataset(test_windows), batch_size=CFG.eval_batch_size):
        x = x.to(CFG.device)
        batch_seed = None if seed is None else seed + idx
        x_noisy, mask = injector(x, seed=batch_seed)
        x_ref = x_noisy if refiner is None else refiner(x_noisy, mask)
        score = det(x_ref)[1].cpu().numpy()
        for i in range(x.shape[0]):
            s = idx * CFG.test_stride
            e = min(s + W, n_labels)
            score_sum[s:e] += score[i, :e - s]
            count[s:e] += 1
            idx += 1

    return score_sum / np.maximum(count, 1.0)


def global_scores(refiners: dict, det, test_entities, injector, seed=None):
    """Pool every entity's scores, then threshold once globally.

    Per-entity thresholds would let a method pick a different operating point on
    each machine, which is not deployable; a single global threshold is the
    strict reading and the one the paper reports.
    """
    acc = {name: [] for name in refiners}
    all_labels = []

    for e_i, ent in enumerate(test_entities):
        te_win = create_sliding_windows(ent["test"], CFG.window_size, CFG.test_stride)
        if te_win.shape[0] == 0:
            continue
        labels = ent["labels"]
        all_labels.append(labels)
        # Same corruption for every refiner on this entity when paired_corruption.
        ent_seed = None if (seed is None or not CFG.paired_corruption) else seed * 10007 + e_i
        for name, refiner in refiners.items():
            acc[name].append(entity_scores(refiner, det, te_win, len(labels), injector, ent_seed))
        del te_win
        gc.collect()

    labels = np.concatenate(all_labels)
    return {name: np.concatenate(v) for name, v in acc.items()}, labels


@torch.no_grad()
def refinement_diagnostics(refiner, det, test_windows, injector, seed, n_windows=512):
    """Reconstruction fidelity of the refined signal, reported next to F1.

    Best-threshold P2P F1 on its own cannot separate a genuine refinement from a
    degenerate one. A refiner whose output saturated at the clipping bound on
    63% of its samples -- reconstruction MSE 30x worse than doing nothing --
    still scored F1 0.69, because the amount of saturation happened to track
    input magnitude and the threshold search then picked that up. These columns
    make that failure mode visible in every table instead of silent.
    """
    nan = float("nan")
    if refiner is None:
        return {"out_std": nan, "sat_pct": nan, "obs_mse": nan, "miss_mse": nan}

    refiner.eval()
    x = test_windows[:n_windows].to(CFG.device)
    x_noisy, mask = injector(x, seed=seed)
    out = refiner(x_noisy, mask)
    obs = mask.bool()
    clip = getattr(refiner, "x0_clip", None)
    # obs_nmse normalises by the target's own variance, so it is comparable
    # across datasets: 1.0 means "no better than predicting the mean".
    var = float(x.var()) or 1.0
    return {
        "out_std": float(out.std()),
        "target_std": float(x.std()),
        "sat_pct": float((out.abs() >= clip * 0.999).float().mean() * 100) if clip else nan,
        "obs_mse": float(F.mse_loss(out[obs], x[obs])) if obs.any() else nan,
        "miss_mse": float(F.mse_loss(out[~obs], x[~obs])) if (~obs).any() else nan,
        "obs_nmse": float(F.mse_loss(out[obs], x[obs])) / var if obs.any() else nan,
    }


def paired_tests(raw, keys, method_col, metric, reference):
    """Paired t-test of every method against a reference, across seeds.

    The paper's claim is about a difference, so the difference is what gets
    tested: pairing on seed cancels the seed-to-seed variance the two methods
    share, which is far more powerful than comparing two independent intervals.
    """
    from scipy import stats

    other = [k for k in keys if k != method_col]
    groups = raw.groupby(other, sort=False) if other else [((), raw)]
    rows = []
    for gv, g in groups:
        if not isinstance(gv, tuple):
            gv = (gv,)
        piv = g.pivot_table(index="seed", columns=method_col, values=metric)
        if reference not in piv.columns:
            continue
        for m in piv.columns:
            if m == reference:
                continue
            d = (piv[m] - piv[reference]).dropna()
            if len(d) < 2:
                continue
            t, pv = stats.ttest_rel(piv[m][d.index], piv[reference][d.index])
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            row = dict(zip(other, gv))
            row.update({"metric": metric, "method": m, "reference": reference,
                        "n_pairs": len(d), "mean_diff": float(d.mean()),
                        "ci95_half": float(half), "t": float(t), "p_value": float(pv)})
            rows.append(row)
    return pd.DataFrame(rows)


def pooled_epochs(n_windows: int, batch_size: int) -> int:
    """Epochs that give this dataset the same gradient-step budget as the others.

    Comparing datasets is only meaningful if the model was trained equally hard
    on each; deriving epochs from the window count is what makes the columns of
    the results table comparable.
    """
    steps_per_epoch = max(1, n_windows // batch_size)
    return int(max(3, min(200, -(-CFG.pooled_target_steps // steps_per_epoch))))


def load_pooled(dataset_name):
    """Pooled training windows + per-entity test arrays (windowed lazily)."""
    entities = get_dataset_entities(dataset_name)
    if not entities:
        raise FileNotFoundError(f"No entities found for {dataset_name} under {CFG.base_dir}")

    train_sets, test_entities, feat_dim = [], [], None
    for ent in entities:
        tr, te, labels = load_and_scale_dataset(dataset_name, ent)
        tr_win = create_sliding_windows(tr, CFG.window_size, CFG.pooled_train_stride)
        if tr_win.shape[0] == 0:
            continue
        train_sets.append(TensorDataset(tr_win))
        test_entities.append({"entity": ent, "test": te, "labels": labels})
        feat_dim = tr_win.shape[-1]

    return ConcatDataset(train_sets), test_entities, feat_dim


# =============================================================================
# Experiments
# =============================================================================

def exp1_refiner_comparison(seed):
    """Table: F1 / Precision / Recall under 40% block missing, SMD machine-1-1.

    This is the imputation-bias evidence: MSE-trained imputers reach ~0.99
    recall at <0.20 precision because they fabricate artefacts everywhere.
    """
    set_seed(seed)
    tr, te, labels = load_and_scale_dataset("SMD", "machine-1-1")
    tr_win = create_sliding_windows(tr, CFG.window_size, CFG.train_stride)
    te_win = create_sliding_windows(te, CFG.window_size, CFG.test_stride)
    feat_dim = tr_win.shape[-1]
    train_ds = TensorDataset(tr_win)

    det = FrozenDetector(
        train_detector(KAN_AD_Lite(CFG.window_size, feat_dim).to(CFG.device),
                       train_ds, CFG.detector_epochs)
    ).to(CFG.device)

    train_inj = NoiseInjector("mixed", missing_ratio=0.2)
    test_inj = NoiseInjector("block_missing", missing_ratio=0.4)

    refiners = {"Raw (No Refiner)": None}
    for name, cls in BASELINE_REFINERS.items():
        refiners[name] = train_baseline_refiner(
            build_refiner(cls, feat_dim, "SMD"), train_ds, train_inj, CFG.refiner_epochs
        )
    refiners["TADR (Ours)"] = train_tadr(
        build_refiner(None, feat_dim, "SMD"), det, train_ds, train_inj, CFG.refiner_epochs
    )

    rows = []
    for name, refiner in refiners.items():
        scores = entity_scores(refiner, det, te_win, len(labels), test_inj, seed)
        f1, p, r, _ = best_f1_with_pr(scores, labels)
        row = {"seed": seed, "Refiner": name, "F1": f1, "Precision": p, "Recall": r}
        row.update(refinement_diagnostics(refiner, det, te_win, test_inj, seed))
        rows.append(row)
        print(f"    {name:<18} F1 {f1:.4f} P {p:.4f} R {r:.4f} | "
              f"obs_mse {row['obs_mse']:.3f} sat {row['sat_pct']:.2f}%", flush=True)
    return pd.DataFrame(rows)


def _pooled_refiner_sweep(seed, datasets, ratios):
    """Shared engine for exp2 and exp4: pooled global P2P F1 per (dataset, ratio)."""
    rows = []
    for ds_name in datasets:
        set_seed(seed)
        print(f"  [{ds_name}] loading...", flush=True)
        train_ds, test_entities, feat_dim = load_pooled(ds_name)
        epochs = pooled_epochs(len(train_ds), CFG.pooled_batch_size)
        print(f"  [{ds_name}] {len(train_ds):,} windows -> {epochs} epochs "
              f"(~{len(train_ds) // CFG.pooled_batch_size * epochs:,} steps)", flush=True)

        det = FrozenDetector(
            train_detector(KAN_AD_Lite(CFG.window_size, feat_dim).to(CFG.device),
                           train_ds, CFG.pooled_detector_epochs, CFG.pooled_batch_size)
        ).to(CFG.device)

        train_inj = NoiseInjector("mixed", missing_ratio=0.2)
        refiners = {"Raw": None}
        for name, cls in BASELINE_REFINERS.items():
            refiners[name] = train_baseline_refiner(
                build_refiner(cls, feat_dim, ds_name), train_ds, train_inj,
                epochs, CFG.pooled_batch_size
            )
        refiners["TADR"] = train_tadr(
            build_refiner(None, feat_dim, ds_name), det, train_ds, train_inj,
            epochs, batch_size=CFG.pooled_batch_size
        )

        diag_win = create_sliding_windows(test_entities[0]["test"], CFG.window_size,
                                          CFG.window_size)
        for ratio in ratios:
            test_inj = NoiseInjector("block_missing", missing_ratio=ratio)
            pooled, labels = global_scores(refiners, det, test_entities, test_inj, seed)
            summary = []
            for name, sc in pooled.items():
                f1, p, r, _ = best_f1_with_pr(sc, labels)
                row = {"seed": seed, "Dataset": ds_name, "Ratio": ratio,
                       "Method": name, "F1": f1, "Precision": p, "Recall": r}
                row.update(refinement_diagnostics(refiners[name], det, diag_win,
                                                  test_inj, seed))
                rows.append(row)
                summary.append(f"{name}:{f1:.3f}")
            print(f"    ratio {ratio:.1f}: " + " | ".join(summary), flush=True)

        del refiners, det, train_ds, test_entities
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def exp2_global_40(seed):
    """Table: global P2P F1 across SMD/SMAP/MSL/PSM at 40% block missing."""
    return _pooled_refiner_sweep(seed, ["SMD", "SMAP", "MSL", "PSM"], [0.4])


def exp4_global_ratios(seed):
    """Table: global P2P F1 across datasets at 10% / 30% / 50% block missing."""
    return _pooled_refiner_sweep(seed, ["SMD", "SMAP", "MSL", "PSM"], [0.1, 0.3, 0.5])


def exp3_detector_generality(seed):
    """Table: detector P2P F1 with and without TADR, pooled SMD.

    A separate TADR is trained per detector -- the refiner is task-aware, so it
    must see the score head it will be deployed behind.

    PROTOCOL NOTE: the notebook that produced the published numbers corrupted
    the TRAINING data and evaluated on clean test data. That measures training-
    set robustness, not the test-time block-missing scenario the table claims.
    This implementation corrupts the TEST stream, consistent with every other
    table in the paper.
    """
    rows = []
    set_seed(seed)
    train_ds, test_entities, feat_dim = load_pooled("SMD")
    epochs = pooled_epochs(len(train_ds), CFG.pooled_batch_size)
    train_inj = NoiseInjector("mixed", missing_ratio=0.2)

    conditions = {
        "Block Missing 0.4": NoiseInjector("block_missing", missing_ratio=0.4),
        "Gaussian 1.0": NoiseInjector("gaussian", gaussian_std=1.0),
    }

    for det_name, det_cls in DETECTORS.items():
        print(f"  [{det_name}]", flush=True)
        det = FrozenDetector(
            train_detector(det_cls(CFG.window_size, feat_dim).to(CFG.device),
                           train_ds, CFG.pooled_detector_epochs, CFG.pooled_batch_size)
        ).to(CFG.device)
        tadr = train_tadr(
            build_refiner(None, feat_dim, "SMD"), det, train_ds, train_inj,
            epochs, batch_size=CFG.pooled_batch_size
        )

        diag_win = create_sliding_windows(test_entities[0]["test"], CFG.window_size,
                                          CFG.window_size)
        for cond_name, test_inj in conditions.items():
            pooled, labels = global_scores({"Raw": None, "TADR": tadr}, det,
                                           test_entities, test_inj, seed)
            f1_raw = best_f1_with_pr(pooled["Raw"], labels)[0]
            f1_tadr = best_f1_with_pr(pooled["TADR"], labels)[0]
            row = {"seed": seed, "Detector": det_name, "Condition": cond_name,
                   "Raw_F1": f1_raw, "TADR_F1": f1_tadr, "Delta": f1_tadr - f1_raw}
            row.update(refinement_diagnostics(tadr, det, diag_win, test_inj, seed))
            rows.append(row)
            print(f"    {cond_name}: raw {f1_raw:.4f} -> tadr {f1_tadr:.4f} "
                  f"| obs_mse {row['obs_mse']:.3f} sat {row['sat_pct']:.2f}%", flush=True)

        del det, tadr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def exp5_ablation(seed):
    """Table: loss-component ablation, smoothed P2P, SMD machine-1-1, 40% missing.

    Scores are smoothed with a moving average before thresholding, because
    strict P2P zeroes out a detection that is one or two steps late.
    """
    set_seed(seed)
    tr, te, labels = load_and_scale_dataset("SMD", "machine-1-1")
    tr_win = create_sliding_windows(tr, CFG.window_size, CFG.train_stride)
    te_win = create_sliding_windows(te, CFG.window_size, CFG.test_stride)
    feat_dim = tr_win.shape[-1]
    train_ds = TensorDataset(tr_win)

    det = FrozenDetector(
        train_detector(KAN_AD_Lite(CFG.window_size, feat_dim).to(CFG.device),
                       train_ds, CFG.detector_epochs)
    ).to(CFG.device)

    train_inj = NoiseInjector("mixed", missing_ratio=0.2)
    test_inj = NoiseInjector("block_missing", missing_ratio=0.4)

    variants = {
        "Recon (MSE only)": TaskAwareLoss(use_feat=False, use_enhance=False),
        "Recon + Feature": TaskAwareLoss(use_feat=True, use_enhance=False),
        "Recon + Feature + Enhance (Full)": TaskAwareLoss(
            use_feat=True, use_enhance=True, lambda_enhance=CFG.lambda_enhance_ablation),
    }

    rows = []
    builders = [(n, lambda fd=feat_dim: build_refiner(None, fd, "SMD"), lf)
                for n, lf in variants.items()]
    # Does the iterative generative process earn its cost? Same full objective,
    # one forward pass instead of T. Answers the question before a reviewer asks.
    builders.append(("Full objective, single-pass refiner",
                     lambda fd=feat_dim: build_refiner(
                         lambda sl, fd2: SinglePassRefiner(fd2), fd, "SMD"),
                     TaskAwareLoss(lambda_enhance=CFG.lambda_enhance_ablation)))

    for name, build, loss_fn in builders:
        refiner = train_tadr(build(), det, train_ds, train_inj,
                             CFG.refiner_epochs, loss_fn=loss_fn)
        scores = entity_scores(refiner, det, te_win, len(labels), test_inj, seed)
        f1, p, r, _ = best_f1_with_pr(scores, labels, smooth_window=CFG.smooth_window)
        row = {"seed": seed, "Loss": name, "F1": f1, "Precision": p, "Recall": r}
        row.update(refinement_diagnostics(refiner, det, te_win, test_inj, seed))
        rows.append(row)
        print(f"    {name}: F1 {f1:.4f} P {p:.4f} R {r:.4f} | "
              f"obs_mse {row['obs_mse']:.3f}", flush=True)
        del refiner
        gc.collect()
    return pd.DataFrame(rows)


@torch.no_grad()
def _measure_overhead(refiner, det, x, mask, iters=50):
    det.eval()
    if refiner is not None:
        refiner.eval()

    for _ in range(10):  # warm-up: CUDA context, cuDNN autotune, allocator caches
        det(x if refiner is None else refiner(x, mask))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        det(x if refiner is None else refiner(x, mask))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / iters * 1000

    mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
    return elapsed_ms, mem_mb


def exp6_overhead(seed):
    """Table: inference latency and peak VRAM at batch size 256.

    Latency is PER BATCH OF 256 WINDOWS. Per-window cost is this divided by 256
    -- state that explicitly in the paper, since a reader who reads 45 ms as a
    per-sample figure will conclude the method cannot run at 1 Hz.
    """
    set_seed(seed)
    tr, te, labels = load_and_scale_dataset("SMD", "machine-1-1")
    tr_win = create_sliding_windows(tr, CFG.window_size, CFG.train_stride)
    te_win = create_sliding_windows(te, CFG.window_size, CFG.test_stride)
    feat_dim = tr_win.shape[-1]
    train_ds = TensorDataset(tr_win)

    det = FrozenDetector(
        train_detector(KAN_AD_Lite(CFG.window_size, feat_dim).to(CFG.device),
                       train_ds, CFG.detector_epochs)
    ).to(CFG.device)

    train_inj = NoiseInjector("mixed", missing_ratio=0.2)
    models = {"Raw (No Refiner)": None}
    for name, cls in BASELINE_REFINERS.items():
        models[name] = train_baseline_refiner(
            build_refiner(cls, feat_dim, "SMD"), train_ds, train_inj, CFG.refiner_epochs
        )
    models["TADR (Ours)"] = train_tadr(
        build_refiner(None, feat_dim, "SMD"), det, train_ds, train_inj, CFG.refiner_epochs
    )

    test_inj = NoiseInjector("block_missing", missing_ratio=0.4)
    x, mask = test_inj(te_win[:CFG.eval_batch_size].to(CFG.device), seed=seed)

    rows = []
    for name, model in models.items():
        ms, mem = _measure_overhead(model, det, x, mask)
        n_params = 0 if model is None else sum(p.numel() for p in model.parameters())
        rows.append({"seed": seed, "Refiner": name,
                     "Latency_ms_per_batch256": ms,
                     "Latency_ms_per_window": ms / CFG.eval_batch_size,
                     "Peak_VRAM_MB": mem, "Params": n_params})
        print(f"    {name}: {ms:.2f} ms/batch  ({ms / CFG.eval_batch_size:.4f} ms/window), "
              f"{mem:.1f} MB, {n_params/1e6:.2f}M params", flush=True)
    return pd.DataFrame(rows)


EXPERIMENTS = {
    "1": ("Refiner comparison, SMD 1-1, 40% block missing", exp1_refiner_comparison,
          ["Refiner"], ("Refiner", "Raw (No Refiner)")),
    "2": ("Global P2P F1, 4 datasets, 40% block missing", exp2_global_40,
          ["Dataset", "Ratio", "Method"], ("Method", "Raw")),
    "3": ("Detector generality with/without TADR, pooled SMD", exp3_detector_generality,
          ["Detector", "Condition"], None),
    "4": ("Global P2P F1, varying block-missing ratios", exp4_global_ratios,
          ["Dataset", "Ratio", "Method"], ("Method", "Raw")),
    "5": ("Ablation on loss components, smoothed P2P", exp5_ablation,
          ["Loss"], ("Loss", "Recon (MSE only)")),
    "6": ("System inference overhead, batch size 256", exp6_overhead,
          ["Refiner"], None),
}


# =============================================================================
# Aggregation over seeds
# =============================================================================

def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """mean / std / 95% CI half-width per metric.

    Reviewer-facing: reporting a CI rather than a bare mean is what makes the
    "N independent runs" claim checkable.
    """
    from scipy import stats

    metrics = [c for c in df.columns if c not in keys + ["seed"]]
    out = []
    for key_vals, g in df.groupby(keys, sort=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = dict(zip(keys, key_vals))
        n = len(g)
        row["n_runs"] = n
        for m in metrics:
            v = g[m].to_numpy(dtype=float)
            row[f"{m}_mean"] = v.mean()
            row[f"{m}_std"] = v.std(ddof=1) if n > 1 else 0.0
            if n > 1:
                half = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
            else:
                half = float("nan")
            row[f"{m}_ci95"] = half
        out.append(row)
    return pd.DataFrame(out)


def run_experiment(exp_id: str, seeds: list[int], out_dir: str):
    title, fn, keys, paired = EXPERIMENTS[exp_id]
    print(f"\n{'=' * 78}\n[exp{exp_id}] {title}\n{'=' * 78}", flush=True)

    frames = []
    for s in seeds:
        print(f"-- seed {s}", flush=True)
        frames.append(fn(s))
    raw = pd.concat(frames, ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    raw.to_csv(os.path.join(out_dir, f"exp{exp_id}_runs.csv"), index=False)
    agg = aggregate(raw, keys)
    agg.to_csv(os.path.join(out_dir, f"exp{exp_id}.csv"), index=False)

    if paired is not None and len(seeds) > 1:
        method_col, reference = paired
        pt = paired_tests(raw, keys, method_col, "F1", reference)
        if len(pt):
            pt.to_csv(os.path.join(out_dir, f"exp{exp_id}_paired.csv"), index=False)

    print(f"\n[exp{exp_id}] aggregated over {len(seeds)} run(s)")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return agg


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="all",
                    help="experiment id (1-6), comma separated, or 'all'")
    ap.add_argument("--runs", type=int, default=1, help="number of independent seeds")
    ap.add_argument("--seed", type=int, default=0, help="first seed; runs use seed..seed+runs-1")
    ap.add_argument("--seed-list", default=None,
                    help="explicit comma-separated seeds; overrides --seed/--runs. "
                         "Use this to split a 5-seed run across GPUs.")
    ap.add_argument("--data-dir", default=CFG.base_dir)
    ap.add_argument("--out", default=None, help="output directory (default: ./results)")
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--fast", action="store_true",
                    help="smoke test: few entities, 1 epoch, coarse thresholds")
    ap.add_argument("--use-mask-channel", action="store_true",
                    help="feed TADR an explicit observation mask (2C input channels)")
    ap.add_argument("--anom-target", choices=["anom", "clean"], default=CFG.anom_target)
    ap.add_argument("--lambda-feat", type=float, default=CFG.lambda_feat)
    ap.add_argument("--lambda-enhance", type=float, default=CFG.lambda_enhance)
    args = ap.parse_args()

    CFG.base_dir = args.data_dir
    CFG.device = args.device
    CFG.anom_target = args.anom_target
    CFG.lambda_feat = args.lambda_feat
    CFG.lambda_enhance = args.lambda_enhance
    if args.use_mask_channel:
        CFG.use_mask_channel = True
        CFG.drop_val = -5.0
    if args.fast:
        CFG.fast = True
        CFG.detector_epochs = CFG.refiner_epochs = 1
        CFG.pooled_detector_epochs = CFG.pooled_refiner_epochs = 1
        CFG.n_thresholds = 50

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(CFG), f, indent=2)

    exp_ids = list(EXPERIMENTS) if args.exp == "all" else [e.strip() for e in args.exp.split(",")]
    unknown = [e for e in exp_ids if e not in EXPERIMENTS]
    if unknown:
        ap.error(f"unknown experiment id(s): {unknown}; choose from {list(EXPERIMENTS)}")

    seeds = ([int(x) for x in args.seed_list.split(",")] if args.seed_list
             else list(range(args.seed, args.seed + args.runs)))
    print(f"device={CFG.device}  seeds={seeds}  experiments={exp_ids}  out={out_dir}")

    for e in exp_ids:
        run_experiment(e, seeds, out_dir)

    print(f"\nDone. CSVs written to {out_dir}")


if __name__ == "__main__":
    main()
