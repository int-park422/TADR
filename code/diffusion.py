#!/usr/bin/env python3
"""
Conditional diffusion refiner for block-missing multivariate time series.

This is the actual generative model the paper describes: a DDPM over the window,
conditioned on the observed samples and their mask, sampled iteratively at
inference. It replaces the single-pass U-Net that earlier notebooks used, and it
is what makes the paper's O(T . C_CNN) complexity claim true.

Two models are built from the same backbone, so the comparison between them
isolates the objective rather than the architecture:

    CSDI-style baseline : trained with the plain epsilon-prediction loss only
    TADR                : same network, same schedule, plus the task-aware terms
                          (feature matching + anomaly enhancement) applied to the
                          one-step x0 estimate

Coupling a diffusion model to a downstream loss is the one non-obvious part.
Back-propagating through a full T-step sampling loop is intractable, so the
task-aware terms are applied to the Tweedie estimate

    x0_hat = (x_t - sqrt(1 - abar_t) * eps_theta(x_t, t, cond)) / sqrt(abar_t)

which is available in the same forward pass that produces the epsilon loss.
Training cost is therefore identical to plain diffusion training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Noise schedule
# =============================================================================

class DiffusionSchedule:
    """Forward-process constants, held on one device.

    "quad" is the schedule CSDI uses for time-series imputation; "cosine" is the
    Nichol-Dhariwal schedule. Quad is the default so the baseline is faithful to
    the method it is named after.
    """

    def __init__(self, num_steps=50, schedule="quad", beta_start=1e-4, beta_end=0.5,
                 device="cpu"):
        self.num_steps = num_steps
        if schedule == "quad":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule == "cosine":
            s, steps = 0.008, num_steps + 1
            t = torch.linspace(0, num_steps, steps) / num_steps
            f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            abar = f / f[0]
            betas = torch.clip(1 - abar[1:] / abar[:-1], 0.0001, 0.9999)
        else:
            raise ValueError(f"unknown schedule: {schedule}")

        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def abar(self, t: torch.Tensor, dtype=None) -> torch.Tensor:
        """alpha_bar for a batch of timesteps, shaped (B, 1, 1) for broadcasting.

        The schedule is stored in float32. Arithmetic against a half-precision
        window would silently promote it back to float32, and the next reverse
        step would then feed float activations into half weights, so the
        constants are cast to the tensor they are combined with.
        """
        a = self.alpha_bars[t].view(-1, 1, 1)
        return a if dtype is None else a.to(dtype)

    def q_sample(self, x0, t, noise):
        a = self.abar(t, x0.dtype)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def x0_from_eps(self, x_t, t, eps, clip=None):
        """Tweedie's formula: the model's implied clean sample.

        The division by sqrt(abar) is ~1/0.006 at the last timestep, so any
        epsilon error is amplified ~170x there. Clipping x0 to a plausible data
        range is what keeps both sampling and the task-aware terms stable; the
        inputs are standardised, so the bound is in units of sigma.
        """
        a = self.abar(t, x_t.dtype)
        x0 = (x_t - (1 - a).sqrt() * eps) / a.sqrt().clamp_min(1e-8)
        return x0.clamp(-clip, clip) if clip else x0


# =============================================================================
# Backbone
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))
        # A buffer tracks .to(dtype) but is left alone by quantisation, whereas a
        # quantised Linear replaces `weight` with a method and would break a
        # weight.dtype lookup here.
        self.register_buffer("_dtype_probe", torch.zeros(1), persistent=False)

    def forward(self, t):
        half = self.dim // 2
        # Build in the module's own dtype: the embedding is computed from an
        # integer timestep, so it would otherwise always come out float32 and
        # fail against half-precision weights when the model is run in FP16.
        dtype = self._dtype_probe.dtype
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=1).to(dtype))


class TimeConvBlock(nn.Module):
    """Two convolutions with the timestep embedding injected between them.

    GroupNorm rather than BatchNorm: it is the diffusion standard and, unlike
    BatchNorm, its behaviour does not depend on batch size -- which matters
    directly here, since the edge deployment this paper targets runs at batch 1.
    """

    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        g = math.gcd(8, out_ch) or 1
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(g, out_ch)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.t_proj(temb).unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class ConditionalUNet1D(nn.Module):
    """Predicts epsilon from [x_t, observed*mask, mask] -- 3C input channels."""

    def __init__(self, c_in, base_ch=64, t_dim=128):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(t_dim)
        self.enc1 = TimeConvBlock(c_in * 3, base_ch, t_dim)
        self.enc2 = TimeConvBlock(base_ch, base_ch * 2, t_dim)
        self.enc3 = TimeConvBlock(base_ch * 2, base_ch * 4, t_dim)
        self.pool = nn.MaxPool1d(2)
        self.bot = TimeConvBlock(base_ch * 4, base_ch * 4, t_dim)
        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = TimeConvBlock(base_ch * 4, base_ch * 2, t_dim)
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = TimeConvBlock(base_ch * 2, base_ch, t_dim)
        self.out = nn.Conv1d(base_ch, c_in, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_t, x_cond, mask, t):
        temb = self.time(t)
        h = torch.cat([x_t, x_cond * mask, mask], dim=-1).permute(0, 2, 1)
        e1 = self.enc1(h, temb)
        e2 = self.enc2(self.pool(e1), temb)
        e3 = self.enc3(self.pool(e2), temb)
        b = self.bot(e3, temb)
        d2 = self.dec2(torch.cat([F.interpolate(self.up2(b), size=e2.shape[-1]), e2], 1), temb)
        d1 = self.dec1(torch.cat([F.interpolate(self.up1(d2), size=e1.shape[-1]), e1], 1), temb)
        return self.out(d1).permute(0, 2, 1)


# =============================================================================
# Refiner
# =============================================================================

class DiffusionRefiner(nn.Module):
    """Conditional DDPM refiner with DDIM sampling.

    `forward(x_cond, mask)` runs the reverse process and returns the refined
    window, so it is a drop-in replacement for the single-pass refiner
    everywhere in the evaluation code.
    """

    use_mask = True   # this refiner always consumes the observation mask

    def __init__(self, c_in, base_ch=64, num_train_steps=50, num_sample_steps=50,
                 schedule="quad", hard_replace=False, x0_clip=10.0, device="cpu"):
        super().__init__()
        self.net = ConditionalUNet1D(c_in, base_ch=base_ch)
        self.sched = DiffusionSchedule(num_train_steps, schedule, device=device)
        self.num_sample_steps = num_sample_steps
        # Inputs are standardised, so +/-10 sigma bounds anything the data can
        # plausibly contain while still admitting large anomalies.
        self.x0_clip = x0_clip
        # Hard replacement pins observed positions to their measured values at
        # every reverse step (RePaint-style inpainting). Left off by default:
        # the paper evaluates a Gaussian-noise condition in which nothing is
        # missing, and under hard replacement the refiner would be a no-op there.
        self.hard_replace = hard_replace

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        dev = next(self.net.parameters()).device
        self.sched.to(dev)
        return out

    # -- training -----------------------------------------------------------
    def loss_terms(self, x_target, x_cond, mask):
        """One forward pass -> (epsilon MSE, x0 estimate, per-sample abar).

        abar is returned so the caller can down-weight the task-aware terms at
        high noise levels, where x0_hat carries almost no signal.
        """
        B = x_target.shape[0]
        dev = x_target.device
        t = torch.randint(0, self.sched.num_steps, (B,), device=dev)
        noise = torch.randn_like(x_target)
        x_t = self.sched.q_sample(x_target, t, noise)

        eps_hat = self.net(x_t, x_cond, mask, t)
        eps_loss = F.mse_loss(eps_hat, noise)
        x0_hat = self.sched.x0_from_eps(x_t, t, eps_hat, clip=self.x0_clip)
        return eps_loss, x0_hat, self.sched.abar(t).view(B)

    # -- sampling -----------------------------------------------------------
    @torch.no_grad()
    def sample(self, x_cond, mask, num_steps=None):
        """Deterministic DDIM over a strided subset of the training timesteps."""
        n = num_steps or self.num_sample_steps
        n = max(1, min(n, self.sched.num_steps))
        steps = torch.linspace(self.sched.num_steps - 1, 0, n).round().long().tolist()

        x = torch.randn_like(x_cond)
        if self.hard_replace:
            x = mask * x_cond + (1 - mask) * x

        for i, ti in enumerate(steps):
            t = torch.full((x.shape[0],), ti, device=x.device, dtype=torch.long)
            eps = self.net(x, x_cond, mask, t)
            x0 = self.sched.x0_from_eps(x, t, eps, clip=self.x0_clip)

            if i + 1 < len(steps):
                a_next = self.sched.alpha_bars[steps[i + 1]].to(x.dtype)
                x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
            else:
                x = x0

            if self.hard_replace:
                x = mask * x_cond + (1 - mask) * x

        return x

    def forward(self, x_cond, mask=None, num_steps=None):
        if mask is None:
            mask = torch.ones_like(x_cond)
        return self.sample(x_cond, mask, num_steps)


def build_refiner(c_in, kind="diffusion", **kw):
    """kind: 'diffusion' (TADR / CSDI backbone) -- kept as a hook for variants."""
    if kind != "diffusion":
        raise ValueError(f"unknown refiner kind: {kind}")
    return DiffusionRefiner(c_in, **kw)


# =============================================================================
# Training
# =============================================================================

def train_diffusion_refiner(refiner, dataset, injector, epochs, det=None,
                            loss_fn=None, task_aware=False, lr=3e-4, batch_size=64,
                            grad_clip=1.0, anom_fn=None, device="cuda",
                            aux_snr_weight=True):
    """Trains either the plain baseline or the task-aware refiner.

    task_aware=False reproduces a CSDI-style imputer: epsilon loss only, and it
    never sees the detector. task_aware=True adds the feature-matching and
    anomaly-enhancement terms on x0_hat, and splits each batch into a clean half
    and a synthetic-anomaly half exactly as the paper describes.
    """
    from torch.utils.data import DataLoader

    opt = torch.optim.Adam(refiner.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=task_aware)

    for _ in range(epochs):
        refiner.train()
        for (x,) in loader:
            x = x.to(device)
            opt.zero_grad()

            if not task_aware:
                x_noisy, mask = injector(x)
                eps_loss, _, _ = refiner.loss_terms(x, x_noisy, mask)
                eps_loss.backward()
            else:
                half = x.shape[0] // 2
                if half == 0:
                    continue
                total = 0.0
                for x_part, is_anom in ((x[:half], False), (anom_fn(x[half:]), True)):
                    x_noisy, mask = injector(x_part)
                    eps_loss, x0_hat, abar = refiner.loss_terms(x_part, x_noisy, mask)
                    # x0_hat is only informative at low noise; weight the
                    # task-aware terms by abar so they do not chase noise.
                    w = abar.mean() if aux_snr_weight else 1.0
                    total = total + eps_loss + w * loss_fn.aux(x_part, x0_hat, det, is_anom)
                total.backward()

            torch.nn.utils.clip_grad_norm_(refiner.parameters(), grad_clip)
            opt.step()
    return refiner
