#!/usr/bin/env python3
"""
One-stop script:
  1) load raw full-d unit vectors (meta.json + hidden_unit_vectors.npy)
  2) apply a chosen transformation regime:
     - fixed full→k projection (shared Gaussian A[d,k] seeded by block_hash+public_key)
     - optionally: sign flips (per nonce)
     - optionally: Householder reflections (per nonce, configurable count)
     - optionally: Haar rotation in k (per nonce)
  3) run bubble test (pick radius or use fixed) and write:
     - summary.json
     - bubble_rate_hist.png
     - bubble_rates.npy
     - k_data.npz (xk + bubble artifacts)

This is intended for fast iteration on different regimes/params using the same raw data.
"""

from __future__ import annotations

import argparse
import json
import math

# Allow importing sibling helper scripts when running as a standalone file.
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from bubble_plot import plot_bubble_rate_hist  # type: ignore
from gpu_random_utils import (
    apply_householder_reflections,
    apply_sign_flips_then_normalize,
    fixed_project_full_to_k,
    haar_rotate_k,
    unit_normalize,
)


@dataclass
class BubbleRunSummary:
    setup: str
    n: int
    d_full: int
    d_analyze: int
    mean_vector_norm: float
    cov_eig_min: float
    cov_eig_max: float
    cov_eig_mean: float
    cov_eig_std: float
    bubble_num_targets: int
    bubble_target_rate: float
    bubble_radius: float
    bubble_rate_mean: float
    bubble_rate_std: float
    bubble_rate_variance: float
    bubble_rate_min: float
    bubble_rate_max: float


def _load_vectors(in_dir: Path, meta: dict[str, Any]) -> np.ndarray:
    vecs_path = in_dir / "hidden_unit_vectors.npy"
    arr = np.load(vecs_path, mmap_mode="r")
    storage_dtype = meta.get("storage_dtype", "float16")
    if storage_dtype == "bfloat16":
        t = torch.from_numpy(arr.astype(np.uint16))
        t = t.view(torch.bfloat16).float()
        return t.numpy()
    return np.asarray(arr)


def _unit_normalize_t(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def _pick_radius_for_target_rate_torch(
    x_unit: torch.Tensor,
    *,
    num_targets: int,
    target_rate: float,
    seed: int,
    max_pairs: int,
    sample_x: int,
    chunk_targets: int,
) -> float:
    tr = float(target_rate)
    if not (0.0 < tr < 1.0):
        raise ValueError(f"target_rate must be in (0,1), got {target_rate}")
    n, d = x_unit.shape
    g = torch.Generator(device=x_unit.device)
    g.manual_seed(int(seed))

    # To estimate r_target accurately without blowing up memory, sample a bounded number
    # of (x, target) pairs. We do this by:
    # - optionally subsampling x rows (uniformly without replacement)
    # - generating many random targets
    # - computing dist^2 in chunks and storing only the sampled dist^2 values on CPU
    #
    # This approximates the quantile of dist^2 over the joint distribution:
    #   x ~ empirical dataset, t ~ Uniform(S^{d-1})
    # while allowing us to "spend more samples" safely.
    mt = int(max(1, num_targets))
    mp = int(max(1, max_pairs))
    sx_arg = int(sample_x)
    if sx_arg > 0:
        sx = min(int(n), sx_arg)
    else:
        # Default: choose sx so that sx * mt <= max_pairs (or full n if already small).
        sx = min(int(n), max(1, mp // mt))

    if sx < int(n):
        idx = torch.randperm(int(n), generator=g, device=x_unit.device)[:sx]
        x_use = x_unit.index_select(0, idx)
    else:
        x_use = x_unit

    ct = int(max(1, chunk_targets))
    total = int(x_use.shape[0]) * mt
    if total > mp:
        # Should not happen due to sx selection, but guard anyway.
        mt = max(1, mp // int(x_use.shape[0]))
        total = int(x_use.shape[0]) * mt

    dist2_cpu = np.empty((total,), dtype=np.float32)
    write = 0
    x_f32 = x_use.to(dtype=torch.float32)
    for j in range(0, mt, ct):
        m = min(ct, mt - j)
        t = torch.randn((m, int(d)), device=x_unit.device, dtype=torch.float32, generator=g)
        t = _unit_normalize_t(t)
        dots = torch.clamp(x_f32 @ t.T, -1.0, 1.0)  # [sx,m]
        dist2 = 2.0 - 2.0 * dots
        block = dist2.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        dist2_cpu[write : write + block.size] = block
        write += block.size

    if write != dist2_cpu.size:
        dist2_cpu = dist2_cpu[:write]

    q = float(np.quantile(dist2_cpu, tr))
    return float(math.sqrt(max(0.0, q)))


@torch.no_grad()
def _bubble_test_rates_torch(
    x_unit: torch.Tensor,
    *,
    num_targets: int,
    radius: float,
    seed: int,
    chunk_targets: int,
) -> np.ndarray:
    n, d = x_unit.shape
    g = torch.Generator(device=x_unit.device)
    g.manual_seed(int(seed))
    rates = np.empty((int(num_targets),), dtype=np.float64)
    ct = int(max(1, chunk_targets))
    r2 = float(radius) * float(radius)
    for j in range(0, int(num_targets), ct):
        m = min(ct, int(num_targets) - j)
        t = torch.randn((m, int(d)), device=x_unit.device, dtype=torch.float32, generator=g)
        t = _unit_normalize_t(t)
        dots = torch.clamp(x_unit.to(dtype=torch.float32) @ t.T, -1.0, 1.0)  # [N,m]
        dist2 = 2.0 - 2.0 * dots
        inside = dist2 <= r2
        rates[j : j + m] = inside.to(dtype=torch.float32).mean(dim=0).detach().cpu().numpy().astype(np.float64)
    return rates


def _device_from_arg(s: str) -> torch.device:
    if s == "cuda":
        return torch.device("cuda")
    if s == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cov_eig_stats(x: np.ndarray) -> tuple[float, float, float, float]:
    cov = np.cov(np.asarray(x, dtype=np.float32), rowvar=False, bias=True).astype(np.float64)
    eigs = np.linalg.eigvalsh(cov).astype(np.float64)
    return float(eigs.min()), float(eigs.max()), float(eigs.mean()), float(eigs.std(ddof=0))


def _write_run_outputs(
    *,
    out_dir: Path,
    setup: str,
    x_unit: torch.Tensor,
    d_full: int,
    d_analyze: int,
    nonces: list[int],
    meta: dict[str, Any],
    bubble_target_rate: float,
    bubble_targets: int,
    bubble_hist_bins: int,
    overlay_ideal: bool,
    radius: float,
    seed: int,
    bubble_chunk_targets: int,
    save_x: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x_np = x_unit.detach().cpu().numpy().astype(np.float32)

    mean_vec = x_np.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))
    cov_eig_min, cov_eig_max, cov_eig_mean, cov_eig_std = _cov_eig_stats(x_np) if d_analyze <= 4096 else (
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    )

    rates = _bubble_test_rates_torch(
        x_unit,
        num_targets=int(bubble_targets),
        radius=float(radius),
        seed=int(seed) + 1337,
        chunk_targets=int(bubble_chunk_targets),
    )

    np.save(out_dir / "bubble_rates.npy", rates.astype(np.float64))
    if save_x:
        np.savez_compressed(
            out_dir / "data.npz",
            x=x_np.astype(np.float32),
            bubble_rates=rates.astype(np.float64),
            bubble_target_rate=float(bubble_target_rate),
            bubble_radius=float(radius),
            setup=str(setup),
            d_full=int(d_full),
            d_analyze=int(d_analyze),
            nonces=np.asarray(nonces, dtype=np.int64),
        )

    plot_bubble_rate_hist(
        out_path=out_dir / "bubble_rate_hist.png",
        rates=rates,
        num_targets=int(bubble_targets),
        radius=float(radius),
        bins=int(bubble_hist_bins),
        overlay_theory=bool(overlay_ideal),
        theory_p=float(bubble_target_rate) if bool(overlay_ideal) else None,
        theory_n=int(x_np.shape[0]) if bool(overlay_ideal) else None,
        title_prefix=str(setup),
    )

    summary = BubbleRunSummary(
        setup=str(setup),
        n=int(x_np.shape[0]),
        d_full=int(d_full),
        d_analyze=int(d_analyze),
        mean_vector_norm=float(mean_norm),
        cov_eig_min=float(cov_eig_min),
        cov_eig_max=float(cov_eig_max),
        cov_eig_mean=float(cov_eig_mean),
        cov_eig_std=float(cov_eig_std),
        bubble_num_targets=int(bubble_targets),
        bubble_target_rate=float(bubble_target_rate),
        bubble_radius=float(radius),
        bubble_rate_mean=float(rates.mean()),
        bubble_rate_std=float(rates.std(ddof=0)),
        bubble_rate_variance=float(rates.var(ddof=0)),
        bubble_rate_min=float(rates.min()),
        bubble_rate_max=float(rates.max()),
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))

    meta_out = dict(meta)
    meta_out.update(
        {
            "setup": str(setup),
            "num_samples_written": int(x_np.shape[0]),
            "d_full": int(d_full),
            "d_analyze": int(d_analyze),
        }
    )
    (out_dir / "meta_effective.json").write_text(json.dumps(meta_out, indent=2))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-raw-dir", type=str, required=True, help="Directory containing raw meta.json + vectors npy")
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for plots + summaries (single run) OR output root (preset mode).",
    )
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-n", type=int, default=0, help="If >0, use only the first max-n samples")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--preset",
        type=str,
        default="none",
        choices=["none", "exp50k_qwen4"],
        help="Run a predefined multi-regime experiment. If not 'none', --out-dir is treated as output root.",
    )

    p.add_argument(
        "--regime",
        type=str,
        default="sf_hh",
        choices=["sf_hh", "haar", "sf_only", "hh_only", "none"],
        help="Which k-space regime to apply after fixed full→k projection.",
    )
    p.add_argument(
        "--num-reflections",
        type=int,
        default=-1,
        help="For hh_only/sf_hh: number of Householder reflections. If -1, uses k.",
    )
    p.add_argument(
        "--num-signflip-rounds",
        type=int,
        default=1,
        help="How many independent sign-flip rounds to apply (uses distinct per-round seeds).",
    )

    # Bubble params
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bubble-targets", type=int, default=5000)
    p.add_argument("--bubble-target-rate", type=float, default=0.01)
    p.add_argument(
        "--bubble-radius-targets",
        type=int,
        default=-1,
        help="Number of random targets used to estimate r_target. If <=0, uses min(bubble_targets, 4096).",
    )
    p.add_argument(
        "--bubble-radius-max-pairs",
        type=int,
        default=50_000_000,
        help="Cap on total (x,target) pairs used for r_target estimation (controls memory/time).",
    )
    p.add_argument(
        "--bubble-radius-sample-x",
        type=int,
        default=-1,
        help="If >0, subsample this many x rows for r_target estimation. If <=0, auto-choose from max-pairs.",
    )
    p.add_argument(
        "--bubble-radius-chunk-targets",
        type=int,
        default=256,
        help="Chunk size (targets per block) for r_target estimation.",
    )
    p.add_argument("--bubble-chunk-targets", type=int, default=512)
    p.add_argument("--bubble-hist-bins", type=int, default=200)
    p.add_argument("--overlay-ideal", action="store_true")
    p.add_argument(
        "--bubble-radius",
        type=float,
        default=-1.0,
        help="If >0, use this fixed radius. If <=0, auto-pick radius by target rate.",
    )
    p.add_argument("--title-prefix", type=str, default="Bubble test rates")
    args = p.parse_args()

    in_dir = Path(args.in_raw_dir)
    out_dir = Path(args.out_dir)

    meta = json.loads((in_dir / "meta.json").read_text())
    vecs = _load_vectors(in_dir, meta).astype(np.float32)

    if int(args.max_n) > 0 and vecs.shape[0] > int(args.max_n):
        vecs = vecs[: int(args.max_n)]

    nonces = meta.get("nonces")
    if not isinstance(nonces, list) or len(nonces) < vecs.shape[0]:
        raise RuntimeError("meta.json must include a `nonces` list at least as long as vectors")
    nonces = nonces[: vecs.shape[0]]

    block_hash = meta.get("block_hash")
    public_key = meta.get("public_key")
    if not (isinstance(block_hash, str) and isinstance(public_key, str)):
        raise RuntimeError("meta.json must include block_hash and public_key")

    device = _device_from_arg(str(args.device))
    x_full = torch.from_numpy(vecs).to(device=device, dtype=torch.float32)
    x_full = unit_normalize(x_full)

    k = int(args.k)
    d_full = int(x_full.shape[1])

    # Bubble radius selection counts.
    rad_targets = int(args.bubble_radius_targets)
    if rad_targets <= 0:
        rad_targets = int(min(int(args.bubble_targets), 4096))

    rad_max_pairs = int(args.bubble_radius_max_pairs)
    rad_sample_x = int(args.bubble_radius_sample_x)
    rad_chunk_targets = int(args.bubble_radius_chunk_targets)

    if str(args.preset) == "exp50k_qwen4":
        # Preset: 4 regimes for exp_50k_raw_qwen, with shared radius across projected regimes.
        out_root = out_dir
        out_root.mkdir(parents=True, exist_ok=True)

        # 1) No projection: full-d + (k signflip rounds) + Householder(k)
        x1 = apply_sign_flips_then_normalize(
            x_full,
            block_hash=str(block_hash),
            public_key=str(public_key),
            nonces=nonces,
            num_rounds=int(k),
        )
        x1 = apply_householder_reflections(
            x1,
            block_hash=str(block_hash),
            public_key=str(public_key),
            nonces=nonces,
            num_reflections=int(k),
        )
        x1 = unit_normalize(x1)
        if float(args.bubble_radius) > 0.0:
            radius_full = float(args.bubble_radius)
        else:
            radius_full = _pick_radius_for_target_rate_torch(
                x1,
                num_targets=int(rad_targets),
                target_rate=float(args.bubble_target_rate),
                seed=int(args.seed) + 9001,
                max_pairs=int(rad_max_pairs),
                sample_x=int(rad_sample_x),
                chunk_targets=int(rad_chunk_targets),
            )
        _write_run_outputs(
            out_dir=out_root / "01_full_sf_k_hh_k",
            setup="01_full_sf(k)_hh(k)",
            x_unit=x1,
            d_full=int(d_full),
            d_analyze=int(d_full),
            nonces=nonces,
            meta=meta,
            bubble_target_rate=float(args.bubble_target_rate),
            bubble_targets=int(args.bubble_targets),
            bubble_hist_bins=int(args.bubble_hist_bins),
            overlay_ideal=bool(args.overlay_ideal),
            radius=float(radius_full),
            seed=int(args.seed) + 9001,
            bubble_chunk_targets=int(args.bubble_chunk_targets),
            save_x=False,
        )

        # Shared projection base.
        xk_base = fixed_project_full_to_k(
            x_full,
            block_hash=str(block_hash),
            public_key=str(public_key),
            k=int(k),
        )
        xk_base = unit_normalize(xk_base)

        # Shared radius for projected regimes (2-4).
        if float(args.bubble_radius) > 0.0:
            radius_k = float(args.bubble_radius)
        else:
            radius_k = _pick_radius_for_target_rate_torch(
                xk_base,
                num_targets=int(rad_targets),
                target_rate=float(args.bubble_target_rate),
                seed=int(args.seed) + 1337,
                max_pairs=int(rad_max_pairs),
                sample_x=int(rad_sample_x),
                chunk_targets=int(rad_chunk_targets),
            )

        # 2) projection + (2 signflip rounds) + Householder(2)
        x2 = apply_sign_flips_then_normalize(
            xk_base,
            block_hash=str(block_hash),
            public_key=str(public_key),
            nonces=nonces,
            num_rounds=2,
        )
        x2 = apply_householder_reflections(
            x2,
            block_hash=str(block_hash),
            public_key=str(public_key),
            nonces=nonces,
            num_reflections=2,
        )
        x2 = unit_normalize(x2)
        _write_run_outputs(
            out_dir=out_root / "02_proj_sf2_hh2",
            setup="02_proj_sf(2)_hh(2)",
            x_unit=x2,
            d_full=int(d_full),
            d_analyze=int(k),
            nonces=nonces,
            meta=meta,
            bubble_target_rate=float(args.bubble_target_rate),
            bubble_targets=int(args.bubble_targets),
            bubble_hist_bins=int(args.bubble_hist_bins),
            overlay_ideal=bool(args.overlay_ideal),
            radius=float(radius_k),
            seed=int(args.seed),
            bubble_chunk_targets=int(args.bubble_chunk_targets),
            save_x=True,
        )

        # 3) projection + (k signflip rounds) + Householder(k)
        x3 = apply_sign_flips_then_normalize(
            xk_base,
            block_hash=str(block_hash),
            public_key=str(public_key),
            nonces=nonces,
            num_rounds=int(k),
        )
        x3 = apply_householder_reflections(
            x3,
            block_hash=str(block_hash),
            public_key=str(public_key),
            nonces=nonces,
            num_reflections=int(k),
        )
        x3 = unit_normalize(x3)
        _write_run_outputs(
            out_dir=out_root / "03_proj_sf_k_hh_k",
            setup="03_proj_sf(k)_hh(k)",
            x_unit=x3,
            d_full=int(d_full),
            d_analyze=int(k),
            nonces=nonces,
            meta=meta,
            bubble_target_rate=float(args.bubble_target_rate),
            bubble_targets=int(args.bubble_targets),
            bubble_hist_bins=int(args.bubble_hist_bins),
            overlay_ideal=bool(args.overlay_ideal),
            radius=float(radius_k),
            seed=int(args.seed),
            bubble_chunk_targets=int(args.bubble_chunk_targets),
            save_x=True,
        )

        # 4) projection + Haar (no sign flips)
        x4 = haar_rotate_k(xk_base, block_hash=str(block_hash), public_key=str(public_key), nonces=nonces)
        x4 = unit_normalize(x4)
        _write_run_outputs(
            out_dir=out_root / "04_proj_haar",
            setup="04_proj_haar",
            x_unit=x4,
            d_full=int(d_full),
            d_analyze=int(k),
            nonces=nonces,
            meta=meta,
            bubble_target_rate=float(args.bubble_target_rate),
            bubble_targets=int(args.bubble_targets),
            bubble_hist_bins=int(args.bubble_hist_bins),
            overlay_ideal=bool(args.overlay_ideal),
            radius=float(radius_k),
            seed=int(args.seed),
            bubble_chunk_targets=int(args.bubble_chunk_targets),
            save_x=True,
        )

        # Write shared radius used for regimes 2-4.
        (out_root / "shared_radius_projected.json").write_text(
            json.dumps(
                {
                    "bubble_target_rate": float(args.bubble_target_rate),
                    "bubble_radius": float(radius_k),
                    "bubble_targets": int(args.bubble_targets),
                    "bubble_hist_bins": int(args.bubble_hist_bins),
                    "note": "radius shared across projected regimes 02/03/04",
                },
                indent=2,
            )
        )
        return 0

    # Single-run mode: fixed projection + chosen k-space regime (backwards compatible).
    out_dir.mkdir(parents=True, exist_ok=True)
    xk_base = fixed_project_full_to_k(
        x_full,
        block_hash=str(block_hash),
        public_key=str(public_key),
        k=k,
    )
    xk_base = unit_normalize(xk_base)

    regime = str(args.regime)
    nref = int(args.num_reflections)
    if nref <= 0:
        nref = k
    nsf = int(args.num_signflip_rounds)
    if nsf <= 0:
        raise ValueError("--num-signflip-rounds must be positive")

    setup_name = f"fixedproj_{regime}"
    if regime in ("sf_hh", "hh_only"):
        setup_name += f"_r{nref}"
    if regime in ("sf_hh", "sf_only"):
        setup_name += f"_sf{nsf}"

    xk = xk_base
    if regime == "sf_hh":
        xk = apply_sign_flips_then_normalize(
            xk,
            block_hash=str(block_hash),
            public_key=str(public_key),
            nonces=nonces,
            num_rounds=nsf,
        )
        xk = apply_householder_reflections(
            xk, block_hash=str(block_hash), public_key=str(public_key), nonces=nonces, num_reflections=nref
        )
    elif regime == "haar":
        xk = haar_rotate_k(xk, block_hash=str(block_hash), public_key=str(public_key), nonces=nonces)
    elif regime == "sf_only":
        xk = apply_sign_flips_then_normalize(
            xk,
            block_hash=str(block_hash),
            public_key=str(public_key),
            nonces=nonces,
            num_rounds=nsf,
        )
    elif regime == "hh_only":
        xk = apply_householder_reflections(
            xk, block_hash=str(block_hash), public_key=str(public_key), nonces=nonces, num_reflections=nref
        )
    elif regime == "none":
        pass
    else:
        raise RuntimeError(f"Unknown regime: {regime}")

    xk = unit_normalize(xk)

    if float(args.bubble_radius) > 0.0:
        radius = float(args.bubble_radius)
    else:
        radius = _pick_radius_for_target_rate_torch(
            xk,
            num_targets=int(rad_targets),
            target_rate=float(args.bubble_target_rate),
            seed=int(args.seed) + 1337,
            max_pairs=int(rad_max_pairs),
            sample_x=int(rad_sample_x),
            chunk_targets=int(rad_chunk_targets),
        )
    _write_run_outputs(
        out_dir=out_dir,
        setup=str(args.title_prefix) if str(args.title_prefix) else setup_name,
        x_unit=xk,
        d_full=int(d_full),
        d_analyze=int(k),
        nonces=nonces,
        meta=meta,
        bubble_target_rate=float(args.bubble_target_rate),
        bubble_targets=int(args.bubble_targets),
        bubble_hist_bins=int(args.bubble_hist_bins),
        overlay_ideal=bool(args.overlay_ideal),
        radius=float(radius),
        seed=int(args.seed),
        bubble_chunk_targets=int(args.bubble_chunk_targets),
        save_x=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


