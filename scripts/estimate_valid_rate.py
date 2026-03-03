#!/usr/bin/env python3
"""
Estimate valid rate distribution for random unit vectors on K-dimensional sphere.

Useful for calibrating r_target values in PoC - understanding the theoretical
distance distribution before dealing with model-specific behavior.

Math: For random unit vectors x, y on S^{K-1}:
  - d = ||x - y|| = sqrt(2 - 2*cos(theta))
  - Mean distance -> sqrt(2) as K -> infinity
  - Distribution tightens with increasing K
"""

import argparse

import numpy as np


def generate_unit_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def compute_distances(vectors: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors - target, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Estimate valid rate in K-dimensional space")
    parser.add_argument("-n", "--n-samples", type=int, default=100_000, help="Number of samples")
    parser.add_argument("-d", "--dim", type=int, default=8192, help="Dimension K")
    parser.add_argument("-p", "--percentiles", type=int, nargs="+", default=[1, 5, 10, 50, 90],
                        help="Percentiles to compute")
    parser.add_argument("-r", "--r-target", type=float, default=None,
                        help="Compute valid rate at this r_target")
    parser.add_argument("-t", "--target-rate", type=float, default=None,
                        help="Target valid rate %% (e.g., 20 for 20%%) - outputs r_target")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Generating {args.n_samples:,} unit vectors in {args.dim}D...")
    vectors = generate_unit_vectors(args.n_samples, args.dim, args.seed)
    target = generate_unit_vectors(1, args.dim, args.seed + 1)[0]
    distances = compute_distances(vectors, target)

    print(f"\nDistance Distribution (K={args.dim}):")
    print(f"  Mean:       {distances.mean():.6f}")
    print(f"  Std:        {distances.std():.6f}")
    print(f"  Min:        {distances.min():.6f}")
    print(f"  Max:        {distances.max():.6f}")
    print(f"  Theoretical mean (sqrt(2)): {np.sqrt(2):.6f}")

    print("\nPercentiles:")
    for p in args.percentiles:
        r = np.percentile(distances, p)
        print(f"  p{p:02d}: {r:.6f}  (r_target for {p}% valid)")

    if args.r_target is not None:
        valid_rate = (distances < args.r_target).mean() * 100
        print(f"\nValid rate at r_target={args.r_target:.6f}: {valid_rate:.2f}%")

    if args.target_rate is not None:
        r = np.percentile(distances, args.target_rate)
        print(f"\nr_target for {args.target_rate:.1f}% valid rate: {r:.6f}")


if __name__ == "__main__":
    main()

