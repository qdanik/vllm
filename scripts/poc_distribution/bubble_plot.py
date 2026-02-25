from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_HAVE_MPL = False
try:  # pragma: no cover
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    _HAVE_MPL = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore


def plot_bubble_rate_hist(
    *,
    out_path: Path,
    rates: np.ndarray,
    num_targets: int,
    radius: float,
    bins: int,
    overlay_theory: bool,
    theory_p: float | None,
    theory_n: int | None,
    title_prefix: str = "Bubble test rates",
) -> None:
    """Plot histogram of bubble hit rates with optional Normal-theory overlay.

    Notes:
    - We never clip outliers: heavy tails are the signal.
    - If overlay_theory is True and theory_p/theory_n are provided, we overlay a
      Normal(μ=p, σ=sqrt(p(1-p)/N)) curve, corresponding to Binomial(N,p)/N.
    """
    if not _HAVE_MPL or plt is None:
        return

    rs = np.asarray(rates, dtype=np.float64)
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return

    rmin = float(rs.min())
    rmax = float(rs.max())
    rmean = float(rs.mean())
    rstd = float(rs.std(ddof=0))

    # Keep narrow plots readable, but never clip true outliers.
    if rstd > 0:
        lo = max(0.0, min(rmin, rmean - 4.0 * rstd))
        hi = min(1.0, max(rmax, rmean + 4.0 * rstd))
    else:
        lo = max(0.0, rmin - 0.05)
        hi = min(1.0, rmax + 0.05)

    if hi - lo < 0.05:
        mid = 0.5 * (lo + hi)
        lo = max(0.0, mid - 0.025)
        hi = min(1.0, mid + 0.025)

    fig = plt.figure(figsize=(9, 4.2), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(rs, bins=int(bins), range=(lo, hi), density=True, alpha=0.80, color="#4A78C2")

    if overlay_theory and theory_p is not None and theory_n is not None and int(theory_n) > 0:
        p = float(theory_p)
        n = int(theory_n)
        var = p * (1.0 - p) / float(n)
        if var > 0:
            sd = math.sqrt(var)
            grid = np.linspace(lo, hi, 800, dtype=np.float64)
            pdf = (1.0 / (sd * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((grid - p) / sd) ** 2)
            ax.plot(
                grid,
                pdf,
                color="#C83A3A",
                linewidth=2.0,
                label=f"Ideal Normal(μ={p:.3g}, σ={sd:.3g})",
            )
            ax.legend(loc="upper right")

    ax.set_title(f"{title_prefix} (M={num_targets}, radius={radius})")
    ax.set_xlabel("rate")
    ax.set_ylabel("density")
    ax.set_xlim(lo, hi)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


