"""Reproduce the encounter-rate heatmap in Fig. 3 and the regime-separation
threshold for the OUF movement model at fixed stationary mean squared speed.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, FixedLocator


def shifted_cmap(cmap_name, norm, midpoint=0.0, n=256):
    """Return a diverging colormap with white at zero."""
    base = plt.get_cmap(cmap_name)
    frac0 = float(norm(midpoint))
    positions = np.linspace(0.0, 1.0, n)
    src = np.empty_like(positions)
    lower = positions <= frac0
    src[lower] = positions[lower] / frac0 * 0.5
    src[~lower] = 0.5 + (positions[~lower] - frac0) / (1.0 - frac0) * 0.5
    return LinearSegmentedColormap.from_list(f"{cmap_name}_shifted", base(src), N=n)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
})

# Model and grid parameters
TAU_Z = 10.341      # OUF home-range crossing time [h]
G_REF = 0.131       # Reference noise amplitude [km^2/h]
TAU_V_REF = 2.0     # Reference directional-persistence time scale [h]
H = 0.05            # Finite-difference step for the rate derivative
TAU_V_MIN = 1.0     # Minimum directional-persistence time in the grid [h]
TAU_V_MAX = 9.0     # Maximum directional-persistence time in the grid [h]
TAU_V_STEP = 0.001  # Directional-persistence grid spacing [h]
TAU_V_THRESHOLD_STEP = 0.01  # Threshold-curve spacing [h]
R_LAMBDA_MIN = 1.0  # Minimum home-range separation in the grid [km]
R_LAMBDA_MAX = 3.5  # Maximum home-range separation in the grid [km]
R_LAMBDA_STEP = 0.01  # Home-range separation grid spacing [km]
TAU_V_PLOT_MAX = 7.0  # Upper x-axis limit [h]
R_LAMBDA_PLOT_MAX = 3.0  # Upper y-axis limit [km]
COLOR_MIN = -0.03    # Lower color-scale limit
COLOR_MAX = 0.04     # Upper color-scale limit

# Stationary velocity variance per coordinate.
SIGMA_ZDOT2 = (G_REF / (2.0 * TAU_V_REF)) * TAU_Z / (TAU_Z + TAU_V_REF)

def sz2(tauv, tau_z=TAU_Z):
    """Stationary location variance at fixed speed"""
    return SIGMA_ZDOT2 * tau_z * tauv


def encrate(s2, rl):
    """Mean encounter rate; 0 where s2 <= 0 (degenerate/clipped variance)."""
    s2b, rlb = np.broadcast_arrays(s2, rl)
    out = np.zeros_like(s2b, dtype=float)
    good = s2b > 0
    out[good] = np.exp(-rlb[good]**2 / (4.0 * s2b[good])) / (4.0 * np.pi * s2b[good])
    return out


def _frange(start, stop, step):
    """Return a range with floating-point values and an inclusive stop rule."""
    n = int((stop - start + step) / step)
    return start + np.arange(n) * step


def make_figure():
    tauv_u = _frange(TAU_V_MIN, TAU_V_MAX, TAU_V_STEP)
    Rl_u = _frange(R_LAMBDA_MIN, R_LAMBDA_MAX, R_LAMBDA_STEP)

    TAUV, RL = np.meshgrid(tauv_u, Rl_u)
    Z = (encrate(sz2(TAUV + H), RL) - encrate(sz2(TAUV - H), RL)) / (2.0 * H)

    tauv_thr = _frange(TAU_V_MIN, TAU_V_MAX, TAU_V_THRESHOLD_STEP)
    Rl_thr = np.sqrt(4.0 * sz2(tauv_thr))

    norm = Normalize(vmin=COLOR_MIN, vmax=COLOR_MAX)
    cmap = shifted_cmap("RdBu_r", norm)

    fig, ax = plt.subplots(figsize=(6.5, 4.9))
    im = ax.pcolormesh(tauv_u, Rl_u, Z, cmap=cmap, norm=norm, shading="auto")

    ax.plot(
        tauv_thr, Rl_thr, color="k", linewidth=1.5, linestyle="--",
        label=r"$R_\lambda = \sqrt{4\sigma_z^2}$",
    )

    ax.set_xlim(TAU_V_MIN, TAU_V_PLOT_MAX)
    ax.set_ylim(R_LAMBDA_MIN, R_LAMBDA_PLOT_MAX)
    ax.set_xlabel(r"Time scale of directional persistence, $\tau_v$")
    ax.set_ylabel(r"Distance between home-range centers, $R_\lambda$", labelpad=10)
    ax.xaxis.set_major_locator(FixedLocator([1, 3, 5, 7]))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.legend(loc="upper left")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        r"$\partial \overline{\mathcal{E}} / \partial \tau_v \ (\times 10^{-2})$"
    )
    tick_step = 1e-2
    n_lo = int(round(COLOR_MIN / tick_step))
    n_hi = int(round(COLOR_MAX / tick_step))
    ticks = np.round(np.arange(n_lo, n_hi + 1) * tick_step, 12)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{round(t * 1e2):g}" for t in ticks])

    fig.tight_layout()
    fig.savefig("fig3.png", dpi=300)
    print("wrote fig3.png")


if __name__ == "__main__":
    make_figure()