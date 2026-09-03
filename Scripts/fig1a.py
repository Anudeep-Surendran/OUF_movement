"""
Script to produce Fig. 1A (variance of the OUF position as a function of time).

Lines   analytical result
Symbols Euler-Maruyama simulation of OUF SDES using Euler-Maruyama.

Initial conditions from Section 2.1 in the main text:
    position   deterministic, at the home-range center,  sigma_z^2(0) = 0
    velocity   drawn from its stationary distribution,   mu_v(0) = 0,
               sigma_v^2(0) = S_v = g / (2 tau_v)
    covariance sigma_zv(0) = 0, forced by Cauchy-Schwarz once sigma_z^2(0) = 0

Parameters are those quoted in the Fig. 1 caption,
    tau_z = 6.855 h, tau_v = 0.486 h, S_z = sigma_z^2(t -> inf) = 0.298 km^2

The noise amplitude is obtained by inverting the expression for S_z
    g = 2 S_z (tau_z + tau_v) / tau_z^2.

    
Usage:
    python fig1a.py
    python fig1a.py --nreal 10000      # faster, coarser symbols
    python fig1a.py --dt 0.002         # reduce short-time discretisation bias
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator, NullLocator

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
})

# --------------------------------------------------------------------------
# Color palette
# --------------------------------------------------------------------------
FIG_COLOR = "rosybrown"  # #BC8F8F

# --------------------------------------------------------------------------
# Fixed plot-box size for the axes area
# --------------------------------------------------------------------------
AX_WIDTH_IN = 7.37 / 2.54
AX_HEIGHT_IN = 6.27 / 2.54
MARGIN_LEFT_IN = 0.62
MARGIN_RIGHT_IN = 0.08
MARGIN_BOTTOM_IN = 0.48
MARGIN_TOP_IN = 0.30

# --------------------------------------------------------------------------
# Movement parameters
# --------------------------------------------------------------------------
TAU_Z = 6.855          # home-range crossing time                 [h]
TAU_V = 0.486          # persistence time scale                   [h]
S_Z = 0.298            # S_z = sigma_z^2(t -> inf)                [km^2]
LAMBDA = 0.0           # home-range center, one coordinate        [km]

G = 2.0 * S_Z * (TAU_Z + TAU_V) / TAU_Z**2   #                    [km^2/h]
S_V = G / (2.0 * TAU_V)                      #          [km^2/h^2]


# --------------------------------------------------------------------------
# Analytical position variance
# --------------------------------------------------------------------------
def sigma_z2(t, tau_z=TAU_Z, tau_v=TAU_V, g=G):
    """Variance of the OUF position in one spatial coordinate.

        sigma_z^2(t) = g tau_z / [2 (tau_z + tau_v)]
                       * { tau_z (1 - e^{-2t/tau_z})
                           - 2 kappa (1 - e^{-t/kappa}) e^{-2t/tau_z} }

    This already assumes the initial conditions of Sec. 2.1, so it takes no
    initial-moment arguments.
    """
    t = np.asarray(t, dtype=float)
    if abs(tau_z - tau_v) < 1e-12:
        raise ValueError("sigma_z^2 is singular at tau_z == tau_v.")
    kappa = tau_z * tau_v / (tau_z - tau_v)
    return g * tau_z / (2.0 * (tau_z + tau_v)) * (
        tau_z * (1.0 - np.exp(-2.0 * t / tau_z))
        - 2.0 * kappa * (1.0 - np.exp(-t / kappa)) * np.exp(-2.0 * t / tau_z)
    )


# --------------------------------------------------------------------------
# Euler-Maruyama integration of Eqs. (2.1)-(2.2), Eqs. (C.1)-(C.2)
# --------------------------------------------------------------------------
def simulate(t_end, dt=0.01, n_real=1000000, tau_z=TAU_Z, tau_v=TAU_V, g=G,
             lam=LAMBDA, seed=0, n_save=200):
    """Return (t_saved, var_saved) from n_real realisations of the OUF SDEs.

    Individuals start at the home-range center with velocity drawn from its
    stationary distribution, matching the initial conditions of Sec. 2.1.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(t_end / dt))

    z = np.full((n_real, 2), lam, dtype=float)
    v = rng.normal(0.0, np.sqrt(g / (2.0 * tau_v)), size=(n_real, 2))

    save_at = set(np.unique(np.linspace(0, n_steps, n_save, dtype=int)).tolist())
    t_saved, var_saved = [], []

    noise_amp = np.sqrt(g) / tau_v * np.sqrt(dt)
    for n in range(n_steps + 1):
        if n in save_at:
            t_saved.append(n * dt)
            var_saved.append(z.var(axis=0, ddof=1).mean())
        if n == n_steps:
            break
        # Eqs. (C.1)-(C.2); z is updated with the pre-update velocity, as written
        z += -dt / tau_z * (z - lam) + dt * v
        v += -dt / tau_v * v + noise_amp * rng.standard_normal((n_real, 2))

    return np.asarray(t_saved), np.asarray(var_saved)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def make_figure(n_real=100_000, dt=0.01, seed=0):
    t_line = np.linspace(0.0, 30.0, 2000)
    t_sim, var_sim = simulate(30.0, dt=dt, n_real=n_real, seed=seed, n_save=200)

    fig_width_in = AX_WIDTH_IN + MARGIN_LEFT_IN + MARGIN_RIGHT_IN
    fig_height_in = AX_HEIGHT_IN + MARGIN_BOTTOM_IN + MARGIN_TOP_IN
    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    ax = fig.add_axes([
        MARGIN_LEFT_IN / fig_width_in,
        MARGIN_BOTTOM_IN / fig_height_in,
        AX_WIDTH_IN / fig_width_in,
        AX_HEIGHT_IN / fig_height_in,
    ])

    ax.plot(t_line, sigma_z2(t_line), "-", color="black", lw=1.6,
            label="Analytical", zorder=2)
    ax.plot(t_sim[::8], var_sim[::8], "o", ms=5.0, mfc=FIG_COLOR, mec="black",
            mew=0.8, label="Simulations", zorder=3)

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 0.32)
    ax.set_xlabel("time [h]")
    ax.set_ylabel(r"$\sigma_z^2(t)$  [km$^2$]")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.legend(frameon=False, fontsize=11, loc="upper right",
              bbox_to_anchor=(0.97, 0.86))
    ax.text(-0.18, 1.04, "A", transform=ax.transAxes, fontsize=11,
            fontweight="bold")

    # inset: short-time behaviour, as in the published panel, on log-log axes
    axin = ax.inset_axes([0.50, 0.16, 0.44, 0.42])
    t_in = np.logspace(-2.0, np.log10(6.0), 400)
    axin.plot(t_in, sigma_z2(t_in), "-", color="black", lw=1.4, zorder=2)

    # ballistic reference, sigma_z^2 ~ t^2.
    t_guide2 = np.array([1e-2, 1.0])
    axin.plot(t_guide2, 3.0 * S_V * t_guide2**2, "--", color="0.5", lw=1.0,
              zorder=1)
    axin.text(0.013, 8e-4, r"$\propto t^2$", fontsize=8, color="0.5")

    axin.set_xscale("log")
    axin.set_yscale("log")
    axin.set_xlim(1e-2, 6.0)
    axin.set_ylim(1e-5, 0.3)
    axin.xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
    axin.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    axin.xaxis.set_minor_locator(NullLocator())
    axin.yaxis.set_minor_locator(NullLocator())
    axin.tick_params(labelsize=9, pad=1.5)

    outfile = "fig1a.pdf"
    fig.savefig(outfile, dpi=300)
    print(f"wrote {outfile}")
    return t_sim, var_sim


# --------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nreal", type=int, default=100_000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    assert abs(float(sigma_z2(1e6)) - S_Z) <= 1e-12 * S_Z
    # short-time ballistic limit, sigma_z^2 -> S_v t^2
    assert abs(float(sigma_z2(1e-4)) / 1e-8 - S_V) <= 1e-3 * S_V
    print("Eqs. (2.6), (2.7) and (2.8) are mutually consistent")

    print(f"tau_z = {TAU_Z} h, tau_v = {TAU_V} h, S_z = {S_Z} km^2")
    print(f"  recovered g = {G:.6f} km^2/h,  S_v = g/(2 tau_v) = {S_V:.6f} km^2/h^2")
    print(f"  mean 2D speed sqrt(pi S_v/2) = {np.sqrt(np.pi * S_V / 2):.4f} km/h"
          f" = {np.sqrt(np.pi * S_V / 2) * 24:.1f} km/d")

    t_sim, var_sim = make_figure(n_real=args.nreal, dt=args.dt, seed=args.seed)