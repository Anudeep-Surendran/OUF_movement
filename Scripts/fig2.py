"""
Mean encounter rate as a function of time for the OUF movement model 
keeping stationary mean squared speed fixed while changing tau_v.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MultipleLocator, FormatStrFormatter

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
TAU_Z = 10.341                  # home-range crossing time                [h]
G_REF = 0.131                   # noise amplitude at TAU_V_REF, tapir fit [km^2/h]
TAU_V_REF = 2.0                 # anchor for the fixed-speed constraint   [h]
GAMMA = 1.0                     # total reactivity                        [km^2/h]
TAU_V_LIST = (2.0, 4.0, 6.0)
R_LAMBDA = {"A": 0.7, "B": 3.0} # home-range center separation            [km]
T_MAX = 20.0

# Stationary variance of the velocity per coordinate, held fixed across the
# sweep.  Anchored so the constraint agrees with g = G_REF at TAU_V_REF.
SIGMA_ZDOT2 = (G_REF / (2.0 * TAU_V_REF)) * TAU_Z / (TAU_Z + TAU_V_REF)


def g_of_tau_v(tau_v, sd2=SIGMA_ZDOT2, tau_z=TAU_Z):
    """Noise amplitude at fixed speed"""
    s_v = sd2 * (tau_z + tau_v) / tau_z
    return 2.0 * tau_v * s_v


# --------------------------------------------------------------------------
# Moments of the OUF location
# --------------------------------------------------------------------------
def sigma_z2(t, tau_v, tau_z=TAU_Z, sd2=SIGMA_ZDOT2):
    """Variance of the OUF location in one spatial coordinate under the initial conditions
    in Section 2.1 of the manuscript"""

    t = np.asarray(t, dtype=float)
    if abs(tau_z - tau_v) < 1e-12:
        raise ValueError("Singularity at tau_z = tau_v.")
    g = g_of_tau_v(tau_v, sd2, tau_z)
    kappa = tau_z * tau_v / (tau_z - tau_v)
    return g * tau_z / (2.0 * (tau_z + tau_v)) * (
        tau_z * (1.0 - np.exp(-2.0 * t / tau_z))
        - 2.0 * kappa * (1.0 - np.exp(-t / kappa)) * np.exp(-2.0 * t / tau_z)
    )


# --------------------------------------------------------------------------
# Mean encounter rate.
# --------------------------------------------------------------------------
def encounter_rate(t, tau_v, r_lambda, tau_z=TAU_Z, sd2=SIGMA_ZDOT2, gamma=GAMMA):
    """Ebar(t) for a pair of identical OUF individuals.
    Evaluated in log space so that the t -> 0 limit, where sigma_r^2 -> 0 and
    the prefactor diverges while the exponential vanishes faster, returns 0.
    """
    s_r2 = 2.0 * sigma_z2(t, tau_v, tau_z, sd2)
    out = np.zeros_like(np.asarray(s_r2, dtype=float))
    ok = s_r2 > 0
    out[ok] = np.exp(np.log(gamma) - np.log(2.0 * np.pi * s_r2[ok])
                     - r_lambda**2 / (2.0 * s_r2[ok]))
    return out

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def configure_fonts(base=11):
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    })
    matplotlib.rcParams.update({
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "xtick.labelsize": base, "ytick.labelsize": base,
        "legend.fontsize": base, "legend.title_fontsize": base,
    })


def _tau_v_legend(ax, colors, **kw):
    handles = [Line2D([0], [0], marker="s", linestyle="None", markersize=6,
                      markerfacecolor=c, markeredgecolor=c) for c in colors]
    labels = [rf"${tv:g}$" for tv in TAU_V_LIST]
    leg = ax.legend(handles, labels, title=r"Directional persistence, $\tau_v$",
                    frameon=False, fontsize=9, handlelength=1.0,
                    handletextpad=0.4, labelspacing=0.25, columnspacing=0.3,
                    borderaxespad=0.0, **kw)
    leg.get_title().set_fontsize(10)
    return leg


def make_figure():
    """Fig. 2 -- Mean encounter rate as a function of time in two home-range configurations"""
    t = np.linspace(0.0, T_MAX, 4000)
    colors = plt.cm.Reds(np.linspace(0.38, 1.0, len(TAU_V_LIST)))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for panel, ax in zip(("A", "B"), axes):
        rl = R_LAMBDA[panel]
        for tau_v, c in zip(TAU_V_LIST, colors):
            ax.plot(t, encounter_rate(t, tau_v, rl), color=c, lw=1.7)
        ax.set_xlim(0, T_MAX)
        ax.set_ylim(0, 1.10 * max(encounter_rate(t, tv, rl).max()
                                  for tv in TAU_V_LIST))
        ax.set_xlabel(r"Time, $t$")
        ax.set_ylabel(r"Mean encounter rate, $\bar{\mathcal{E}}(t)$")
        ax.text(-0.30, 1.13, panel, transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top")
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.get_offset_text().set_fontsize(11)

    _tau_v_legend(axes[1], colors, loc="upper left", ncol=3)

    # Inset on panel A: zoom on the short-time behavior
    ax_in = axes[0].inset_axes([0.50, 0.66, 0.46, 0.28])
    ax_in.set_box_aspect(0.56)  # width:height ~ 1.8:1
    for tau_v, c in zip(TAU_V_LIST, colors):
        ax_in.plot(t, encounter_rate(t, tau_v, R_LAMBDA["A"]), color=c, lw=1.2)
    ax_in.set_xlim(0.5, 1.5)
    ax_in.set_ylim(0, 0.15)
    ax_in.xaxis.set_major_locator(MultipleLocator(0.5))
    ax_in.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_in.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_in.tick_params(labelsize=8, pad=1.5)

    fig.tight_layout()
    fig.savefig("fig2.pdf")
    print("wrote fig2.pdf")
    return fig

# --------------------------------------------------------------------------
def main():
    configure_fonts(base=11)
    make_figure()

if __name__ == "__main__":
    main()
