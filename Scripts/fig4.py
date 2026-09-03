"""Reproduce Fig. 4, comparing OUF and IOU movement and encounter statistics."""

import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

# Model and figure parameters
TAU_Z = 10.341                  # home-range crossing time                 [h]
TAU_V = 2.0                     # persistence time scale                   [h]
S_V = 0.131 / (2.0 * 2.0)       # stationary velocity variance             [km^2/h^2]
GAMMA = 1.0                     # total reactivity                         [km^2/h]
D0 = 0.4                        # separation of the release points          [km]
R_LIST = (0.0, 0.7, 1.5, 3.0)   # home-range center separations             [km]


# --------------------------------------------------------------------------
# Position variances
# --------------------------------------------------------------------------
def sigma_z2(t, tau_v=TAU_V, tau_z=TAU_Z, s_v=S_V):
    """OUF position variance one spatial coordinate."""
    t = np.asarray(t, dtype=float)
    g = 2.0 * s_v * tau_v
    kappa = tau_z * tau_v / (tau_z - tau_v)
    return g * tau_z / (2.0 * (tau_z + tau_v)) * (
        tau_z * (1.0 - np.exp(-2.0 * t / tau_z))
        - 2.0 * kappa * (1.0 - np.exp(-t / kappa)) * np.exp(-2.0 * t / tau_z)
    )


def sigma_z2_iou(t, tau_v=TAU_V, s_v=S_V):
    """IOU position variance"""
    t = np.asarray(t, dtype=float)
    return 2.0 * s_v * tau_v * (t - tau_v * (1.0 - np.exp(-t / tau_v)))


# --------------------------------------------------------------------------
# Squared separation of the mean positions
# --------------------------------------------------------------------------
def lambda2_ouf(t, r_lambda, d0=D0, tau_z=TAU_Z):
    t = np.asarray(t, dtype=float)
    e = np.exp(-t / tau_z)
    return (d0 * e + r_lambda * (1.0 - e))**2


def lambda2_iou(t, d0=D0):
    return np.full_like(np.asarray(t, dtype=float), d0**2)


# --------------------------------------------------------------------------
# Encounter rate and accumulated encounters
# --------------------------------------------------------------------------
def encounter_rate(s2z, lam2, gamma=GAMMA):
    """Ebar for identical individuals, so that sigma_r^2 = 2 sigma_z^2."""
    s_r2 = 2.0 * np.asarray(s2z, dtype=float)
    out = np.zeros_like(s_r2)
    ok = s_r2 > 0
    out[ok] = np.exp(np.log(gamma) - np.log(2.0 * np.pi * s_r2[ok])
                     - np.asarray(lam2, dtype=float)[ok] / (2.0 * s_r2[ok]))
    return out


def cumulative(y, t):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def configure_fonts(usetex=True, base=11):
    if usetex:
        matplotlib.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        })
    else:
        matplotlib.rcParams.update({
            "text.usetex": False, "font.family": "serif",
            "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
        })
    matplotlib.rcParams.update({
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "xtick.labelsize": base, "ytick.labelsize": base,
        "legend.fontsize": base, "legend.title_fontsize": base,
    })


C_IOU = "0.15"
C_OUF = "#B2182B"
C_BAND = "0.88"
T_LO, T_HI = 0.3, 40.0
AGREE = 0.10          # tolerance defining the "indistinguishable" window


def agreement_time(tol=AGREE):
    """Last time at which the two position variances still agree to within tol."""
    t = np.linspace(1e-4, T_HI, 400000)
    rel = np.abs(sigma_z2_iou(t) - sigma_z2(t)) / sigma_z2(t)
    return float(t[int(np.argmax(rel > tol))])


def make_figure():
    colors = plt.cm.Blues(np.linspace(0.42, 1.0, len(R_LIST)))
    t_agree = agreement_time()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))

    ax = axes[0]
    ta = np.logspace(-2, 2, 3000)
    ax.axvspan(1e-2, t_agree, color=C_BAND, lw=0, zorder=0)
    ax.loglog(ta, sigma_z2(ta), "-", color=C_OUF, lw=1.8, label="OUF")
    ax.loglog(ta, sigma_z2_iou(ta), "--", color=C_IOU, lw=1.6, label="IOU")
    for tau, name in ((TAU_V, r"$\tau_v$"), (TAU_Z, r"$\tau_z$")):
        ax.axvline(tau, ls=":", color="0.55", lw=1.0, zorder=1)
        ax.text(tau * 1.3, 5.0, name, fontsize=10, color="0.4", va="center")
    pc = r"\%" if matplotlib.rcParams["text.usetex"] else r"%"
    ax.text(1.4e-2, 5.0, rf"$\sigma_z^2$ within {100 * AGREE:.0f}{pc}",
            fontsize=9, color="0.35", va="center")
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(1e-6, 20)
    ax.set_xlabel(r"Time, $t$")
    ax.set_ylabel(r"$\sigma_z^2(t)$")
    ax.legend(frameon=False, loc="lower right")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda y, pos: (rf"$10^{{{round(np.log10(y))}}}$"
                        if round(np.log10(y)) % 2 == 0 else "")))

    ax = axes[1]
    tb = np.logspace(np.log10(T_LO), np.log10(T_HI), 3000)
    ax.axvspan(T_LO, t_agree, color=C_BAND, lw=0, zorder=0)
    h_iou, = ax.semilogx(tb, encounter_rate(sigma_z2_iou(tb), lambda2_iou(tb)),
                         "--", color=C_IOU, lw=1.6, label="IOU", zorder=5)
    handles = []
    for r, c in zip(R_LIST, colors):
        h, = ax.semilogx(tb, encounter_rate(sigma_z2(tb), lambda2_ouf(tb, r)),
                         "-", color=c, lw=1.7, label=rf"${r:g}$")
        handles.append(h)
    ax.set_xlim(T_LO, T_HI)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Time, $t$")
    ax.set_ylabel(r"Mean encounter rate, $\bar{\mathcal{E}}(t)$")
    leg = ax.legend(handles=handles, title=r"OUF, $R_\lambda$", frameon=False,
                    loc="upper right", handlelength=1.3, labelspacing=0.25,
                    ncol=2, columnspacing=1.2, borderaxespad=0.2)
    leg.get_title().set_fontsize(11)
    ax.add_artist(leg)
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(ax.transAxes.inverted())
    ax.legend(handles=[h_iou], frameon=False, loc="upper center",
              bbox_to_anchor=(0.5 * (bbox.x0 + bbox.x1), bbox.y0),
              handlelength=1.6, borderaxespad=0.2)

    ax = axes[2]
    tc = np.linspace(0.0, T_HI, 40001)
    n_iou = cumulative(encounter_rate(sigma_z2_iou(tc), lambda2_iou(tc)), tc)
    ax.axvspan(T_LO, t_agree, color=C_BAND, lw=0, zorder=0)
    for r, c in zip(R_LIST, colors):
        n_ouf = cumulative(encounter_rate(sigma_z2(tc), lambda2_ouf(tc, r)), tc)
        m = (tc >= T_LO) & (n_iou > 0)
        ax.loglog(tc[m], n_ouf[m] / n_iou[m], "-", color=c, lw=1.8, zorder=3)
    ax.axhline(1.0, ls="--", color=C_IOU, lw=1.4, zorder=2)
    for tau in (TAU_V, TAU_Z):
        ax.axvline(tau, ls=":", color="0.55", lw=1.0, zorder=1)
    ax.set_xlim(T_LO, T_HI)
    ax.set_ylim(5e-3, 5.0)
    ax.set_xlabel(r"Observation window, $T$")
    ax.set_ylabel(r"$n_{\mathrm{OUF}}(T)\,/\,n_{\mathrm{IOU}}(T)$")

    for panel, ax in zip("ABC", axes):
        ax.text(-0.26, 1.14, panel, transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig("fig4.pdf")
    print("wrote fig4.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-usetex", action="store_true")
    args = p.parse_args()

    configure_fonts(usetex=not args.no_usetex, base=11)
    make_figure()