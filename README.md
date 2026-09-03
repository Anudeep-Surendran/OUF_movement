# Ornstein-Uhlenbeck Foraging: Manuscript Scripts

This directory contains the Python and R scripts used to reproduce manuscript figures for the Ornstein-Uhlenbeck-foraging (OUF) movement model and its infinite-home-range limit (IOU).

The scripts calculate position variances, variogram, mean encounter rates, accumulated encounter counts, and the effect of directional persistence while holding the stationary speed fixed. Parameters are defined in each script and correspond to the manuscript figure captions and equations.

## Requirements

- Python 3.9 or newer
- NumPy
- Matplotlib
- R 4.6.1 or newer
- RStudio 2026.08.2 or newer
- ctmm
- A working LaTeX installation for the default figure rendering

Install the Python dependencies with:

```bash
python -m pip install numpy matplotlib
```

The scripts use Matplotlib's `usetex` mode by default. Install a LaTeX distribution such as MacTeX on macOS, or use the `--no-usetex` option supported by `fig4.py` when LaTeX is unavailable.

## Scripts


| `fig1a.py` | Compares the analytical OUF position variance with Euler-Maruyama simulations for Figure 1A. | `fig1a.pdf` |
| `fig1b.R` | Produces variogram and OUF model fit for animal movement tracking data. | `fig1b.pdf` |
| `fig2.py` | Produces encounter-rate curves for two home-range configurations. | `fig2.pdf` |
| `fig3.py` | Produces the heatmap of the encounter-rate derivative with respect to directional persistence. | `fig3.png` |
| `fig4.py` | Compares OUF and IOU movement and encounter statistics. | `fig4.pdf` |

## Usage

Run commands from this directory:

```bash
python fig1a.py
python fig2.py
python fig3.py
python fig4.py
```
Run `fig1b.R` using RStudio 2026.08.2 or its newer versions.
 
Figure 1A accepts simulation controls. Its output filename is fixed:

```bash
python fig1a.py --nreal 10000 --dt 0.002 --seed 0
```

Figure 2 uses fixed output filenames:

```bash
python fig2.py
```

Figure 4 can disable LaTeX rendering and uses a fixed output filename:

```bash
python fig4.py --no-usetex
```

Generated figures are written to the current working directory.

## Reproducibility notes

- Figure 1A uses a fixed random seed by default (`0`), so its simulation is reproducible for the same NumPy version and settings.
- Figure 1A checks the analytical limits before plotting.
- The fixed-speed scripts vary the driving-process variance and noise amplitude with `tau_v` so that the stationary mean squared speed remains constant.
- Units are hours for time, kilometres for distance, and the corresponding derived units for variances and encounter rates.

## Repository layout

This README documents the scripts in `Manuscript/Scripts`. The manuscript text and other project materials are located in the parent project directories.