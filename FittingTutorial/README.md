# Tutorial: fitting a landscape model to cell-state proportions

A standalone worked example accompanying:

> M. Fontaine, J.M. Delas, M. Saez, R.J. Maizels, E. Finnie, J. Briscoe, D.A. Rand (2025).
> *Dynamic Landscape Analysis of Cell Fate Decisions: Predictive Models of Neural
> Development From Single-Cell Data*. bioRxiv. https://doi.org/10.1101/2025.05.28.656648

The method is described in **Appendix B (Modelling & estimation)**:
https://doi.org/10.6084/m9.figshare.31342816 — specifically Section B3 (Parameterising
and simulating the landscape) and Section B4 (Fitting the model to simulated
proportions).

This tutorial fits a small **first-decisions** landscape rather than the full global
model: the two binary decisions taken between days D3 and D4, from NMP (T/Bra) to either
PreNeural (Nkx1.2) or Early Mesoderm/Mesoderm (Tbx6/Foxc2), and from PreNeural (Nkx1.2) to either EarlyVentral (Foxa2) or p0/p1 (Pax6). Six parameters
are fitted instead of nineteen, so a full ABC run finishes in minutes rather than hours —
which makes it the right place to start, and the right place to plug in your own
landscape.

![First Decisions Landscape Simulation](Github_SimulationNMP.jpg)

## Contents

| File | Purpose |
|------|---------|
| `Fitting_RNAseq_FirstDecisions_Tutorial.ipynb` | The tutorial. Simulates the landscape, fits it with PyABC, and plots the posterior and the simulated proportions. |
| `landscape_first_decisions.py` | The first-decisions landscape: `compute_landscape`, the attractor positions `cellstates`, and `default_parSDE`. |
| `fitting_core.py` | The model-agnostic fitting machinery (same file as in the parent folder). |
| `fitting_plots.py` | The plotting helpers (same file as in the parent folder). |
| `requirements.txt` | Python dependencies. |
| `Experimental_and_Simulated_Proportions_NMPTransition/` | Input proportions and fitted parameters (see *Data* below). |
| `Github_SimulationNMP.jpg` | Figure shown in the notebook: simulated cell trajectories on the landscape. |
| `Github_FirstDecisionsLandscapes.jpg` | Figure shown in the notebook: the two sub-landscapes and their bifurcations. |
| `Github_DecisionLandscape.jpg` | Figure of the decision landscape. |

## Data

Everything the tutorial needs:
`Experimental_and_Simulated_Proportions_NMPTransition/`:

| File | Purpose |
|------|---------|
| `Proportions_RNAseq_FirstDecisions.csv` | Experimental proportions of Meso, NMP, PreNeural, p0/p1 and EarlyVentral at each timepoint — the input the model is fitted to. |
| `NMPFlip_parameters.csv` | Posterior parameters accepted by the ABC fit. |
| `NMPFlip_simulation.csv` | The corresponding simulated proportions. |

The proportions derive from the mouse scRNA-seq data of Maizels, Snell & Briscoe,
*Cell Systems* 15(5) (2024), https://doi.org/10.1016/j.cels.2024.04.004.


## Setup

```bash
pip install -r requirements.txt
jupyter lab        # or: jupyter notebook
```

Then run the cells top to bottom.

## What the notebook does

1. **Import experimental proportions** — read the CSV, merge the transitioning clusters
   into their parent states, and flatten to the vector PyABC compares against.
2. **Set up the model** — pass `compute_landscape` from `landscape_first_decisions` to
   `fitting_core.make_model`, together with the attractor positions, the SDE settings, and
   the initial mixture (~5% Meso, 85% NMP, 10% PreNeural). `make_distance()` keeps every
   timepoint here, so nothing is dropped.
3. **Fit with ABC** — uniform priors on $(u_1, v_1, u_2, v_2, \sigma_A, vel_A)$,
   population size 1500, up to 8 generations (Appendix B Section B4).
4. **Inspect the posterior** — parameter distributions and the decrease of the
   $\varepsilon$-threshold across generations.
5. **Check the fit** — simulated versus experimental proportions, as tendency plots with
   a 5–95 percentile band and as stacked barplots.
6. **Locate the posterior** — the fitted parameters plotted over the bifurcation sets of
   the two sub-landscapes, showing which decision is bistable and which has already
   collapsed.
7. **Save results** — accepted simulations and posterior parameters as DataFrames.

## Fitting your own model

The machinery in `fitting_core.py` contains no landscape. To fit a different one, write

```python
@numba.jit(nopython=True)
def compute_landscape(x, y, params, perturb_std):
    ...
    return L_x, L_y, noise
```

then pass it to `make_model(...)` in place of the tutorial's landscape, along with your
own `cellstates` (attractor name → `[x, y]`) and your own priors. Note that the order of
`cellstates` must match the column order of your experimental data, because the distance
compares the two position by position.

## Reproducibility

The initial cell positions are seeded (`make_blobs`, `random_state=1`), but the SDE noise
and the per-step parameter perturbation are not — ABC needs stochastic simulations. Runs
are therefore not bit-identical.
