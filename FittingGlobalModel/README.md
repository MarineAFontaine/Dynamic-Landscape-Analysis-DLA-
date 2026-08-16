# Fitting the dynamical landscape model to data

Code reproducing the model fitting and prediction for the paper:

> M. Fontaine, J.M. Delas, M. Saez, R.J. Maizels, E. Finnie, J. Briscoe, D.A. Rand (2025).
> *Dynamic Landscape Analysis of Cell Fate Decisions: Predictive Models of Neural
> Development From Single-Cell Data*. bioRxiv. https://doi.org/10.1101/2025.05.28.656648

All the methods used here are described in **Appendix B (Modelling & estimation)**:
https://doi.org/10.6084/m9.figshare.31342816


## Contents

### Models

| File | Purpose | Appendix B |
|------|---------|------------|
| `fitting_core.py` | Model-agnostic: SDE settings, Euler–Maruyama integration, proportion counting, PyABC model and distance builders. Contains no landscape — you pass your own `compute_landscape`. | Sections B3, B4 |
| `landscape_global.py` | The global landscape model (mouse; used for the flow cytometry and scRNA-seq fits). | B2 |
| `landscape_human.py` | The landscape model fitted to the human dataset of Rito et al. | Section B9 |
| `fitting_plots.py` | Plotting helpers shared by the fitting notebooks (stacked proportion bars, histogram grids, tendency plots with percentile bands, posterior parameter densities). | — |
| `palette.py` | Cell-type and timepoint colour palettes. | — |
| `prediction_global_earlypulse.py` | Prediction machinery for the Early Pulse (SAGUP) experiment: the global landscape with time-varying parameters interpolated between the 0nM and 500nM fits. Not model-agnostic — the pulse protocol is written in directly. | Section B6 |

### Fitting notebooks

| File | Purpose | Appendix B |
|------|---------|------------|
| `Fitting_Flow_SAG0.ipynb` | Fit the global model to flow cytometry data, **0 nM SAG**. | Section B5 |
| `Fitting_Flow_SAG10.ipynb` | Same, **10 nM SAG**. | Section B5 |
| `Fitting_Flow_SAG100.ipynb` | Same, **100 nM SAG**. | Section B5 |
| `Fitting_Flow_SAG500.ipynb` | Same, **500 nM SAG**. | Section B5 |
| `Fitting_RNAseq_SAG500.ipynb` | Fit the global model to mouse scRNA-seq data, 500 nM SAG. | Section B7 |
| `Fitting_RNAseq_Human_18hDelay.ipynb` | Fit the human landscape model, 18h TGFβ delay. | Section B9 |
| `Fitting_RNAseq_Human_24hDelay.ipynb` | Fit the human landscape model, 24h TGFβ delay. | Section B9 |

### Analysis and prediction notebooks

| File | Purpose | Appendix B |
|------|---------|------------|
| `ParametersDistributions.ipynb` | Posterior parameter distributions from the four flow cytometry fits. | Section B5.1 |
| `ModelPrediction_Flow_EarlyPulse.ipynb` | Early Pulse (SAGUP) experiment from the constant-SAG fits; only the two response times $T_1$, $T_2$ are fitted. | Section B6 |

### Other

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies. |
| `Experimental_and_Simulated_Proportions_Flow/` | Input proportions and fitted parameters (see *Data* below). |


The notebooks are independent of one another, with two exceptions:
`ParametersDistributions.ipynb` and `ModelPrediction_Flow_EarlyPulse.ipynb` both read the
posterior parameters written by the `Fitting_Flow_SAG*.ipynb` notebooks. Those parameter
files are in this repository, so you can run either notebook without refitting. Within a
notebook, run the cells top to bottom.

## Data

The notebooks read CSV files from `Experimental_and_Simulated_Proportions_Flow/`:

- `Average_SAG{0,10,100,500}.csv` — experimental proportions per timepoint and cell state.
- `Flow_SAG{0,10,100,500}_parameters.csv` — posterior parameters accepted by the ABC fit.
- `Flow_SAG{0,10,100,500}_simulation.csv` — the corresponding simulated proportions.
- `Proportions_EarlyPulse_SAGUP.csv` — experimental proportions for the Early Pulse experiment.


The mouse scRNA-seq data originate from Maizels, Snell & Briscoe (2024); the human data
fitted by `Fitting_RNAseq_Human_18hDelay.ipynb` and `Fitting_RNAseq_Human_24hDelay.ipynb`
originate from Rito et al. (2025). Both are listed under *References* below.

## Setup

```bash
pip install -r requirements.txt
jupyter lab        # or: jupyter notebook
```

Then open a notebook and run the cells in order. The ABC fits are the expensive step:
each generation simulates `population_size` parameter sets, so a full fit can take an hour on
a laptop for the full model. 

## Method, in brief

1. **Build the landscape** (`landscape_*.py`, Appendix B Section B2) — sub-landscapes for
   each binary decision are joined along their trajectories into one global landscape;
   `compute_landscape(x, y, params, perturb_std)` returns the drift and the noise level.
2. **Simulate** (`fitting_core.make_parSDE`, `SDE`, Section B3) — cells start from a given
   mixture of clusters and are integrated with Euler–Maruyama; positions are recorded at
   the measurement times.
3. **Count proportions** (`fitting_core.make_model`, Section B4.1) — cells within `radius`
   of an attractor are counted as being in that state, giving a proportion per timepoint
   and state.
4. **Compare to data** (`fitting_core.make_distance`, Section B4.2) — sum of absolute
   differences between simulated and experimental proportions, after dropping the
   timepoints excluded from the fit.
5. **Fit with ABC** (PyABC, Section B4.3) — uniform priors over the parameter domain $D$,
   population size $N = 1500$, up to 8 generations.
6. **Interpret** (Section B5.1) — locate the posterior in parameter space relative to the
   bifurcation sets of each sub-landscape.
7. **Predict** (Section B6) — hold the fitted parameters fixed and switch between the 0nM
   and 500nM fits over time to simulate a pulsed signal.

## Reproducibility

The initial cell positions are seeded (`make_blobs`, `random_state=1`), but the SDE noise
and the per-step parameter perturbation are not — ABC needs stochastic simulations. Runs
are therefore not identical.

## References

**Appendix B (Modelling & estimation)** — the methods reproduced here:
https://doi.org/10.6084/m9.figshare.31342816

**Data**

- Maizels, R.J., Snell, D.M., Briscoe, J. Reconstructing developmental trajectories using
  latent dynamical systems and time-resolved transcriptomics. *Cell Systems* **15**(5),
  411–424 (2024). https://doi.org/10.1016/j.cels.2024.04.004 — mouse scRNA-seq.
- Rito, T., Libby, A.R.G., Demuth, M. et al. Timely TGFβ signalling inhibition induces
  notochord. *Nature* **637**, 673–682 (2025). https://doi.org/10.1038/s41586-024-08332-w
  — human scRNA-seq, fitted in the two `Fitting_RNAseq_Human_*` notebooks.

**Methods**

- Schälte, Y., Klinger, E., Alamoudi, E., Hasenauer, J. pyABC: Efficient and robust
  easy-to-use approximate Bayesian computation. *Journal of Open Source Software*
  **7**(74), 4304 (2022). https://doi.org/10.21105/joss.04304
- Klinger, E., Rickert, D., Hasenauer, J. pyABC: distributed, likelihood-free inference.
  *Bioinformatics* **34**(20) (2018). https://doi.org/10.1093/bioinformatics/bty361
