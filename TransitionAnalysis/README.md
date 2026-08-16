# Transition analysis of scRNA-seq data

Code reproducing the transition analysis for the paper:

> M. Fontaine, J.M. Delas, M. Saez, R.J. Maizels, E. Finnie, J. Briscoe, D.A. Rand (2025).
> *Dynamic Landscape Analysis of Cell Fate Decisions: Predictive Models of Neural
> Development From Single-Cell Data*. bioRxiv. https://doi.org/10.1101/2025.05.28.656648

All the methods used here are described in **Appendix A (Data analysis)**:
https://doi.org/10.6084/m9.figshare.31342810

A transition between two attractor clusters is followed by approximating the unstable manifold
that connects them. A group of attractor clusters and transitionning cells (a *sub-landscape*) is projected into LDA space, the densest point of each cluster gives a *centre*, a spline through
those centres in gene space gives a *route*, and gene expression is averaged in a ball
around each point of the route — with a bootstrap band — to show how expression varies
along the transition.

## Contents

| File | Purpose | Appendix A |
|------|---------|------------|
| `transitions_functions_rnaseq.py` | Helper functions: LDA sub-landscapes, cluster centres, routes through gene space, ball averages and bootstrap bands. | Section A2 |
| `RNAseq_Transitions_Early_Neurals.ipynb` | Transition analysis on the early neural data: the NMP and PreNeural sub-landscapes (mouse) and the PreNeural sub-landscape (human). | Section3 A2, A5.2 |
| `requirements.txt` | Python dependencies. | — |

The gene modules the notebook restricts to are selected in
[`../RNAseqDataAnalysis/Clustering/`](../RNAseqDataAnalysis/Clustering/) (Appendix A,
Sections A1.1–A1.2, and A5 for the human data).

## Data

The notebook reads AnnData (`.h5ad`) files from
[`../RNAseqDataAnalysis/`](../RNAseqDataAnalysis/) at Zenodo under the DOI https://doi.org/10.5281/zenodo.15584010:

| File | Used for |
|------|----------|
| `NeuralEarly_celltype.h5ad` | mouse, early timepoints (Parts 1 and 2) |
| `Human_celltype.h5ad` | human (Part 3) |


The mouse data originate from Maizels, Snell & Briscoe, *Cell Systems* 15(5) (2024),
https://doi.org/10.1016/j.cels.2024.04.004; the human data from Rito et al., *Nature*
637, 673–682 (2025), https://doi.org/10.1038/s41586-024-08332-w.

## Setup

```bash
pip install -r requirements.txt
jupyter lab        # or: jupyter notebook
```

Then open the notebook and run the cells in order.

## Method, in brief

1. **Load & normalise** (`sc.read_h5ad`) — read raw counts, then normalise and `log1p`.
2. **Restrict to the gene module** — subset the data to `G_early` (or `G_human`) before
   projecting, so the LDA axes are built from the genes that separate the states.
3. **Build the sub-landscape** (`perform_lda`, Appendix A Section A1.4) — LDA of a chosen
   list of cell types; the embedding is stored in `obsm['X_LDA']`.
4. **Find the cluster centres** (`find_high_density_point`) — the densest point of each
   cluster in LDA space, by KDE.
5. **Trace the route** (`unstable_ges`, `lda_unst`, `compute_routes`, Section A2.2) — a
   quadratic or cubic spline through those centres in gene space, then projected into LDA
   space. This approximates the unstable manifold along which cells transition.
6. **Visualise** (`plot_lda`, `render_plot`) — scatter the sub-landscape per timepoint and
   overlay the route.
7. **Gene variation along the route** (`bootstrap_route_band`, `bootstrap_route_grid`,
   Section A2.4) — average each gene over the cells within `radius` of every route point,
   bootstrapped to give a mean ± SD band. Run `bootstrap_sanity_check` first to confirm
   each subsample holds enough cells.

**Prerequisite.** The transition functions need `obsm['X_LDA']`, so `perform_lda` must be
run before them, and every gene plotted must belong to the module used to build the
sub-landscape. If you change the module, re-run the subset and `perform_lda` cells before
plotting.

## References

**Appendix A (Data analysis)** — the methods reproduced here:
https://doi.org/10.6084/m9.figshare.31342810

**Data**

- Maizels, R.J., Snell, D.M., Briscoe, J. Reconstructing developmental trajectories using
  latent dynamical systems and time-resolved transcriptomics. *Cell Systems* **15**(5),
  411–424 (2024). https://doi.org/10.1016/j.cels.2024.04.004 — mouse scRNA-seq.
- Rito, T., Libby, A.R.G., Demuth, M. et al. Timely TGFβ signalling inhibition induces
  notochord. *Nature* **637**, 673–682 (2025). https://doi.org/10.1038/s41586-024-08332-w
  — human scRNA-seq.
