# Gene module selection from scRNA-seq data

Code reproducing the gene-module selection and clustering for the paper:

> M. Fontaine, J.M. Delas, M. Saez, R.J. Maizels, E. Finnie, J. Briscoe, D.A. Rand (2025).
> *Dynamic Landscape Analysis of Cell Fate Decisions: Predictive Models of Neural
> Development From Single-Cell Data*. bioRxiv. https://doi.org/10.1101/2025.05.28.656648

All the methods used here are described in **Appendix A (Data analysis)**:
https://doi.org/10.6084/m9.figshare.31342810

These notebooks select a module of genes that distinguishes cell states and/or transition
routes between cell states, then cluster and annotate cells using that module. The same
method is applied to three datasets — early mouse, late mouse and human.

## Contents

| File | Purpose | Appendix A |
|------|---------|------------|
| `GeneModules_RNAseq.py` | Helper functions used by all three notebooks: marker-centred samples and differential-expression gene lists. | Sections A1.1, A1.2 |
| `Clustering_and_gene_module_EarlyTimepoints.ipynb` | Early mouse timepoints (D3–D4) → module `G_early`. | Sections A1.1, A1.2 |
| `Clustering_and_gene_module_LateTimepoints.ipynb` | Late mouse timepoints (D5–D8) → module `G_late`. | Sections A1.1, A1.2 |
| `Clustering_and_gene_module_Human.ipynb` | Human notochord data (D3, D5, D7) → module `G_human`. | Section A5 |
| `requirements.txt` | Python dependencies. | — |

The relevant sections of Appendix A are:

* __Section A1.1:__ Outward clustering and gene module selection in scRNA-seq data,
* __Section A1.2:__ Choice and use of marker genes,
* __Section A5:__ Testing the model on human data *(the human notebook)*.

The three notebooks are independent of one another; run whichever you need. Within a
notebook, run the cells top to bottom.

## Data

Each notebook reads an AnnData (`.h5ad`) file that can be downloaded at Zenodo under the DOI https://doi.org/10.5281/zenodo.15584010:

| File | Notebook | Module produced |
|------|----------|-----------------|
| `NeuralEarly_celltype.h5ad` | Mouse EarlyTimepoints | `G_early` |
| `NeuralLate_celltype.h5ad` | Mouse LateTimepoints | `G_late` |
| `Human_celltype.h5ad` | Human | `G_human` |


The mouse data originate from Maizels, Snell & Briscoe, *Cell Systems* 15(5) (2024),
https://doi.org/10.1016/j.cels.2024.04.004; the human data from Rito et al., *Nature* 637,
673–682 (2025), https://doi.org/10.1038/s41586-024-08332-w.

## Setup

```bash
pip install -r requirements.txt
jupyter lab        # or: jupyter notebook
```

Then open a notebook and run the cells in order.

## Method, in brief

1. **Load & normalise** — read raw counts, then CPM-normalise and `log1p`.
2. **Define "1-marker samples"** (`collect_samples`, Appendix A Section A1.2) — for each
   marker gene, group the cells whose expression sits near that marker's centre in
   expression space. `calculate_mean_and_percentile_distance` sets how wide that
   neighbourhood is.
3. **Select the gene module** (Section A1.1) — find the genes differentially expressed
   between those samples: `get_genelist_by_day` (per timepoint, mouse) or
   `get_genelist_human` (per sample, human).
4. **Restrict, cluster, annotate** — subset the data to the module, run PCA and Leiden
   clustering, then map the clusters to cell type / domain labels. Clusters that sit
   between two attractors are labelled as transitioning.

## What this feeds

The annotated `.h5ad` files produced here are the input to:

- [`../../TransitionAnalysis/`](../../TransitionAnalysis/) — expression along transitions
  (Appendix A, Section A2),
- [`../../FittingModel/`](../../FittingModel/) — the landscape model fitted to the
  resulting cell-state proportions (Appendix B).

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
