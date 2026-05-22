# Gene module selection from scRNA-seq data

Code accompanying Appendix A (Data Analysis) of:

> M. Fontaine, J.M. Delas, M. Saez, R.J. Maizels, E. Finnie, J. Briscoe, D.A. Rand (2025).
> *Dynamic Landscape Analysis of Cell Fate Decisions: Predictive Models of Neural
> Development From Single-Cell Data*. bioRxiv. https://doi.org/10.1101/2025.05.28.656648

These notebooks select a module of genes that distinguishes cell states, then cluster
and annotate cells using that module. The same method is applied to three datasets
(early mouse, late mouse, and human), as described in Appendix A, Sections A1.1–A1.2
(and Section A5 for the human data).

## Contents

| File | Purpose |
|------|---------|
| `GeneModules_RNAseq.py` | Helper functions used by all three notebooks (gene-module selection). |
| `Clustering_and_gene_module_EarlyTimepoints.ipynb` | Early mouse timepoints (D3–D4) → module `G_early`. |
| `Clustering_and_gene_module_LateTimepoints.ipynb` | Late mouse timepoints (D5–D8) → module `G_late`. |
| `Clustering_and_gene_module_Human.ipynb` | Human notochord data → module `G_human`. |
| `requirements.txt` | Python dependencies. |
| `*.h5ad` | Input datasets (see *Data* below). |

The three notebooks are independent of one another; run whichever you need. Within a
notebook, run the cells top to bottom.

## Data

Each notebook reads an AnnData (`.h5ad`) file that ships in this same folder:

- `NeuralEarly_celltype.h5ad` (early notebook)
- `NeuralLate_celltype.h5ad` (late notebook)
- `Human_celltype.h5ad` (human notebook)

The mouse data originate from Maizels, Snell & Briscoe, *Cell Systems* 15(5) (2024),
https://doi.org/10.1016/j.cels.2024.04.004; the human data from Rito et al., *Nature*
637 (2025), https://doi.org/10.1038/s41586-024-08332-w.

## Setup

```bash
pip install -r requirements.txt
jupyter lab        # or: jupyter notebook
```

Then open a notebook and run the cells in order.

## Method, in brief

1. **Load & normalise** — read raw counts, then CPM-normalise and `log1p`.
2. **Define "1-marker samples"** (`collect_samples`) — for each marker gene, group the
   cells whose expression sits near that marker's centre in expression space.
3. **Select the gene module** — find genes that are differentially expressed between
   those samples: `get_genelist_by_day` (per timepoint, mouse data) or
   `get_genelist_human` (per sample, human data).
4. **Restrict, cluster, annotate** — subset the data to the module, run PCA and Leiden
   clustering, then map clusters to cell-type / domain labels.
