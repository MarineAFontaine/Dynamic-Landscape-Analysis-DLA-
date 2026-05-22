# Author: Marine Fontaine (Warwick)
#
# Helper functions for gene-module selection from scRNA-seq data.
# Used by the notebook `Clustering_and_gene_module_EarlyTimepoints.ipynb`.
# Method described in Appendix A (Data Analysis), Sections A1.1-A1.2, of:
#   Fontaine, Delas, Saez, Maizels, Finnie, Briscoe, Rand (2025),
#   "Dynamic Landscape Analysis of Cell Fate Decisions", bioRxiv
#   doi: 10.1101/2025.05.28.656648
#
# Pipeline overview:
#   1. calculate_mean_and_percentile_distance : per-marker expression centre + spread
#   2. collect_samples                        : build "1-marker samples" of cells near each marker's centre
#   3. get_genelist_by_day                    : differentially-expressed genes between sample pairs, per timepoint
#   4. get_genelist_human                     : differentially-expressed genes per sample (no timepoint split)
#
# Expected input: a log-normalised AnnData object (cells x genes).

import itertools
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import scipy.stats as stats
import anndata as ann
import scanpy as sc


def _to_dense(matrix):
    """Return a dense numpy array whether `matrix` is sparse or already dense.

    AnnData's `.X` may be a scipy sparse matrix or a plain numpy array. The
    sparse `.A` attribute does not exist on dense arrays, so accessing it
    directly (`adata.X.A`) crashes on dense data. This helper handles both.
    """
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def calculate_mean_and_percentile_distance(data_LogNorm, genes, confidence):
    """
    For each marker gene, estimate its expression centre and spread.

    A Gaussian is fitted to the non-zero log-normalised expression values of
    each gene. The "centre" is the fitted mean; the "spread" is the distance
    from that mean to the requested percentile (used later as a search radius).

    Parameters
    ----------
    data_LogNorm : AnnData
        Log-normalised expression data (cells x genes).
    genes : list[str]
        Marker genes to characterise.
    confidence : float
        Percentile (e.g. 95) at which to measure the spread above the mean.

    Returns
    -------
    mean : list[float]
        Fitted Gaussian mean per gene.
    percentile_distance : list[float]
        Distance from mean to the `confidence` percentile per gene.
    """
    # Fail early with a clear message if any requested gene is missing
    missing = [g for g in genes if g not in data_LogNorm.var_names]
    if missing:
        raise ValueError(f"Gene(s) not found in data.var_names: {missing}")

    mean, percentile_distance = [], []
    for x in genes:
        # Expression vector for gene x (dense), across all cells
        data = _to_dense(data_LogNorm[:, data_LogNorm.var_names.isin([x])].X)
        # Drop non-expressing cells so the fit reflects expressing cells only
        data = data[data != 0]
        if data.size == 0:
            raise ValueError(
                f"Gene '{x}' has no non-zero expression values; cannot fit a "
                f"distribution. Remove it from `genes` or check the data."
            )
        # Fit a Gaussian; centre = mean (std is not used here)
        mu, _ = stats.norm.fit(data)
        # Spread = distance from the mean to the chosen percentile (search radius)
        dist = np.percentile(data, confidence) - mu

        mean.append(mu)
        percentile_distance.append(dist)
    return mean, percentile_distance


def collect_samples(data, genes, confidence, key):
    """
    Define one "1-marker sample" of cells per marker gene.

    Each marker defines a point in expression space whose coordinate is high
    for that marker and zero for the others. Cells lying
    within the marker's spread (radius) of that centre are grouped together as
    that marker's sample. Group labels are written back into `data.obs[key]`.

    Parameters
    ----------
    data : AnnData
        Log-normalised expression data (cells x genes).
    genes : list[str]
        Marker genes; each defines one sample.
    confidence : float
        Percentile passed to `calculate_mean_and_percentile_distance` (sets the radius).
    key : str
        Name of the obs column in which to store the sample labels.

    Returns
    -------
    AnnData
        The input `data` with a new categorical obs column `key`.
    """
    # Fail early with a clear message if any requested gene is missing
    missing = [g for g in genes if g not in data.var_names]
    if missing:
        raise ValueError(f"Gene(s) not found in data.var_names: {missing}")

    # Restrict to the marker genes (defines the coordinate space for the search).
    # Reindexing by `genes` keeps the column order aligned with the diagonal
    # centres built from `means` below (boolean masks would use var_names order).
    data_TF = data[:, genes]
    means, radii = calculate_mean_and_percentile_distance(data, genes, confidence)

    # Diagonal centres: marker i is high on its own axis, zero on the others
    centers = np.diag(means).tolist()
    point_tree = cKDTree(_to_dense(data_TF.X))

    # Collect the cells belonging to each marker's sample
    groups_to_test = []
    for i, (center, radius) in enumerate(zip(centers, radii)):
        idx = point_tree.query_ball_point(center, radius)
        gene_data = data[idx, :].copy()
        gene_data.obs[key] = pd.Categorical([f'Sample {genes[i]}'] * len(gene_data))
        groups_to_test.append(gene_data)

    # Concatenate the per-marker samples and copy labels back onto `data`
    groups_to_test = ann.concat(groups_to_test)
    data.obs[key] = groups_to_test.obs[key]

    return data


def get_genelist_by_day(data, group, alpha, days, timepoint_key='timepoint'):
    """
    Find differentially-expressed genes between sample pairs, at each timepoint.

    Every ordered sample pair is tested at every timepoint in `days`. For each
    (timepoint, sample-pair) combination a Wilcoxon rank-genes test is run and
    the genes whose scores fall in the extreme tails (below the (100-alpha)
    percentile or above the alpha percentile) are kept.

    Parameters
    ----------
    data : AnnData
        Data containing the sample labels in `data.obs[group]`.
    group : str
        Name of the obs column holding the sample labels.
    alpha : float
        Upper-tail percentile (e.g. 99.95). The lower tail is 100-alpha.
    days : list
        Timepoint values to iterate over (matched against data.obs[timepoint_key]).
    timepoint_key : str, optional
        Name of the obs column holding the timepoint of each cell.
        Defaults to 'timepoint'.

    Returns
    -------
    np.ndarray
        Sorted, unique gene names selected across all comparisons.
    """
    # Validate required obs columns up front for clear error messages
    if group not in data.obs:
        raise ValueError(f"Group column '{group}' not found in data.obs.")
    if timepoint_key not in data.obs:
        raise ValueError(
            f"Timepoint column '{timepoint_key}' not found in data.obs. "
            f"Pass timepoint_key=<your column name>."
        )
    # Keep only cells assigned to a 1-marker sample
    grouped_data = data[data.obs[group].notna()]
    samples = grouped_data.obs[group].unique().tolist()  # sample names

    # Test every sample pair at every timepoint; keep extreme-tail DE genes.
    genelist = []
    for day in days:
        grouped_data_day = grouped_data[grouped_data.obs[timepoint_key] == day].copy()
        # Skip timepoints with fewer than two samples present (no comparison possible)
        present = grouped_data_day.obs[group].unique().tolist()
        for pair in itertools.combinations(samples, 2):
            if pair[0] not in present or pair[1] not in present:
                continue
            sc.tl.rank_genes_groups(
                grouped_data_day, groupby=group,
                groups=[pair[0]], reference=pair[1], method='wilcoxon'
            )
            scores = grouped_data_day.uns['rank_genes_groups']['scores'][pair[0]]
            # Tail thresholds: lower = (100-alpha) percentile, upper = alpha percentile
            lower_tail = np.percentile(scores, 100 - alpha)
            upper_tail = np.percentile(scores, alpha)
            # Indices of genes in either tail
            score_indices = [i for i, score in enumerate(scores)
                             if score < lower_tail or score > upper_tail]
            # Corresponding gene names
            genes = grouped_data_day.uns['rank_genes_groups']['names'][pair[0]][score_indices]
            genelist.extend(genes)
    return np.unique(np.array(list(set(genelist))))


def get_genelist_human(data, group, score_min, n_genes_max):
    """
    Find differentially-expressed genes per sample (no timepoint split).

    For each sample, the genes with a Wilcoxon score above `score_min` are kept,
    taking at most the top `n_genes_max` of them (they are already sorted by
    descending score).

    Parameters
    ----------
    data : AnnData
        Data containing the sample labels in `data.obs[group]`.
    group : str
        Name of the obs column holding the sample labels.
    score_min : float
        Minimum Wilcoxon score a gene must exceed to be kept.
    n_genes_max : int
        Maximum NUMBER of genes to keep per sample (a count, not a score):
        the top `n_genes_max` genes that pass the `score_min` threshold.

    Returns
    -------
    np.ndarray
        Sorted, unique gene names selected across all samples.
    """
    # Validate that the grouping column exists
    if group not in data.obs:
        raise ValueError(f"Group column '{group}' not found in data.obs.")

    # Keep only cells assigned to a 1-marker sample
    grouped_data = data[data.obs[group].notna()].copy()
    samples = grouped_data.obs[group].unique().tolist()  # sample names

    # Rank genes once for all groups 
    sc.tl.rank_genes_groups(grouped_data, group, method='wilcoxon')

    # For each sample: keep genes above score_min, then take at most n_genes_max
    genelist = []
    for sample in samples:
        df = sc.get.rank_genes_groups_df(grouped_data, group=sample)
        genes = df.loc[df["scores"] > score_min]['names'][0:n_genes_max].tolist()
        genelist.extend(genes)
    return np.unique(np.array(list(set(genelist))))
