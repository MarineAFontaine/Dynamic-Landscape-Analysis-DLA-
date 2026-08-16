### transitions_functions_rnaseq.py -- transition analysis of scRNA-seq data.
### Marine Fontaine (University of Warwick)
#
# Used by the notebook `RNAseq_Transitions_Early_Neurals.ipynb`. The method is described
# in Appendix A (Data analysis) [2] of [1]:
#     A2    Probing expression changes during transition using scRNA-seq,
#     A2.2  Computational approximation of gene dynamics along unstable manifolds,
#     A2.4  Example applications to key sub-landscapes,
#     A5.2  Sub-landscapes and gene expression (the human dataset).
#
# [1] M. Fontaine, J.M. Delas, M. Saez, R.J. Maizels, E. Finnie, J. Briscoe, D.A. Rand,
#     (2025). *Dynamic Landscape Analysis of Cell Fate Decisions: Predictive Models of
#     Neural Development From Single-Cell Data*. bioRxiv doi:
#     https://doi.org/10.1101/2025.05.28.656648
# [2] M. Fontaine, J.M. Delas, M. Saez, R.J. Maizels, E. Finnie, J. Briscoe, D.A. Rand,
#     (2025). *Appendix A: Data analysis*. figshare doi:
#     https://doi.org/10.6084/m9.figshare.31342810
#
# Pipeline overview:
#   1. perform_lda              : LDA projection of chosen cell types (a "sub-landscape")
#   2. find_high_density_point  : per-cluster centre = densest point in LDA space (KDE)
#   3. unstable_ges             : spline through the centres in gene space (a "route")
#   4. lda_unst                 : project a route into LDA space
#   5. compute_routes           : run 2-4 for any number of reference routes
#   6. plot_lda / render_plot   : scatter the sub-landscape, overlay a route
#   7. bootstrap_sanity_check   : report cells per subsample before bootstrapping
#   8. bootstrap_route_band     : bootstrap a gene's profile along each route (mean +/- SD)
#   9. bootstrap_route_grid     : bootstrap_route_band for many genes, as a grid
#
# Expected input: a log-normalised AnnData with obs columns 'celltype' and
# 'timepoint', and an LDA embedding in obsm['X_LDA'] (set by perform_lda).
# Configurable via celltype_key / lda_key / timepoint_key arguments.
from scipy import spatial
from scipy.stats import gaussian_kde
import scipy.linalg as la
from scipy.interpolate import CubicSpline
from scipy import interpolate
from scipy.interpolate import splrep, BSpline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
def _to_dense(matrix):
    """Return a dense numpy array whether `matrix` is sparse or already dense."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
def _suggest_radius(data, routes, dimension, lda_key='X_LDA'):
    """For each route, the max distance from a route point to its nearest cell.
    A `radius` larger than this value guarantees no empty balls on that route.
    Used to advise the user when a route comes out empty (all-NaN).
    """
    tree = spatial.cKDTree(data.obsm[lda_key][:, :dimension])
    out = []
    for route in routes:
        pts = np.column_stack([route[:, d] for d in range(dimension)])
        d, _ = tree.query(pts, k=1)
        out.append(float(d.max()))
    return out
def perform_lda(data, cluster_list, genes_to_drop, celltype_key='celltype', lda_key='X_LDA'):
    """
    Fit Linear Discriminant Analysis on a chosen set of cell types.
    The cell types in `cluster_list` define a "sub-landscape". LDA finds the
    directions that best separate them; the projected coordinates are stored in
    `obsm[lda_key]` and the discriminant axes are returned as `loadings`.
    Parameters
    ----------
    data : AnnData
        Log-normalised data with an obs column of cell-type labels.
    cluster_list : list[str]
        Cell types to include in the sub-landscape.
    genes_to_drop : list[str] or None
        Genes to exclude before fitting (None keeps all genes).
    celltype_key : str, optional
        obs column holding the cell-type labels. Default 'celltype'.
    lda_key : str, optional
        obsm key under which to store the LDA projection. Default 'X_LDA'.
    Returns
    -------
    sublandscape : AnnData
        Subset of `data` with the LDA projection in obsm[lda_key].
    loadings : np.ndarray
        LDA scalings (eigenvectors of Sw^{-1} Sb), i.e. the discriminant axes.
    """
    subset = data[data.obs[celltype_key].isin(cluster_list)]
    if genes_to_drop != None:
        subset = subset[:, ~subset.var_names.isin(genes_to_drop)]
    X_data = _to_dense(subset.X)
    y_target = subset.obs[celltype_key]
    lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
    X_r2 = lda.fit(X_data, y_target).transform(X_data)
    loadings = lda.scalings_  # eigenvectors of Sw^{-1} Sb
    sublandscape = data[data.obs[celltype_key].isin(cluster_list)].copy()
    sublandscape.obsm[lda_key] = X_r2
    return sublandscape, loadings
def plot_lda(ax, data, day, kde, s0, colorpalette,
             celltype_key='celltype', lda_key='X_LDA', timepoint_key='timepoint'):
    """
    Scatter the LDA sub-landscape (LD1 vs LD2), optionally for a single day.
    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw on.
    data : AnnData
        Data with obsm[lda_key] and the cell-type / timepoint obs columns.
    day : str or None
        If given, restrict to that timepoint; None plots all cells.
    kde : bool
        Overlay KDE contour lines if True.
    s0 : float
        Scatter point size.
    colorpalette : dict
        Maps cell-type name -> colour.
    celltype_key, lda_key, timepoint_key : str, optional
        obs/obsm key names. Defaults 'celltype', 'X_LDA', 'timepoint'.
    """
    if day != None:
        subset_specific = data[data.obs[timepoint_key] == day]
    else:
        subset_specific = data
    x = subset_specific.obsm[lda_key][:, 0]
    y = subset_specific.obsm[lda_key][:, 1]
    celltype_colors = [colorpalette[ci] for ci in subset_specific.obs[celltype_key]]
    ax.scatter(x, y, c=celltype_colors, edgecolor='black', linewidth=0.1, s=s0, alpha=0.8)
    if kde == True:
        sns.kdeplot(x=x, y=y, color='k', levels=10, linewidths=1, alpha=0.8, ax=ax)
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.grid(False)
def find_high_density_point(data, refs, dimension, grid_size=100,
                            celltype_key='celltype', lda_key='X_LDA', timepoint_key='timepoint'):
    """
    For each reference cluster, find a representative high-density cell in LDA space.
    A KDE is fitted to the cluster's LDA coordinates, the densest grid point is
    located, and the actual cell closest to it is taken as the cluster centre.
    Parameters
    ----------
    data : AnnData
        Data with obsm[lda_key] and the cell-type / timepoint obs columns.
    refs : list[tuple]
        Each entry is (celltype, day): day=False uses all timepoints, otherwise
        restricts to that timepoint.
    dimension : int
        Number of LDA dimensions to use.
    grid_size : int, optional
        Grid resolution per dimension for the KDE search. Default 100.
    celltype_key, lda_key, timepoint_key : str, optional
        obs/obsm key names. Defaults 'celltype', 'X_LDA', 'timepoint'.
    Returns
    -------
    dict
        Maps cluster key -> centre coordinates (key is 'celltype' or
        'celltype-day' depending on whether a day was specified).
    """
    if dimension > data.shape[1]:
        raise ValueError(f"Error: Dimension {dimension} exceeds available dimensions ({data.shape[1]}).")
    centers = {}
    for ref in refs:
        if ref[1] == False:
            subset_data = data[data.obs[celltype_key] == ref[0]]
        else:
            subset_data = data[(data.obs[timepoint_key] == ref[1]) & (data.obs[celltype_key] == ref[0])]
        data_selected = subset_data.obsm[lda_key][:, :dimension].T  # (dimension, N) for KDE
        kde = gaussian_kde(data_selected, bw_method='scott')
        grid_ranges = [np.linspace(data_selected[i].min(), data_selected[i].max(), grid_size) for i in range(dimension)]
        grid = np.meshgrid(*grid_ranges, indexing='ij')
        grid_points = np.vstack([g.ravel() for g in grid])
        densities = kde(grid_points)
        high_density_point = grid_points[:, np.argmax(densities)]
        distances = np.linalg.norm(subset_data.obsm[lda_key][:, :dimension] - high_density_point, axis=1)
        closest_row = subset_data.obsm[lda_key][np.argmin(distances), :dimension]
        if ref[1] == False:
            centers[ref[0]] = np.array(closest_row)
        else:
            centers[f'{ref[0]}-{ref[1]}'] = np.array(closest_row)
    return centers
def unstable_ges(data, refs, centers_indices, sorder, quadratic):
    """
    Fit a spline through the cluster centres in effective gene space ("route").
    The centres are placed at evenly spaced knots in [0, 1] and a spline is fitted
    per gene, then evaluated on a fine grid `xnew`. The number of knots adapts to
    the number of centres.
    - quadratic=True : degree-2 spline through exactly 3 centres (interpolating).
    - quadratic=False: smoothing cubic spline (degree 3), which needs >= 4 centres.
    Parameters
    ----------
    data : AnnData
        Data whose .X holds the effective gene space.
    refs : list[tuple]
        Reference clusters; its length is the number of centres on the route.
    centers_indices : dict
        Maps cluster key -> row index of its centre cell in `data`.
    sorder : float
        Smoothing factor `s` passed to splrep (cubic case).
    quadratic : bool
        If True, fit a quadratic spline through exactly 3 centres.
    Returns
    -------
    xnew : np.ndarray
        Parameter grid the spline is evaluated on (0 to 1).
    S : list[np.ndarray]
        Interpolated values per gene along `xnew`.
    """
    xnew = np.arange(0, 1.05, 0.05)
    n_centres = len(centers_indices)
    T = np.linspace(0, 1, n_centres)  # n=3 -> [0,0.5,1]; n=5 -> [0,0.25,0.5,0.75,1]
    MU = _to_dense(data.X)[np.array(list(centers_indices.values())), :]
    if quadratic:
        if n_centres != 3:
            raise ValueError(f"Quadratic route requires exactly 3 centres, got {n_centres}.")
        splines = [splrep(T, MU[:, i], s=0, k=2) for i in range(len(data.var))]
    else:
        if n_centres < 4:
            raise ValueError(
                f"Cubic (non-quadratic) route requires at least 4 centres, got "
                f"{n_centres}. Provide more clusters or set quadratic=True (needs 3)."
            )
        splines = [splrep(T, MU[:, i], s=sorder) for i in range(len(data.var))]
    S = [BSpline(*spline)(xnew) for spline in splines]
    return xnew, S
def lda_unst(data, unst, celltype_key='celltype'):
    """
    Project a route (`unst`) into the same LDA space as `data`.
    Re-fits LDA on the data's cell types, then applies that transform to the
    route points so they can be overlaid on the LDA scatter.
    Parameters
    ----------
    data : AnnData
        Data with a cell-type obs column; .X is the gene space the route lives in.
    unst : np.ndarray
        Route points in gene space (rows = points).
    celltype_key : str, optional
        obs column with cell-type labels. Default 'celltype'.
    Returns
    -------
    np.ndarray
        Route points projected into LDA space.
    """
    X_data = _to_dense(data.X)
    y_target = data.obs[celltype_key]
    lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
    X_r2 = lda.fit(X_data, y_target).transform(X_data)
    unst_r2 = lda.transform(unst)
    return unst_r2
def compute_routes(data, refs_list, dimension, sorder, quadratic,
                   celltype_key='celltype', lda_key='X_LDA', timepoint_key='timepoint'):
    """
    Build one or more routes (unstable manifolds) and project them into LDA space.
    For each reference set in `refs_list`: find cluster centres (high-density
    points), look up the cells at those centres, fit a spline through them in
    gene space, then project the spline into LDA space.
    Parameters
    ----------
    data : AnnData
        Sub-landscape with obsm[lda_key] (output of perform_lda).
    refs_list : list[list[tuple]]
        A list of routes. Each route is a list of (celltype, day) tuples
        (day=False = all timepoints). Pass any number of routes, e.g.
        [refs1, refs2] or [refs1, refs2, refs3].
    dimension : int
        Number of LDA dimensions used to locate centres.
    sorder : float
        Spline smoothing factor (see unstable_ges).
    quadratic : bool
        Quadratic spline if True (needs 3 centres per route), else cubic
        (needs >= 4 centres per route).
    celltype_key, lda_key, timepoint_key : str, optional
        obs/obsm key names. Defaults 'celltype', 'X_LDA', 'timepoint'.
    Returns
    -------
    list[np.ndarray]
        One array per route (same order as `refs_list`).
    """
    routes = []
    for refs in refs_list:
        means_centers = find_high_density_point(
            data, refs, dimension,
            celltype_key=celltype_key, lda_key=lda_key, timepoint_key=timepoint_key)
        centers_indices = {}
        for ref in refs:
            key = ref[0] if ref[1] == False else f"{ref[0]}-{ref[1]}"
            target_row = np.array(means_centers[key])
            indices = np.where((data.obsm[lda_key][:, :dimension] == target_row).all(axis=1))[0][0]
            if ref[1] == False:
                centers_indices[ref[0]] = indices
            else:
                centers_indices[f'{ref[0]}-{ref[1]}'] = indices
        xnew, unst = unstable_ges(data, refs, centers_indices, sorder, quadratic)
        unst_proj = lda_unst(data, np.array(unst).T, celltype_key=celltype_key)
        routes.append(unst_proj)
    return routes
def render_plot(data, unstx, unsty, kde, s0, color_unst, h, colorpalette,
                figsize=(5, 5), celltype_key='celltype', lda_key='X_LDA'):
    """
    Draw the LDA sub-landscape and overlay one route with annotated waypoints.
    Parameters
    ----------
    data : AnnData
        Sub-landscape with obsm[lda_key].
    unstx, unsty : np.ndarray
        Route coordinates (LD1, LD2).
    kde : bool
        Overlay KDE contours on the scatter.
    s0 : float
        Scatter point size.
    color_unst : str
        Colour of the route line.
    h : list[float]
        [x, y] offsets for the waypoint labels.
    colorpalette : dict
        Cell-type -> colour map.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default (5, 5).
    celltype_key, lda_key : str, optional
        obs/obsm key names. Defaults 'celltype', 'X_LDA'.
    """
    sc.set_figure_params(scanpy=True, fontsize=20)
    fig, ax = plt.subplots(figsize=figsize)
    plot_lda(ax, data, None, kde, s0, colorpalette, celltype_key=celltype_key, lda_key=lda_key)
    ax.plot(unstx, unsty, linewidth=3, color=color_unst, zorder=1)
    ax.scatter(unstx, unsty, color='k', s=15, edgecolor='k', zorder=2)
    for i in [0, 5, 10, 15, 20]:
        ax.annotate(str(i), (unstx[i] - h[0], unsty[i] + h[1]), color='k', fontweight='bold', fontsize=15,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), zorder=4)
        ax.scatter(unstx[i], unsty[i], color='white', s=25, edgecolor='k', zorder=3)
    ax.add_patch(patches.Circle([unstx[0], unsty[0]], 0.8, edgecolor='black', facecolor='none', linewidth=2, linestyle='dashed'))
    plt.show()
def _ball_average(data, gene, unst, dimension, radius, layer='total', lda_key='X_LDA'):
    """Mean expression of one gene in a ball (radius) around each route point."""
    # The gene must be present: the sub-landscape only contains its module genes.
    if gene not in data.var_names:
        raise ValueError(
            f"Gene '{gene}' not found in data.var_names. It must be in the module used "
            f"to build this sub-landscape. If you changed the module, rebuild the "
            f"sub-landscape (re-run the subset + perform_lda cells) before plotting."
        )
    # layer='total' needs a raw-counts layer you created (adata.layers['total'] =
    # adata_counts.X before normalising); layer=None uses data.X (log-normalised).
    if layer is None:
        expr = data.X
    elif layer not in data.layers:
        raise ValueError(
            f"Layer '{layer}' not found in data.layers. Create it (e.g. "
            f"adata.layers['{layer}'] = adata_counts.X before normalising) "
            f"or pass layer=None to use the log-normalised data.X."
        )
    else:
        expr = data.layers[layer]
    expr = _to_dense(expr)
    tree = spatial.cKDTree(data.obsm[lda_key][:, :dimension])
    g = np.where(data.var_names == gene)[0][0]
    n_points = len(unst.T[0])
    out = np.zeros(n_points)
    for k in range(n_points):
        group = tree.query_ball_point([unst.T[d][k] for d in range(dimension)], radius)
        out[k] = expr[group, g].sum() / len(group) if len(group) else np.nan
    return out
def bootstrap_sanity_check(data, fraction, refs_list=None, celltype_key='celltype'):
    """
    Report how many cells a bootstrap subsample (`fraction`) will contain.
    Run before bootstrap_route_band / bootstrap_route_grid to check the subsample
    isn't too small. If `refs_list` is given, also reports the per-cluster count.
    Returns the approximate number of cells per subsample.
    """
    n_total = data.n_obs
    n_sub = int(round(n_total * fraction))
    print(f"Dataset: {n_total} cells. fraction={fraction} -> ~{n_sub} cells per subsample.")
    if refs_list is not None:
        clusters = {ref[0] for refs in refs_list for ref in refs}
        print("Approx. cells per route cluster in a subsample:")
        for cl in sorted(clusters):
            n_cl = int((data.obs[celltype_key] == cl).sum())
            print(f"  {cl:>14}: {n_cl} total -> ~{int(round(n_cl * fraction))} per subsample")
    return n_sub
def bootstrap_route_band(gene, refs_list, data, colors_unst, dimension=2, sorder=2,
                         quadratic=True, n_bootstrap=40, fraction=0.12, radius=0.8,
                         ylim=None, layer='total', T=None, splrep_s=1, labels_unst=None,
                         celltype_key='celltype', lda_key='X_LDA', timepoint_key='timepoint',
                         ax=None, figsize=(4, 4)):
    """
    Bootstrap the expression of one gene along each route and plot mean +/- SD bands.
    The sub-landscape is repeatedly subsampled; for each subsample every route is
    recomputed and the gene profile along it is averaged in balls and smoothed. The
    per-bootstrap curves give a mean line and a shaded SD band per route.
    Curves are combined across bootstraps with np.nanmean/np.nanstd, so a point
    that was empty in some subsamples (no cells in its ball) is averaged over only
    the subsamples that did have cells, instead of turning the whole curve to NaN.
    Treated in the paper, Appendix A (Data Analysis), Section A2.
    Parameters
    ----------
    gene : str
        Gene to profile (must be in the module of this sub-landscape).
    refs_list : list[list[tuple]]
        One entry per route; each is a list of (celltype, day) tuples.
    data : AnnData
        Sub-landscape with obsm[lda_key] (output of perform_lda).
    colors_unst : list[str]
        One colour per route.
    dimension, sorder, quadratic : see compute_routes / unstable_ges.
    n_bootstrap : int
        Subsampling iterations (default 40). Lower it (e.g. 5-10) for quick looks.
    fraction : float
        Subsample fraction per iteration. Default 0.12.
    radius : float
        Ball radius for the neighbour average. Default 0.8. NOTE: the right value
        depends on the LDA coordinate scale, which differs between datasets; for
        sparser data (e.g. human) you often need a larger radius (e.g. 1.5-3).
        If a route comes out empty, the function prints the radius it needs.
    ylim : float or None
        Upper y-limit (lower fixed at 0). None = auto.
    layer : str or None
        Expression matrix: layer name (raw counts) or None for data.X.
    T : np.ndarray or None
        Parameter grid; if None, derived from the route length (adaptive).
    splrep_s : float
        Smoothing factor for the per-curve spline. Default 1.
    labels_unst : list[str] or None
        Legend label per route. If None, ['Route 1', 'Route 2', ...]. Any number.
    celltype_key, lda_key, timepoint_key : str
        obs/obsm key names.
    ax : matplotlib axis or None
        Axis to draw on; created if None.
    figsize : tuple
        Figure size used only when ax is None.
    Returns
    -------
    ax : matplotlib axis
    """
    if labels_unst is None:
        labels_unst = [f'Route {r + 1}' for r in range(len(refs_list))]
    curves_per_route = {r: [] for r in range(len(refs_list))}
    for i in range(n_bootstrap):
        sub = sc.pp.subsample(data, fraction=fraction, copy=True, random_state=i)
        routes = compute_routes(sub, refs_list, dimension, sorder, quadratic,
                                celltype_key=celltype_key, lda_key=lda_key,
                                timepoint_key=timepoint_key)
        if T is None:
            T = np.arange(len(routes[0]))  # adaptive: follow the route length
        for r, route in enumerate(routes):
            power = _ball_average(sub, gene, route, dimension, radius, layer=layer, lda_key=lda_key)
            curves_per_route[r].append(BSpline(*splrep(T, power, s=splrep_s))(T))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    empty_routes = set()
    for r, color in enumerate(colors_unst):
        curves = np.asarray(curves_per_route[r])
        # nanmean/nanstd: a point empty in some subsamples is averaged over the rest,
        # so a few empty balls don't blank the whole curve.
        with np.errstate(invalid='ignore'):
            mean = np.nanmean(curves, axis=0)
            sd = np.nanstd(curves, axis=0)
        ax.plot(T, mean, color=color, lw=3, label=labels_unst[r])
        ax.fill_between(T, mean - sd, mean + sd, color=color, alpha=0.20)
        if np.all(np.isnan(mean)):
            empty_routes.add(r)
    # If a route was empty at EVERY point in every subsample, advise the radius needed.
    if empty_routes:
        full_routes = compute_routes(data, refs_list, dimension, sorder, quadratic,
                                     celltype_key=celltype_key, lda_key=lda_key,
                                     timepoint_key=timepoint_key)
        need = _suggest_radius(data, full_routes, dimension, lda_key=lda_key)
        for r in sorted(empty_routes):
            print(f"Route {r+1} ({labels_unst[r]}) is empty at radius={radius}: its balls "
                  f"caught no cells. Try radius >= {need[r]:.2f}.")
    ax.set(xticks=np.arange(0, len(T), 5))
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.set_title(gene, fontsize=20)
    ax.set_xlabel('Ball indices', fontsize=14)
    ax.set_ylabel('Average counts', fontsize=14)
    ax.legend(frameon=False)
    ax.grid(False)
    return ax
def bootstrap_route_grid(genes, refs_list, data, colors_unst, high=None, dimension=2,
                         sorder=2, quadratic=True, n_bootstrap=40, fraction=0.12, radius=0.8,
                         layer='total', T=None, splrep_s=1, labels_unst=None,
                         celltype_key='celltype', lda_key='X_LDA', timepoint_key='timepoint',
                         figsize=(13, 10)):
    """
    Bootstrap many genes along the routes and draw one mean +/- SD panel per gene.
    The route bootstrap is done ONCE per subsample and all genes are averaged within
    that same subsample (efficient vs. calling bootstrap_route_band per gene).
    Curves are combined across bootstraps with np.nanmean/np.nanstd (see
    bootstrap_route_band), so a few empty balls don't blank a curve.
    Parameters
    ----------
    genes : list[list[str]]
        Grid of gene names; shape defines the subplot grid. Every gene must be in
        the module of this sub-landscape.
    high : list[list[float]] or None
        Per-panel y-axis upper limit, same shape as `genes`. If None (default),
        each panel auto-scales to 1.1 * max(mean + SD). Normally leave None.
    radius : float
        Ball radius. Default 0.8. The right value depends on the LDA scale, which
        differs between datasets; sparser data (e.g. human) often needs a larger
        radius. If a route comes out empty, the function prints the radius needed.
    Other parameters: as in bootstrap_route_band.
    Returns
    -------
    fig, ax
    """
    if labels_unst is None:
        labels_unst = [f'Route {r + 1}' for r in range(len(refs_list))]
    n_rows = len(genes)
    n_cols = max(len(row) for row in genes)
    flat_genes = [g for row in genes for g in row]
    curves = {g: {r: [] for r in range(len(refs_list))} for g in flat_genes}
    for i in range(n_bootstrap):
        sub = sc.pp.subsample(data, fraction=fraction, copy=True, random_state=i)
        routes = compute_routes(sub, refs_list, dimension, sorder, quadratic,
                                celltype_key=celltype_key, lda_key=lda_key,
                                timepoint_key=timepoint_key)
        if T is None:
            T = np.arange(len(routes[0]))  # adaptive
        for r, route in enumerate(routes):
            for g in flat_genes:
                power = _ball_average(sub, g, route, dimension, radius, layer=layer, lda_key=lda_key)
                curves[g][r].append(BSpline(*splrep(T, power, s=splrep_s))(T))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    ax = np.atleast_2d(ax)
    plt.subplots_adjust(hspace=0.1)
    empty_routes = set()
    for row in range(n_rows):
        for col in range(len(genes[row])):
            g = genes[row][col]
            a = ax[row, col]
            panel_max = 0.0
            for r, color in enumerate(colors_unst):
                cc = np.asarray(curves[g][r])
                # nanmean/nanstd: ignore empty-ball points from individual subsamples
                with np.errstate(invalid='ignore'):
                    mean = np.nanmean(cc, axis=0)
                    sd = np.nanstd(cc, axis=0)
                a.plot(T, mean, color=color, lw=2, label=labels_unst[r])
                a.fill_between(T, mean - sd, mean + sd, color=color, alpha=0.20)
                if np.all(np.isnan(mean)):
                    empty_routes.add(r)        # nothing to draw, no y-limit contribution
                else:
                    panel_max = max(panel_max, float(np.nanmax(mean + sd)))
            a.set(xticks=np.arange(0, len(T), 5), title=g)
            if high is not None:
                a.set_ylim(0, high[row][col])
            elif panel_max > 0:
                a.set_ylim(0, 1.1 * panel_max)
            a.grid(False)
            a.set_title(g, fontsize=20)
    # If any route was empty everywhere (no cells in its balls), advise a radius.
    if empty_routes:
        full_routes = compute_routes(data, refs_list, dimension, sorder, quadratic,
                                     celltype_key=celltype_key, lda_key=lda_key,
                                     timepoint_key=timepoint_key)
        need = _suggest_radius(data, full_routes, dimension, lda_key=lda_key)
        for r in sorted(empty_routes):
            print(f"Route {r+1} ({labels_unst[r]}) is empty at radius={radius}: its balls "
                  f"caught no cells. Try radius >= {need[r]:.2f}.")
    ax[n_rows - 1, 0].set_ylabel("Average counts", fontsize=18)
    ax[n_rows - 1, 0].set_xlabel("Ball indices", fontsize=18)
    plt.tight_layout(pad=1.0)
    plt.show()
    return fig, ax
