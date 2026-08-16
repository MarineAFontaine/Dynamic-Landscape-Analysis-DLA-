### flow_functions.py -- helpers for the flow cytometry analysis notebooks.
### Marine Fontaine (University of Warwick)
#
# Used by the `FlowAnalysis_*.ipynb` notebooks. This covers Appendix A (Data analysis)
# in [1]:
#     A3  Flow cytometry analysis (A3.1 clustering, A3.2 cell identities, A3.3 quality),
#     A4  Decision landscape and transitions in flow cytometry data in response to SAG
#         perturbations.
#
# [1] M. Fontaine, J.M. Delas, M. Saez, R.J. Maizels, E. Finnie, J. Briscoe, D.A. Rand,
#     (2025). *Appendix A: Data analysis*. figshare doi:
#     https://doi.org/10.6084/m9.figshare.31342810

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
from scipy import spatial
from sklearn.mixture import GaussianMixture


# Functions for cell states proportions

def proportions(data, states, days):
    """Proportion of cells in each state at each timepoint.

    Parameters
    ----------
    data : AnnData
        Annotated flow cytometry data; `obs` must carry 'timepoint' and 'celltype'.
    states : list[str]
        Cell states to count (one column of the result per state).
    days : list[str]
        Timepoints to report (one row per day; a missing day gives zeros).
    """
    # Initialise results dictionary with states
    results = {state: [] for state in states}
    grouped_by_day = data.obs.groupby('timepoint', observed=True)
    for day in days:
        try:
            day_data = grouped_by_day.get_group(day)
            total_cells = len(day_data)

            for state in states:
                # Filter by state within the day's data
                state_count = len(day_data[day_data['celltype'] == state])
                proportion = state_count / total_cells if total_cells > 0 else 0
                results[state].append(proportion)
        except KeyError:
            # Handle case where 'day' does not exist
            for state in states:
                results[state].append(0)

    return pd.DataFrame(results, index=days)


def plot_data(ax, data, title, x_labels, color_dict):
    """Stacked bar plot of cell-state proportions per timepoint (one axis)."""
    df = data
    df.plot.bar(stacked=True, color=[color_dict[col] for col in df.columns], rot=0, ax=ax, edgecolor="black", linewidth=1, width=0.8)
    ax.set_title(title, fontsize=40, pad=5)
    ax.get_legend().remove()
    ax.set_xticklabels(x_labels, rotation=0, fontsize=40)
    ax.tick_params(axis='y', labelsize=30)
    ax.grid(False)
    # Adding labels to the bars
    for container in ax.containers:
        labels = [f"{int(bar.get_height())}" if bar.get_height() > 4 else '' for bar in container]
        ax.bar_label(container, labels=labels, label_type='center', fontsize=25, color='white', fontweight="bold")


## Functions for gene plots

def cluster_gs(data, cluster_ref, per):
    """The `per` percent of cells of `cluster_ref` nearest to its Gaussian mean.

    Fits a single-component Gaussian to the cluster and keeps the cells closest to that
    mean, which trims the transitional cells at the cluster edge before plotting gene
    expression (Appendix A Section A3.3).
    """
    cluster = data[data.obs['celltype'] == cluster_ref]

    # Assuming GMM returns means and you are using the first mean
    gmm = GaussianMixture(n_components=1).fit(cluster.X.A)
    mean_clust = gmm.means_[0]
    point_tree = spatial.cKDTree(cluster.X.A)

    # Take `per` % of cells nearest to the mean, ensure k is an integer
    k_value = int(round(len(cluster) * per / 100))
    nearest_dist, nearest_idx = point_tree.query(mean_clust, k=k_value)
    return cluster[nearest_idx, :]


## Functions for Linear Discriminant Analysis

def perform_lda(data, cluster_list):
    """Project the clusters in `cluster_list` into LDA space.

    Returns the extracted sub-landscape with the LDA coordinates stored in
    `obsm['X_LDA']` (Appendix A Section A1.5).
    """
    # extract sublandscape
    sublandscape = data[data.obs['celltype'].isin(cluster_list)].copy()
    X_data = sublandscape.X.A
    y_target = sublandscape.obs['celltype']
    # perform LDA
    lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
    X_r2 = lda.fit(X_data, y_target).transform(X_data)
    sublandscape.obsm['X_LDA'] = X_r2
    return sublandscape


def plot_lda(ax, data, obs_key, day, label_suffix, color_dict):
    """Scatter + KDE of one (SAG condition, day) sample in LDA coordinates."""
    subset = data[(data.obs['timepoint'] == day) & (data.obs['signal'] == obs_key)]
    x = subset.obsm['X_LDA'][:, 0]
    y = subset.obsm['X_LDA'][:, 1]
    celltype_colors = [color_dict[ci] for ci in subset.obs.celltype]

    # Scatter plot with KDE
    ax.scatter(x, y, c=celltype_colors, s=100, alpha=0.8)
    ax.scatter(data.obsm['X_LDA'][:, 0], data.obsm['X_LDA'][:, 1], alpha=0.01, c='gray', s=200, zorder=-1)
    sns.kdeplot(x=x, y=y, color='k', levels=10, linewidths=5.5, alpha=0.5, ax=ax)

    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_title(f'SAG{label_suffix} {day}', fontweight="bold", fontsize=100)
    ax.grid(False)
