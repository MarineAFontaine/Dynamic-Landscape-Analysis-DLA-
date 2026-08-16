### Plotting helpers for all the model-fitting notebooks (flow cytometry, scRNA-seq, human,
### first-decisions). All fitting plots live in this one file.
### Marine Fontaine (University of Warwick)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats


def plot_data(ax, data, title, x_labels, palette, background=None):
    """
    Stacked bar plot of cell-state proportions per timepoint (one axis).

    Used to show experimental vs simulated proportions side by side; pass
    background='data' or 'simulation' to tint the panel and label it accordingly.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw on.
    data : pandas.DataFrame
        Rows = timepoints, columns = cell states; values are proportions (0-100).
    title : str
        Panel title (the word "Experiments"/"Simulation" is appended by `background`).
    x_labels : list[str]
        Tick labels for the timepoints.
    palette : dict
        Cell-state name -> colour (one per column of `data`).
    background : {'data', 'simulation', None}
        Tints the panel and sets the title suffix. None leaves it plain.
    """
    df = data
    df.plot.bar(
        stacked=True,
        color=[palette[col] for col in df.columns],
        rot=0,
        ax=ax,
        edgecolor="black",
        linewidth=1,
        width=0.7,
        alpha=1
    )

    ax.get_legend().remove()
    ax.set_xticklabels(x_labels, rotation=45, fontsize=50)
    ax.tick_params(axis='y', labelsize=0)    # hide y tick labels (proportions shown in-bar)
    ax.grid(False)
    # Panel tint + title suffix depending on data vs simulation
    if background == "data":
        ax.set_facecolor('#fffacd')          # light yellow
        ax.set_title(f'{title}\n Experiments', fontsize=40, pad=20)
    if background == "simulation":
        ax.set_facecolor('#e0f3f8')          # light blue
        ax.set_title(f'{title}\n Simulation', fontsize=40, pad=20)

    # Write each segment's value inside the bar (only if tall enough to read)
    for i, container in enumerate(ax.containers):
        for j, bar in enumerate(container):
            label = f"{int(bar.get_height())}" if bar.get_height() > 4 else ''
            ax.text(
                bar.get_x() + bar.get_width() / 2,   # x position
                bar.get_y() + bar.get_height() / 2,  # y position
                label,
                ha='center', va='center', fontsize=25,
                color='white', fontweight="bold"
            )


def plot_histogram_grid(states, times, simulation_accepted, data, colorpalette, bins):
    """
    Grid of histograms: the bootstrap/accepted simulated proportions per
    (timepoint, state), with the experimental value marked by a dashed line.

    One column per state, one row per timepoint.

    Parameters
    ----------
    states : list[str]
        Cell states (columns of the grid).
    times : list[str]
        Timepoints to show (rows of the grid).
    simulation_accepted : pandas.DataFrame or dict-like
        Indexed by '<state>-<day>'; each entry is an array of accepted simulated
        proportions (0-1) for that state/timepoint.
    data : pandas.DataFrame
        Experimental proportions (rows indexed by timepoint, columns = states).
    colorpalette : dict
        State name -> histogram colour.
    bins : int
        Number of histogram bins.
    """
    filtered_days = [f"{state}-{day}" for day in times for state in states]
    reshaped_array = np.array(filtered_days).reshape(-1, len(states))
    subset_data = data.loc[times]

    # Figure size scales with the grid
    num_rows, num_cols = len(reshaped_array), len(states)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows), sharey=True)

    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i, j]
            data_index = reshaped_array[i, j]
            sns.histplot(simulation_accepted[data_index] * 100, color=colorpalette[states[j]], bins=bins, ax=ax)
            ax.set_title(f'{data_index}', fontsize=25)

            # Dashed line at the experimental value
            vertical_line_x = subset_data.values[i, j]
            ax.vlines(x=vertical_line_x, ymin=0, ymax=700, colors='k', linestyle='dashed', linewidth=3,
                      label=f"data {round(vertical_line_x, 2)}")

            ax.grid(False)
            ax.set_xlim(0, 40)
            ax.set_ylim(0, 700)
            ax.legend(fontsize=15)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_xlabel("", fontsize=14)
            ax.set_ylabel("", fontsize=14)

    axes[-1, 0].set_xlabel("Proportions", fontsize=20)
    axes[-1, 0].set_ylabel("Count", fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_tendencies(states, x_labels, Xsim, data, colorpalette):
    """Per-state tendency plot: data vs simulation mean, with a 5-95 percentile band.

    Parameters
    ----------
    states : list[str]
        Cell states (one panel per state).
    x_labels : list[str]
        Timepoint labels along the x-axis.
    Xsim : array-like
        Accepted simulated proportions; reshaped to '<state>-<day>' columns.
    data : pandas.DataFrame
        Experimental proportions (rows indexed by timepoint, columns = states).
    colorpalette : dict
        State name -> colour.
    """
    states_byDay = [f"{state}-{day}" for day in x_labels for state in states]
    reshaped_array = np.array(states_byDay).reshape(-1, len(states))
    simulation_accepted = pd.DataFrame(np.array(Xsim), columns=states_byDay)

    times = list(range(len(x_labels)))
    subset_data = data.loc[x_labels]

    num_rows, num_cols = len(reshaped_array), len(states)
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 5), sharey=False)

    for j in range(num_cols):
        minq_val, maxq_val, min_val, max_val, data_average, means = [], [], [], [], [], []
        for i in range(num_rows):
            data_index = reshaped_array[i, j]
            data_values = simulation_accepted[data_index]
            min_val.append(min(data_values))
            max_val.append(max(data_values))
            minq_val.append(np.percentile(data_values, 5))
            maxq_val.append(np.percentile(data_values, 95))
            data_average.append(subset_data.values[i, j] / 100)
            means.append(np.mean(data_values))

        ax = axes[j]
        ax.plot(times, data_average, label='data', color='b', linestyle='-', linewidth=4)
        ax.plot(times, means, label='simulation mean', color=colorpalette[states[j]], linestyle='--', linewidth=4)
        ax.fill_between(times, minq_val, maxq_val, color=colorpalette[states[j]], alpha=0.5, label='5-95 Percentile Range')
        ax.fill_between(times, min_val, minq_val, color=colorpalette[states[j]], alpha=0.1)
        ax.fill_between(times, maxq_val, max_val, color=colorpalette[states[j]], alpha=0.1)
        ax.set_title(f"{states[j]}", fontsize=40)
        ax.set_xticks(times)
        ax.set_xticklabels(x_labels, fontsize=30, rotation=45)

    axes[0].set_xlabel("Days", fontsize=35)
    axes[0].set_ylabel("Proportions", fontsize=35)
    axes[-1].legend(fontsize=30, loc='upper left', bbox_to_anchor=(1, 1.3))
    plt.tight_layout()
    plt.show()


def plot_parameters(df, ref_par=None, lim_par=None, lim_priors=None):
    """Posterior parameter densities (KDE contours + scatter) per sub-landscape.

    Defaults plot the NMP (u1,v1) and PreNeural (u2,v2) parameter pairs (the
    first-decisions model). Pass ref_par / lim_par / lim_priors to plot other
    parameter pairs / axis limits / prior boxes.

    Parameters
    ----------
    df : pandas.DataFrame
        Accepted posterior parameter samples (columns include the names in ref_par).
    ref_par : list[list[str]] or None
        Pairs of parameter names to plot, one panel each. Default [['u1','v1'],['u2','v2']].
    lim_par, lim_priors : list or None
        Axis limits and prior-box limits per panel (same length as ref_par).
    """
    if ref_par is None:
        ref_par = [['u1', 'v1'], ['u2', 'v2']]
    if lim_par is None:
        lim_par = [[[-4.5, 4.5], [-2.5, 0]], [[-4.5, 4.5], [-2.5, 0]]]
    if lim_priors is None:
        lim_priors = [[[-1, 1], [-2.5, -1]], [[-1, 1], [-2.5, -1]]]

    fig, axes = plt.subplots(1, len(ref_par), figsize=(15 * len(ref_par), 5), dpi=300)
    if len(ref_par) == 1:
        axes = [axes]
    plt.rcParams.update({'lines.linewidth': 2, 'xtick.labelsize': 20, 'ytick.labelsize': 20})

    def plot_density(ax, x, y, z, levels, cmap, contour_color, label, max_idx):
        ax.contourf(x, y, z, levels=levels, cmap=cmap, alpha=0.3)
        ax.contour(x, y, z, levels=levels, linewidths=2, colors=contour_color)
        ax.scatter(x.ravel()[max_idx], y.ravel()[max_idx], marker="X", edgecolor='black', s=500, c=contour_color, label=label, linewidth=2, zorder=20)

    for idx, ref in enumerate(ref_par):
        ax = axes[idx]
        X = df[ref].values
        kernel = stats.gaussian_kde(np.vstack([X[:, 0], X[:, 1]]))
        x, y = np.mgrid[lim_par[idx][0][0]:lim_par[idx][0][1]:100j, lim_par[idx][1][0]:lim_par[idx][1][1]:100j]
        z = kernel(np.vstack([x.ravel(), y.ravel()])).reshape(x.shape)
        max_idx = np.argmax(z.ravel())
        levels = np.linspace(0, z.max(), 8)[1:]

        plot_density(ax, x, y, z, levels, 'Blues', 'blue', 'RNAseq', max_idx)
        ax.scatter(X[:, 0], X[:, 1], color='blue', s=1, alpha=0.5)

        xlim, ylim = lim_priors[idx]
        rect = patches.Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                                 linewidth=2, edgecolor='black', linestyle='--', facecolor='none', alpha=0.8, label="Priors")
        ax.add_patch(rect)
        ax.set_facecolor('white')
        ax.grid(False)
        ax.set_xlabel(ref[0], fontsize=25, fontweight='bold')
        ax.set_ylabel(ref[1], fontsize=25, fontweight='bold')
        ax.set_xlim(lim_par[idx][0])
        ax.set_ylim(lim_par[idx][1])

    axes[0].legend(facecolor='white', framealpha=1, fontsize=30, loc="lower right")
    plt.tight_layout()
    plt.show()
