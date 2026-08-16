### fitting_core.py -- model-agnostic machinery for the landscape fits.
### Marine Fontaine (University of Warwick)
#
# This file contains NO specific landscape. You supply your own `compute_landscape`
# (a function (x, y, params, perturb_std) -> (L_x, L_y, noise), normally @njit) and
# pass it to make_model(). The same machinery then fits ANY landscape.
#
# A landscape file just needs:
#     @njit
#     def compute_landscape(x, y, params, perturb_std):
#         ...
#         return L_x, L_y, noise
# and a default parSDE dict (see make_parSDE) and a cellstates dict.
#
# Reproducibility note: the initial cell positions are seeded (make_blobs,
# random_state=1), but the SDE noise (np.random.normal in update_positions) and
# the per-step parameter perturbation inside compute_landscape are NOT seeded.
# Runs are therefore stochastic by design (ABC needs stochastic simulations), so
# do not expect bit-identical output between runs.

import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.datasets import make_blobs
from numba.typed import Dict


def make_parSDE(T=4.0, dt=0.01, ncells=500, measurements=None, perturb_std=0.02):
    """Build the SDE settings dict.

    Parameters
    ----------
    T, dt : float
        Total integration time and time step (tsteps = T/dt).
    ncells : int
        Number of simulated cells.
    measurements : list[int] or None
        Time-step indices recorded as the experimental timepoints. Defaults to 5
        evenly spaced steps.
    perturb_std : float
        Std of the per-step Gaussian perturbation applied to each parameter inside
        `compute_landscape` (0 turns it off). Set this to control the model noise.
    """
    tsteps = int(T / dt)
    if measurements is None:
        measurements = list(np.linspace(0, tsteps - 1, 5).astype(int))
    return {
        'dt': dt, 'T': T, 'tsteps': tsteps,
        't': np.linspace(0., T, tsteps), 'sqrtdt': np.sqrt(dt),
        'ncells': ncells, 'measurements': measurements,
        'perturb_std': perturb_std,
    }


def get_idx_remove(states_byDay, states, days, drop=()):
    """Indices of '<state>-<day>' entries to drop, matched on the *known* state and
    day lists rather than by substring.

    `states_byDay` is the flattened list built as
    `[f'{state}-{day}' for day in days for state in states]`.
    `drop` may contain state names (drop that state at every day) and/or day tokens
    (drop every state at that day). Matching is exact against `states` / `days`, so
    dropping 'p3' will NOT also drop 'Earlyp3', and state names may contain hyphens.

    Parameters
    ----------
    states_byDay : list[str]
        The flattened '<state>-<day>' labels, in the order the simulation produces.
    states : list[str]
        The known cell-state names.
    days : list[str]
        The known day/timepoint tokens.
    drop : iterable[str]
        State names and/or day tokens to drop.
    """
    drop = set(drop)
    states_set, days_set = set(states), set(days)
    unknown = drop - states_set - days_set
    if unknown:
        raise ValueError(
            f"get_idx_remove: {sorted(unknown)} are neither states {states} "
            f"nor days {days}. Check the tokens passed in `drop`.")

    drop_states = drop & states_set
    drop_days = drop & days_set

    # Rebuild the labels with their (state, day) split known explicitly, so we never
    # have to parse a label string (state names may contain '-').
    idx = []
    k = 0
    for day in days:
        for state in states:
            if state in drop_states or day in drop_days:
                idx.append(k)
            k += 1
    if k != len(states_byDay):
        raise ValueError(
            f"get_idx_remove: states_byDay has {len(states_byDay)} entries but "
            f"len(states)*len(days) = {k}. They must be built from the same lists.")
    return idx


def update_positions(compute_landscape, x, y, params, parSDE):
    """Euler-Maruyama integration of the SDE using the supplied `compute_landscape`."""
    dt, sqrtdt, tsteps, ncells = parSDE['dt'], parSDE['sqrtdt'], parSDE['tsteps'], parSDE['ncells']
    perturb_std = parSDE['perturb_std']
    for i in range(tsteps - 1):
        L_x, L_y, noise = compute_landscape(x[:, i], y[:, i], params, perturb_std)
        x[:, i + 1] = x[:, i] + dt * (-L_x) + sqrtdt * noise * np.random.normal(size=ncells)
        y[:, i + 1] = y[:, i] + dt * (-L_y) + sqrtdt * noise * np.random.normal(size=ncells)
    return x, y


def SDE(compute_landscape, params, parSDE, samples, start_centers, cluster_std=0.05):
    """Simulate all cells from the given start clusters; return positions at the
    measurement times. `samples`/`start_centers` set the initial mixture."""
    if len(samples) != len(start_centers):
        raise ValueError(
            f"SDE: samples ({len(samples)}) and start_centers ({len(start_centers)}) "
            f"must have the same length (one count per start cluster).")
    if sum(samples) != parSDE['ncells']:
        raise ValueError(
            f"SDE: samples must sum to parSDE['ncells'] ({parSDE['ncells']}), "
            f"got sum(samples) = {sum(samples)}.")
    stds = [cluster_std] * len(start_centers)
    X, _ = make_blobs(n_samples=samples, cluster_std=stds, centers=start_centers,
                      n_features=2, random_state=1)
    x = np.zeros((parSDE['ncells'], parSDE['tsteps']))
    y = np.zeros((parSDE['ncells'], parSDE['tsteps']))
    x[:, 0], y[:, 0] = X[:, 0], X[:, 1]
    x, y = update_positions(compute_landscape, x, y, params, parSDE)
    return np.column_stack((x[:, parSDE['measurements']].flatten('F'),
                            y[:, parSDE['measurements']].flatten('F')))


def get_index_range(block_size, index):
    """Row indices of measurement block `index` in the stacked SDE output."""
    return range(block_size * index, block_size * (index + 1))


def make_model(compute_landscape, cellstates, parSDE, samples, start_centers,
               radius=0.4, states=None):
    """Build a PyABC `model(pars)` for YOUR landscape.

    Parameters
    ----------
    compute_landscape : callable
        Your landscape function (x, y, params, perturb_std) -> (L_x, L_y, noise).
        Normally @njit. This is where you 'insert the model' -- pass your own function.
    cellstates : dict
        Attractor name -> [x, y] position.
    parSDE : dict
        SDE settings (use make_parSDE()).
    samples : list[int]
        Cells per start cluster (must sum to parSDE['ncells']).
    start_centers : list[tuple]
        (x, y) of each start cluster, same order/length as `samples`.
    radius : float
        Ball radius for counting cells near each attractor.
    states : list[str] or None
        Cluster order; defaults to list(cellstates). NOTE: this order must match the
        column order of your experimental data, because the fit compares the two
        position-by-position. If you reorder the cellstates dict, reorder the data too.

    Returns
    -------
    model(pars) : callable for ABCSMC(models=model, ...).
    """
    if states is None:
        states = list(cellstates)

    missing = [s for s in states if s not in cellstates]
    if missing:
        raise ValueError(f"make_model: states {missing} are not in cellstates.")

    def model(pars):
        numba_pars = Dict()
        for k, v in pars.items():
            numba_pars[k] = float(v)

        S = SDE(compute_landscape, numba_pars, parSDE, samples, start_centers)

        kdtree = spatial.cKDTree(S)
        ball_index = [kdtree.query_ball_point(cellstates[s], radius) for s in states]

        proportions = []
        for i in range(len(parSDE['measurements'])):
            day_index = list(get_index_range(parSDE['ncells'], i))
            total = [len(np.intersect1d(ball_index[s], day_index)) for s in range(len(ball_index))]
            # Proportions are kept at full precision (not rounded) so small-state
            # differences are not quantised away before ABC compares.
            proportions.append(np.divide(total, parSDE['ncells']).tolist())

        return {"X_2": np.array(proportions).flatten()}

    return model


def make_distance(idx_remove=()):
    """Return a PyABC distance(simulation, data) that drops `idx_remove` entries
    from the simulation before comparing (pass () to keep all)."""
    idx = list(idx_remove)

    def distance(simulation, data):
        sim = np.delete(simulation['X_2'], idx, 0) if idx else np.asarray(simulation['X_2'])
        # Flatten both so a (1, N) data row-vector and an (N,) simulation vector
        # compare as the same flat array (PyABC may store the data 2-D).
        sim = np.asarray(sim).ravel()
        obs = np.asarray(data['X_2']).ravel()
        if sim.size != obs.size:
            raise ValueError(
                f"make_distance: simulated vector has {sim.size} entries after "
                f"dropping {len(idx)}, but data has {obs.size}. "
                f"Check that idx_remove matches the dropped timepoints/states.")
        return np.absolute(sim - obs).sum()

    return distance


def process_simulations(n, history, Xdata, parSDE, states, idx_remove=()):
    """Pull n accepted sims from a PyABC history; return them, distances, and the
    mean proportions (%, timepoints x states). Pass the same idx_remove used to fit."""
    distance = make_distance(idx_remove)
    sum_stats = history.get_weighted_sum_stats_for_model(m=0, t=history.max_t)
    simulation_accepted = np.array([sum_stats[1][i]['X_2'].tolist() for i in range(n)])
    dist = np.array([distance(sum_stats[1][i], Xdata) for i in range(n)])
    column_means = np.mean(simulation_accepted, axis=0)

    expected = len(parSDE['measurements']) * len(states)
    if column_means.size != expected:
        raise ValueError(
            f"process_simulations: simulated vector length {column_means.size} does "
            f"not equal len(measurements)*len(states) = "
            f"{len(parSDE['measurements'])}*{len(states)} = {expected}. "
            f"Check that `states` matches the cellstates used to fit.")
    return simulation_accepted, dist, np.reshape(column_means * 100,
                                                 (len(parSDE['measurements']), len(states)))


def simulation_todf(states, x_labels, Xsim):
    """Wrap simulated proportions into a DataFrame with '<state>-<day>' columns."""
    states_byDay = [f"{state}-{day}" for day in x_labels for state in states]
    return pd.DataFrame(np.array(Xsim), columns=states_byDay)
