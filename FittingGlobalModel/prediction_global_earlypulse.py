### prediction_global_earlypulse.py -- global landscape model prediction for the
### Early Pulse (SAGUP) experiment.
### Marine Fontaine (University of Warwick)
#
# Used by `ModelPrediction_Flow_EarlyPulse.ipynb`. This covers Appendix B (Modelling &
# estimation) Section B6 (Model predictions using flow cytometry data) in [1]:
#     B6.1  Early pulse experiment,
#     B6.2  Model simulation and prediction.
#
# Unlike `fitting_core.py`, this file is NOT model-agnostic: the global landscape and
# the SAGUP protocol are written in directly, because the prediction is a time-varying
# simulation rather than a fit. Cells see 0nM SAG, then 500nM SAG from step T1, then
# 0nM SAG again from step T2; the landscape parameters are interpolated between the
# posterior means fitted at constant 0nM and constant 500nM (see the notebooks
# `Fitting_Flow_SAG0.ipynb` and `Fitting_Flow_SAG500.ipynb`) using the bump function
# memory1 * (1 - memory2).
#
# Only T1 and T2 are fitted here; every other parameter is taken from the constant-SAG
# fits, which is what makes this a prediction rather than a new fit.
#
# [1] M. Fontaine, J.M. Delas, M. Saez, R.J. Maizels, E. Finnie, J. Briscoe, D.A. Rand,
#     (2025). Dynamic Landscape Analysis of Cell Fate Decisions: Predictive Models of
#     Neural Development From Single-Cell Data. bioRxiv.
#     https://doi.org/10.1101/2025.05.28.656648

import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.datasets import make_blobs

### Attractor positions of the global landscape (one [x, y] per cell state)
cellstates = {
    "PreNeural":   [-0.2, 4.85763],
    "p0/p1":       [-1.37061, 2.8106],
    "p2":          [-1.84471, 1.13623],
    "pMN":         [-2.56928, 0.0889825],
    "MNDiff":      [-3.34239, -0.520129],
    "DP":          [-1.17209, 0.13729],
    "EarlyVentral":[1.37061, 2.8106],
    "Early p3":    [1.70351, 0.457739],
    "p3":          [0.0443102, -0.34472],
    "FP":          [1.83068, -1.3856],
}

### Parameters obtained from the constant-SAG fits (Appendix B Section B5)
wd = 'Experimental_and_Simulated_Proportions_Flow/'  # data sits next to this file

adata0 = pd.read_csv(wd + 'Flow_SAG0_parameters.csv')
adata500 = pd.read_csv(wd + 'Flow_SAG500_parameters.csv')

states = list(cellstates.keys())
timepoints = ['D3', 'D3.5', 'D4', 'D5', 'D6', 'D7']
states_byDay = [f"{state}-{day}" for day in timepoints for state in states]
# Indices to remove: D3 (not D3.5), D7, and FP (FP is held out as a prediction).
idx_remove = [i for i, day in enumerate(states_byDay)
              if ('D3' in day and 'D3.5' not in day) or 'D7' in day or 'FP' in day]

### SDE settings (time step, total time, number of cells, measurement times)
parSDE = {}
parSDE['dt'] = .01  # Time step.
parSDE['T'] = 4.  # Total time.
parSDE['tsteps'] = int(parSDE['T'] / parSDE['dt'])  # Number of time steps.
parSDE['t'] = np.linspace(0., parSDE['T'], parSDE['tsteps'])  # Vector of times.
parSDE['sqrtdt'] = np.sqrt(parSDE['dt'])
parSDE['ncells'] = 1500
parSDE['measurements'] = [0, 50, 100, 200, 300, 399]


### Landscape simulation

def compute_landscape(x, y, params):
    """Drift + noise of the global landscape at (x, y) for the given parameters.

    Returns (L_x, L_y, noise); the SDE integrates -grad(L) plus `noise` times a
    Wiener increment. See Appendix B Section B2 for the assembly of the
    sub-landscapes and Section B3 for the stochastic differential equation.
    """
    # Generate perturbed parameters
    par_perturbed = params
    # NMP sub-landscape
    L1x = -2*(x-1.5)+6*(x-1.5)*(y-6)+4*(x-1.5)**3-par_perturbed['u1']
    L1y = -2*(y-6)+(3*(x-1.5)**2-3*(y-6)**2)+4*(y-6)**3-par_perturbed['v1']

    # PreNeural sub-landscape
    L2x = (-2*x+6*x*(y-4)+4*x**3-par_perturbed['u2'])
    L2y = (-2*(y-4)+(3*x**2-3*(y-4)**2)+4*(y-4)**3-par_perturbed['v2'])

    # Circular sub-landscape
    L3x = (par_perturbed['v3']/100-2*(0.4*x)+6*(0.4*x)*((0.4*x)**2+(0.4*y-0.6)**2)**2+(0.4*x)**3*(0.4*y-0.6)-(0.4*x)*(0.4*y-0.6)**3)
    L3y = (par_perturbed['u3']/100-2*(0.4*y-0.6)+6*(0.4*y-0.6)*((0.4*x)**2+(0.4*y-0.6)**2)**2+0.25*(0.4*x)**4-1.5*(0.4*x)**2*(0.4*y-0.6)**2+0.25*(0.4*y-0.6)**4)

    # p2 transition sub-landscape
    L4x = (-2*(2*x+3.5)+2*6*(2*x+3.5)*(2*y-1)+4*(2*x+3.5)**3-par_perturbed['u4'])
    L4y = (-2*(2*y-1)+2*(3*(2*x+3.5)**2-3*(2*y-1)**2)+4*(2*y-1)**3-par_perturbed['v4'])

    # pMN sub-landscape
    L5x = (4*(x+1.8)*((-y-0.2)**2+(x+1.8)**2)-8*(x+1.8)-12*(x+1.8)*(-y-0.2)+2*(x+1.8)+par_perturbed['v5'])
    L5y = (4*(-y-0.2)*((-y-0.2)**2+(x+1.8)**2)-8*(-y-0.2)+2*(3*(-y-0.2)**2-3*(x+1.8)**2)+200*(-y-0.2)**3+2*0.7*(x+1.8)+2*(-y-0.2)+par_perturbed['u5'])

    # p3 sub-landscape
    L6x = (4*(2*x-2.6)*((2*x-2.6)**2+(2*y+0.5)**2)-8*(2*x-2.6)+2*(3*(2*x-2.6)**2-3*(2*y+0.5)**2)+par_perturbed['u6'])
    L6y = (4*(2*y+0.5)*((2*x-2.6)**2+(2*y+0.5)**2)-8*(2*y+0.5)-12*(2*y+0.5)*(2*x-2.6)+par_perturbed['v6'])

    chi1 = 0.5*(np.tanh(10*((y-6)**2+(x-1.5)**2-2.5**2))+1)  # 0 in B1 p3 landscape
    chi2 = 0.5*(np.tanh(10*((y-3.4)**2+x**2-1.9**2))+1)  # 0 in B2 PreNeural landscape
    chi3 = 0.5*(np.tanh(10*((y-1.6)**2+x**2-2.4**2))+1)  # 0 in B3 circle landscape

    tau = 0.5*(np.tanh(10*(y-0.5))+1)  # 0 when y<0.2 this is for p2 landscape valid in B4
    chi4 = 0.5*(np.tanh(10*((y)**2+(x+2)**2-2**2))+1)  # 0 in B4 ball containing intermediate landscape
    chi6 = 0.5*(np.tanh(10*((y+0.5)**2+(x-1.2)**2-1.6**2))+1)  # 0 in B6 ball containing p6 landscape

    L_x = (5*(1-chi1)*chi2*chi3*L1x+5*(1-chi2)*chi3*par_perturbed['velA']*L2x+chi6*chi4*(1-chi3)*50*par_perturbed['velB']*L3x+chi6*(1-chi4)*(tau*L4x+par_perturbed['velC']*(1-tau)*L5x)+(1-chi6)*L6x)
    L_y = (5*(1-chi1)*chi2*chi3*L1y+5*(1-chi2)*chi3*par_perturbed['velA']*L2y+chi6*chi4*(1-chi3)*50*par_perturbed['velB']*L3y+chi6*(1-chi4)*(tau*L4y-par_perturbed['velC']*(1-tau)*L5y)+(1-chi6)*L6y)

    noise = ((1-chi1*chi2)*chi3*par_perturbed['noiseA']+chi4*chi6*(1-chi3)*par_perturbed['noiseB']+chi6*(1-chi4)*par_perturbed['noiseC']+(1-chi6)*par_perturbed['noiseD'])

    return L_x, L_y, noise


def update_landscape_parameters(ts, params):
    """Landscape parameters at time step `ts` for the SAGUP pulse.

    The two bump functions switch the parameters from the 0nM fit to the 500nM fit at
    step T1, then back to 0nM at step T2 (Appendix B Section B6.2).
    """
    memory1 = 0.5*(np.tanh(1*(ts-params['T1']))+1)  # memory is 1 after tstep T1=12h
    memory2 = 0.5*(np.tanh(0.1*(ts-params['T2']))+1)  # memory is 1 after tstep T2=24h
    updated_values = (1-memory1*(1-memory2))*adata0+memory1*(1-memory2)*adata500  # memory is 1 after tstep Ti
    return updated_values


def update_positions(x, y, params):
    """Euler-Maruyama integration of the SDE, re-evaluating the pulsed parameters at
    every time step."""
    for i in range(parSDE['tsteps'] - 1):

        # Update landscape parameters
        params_land = update_landscape_parameters(i, params)

        # compute vectorfield + noise
        L_x, L_y, noise = compute_landscape(x[:, i], y[:, i], params_land)

        # return position
        x[:, i + 1] = x[:, i] + parSDE['dt'] * (-L_x) + parSDE['sqrtdt'] * noise * np.random.normal(size=parSDE['ncells'])
        y[:, i + 1] = y[:, i] + parSDE['dt'] * (-L_y) + parSDE['sqrtdt'] * noise * np.random.normal(size=parSDE['ncells'])
    return x, y


def sde(params):
    """Simulate all cells from the NMP initial condition; return their positions at the
    measurement times."""
    # Generate initial conditions
    NMP = np.array([1.40765, 6.86026])  # NMP initial condition
    centers = [(NMP[0], NMP[1])]
    cluster_std = [0.05]
    X, w = make_blobs(n_samples=parSDE['ncells'], cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)

    # Initialize positions
    x, y = np.zeros((parSDE['ncells'], parSDE['tsteps'])), np.zeros((parSDE['ncells'], parSDE['tsteps']))
    x[:, 0], y[:, 0] = X[:, 0], X[:, 1]

    # Update positions
    x, y = update_positions(x, y, params)

    # Sampling at specific measurement times and return
    return np.column_stack((x[:, parSDE["measurements"]].flatten('F'), y[:, parSDE["measurements"]].flatten('F')))


def get_index_range(block_size, index):
    """Row indices of measurement block `index` in the stacked SDE output."""
    return range(block_size * index, block_size * (index + 1))


def model(pars):
    """PyABC model: simulate the pulse for `pars` = {'T1': ..., 'T2': ...} and return
    the per-timepoint cluster proportions (Appendix B Section B4.1)."""
    # Generate data generation with SDE
    S = sde(pars)

    # Precompute and use KDTree for all data
    kdtree = spatial.cKDTree(S)
    ball_index = [(kdtree.query_ball_point(cellstates[s], 0.4)) for s in states]

    proportions = []

    # Calculate proportions for each timepoint using the precomputed KDTree
    for i in range(len(parSDE["measurements"])):
        day_index = list(get_index_range(parSDE["ncells"], i))
        total = [len(np.intersect1d(ball_index[s], day_index)) for s in range(0, len(ball_index))]
        proportions.append(np.around(np.divide(total, parSDE["ncells"]), 2).tolist())

    return {"X_2": np.array(proportions).flatten()}


def distance_sim(simulation, data):
    """PyABC distance: drop the timepoints/states in `idx_remove` (D3, D7, FP) from the
    simulation, then sum the absolute differences to the data."""
    # remove the timepoints that are not compared in the simulation
    simulation_trunc = {'X_2': np.delete(simulation['X_2'], idx_remove, 0)}
    return np.absolute(np.array(simulation_trunc["X_2"])-np.array(data["X_2"])).sum()


def process_simulations(n, history, Xdata):
    """Pull n accepted simulations from a PyABC history; return them, their distances,
    and the mean proportions (%, timepoints x states)."""
    sum_stats = history.get_weighted_sum_stats_for_model(m=0, t=history.max_t)
    simulation_accepted = np.array([sum_stats[1][i]['X_2'].tolist() for i in range(n)])
    dist = np.array([distance_sim(sum_stats[1][i], Xdata) for i in range(n)])
    column_means = np.mean(simulation_accepted, axis=0)
    return simulation_accepted, dist, np.reshape(column_means*100, (len(parSDE["measurements"]), len(states)))


def simulation_todf(states, x_labels, Xsim):
    """Wrap simulated proportions into a DataFrame with '<state>-<day>' columns."""
    states_byDay = [f"{state}-{day}" for day in x_labels for state in states]
    simulation_df = pd.DataFrame(np.array(Xsim), columns=states_byDay)
    return simulation_df


def sim_predict(param, ncells):
    """Run `ncells` independent noise realisations of the pulse for a fixed (T1, T2)
    and return the simulated proportions as a DataFrame."""
    simulation_noise = []
    for _ in range(ncells):
        simul_noise = model(param)
        simulation_noise.append(simul_noise["X_2"].tolist())
    sim_noise = pd.DataFrame(np.array(simulation_noise), columns=states_byDay)
    return (sim_noise)
