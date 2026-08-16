### landscape_first_decisions.py -- the FIRST-DECISIONS landscape.
### Marine Fontaine (University of Warwick)
#
# Use with fitting_core.py. Cells start from a
# Meso/NMP/PreNeural mixture.

import numpy as np
import numba
from fitting_core import make_parSDE

default_parSDE = make_parSDE(T=1.0, dt=0.01, ncells=500, measurements=[0, 20, 40, 60, 80, 99])

cellstates = {
    "Meso":        [3, 4.8],
    "NMP":         [1.40765, 6.86026],
    "PreNeural":   [-0.0497558, 4.85763],
    "p0/p1":       [-1.37061, 2.8106],
    "EarlyVentral":[1.37061, 2.8106],
}


@numba.jit(nopython=True)
def compute_landscape(x, y, params, perturb_std):
    """First-decisions landscape: only NMP (L1) and PreNeural (L2), blended by chi1/chi2.

    `perturb_std` is the std of the per-step Gaussian perturbation applied to each
    parameter (set via parSDE['perturb_std']; 0 turns it off)."""
    par_perturbed = {key: value + np.random.normal(0, perturb_std) for key, value in params.items()}

    # NMP sublandscape
    L1x = -2*(x-1.5)+6*(x-1.5)*(y-6)+4*(x-1.5)**3-par_perturbed['u1']
    L1y = -2*(y-6)+(3*(x-1.5)**2-3*(y-6)**2)+4*(y-6)**3-par_perturbed['v1']
    # PreNeural sublandscape
    L2x =(-2*x+6*x*(y-4)+4*x**3-par_perturbed['u2'])
    L2y =(-2*(y-4)+(3*x**2-3*(y-4)**2)+4*(y-4)**3-par_perturbed['v2'])

    # Bump functions
    chi1=0.5*(np.tanh(10*((y-6)**2+(x-1.5)**2-2.5**2))+1)
    chi2=0.5*(np.tanh(10*((y-3.4)**2+x**2-1.9**2))+1)

    # First Decisions Landscape (FDL)
    L_x=(5*(1-chi1)*chi2*L1x+5*(1-chi2)*par_perturbed['velA']*L2x)
    L_y=(5*(1-chi1)*chi2*L1y+5*(1-chi2)*par_perturbed['velA']*L2y)
    # Space-dependent diffusion coefficient
    noise=((1-chi1*chi2)*par_perturbed['noiseA'])

    return L_x, L_y, noise
