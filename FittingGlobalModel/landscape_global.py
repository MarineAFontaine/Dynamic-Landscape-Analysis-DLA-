### landscape_global.py -- the GLOBAL landscape model (mouse / flow cytometry / scRNA-seq).
### Marine Fontaine (University of Warwick)
#
# A "model" file: just the @njit compute_landscape plus sensible defaults.
# Use it with the model-agnostic machinery in fitting_core.py:
#     from fitting_core import make_model, make_parSDE
#     from landscape_global import compute_landscape, default_parSDE, cellstates_Flow
#     model = make_model(compute_landscape, cellstates_Flow, default_parSDE,
#                        samples=[parSDE['ncells']], start_centers=[(1.40765, 6.86026)])

import numpy as np
import numba
from fitting_core import make_parSDE

default_parSDE = make_parSDE(T=4.0, dt=0.01, ncells=500, measurements=[0, 100, 200, 300, 399])

# Example cellstates (flow cytometry SAG0); other SAG/RNAseq fits override these in the notebook.
cellstates_Flow = {
    "PreNeural":   [-0.2, 4.85763],
    "p0/p1":       [-1.37061, 2.8106],
    "p2":          [-1.84471, 1.13623],
    "pMN":         [-2.56928, 0.0889825],
    "MNDiff":      [-3.34239, -0.520129],
    "DP":          [-1.17209, 0.13729],
    "EarlyVentral":[1.37061, 2.8106],
    "Earlyp3":     [1.70351, 0.457739],
    "p3":          [0.0443102, -0.34472],
    "FP":          [1.83068, -1.3856],
}


@numba.jit(nopython=True)
def compute_landscape(x, y, params, perturb_std):
    """Global landscape: six sub-landscapes blended by bump functions chi1..chi6, tau.

    `perturb_std` is the std of the per-step Gaussian perturbation applied to each
    parameter (set via parSDE['perturb_std']; 0 turns it off)."""
    par_perturbed = {key: value + np.random.normal(0, perturb_std) for key, value in params.items()}

    # NMP sub-landscape
    L1x = -2*(x-1.5)+6*(x-1.5)*(y-6)+4*(x-1.5)**3-par_perturbed['u1']
    L1y = -2*(y-6)+(3*(x-1.5)**2-3*(y-6)**2)+4*(y-6)**3-par_perturbed['v1']
    # PreNeural sub-landscape
    L2x =(-2*x+6*x*(y-4)+4*x**3-par_perturbed['u2'])
    L2y =(-2*(y-4)+(3*x**2-3*(y-4)**2)+4*(y-4)**3-par_perturbed['v2'])
    # Circular sub-landscape
    L3x=(par_perturbed['v3']/100-2*(0.4*x)+6*(0.4*x)*((0.4*x)**2+(0.4*y-0.6)**2)**2+(0.4*x)**3*(0.4*y-0.6)-(0.4*x)*(0.4*y-0.6)**3)
    L3y=(par_perturbed['u3']/100-2*(0.4*y-0.6)+6*(0.4*y-0.6)*((0.4*x)**2+(0.4*y-0.6)**2)**2+0.25*(0.4*x)**4-1.5*(0.4*x)**2*(0.4*y-0.6)**2+0.25*(0.4*y-0.6)**4)
    # p2 transition
    L4x =(-2*(2*x+3.5)+2*6*(2*x+3.5)*(2*y-1)+4*(2*x+3.5)**3-par_perturbed['u4'])
    L4y =(-2*(2*y-1)+2*(3*(2*x+3.5)**2-3*(2*y-1)**2)+4*(2*y-1)**3-par_perturbed['v4'])
    # Intermediate sub-landscape
    L5x=(4*(x+1.8)*((-y-0.2)**2+(x+1.8)**2)-8*(x+1.8)-12*(x+1.8)*(-y-0.2)+2*(x+1.8)+par_perturbed['v5'])
    L5y=(4*(-y-0.2)*((-y-0.2)**2+(x+1.8)**2)-8*(-y-0.2)+2*(3*(-y-0.2)**2-3*(x+1.8)**2)+200*(-y-0.2)**3+2*0.7*(x+1.8)+2*(-y-0.2)+par_perturbed['u5'])
    # Ventral sub-landscape
    L6x=(4*(2*x-2.6)*((2*x-2.6)**2+(2*y+0.5)**2)-8*(2*x-2.6)+2*(3*(2*x-2.6)**2-3*(2*y+0.5)**2)+par_perturbed['u6'])
    L6y=(4*(2*y+0.5)*((2*x-2.6)**2+(2*y+0.5)**2)-8*(2*y+0.5)-12*(2*y+0.5)*(2*x-2.6)+par_perturbed['v6'])

    # Bump functions
    chi1=0.5*(np.tanh(10*((y-6)**2+(x-1.5)**2-2.5**2))+1)
    chi2=0.5*(np.tanh(10*((y-3.4)**2+x**2-1.9**2))+1)
    chi3=0.5*(np.tanh(10*((y-1.6)**2+x**2-2.4**2))+1)
    tau=0.5*(np.tanh(10*(y-0.5))+1)
    chi4=0.5*(np.tanh(10*((y)**2+(x+2)**2-2**2))+1)
    chi6=0.5*(np.tanh(10*((y+0.5)**2+(x-1.2)**2-1.6**2))+1)

    # Vectorfields
    L_x=(5*(1-chi1)*chi2*chi3*L1x+5*(1-chi2)*chi3*par_perturbed['velA']*L2x+chi6*chi4*(1-chi3)*50*par_perturbed['velB']*L3x+chi6*(1-chi4)*(tau*L4x+par_perturbed['velC']*(1-tau)*L5x)+(1-chi6)*L6x)
    L_y=(5*(1-chi1)*chi2*chi3*L1y+5*(1-chi2)*chi3*par_perturbed['velA']*L2y+chi6*chi4*(1-chi3)*50*par_perturbed['velB']*L3y+chi6*(1-chi4)*(tau*L4y-par_perturbed['velC']*(1-tau)*L5y)+(1-chi6)*L6y)
    # Space-dependent diffusion coefficient
    noise=((1-chi1*chi2)*chi3*par_perturbed['noiseA']+chi4*chi6*(1-chi3)*par_perturbed['noiseB']+chi6*(1-chi4)*par_perturbed['noiseC']+(1-chi6)*par_perturbed['noiseD'])

    return L_x, L_y, noise
