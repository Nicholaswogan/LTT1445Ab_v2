import numpy as np

import LTT1445Ab_grid
import models
from gridutils import make_grid

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

def initialize():
    surface_albedo = 0.2
    m = models.LTTCoupledModel('inputs/gj176_scaled_to_ltt1445ab.txt', surface_albedo)
    c, _ = LTT1445Ab_grid.initialize_model()
    c.set_custom_albedo(np.array([1.0]), np.array([surface_albedo]))
    return m, c

COUPLED_MODEL, NOMINAL_MODEL = initialize()

def model(x):

    PCO2 = 10.0**x[0]

    m = COUPLED_MODEL
    c = NOMINAL_MODEL

    Pi = {
        'O2': 1*1e6,
        'CO2': PCO2*1e6,
    }
    Kzz = 1.0e6

    m.initial_guess(Pi, Kzz)
    result = m.solve(Pi, Kzz, tol=1e-1, max_tol=1e-1, verbose=True, max_iter=20)
    assert result.converged

    P_i = np.ones(len(c.species_names))*1e-15
    for sp in Pi:
        ind = c.species_names.index(sp)
        P_i[ind] = Pi[sp]
    converged = c.RCE_robust(P_i)
    assert converged
    
    P = np.append(c.P_surf, c.P)
    T = np.append(c.T_surf, c.T)
    f_i = np.concatenate((np.array([c.f_i[0,:]]),c.f_i),axis=0)
    mix = {}
    for i,sp in enumerate(c.species_names):
        mix[sp] = f_i[:,i]
    _, _, fpfs = c.fpfs_picaso(wavl=LTT1445Ab_grid.WAVL)

    result = {
        'P_n': P,
        'T_n': T, 
        'fpfs_n': fpfs
    }
    for key in mix:
        result[key+'_n'] = mix[key]

    P = m.P_c.copy()
    T = m.T_c.copy()
    mix = m.mix_c.copy()
    _, _, fpfs = m.c.fpfs_picaso(wavl=LTT1445Ab_grid.WAVL)
    result['P_c'] = P
    result['T_c'] = T
    result['fpfs_c'] = fpfs
    for key in mix:
        result[key+'_c'] = mix[key]

    return result

def get_gridvals():
    log10PCO2 = np.array([-6.0, -5.0, -4.0, -3.0, -2.0, -1.0])
    gridvals = (log10PCO2,)
    gridnames = ['log10PCO2']
    return gridvals, gridnames

def main():

    gridvals, gridnames = get_gridvals()

    make_grid(
        model_func=model, 
        gridvals=gridvals,
        gridnames=gridnames, 
        filename='results/LTT1445Ab_coupled.h5', 
        progress_filename='results/LTT1445Ab_coupled.log',
        common={'wavl': LTT1445Ab_grid.WAVL},
        compression='lzf',
        shuffle=True,
    )

if __name__ == "__main__":
    # mpiexec -n X python filename.py
    main()

