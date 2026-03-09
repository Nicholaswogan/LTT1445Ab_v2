import numpy as np

import LTT1445Ab_grid
import dust
from gridutils import make_grid

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

def initialize():
    surface_albedo = 0.2

    m = dust.DustSolver()
    c_nom, _ = LTT1445Ab_grid.initialize_model()

    m.c.set_custom_albedo(np.array([1.0]), np.array([surface_albedo]))
    c_nom.set_custom_albedo(np.array([1.0]), np.array([surface_albedo]))

    return m, c_nom

DUST_MODEL, NOMINAL_MODEL = initialize()

def model(x):

    tau_9_3, dust_radius = x

    m = DUST_MODEL
    c_dust = m.c
    c_nom = NOMINAL_MODEL

    P_i = np.ones(len(c_dust.species_names))*1e-15
    P_i[c_dust.species_names.index('O2')] = 1.0
    P_i[c_dust.species_names.index('CO2')] = 1e-2
    P_i *= 1.0e6 # convert to dynes/cm^2

    assert c_dust.species_names == c_nom.species_names

    result = m.solve(P_i, tau_9_3=tau_9_3, dust_radius=dust_radius, verbose=True)
    assert result.converged

    converged = c_nom.RCE_robust(P_i)
    assert converged
    
    result = {}

    _c = c_dust
    P = np.append(_c.P_surf, _c.P)
    T = np.append(_c.T_surf, _c.T)
    f_i = np.concatenate((np.array([_c.f_i[0,:]]),_c.f_i),axis=0)
    mix = {}
    for i,sp in enumerate(_c.species_names):
        mix[sp] = f_i[:,i]
    _, _, fpfs = dust.fpfs_picaso_with_dust(_c, wavl=LTT1445Ab_grid.WAVL)
    result['P_d'] = P
    result['T_d'] = T
    result['fpfs_d'] = fpfs
    for key in mix:
        result[key+'_d'] = mix[key]

    _c = c_nom
    P = np.append(_c.P_surf, _c.P)
    T = np.append(_c.T_surf, _c.T)
    f_i = np.concatenate((np.array([_c.f_i[0,:]]),_c.f_i),axis=0)
    mix = {}
    for i,sp in enumerate(_c.species_names):
        mix[sp] = f_i[:,i]
    _, _, fpfs = _c.fpfs_picaso(wavl=LTT1445Ab_grid.WAVL)
    result['P_n'] = P
    result['T_n'] = T
    result['fpfs_n'] = fpfs
    for key in mix:
        result[key+'_n'] = mix[key]

    return result

def get_gridvals():
    tau_9_3 = np.array([0.1, 1.0, 10.0])
    dust_radius = np.array([0.5e-4, 1.0e-4, 5.0e-4])
    gridvals = (tau_9_3, dust_radius)
    gridnames = ['tau_9_3', 'dust_radius']
    return gridvals, gridnames

def main():

    gridvals, gridnames = get_gridvals()

    make_grid(
        model_func=model, 
        gridvals=gridvals,
        gridnames=gridnames, 
        filename='results/LTT1445Ab_dust.h5', 
        progress_filename='results/LTT1445Ab_dust.log',
        common={'wavl': LTT1445Ab_grid.WAVL},
        compression='lzf',
        shuffle=True,
    )

if __name__ == "__main__":
    # mpiexec -n X python filename.py
    main()

