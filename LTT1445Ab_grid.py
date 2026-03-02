import input_files

import numpy as np
from photochem.extensions import hotrocks
from photochem.utils import stars
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

import planets
from gridutils import make_grid

def initialize_model():

    # Make climate model
    pl = planets.LTT1445Ab
    st = planets.LTT1445A
    c = hotrocks.AdiabatClimateThermalEmission(
        Teq=pl.Teq,
        M_planet=pl.mass,
        R_planet=pl.radius,
        R_star=st.radius,
        Teff=st.Teff,
        metal=st.metal,
        logg=st.logg,
        catdir='sphinx',
        sphinx_filename='inputs/sphinx.h5'
    )
    c.verbose = False
    c.P_top = 1.0

    # Initialize PICASO
    filename_db = "picasofiles/opacities_photochem_0.1_250.0_R15000.db"
    c.initialize_picaso_from_clima(filename_db, opannection_kwargs={'wave_range': [4.0, 25.0]})

    # Make a sensible grid to save spectra at
    wavl = stars.grid_at_resolution(min_wv=np.min(c.ptherm.opa.wave), max_wv=np.max(c.ptherm.opa.wave), R=100)

    return c, wavl

def model(x):
    log10PH2O, log10PCO2, log10PO2, log10PSO2, chi, albedo, Teq = x

    # Unpack global
    c = NOMINAL_CLIMATE_MODEL
    
    # Set bolometric flux
    flux = stars.equilibrium_temperature_inverse(Teq, 0.0)
    c.rad.set_bolometric_flux(flux)
  
    # Set albedos
    c.set_custom_albedo(np.array([1.0]), np.array([albedo]))

    # Set chi
    c.chi = chi

    # Atmospheric composition
    P_i = np.ones(len(c.species_names))*1e-15
    P_i[c.species_names.index('H2O')] = 10.0**log10PH2O
    P_i[c.species_names.index('CO2')] = 10.0**log10PCO2
    P_i[c.species_names.index('O2')] = 10.0**log10PO2
    P_i[c.species_names.index('SO2')] = 10.0**log10PSO2
    P_i *= 1.0e6 # convert to dynes/cm^2

    # Compute climate
    converged = c.RCE_robust(P_i)

    # Save the P-T profile
    P = np.append(c.P_surf,c.P)
    T = np.append(c.T_surf,c.T)

    # Get emission spectra
    _, fp, _ = c.fpfs_picaso(wavl=WAVL)

    # Save
    result = {
        'converged': np.array(converged),
        'x': x.astype(np.float32),
        'P': P.astype(np.float32),
        'T': T.astype(np.float32),
        'fp': fp.astype(np.float32)
    }

    return result

def get_gridvals():
    log10PH2O = np.arange(-7.0, 2.01, 1.0)
    log10PCO2 = np.arange(-7.0, 2.01, 0.5)
    log10PO2 = np.arange(-5.0, 2.01, 1.0)
    log10PSO2 = np.arange(-7.0, 2.01, 1.0)
    chi = np.array([0.05, 0.2, 0.8]) # Factor of 4
    albedo = np.arange(0, 0.401, 0.1)
    Teq = np.array([431.0 - 23.0*2.0, 431.0, 431.0 + 23.0*2.0]) # +/- 2 sigma uncertainty
    gridvals = (log10PH2O, log10PCO2, log10PO2, log10PSO2, chi, albedo, Teq)
    gridnames = ['log10PH2O','log10PCO2','log10PO2','log10PSO2','chi','albedo','Teq']
    return gridvals, gridnames

NOMINAL_CLIMATE_MODEL, WAVL = initialize_model()

def main():

    gridvals, gridnames = get_gridvals()
    make_grid(
        model_func=model, 
        gridvals=gridvals,
        gridnames=gridnames, 
        filename='results/LTT1445Ab.h5', 
        progress_filename='results/LTT1445Ab.log',
        common={'wavl': WAVL},
        flush_every_n=100,
        batch_size=10
    )

if __name__ == "__main__":
    # mpiexec -n X python filename.py
    main()
