import input_files

import numpy as np
from astropy import constants
from tempfile import NamedTemporaryFile
from copy import deepcopy
from scipy import constants as const
from scipy import interpolate
from scipy import integrate
import numba as nb
from numba import types

from photochem.extensions import hotrocks
from photochem.utils import zahnle_rx_and_thermo_files
from photochem.utils._format import yaml, FormatSettings_main, MyDumper
from photochem import EvoAtmosphere, PhotoException

from fixedpoint import RobustFixedPointSolver
import planets

class LTTCoupledModel():

    def __init__(self, stellar_uv_file, surface_albedo, atoms=['O','C']):

        pl = planets.LTT1445Ab
        st = planets.LTT1445A

        # Photochemistry init
        settings_file = {
            'atmosphere-grid': {
                'bottom': 0.0, 
                'top': 'atmospherefile', 
                'number-of-layers': 100
            },
            'planet': {
                'planet-mass': float(pl.mass*constants.M_earth.value),
                'planet-radius': float(pl.radius*constants.R_earth.value),
                'surface-albedo': float(surface_albedo),
                'solar-zenith-angle': 60.0,
                'hydrogen-escape': {'type': 'none'},
                'water': {'fix-water-in-troposphere': False, 'gas-rainout': False, 'water-condensation': False}
            },
            'boundary-conditions': [{
                'name': 'O2',
                'lower-boundary': {'type': 'vdep', 'vdep': 0.0},
                'upper-boundary': {'type': 'veff', 'veff': 0.0}
            }]
        }
        settings_file = FormatSettings_main(settings_file)
        
        with NamedTemporaryFile('w',suffix='.yaml') as ff:
            # Get mechanism
            zahnle_rx_and_thermo_files(
                atoms_names=atoms,
                rxns_filename=ff.name,
                thermo_filename=None,
                remove_reaction_particles=True
            )
            with NamedTemporaryFile('w',suffix='.yaml') as f:
                yaml.dump(settings_file, f, Dumper=MyDumper)

                pc = EvoAtmosphereRobust(
                    ff.name,
                    f.name,
                    stellar_uv_file
                )
        pc.var.verbose = 1
        pc.rdat.verbose = False
        pc.rdat.max_dT_tol = 1.0
        pc.var.diurnal_fac = 1 
        self.pc = pc

        # Climate model
        c = hotrocks.AdiabatClimateThermalEmission(
            Teq=pl.Teq,
            M_planet=pl.mass,
            R_planet=pl.radius,
            R_star=st.radius,
            Teff=st.Teff,
            metal=st.metal,
            logg=st.logg,
            catdir='sphinx',
            sphinx_filename='inputs/sphinx.h5',
            species=pc.dat.species_names[pc.dat.np:-2],
            condensates=[]
        )
        c.verbose = True
        c.P_top = 1.0

        # Initialize PICASO
        filename_db = "picasofiles/opacities_photochem_0.1_250.0_R15000.db"
        c.initialize_picaso_from_clima(filename_db, opannection_kwargs={'wave_range': [4.0, 25.0]})

        c.set_custom_albedo(np.array([1.0]), np.array([surface_albedo]))
        self.c = c
        

    def set_PTmix_from_climate(self):
        c = self.c
        self.P_c = np.append(c.P_surf, c.P)
        self.T_c = np.append(c.T_surf, c.T)
        f_i = np.concatenate((np.array([self.c.f_i[0,:]]),self.c.f_i),axis=0)
        self.mix_c = {}
        for i,sp in enumerate(c.species_names):
            self.mix_c[sp] = f_i[:,i]

    def set_PTmix_from_photochem(self):
        pc = self.pc
        sol = pc.mole_fraction_dict()
        self.P = sol['pressure']
        self.T = sol['temp']
        self.mix = {}
        species_names = pc.dat.species_names[:-2-pc.dat.nsl]
        for i,sp in enumerate(species_names):
            self.mix[sp] = np.maximum(sol[sp], 1.0e-200)

    def initial_guess(self, Pi, Kzz):

        # Climate
        P_i = np.ones(len(self.c.species_names))*1e-15
        for sp in Pi:
            ind = self.c.species_names.index(sp)
            P_i[ind] = Pi[sp]
        converged = self.c.RCE_robust(P_i)
        assert converged
        self.set_PTmix_from_climate()

        # Photochemistry
        Kzz_ = np.ones_like(self.P_c)*Kzz
        self.pc.initialize_to_PT_bcs(self.P_c, self.T_c, Kzz_, self.mix_c, Pi)
        converged = self.pc.find_steady_state_robust()
        assert converged
        self.set_PTmix_from_photochem()

    def g_eval(self, x, Pi, Kzz):

        c = self.c
        pc = self.pc

        # Unpack
        log10P, T = x[:len(self.P_c)], x[len(self.P_c):]
        P = 10.0**log10P

        # Get an intial mixing ratio guess, and run photochemistry
        mix = {}
        for sp in self.mix:
            mix[sp] = 10.0**np.interp(log10P[::-1].copy(), np.log10(self.P[::-1]).copy(), np.log10(self.mix[sp])[::-1].copy())[::-1].copy()
        Kzz_ = np.ones_like(P)*Kzz
        pc.initialize_to_PT_bcs(P, T, Kzz_, mix, Pi)
        converged = pc.find_steady_state_robust()
        assert converged
        self.set_PTmix_from_photochem()

        # Unpack photochemistry
        sol = pc.mole_fraction_dict()
        custom_dry_mix = {'pressure': sol['pressure']}
        P_i = np.ones(len(c.species_names))*1e-15
        for i,sp in enumerate(c.species_names):
            P_i[c.species_names.index(sp)] = np.maximum(sol[sp][0],1e-30)*pc.var.surface_pressure*1e6
            custom_dry_mix[sp] = np.maximum(sol[sp],1e-200)
        
        # Run climate
        converged = c.RCE(P_i, c.T_surf, c.T, c.convecting_with_below, custom_dry_mix)
        assert converged
        self.set_PTmix_from_climate()

        # results
        result = np.append(np.log10(self.P_c), self.T_c)
        return result
    
    def solve(self, Pi, Kzz, tol=1, max_tol=2, **kwargs): 

        def g(PT):
            return self.g_eval(PT, Pi, Kzz)

        guess = np.append(np.log10(self.P_c), self.T_c)

        solver = RobustFixedPointSolver(
            g=g,
            x0=guess,
            tol=tol,
            max_tol=max_tol,
            **kwargs
        )
        result = solver.solve()
        return result
    
    
class RobustData():
    
    def __init__(self):

        # Parameters for determining steady state
        self.atols = [1e-23, 1e-22, 1e-20, 1e-18]
        self.min_mix_reset = -1e-13
        self.TOA_pressure_avg = 1.0e-7*1e6 # mean TOA pressure (dynes/cm^2)
        self.max_dT_tol = 3 # The permitted difference between T in photochem and desired T
        self.max_dlog10edd_tol = 0.2 # The permitted difference between Kzz in photochem and desired Kzz
        self.freq_update_PTKzz = 1000 # step frequency to update PTKzz profile.
        self.freq_update_atol = 10_000
        self.max_total_step = 100_000 # Maximum total allowed steps before giving up
        self.min_step_conv = 300 # Min internal steps considered before convergence is allowed
        self.verbose = True # print information or not?
        self.freq_print = 100 # Frequency in which to print

        # Below for interpolation
        self.log10P_interp = None
        self.T_interp = None
        self.log10edd_interp = None
        self.P_desired = None
        self.T_desired = None
        self.Kzz_desired = None
        # information needed during robust stepping
        self.total_step_counter = None
        self.nerrors = None
        self.max_time = None
        self.robust_stepper_initialized = None
        # Surface pressures
        self.Pi = None

class EvoAtmosphereRobust(EvoAtmosphere):

    def __init__(self, mechanism_file, settings_file, flux_file, data_dir=None):

        with NamedTemporaryFile('w',suffix='.txt') as f:
            f.write(ATMOSPHERE_INIT)
            f.flush()
            super().__init__(
                mechanism_file, 
                settings_file, 
                flux_file,
                f.name,
                data_dir
            )

        self.rdat = RobustData()

        # Values in photochem to adjust
        self.var.verbose = 0
        self.var.upwind_molec_diff = True
        self.var.autodiff = True
        self.var.atol = 1.0e-23
        self.var.equilibrium_time = 1e15

        # Model state
        self.max_time_state = None

        for i in range(len(self.var.cond_params)):
            self.var.cond_params[i].smooth_factor = 1
            self.var.cond_params[i].k_evap = 0

    def set_surface_pressures(self, Pi):
        
        for sp in Pi:
            self.set_lower_bc(sp, bc_type='press', press=Pi[sp])

    def initialize_to_PT(self, P, T, Kzz, mix):

        P, T, mix = deepcopy(P), deepcopy(T), deepcopy(mix)

        rdat = self.rdat

        # Ensure X sums to 1
        ftot = np.zeros(P.shape[0])
        for key in mix:
            ftot += mix[key]
        for key in mix:
            mix[key] = mix[key]/ftot

        # Compute mubar at all heights
        mu = {}
        for i,sp in enumerate(self.dat.species_names[:-2]):
            mu[sp] = self.dat.species_mass[i]
        mubar = np.zeros(P.shape[0])
        for key in mix:
            mubar += mix[key]*mu[key]

        # Altitude of P-T grid
        P1, T1, mubar1, z1 = compute_altitude_of_PT(P, T, mubar, self.dat.planet_radius, self.dat.planet_mass, rdat.TOA_pressure_avg)
        # If needed, extrapolate Kzz and mixing ratios
        if P1.shape[0] != Kzz.shape[0]:
            Kzz1 = np.append(Kzz,Kzz[-1])
            mix1 = {}
            for sp in mix:
                mix1[sp] = np.append(mix[sp],mix[sp][-1])
        else:
            Kzz1 = Kzz.copy()
            mix1 = mix

        rdat.log10P_interp = np.log10(P1.copy()[::-1])
        rdat.T_interp = T1.copy()[::-1]
        rdat.log10edd_interp = np.log10(Kzz1.copy()[::-1])
        
        # extrapolate to 1e6 bars
        T_tmp = interpolate.interp1d(rdat.log10P_interp, rdat.T_interp, bounds_error=False, fill_value='extrapolate')(12)
        edd_tmp = interpolate.interp1d(rdat.log10P_interp, rdat.log10edd_interp, bounds_error=False, fill_value='extrapolate')(12)
        rdat.log10P_interp = np.append(rdat.log10P_interp, 12)
        rdat.T_interp = np.append(rdat.T_interp, T_tmp)
        rdat.log10edd_interp = np.append(rdat.log10edd_interp, edd_tmp)

        rdat.P_desired = P1.copy()
        rdat.T_desired = T1.copy()
        rdat.Kzz_desired = Kzz1.copy()

        # Calculate the photochemical grid
        ind_t = np.argmin(np.abs(P1 - rdat.TOA_pressure_avg))
        z_top = z1[ind_t]
        z_bottom = 0.0
        dz = (z_top - z_bottom)/self.var.nz
        z_p = np.empty(self.var.nz)
        z_p[0] = dz/2.0
        for i in range(1,self.var.nz):
            z_p[i] = z_p[i-1] + dz

        # Now, we interpolate all values to the photochemical grid
        P_p = 10.0**np.interp(z_p, z1, np.log10(P1))
        T_p = np.interp(z_p, z1, T1)
        Kzz_p = 10.0**np.interp(z_p, z1, np.log10(Kzz1))
        mix_p = {}
        for sp in mix1:
            mix_p[sp] = 10.0**np.interp(z_p, z1, np.log10(mix1[sp]))
        k_boltz = const.k*1e7
        den_p = P_p/(k_boltz*T_p)

        # Update photochemical model grid
        self.update_vertical_grid(TOA_alt=z_top) # this will update gravity for new planet radius
        self.set_temperature(T_p)
        self.var.edd = Kzz_p
        usol = np.ones(self.wrk.usol.shape)*1e-40
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        for sp in mix_p:
            if sp in species_names:
                ind = species_names.index(sp)
                usol[ind,:] = mix_p[sp]*den_p
        self.wrk.usol = usol

        self.prep_atmosphere(self.wrk.usol)

    def initialize_to_PT_bcs(self, P, T, Kzz, mix, Pi):
        self.rdat.Pi = Pi
        self.set_surface_pressures(Pi)
        self.initialize_to_PT(P, T, Kzz, mix)

    def set_particle_radii(self, radii):
        particle_radius = self.var.particle_radius
        for key in radii:
            ind = self.dat.species_names.index(key)
            particle_radius[ind,:] = radii[key]
        self.var.particle_radius = particle_radius
        self.update_vertical_grid(TOA_alt=self.var.top_atmos)

    def initialize_robust_stepper(self, usol):
        """Initialized a robust integrator.

        Parameters
        ----------
        usol : ndarray[double,dim=2]
            Input number densities
        """
        rdat = self.rdat  
        rdat.total_step_counter = 0
        rdat.nerrors = 0
        rdat.max_time = 0
        self.max_time_state = None
        self.initialize_stepper(usol)
        rdat.robust_stepper_initialized = True

    def robust_step(self):
        """Takes a single robust integrator step

        Returns
        -------
        tuple
            The tuple contains two bools `give_up, reached_steady_state`. If give_up is True
            then the algorithm things it is time to give up on reaching a steady state. If
            reached_steady_state then the algorithm has reached a steady state within
            tolerance.
        """

        rdat = self.rdat

        if not rdat.robust_stepper_initialized:
            raise Exception('This routine can only be called after `initialize_robust_stepper`')

        give_up = False
        reached_steady_state = False

        for i in range(1):
            try:
                self.step()
                rdat.total_step_counter += 1
            except PhotoException as e:
                # If there is an error, lets reinitialize, but get rid of any
                # negative numbers
                usol = np.clip(self.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf)
                self.initialize_stepper(usol)
                rdat.nerrors += 1

                if rdat.nerrors > 15:
                    give_up = True
                    break

            # Reset integrator if we get large magnitude negative numbers
            if not self.healthy_atmosphere():
                usol = np.clip(self.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf)
                self.initialize_stepper(usol)
                rdat.nerrors += 1

                if rdat.nerrors > 15:
                    give_up = True
                    break

            # Update the max time achieved
            if self.wrk.tn > rdat.max_time:
                rdat.max_time = self.wrk.tn
                self.max_time_state = self.model_state_to_dict() # save the model state

            # convergence checking
            converged = self.check_for_convergence()

            # Compute the max difference between the P-T profile in photochemical model
            # and the desired P-T profile
            T_p = np.interp(np.log10(self.wrk.pressure_hydro.copy()[::-1]), rdat.log10P_interp, rdat.T_interp)
            T_p = T_p.copy()[::-1]
            max_dT = np.max(np.abs(T_p - self.var.temperature))

            # Compute the max difference between the P-edd profile in photochemical model
            # and the desired P-edd profile
            log10edd_p = np.interp(np.log10(self.wrk.pressure_hydro.copy()[::-1]), rdat.log10P_interp, rdat.log10edd_interp)
            log10edd_p = log10edd_p.copy()[::-1]
            max_dlog10edd = np.max(np.abs(log10edd_p - np.log10(self.var.edd)))

            # TOA pressure
            TOA_pressure = self.wrk.pressure_hydro[-1]

            condition1 = converged and self.wrk.nsteps > rdat.min_step_conv or self.wrk.tn > self.var.equilibrium_time
            condition2 = max_dT < rdat.max_dT_tol and max_dlog10edd < rdat.max_dlog10edd_tol and rdat.TOA_pressure_avg/3 < TOA_pressure < rdat.TOA_pressure_avg*3

            if condition1 and condition2:
                if rdat.verbose:
                    print('nsteps = %i  longdy = %.1e  max_dT = %.1e  max_dlog10edd = %.1e  TOA_pressure = %.1e'% \
                        (rdat.total_step_counter, self.wrk.longdy, max_dT, max_dlog10edd, TOA_pressure/1e6))
                # success!
                reached_steady_state = True
                break

            if not (rdat.total_step_counter % rdat.freq_update_atol):
                ind = int(rdat.total_step_counter/rdat.freq_update_atol)
                ind1 = ind - len(rdat.atols)*int(ind/len(rdat.atols))
                self.var.atol = rdat.atols[ind1]
                if rdat.verbose:
                    print('new atol = %.1e'%(self.var.atol))
                self.initialize_stepper(self.wrk.usol)
                break

            if not (self.wrk.nsteps % rdat.freq_update_PTKzz) or (condition1 and not condition2):
                # After ~1000 steps, lets update P,T, edd and vertical grid, if possible.
                try:
                    self.set_press_temp_edd(rdat.P_desired,rdat.T_desired,rdat.Kzz_desired,hydro_pressure=True)
                except PhotoException:
                    pass
                try:
                    self.update_vertical_grid(TOA_pressure=rdat.TOA_pressure_avg)
                except PhotoException:
                    pass
                self.initialize_stepper(self.wrk.usol)

            if rdat.total_step_counter > rdat.max_total_step:
                give_up = True
                break

            if not (self.wrk.nsteps % rdat.freq_print) and rdat.verbose:
                print('nsteps = %i  longdy = %.1e  max_dT = %.1e  max_dlog10edd = %.1e  TOA_pressure = %.1e'% \
                    (rdat.total_step_counter, self.wrk.longdy, max_dT, max_dlog10edd, TOA_pressure/1e6))
                
        return give_up, reached_steady_state
    
    def find_steady_state(self):
        """Attempts to find a photochemical steady state.

        Returns
        -------
        bool
            If True, then the routine was successful.
        """    

        self.initialize_robust_stepper(self.wrk.usol)
        success = True
        while True:
            give_up, reached_steady_state = self.robust_step()
            if reached_steady_state:
                break
            if give_up:
                success = False
                break
        return success
    
    def healthy_atmosphere(self):
        return np.min(self.wrk.mix_history[:,:,0]) > self.rdat.min_mix_reset
    
    def find_steady_state_robust(self):

        # Change some rdat settings
        self.rdat.freq_update_atol = 100_000
        self.rdat.max_total_step = 10_000

        # First just try to get to steady-state with standard atol
        self.var.atol = 1.0e-23
        converged = self.find_steady_state()
        if converged:
            return converged

        # Convergence did not happen. Save the max time state.
        max_time = self.rdat.max_time
        max_time_state = deepcopy(self.max_time_state)

        # Lets try a couple different atols.
        for atol in [1.0e-18, 1.0e-15]:
            # Lets initialize to max time state
            self.initialize_from_dict(max_time_state)
            # Do some smaller number of steps
            self.rdat.max_total_step = 5_000
            self.var.atol = atol # set the atol
            converged = self.find_steady_state() # Integrate
            if converged:
                # If converged then lets return
                return converged

            # No convergence. We re-save max time state
            if self.rdat.max_time > max_time:
                max_time = self.rdat.max_time
                max_time_state = deepcopy(self.max_time_state)

        # No convergence, we reinitialize to max time state and return
        self.initialize_from_dict(max_time_state)

        return converged
        
    def model_state_to_dict(self):
        """Returns a dictionary containing all information needed to reinitialize the atmospheric
        state. This dictionary can be used as an input to "initialize_from_dict".
        """

        if self.rdat.log10P_interp is None:
            raise Exception('This routine can only be called after `initialize_to_PT_bcs`')

        out = {}
        out['rdat'] = deepcopy(self.rdat.__dict__)
        out['top_atmos'] = self.var.top_atmos
        out['temperature'] = self.var.temperature
        out['edd'] = self.var.edd
        out['usol'] = self.wrk.usol
        out['particle_radius'] = self.var.particle_radius

        # Other settings
        out['equilibrium_time'] = self.var.equilibrium_time
        out['verbose'] = self.var.verbose
        out['atol'] = self.var.atol
        out['autodiff'] = self.var.autodiff

        return out

    def initialize_from_dict(self, out):
        """Initializes the model from a dictionary created by the "model_state_to_dict" routine.
        """

        for key, value in out['rdat'].items():
            setattr(self.rdat, key, value)

        self.update_vertical_grid(TOA_alt=out['top_atmos'])
        self.set_temperature(out['temperature'])
        self.var.edd = out['edd']
        self.wrk.usol = out['usol']
        self.var.particle_radius = out['particle_radius']
        self.update_vertical_grid(TOA_alt=out['top_atmos'])

        # Other settings
        self.var.equilibrium_time = out['equilibrium_time']
        self.var.verbose = out['verbose']
        self.var.atol = out['atol']
        self.var.autodiff = out['autodiff']
        
        # Now set boundary conditions
        Pi = self.rdat.Pi
        for sp in Pi:
            self.set_lower_bc(sp, bc_type='press', press=Pi[sp])

        self.prep_atmosphere(self.wrk.usol)

@nb.experimental.jitclass()
class TempPressMubar:

    log10P : types.double[:] # type: ignore
    T : types.double[:] # type: ignore
    mubar : types.double[:] # type: ignore

    def __init__(self, P, T, mubar):
        self.log10P = np.log10(P)[::-1].copy()
        self.T = T[::-1].copy()
        self.mubar = mubar[::-1].copy()

    def temperature_mubar(self, P):
        T = np.interp(np.log10(P), self.log10P, self.T)
        mubar = np.interp(np.log10(P), self.log10P, self.mubar)
        return T, mubar

@nb.njit()
def gravity(radius, mass, z):
    G_grav = const.G
    grav = G_grav * (mass/1.0e3) / ((radius + z)/1.0e2)**2.0
    grav = grav*1.0e2 # convert to cgs
    return grav

@nb.njit()
def hydrostatic_equation(P, u, planet_radius, planet_mass, ptm):
    z = u[0]
    grav = gravity(planet_radius, planet_mass, z)
    T, mubar = ptm.temperature_mubar(P)
    k_boltz = const.Boltzmann*1e7
    dz_dP = -(k_boltz*T*const.Avogadro)/(mubar*grav*P)
    return np.array([dz_dP])

def compute_altitude_of_PT(P, T, mubar, planet_radius, planet_mass, P_top):
    ptm = TempPressMubar(P, T, mubar)
    args = (planet_radius, planet_mass, ptm)

    if P_top < P[-1]:
        # If P_top is lower P than P grid, then we extend it
        P_top_ = P_top
        P_ = np.append(P,P_top_)
        T_ = np.append(T,T[-1])
        mubar_ = np.append(mubar,mubar[-1])
    else:
        P_top_ = P[-1]
        P_ = P.copy()
        T_ = T.copy()
        mubar_ = mubar.copy()

    # Integrate to TOA
    out = integrate.solve_ivp(hydrostatic_equation, [P_[0], P_[-1]], np.array([0.0]), t_eval=P_, args=args, rtol=1e-6)
    assert out.success

    # Stitch together
    z_ = out.y[0]

    return P_, T_, mubar_, z_

ATMOSPHERE_INIT = \
"""alt      den        temp       eddy                       
0.0      1          1000       1e6              
1.0e3    1          1000       1e6         
"""

