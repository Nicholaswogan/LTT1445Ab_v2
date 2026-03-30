import os
import re
import numpy as np
import h5py
import pandas as pd


KBOLTZ_CGS = 1.380649e-16  # erg/K
TAU_WAVELENGTH_NM = 9300.0  # 9.3 micron
N_AVO = 6.02214076e23  # 1/mol
AREA_OF_MOLECULE = 6.0e-15  # cm^2
ATOMIC_WEIGHTS = {
    "H": 1.00794,
    "He": 4.002602,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "Na": 22.98976928,
    "Mg": 24.3050,
    "Al": 26.9815385,
    "Si": 28.0855,
    "P": 30.973761998,
    "S": 32.065,
    "Cl": 35.453,
    "K": 39.0983,
    "Ca": 40.078,
    "Ti": 47.867,
    "V": 50.9415,
    "Fe": 55.845,
}


def _default_opacity_file():
    try:
        from photochem_clima_data import DATA_DIR
    except Exception as exc:
        raise FileNotFoundError(
            "Failed to import `photochem_clima_data.DATA_DIR` to locate Mars dust opacity."
        ) from exc
    opacity_file = os.path.join(DATA_DIR, "aerosol_xsections", "marsdust", "mie_marsdust.h5")
    if not os.path.exists(opacity_file):
        raise FileNotFoundError(f"Opacity file not found: {opacity_file}")
    return opacity_file


def _interp_loglog_1d(x, y, xq):
    xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
    if np.any(x <= 0.0) or np.any(y <= 0.0) or np.any(xq_arr <= 0.0):
        raise ValueError("Log-log interpolation requires positive x, y, and query value.")
    out = np.exp(np.interp(np.log(xq_arr), np.log(x), np.log(y)))
    if np.isscalar(xq):
        return float(out[0])
    return out


def _qext_at_radius_and_wavelength(wavelengths_nm, radii_um, qext, radius_um, wavelength_nm):
    if wavelength_nm < wavelengths_nm.min() or wavelength_nm > wavelengths_nm.max():
        raise ValueError(
            f"Requested wavelength {wavelength_nm:.3f} nm is outside opacity table range "
            f"[{wavelengths_nm.min():.3f}, {wavelengths_nm.max():.3f}] nm."
        )
    if radius_um < radii_um.min() or radius_um > radii_um.max():
        raise ValueError(
            f"Requested dust radius {radius_um:.6g} um is outside opacity table range "
            f"[{radii_um.min():.6g}, {radii_um.max():.6g}] um."
        )

    # Robustly handle either (nw, nr) or (nr, nw) table orientation.
    if qext.shape == (wavelengths_nm.size, radii_um.size):
        qext_wr = qext
    elif qext.shape == (radii_um.size, wavelengths_nm.size):
        qext_wr = qext.T
    else:
        raise ValueError(
            "Unexpected qext shape. Expected either "
            f"({wavelengths_nm.size}, {radii_um.size}) or "
            f"({radii_um.size}, {wavelengths_nm.size}), got {qext.shape}."
        )

    # First interpolate in wavelength for each radius, then interpolate in radius.
    qext_vs_radius = np.empty(radii_um.size, dtype=float)
    for i in range(radii_um.size):
        qext_vs_radius[i] = _interp_loglog_1d(wavelengths_nm, qext_wr[:, i], wavelength_nm)
    return _interp_loglog_1d(radii_um, qext_vs_radius, radius_um)


def make_dust_profile(
    pressure,
    temperature,
    dz,
    tau_9_3,
    dust_radius,
):
    """Build a constant-radius dust profile and normalize it to a target tau at 9.3 micron.

    Parameters
    ----------
    pressure : array-like
        Pressure profile [dynes/cm^2].
    temperature : array-like
        Temperature profile [K].
    dz : array-like
        Layer thickness profile [cm], same length as pressure.
    tau_9_3 : float
        Target optical depth at 9.3 micron.
    dust_radius : float
        Constant particle radius [cm].

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        `(pressure_out, n_dust, r_dust)` with units:
        pressure [dynes/cm^2], n_dust [particles/cm^3], r_dust [cm].
    """
    pressure = np.asarray(pressure, dtype=float).ravel()
    temperature = np.asarray(temperature, dtype=float).ravel()
    dz = np.asarray(dz, dtype=float).ravel()

    if pressure.ndim != 1 or temperature.ndim != 1 or dz.ndim != 1:
        raise ValueError("pressure, temperature, and dz must be 1D arrays.")
    if not (pressure.size == temperature.size == dz.size):
        raise ValueError("pressure, temperature, and dz must have the same length.")
    if pressure.size == 0:
        raise ValueError("Input arrays must be non-empty.")
    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(dz)):
        raise ValueError("pressure, temperature, and dz must be finite.")
    if np.any(pressure <= 0.0):
        raise ValueError("pressure must be > 0.")
    if np.any(temperature <= 0.0):
        raise ValueError("temperature must be > 0.")
    if np.any(dz <= 0.0):
        raise ValueError("dz must be > 0.")

    if not np.isfinite(tau_9_3) or tau_9_3 < 0.0:
        raise ValueError("tau_9_3 must be finite and >= 0.")
    if not np.isfinite(dust_radius) or dust_radius <= 0.0:
        raise ValueError("dust_radius must be finite and > 0.")

    opacity_file = _default_opacity_file()

    # Gas profile and provisional dust-shape profile (n_raw ∝ n_gas).
    n_gas = pressure / (KBOLTZ_CGS * temperature)
    n_raw = n_gas.copy()
    r_dust = np.full(pressure.shape, float(dust_radius), dtype=float)

    # Load optical properties and evaluate qext at 9.3 micron and chosen radius.
    with h5py.File(opacity_file, "r") as f:
        wavelengths_nm = np.asarray(f["wavelengths"][:], dtype=float).ravel()
        radii_um = np.asarray(f["radii"][:], dtype=float).ravel()
        qext = np.asarray(f["qext"][:], dtype=float)

    if wavelengths_nm.size < 2 or radii_um.size < 2:
        raise ValueError("Opacity table must contain at least two wavelength and radius points.")
    if np.any(wavelengths_nm <= 0.0) or np.any(radii_um <= 0.0):
        raise ValueError("Opacity wavelengths and radii must be strictly positive.")
    if np.any(qext <= 0.0):
        raise ValueError("qext must be strictly positive for log-log interpolation.")

    radius_um = dust_radius * 1.0e4  # cm -> um
    qext_9p3 = _qext_at_radius_and_wavelength(
        wavelengths_nm=wavelengths_nm,
        radii_um=radii_um,
        qext=qext,
        radius_um=radius_um,
        wavelength_nm=TAU_WAVELENGTH_NM,
    )

    sigma_ext = qext_9p3 * np.pi * (dust_radius ** 2)  # cm^2 / particle
    tau_raw = float(np.sum(n_raw * sigma_ext * dz))

    if tau_9_3 == 0.0:
        n_dust = np.zeros_like(n_raw)
    else:
        if tau_raw <= 0.0:
            raise ValueError(
                "Computed raw tau at 9.3 micron is <= 0 while tau_9_3 > 0. "
                "Check pressure/temperature/dz units and dust_radius."
            )
        n_dust = n_raw * (tau_9_3 / tau_raw)

    return pressure.copy(), n_dust, r_dust


def _species_mass_from_formula(species_name):
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", species_name)
    if not tokens:
        raise ValueError(f"Could not parse molecular formula for species `{species_name}`.")
    consumed = "".join(atom + count for atom, count in tokens)
    if consumed != species_name:
        raise ValueError(f"Unsupported species formula `{species_name}`.")

    mass = 0.0
    for atom, count_str in tokens:
        if atom not in ATOMIC_WEIGHTS:
            raise ValueError(f"Unsupported atom `{atom}` in species `{species_name}`.")
        count = int(count_str) if count_str else 1
        mass += ATOMIC_WEIGHTS[atom] * count
    return mass


def _species_masses_from_names(species_names):
    return np.array([_species_mass_from_formula(name) for name in species_names], dtype=float)


def _dynamic_viscosity_air(T):
    T = np.asarray(T, dtype=float)
    if np.any(T <= 0.0):
        raise ValueError("Temperature must be > 0 for dynamic viscosity.")
    eta0 = 1.716e-5
    T0 = 273.15
    S = 111.0
    unit_conversion = 10.0
    return unit_conversion * eta0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)


def _slip_correction_factor(particle_radius, number_density):
    particle_radius = np.asarray(particle_radius, dtype=float)
    number_density = np.asarray(number_density, dtype=float)
    if np.any(particle_radius <= 0.0) or np.any(number_density <= 0.0):
        raise ValueError("particle_radius and number_density must be > 0 for slip correction.")
    lambda_mfp = 1.0 / (number_density * AREA_OF_MOLECULE)
    return 1.0 + (lambda_mfp / particle_radius) * (
        1.257 + 0.4 * np.exp((-1.1 * particle_radius) / lambda_mfp)
    )


def _fall_velocity(grav, particle_radius, particle_density, air_density, viscosity):
    return (2.0 / 9.0) * grav * particle_radius ** 2 * (particle_density - air_density) / viscosity


def _gas_number_density(pressure, temperature):
    return pressure / (KBOLTZ_CGS * temperature)


def _mean_molecular_weight(mix, species_masses):
    mix = np.asarray(mix, dtype=float)
    species_masses = np.asarray(species_masses, dtype=float).ravel()
    if mix.ndim != 2:
        raise ValueError("mix must be a 2D array with shape (nlevel, nspecies).")
    if mix.shape[1] != species_masses.size:
        raise ValueError("mix and species_masses have incompatible shapes.")
    return np.sum(mix * species_masses[None, :], axis=1)


def _gas_mass_density(number_density, mubar):
    return number_density * (mubar / N_AVO)


def _dust_particle_mass(dust_radius, dust_density):
    return (4.0 / 3.0) * np.pi * dust_radius ** 3 * dust_density


def _tau_9p3_from_profile(n_dust, dz, dust_radius):
    opacity_file = _default_opacity_file()
    with h5py.File(opacity_file, "r") as f:
        wavelengths_nm = np.asarray(f["wavelengths"][:], dtype=float).ravel()
        radii_um = np.asarray(f["radii"][:], dtype=float).ravel()
        qext = np.asarray(f["qext"][:], dtype=float)
    radius_um = dust_radius * 1.0e4
    qext_9p3 = _qext_at_radius_and_wavelength(
        wavelengths_nm=wavelengths_nm,
        radii_um=radii_um,
        qext=qext,
        radius_um=radius_um,
        wavelength_nm=TAU_WAVELENGTH_NM,
    )
    sigma_ext = qext_9p3 * np.pi * dust_radius ** 2
    return float(np.sum(n_dust * sigma_ext * dz))


def compute_lofted_dust_diagnostics(
    pressure,
    temperature,
    dz,
    mix,
    species_masses,
    n_dust,
    dust_radius,
    dust_density,
):
    pressure = np.asarray(pressure, dtype=float).ravel()
    temperature = np.asarray(temperature, dtype=float).ravel()
    dz = np.asarray(dz, dtype=float).ravel()
    n_dust = np.asarray(n_dust, dtype=float).ravel()
    mix = np.asarray(mix, dtype=float)
    species_masses = np.asarray(species_masses, dtype=float).ravel()

    n_gas = _gas_number_density(pressure, temperature)
    mubar = _mean_molecular_weight(mix, species_masses)
    rho_gas = _gas_mass_density(n_gas, mubar)
    particle_mass = _dust_particle_mass(dust_radius, dust_density)
    rho_dust = n_dust * particle_mass

    gas_column_mass = float(np.sum(rho_gas * dz))
    dust_column_mass = float(np.sum(rho_dust * dz))
    epsilon_col = dust_column_mass / gas_column_mass if gas_column_mass > 0.0 else np.nan
    q_number = np.divide(n_dust, n_gas, out=np.zeros_like(n_dust), where=n_gas > 0.0)
    q_mass = np.divide(rho_dust, rho_gas, out=np.zeros_like(rho_dust), where=rho_gas > 0.0)

    return {
        "n_gas": n_gas,
        "mubar": mubar,
        "rho_gas": rho_gas,
        "rho_dust": rho_dust,
        "q_number": q_number,
        "q_mass": q_mass,
        "dust_column_mass": dust_column_mass,
        "gas_column_mass": gas_column_mass,
        "epsilon_col": epsilon_col,
        "tau_9_3": _tau_9p3_from_profile(n_dust, dz, dust_radius),
    }


def solve_lofted_dust_profile(
    pressure,
    temperature,
    dz,
    mix,
    species_masses,
    Kzz,
    dust_radius,
    dust_density,
    epsilon_col,
    grav,
):
    """Solve a simple steady lofted-dust profile on a fixed climate column.

    The profile shape is set by balancing eddy mixing against sedimentation for
    a fixed particle size. The resulting shape is normalized to the target
    column dust-to-gas mass ratio `epsilon_col`.

    Parameters
    ----------
    pressure : array-like
        Pressure profile [dynes/cm^2].
    temperature : array-like
        Temperature profile [K].
    dz : array-like
        Layer thickness profile [cm].
    mix : array-like
        Gas mixing-ratio profile with shape `(nlevel, nspecies)` [unitless].
    species_masses : array-like
        Molecular masses corresponding to `mix` columns [g/mol].
    Kzz : array-like
        Eddy diffusion coefficient profile [cm^2/s].
    dust_radius : float
        Dust particle radius [cm].
    dust_density : float
        Material density of dust grains [g/cm^3].
    epsilon_col : float
        Target column dust-to-gas mass ratio [unitless].
    grav : float or array-like
        Gravitational acceleration [cm/s^2]. May be a scalar or a profile with
        the same length as `pressure`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, dict]
        `(pressure_out, n_dust, r_dust, diagnostics)` where:
        `pressure_out` is pressure [dynes/cm^2],
        `n_dust` is dust particle density [particles/cm^3],
        `r_dust` is dust radius [cm],
        and `diagnostics` contains derived profiles and column-integrated
        quantities such as `tau_9_3`, `epsilon_col`, `q_number`, `q_mass`,
        and `v_set`.
    """
    pressure = np.asarray(pressure, dtype=float).ravel()
    temperature = np.asarray(temperature, dtype=float).ravel()
    dz = np.asarray(dz, dtype=float).ravel()
    mix = np.asarray(mix, dtype=float)
    species_masses = np.asarray(species_masses, dtype=float).ravel()
    Kzz = np.asarray(Kzz, dtype=float).ravel()

    nlevel = pressure.size
    if nlevel == 0:
        raise ValueError("pressure must be non-empty.")
    if not (temperature.size == nlevel == dz.size == Kzz.size):
        raise ValueError("pressure, temperature, dz, and Kzz must have the same length.")
    if mix.shape != (nlevel, species_masses.size):
        raise ValueError("mix must have shape (nlevel, nspecies).")
    if np.any(pressure <= 0.0) or np.any(temperature <= 0.0) or np.any(dz <= 0.0) or np.any(Kzz <= 0.0):
        raise ValueError("pressure, temperature, dz, and Kzz must be > 0.")
    if not np.isfinite(dust_radius) or dust_radius <= 0.0:
        raise ValueError("dust_radius must be finite and > 0.")
    if not np.isfinite(dust_density) or dust_density <= 0.0:
        raise ValueError("dust_density must be finite and > 0.")
    if not np.isfinite(epsilon_col) or epsilon_col < 0.0:
        raise ValueError("epsilon_col must be finite and >= 0.")
    grav = np.asarray(grav, dtype=float).ravel()
    if grav.size == 1:
        grav = np.full(nlevel, grav[0], dtype=float)
    if grav.size != nlevel:
        raise ValueError("grav must be a scalar or have the same length as pressure.")
    if np.any(~np.isfinite(grav)) or np.any(grav <= 0.0):
        raise ValueError("grav must be finite and > 0.")

    n_gas = _gas_number_density(pressure, temperature)
    mubar = _mean_molecular_weight(mix, species_masses)
    rho_gas = _gas_mass_density(n_gas, mubar)
    viscosity = _dynamic_viscosity_air(temperature)
    v_set = _fall_velocity(grav, dust_radius, dust_density, rho_gas, viscosity)
    v_set *= _slip_correction_factor(dust_radius, n_gas)
    v_set = np.clip(v_set, 0.0, np.inf)

    # Shape-only solve: fix arbitrary surface mixing ratio and march upward.
    q_shape = np.zeros(nlevel, dtype=float)
    q_shape[0] = 1.0
    for i in range(nlevel - 1):
        dqdz = -(v_set[i] * q_shape[i]) / Kzz[i]
        q_next = q_shape[i] + dqdz * dz[i]
        if q_next <= 0.0 or not np.isfinite(q_next):
            q_shape[i + 1 :] = 0.0
            break
        q_shape[i + 1] = q_next

    n_dust_shape = q_shape * n_gas
    diagnostics_shape = compute_lofted_dust_diagnostics(
        pressure=pressure,
        temperature=temperature,
        dz=dz,
        mix=mix,
        species_masses=species_masses,
        n_dust=n_dust_shape,
        dust_radius=dust_radius,
        dust_density=dust_density,
    )

    if epsilon_col == 0.0:
        n_dust = np.zeros_like(n_dust_shape)
        scale = 0.0
    else:
        epsilon_shape = diagnostics_shape["epsilon_col"]
        if not np.isfinite(epsilon_shape) or epsilon_shape <= 0.0:
            raise ValueError("Unnormalized dust shape has non-positive epsilon_col; cannot normalize.")
        scale = epsilon_col / epsilon_shape
        n_dust = n_dust_shape * scale

    r_dust = np.full(nlevel, dust_radius, dtype=float)
    diagnostics = compute_lofted_dust_diagnostics(
        pressure=pressure,
        temperature=temperature,
        dz=dz,
        mix=mix,
        species_masses=species_masses,
        n_dust=n_dust,
        dust_radius=dust_radius,
        dust_density=dust_density,
    )
    diagnostics["v_set"] = v_set
    diagnostics["Kzz"] = Kzz.copy()
    diagnostics["shape_scale"] = scale
    diagnostics["n_dust_shape"] = n_dust_shape

    return pressure.copy(), n_dust, r_dust, diagnostics


def solve_lofted_dust_profile_from_climate(
    c,
    Kzz,
    dust_radius,
    dust_density,
    epsilon_col,
    species_masses=None,
):
    """Extract a climate column and solve the lofted dust profile.

    Parameters
    ----------
    c : object
        Climate object with `P_surf`, `P`, `dz`, `T_surf`, `T`, `f_i_surf`,
        `f_i`, `species_names`, and `gravity` attributes.
    Kzz : array-like
        Eddy diffusion coefficient profile [cm^2/s]. May have length `nlevel`
        or `nlevel-1`; if `nlevel-1`, the first value is prepended for the
        surface level.
    dust_radius : float
        Dust particle radius [cm].
    dust_density : float
        Material density of dust grains [g/cm^3].
    epsilon_col : float
        Target column dust-to-gas mass ratio [unitless].
    species_masses : array-like, optional
        Molecular masses for `c.species_names` [g/mol]. If omitted, they are
        inferred from the species names using a simple chemical-formula parser.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, dict]
        `(pressure_out, n_dust, r_dust, diagnostics)` on the extracted climate
        level grid, with pressure in [dynes/cm^2], dust number density in
        [particles/cm^3], and dust radius in [cm].
    """
    if not hasattr(c, "P_surf") or not hasattr(c, "P") or not hasattr(c, "dz"):
        raise ValueError("Climate object must provide `P_surf`, `P`, and `dz`.")
    if not hasattr(c, "T_surf") or not hasattr(c, "T"):
        raise ValueError("Climate object must provide `T_surf` and `T`.")
    if not hasattr(c, "f_i_surf") or not hasattr(c, "f_i"):
        raise ValueError("Climate object must provide `f_i_surf` and `f_i`.")
    if not hasattr(c, "species_names"):
        raise ValueError("Climate object must provide `species_names`.")
    if not hasattr(c, "gravity"):
        raise ValueError("Climate object must provide `gravity`.")

    pressure = np.append(float(c.P_surf), np.asarray(c.P, dtype=float).ravel())
    temperature = np.append(float(c.T_surf), np.asarray(c.T, dtype=float).ravel())
    dz_internal = np.asarray(c.dz, dtype=float).ravel()
    dz = np.append(dz_internal[0], dz_internal)
    mix = np.vstack((np.asarray(c.f_i_surf, dtype=float), np.asarray(c.f_i, dtype=float)))

    Kzz = np.asarray(Kzz, dtype=float).ravel()
    if Kzz.size == pressure.size - 1:
        Kzz = np.append(Kzz[0], Kzz)
    if Kzz.size != pressure.size:
        raise ValueError("Kzz must have length nlevel or nlevel-1 relative to the extracted climate column.")

    if species_masses is None:
        species_masses = _species_masses_from_names(list(c.species_names))

    return solve_lofted_dust_profile(
        pressure=pressure,
        temperature=temperature,
        dz=dz,
        mix=mix,
        species_masses=species_masses,
        Kzz=Kzz,
        dust_radius=dust_radius,
        dust_density=dust_density,
        epsilon_col=epsilon_col,
        grav=np.append(float(c.gravity_surf), np.asarray(c.gravity, dtype=float).ravel()),
    )


def _load_marsdust_optics():
    opacity_file = _default_opacity_file()
    with h5py.File(opacity_file, "r") as f:
        wavelengths_nm = np.asarray(f["wavelengths"][:], dtype=float).ravel()
        radii_um = np.asarray(f["radii"][:], dtype=float).ravel()
        qext = np.asarray(f["qext"][:], dtype=float)
        w0 = np.asarray(f["w0"][:], dtype=float)
        g0 = np.asarray(f["g0"][:], dtype=float)

    if wavelengths_nm.size < 2 or radii_um.size < 2:
        raise ValueError("Opacity table must contain at least two wavelength and radius points.")
    if np.any(wavelengths_nm <= 0.0) or np.any(radii_um <= 0.0):
        raise ValueError("Opacity wavelengths and radii must be strictly positive.")
    if np.any(qext <= 0.0):
        raise ValueError("qext must be strictly positive for log-log interpolation.")

    expected_wr = (wavelengths_nm.size, radii_um.size)
    expected_rw = (radii_um.size, wavelengths_nm.size)

    def _to_wr(arr, name):
        if arr.shape == expected_wr:
            return arr
        if arr.shape == expected_rw:
            return arr.T
        raise ValueError(
            f"Unexpected {name} shape {arr.shape}. Expected {expected_wr} or {expected_rw}."
        )

    qext = _to_wr(qext, "qext")
    w0 = _to_wr(w0, "w0")
    g0 = _to_wr(g0, "g0")

    return wavelengths_nm, radii_um, qext, w0, g0


def _interp_linear_over_logx(x, y, xq):
    if np.any(x <= 0.0) or np.any(xq <= 0.0):
        raise ValueError("Interpolation in log-x requires positive x and query values.")
    return np.interp(np.log(xq), np.log(x), y)


def _interp_optics_at_radius_and_wavelengths(
    wavelengths_nm,
    radii_um,
    qext_wr,
    w0_wr,
    g0_wr,
    radius_um,
    wavelength_nm,
):
    if np.any(wavelength_nm < wavelengths_nm.min()) or np.any(wavelength_nm > wavelengths_nm.max()):
        raise ValueError(
            f"Requested wavelength range [{wavelength_nm.min():.3f}, {wavelength_nm.max():.3f}] nm "
            f"is outside opacity table range [{wavelengths_nm.min():.3f}, {wavelengths_nm.max():.3f}] nm."
        )
    if radius_um < radii_um.min() or radius_um > radii_um.max():
        raise ValueError(
            f"Requested dust radius {radius_um:.6g} um is outside opacity table range "
            f"[{radii_um.min():.6g}, {radii_um.max():.6g}] um."
        )

    nr = radii_um.size
    nw = wavelength_nm.size

    # Interpolate in wavelength at every radius.
    qext_r_w = np.empty((nr, nw), dtype=float)
    w0_r_w = np.empty((nr, nw), dtype=float)
    g0_r_w = np.empty((nr, nw), dtype=float)
    for i in range(nr):
        qext_r_w[i, :] = _interp_loglog_1d(wavelengths_nm, qext_wr[:, i], wavelength_nm)
        w0_r_w[i, :] = _interp_linear_over_logx(wavelengths_nm, w0_wr[:, i], wavelength_nm)
        g0_r_w[i, :] = _interp_linear_over_logx(wavelengths_nm, g0_wr[:, i], wavelength_nm)

    # Interpolate in radius at every wavelength.
    qext_out = np.empty(nw, dtype=float)
    w0_out = np.empty(nw, dtype=float)
    g0_out = np.empty(nw, dtype=float)
    lr = np.log(radii_um)
    lrq = np.log(radius_um)
    for k in range(nw):
        qext_out[k] = _interp_loglog_1d(radii_um, qext_r_w[:, k], radius_um)
        w0_out[k] = np.interp(lrq, lr, w0_r_w[:, k])
        g0_out[k] = np.interp(lrq, lr, g0_r_w[:, k])

    w0_out = np.clip(w0_out, 0.0, 1.0)
    g0_out = np.clip(g0_out, 0.0, 1.0)
    return qext_out, w0_out, g0_out


def build_picaso_cloud_df(
    pressure_level_bar,
    dz_layer_cm,
    n_dust_layer,
    r_dust_layer_cm,
    wno_cm1,
):
    """Build PICASO-compatible cloud dataframe from a dust profile.

    Parameters
    ----------
    pressure_level_bar : array-like
        Layer pressures in bars (length nlayer).
    dz_layer_cm : array-like
        Layer thicknesses in cm (length nlayer).
    n_dust_layer : array-like
        Dust particle number density in particles/cm^3 (length nlayer).
    r_dust_layer_cm : array-like
        Dust particle radius in cm (length nlayer).
    wno_cm1 : array-like
        PICASO wavenumber grid in cm^-1 (length nwave).

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns required by `case.clouds(df=...)`:
        `pressure`, `wavenumber`, `opd`, `w0`, `g0`.
    """
    pressure_level_bar = np.asarray(pressure_level_bar, dtype=float).ravel()
    dz_layer_cm = np.asarray(dz_layer_cm, dtype=float).ravel()
    n_dust_layer = np.asarray(n_dust_layer, dtype=float).ravel()
    r_dust_layer_cm = np.asarray(r_dust_layer_cm, dtype=float).ravel()
    wno_cm1 = np.asarray(wno_cm1, dtype=float).ravel()

    nlayer = pressure_level_bar.size
    if nlayer == 0:
        raise ValueError("Layer arrays must be non-empty.")
    if not (dz_layer_cm.size == nlayer == n_dust_layer.size == r_dust_layer_cm.size):
        raise ValueError("pressure_level_bar, dz_layer_cm, n_dust_layer, and r_dust_layer_cm must match in length.")
    if wno_cm1.size == 0:
        raise ValueError("wno_cm1 must be non-empty.")
    if np.any(~np.isfinite(pressure_level_bar)) or np.any(~np.isfinite(dz_layer_cm)) or np.any(~np.isfinite(n_dust_layer)) or np.any(~np.isfinite(r_dust_layer_cm)) or np.any(~np.isfinite(wno_cm1)):
        raise ValueError("All inputs must be finite.")
    if np.any(pressure_level_bar <= 0.0):
        raise ValueError("pressure_level_bar must be > 0.")
    if np.any(dz_layer_cm <= 0.0):
        raise ValueError("dz_layer_cm must be > 0.")
    if np.any(n_dust_layer < 0.0):
        raise ValueError("n_dust_layer must be >= 0.")
    if np.any(r_dust_layer_cm <= 0.0):
        raise ValueError("r_dust_layer_cm must be > 0.")
    if np.any(wno_cm1 <= 0.0):
        raise ValueError("wno_cm1 must be > 0.")

    wavelengths_nm, radii_um, qext_wr, w0_wr, g0_wr = _load_marsdust_optics()
    wavelength_query_nm = 1.0e7 / wno_cm1
    radius_layer_um = r_dust_layer_cm * 1.0e4

    if np.any(wavelength_query_nm < wavelengths_nm.min()) or np.any(wavelength_query_nm > wavelengths_nm.max()):
        raise ValueError(
            f"Requested wavelength range [{wavelength_query_nm.min():.3f}, {wavelength_query_nm.max():.3f}] nm "
            f"is outside opacity table range [{wavelengths_nm.min():.3f}, {wavelengths_nm.max():.3f}] nm."
        )
    if np.any(radius_layer_um < radii_um.min()) or np.any(radius_layer_um > radii_um.max()):
        raise ValueError(
            f"Requested dust radius range [{radius_layer_um.min():.6g}, {radius_layer_um.max():.6g}] um "
            f"is outside opacity table range [{radii_um.min():.6g}, {radii_um.max():.6g}] um."
        )

    nwave = wno_cm1.size
    nr = radii_um.size

    # First interpolate each table radius onto the PICASO wavelength grid.
    qext_r_w = np.empty((nr, nwave), dtype=float)
    w0_r_w = np.empty((nr, nwave), dtype=float)
    g0_r_w = np.empty((nr, nwave), dtype=float)
    for i in range(nr):
        qext_r_w[i, :] = _interp_loglog_1d(wavelengths_nm, qext_wr[:, i], wavelength_query_nm)
        w0_r_w[i, :] = _interp_linear_over_logx(wavelengths_nm, w0_wr[:, i], wavelength_query_nm)
        g0_r_w[i, :] = _interp_linear_over_logx(wavelengths_nm, g0_wr[:, i], wavelength_query_nm)

    # Then interpolate in radius for all layers in bulk.
    lr = np.log(radii_um)
    lrq = np.log(radius_layer_um)
    idx_hi = np.searchsorted(lr, lrq, side="right")
    idx_hi = np.clip(idx_hi, 1, nr - 1)
    idx_lo = idx_hi - 1
    frac = (lrq - lr[idx_lo]) / (lr[idx_hi] - lr[idx_lo])
    frac2d = frac[:, None]

    qext_lo = qext_r_w[idx_lo, :]
    qext_hi = qext_r_w[idx_hi, :]
    qext = np.exp((1.0 - frac2d) * np.log(qext_lo) + frac2d * np.log(qext_hi))

    w0_lo = w0_r_w[idx_lo, :]
    w0_hi = w0_r_w[idx_hi, :]
    w0 = (1.0 - frac2d) * w0_lo + frac2d * w0_hi

    g0_lo = g0_r_w[idx_lo, :]
    g0_hi = g0_r_w[idx_hi, :]
    g0 = (1.0 - frac2d) * g0_lo + frac2d * g0_hi

    w0 = np.clip(w0, 0.0, 1.0)
    g0 = np.clip(g0, 0.0, 1.0)

    sigma_ext = qext * (np.pi * (r_dust_layer_cm[:, None] ** 2))
    opd = sigma_ext * (n_dust_layer[:, None] * dz_layer_cm[:, None])

    pressure_col = np.repeat(pressure_level_bar, nwave)
    wno_col = np.tile(wno_cm1, nlayer)
    df = pd.DataFrame(
        {
            "pressure": pressure_col,
            "wavenumber": wno_col,
            "opd": opd.reshape(-1),
            "w0": w0.reshape(-1),
            "g0": g0.reshape(-1),
        }
    ).sort_values(["pressure", "wavenumber"]).reset_index(drop=True)

    expected_rows = nlayer * nwave
    if df.shape[0] != expected_rows:
        raise ValueError(f"Cloud dataframe row count {df.shape[0]} does not match expected {expected_rows}.")

    return df


def apply_picaso_dust_clouds(
    c,
    particle_index=0,
    particle_name=None,
):
    """Apply dust clouds to an initialized hotrocks PICASO case.

    Parameters
    ----------
    c : object
        Hotrocks climate object with initialized `ptherm` (`initialize_picaso_from_clima` called).
    particle_index : int, optional
        Which particle to use from `c.pdensities/c.pradii` if multiple are present.
    particle_name : str, optional
        Alternative to `particle_index`. If given, selects that particle by name.

    Returns
    -------
    pandas.DataFrame
        PICASO cloud dataframe that was applied to `c.ptherm.case`.
    """
    if not hasattr(c, "ptherm") or c.ptherm is None:
        raise ValueError("PICASO is not initialized. Call `c.initialize_picaso_from_clima(...)` first.")
    if not hasattr(c, "make_picaso_atm"):
        raise ValueError("Input object must provide `make_picaso_atm()`.")
    if not hasattr(c, "pdensities") or not hasattr(c, "pradii"):
        raise ValueError("Input object must provide `pdensities` and `pradii`.")
    if not hasattr(c, "P") or not hasattr(c, "P_surf") or not hasattr(c, "dz"):
        raise ValueError("Input object must provide `P`, `P_surf`, and `dz`.")

    pdens = np.asarray(c.pdensities, dtype=float)
    pradii = np.asarray(c.pradii, dtype=float)
    if pdens.ndim != 2 or pradii.ndim != 2:
        raise ValueError("`c.pdensities` and `c.pradii` must be 2D arrays (nz,np).")
    if pdens.shape != pradii.shape:
        raise ValueError("`c.pdensities` and `c.pradii` must have identical shapes.")
    if pdens.shape[1] < 1:
        raise ValueError("No particles found in `c.pdensities`.")

    if particle_name is not None:
        if not hasattr(c, "particle_names"):
            raise ValueError("`particle_name` was provided, but object has no `particle_names`.")
        names = list(c.particle_names)
        if particle_name not in names:
            raise ValueError(f"particle_name `{particle_name}` not found. Available: {names}")
        ip = names.index(particle_name)
    else:
        ip = int(particle_index)
    if ip < 0 or ip >= pdens.shape[1]:
        raise ValueError(f"particle_index {ip} is out of bounds for np={pdens.shape[1]}.")

    P_internal = np.asarray(c.P, dtype=float).ravel()
    dz_layer_cm = np.asarray(c.dz, dtype=float).ravel()
    if P_internal.size < 1:
        raise ValueError("`c.P` must be non-empty.")
    if dz_layer_cm.size != P_internal.size:
        raise ValueError("Expected `len(c.dz) == len(c.P)`.")

    # Build level arrays matching the PICASO atmosphere levels used by make_picaso_atm:
    # [surface, internal_levels...]. Dust at the surface is set equal to first internal level.
    pressure_level_dyn = np.append(float(c.P_surf), P_internal)
    n_dust_level = np.append(pdens[0, ip], pdens[:, ip])
    r_dust_level_cm = np.append(pradii[0, ip], pradii[:, ip])

    nlevel = pressure_level_dyn.size
    if nlevel < 2:
        raise ValueError("pressure_level_dyn must have at least two levels.")
    if not (n_dust_level.size == nlevel and r_dust_level_cm.size == nlevel):
        raise ValueError("n_dust_level and r_dust_level_cm must match pressure_level_dyn length.")
    if dz_layer_cm.size != (nlevel - 1):
        raise ValueError("dz_layer_cm must have length nlevel-1.")
    if np.any(pressure_level_dyn <= 0.0):
        raise ValueError("pressure_level_dyn must be > 0.")
    if np.any(r_dust_level_cm <= 0.0):
        raise ValueError("r_dust_level_cm must be > 0.")
    if np.any(n_dust_level < 0.0):
        raise ValueError("n_dust_level must be >= 0.")
    if np.any(dz_layer_cm <= 0.0):
        raise ValueError("dz_layer_cm must be > 0.")

    # Ensure PICASO has atmosphere loaded so `nlevel` is known before clouds(df=...).
    atm = c.make_picaso_atm()
    c.ptherm.case.atmosphere(df=atm, verbose=False)

    pressure_layer_bar = np.sqrt(pressure_level_dyn[:-1] * pressure_level_dyn[1:]) / 1.0e6
    n_dust_layer = 0.5 * (n_dust_level[:-1] + n_dust_level[1:])
    r_dust_layer_cm = np.sqrt(r_dust_level_cm[:-1] * r_dust_level_cm[1:])

    wno_cm1 = np.asarray(c.ptherm.opa.wno, dtype=float)
    cloud_df = build_picaso_cloud_df(
        pressure_level_bar=pressure_layer_bar,
        dz_layer_cm=dz_layer_cm,
        n_dust_layer=n_dust_layer,
        r_dust_layer_cm=r_dust_layer_cm,
        wno_cm1=wno_cm1,
    )
    c.ptherm.case.clouds(df=cloud_df)
    return cloud_df


def fpfs_picaso_with_dust(
    c,
    particle_index=0,
    particle_name=None,
    R=100,
    wavl=None,
    atmosphere_kwargs=None,
    **kwargs,
):
    """Apply dust clouds and compute thermal emission with PICASO."""
    if atmosphere_kwargs is None:
        atmosphere_kwargs = {}
    apply_picaso_dust_clouds(
        c=c,
        particle_index=particle_index,
        particle_name=particle_name,
    )
    return c.fpfs_picaso(R=R, wavl=wavl, atmosphere_kwargs=atmosphere_kwargs, **kwargs)

import LTT1445Ab_grid
import planets
from fixedpoint import RobustFixedPointSolver
from photochem.extensions import hotrocks

class DustSolver():

    def __init__(self):

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
            sphinx_filename='inputs/sphinx.h5',
            species_file='inputs/species_dust.yaml',
            opacities_file='inputs/opacities_dust.yaml'
        )
        c.verbose = False
        c.P_top = 1.0

        # Initialize PICASO
        filename_db = "picasofiles/opacities_photochem_0.1_250.0_R15000.db"
        c.initialize_picaso_from_clima(filename_db, opannection_kwargs={'wave_range': [4.0, 25.0]})

        self.c = c
        self.species_masses = _species_masses_from_names(list(c.species_names))

    def compute_P_grid(self, P_surf):
        """Construct the pressure grid used for equilibrium calculations.

        Parameters
        ----------
        P_surf : float
            Surface pressure in dyn/cm^2.

        Returns
        -------
        ndarray
            1D pressure grid in dyn/cm^2, including the surface level and
            extending to ``self.P_top``.
        """
        c = self.c
        P_top = c.P_top
        nz = len(c.T)
        P_grid = np.logspace(np.log10(P_surf), np.log10(P_top), 2*nz+1)
        P_grid = np.append(P_grid[0], P_grid[1::2])
        return P_grid

    def _coerce_Kzz_profile(self, Kzz, nlevel):
        Kzz = np.asarray(Kzz, dtype=float).ravel()
        if Kzz.size == 1:
            Kzz = np.full(nlevel, Kzz[0], dtype=float)
        elif Kzz.size == nlevel - 1:
            Kzz = np.append(Kzz[0], Kzz)
        elif Kzz.size != nlevel:
            raise ValueError("Kzz must be a scalar or have length nlevel or nlevel-1.")
        if np.any(~np.isfinite(Kzz)) or np.any(Kzz <= 0.0):
            raise ValueError("Kzz must be finite and > 0.")
        return Kzz

    def _build_legacy_dust_profile(self, tau_9_3, dust_radius):
        c = self.c
        pressure = np.append(c.P_surf, c.P)
        temperature = np.append(c.T_surf, c.T)
        dz = np.append(c.dz[0], c.dz)
        _, n_dust, r_dust = make_dust_profile(
            pressure=pressure,
            temperature=temperature,
            dz=dz,
            tau_9_3=tau_9_3,
            dust_radius=dust_radius,
        )
        diagnostics = {
            "tau_9_3": tau_9_3,
            "epsilon_col": np.nan,
        }
        return n_dust, r_dust, diagnostics

    def _build_lofted_dust_profile(self, epsilon_col, Kzz, dust_radius, dust_density):
        c = self.c
        nlevel = len(c.P) + 1
        Kzz_profile = self._coerce_Kzz_profile(Kzz, nlevel)
        _, n_dust, r_dust, diagnostics = solve_lofted_dust_profile_from_climate(
            c,
            Kzz=Kzz_profile,
            dust_radius=dust_radius,
            dust_density=dust_density,
            epsilon_col=epsilon_col,
            species_masses=self.species_masses,
        )
        return n_dust, r_dust, diagnostics

    def g_eval(self, x, P_i, dust_model, **dust_kwargs):

        c = self.c

        # State vector is the full temperature profile on the pressure grid.
        T = np.asarray(x, dtype=float).ravel()

        # Compute P and dz
        P = self.compute_P_grid(np.sum(P_i))
        if T.size != len(P):
            raise ValueError("Temperature state vector is incompatible with the current pressure grid.")
        f_i = np.empty((len(P),len(c.species_names)))
        f_i_ = P_i/np.sum(P_i)
        for i,sp in enumerate(c.species_names):
            f_i[:,i] = f_i_[i]
        c.make_profile_dry(P, T, f_i)

        # Build dust profile using the selected model.
        if dust_model == "legacy_tau":
            n_dust, r_dust, dust_diag = self._build_legacy_dust_profile(
                tau_9_3=dust_kwargs["tau_9_3"],
                dust_radius=dust_kwargs["dust_radius"],
            )
        elif dust_model == "lofted_epsilon":
            n_dust, r_dust, dust_diag = self._build_lofted_dust_profile(
                epsilon_col=dust_kwargs["epsilon_col"],
                Kzz=dust_kwargs["Kzz"],
                dust_radius=dust_kwargs["dust_radius"],
                dust_density=dust_kwargs["dust_density"],
            )
        else:
            raise ValueError(f"Unsupported dust_model `{dust_model}`.")
        self.last_dust_diagnostics = dust_diag

        n_dust = n_dust.reshape(len(n_dust), len(c.particle_names))
        r_dust = r_dust.reshape(len(r_dust), len(c.particle_names))

        # set dust profile
        c.set_particle_density_and_radii(np.append(c.P_surf, c.P), n_dust, r_dust)

        # Run climate
        converged = c.RCE_robust(P_i)
        assert converged

        # results
        result = np.append(c.T_surf, c.T)
        return result
    
    def solve(self, P_i, dust_model="legacy_tau", dust_kwargs=None, tol=1, max_tol=1, **solver_kwargs): 

        if dust_kwargs is None:
            dust_kwargs = {}

        def g(x):
            return self.g_eval(x, P_i, dust_model, **dust_kwargs)

        self.c.set_particle_density_and_radii(np.array([1.0]), np.array([[0.0]]), np.array([[1.0e-4]]))
        converged = self.c.RCE_robust(P_i)
        assert converged

        guess = np.append(self.c.T_surf, self.c.T)

        solver = RobustFixedPointSolver(
            g=g,
            x0=guess,
            tol=tol,
            max_tol=max_tol,
            **solver_kwargs
        )
        result = solver.solve()
        return result
    
