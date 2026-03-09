import os
import numpy as np
import h5py


KBOLTZ_CGS = 1.380649e-16  # erg/K
TAU_WAVELENGTH_NM = 9300.0  # 9.3 micron


def _default_opacity_file():
    local = os.path.join(os.path.dirname(__file__), "mie_marsdust.h5")
    if os.path.exists(local):
        return local
    try:
        from photochem_clima_data import DATA_DIR
    except Exception as exc:
        raise FileNotFoundError(
            "Could not find local mie_marsdust.h5 and failed to import "
            "`photochem_clima_data.DATA_DIR`."
        ) from exc
    return os.path.join(DATA_DIR, "aerosol_xsections", "marsdust", "mie_marsdust.h5")


def _interp_loglog_1d(x, y, xq):
    if np.any(x <= 0.0) or np.any(y <= 0.0) or xq <= 0.0:
        raise ValueError("Log-log interpolation requires positive x, y, and query value.")
    return float(np.exp(np.interp(np.log(xq), np.log(x), np.log(y))))


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
    opacity_file=None,
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
    opacity_file : str, optional
        Path to `mie_marsdust.h5`. If None, defaults to local `mie_marsdust.h5` in this
        directory, otherwise `DATA_DIR/aerosol_xsections/marsdust/mie_marsdust.h5`.

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

    if opacity_file is None:
        opacity_file = _default_opacity_file()
    if not os.path.exists(opacity_file):
        raise FileNotFoundError(f"Opacity file not found: {opacity_file}")

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


