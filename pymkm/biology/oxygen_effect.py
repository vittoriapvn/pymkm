import numpy as np
from typing import Optional, Tuple

def compute_relative_radioresistance(K: float, pO2: float, K_mult: np.ndarray) -> np.ndarray:
    """
    Compute the relative radioresistance R using a general formulation,
    valid for both OSMK 2021 and 2023 versions.

    Parameters
    ----------
    K : float
        Oxygen tension [mmHg] at which R = (1 + Rmax)/2.
    pO2 : float
        Partial pressure of oxygen [mmHg] at which to evaluate R.
    K_mult : np.ndarray
        Multiplicative factor depending on OSMK version and microdosimetric quantity.

    Returns
    -------
    R : np.ndarray
        Relative radioresistance values.
    """
    denominator = pO2 + K * K_mult
    return (pO2 + K) / denominator


def _compute_scaling_factor_component(R: np.ndarray, f_max: float, Rmax: float) -> np.ndarray:
    """
    Compute scaling factor component for OSMK 2023, used for either rd or z0.

    Parameters
    ----------
    R : np.ndarray
        Relative radioresistance.
    f_max : float
        Maximum scaling factor at pO2 = 0.
    Rmax : float
        Maximum radioresistance at pO2 = 0.

    Returns
    -------
    scaling_factor : np.ndarray
        Scaling factors for the microdosimetric quantity.
    """
    return 1 + (R - 1) * (f_max - 1) / (Rmax - 1)


def compute_scaling_factors(
    R: np.ndarray, f_rd_max: float, f_z0_max: float, Rmax: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute both rd and z0 scaling factors for OSMK 2023.

    Parameters
    ----------
    R : np.ndarray
        Relative radioresistance.
    f_rd_max : float
        Max scaling factor for domain radius.
    f_z0_max : float
        Max scaling factor for z0.
    Rmax : float
        Maximum radioresistance at pO2 = 0.

    Returns
    -------
    f_rd : np.ndarray
        Scaling factor for domain radius.
    f_z0 : np.ndarray
        Scaling factor for z0 (already squared).
    """
    f_rd = _compute_scaling_factor_component(R, f_rd_max, Rmax)
    f_z0 = _compute_scaling_factor_component(R, f_z0_max, Rmax) ** 2
    return f_rd, f_z0


def compute_osmk_radioresistance(
    version: str, z_bar_domain: np.ndarray, params
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute radioresistance R and optionally the OSMK 2023 scaling factors.

    Parameters
    ----------
    version : str
        Either "2021" or "2023".
    z_bar_domain : np.ndarray
        Dose-mean specific energy of the domain.
    params : object
        Parameter container (must have attributes depending on OSMK version).

    Returns
    -------
    R : np.ndarray
        Relative radioresistance.
    f_rd : np.ndarray or None
        Scaling factor for domain radius (only for 2023).
    f_z0 : np.ndarray or None
        Scaling factor for z0 (only for 2023).
    """
    if version == "2021":
        zR = params.zR
        gamma = params.gamma
        Rm = params.Rm
        z_ratio = (z_bar_domain / zR) ** gamma
        K_mult = (z_ratio + 1) / (z_ratio + Rm)
        R = compute_relative_radioresistance(params.K, params.pO2, K_mult)
        return R, None, None

    elif version == "2023":
        Rmax = params.Rmax
        K_mult = 1 / Rmax
        R = compute_relative_radioresistance(params.K, params.pO2, K_mult)
        f_rd, f_z0 = compute_scaling_factors(R, params.f_rd_max, params.f_z0_max, Rmax)
        return R, f_rd, f_z0

    else:
        raise ValueError(f"Unsupported OSMK version: {version}")

def apply_oxygen_correction_alpha(z_bar_star_domain: np.ndarray, R: np.ndarray, params) -> np.ndarray:
    """
    Compute oxygen-corrected alpha for OSMK.
    """
    alphaL = params.alphaL
    alphaS = params.alphaS
    beta0 = params.beta0
    
    return alphaL + (alphaS / R) + (beta0 * z_bar_star_domain / (R**2))

def apply_oxygen_correction_beta(z_bar_star_domain: np.ndarray, z_bar_domain: np.ndarray, R: np.ndarray, params) -> np.ndarray:
    """
    Compute oxygen-corrected beta for OSMK.
    """
    beta0 = params.beta0
    
    return (z_bar_star_domain / z_bar_domain) * beta0 / (R**2)
