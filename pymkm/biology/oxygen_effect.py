"""
Oxygen effect modeling (OSMK 2021 and 2023) for microdosimetry.

This module implements functions to compute:

- Relative radioresistance :func:`compute_relative_radioresistance`
- OSMK 2023 scaling factors :func:`compute_scaling_factors`
- Oxygen-corrected α and β LQ parameters for hypoxia

It supports both OSMK 2021 (γ, zR, Rm formulation) and OSMK 2023 (scaling model)
based on microdosimetric specific energies and a parameter container.

Intended for use during SMK post-processing when `apply_oxygen_effect=True`.
"""

import numpy as np
from typing import Optional, Tuple

def compute_relative_radioresistance(K: float, pO2: float, K_mult: np.ndarray) -> np.ndarray:
    """
    Compute the relative radioresistance R based on oxygen tension and model-dependent scaling.

    This function is used in both OSMK 2021 and OSMK 2023 models.
    
    :param K: Oxygen pressure [mmHg] at which R = (1 + Rmax)/2.
    :param pO2: Partial pressure of oxygen [mmHg].
    :param K_mult: Model-specific multiplicative term for the denominator.
    :returns: Relative radioresistance values (R).
    :rtype: np.ndarray
    """
    denominator = pO2 + K * K_mult
    return (pO2 + K) / denominator


def _compute_scaling_factor_component(R: np.ndarray, f_max: float, Rmax: float) -> np.ndarray:
    """
    Internal helper to compute a scaling factor (e.g., for rd or z0) based on R.

    :param R: Relative radioresistance.
    :param f_max: Maximum scaling factor at full hypoxia (pO2 = 0).
    :param Rmax: Maximum radioresistance at pO2 = 0.
    :returns: Scaling factor array for the given quantity.
    :rtype: np.ndarray
    """
    return 1 + (R - 1) * (f_max - 1) / (Rmax - 1)


def compute_scaling_factors(
    R: np.ndarray, f_rd_max: float, f_z0_max: float, Rmax: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaling factors for domain radius and z0 (OSMK 2023).

    :param R: Relative radioresistance.
    :param f_rd_max: Max scaling factor for domain radius at pO2 = 0.
    :param f_z0_max: Max scaling factor for z0 at pO2 = 0.
    :param Rmax: Maximum radioresistance at pO2 = 0.
    :returns: Tuple of (f_rd, f_z0) scaling factors.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    f_rd = _compute_scaling_factor_component(R, f_rd_max, Rmax)
    f_z0 = _compute_scaling_factor_component(R, f_z0_max, Rmax) ** 2
    return f_rd, f_z0


def compute_osmk_radioresistance(
    version: str, z_bar_domain: np.ndarray, params
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute relative radioresistance R using the OSMK model.

    Supports both OSMK 2021 and OSMK 2023. Also computes scaling factors
    (f_rd, f_z0) for OSMK 2023 if requested.

    :param version: OSMK model version ('2021' or '2023').
    :param z_bar_domain: Dose-mean specific energy of the domain.
    :param params: Parameter object with required attributes.
    :returns: Tuple (R, f_rd, f_z0). f_rd and f_z0 are None for version '2021'.
    :rtype: tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
    :raises ValueError: If the version is unsupported.
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
    Compute oxygen-corrected alpha (LQ model) using the OSMK model.

    :param z_bar_star_domain: Saturation-corrected specific energy.
    :param R: Relative radioresistance.
    :param params: Object with attributes: alphaL, alphaS, beta0.
    :returns: Oxygen-corrected alpha values.
    :rtype: np.ndarray
    """
    alphaL = params.alphaL
    alphaS = params.alphaS
    beta0 = params.beta0
    
    return alphaL + (alphaS / R) + (beta0 * z_bar_star_domain / (R**2))

def apply_oxygen_correction_beta(z_bar_star_domain: np.ndarray, z_bar_domain: np.ndarray, R: np.ndarray, params) -> np.ndarray:
    """
    Compute oxygen-corrected beta (LQ model) using the OSMK model.

    :param z_bar_star_domain: Saturation-corrected specific energy (Gy).
    :param z_bar_domain: Dose-mean specific energy (Gy).
    :param R: Relative radioresistance.
    :param params: Object with attribute beta0.
    :returns: Oxygen-corrected beta values.
    :rtype: np.ndarray
    """
    beta0 = params.beta0
    
    return (z_bar_star_domain / z_bar_domain) * beta0 / (R**2)
