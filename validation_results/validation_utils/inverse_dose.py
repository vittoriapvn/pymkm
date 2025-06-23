import math
import numpy as np

def compute_lq_dose_from_survival(alpha: float, beta: float, S: float) -> float:
    """
    Compute the radiation dose corresponding to a given survival fraction
    using the linear-quadratic (LQ) model:
    
        S(D) = exp(-alpha * D - beta * D^2)
    
    This function solves the inverse problem to find D given alpha, beta, and S.
    
    Parameters:
    - alpha (float): linear coefficient (must be >= 0)
    - beta (float): quadratic coefficient (must be >= 0)
    - S (float): survival fraction (must be in the open interval (0, 1])
    
    Returns:
    - D (float): radiation dose corresponding to the input survival fraction
    
    Raises:
    - ValueError: if inputs are outside the valid range or the solution is not real
    """
    if not (0 < S <= 1):
        raise ValueError("Survival fraction S must be in the open interval (0, 1].")
    if alpha < 0 or beta < 0:
        raise ValueError("Alpha and beta must be non-negative.")
    if beta == 0:
        if alpha == 0:
            raise ValueError("At least one of alpha or beta must be non-zero.")
        return -math.log(S) / alpha

    discriminant = alpha**2 - 4 * beta * math.log(S)
    if discriminant < 0:
        raise ValueError("No real solution exists for the given parameters.")
    
    D = (-alpha + math.sqrt(discriminant)) / (2 * beta)
    return D


def inverse_dose_from_survival(doses: np.ndarray, survivals: np.ndarray, S_target: float) -> float:
    """
    Estimate the radiation dose corresponding to a given survival fraction using
    numpy linear interpolation.
    
    Assumes the survival curve is monotonic and defined over a 1D grid of doses.
    
    Parameters
    ----------
    doses : np.ndarray
        Array of dose values (must be increasing).
    survivals : np.ndarray
        Array of survival fractions corresponding to doses (must be monotonic).
    S_target : float
        Target survival fraction (must be within survivals.min() and survivals.max()).
    
    Returns
    -------
    float
        Dose corresponding to the given survival value.
    
    Raises
    ------
    ValueError
        If inputs are invalid or S_target is out of range.
    """
    doses = np.asarray(doses)
    survivals = np.asarray(survivals)
    
    if not (0 < S_target <= 1):
        raise ValueError("S_target must be in the interval (0, 1].")
    if not np.all(np.diff(doses) > 0):
        raise ValueError("Dose values must be strictly increasing.")
    
    # Determine monotonicity
    if np.all(np.diff(survivals) < 0):  # survival is decreasing
        inv_dose = np.interp(S_target, survivals[::-1], doses[::-1])
    elif np.all(np.diff(survivals) > 0):  # survival is increasing
        inv_dose = np.interp(S_target, survivals, doses)
    else:
        raise ValueError("Survival curve must be monotonic for interpolation.")
    
    return inv_dose
