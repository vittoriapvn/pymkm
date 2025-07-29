import numpy as np

def log_error_metrics(x_ref, y_ref, x_model, y_model):
    """
    Compute multiple log-scale error metrics between a reference curve and a model prediction.

    The function interpolates the model data onto the reference domain and evaluates
    pointwise differences in base-10 logarithmic space. It is particularly suited for
    comparing local dose or specific energy distributions in log-log scale.

    Parameters
    ----------
    x_ref : array-like
        X-values of the reference dataset (e.g., radial distances).
    y_ref : array-like
        Y-values of the reference dataset (e.g., dose values).
    x_model : array-like
        X-values of the model-predicted dataset.
    y_model : array-like
        Y-values of the model-predicted dataset.

    Returns
    -------
    dict
        Dictionary containing the following error metrics:
        
        - 'mean_log_error' : float
            Mean absolute log₁₀ error between reference and interpolated model.
        
        - 'rms_log_error' : float
            Root mean squared log₁₀ error.
        
        - 'max_log_error' : float
            Maximum absolute log₁₀ deviation.
        
        - 'smape_log' : float
            Symmetric mean absolute percentage error in log₁₀ space (as a percentage).

    Raises
    ------
    ValueError
        If no valid overlapping domain with positive, finite values exists.

    Notes
    -----
    - All metrics are computed in base-10 logarithmic space.
    - Only strictly positive and finite values are considered valid.
    - Interpolation is linear and extrapolated values are excluded.
    """
    
    x_ref = np.asarray(x_ref)
    y_ref = np.asarray(y_ref)
    x_model = np.asarray(x_model)
    y_model = np.asarray(y_model)

    y_model_interp = np.interp(x_ref, x_model, y_model, left=np.nan, right=np.nan)
    valid = (~np.isnan(y_model_interp)) & (y_ref > 0) & (y_model_interp > 0)

    if not np.any(valid):
        raise ValueError("No valid overlapping domain between reference and model.")

    log_ref = np.log10(y_ref[valid])
    log_model = np.log10(y_model_interp[valid])
    delta = log_model - log_ref

    denom = np.abs(log_ref) + np.abs(log_model)
    smape = np.mean(np.abs(delta) / denom) * 100 if np.any(denom > 0) else np.nan

    return {
        'mean_log_error': np.mean(np.abs(delta)),
        'rms_log_error': np.sqrt(np.mean(delta**2)),
        'max_log_error': np.max(np.abs(delta)),
        'smape_log': smape
    }


def semi_log_error_metrics(x_ref, y_ref, x_model, y_model):
    """
    Compute error metrics in semi-log scale: X is logarithmic, Y is linear.

    Parameters
    ----------
    x_ref : array-like
        X-values (e.g. energy or LET) from the reference dataset.
    y_ref : array-like
        Corresponding Y-values (e.g. z_d, D10) from the reference dataset.
    x_model : array-like
        X-values from the model-generated dataset.
    y_model : array-like
        Corresponding Y-values from the model-generated dataset.

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean_abs_error'
        - 'rms_error'
        - 'max_abs_error'
        - 'smape' (Symmetric Mean Absolute Percentage Error)
    """
    x_ref = np.asarray(x_ref)
    y_ref = np.asarray(y_ref)
    x_model = np.asarray(x_model)
    y_model = np.asarray(y_model)

    # Interpolate model values on the log10(x_ref) domain
    logx_ref = np.log10(x_ref)
    logx_model = np.log10(x_model)

    y_model_interp = np.interp(logx_ref, logx_model, y_model, left=np.nan, right=np.nan)
    valid = (~np.isnan(y_model_interp)) & np.isfinite(y_ref) & np.isfinite(y_model_interp)

    if not np.any(valid):
        raise ValueError("No valid overlapping semi-log domain.")

    diff = y_model_interp[valid] - y_ref[valid]
    denom = np.abs(y_ref[valid]) + np.abs(y_model_interp[valid])
    smape = np.mean(np.abs(diff) / denom) * 100 if np.any(denom > 0) else np.nan

    return {
        'mean_abs_error': np.mean(np.abs(diff)),
        'rms_error': np.sqrt(np.mean(diff**2)),
        'max_abs_error': np.max(np.abs(diff)),
        'smape_percent': smape
    }


def log_linear_error_metrics(x_ref, y_ref, x_model, y_model):
    """
    Compute error metrics in log-linear scale: X is linear (e.g., dose), Y is logarithmic (e.g., survival fraction).

    Parameters
    ----------
    x_ref : array-like
        X-values (e.g., dose) from the reference dataset.
    y_ref : array-like
        Corresponding Y-values (e.g., survival fraction) from the reference dataset.
    x_model : array-like
        X-values from the model-generated dataset.
    y_model : array-like
        Corresponding Y-values from the model-generated dataset.

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean_log_error': Mean absolute log₁₀ error
        - 'rms_log_error' : Root mean squared log₁₀ error
        - 'max_log_error' : Maximum absolute log₁₀ error
        - 'smape_log'     : Symmetric Mean Absolute Percentage Error in log₁₀ scale (percentage)

    Raises
    ------
    ValueError
        If no valid overlapping domain exists or all values are non-positive.
    """
    x_ref = np.asarray(x_ref)
    y_ref = np.asarray(y_ref)
    x_model = np.asarray(x_model)
    y_model = np.asarray(y_model)

    # Interpolate model onto reference X (dose) domain
    y_model_interp = np.interp(x_ref, x_model, y_model, left=np.nan, right=np.nan)

    # Valid if both survival fractions are > 0 and finite
    valid = (~np.isnan(y_model_interp)) & (y_ref > 0) & (y_model_interp > 0)

    if not np.any(valid):
        raise ValueError("No valid overlapping log-linear domain.")

    log_ref = np.log10(y_ref[valid])
    log_model = np.log10(y_model_interp[valid])
    delta = log_model - log_ref

    denom = np.abs(log_ref) + np.abs(log_model)
    smape = np.mean(np.abs(delta) / denom) * 100 if np.any(denom > 0) else np.nan

    return {
        'mean_log_error': np.mean(np.abs(delta)),
        'rms_log_error': np.sqrt(np.mean(delta**2)),
        'max_log_error': np.max(np.abs(delta)),
        'smape_log': smape
    }
