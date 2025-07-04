"""
Computation of survival fraction (SF) curves from microdosimetric input.

This module defines the method :meth:`SFTable.compute`, which calculates SF
values over a dose grid based on results from an MKTable.

Supported models include:

- MKM (classic)
- SMK (stochastic)
- OSMK 2021/2023 (stochastic with oxygen correction)

The method supports optional interpolation, recomputation, and model switching.
"""

from .core import SFTable
from typing import Union, Literal, Optional
import numpy as np
import pandas as pd
import warnings
from pymkm.mktable.compute import _compute_for_energy_let_pair
from dataclasses import replace
from pymkm.physics.specific_energy import SpecificEnergy
from pymkm.biology.oxygen_effect import (
    compute_osmk_radioresistance,
    apply_oxygen_correction_alpha,
    apply_oxygen_correction_beta
    )

def compute(
    self: SFTable,
    *,
    ion: Union[str, int],
    energy: Optional[float] = None,
    let: Optional[float] = None,
    force_recompute: Optional[bool] = True,
    model: Optional[Literal["classic", "stochastic"]] = None,
    apply_oxygen_effect: bool = False
) -> None:
    """
    Compute survival fraction curve(s) based on microdosimetric inputs.
    
    This function stores the results in ``self.table`` as a list of result dictionaries.
    
    :param ion: Ion name, symbol, or atomic number.
    :type ion: Union[str, int]
    :param energy: Kinetic energy per nucleon [MeV/u]. Interpolated if not provided.
    :type energy: Optional[float]
    :param let: Linear energy transfer [MeV/cm]. Interpolated if not provided.
    :type let: Optional[float]
    :param force_recompute: If True, recomputes results even if cached data exist.
    :type force_recompute: Optional[bool]
    :param model: Microdosimetric model to use: "classic" or "stochastic".
    :type model: Literal["classic", "stochastic"], optional
    :param apply_oxygen_effect: Whether to apply OSMK model. Ignored if model != "stochastic".
    :type apply_oxygen_effect: Optional[bool]
    
    :returns: None. Results are stored in ``self.table``.
    
    :raises ValueError: If inputs are inconsistent, or if required parameters are missing.
    """
   
    params = self.params 
   
    model = model or params.mktable.model_version
    
    if apply_oxygen_effect and model != "stochastic":
        raise ValueError("Oxygen effect (OSMK) can only be applied with model='stochastic'.")

    # Determine if OSMK effect is active
    is_osmk = apply_oxygen_effect and params.pO2 is not None
    osmk_version = None
    if is_osmk:
        has_2021 = all(x is not None for x in (params.zR, params.gamma, params.Rm))
        has_2023 = all(x is not None for x in (params.f_rd_max, params.f_z0_max, params.Rmax))

        if has_2021 and not has_2023:
            osmk_version = "2021"
        elif has_2023 and not has_2021:
            osmk_version = "2023"
            force_recompute = True # <-- force recompute for OSMK 2023
        elif has_2021 and has_2023:
            raise ValueError("Inconsistent OSMK input: cannot provide both 2021 and 2023 parameter sets.")
        else:
            raise ValueError("OSMK model requested but required parameters are missing.")

    mktable = params.mktable
    original_params = replace(mktable.params)

    z0 = mktable.params.z0 or SpecificEnergy.compute_saturation_parameter(
        domain_radius=mktable.params.domain_radius,
        nucleus_radius=mktable.params.nucleus_radius,
        beta0=params.beta0
    )
    mktable.params.z0 = round(z0, 2)
    mktable._refresh_parameters(original_params)

    full_ion_name = mktable.sp_table_set._map_to_fullname(ion)

    if model == "stochastic" and not mktable.params.use_stochastic_model:
        raise ValueError("Stochastic output requested but MKTable was computed in classic mode.")

    ion_data = None
    if mktable.table and full_ion_name in mktable.table:
        ion_data = mktable.table[full_ion_name].get("data")
    else:
        warnings.warn(f"No precomputed data found for ion '{full_ion_name}'. Proceeding with direct computation.")

    def _compute(energy_val, let_val):
        sp = mktable.sp_table_set.get(full_ion_name)
        Z = sp.atomic_number
        params_dict = {
            "model_name": mktable.params.model_name,
            "core_radius_type": mktable.params.core_radius_type,
            "base_points_b": mktable.params.base_points_b,
            "base_points_r": mktable.params.base_points_r,
            "domain_radius": mktable.params.domain_radius,
            "nucleus_radius": mktable.params.nucleus_radius,
            "z0": mktable.params.z0,
            "use_stochastic_model": mktable.params.use_stochastic_model,
            "integration_method": "trapz"
        }
        return _compute_for_energy_let_pair(params_dict, energy=energy_val, let=let_val, atomic_number=Z)

    sp = mktable.sp_table_set.get(full_ion_name)

    if energy is None and let is None:
        raise ValueError("At least one of 'energy' or 'let' must be specified.")

    combinations = []
    calc_info = "interpolated"

    if force_recompute or ion_data is None:
        calc_info = "computed"
        if energy is not None and let is None:
            let = sp.interpolate(energy=energy)[0]
            combinations = [(energy, let)]
        elif let is not None and energy is None:
            energy_map = sp.interpolate(let=let)
            energy_vals = energy_map[let]
            combinations = [(e, let) for e in energy_vals]
        else:
            combinations = [(energy, let)]

    else:
        if energy is not None and let is not None:
            combinations = [(energy, let)]
        elif energy is not None:
            match = ion_data[ion_data["energy"] == energy]
            if not match.empty:
                lets = match["let"].unique()
                combinations = [(energy, l) for l in lets]
            else:
                let = sp.interpolate(energy=energy)[0]
                combinations = [(energy, let)]
                calc_info = "computed"
        elif let is not None:
            energy_map = sp.interpolate(let=let)
            energy_vals = energy_map[let]
            combinations = [(e, let) for e in energy_vals]
            calc_info = "computed"

    results = []
    for E, L in combinations:
        result = _compute(E, L)
        z_bar_star_domain = result["z_bar_star_domain"]
        z_bar_domain = result.get("z_bar_domain")
        z_bar_nucleus = result.get("z_bar_nucleus")

        alpha0 = params.alpha0
        beta0 = params.beta0
        dose_grid = params.dose_grid

        alpha_MKM = alpha0 + beta0 * z_bar_star_domain
        beta_MKM = beta0

        if model == "classic":
            sf_curve = np.exp(-alpha_MKM * dose_grid - beta_MKM * dose_grid ** 2)

        elif model == "stochastic":                       
            if is_osmk:                
                R, f_rd, f_z0 = compute_osmk_radioresistance(osmk_version, z_bar_domain, params)
                               
                if f_rd is not None and f_z0 is not None:
                    rd_OSMK = round(mktable.params.domain_radius / f_rd, 2)
                    z0_OSMK = round(z0 * f_z0, 2)
                    
                    # Apply OSMK-adjusted parameters and recompute specific energies
                    original_mktable_params = replace(mktable.params)
                    mktable.params.domain_radius = rd_OSMK
                    mktable.params.z0 = z0_OSMK
                    mktable._refresh_parameters(original_mktable_params)
                    
                    # Recompute microdosimetric results with OSMK-adjusted parameters
                    result_osmk = _compute(E, L)
                    z_bar_star_domain = result_osmk["z_bar_star_domain"]
                    z_bar_domain = result_osmk["z_bar_domain"]
                    z_bar_nucleus = result_osmk["z_bar_nucleus"]
                    
                    # Restore original parameters after OSMK computation
                    mktable._refresh_parameters(original_params)
                    
                z_bar_domain_safe = np.where(z_bar_domain == 0, np.finfo(float).eps, z_bar_domain)                
                alpha_SMK = apply_oxygen_correction_alpha(z_bar_star_domain, R, params)
                beta_SMK  = apply_oxygen_correction_beta(z_bar_star_domain, z_bar_domain_safe, R, params)
            
            else:
                z_bar_domain_safe = np.where(z_bar_domain == 0, np.finfo(float).eps, z_bar_domain)
                alpha_SMK = alpha_MKM
                beta_SMK  = (z_bar_star_domain / z_bar_domain_safe) * beta_MKM 

            gamma_SMK = z_bar_nucleus * (0.5 * (alpha_SMK + 2 * beta_SMK * dose_grid) ** 2 - beta_SMK)
            exponent = alpha_SMK * dose_grid + beta_SMK * dose_grid ** 2
            correction_factor = 1 + gamma_SMK * dose_grid
            sf_curve = np.exp(-exponent) * correction_factor
       
        results.append({
            "params": {
                "ion": full_ion_name,
                "energy": float(E),
                "let": float(L),
                "model": model,
                "osmk_version": osmk_version if is_osmk else None
            },
            "calculation_info": calc_info,
            "data": pd.DataFrame({
                "dose": dose_grid,
                "survival_fraction": sf_curve
            })
        })
        
    self.table = results

SFTable.compute = compute