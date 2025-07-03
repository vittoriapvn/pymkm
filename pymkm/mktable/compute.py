"""
Computation engine for MKTable.

This module defines the logic to compute microdosimetric quantities
(z̄*, z̄_d, z̄_n) for a set of ions based on the MKM or SMK model using energy–LET tables.

It integrates track structure modeling, specific energy calculation,
saturation correction, and optional oxygen-effect scaling (OSMK 2023).
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dataclasses import asdict, replace

from pymkm.physics.particle_track import ParticleTrack
from pymkm.physics.specific_energy import SpecificEnergy
from pymkm.utils.parallel import optimal_worker_count
from pymkm.biology.oxygen_effect import compute_relative_radioresistance, compute_scaling_factors

from .core import MKTable

def _run_energy_let_task(func, args):
    """
    Internal wrapper for multiprocessing task execution.

    :param func: Callable function to execute.
    :type func: Callable
    :param args: Tuple of arguments for the function.
    :type args: tuple
    :return: Output of func(*args)
    """
    return func(*args)

def _compute_for_energy_let_pair(
    params: dict,
    energy: float,
    let: float,
    atomic_number: int
) -> dict:
    """
    Compute specific energies for one (energy, LET) pair.

    :param params: Flattened MKTableParameters dictionary.
    :type params: dict
    :param energy: Kinetic energy per nucleon [MeV/u].
    :type energy: float
    :param let: Linear energy transfer [MeV/cm].
    :type let: float
    :param atomic_number: Atomic number of the ion.
    :type atomic_number: int

    :return: Dictionary of computed microdosimetric quantities.
    :rtype: dict
    """
    track = ParticleTrack(
        model_name=params["model_name"],
        core_radius_type=params["core_radius_type"],
        energy=energy,
        atomic_number=atomic_number,
        let=let,
        base_points=params["base_points_r"],
    )

    se_domain = SpecificEnergy(track, region_radius=params["domain_radius"])

    z_domain, b_domain = se_domain.single_event_specific_energy(
        base_points_b=params["base_points_b"],
        base_points_r=params["base_points_r"]
    )

    z_prime_domain = se_domain.saturation_corrected_single_event_specific_energy(
        z0=params["z0"], z_array=z_domain
    )

    z_bar_star_domain = se_domain.dose_averaged_specific_energy(
        z_array=z_domain,
        b_array=b_domain,
        z_corrected=z_prime_domain,
        integration_method=params["integration_method"]
    )

    result = {
        "z_bar_star_domain": z_bar_star_domain
    }

    if params["use_stochastic_model"]:
        z_bar_domain = se_domain.dose_averaged_specific_energy(
            z_array=z_domain,
            b_array=b_domain,
            integration_method=params["integration_method"]
        )

        se_nucleus = SpecificEnergy(track, region_radius=params["nucleus_radius"])

        z_nucleus, b_nucleus = se_nucleus.single_event_specific_energy(
            base_points_b=params["base_points_b"],
            base_points_r=params["base_points_r"]
        )

        z_bar_nucleus = se_nucleus.dose_averaged_specific_energy(
            z_array=z_nucleus,
            b_array=b_nucleus,
            integration_method=params["integration_method"]
        )

        result.update({
            "z_bar_domain": z_bar_domain,
            "z_bar_nucleus": z_bar_nucleus
        })

    return result

def _get_osmk2023_corrected_parameters(mktable: MKTable) -> tuple[float, float]:
    """
    Compute oxygen-effect–corrected domain radius and z₀ using the OSMK 2023 model.

    :param mktable: MKTable instance containing OSMK parameters.
    :type mktable: MKTable

    :returns: Tuple with (corrected_domain_radius, corrected_z0).
    :rtype: tuple[float, float]

    :raises ValueError: If required OSMK parameters are missing.
    """
    
    p = mktable.params

    R = compute_relative_radioresistance(K=p.K, pO2=p.pO2, K_mult=1 / p.Rmax)
    f_rd, f_z0 = compute_scaling_factors(R, p.f_rd_max, p.f_z0_max, p.Rmax)

    rd_eff = round(p.domain_radius / f_rd, 3)
    z0_eff = round(p.z0 * f_z0, 2)
    return rd_eff, z0_eff

def _compute_for_ion(self: MKTable, ion: str, parallel: bool = True, integration_method: str = "trapz"):
    """
    Compute all specific energies for a given ion in the table set.

    :param ion: Ion identifier (e.g., "C", "Carbon", 6).
    :type ion: str
    :param parallel: Whether to use parallel processing.
    :type parallel: bool
    :param integration_method: Numerical integration method ('trapz', 'simps', or 'quad').
    :type integration_method: str

    :returns: Tuple of ion name and list of computed data entries.
    :rtype: tuple[str, list[dict]]
    """
    
    start_time = time.time()
    sp = self.sp_table_set.get(ion)
    energy_grid = sp.energy
    let_grid = sp.let

    job_list = [(E, L, sp.atomic_number) for E, L in zip(energy_grid, let_grid)]
    results = []
    
    # Prepare oxygen-corrected geometry if requested
    if self.params.apply_oxygen_effect:
        rd_eff, z0_eff = _get_osmk2023_corrected_parameters(self)
        
        print("✔ Using OSMK2023-corrected values:")
        print(f" - domain_radius: {self.params.domain_radius} → {rd_eff}")
        print(f" - z0: {self.params.z0} → {z0_eff}")
    else:
        rd_eff = self.params.domain_radius
        z0_eff = self.params.z0

    if parallel:
        worker_count = optimal_worker_count(job_list)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            params_dict = {
                "model_name": self.params.model_name,
                "core_radius_type": self.params.core_radius_type,
                "base_points_b": self.params.base_points_b,
                "base_points_r": self.params.base_points_r,
                "domain_radius": rd_eff,
                "nucleus_radius": self.params.nucleus_radius,
                "z0": z0_eff,
                "use_stochastic_model": self.params.use_stochastic_model,
                "integration_method": integration_method
            }
            func = partial(self._compute_for_energy_let_pair, params_dict)
            results = list(tqdm(
                executor.map(partial(_run_energy_let_task, func), job_list),
                total=len(job_list),
                desc=f"[{worker_count} workers] {sp.ion_name} ({sp.atomic_number},{sp.mass_number})",
                unit="energy"
            ))
    else:
        for args in tqdm(job_list, desc=f"{sp.ion_name} ({sp.atomic_number},{sp.mass_number})", unit="energy"):
            params_dict = {
                "model_name": self.params.model_name,
                "core_radius_type": self.params.core_radius_type,
                "base_points_b": self.params.base_points_b,
                "base_points_r": self.params.base_points_r,
                "domain_radius": rd_eff,
                "nucleus_radius": self.params.nucleus_radius,
                "z0": z0_eff,
                "use_stochastic_model": self.params.use_stochastic_model,
                "integration_method": integration_method
            }
            results.append(self._compute_for_energy_let_pair(params_dict, *args))

    for job, result in zip(job_list, results):
        E, L, _ = job
        result.update({"energy": E, "let": L})

    elapsed = time.time() - start_time
    print(f"\n  ... completed in {elapsed:.2f} seconds.\n")
    return ion, results

def compute(
    self: MKTable,
    ions: Optional[List[Union[str, int]]] = None,
    energy: Optional[Union[float, List[float], np.ndarray]] = None,
    parallel: bool = True,
    integration_method: str = "trapz"
) -> None:
    """
    Compute per-ion microdosimetric tables using MKM or SMK model.

    For each ion:
      - Retrieves energy–LET grid
      - Computes specific energy and dose-averaged quantities
      - Applies saturation correction and optional OSMK
      - Aggregates into a structured table

    :param self: MKTable instance.
    :type self: MKTable
    :param ions: Ion identifiers to compute. If None, all available ions are used.
    :type ions: list[str or int], optional
    :param energy: Custom energy grid for resampling (if any).
    :type energy: float or list or np.ndarray, optional
    :param parallel: Whether to enable multiprocessing.
    :type parallel: bool
    :param integration_method: Integration scheme ('trapz', 'simps', 'quad').
    :type integration_method: str

    :raises RuntimeError: If MKTable is not initialized.
    """
    
    if not self.sp_table_set or not self.params:
        raise RuntimeError("MKTable is not properly initialized.")

    ions = ions or self.sp_table_set.get_available_ions()
    ions = [self.sp_table_set._map_to_fullname(ion) for ion in ions]

    if energy is not None:
        custom_energy = np.atleast_1d(np.array(energy, dtype=float))
        self.sp_table_set.resample_all(custom_energy)

    original_params = replace(self.params)

    z0 = self.params.z0 or SpecificEnergy.compute_saturation_parameter(
        domain_radius=self.params.domain_radius,
        nucleus_radius=self.params.nucleus_radius,
        beta0=self.params.beta0
    )
    self.params.z0 = round(z0, 2)

    self._refresh_parameters(original_params)

    print("\nStarting table computation in {} mode ...".format("parallel" if parallel else "serial"))
    print(f"\nIons to be computed: {', '.join(ions)}\n")
    overall_start = time.time()
    results = {}
    for ion in ions:
        ion_key = self.sp_table_set._map_to_fullname(ion)
        ion, ion_results = self._compute_for_ion(ion, parallel=parallel, integration_method=integration_method)
        results[ion_key] = ion_results

    enriched_results = {}
    for ion_key, data in results.items():
        sp = self.sp_table_set.get(ion_key)
        rows = []
        for entry in data:
            row = {
                "energy": entry["energy"],
                "let": entry["let"],
                "z_bar_star_domain": entry.get("z_bar_star_domain"),
                "z_bar_domain": entry.get("z_bar_domain"),
                "z_bar_nucleus": entry.get("z_bar_nucleus")
            }
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("energy").reset_index(drop=True)

        sp_metadata = sp.to_dict()
        sp_metadata.pop("energy", None)
        sp_metadata.pop("let", None)

        enriched_results[ion_key] = {
            "params": asdict(self.params),
            "stopping_power_info": sp_metadata,
            "data": df
        }

    print("\n... finalizing results and updating ...")
    sorted_keys = sorted(
        enriched_results.keys(),
        key=lambda k: enriched_results[k]["stopping_power_info"].get("atomic_number")
        )
    self.table = {k: enriched_results[k] for k in sorted_keys}

    total_elapsed = time.time() - overall_start
    print(f"\n... done. Total elapsed time: {total_elapsed:.2f} seconds.")

MKTable.compute = compute
MKTable._compute_for_energy_let_pair = staticmethod(_compute_for_energy_let_pair)
MKTable._compute_for_ion = _compute_for_ion
