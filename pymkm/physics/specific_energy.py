"""
Computation of microdosimetric specific energy quantities.

This module implements the :class:`SpecificEnergy`, which calculates:

- Single-event specific energy z(b)
- Saturation correction using z₀ (square-root or quadratic model)
- Dose-averaged specific energy z̄
- OSMK 2023 oxygen-effect corrections (via domain radius and z₀ scaling)

All calculations are based on radial dose profiles provided by a ParticleTrack instance,
and integrate over geometry defined by a sensitive cylindrical volume.
"""

import time
import numpy as np
from typing import Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from scipy.integrate import simpson, quad
from scipy.interpolate import interp1d

from pymkm.physics.particle_track import ParticleTrack
from pymkm.utils.geometry_tools import GeometryTools
from pymkm.utils.parallel import optimal_worker_count

class SpecificEnergy:
    """
    Compute microdosimetric specific energy quantities for a single ion track.

    This class provides methods for calculating:
      - Single-event specific energy z(b)
      - Saturation parameter z₀
      - Saturation-corrected specific energy z′(b), z_sat(b)
      - Dose-averaged specific energy z̄

    The sensitive region is modeled as a cylinder perpendicular to the ion track.
    """

    def __init__(
        self, particle_track: ParticleTrack, region_radius: float
    ) -> None:
        """
        Initialize a SpecificEnergy instance.
    
        :param particle_track: A ParticleTrack instance describing the ion's radial dose distribution.
        :type particle_track: pymkm.physics.particle_track.ParticleTrack
        :param region_radius: Radius of the sensitive region (e.g., domain or nucleus), in micrometers.
        :type region_radius: float
    
        :raises TypeError: If `particle_track` is not a ParticleTrack instance.
        :raises ValueError: If `region_radius` is not positive.
        """
        if not isinstance(particle_track, ParticleTrack):
            raise TypeError("particle_track must be an instance of ParticleTrack.")
        if region_radius <= 0:
            raise ValueError("region_radius must be positive.")

        self.track = particle_track
        self.region_radius = float(region_radius)
        self.penumbra_radius = self.track.penumbra_radius

    def single_event_specific_energy(
        self,
        impact_parameters: Optional[np.ndarray] = None,
        base_points_b: Optional[int] = None,
        base_points_r: Optional[int] = None,
        parallel: bool = False,
        return_time: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float]]:
        """
        Compute the single-event specific energy z(b) as a function of impact parameter.
    
        :param impact_parameters: Optional array of impact parameters in micrometers. If None, a default grid is generated.
        :type impact_parameters: np.ndarray, optional
        :param base_points_b: Number of impact parameter points. If None, defaults to internal base setting.
        :type base_points_b: int, optional
        :param base_points_r: Number of radial sampling points. If None, defaults to internal base setting.
        :type base_points_r: int, optional
        :param parallel: Whether to parallelize over impact parameters.
        :type parallel: bool, optional
        :param return_time: Whether to also return elapsed wall time in seconds.
        :type return_time: bool, optional
    
        :returns: Tuple of (z_array, b_array) or (z_array, b_array, elapsed_time).
        :rtype: tuple[np.ndarray, np.ndarray] or tuple[np.ndarray, np.ndarray, float]
        """
        start_time = time.perf_counter()

        if impact_parameters is None or (
            isinstance(impact_parameters, np.ndarray) and impact_parameters.size == 0
        ):
            b_max = self.region_radius + self.penumbra_radius
            default_b_pts = GeometryTools.generate_default_radii.__defaults__[1]
            impact_parameters = GeometryTools.generate_default_radii(
                energy=self.track.energy,
                radius_max=b_max,
                base_points=base_points_b if base_points_b is not None else default_b_pts
            )

        b_array = np.sort(np.asarray(impact_parameters).flatten())
        if parallel:
            workers = optimal_worker_count(b_array)
            func = partial(self._compute_z_single_b, base_points_r=base_points_r)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                z_vals = list(executor.map(func, b_array))
        else:
            z_vals = [self._compute_z_single_b(b, base_points_r) for b in b_array]

        z_array = np.array(z_vals)
        elapsed_time = time.perf_counter() - start_time if return_time else None
        return (z_array, b_array, elapsed_time) if return_time else (z_array, b_array)

    def _compute_z_single_b(
        self, b: float, base_points_r: Optional[int] = None
    ) -> float:
        """
        Compute the single-event specific energy z(b) for a given impact parameter.
    
        :param b: Impact parameter in micrometers.
        :type b: float
        :param base_points_r: Number of radial sampling points.
        :type base_points_r: int, optional
    
        :returns: Specific energy value at impact parameter b in Gy.
        :rtype: float
        """
        r_min = max(1e-6, b - self.region_radius)
        r_max = min(b + self.region_radius, self.penumbra_radius)
        default_r_pts = GeometryTools.generate_default_radii.__defaults__[1]
        r_array = GeometryTools.generate_default_radii(
            energy=self.track.energy,
            radius_max=r_max,
            radius_min=r_min,
            base_points=base_points_r if base_points_r is not None else default_r_pts
        )
        dose_profile, _ = self.track.initial_local_dose(radius=r_array)
        values = dose_profile.ravel()[:-1]
        areas = GeometryTools.calculate_intersection_area(r_array, self.region_radius, b)
        diff_areas = np.diff(areas)
        z_b = np.sum(values * diff_areas)
        return z_b / (np.pi * self.region_radius**2)

    @staticmethod
    def compute_saturation_parameter(
        domain_radius: float,
        nucleus_radius: float,
        beta0: float
    ) -> float:
        """
        Compute the saturation parameter z₀ for overkill correction.
    
        :param domain_radius: Radius of the sensitive domain in micrometers.
        :type domain_radius: float
        :param nucleus_radius: Radius of the nucleus in micrometers.
        :type nucleus_radius: float
        :param beta0: Quadratic coefficient β₀ of the LQ model at low LET (Gy⁻²).
        :type beta0: float
    
        :returns: Saturation parameter z₀ in Gy.
        :rtype: float
    
        :raises ValueError: If any input is non-positive.
        """
        if nucleus_radius <= 0 or domain_radius <= 0 or beta0 <= 0:
            raise ValueError("All input parameters (domain_radius, nucleus_radius, beta0) must be > 0.")
        ratio = nucleus_radius / domain_radius
        return (ratio ** 2) / np.sqrt(beta0 * (1 + ratio ** 2))

    def saturation_corrected_single_event_specific_energy(
        self,
        z0: float,
        z_array: np.ndarray,
        model: str = "square_root"
    ) -> np.ndarray:
        """
        Apply saturation correction to single-event specific energy.
    
        :param z0: Saturation parameter z₀ in Gy.
        :type z0: float
        :param z_array: Array of uncorrected z(b) values.
        :type z_array: np.ndarray
        :param model: Correction model to use ('square_root' or 'quadratic').
        :type model: str, optional
    
        :returns: Array of corrected specific energy values.
        :rtype: np.ndarray
    
        :raises ValueError: If model is unsupported.
        """
        z_prime = z0 * np.sqrt(1 - np.exp(-(z_array ** 2) / (z0 ** 2)))
        if model == "square_root":
            return z_prime
        elif model == "quadratic":
            eps = 1e-12
            z_sat = np.zeros_like(z_array)
            mask = z_array > eps
            z_sat[mask] = (z_prime[mask] ** 2) / z_array[mask]
            return z_sat
        else:
            raise ValueError(f"Unsupported model '{model}'.")

    def dose_averaged_specific_energy(
        self,
        z_array: np.ndarray,
        b_array: np.ndarray,
        z_corrected: Optional[np.ndarray] = None,
        model: str = "square_root",
        integration_method: str = "trapz"
    ) -> float:
        """
        Compute the dose-averaged specific energy z̄.
    
        :param z_array: Uncorrected z(b) values (Gy).
        :type z_array: np.ndarray
        :param b_array: Impact parameter values (μm), must be sorted.
        :type b_array: np.ndarray
        :param z_corrected: Optional corrected z_eff(b) values.
        :type z_corrected: np.ndarray, optional
        :param model: Correction model ('square_root' or 'quadratic').
        :type model: str, optional
        :param integration_method: Integration rule to use: 'trapz', 'simps', or 'quad'.
        :type integration_method: str, optional
    
        :returns: Dose-averaged specific energy z̄ in Gy.
        :rtype: float
    
        :raises ValueError: If integration method or model is invalid.
        """
    
        def integrate(y: np.ndarray) -> float:
            if integration_method == "trapz":
                return np.trapz(y, b_array)
            elif integration_method == "simps":
                return simpson(y=y, x=b_array)
            elif integration_method == "quad":
                f_interp = interp1d(b_array, y, kind="cubic", fill_value="extrapolate")
                result, _ = quad(f_interp, b_array[0], b_array[-1], limit=100)
                return result
            else:
                raise ValueError(f"Unsupported integration method: '{integration_method}'")
    
        if z_corrected is None:
            numerator = integrate(z_array ** 2 * b_array)
        else:
            if model == "square_root":
                numerator = integrate(z_corrected ** 2 * b_array)
            elif model == "quadratic":
                numerator = integrate(z_array * z_corrected * b_array)
            else:
                raise ValueError("Model must be 'square_root' or 'quadratic' when providing z_corrected.")
    
        denom = integrate(z_array * b_array)
        return 0.0 if denom == 0 else numerator / denom