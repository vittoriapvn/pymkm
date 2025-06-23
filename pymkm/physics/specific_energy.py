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
    SpecificEnergy handles all computations related to microdosimetric specific energy,
    including:
      - Single-event specific energy z(b) as a function of impact parameter b
      - Saturation parameter z0 for overkill correction
      - Saturation-corrected profiles z'(b) and z_sat(b)
      - Dose-averaged specific energy z̄

    The sensitive region is modeled as a cylinder of radius `region_radius`,
    perpendicular to the ion track.
    """

    def __init__(
        self, particle_track: ParticleTrack, region_radius: float
    ) -> None:
        """
        Initialize the SpecificEnergy instance.

        Parameters
        ----------
        particle_track : ParticleTrack
            An instance providing local dose distribution and penumbra_radius.
        region_radius : float
            Radius (in μm) of the sensitive region (e.g., domain or nucleus).

        Raises
        ------
        TypeError
            If `particle_track` is not a ParticleTrack instance.
        ValueError
            If `region_radius` ≤ 0.
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
        Compute single-event specific energy z(b) over a set of impact parameters.

        Parameters
        ----------
        impact_parameters : Optional[np.ndarray]
            Array of impact parameters (μm). If None, a default array is generated based on geometry.
        base_points_b : Optional[int]
            Number of sampling points for b. Defaults to GeometryTools default if None.
        base_points_r : Optional[int]
            Number of radial sampling points. Defaults to GeometryTools default if None.
        parallel : bool, optional
            Whether to parallelize computation over b values (default: False).
        return_time : bool, optional
            If True, also return elapsed computation time (s) (default: False).

        Returns
        -------
        z_array : np.ndarray
            Specific energy values (Gy) sorted by b.
        b_array : np.ndarray
            Corresponding sorted impact parameters (μm).
        elapsed_time : float, optional
            Elapsed wall time if return_time is True.
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
        Compute single-event specific energy z(b) for a single impact parameter.

        Parameters
        ----------
        b : float
            Impact parameter (μm).
        base_points_r : Optional[int]
            Number of radial sampling points.

        Returns
        -------
        float
            Specific energy (Gy) at impact parameter b.
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
        Compute the saturation parameter z₀ (in Gy) based on the geometric ratio between 
        nucleus and domain, and the empirical coefficient β₀.
       
        Parameters
        ----------
        domain_radius : float
            Radius of the sensitive domain within the nucleus in μm (e.g., 0.2–0.4 μm).
        nucleus_radius : float
            Radius of the cell nucleus in μm (e.g., 5–10 μm).
        beta0 : float
            Coefficient in Gy⁻² of the quadratic term in the linear-quadratic (LQ)
            survival model in the limit of vanishing LET (e.g., 0.05 Gy⁻²).
    
        Returns
        -------
        float
            The saturation parameter z₀, in units of Gy.
    
        Raises
        ------
        ValueError
            If any input is non-positive.
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
        Compute the saturation-corrected single-event specific energy using either
        a square-root or quadratic correction model.

        Parameters
        ----------
        z0 : float
            Saturation parameter z₀ (Gy), provided by the user.
        z_array : np.ndarray
            Precomputed single-event specific energy.
        model : str, optional
            Saturation correction model:
                - "square_root" (default): 
                    z′(b) = z₀ · √[1 − exp(−z² / z₀²)] 
                    (Inaniwa et al., 2021)
                - "quadratic": 
                    z_sat(b) = z′² / z 
                    (based on Inaniwa et al., 2010)

        Returns
        -------
        np.ndarray
            Corrected specific energy: z' or z_sat (Gy).

        Raises
        ------
        ValueError
            If model not in ['square_root','quadratic'].
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
        Compute the dose-averaged specific energy z̄ using the selected integration method.
    
        The computation follows the general form:
            z̄ = ( ∫ z_eff(b) * z(b) * b  db ) / ( ∫ z(b) * b  db )
        where z_eff depends on the chosen model:
            - 'square_root': z_eff = z'(b)
            - 'quadratic' : z_eff = z_sat(b)
            - uncorrected: z_eff = z(b)
    
        Parameters
        ----------
        z_array : np.ndarray
            Uncorrected specific energy values z(b) [Gy].
        b_array : np.ndarray
            Impact parameters b [μm]. Must be 1D and sorted.
        z_corrected : Optional[np.ndarray], optional
            Saturation-corrected values z_eff(b), by default None.
        model : str, optional
            Saturation correction model: 'square_root' or 'quadratic'.
        integration_method : str, optional
            Integration method to use: 'trapz', 'simps', 'romb', or 'quad'.
            - 'trapz': Trapezoidal rule (numpy)
            - 'simps': Simpson's rule (scipy)
            - 'quad' : Quadrature with cubic interpolation (scipy)
    
        Returns
        -------
        float
            Dose-averaged specific energy z̄ [Gy].
    
        Raises
        ------
        ValueError
            If an unsupported model or integration method is provided.
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