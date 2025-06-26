"""
Radial dose models for charged particle tracks.

This module defines the :class:`ParticleTrack`, which implements two analytical models:

- Scholz-Kraft model (Phys. Med. Biol., 1996)
- Kiefer-Chatterjee model (Radiat. Environ. Biophys., 1988)

Both models compute local dose as a function of radial distance from the ion trajectory,
based on physical parameters such as energy, atomic number, LET, and core radius type.

These dose profiles serve as the basis for computing specific energy deposition
in MKM and SMK.
"""

import numpy as np
import logging
from typing import Optional, Union, Tuple

from pymkm.utils.geometry_tools import GeometryTools

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class ParticleTrack:
    """
    Model of radial dose deposition around a particle track.

    Supports the Scholz-Kraft and Kiefer-Chatterjee analytical models
    for describing the initial local dose as a function of radial distance
    from the ion trajectory.
    """

    def __init__(self,
                 model_name: str = 'Kiefer-Chatterjee',
                 core_radius_type: str = 'energy-dependent',
                 energy: Optional[float] = None,
                 atomic_number: Optional[int] = None,
                 let: Optional[float] = None,
                 base_points: int = GeometryTools.generate_default_radii.__defaults__[1]) -> None:
        """
        Initialize a ParticleTrack instance.
    
        :param model_name: The dose model to use ('Scholz-Kraft' or 'Kiefer-Chatterjee').
        :type model_name: str
        :param core_radius_type: Core radius mode, either 'constant' or 'energy-dependent'.
        :type core_radius_type: str
        :param energy: Kinetic energy in MeV/u. Required for most calculations.
        :type energy: float, optional
        :param atomic_number: Atomic number (Z) of the ion. Required for the Kiefer-Chatterjee model.
        :type atomic_number: int, optional
        :param let: Unrestricted LET in MeV/cm.
        :type let: float, optional
        :param base_points: Base number of sampling points used to generate radius grid.
        :type base_points: int
    
        :raises ValueError: If required inputs are missing or invalid.
        """
        if core_radius_type not in ['constant', 'energy-dependent']:
            raise ValueError("Invalid core_radius_type. Choose 'constant' or 'energy-dependent'.")
        if model_name not in ['Scholz-Kraft', 'Kiefer-Chatterjee']:
            raise ValueError("Invalid model_name. Choose 'Scholz-Kraft' or 'Kiefer-Chatterjee'.")

        if model_name == 'Kiefer-Chatterjee':
            if core_radius_type == 'constant':
                logger.warning("'Kiefer-Chatterjee' model is incompatible with 'constant' core radius type. Switching to 'energy-dependent'.")
                core_radius_type = 'energy-dependent'
            if atomic_number is None:
                raise ValueError("Atomic number must be provided for the Kiefer-Chatterjee model.")

        if model_name == 'Scholz-Kraft':
            if core_radius_type == 'energy-dependent' and energy is None:
                raise ValueError("Energy must be provided for energy-dependent core radius type in the Scholz-Kraft model.")
            if core_radius_type == 'constant' and energy is not None:
                logger.warning("Energy is ignored when core radius type is 'constant' in the Scholz-Kraft model.")

        self.model_name = model_name
        self.core_radius_type = core_radius_type
        self.energy = float(energy) if energy is not None else None
        self.atomic_number = atomic_number
        self.let = self._convert_let(float(let)) if let is not None else None
        self.base_points = base_points

        # Compute derived quantities
        self.penumbra_radius = self._compute_penumbra_radius()
        self.velocity = self._compute_velocity()
        self.rho = 1e-12  # Density of water in g/μm³

    def _convert_let(self, let_mev_per_cm: float) -> float:
        """
        Convert LET from MeV/cm to Gy·g/μm.
    
        :param let_mev_per_cm: LET value in MeV/cm.
        :type let_mev_per_cm: float
    
        :returns: LET converted to Gy·g/μm.
        :rtype: float
        """
        GEVGRA = 1.602176462e-7
        let_gev_per_um = let_mev_per_cm * 1e-7
        return let_gev_per_um * GEVGRA

    def _compute_velocity(self) -> float:
        """
        Compute the particle's velocity as a fraction of the speed of light.
        
        :returns: Normalized particle velocity (unitless).
        :rtype: float
        
        :raises ValueError: If energy is not provided.
        """
        if self.energy is None:
            raise ValueError("Kinetic energy is required to compute velocity.")
        m0 = 931.5  # Rest mass in MeV/u
        return np.sqrt(1 - 1 / ((1 + self.energy / m0) ** 2))

    def _compute_core_radius(self) -> float:
        """
        Compute the core radius in micrometers.
    
        :returns: Core radius value.
        :rtype: float
    
        :raises ValueError: If core_radius_type is invalid.
        """
        if self.core_radius_type == 'constant':
            return 1.0e-2
        elif self.core_radius_type == 'energy-dependent':
            Rc0 = 11.6e-3
            Rc = Rc0 * self.velocity
            return max(Rc, 3.0e-4)
        else:
            raise ValueError("Unsupported core radius type.")

    def _compute_effective_atomic_number(self) -> float:
        """
        Compute the effective atomic number (Z_eff) using the Barkas expression.
    
        :returns: Effective atomic number.
        :rtype: float
    
        :raises ValueError: If atomic number is not provided.
        """
        if self.atomic_number is None:
            raise ValueError("Atomic number is required to compute effective atomic number.")
        Z = self.atomic_number
        return Z * (1 - np.exp(-125 * self.velocity * Z ** (-2/3)))

    def _compute_Kp(self) -> float:
        """
        Compute the auxiliary constant Kp used in the Kiefer-Chatterjee model.
        
        :returns: Kp value.
        :rtype: float
        """
        Zeff = self._compute_effective_atomic_number()
        return 1.25e-4 * (Zeff / self.velocity) ** 2

    def _compute_lambda0(self) -> float:
        """
        Compute the auxiliary constant λ₀ (lambda0), used in both models.
    
        :returns: Lambda0 value.
        :rtype: float
        """
        return 1 / (np.pi * self.rho)

    def _compute_penumbra_radius(self) -> float:
        """
        Compute the penumbra radius (Rp) of the particle track.
    
        :returns: Penumbra radius.
        :rtype: float
    
        :raises ValueError: If energy is not provided or model is invalid.
        """
        if self.energy is None:
            raise ValueError("Kinetic energy is required to compute penumbra radius.")
        delta = 1.7
        if self.model_name == 'Scholz-Kraft':
            gamma = 0.05
        elif self.model_name == 'Kiefer-Chatterjee':
            gamma = 0.0616
        else:
            raise ValueError("Invalid model name for penumbra radius computation.")
        return gamma * (self.energy ** delta)

    def _kiefer_chatterjee_dose(self, radius: np.ndarray) -> np.ndarray:
        """
        Compute the local dose using the Kiefer-Chatterjee model.
    
        :param radius: Radii in micrometers at which to evaluate the dose.
        :type radius: np.ndarray
    
        :returns: Dose values at each radius.
        :rtype: np.ndarray
    
        :raises ValueError: If core_radius_type is not 'energy-dependent'.
        """
        if self.core_radius_type != 'energy-dependent':
            raise ValueError("Kiefer-Chatterjee model requires energy-dependent core radius.")
        core_radius = self._compute_core_radius()
        Kp = self._compute_Kp()
        lambda0 = self._compute_lambda0()
        LET = self.let
        Rp = self.penumbra_radius

        dose = np.where(radius < core_radius,
                        (lambda0 / core_radius**2) * (LET - 2 * np.pi * self.rho * Kp * np.log(Rp / core_radius)),
                        Kp / radius**2)
        dose[radius >= Rp] = 0.0
        return dose

    def _scholz_kraft_dose(self, radius: np.ndarray) -> np.ndarray:
        """
        Compute the local dose using the Scholz-Kraft model.
    
        :param radius: Radii in micrometers at which to evaluate the dose.
        :type radius: np.ndarray
    
        :returns: Dose values at each radius.
        :rtype: np.ndarray
        """
        core_radius = self._compute_core_radius()
        lambda0 = self._compute_lambda0()
        LET = self.let
        Rp = self.penumbra_radius

        lambda_ = lambda0 if Rp <= core_radius else lambda0 / (1 + 2 * np.log(Rp / core_radius))
        dose = lambda_ * LET / radius**2
        mask = radius < core_radius
        dose[mask] *= (radius[mask]**2) / (core_radius**2)
        dose[radius >= Rp] = 0.0
        return dose

    def initial_local_dose(self, radius: Optional[Union[float, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the initial radial dose distribution.
    
        :param radius: Optional radius or array of radii in micrometers.
            If None, a default grid is generated from energy and penumbra radius.
        :type radius: float or np.ndarray, optional
    
        :returns: Tuple containing (dose, radius) arrays.
        :rtype: tuple[np.ndarray, np.ndarray]
    
        :raises ValueError: If radius values are invalid or energy is missing when required.
        """
        if radius is None:
            if self.energy is None:
                raise ValueError("Energy must be provided to generate default radii.")
            radius = GeometryTools.generate_default_radii(self.energy, self.penumbra_radius, base_points=self.base_points)
        radii = np.asarray(radius, dtype=float).reshape(-1, 1)
        if np.any(radii <= 0):
            raise ValueError("Radius values must be positive.")
        if self.model_name == 'Kiefer-Chatterjee':
            dose = self._kiefer_chatterjee_dose(radii)
        elif self.model_name == 'Scholz-Kraft':
            dose = self._scholz_kraft_dose(radii)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        return dose, radii
