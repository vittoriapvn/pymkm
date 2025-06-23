import numpy as np
import logging
from typing import Optional, Union, Tuple

from pymkm.utils.geometry_tools import GeometryTools

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class ParticleTrack:
    """
    ParticleTrack models the local dose distribution along a particle track
    using either the 'Scholz-Kraft' or 'Kiefer-Chatterjee' model.
    """

    def __init__(self,
                 model_name: str = 'Kiefer-Chatterjee',
                 core_radius_type: str = 'energy-dependent',
                 energy: Optional[float] = None,
                 atomic_number: Optional[int] = None,
                 let: Optional[float] = None,
                 base_points: int = GeometryTools.generate_default_radii.__defaults__[1]) -> None:
        """
        Initialize the ParticleTrack instance.

        Parameters:
          model_name (str): The model to use for dose calculation ('Scholz-Kraft' or 'Kiefer-Chatterjee').
          core_radius_type (str): 'constant' or 'energy-dependent' (default is 'energy-dependent').
          energy (Optional[float]): Kinetic energy in MeV/u. Required for energy-dependent calculations.
          atomic_number (Optional[int]): Atomic number (Z) of the particle. Required for Kiefer-Chatterjee.
          let (Optional[float]): Unrestricted LET in MeV/cm.
          base_points (int): Base number of sampling points (N0) for generating default radii (default is 150).

        Raises:
          ValueError: if required parameters are missing or invalid.
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
        Convert LET from MeV/cm to g*Gy/μm.
        """
        GEVGRA = 1.602176462e-7
        let_gev_per_um = let_mev_per_cm * 1e-7
        return let_gev_per_um * GEVGRA

    def _compute_velocity(self) -> float:
        """
        Compute the particle's velocity (fraction of the speed of light) based on kinetic energy.
        """
        if self.energy is None:
            raise ValueError("Kinetic energy is required to compute velocity.")
        m0 = 931.5  # Rest mass in MeV/u
        return np.sqrt(1 - 1 / ((1 + self.energy / m0) ** 2))

    def _compute_core_radius(self) -> float:
        """
        Compute the core radius in micrometers.
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
        """
        if self.atomic_number is None:
            raise ValueError("Atomic number is required to compute effective atomic number.")
        Z = self.atomic_number
        return Z * (1 - np.exp(-125 * self.velocity * Z ** (-2/3)))

    def _compute_Kp(self) -> float:
        """
        Compute the auxiliary variable Kp.
        """
        Zeff = self._compute_effective_atomic_number()
        return 1.25e-4 * (Zeff / self.velocity) ** 2

    def _compute_lambda0(self) -> float:
        """
        Compute the auxiliary variable lambda0.
        """
        return 1 / (np.pi * self.rho)

    def _compute_penumbra_radius(self) -> float:
        """
        Compute the penumbra radius (Rp) based on the model and energy.
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

        Parameters:
          radius (np.ndarray): Radii (in μm) at which to compute the dose.

        Returns:
          np.ndarray: Dose values corresponding to the provided radii.
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

        Parameters:
          radius (np.ndarray): Radii (in μm) at which to compute the dose.

        Returns:
          np.ndarray: Dose values corresponding to the provided radii.
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
        Compute the initial local dose as a function of radial distance.

        Parameters:
          radius (Optional[Union[float, np.ndarray]]): Radius (or array of radii) in micrometers.
              If not provided, a default logarithmic grid is generated using GeometryTools.generate_default_radii,
              using the user-controlled base_points value.

        Returns:
          Tuple[np.ndarray, np.ndarray]: A tuple (dose, radii), where 'dose' is the computed dose values (Gy)
                                         and 'radii' is a column vector of radii (μm).
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
