import numpy as np
from typing import Optional

class GeometryTools:
    """
    Collection of geometric helper methods for particle interaction modeling.
    """

    @staticmethod
    def determine_sampling_points(energy: float, radius_max: float, base_points: int = 150) -> int:
        """
        Determine the number of sampling points (N) for dose calculation 
        based on the particle's energy and a specified maximum radius.

        Parameters:
          energy (float): The particle's kinetic energy in MeV/u.
          radius_max (float): The maximum radius in micrometers.
          base_points (int): The base number of sampling points (default is 150).

        Returns:
          int: The computed number of sampling points.
        """
        N0 = base_points

        if energy <= 10 or radius_max <= 0.05:
            multiplier = 20
        elif energy <= 20 or radius_max <= 0.1:
            multiplier = 15
        elif energy <= 40 or radius_max <= 0.5:
            multiplier = 12
        elif energy <= 60 or radius_max <= 1:
            multiplier = 8
        elif energy <= 80 or radius_max <= 2:
            multiplier = 6
        elif energy <= 100 or radius_max <= 5:
            multiplier = 5
        elif energy <= 150 or radius_max <= 10:
            multiplier = 4
        elif energy <= 200 or radius_max <= 20:
            multiplier = 3
        else:
            multiplier = 2

        return N0 * multiplier

    @staticmethod
    def generate_default_radii(energy: float, radius_max: float, radius_min: Optional[float] = 1e-3, base_points: int = 150) -> np.ndarray:
        """
        Generate a default logarithmic scale of radii based on the particle's energy 
        and maximum radius.

        Parameters:
          energy (float): The particle's kinetic energy in MeV/u.
          radius_max (float): The maximum radius in micrometers.
          radius_min (Optional[float]): The minimum radius in micrometers (default is 1e-3).
          base_points (int): The base number of sampling points (default is 150).

        Returns:
          np.ndarray: An array of radii values computed on a logarithmic scale.
        """
        if energy is None or radius_max is None:
            raise ValueError("Both energy and maximum radius must be provided.")

        N = GeometryTools.determine_sampling_points(energy, radius_max, base_points)
        return np.logspace(np.log10(radius_min), np.ceil(np.log10(radius_max)), N)

    @staticmethod
    def calculate_intersection_area(r1: np.ndarray, r2: float, d: float) -> np.ndarray:
        """
        Calculate the area of intersection between circles where one set of circles has varying radii
        and the other set has a constant radius with a fixed center-to-center distance.

        Parameters:
          r1 (np.ndarray): Array of radii for the first set of circles.
          r2 (float): Radius of the second set of circles (constant).
          d (float): Distance between the centers of the circles.

        Returns:
          np.ndarray: Array of intersection areas for each value in r1.
        """
        r1 = np.asarray(r1).flatten()
        intersection_area = np.zeros_like(r1)

        non_overlap_mask = d >= (r1 + r2)
        intersection_area[non_overlap_mask] = 0.0

        contained_mask = d <= np.abs(r1 - r2)
        intersection_area[contained_mask] = np.pi * np.minimum(r1[contained_mask], r2)**2

        overlap_mask = ~non_overlap_mask & ~contained_mask
        if np.any(overlap_mask):
            r1_overlap = r1[overlap_mask]
            r1_sq = r1_overlap**2
            r2_sq = r2**2
            d_overlap = d

            arg1 = (d_overlap**2 + r1_sq - r2_sq) / (2 * d_overlap * r1_overlap)
            arg2 = (d_overlap**2 + r2_sq - r1_sq) / (2 * d_overlap * r2)
            arg1 = np.clip(arg1, -1.0, 1.0)
            arg2 = np.clip(arg2, -1.0, 1.0)

            part1 = r1_sq * np.arccos(arg1)
            part2 = r2_sq * np.arccos(arg2)
            part3 = 0.5 * np.sqrt(
                (-d_overlap + r1_overlap + r2) *
                (d_overlap + r1_overlap - r2) *
                (d_overlap - r1_overlap + r2) *
                (d_overlap + r1_overlap + r2)
            )

            intersection_area[overlap_mask] = part1 + part2 - part3

        return intersection_area
