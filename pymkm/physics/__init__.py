"""
Physics models and computational core for pyMKM.

This subpackage contains the physical and mathematical implementations required
to compute microdosimetric quantities for the Microdosimetric Kinetic Model (MKM)
and its stochastic extensions.

Modules
-------

- :mod:`particle_track`:
  Implements :class:`~pymkm.physics.particle_track.ParticleTrack`, a model for radial
  dose distributions around ion tracks using the Scholz-Kraft and Kiefer-Chatterjee formalisms.

- :mod:`specific_energy`:
  Provides the :class:`~pymkm.physics.specific_energy.SpecificEnergy` class to compute
  microdosimetric specific energy quantities, including single-event saturation-corrected
  values, and dose-averaged specific energy.
"""

from .particle_track import ParticleTrack
from .specific_energy import SpecificEnergy

__all__ = ["ParticleTrack", "SpecificEnergy"]
