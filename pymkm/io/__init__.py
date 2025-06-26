"""
I/O submodule for pyMKM.

This package provides functionality to load, manage, and serialize stopping power (LET) data
used by the Microdosimetric Kinetic Model. It handles both individual ion tables and structured
collections of precomputed datasets.

Modules
-------

- :mod:`data_registry`: 
  Functions for discovering and loading default `.txt` stopping power tables and the 
  periodic table lookup (`elements.json`). See functions like 
  :func:`~pymkm.io.data_registry.get_default_txt_path` and 
  :func:`~pymkm.io.data_registry.load_lookup_table`.

- :mod:`stopping_power`: 
  Defines the :class:`~pymkm.io.stopping_power.StoppingPowerTable` for parsing,
  validating, interpolating and plotting LET curves for a single ion.

- :mod:`table_set`: 
  Provides :class:`~pymkm.io.table_set.StoppingPowerTableSet`, a high-level container 
  for multiple ion tables with utilities for batch resampling, filtering, and plotting.
"""

from .stopping_power import StoppingPowerTable
from .table_set import StoppingPowerTableSet

__all__ = ["StoppingPowerTable", "StoppingPowerTableSet"]
