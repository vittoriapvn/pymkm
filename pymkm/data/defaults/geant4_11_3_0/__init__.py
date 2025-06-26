# This file marks this directory as a Python package
"""
Stopping power tables from Geant4 11.3.0.

This dataset contains LET tables computed with the Geant4 toolkit (version 11.3.0)
using water as the target medium.

Each `.txt` file corresponds to one ion species and can be accessed using pyMKM's data utilities.

These values can be used to initialize:
- :class:`~pymkm.io.stopping_power.StoppingPowerTable`
- :class:`~pymkm.io.table_set.StoppingPowerTableSet`
"""
