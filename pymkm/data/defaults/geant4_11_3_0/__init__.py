# This file marks this directory as a Python package
"""
Stopping power tables from Geant4 11.3.0.

This dataset contains precomputed LET tables for ions in liquid water
generated using the Geant4 toolkit (version 11.3.0).

Each .txt file contains stopping power values for a specific ion, tabulated
as energy (MeV/u) vs. LET (MeV/cm), for water as the reference medium.

These values can be used to initialize:
- :class:`~pymkm.io.stopping_power.StoppingPowerTable`
- :class:`~pymkm.io.table_set.StoppingPowerTableSet`
"""
