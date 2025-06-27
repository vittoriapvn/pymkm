# This file marks this directory as a Python package
"""
Stopping power tables from FLUKA 2020.0.

This dataset contains precomputed LET tables for ions in liquid water
generated using the FLUKA Monte Carlo code (version 2020.0).

Each .txt file contains stopping power values for a specific ion, tabulated
as energy (MeV/u) vs. LET (MeV/cm), for water as the reference medium.

These values can be used to initialize:
- :class:`~pymkm.io.stopping_power.StoppingPowerTable`
- :class:`~pymkm.io.table_set.StoppingPowerTableSet`
"""
