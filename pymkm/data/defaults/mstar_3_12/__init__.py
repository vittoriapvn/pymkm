# This file marks this directory as a Python package
"""
Stopping power tables from MSTAR 3.12.

This dataset contains precomputed LET tables for ions in liquid water
generated using the MSTAR code (version 3.12, https://nds.iaea.org/stopping-legacy/MstarWWW/MSTARInstr.htmlmai).

Each .txt file contains stopping power values for a specific ion, tabulated
as energy (MeV/u) vs. LET (MeV/cm), for water as the reference medium.

These values can be used to initialize:
- :class:`~pymkm.io.stopping_power.StoppingPowerTable`
- :class:`~pymkm.io.table_set.StoppingPowerTableSet`
"""
