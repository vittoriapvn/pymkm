# This file marks this directory as a Python package
"""
Stopping power tables from FLUKA 2020.0.

This dataset contains precomputed LET tables for ions in liquid water
generated using the FLUKA Monte Carlo code (version 2020.0).

Each file corresponds to a single ion (e.g., Hydrogen, Carbon, Oxygen),
formatted as a `.txt` table with energy vs. stopping power values.

These data can be loaded using:
- :func:`~pymkm.io.data_registry.get_default_txt_path`
- :class:`~pymkm.io.table_set.StoppingPowerTableSet.from_default_source`
"""
