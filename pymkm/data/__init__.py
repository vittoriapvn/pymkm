# This file marks this directory as a Python package
"""
Data resources for pyMKM.

This subpackage provides structured data required for microdosimetric modeling,
including periodic element definitions and reference stopping power tables
from external Monte Carlo codes.

Contents
--------

- ``elements.json``:
  Lookup table mapping element symbols and names to atomic number, mass number,
  and display color. Used for validating and initializing LET table inputs.
  Elemental data are based on values published by the IUPAC:
  https://ciaaw.org/atomic-weights.htm

- ``defaults/``:
  Contains precomputed stopping power tables (in .txt format) for various Monte Carlo
  transport codes and versions. These files can be used to initialize
  :class:`~pymkm.io.stopping_power.StoppingPowerTable` via
  :func:`~pymkm.io.data_registry.get_default_txt_path` or
  :meth:`~pymkm.io.table_set.StoppingPowerTableSet.from_default_source`.
"""
