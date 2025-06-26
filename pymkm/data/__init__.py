# This file marks this directory as a Python package
"""
Data resources for pyMKM.

This subpackage provides structured data used in microdosimetric calculations,
including periodic element properties and default stopping power tables.

Contents
--------

- ``elements.json``:
  Lookup table mapping element names and symbols to atomic number, mass number, and visualization color.
  Used for consistency and validation when initializing LET tables.

- ``defaults/``:
  Contains precomputed stopping power tables (.txt files) for various Monte Carlo codes
  and versions, organized by subfolder. These files are used to initialize
  :class:`~pymkm.io.stopping_power.StoppingPowerTable` instances via 
  :func:`~pymkm.io.data_registry.get_default_txt_path` or
  :class:`~pymkm.io.table_set.StoppingPowerTableSet.from_default_source`.

Usage
-----

These resources are accessed internally using :mod:`importlib.resources`,
so that they can be bundled as package data and accessed seamlessly whether installed or used locally.
"""
