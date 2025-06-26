# This file marks this directory as a Python package
"""
Biological effect models and oxygen correction tools.

This subpackage contains functions and models used to compute
biologically relevant quantities such as relative radioresistance
under hypoxic conditions (OSMK 2021/2023) and corrections to
linear–quadratic model parameters (α, β) in the presence of oxygen.

Modules
-------

- :mod:`oxygen_effect`:
  Provides utilities for computing relative radioresistance (R), 
  OSMK 2023 scaling factors (f_rd, f_z0), and oxygen-corrected α and β.

These functions are typically used during post-processing of MKTable results
when the oxygen effect correction is enabled.
"""

