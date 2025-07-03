# This file marks this directory as a Python package
"""
Default stopping power tables organized by source.

Each subfolder (e.g., `mstar_3_12`, `geant4_11_3_0`, `fluka_2020_0`) contains
precomputed LET tables in `.txt` format for common ions (H, He, C, O, etc.).

These are loaded dynamically using :mod:`importlib.resources` and used for testing,
validation, or default initialization.
"""

