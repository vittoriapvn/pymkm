"""
Survival fraction table generation for MKM, SMK, and OSMK models.

This subpackage provides tools for computing and visualizing survival fraction (SF)
curves based on microdosimetric quantities from an MKTable.

After running `compute()`, the results are stored in `self.table` as a list of result dictionaries:

.. code-block:: python

    SFTable.table = [
        {
            "params": {
                "ion": "12C",
                "energy": 160.0,
                "let": 28.3,
                "model": "stochastic",
                "osmk_version": "2023"
            },
            "calculation_info": "computed",
            "data": pd.DataFrame({
                "dose": [...],                 # [Gy]
                "survival_fraction": [...]     # dimensionless
            })
        },
        ...
    ]

Models supported
----------------

- **MKM (classic)**: Linear-quadratic model with saturation correction.
- **SMK (stochastic)**: Extension of MKM using dose-averaged microdosimetric inputs.
- **OSMK (oxygen-aware SMK)**: Includes hypoxia corrections using 2021 or 2023 formulations
   by Inaniwa et al., (Phys. Med. Biol., 2021) and Inaniwa & Kanematsu, (JRR, 2023).

Modules
-------

- :mod:`core`:
  Defines :class:`~pymkm.sftable.core.SFTableParameters` and :class:`~pymkm.sftable.core.SFTable`
  for configuring and managing SF table generation.

- :mod:`compute`:
  Provides :meth:`~pymkm.sftable.compute.compute`, the main routine for calculating survival curves
  from LET and energy data.

- :mod:`plot`:
  Contains plotting utilities for visualizing survival curves across doses, energies, or ions.

Usage
-----

The `sftable` module is typically used after computing an MKTable:

.. code-block:: python

    from pymkm.sftable.core import SFTableParameters, SFTable

    params = SFTableParameters(mktable=mktable, alpha0=0.1, beta0=0.05)
    sftable = SFTable(params)
    sftable.compute(ion="C")
    sftable.plot()
"""

from .core import SFTable, SFTableParameters
from . import compute  # noqa
from . import plot  # noqa

__all__ = ["SFTable", "SFTableParameters"]
