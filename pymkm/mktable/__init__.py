"""
Tools for generating microdosimetric model tables (MKTable).

This subpackage provides all components to compute, analyze, and export
dose-averaged specific energy quantities used in the Microdosimetric Kinetic Model (MKM),
its stochastic extension (SMK), and the oxygen-enhanced variant as described by Inaniwa & Kanematsu, (JRR, 2023).

The typical workflow includes:
- configuration of model and geometry parameters
- per-ion table computation from LET data
- saturation correction and optional hypoxia adjustment
- interactive visualization and export to clinical-compatible formats

Each computed result is stored in the `self.table` dictionary with the structure:

.. code-block:: python

    MKTable.table = {
        "C": {
            "params": {...},                   # Input model parameters
            "stopping_power_info": {...},     # Metadata (LET, Z, A, source...)
            "data": pd.DataFrame([            # Main output table
                {
                    "energy": ...,            # [MeV/u]
                    "let": ...,               # [MeV/cm]
                    "z_bar_star_domain": ..., # [Gy]
                    "z_bar_domain": ...,      # [Gy] (SMK only)
                    "z_bar_nucleus": ...      # [Gy] (SMK only)
                },
                ...
            ])
        },
        ...
    }

Modules
-------

- :mod:`core`:
  Defines :class:`~pymkm.mktable.core.MKTableParameters` and :class:`~pymkm.mktable.core.MKTable`,
  which manage configuration, geometry, stopping power sets, and result storage.

- :mod:`compute`:
  Implements :meth:`~pymkm.mktable.core.MKTable.compute`, the numerical engine that builds
  microdosimetric tables from energy–LET inputs.

- :mod:`plot`:
  Adds :meth:`~pymkm.mktable.core.MKTable.plot`, enabling multi-ion plots of 
  dose-averaged specific energy quantities vs. energy or LET.

Usage
-----

This subpackage is the main high-level interface to pyMKM.

.. code-block:: python

    from pymkm.mktable import MKTable, MKTableParameters

    params = MKTableParameters(domain_radius=0.5, nucleus_radius=3.0, beta0=0.05)
    mktable = MKTable(params)
    mktable.compute()
    mktable.plot()
    mktable.save()
    mktable.write_txt(...)

"""

from .core import MKTable, MKTableParameters
from . import compute  # noqa
from . import plot  # noqa

__all__ = ["MKTable", "MKTableParameters"]

