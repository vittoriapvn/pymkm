"""
Tools for generating microdosimetric model tables (MKTable).

This subpackage provides all components needed to generate, manage,
analyze and export microdosimetric tables used in the MKM and SMK
radiobiological models, including the oxygen-enhanced OSMK 2023 variant.

The core workflow includes configuration of physical and numerical parameters,
computation of dose-averaged specific energies, optional overkill corrections,
and export to interoperable formats.

Modules
-------

- :mod:`core`:
  Defines :class:`~pymkm.mktable.core.MKTableParameters` and :class:`~pymkm.mktable.core.MKTable`,
  which manage model configuration, geometry, stopping power sets, and computed data.

- :mod:`compute`:
  Implements :meth:`~pymkm.mktable.core.MKTable.compute`, the main engine to compute 
  microdosimetric quantities (z̄*, z̄_d, z̄_n) from input LET tables.

- :mod:`plot`:
  Provides :meth:`~pymkm.mktable.core.MKTable.plot` for visualizing results for multiple ions,
  with energy or LET on the x-axis.

Usage
-----

This subpackage is the main user-facing interface of pyMKM.
After configuring a :class:`~pymkm.mktable.core.MKTableParameters` object, you can:

.. code-block:: python

    mktable = MKTable(parameters)
    mktable.compute()
    mktable.plot()
    mktable.save()
    mktable.write_txt(...)

"""

from .core import MKTable, MKTableParameters
from . import compute  # noqa
from . import plot  # noqa

__all__ = ["MKTable", "MKTableParameters"]

