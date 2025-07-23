"""
Management of multiple ion stopping power tables.

This module defines the :class:`StoppingPowerTableSet`, a high-level container
for managing multiple :class:`~pymkm.io.stopping_power.StoppingPowerTable` instances,
each representing LET vs. energy data for a different ion in liquid water.

Main features
-------------

- Add, remove, or retrieve tables by ion name, symbol, or atomic number
- Batch interpolation, resampling, filtering, and plotting
- JSON and directory-based I/O
- Support for internal default sources: ``mstar_3_12``, ``geant4_11_3_0``, ``fluka_2020_0``

Used throughout pyMKM to provide LET data to physical and biological models.

Examples
--------

>>> from pymkm.io.table_set import StoppingPowerTableSet
>>> s = StoppingPowerTableSet.from_default_source("mstar_3_12")
>>> sorted(s.get_available_ions())
['Beryllium', 'Boron', 'Carbon', 'Fluorine', 'Helium', 'Hydrogen',
 'Lithium', 'Neon', 'Nitrogen', 'Oxygen']
>>> s.get("Carbon").plot(show=False)
"""

import json
import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from pymkm.io.stopping_power import StoppingPowerTable
from pymkm.io.data_registry import list_available_defaults, get_default_txt_path


class StoppingPowerTableSet:
    """
    A container for managing multiple StoppingPowerTable instances.

    Provides unified access to stopping power curves for multiple ions, identified
    by name, symbol, or atomic number. Supports serialization, resampling, filtering,
    and plotting.
    """

    def __init__(self):
        """
        Initialize an empty StoppingPowerTableSet.
        """
        self.tables: Dict[str, StoppingPowerTable] = {}
        self.source_info: Optional[str] = None  # Track origin

    def add(self, ion_input: str, table: StoppingPowerTable):
        """
        Add a stopping power table to the set.
        
        :param ion_input: Ion identifier (name, symbol, or atomic number).
        :type ion_input: str
        :param table: Instance of :class:`~pymkm.io.stopping_power.StoppingPowerTable` to add.
        :type table: pymkm.io.stopping_power.StoppingPowerTable
        """
        key = self._map_to_fullname(ion_input)
        self.tables[key] = table

    def remove(self, ion_input: str):
        """
        Remove a table by ion identifier.
        
        :param ion_input: Ion name, symbol, or atomic number.
        :type ion_input: str
        """
        key = self._map_to_fullname(ion_input)
        self.tables.pop(key, None)

    def get(self, ion_input: str) -> Optional[StoppingPowerTable]:
        """
        Retrieve a table by ion identifier.
    
        :param ion_input: Ion name, symbol, or atomic number.
        :type ion_input: str
    
        :returns: Corresponding :class:`~pymkm.io.stopping_power.StoppingPowerTable` or None.
        :rtype: pymkm.io.stopping_power.StoppingPowerTable or None
        """
        key = self._map_to_fullname(ion_input)
        return self.tables.get(key)

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, ion_input: str) -> StoppingPowerTable:
        table = self.get(ion_input)
        if table is None:
            raise KeyError(f"No table found for ion: {ion_input}")
        return table

    def __contains__(self, ion_input: str) -> bool:
        return self.get(ion_input) is not None

    def __iter__(self):
        return iter(self.tables.items())

    def keys(self):
        return self.tables.keys()

    def values(self):
        return self.tables.values()

    def items(self):
        return self.tables.items()

    def to_dict(self) -> Dict[str, dict]:
        """
        Serialize all tables to a dictionary.
        
        :returns: Dictionary of ion names to serialized data.
        :rtype: dict[str, dict]
        """
        return {k: v.to_dict() for k, v in self.tables.items()}

    def to_json(self) -> str:
        """
        Serialize the set to a JSON string.
        
        :returns: JSON-formatted string.
        :rtype: str
        """
        return json.dumps(self.to_dict(), indent=2)

    def save(self, filepath: str):
        """
        Save the table set to a JSON file.
        
        :param filepath: Output file path.
        :type filepath: str
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, dict]) -> "StoppingPowerTableSet":
        """
        Create a table set from a dictionary.
        
        :param data: Dictionary mapping ion names to serialized tables.
        :type data: dict[str, dict]
        
        :returns: StoppingPowerTableSet instance.
        :rtype: StoppingPowerTableSet
        
        :raises ValueError: If tables have inconsistent source program or ionization potential.
        """
        instance = cls()
        reference_program = None
        reference_ipot = None

        for name, table_data in data.items():
            table = StoppingPowerTable.from_dict(table_data)
            if reference_program is None:
                reference_program = table.source_program
                reference_ipot = table.ionization_potential
            else:
                if table.source_program != reference_program:
                    raise ValueError(f"Inconsistent source_program in table '{name}': expected '{reference_program}', got '{table.source_program}'")
                if table.ionization_potential != reference_ipot:
                    raise ValueError(f"Inconsistent ionization_potential in table '{name}': expected {reference_ipot}, got {table.ionization_potential}")
            instance.add(name, table)
        instance.source_info = "dict"
        return instance

    @classmethod
    def from_json(cls, json_str: str) -> "StoppingPowerTableSet":
        """
        Create a table set from a JSON string.
    
        :param json_str: JSON-formatted table set string.
        :type json_str: str
    
        :returns: Deserialized StoppingPowerTableSet.
        :rtype: StoppingPowerTableSet
        """        
        data = json.loads(json_str)
        instance = cls.from_dict(data)
        instance.source_info = "json"
        return instance

    @classmethod
    def load(cls, filepath: str) -> "StoppingPowerTableSet":
        """
        Load a table set from a JSON file.
        
        :param filepath: Path to JSON file.
        :type filepath: str
        
        :returns: Loaded StoppingPowerTableSet.
        :rtype: StoppingPowerTableSet
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        instance = cls.from_dict(data)
        instance.source_info = f"loaded:{filepath}"
        return instance

    @classmethod
    def from_directory(cls, directory: str) -> "StoppingPowerTableSet":
        """
        Load all .txt stopping power tables from a directory.
    
        :param directory: Path to directory containing .txt files.
        :type directory: str
    
        :returns: StoppingPowerTableSet with all successfully loaded tables.
        :rtype: StoppingPowerTableSet
        """
        instance = cls()
        reference_program = None
        reference_ipot = None

        for fname in os.listdir(directory):
            if fname.endswith(".txt"):
                path = os.path.join(directory, fname)
                try:
                    table = StoppingPowerTable.from_txt(path)
                    if reference_program is None:
                        reference_program = table.source_program
                        reference_ipot = table.ionization_potential
                    else:
                        if table.source_program != reference_program:
                            raise ValueError(f"Inconsistent source_program in file '{fname}': expected '{reference_program}', got '{table.source_program}'")
                        if table.ionization_potential != reference_ipot:
                            raise ValueError(f"Inconsistent ionization_potential in file '{fname}': expected {reference_ipot}, got {table.ionization_potential}")

                    key = cls._map_to_fullname(table.ion_name)
                    instance.add(key, table)
                except Exception as e:
                    print(f"Warning: Failed to load {fname}: {e}")
        instance.source_info = f"directory:{directory}"
        return instance

    @classmethod
    def from_default_source(cls, source: str) -> "StoppingPowerTableSet":
        """
        Load tables from a default internal source (e.g., "mstar_3_12").
        
        :param source: Name of the predefined source directory.
        :type source: str
        
        :returns: Table set initialized from source.
        :rtype: StoppingPowerTableSet
        
        :raises RuntimeError: If any file cannot be loaded.
        """
        instance = cls()
        for filename in list_available_defaults(source):
            try:
                path = get_default_txt_path(source, filename)
                table = StoppingPowerTable.from_txt(path)
                key = cls._map_to_fullname(table.ion_name)
                instance.add(key, table)
            except Exception as e:
                raise RuntimeError(f"Failed to load {filename} from {source}: {e}") from e
        instance.source_info = f"default:{source}"
        return instance

    @staticmethod
    def _map_to_fullname(ion_input: str) -> str:
        """
        Convert ion identifier to full element name.
    
        Supports input as name, symbol, or atomic number.
    
        :param ion_input: Ion identifier.
        :type ion_input: str
    
        :returns: Full name of the ion.
        :rtype: str
        """
        lookup = StoppingPowerTable.get_lookup_table()
    
        # Already a full name
        if ion_input in lookup:
            return ion_input
    
        # Reverse mapping: symbol -> full name
        reverse = {v["symbol"]: k for k, v in lookup.items()}
        if ion_input in reverse:
            return reverse[ion_input]
    
        # Try atomic number
        try:
            z = int(ion_input)
            for name, v in lookup.items():
                if v["atomic_number"] == z:
                    return name
        except ValueError:
            pass  # fallback below if not an integer
    
        # Fallback: return input as-is (may fail in get())
        return ion_input

    def get_available_ions(self) -> List[str]:
        """
        Get the list of ion names present in the set.
        
        :returns: List of ion names.
        :rtype: list[str]
        """
        return list(self.tables.keys())

    def filter_by_ions(self, ion_inputs: List[str]) -> "StoppingPowerTableSet":
        """
        Create a subset containing only the specified ions.
        Each table is a :class:`~pymkm.io.stopping_power.StoppingPowerTable` instance.
        
        :param ion_inputs: List of identifiers (names, symbols, or atomic numbers).
        :type ion_inputs: list[str]
        
        :returns: New StoppingPowerTableSet with selected ions.
        :rtype: StoppingPowerTableSet
        """
        subset = StoppingPowerTableSet()
        for ion in ion_inputs:
            key = self._map_to_fullname(ion)
            if key in self.tables:
                subset.add(key, self.tables[key])
        subset.source_info = self.source_info
        return subset

    def get_energy_grid(self, ion_input: str) -> np.ndarray:
        """
        Get the energy grid for a given ion.
    
        :param ion_input: Ion identifier.
        :type ion_input: str
    
        :returns: Energy array.
        :rtype: np.ndarray
        """
        return self.get(ion_input).energy_grid

    def get_stopping_power(self, ion_input: str) -> np.ndarray:
        """
        Get the stopping power values for a given ion.
    
        :param ion_input: Ion identifier.
        :type ion_input: str
    
        :returns: LET array.
        :rtype: np.ndarray
        """
        return self.get(ion_input).stopping_power

    def get_common_energy_range(self) -> Optional[List[float]]:
        """
        Get the overlapping energy range across all tables.
        
        :returns: [min, max] energy range, or None if no common range exists.
        :rtype: list[float] or None
        """
        if not self.tables:
            return None
        mins = [np.min(t.energy_grid) for t in self.tables.values()]
        maxs = [np.max(t.energy_grid) for t in self.tables.values()]
        common_min = max(mins)
        common_max = min(maxs)
        return [common_min, common_max] if common_max > common_min else None

    def resample_all(self, new_grid: np.ndarray):
        """
        Resample the LET curves of all tables onto a new energy grid.
        
        :param new_grid: Strictly increasing energy grid in MeV/u.
        :type new_grid: np.ndarray
        
        :raises ValueError: If the grid is not strictly increasing.
        """
        for t in self.tables.values():
            t.resample(new_grid)
    
    def interpolate_all(self, energy: np.ndarray, loglog: bool = True) -> Dict[str, np.ndarray]:
        """
        Interpolate LET values at given energies for all tables.
    
        :param energy: Energy values at which to interpolate.
        :type energy: np.ndarray
        :param loglog: Use log-log interpolation if True.
        :type loglog: bool
    
        :returns: Dictionary mapping ion names to interpolated LET arrays.
        :rtype: dict[str, np.ndarray]
        """
        return {
            ion: table.interpolate(energy=energy, loglog=loglog)
            for ion, table in self.tables.items()
        }

    def plot(self,
             ions: Optional[List[str]] = None,
             show: bool = True,
             ax: Optional[plt.Axes] = None,
             single_plot: bool = True
        ):
        """
        Plot stopping power curves for one or more ions.
    
        :param ions: List of ion identifiers to plot. If None, all are plotted.
        :type ions: Optional[List[str]]
        :param show: Whether to call plt.show().
        :type show: bool
        :param ax: Matplotlib Axes object to draw on. If None, a new figure is created.
        :type ax: Optional[matplotlib.axes.Axes]
        :param single_plot: If True, plot all ions on one figure; otherwise, one figure per ion.
        :type single_plot: bool
        """
        ions_to_plot = ions if ions is not None else list(self.tables.keys())    

        if single_plot:
            
            # Create figure/axes if not provided
            created_fig = False
            if ax is None:
                _, ax = plt.subplots()
                created_fig = True

            for ion in ions_to_plot:
                table = self.get(ion)
                if table is not None:
                    table.plot(label=ion, show=False, ax=ax)

            ax.set_xlabel("Energy [MeV/u]")
            ax.set_ylabel("Stopping Power [MeV/cm]")
            ax.set_xscale("log")
            ax.legend()
            ax.grid(True)
            if show and created_fig:
                plt.show()
        else:
            for ion in ions_to_plot:
                table = self.get(ion)
                if table is not None:
                    table.plot(label=ion, show=show, new_figure=True)
