import json
import os
from typing import Dict, List, Optional, Union
import numpy as np

from pymkm.io.stopping_power import StoppingPowerTable
from pymkm.io.data_registry import list_available_defaults, get_default_txt_path


class StoppingPowerTableSet:
    """
    A container for multiple StoppingPowerTable instances.

    Each table is keyed by the ion's full name (e.g. "Carbon", "Oxygen"),
    and can be accessed via various identifier formats. All public methods
    accept the following formats to reference a given ion:
    
    - Full name (e.g. "Carbon")
    - Symbol (e.g. "C")
    - Atomic number (integer or string, e.g. 6 or "6")
    """

    def __init__(self):
        self.tables: Dict[str, StoppingPowerTable] = {}
        self.source_info: Optional[str] = None  # Track origin

    def add(self, ion_input: str, table: StoppingPowerTable):
        """
        Add a table to the set. The ion identifier can be a full name (e.g., "Carbon"),
        a symbol (e.g., "C"), or an atomic number (e.g., 6 or "6"). It will be normalized to the full name.
        """
        key = self._map_to_fullname(ion_input)
        self.tables[key] = table

    def remove(self, ion_input: str):
        """
        Remove a table from the set. The ion identifier can be a full name, symbol,
        or atomic number. It will be normalized to the full name key.
        """
        key = self._map_to_fullname(ion_input)
        self.tables.pop(key, None)

    def get(self, ion_input: str) -> Optional[StoppingPowerTable]:
        """
        Remove a table from the set. The ion identifier can be a full name, symbol,
        or atomic number. It will be normalized to the full name key.
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
        return {k: v.to_dict() for k, v in self.tables.items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, dict]) -> "StoppingPowerTableSet":
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
        data = json.loads(json_str)
        instance = cls.from_dict(data)
        instance.source_info = "json"
        return instance

    @classmethod
    def load(cls, filepath: str) -> "StoppingPowerTableSet":
        with open(filepath, 'r') as f:
            data = json.load(f)
        instance = cls.from_dict(data)
        instance.source_info = f"loaded:{filepath}"
        return instance

    @classmethod
    def from_directory(cls, directory: str) -> "StoppingPowerTableSet":
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
        Infer the full ion name from the provided identifier.
        Supports:
        - Full names (e.g. "Carbon")
        - Symbols (e.g. "C")
        - Atomic numbers (e.g. 6 or "6")
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
        return list(self.tables.keys())

    def filter_by_ions(self, ion_inputs: List[str]) -> "StoppingPowerTableSet":
        """
        Return a new table set containing only the tables matching the provided ion identifiers.
        Each identifier can be a full name, symbol, or atomic number; all are normalized to full names.
        """
        subset = StoppingPowerTableSet()
        for ion in ion_inputs:
            key = self._map_to_fullname(ion)
            if key in self.tables:
                subset.add(key, self.tables[key])
        subset.source_info = self.source_info
        return subset

    def get_energy_grid(self, ion_input: str) -> np.ndarray:
        return self.get(ion_input).energy_grid

    def get_stopping_power(self, ion_input: str) -> np.ndarray:
        return self.get(ion_input).stopping_power

    def get_common_energy_range(self) -> Optional[List[float]]:
        if not self.tables:
            return None
        mins = [np.min(t.energy_grid) for t in self.tables.values()]
        maxs = [np.max(t.energy_grid) for t in self.tables.values()]
        common_min = max(mins)
        common_max = min(maxs)
        return [common_min, common_max] if common_max > common_min else None

    def resample_all(self, new_grid: np.ndarray):
        """
        Resample the LET values of all tables in the set to a new common energy grid.
        
        Parameters
        ----------
        new_grid : np.ndarray
            A strictly increasing array of energy values [MeV/u] on which to resample
            the stopping power curves of all ions in the set.
        
        Notes
        -----
        This modifies the internal `energy` and `let` arrays of each
        `StoppingPowerTable` instance by interpolating in log-log space.
        """
        for t in self.tables.values():
            t.resample(new_grid)
    
    def interpolate_all(self, energy: np.ndarray, loglog: bool = True) -> Dict[str, np.ndarray]:
        """
        Perform forward interpolation (energy → LET) for all ions.
    
        Parameters:
          energy: Energy values (array-like) to interpolate LET.
          loglog: Use log-log interpolation if True.
    
        Returns:
          Dict of interpolated LET arrays for each ion.
        """
        return {
            ion: table.interpolate(energy=energy, loglog=loglog)
            for ion, table in self.tables.items()
        }

    def plot(self, ions: Optional[List[str]] = None, show: bool = True, single_plot: bool = True):
        """
        Plot the stopping power curves.
        If single_plot is True, all selected ions are plotted on one figure.
        Otherwise, a new figure is created for each ion.
        The ion identifier provided in the plot call can be either full name or symbol.
        """
        import matplotlib.pyplot as plt
        ions_to_plot = ions if ions is not None else list(self.tables.keys())
        if single_plot:
            plt.figure(figsize=(8, 5))
            for ion in ions_to_plot:
                table = self.get(ion)
                if table is not None:
                    table.plot(label=ion, show=False, new_figure=False)
            plt.xlabel("Energy [MeV/u]")
            plt.ylabel("Stopping Power [MeV/cm]")
            plt.xscale("log")
            plt.legend()
            plt.grid(True)
            if show:
                plt.show()
        else:
            for ion in ions_to_plot:
                table = self.get(ion)
                if table is not None:
                    table.plot(label=ion, show=show, new_figure=True)
