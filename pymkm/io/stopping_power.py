"""
Representation of stopping power data for individual ions.

This module defines the :class:`StoppingPowerTable`, which stores and validates
LET vs. energy curves for a given ion in liquid water.

Main features
-------------

- Parsing from text files or dictionaries
- Interpolation and resampling
- Plotting
- Metadata handling (Z, A, source, ionization potential)

This class serves as the core data structure for model computations.

Examples
--------

>>> from pymkm.io.stopping_power import StoppingPowerTable
>>> spt = StoppingPowerTable.from_txt("defaults/mstar_3_12/Z06_A12_Carbon.txt")
>>> spt.ion_symbol
'C'
>>> spt.energy[:3]
array([1.0, 2.0, 3.0])
>>> spt.plot(show=False)
"""

from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union
import warnings

from pymkm.io.data_registry import load_lookup_table
from pymkm.utils.interpolation import Interpolator

# Always show warnings
warnings.simplefilter("always", UserWarning)

class StoppingPowerTable:
    """
    Class for representing and handling stopping power (LET) data for ions in liquid water.
    
    Provides utilities for data validation, plotting, serialization, and interpolation.
    :attr REQUIRED_HEADER_KEYS: Required header keys for parsing stopping power tables.
    :attr REQUIRED_DICT_KEYS: Required dictionary keys for internal representation.
    :attr DEFAULT_TARGET: Default target material (``WATER_LIQUID``).
    """
    
    REQUIRED_HEADER_KEYS = ["Ion", "AtomicNumber", "MassNumber", "SourceProgram", "IonizationPotential", "Target"]
    REQUIRED_DICT_KEYS = ["ion_symbol", "atomic_number", "mass_number", "source_program", "ionization_potential", "target"]
    DEFAULT_TARGET = "WATER_LIQUID" 

    @classmethod
    def get_lookup_table(cls) -> Dict[str, Dict[str, int]]:
        """
        Retrieve the ion properties lookup table used to resolve ion metadata.
    
        The table maps element names (e.g., "Carbon") to their symbol, atomic number,
        mass number, and display color. Based on IUPAC reference data from:
        https://ciaaw.org/atomic-weights.htm
    
        :returns: A dictionary with element names as keys and their properties as nested dictionaries.
        :rtype: dict[str, dict[str, int]]
        """
        return load_lookup_table()

    def __init__(self, ion_input: Union[str, int], energy: np.ndarray, let: np.ndarray,
                 mass_number: Optional[int] = None, source_program: Optional[str] = None,
                 ionization_potential: Optional[float] = None):
        """
        Initialize a StoppingPowerTable for a given ion.
    
        :param ion_input: Ion identifier (element name, symbol, or atomic number).
        :type ion_input: str or int
        :param energy: Array of energy values in MeV/u.
        :type energy: np.ndarray
        :param let: Corresponding LET values in MeV/cm.
        :type let: np.ndarray
        :param mass_number: Optional mass number override.
        :type mass_number: Optional[int]
        :param source_program: Optional name of the source that provided the data.
        :type source_program: Optional[str]
        :param ionization_potential: Optional ionization potential of the medium.
        :type ionization_potential: Optional[float]
    
        :raises TypeError: If ion_input is not a valid type.
        :raises ValueError: If ion cannot be resolved in the lookup table.
        """
        lookup = self.get_lookup_table()

        # Try matching a full element name (e.g., "Carbon")
        if isinstance(ion_input, str) and ion_input in lookup:
            ion_data = lookup[ion_input]
            self.ion_symbol = ion_data["symbol"]
        
        # Try matching an element symbol (e.g., "C")
        elif isinstance(ion_input, str) and ion_input in {v["symbol"] for v in lookup.values()}:
            self.ion_symbol = ion_input
        
        # Try matching an atomic number (e.g., 6 or "6")
        elif isinstance(ion_input, (int, str)) and str(ion_input).isdigit():
            atomic_number = int(ion_input)
            # Look up the first element that matches the given atomic number
            match = next((v["symbol"] for v in lookup.values() if v["atomic_number"] == atomic_number), None)
            if match:
                self.ion_symbol = match
            else:
                raise ValueError(f"No ion found with atomic number {atomic_number}")
        
        # If none of the above matches, raise an error
        else:
            raise ValueError(f"Unknown ion identifier: {ion_input}")

        # Retrieve the corresponding record key.
        reverse = {v["symbol"]: k for k, v in lookup.items()}
        record_key = reverse.get(self.ion_symbol, ion_input)
        self.atomic_number = lookup[record_key]['atomic_number']
        self.mass_number = mass_number or lookup[record_key]['mass_number']
        self.color = lookup[record_key]['color']
        self.energy = np.asarray(energy)
        self.let = np.asarray(let)
        self.source_program = source_program
        self.ionization_potential = ionization_potential
        self.target = StoppingPowerTable.DEFAULT_TARGET

        self._validate()

    @property
    def ion_name(self) -> str:
        """
         Get the ion symbol (e.g., "C" for carbon).
        
         :returns: Ion symbol string.
         :rtype: str
         """
        return self.ion_symbol

    @property
    def energy_grid(self) -> np.ndarray:
        """
        Access the energy array (grid).
        
        :returns: Energy values.
        :rtype: np.ndarray
        """
        return self.energy

    @property
    def stopping_power(self) -> np.ndarray:
        """
        Access the LET (stopping power) array.
        
        :returns: LET values.
        :rtype: np.ndarray
        """
        return self.let

    def _validate(self):
        """
        Perform internal consistency checks on the input data.
    
        Validates that:
          - Energy and LET arrays have identical shape.
          - At least a minimum number of points are provided 
            (150 by default, unless source is 'mstar_3_12').
          - Energy values are finite, non-NaN, strictly increasing,
            and (ideally) within the validated energy range (0.1–1000 MeV/u).
            Values outside this range trigger a strong visible warning.
    
        :raises ValueError: If shapes mismatch, insufficient points, non-finite values,
                            or non-monotonic energy sequence.
        :warns UserWarning: If energy values are outside the validated range.
        """
        if self.energy.shape != self.let.shape:
            raise ValueError(
                f"Shape mismatch: energy {self.energy.shape}, LET {self.let.shape}."
            )
    
        if not np.isfinite(self.energy).all() or not np.isfinite(self.let).all():
            raise ValueError("Energy and LET arrays must contain only finite values.")
    
        if len(self.energy) < 150 and self.source_program != "mstar_3_12":
            raise ValueError(
                f"Insufficient data points: got {len(self.energy)}, need at least 150."
            )
    
        if not np.all(np.diff(self.energy) > 0):
            raise ValueError("Energy values must be strictly increasing with no duplicates.")
    
        # Energy range check (validated domain of the package)
        if (self.energy.min() < 0.1) or (self.energy.max() > 1000):
            msg = (
                "\n\033[93m"  # bright yellow
                + "═" * 72 + "\n"
                + f"⚠️  WARNING on {self.ion_name}: Energy values outside validated range (0.1–1000 MeV/u)\n"
                + f"    Found range: {self.energy.min():.3f} – {self.energy.max():.1f} MeV/u\n"
                + "    Results may be unreliable outside this domain.\n"
                + "═" * 72
                + "\033[0m\n"  # reset
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

    def to_dict(self) -> Dict:
        """
        Serialize the object to a dictionary.
    
        :returns: Dictionary containing ion metadata and LET table.
        :rtype: dict
        """
        return {
            "ion_symbol": self.ion_symbol,
            "mass_number": self.mass_number,
            "atomic_number": self.atomic_number,
            "energy": self.energy.tolist(),
            "let": self.let.tolist(),
            "source_program": self.source_program,
            "ionization_potential": self.ionization_potential,
            "target": StoppingPowerTable.DEFAULT_TARGET
        }

    @staticmethod
    def from_dict(data: Dict) -> "StoppingPowerTable":
        """
        Create a :class:`StoppingPowerTable` instance from a serialized dictionary.

        The input dictionary must contain at least the required fields
        listed in :attr:`StoppingPowerTable.REQUIRED_DICT_KEYS`, also listed below.

        **Expected dictionary format**::

            {
                "ion_symbol": H,                    # str, symbol of the ion
                "energy": [...],                    # list/array of float, energy values (MeV/u)
                "let": [...],                       # list/array of float, corresponding stopping powers (MeV/cm)
                "mass_number": 1,                   # int, mass number of the ion
                "source_program": fluka_2020_0      # str, program used to generate data
                "ionization_potential": 75.0        # float, ionization potential (eV)
            }

        :param data: Dictionary containing serialized table data.
        :type data: dict

        :returns: A new :class:`StoppingPowerTable` instance.
        :rtype: StoppingPowerTable

        :raises ValueError: If required fields are missing.
        """
        missing = [key for key in StoppingPowerTable.REQUIRED_DICT_KEYS if key not in data]
        if missing:
            raise ValueError(f"Missing required field(s) in dictionary: {', '.join(missing)}")
                    
        return StoppingPowerTable(
            ion_input=data["ion_symbol"],
            energy=np.array(data["energy"]),
            let=np.array(data["let"]),
            mass_number=data.get("mass_number"),
            source_program=data.get("source_program"),
            ionization_potential=data.get("ionization_potential")
        )

    def plot(
        self,
        label: Optional[str] = None,
        show: bool = True,
        ax: Optional[plt.Axes] = None
    ):
        """
        Plot stopping power (LET) as a function of energy.
    
        :param label: Optional label for the plot legend.
        :type label: Optional[str]
        :param show: Whether to call plt.show().
        :type show: bool
        :param ax: Matplotlib Axes object to draw on. If None, a new figure is created.
        :type ax: Optional[matplotlib.axes.Axes]
        """

        # Create figure/axes if not provided
        created_fig = False
        if ax is None:
            _, ax = plt.subplots()
            ax.set_title(f'{self.ion_symbol}: Stopping Power vs Energy')
            created_fig = True

        else:
            ax.set_title('Stopping Power vs Energy')

        ax.plot(self.energy, self.let, label=label or self.ion_symbol, color=self.color, alpha=0.5, linewidth=6)
        ax.set_xscale('log')
        ax.set_xlabel('Energy [MeV/u]')
        ax.set_ylabel('Stopping Power [MeV/cm]')
        ax.grid(True)
        ax.legend()

        if show and created_fig:
            plt.tight_layout()
            plt.show()

    def resample(self, new_grid: np.ndarray):
        """
        Resample the LET curve onto a new energy grid using log-log interpolation.
    
        :param new_grid: The target energy grid (must be strictly increasing).
        :type new_grid: np.ndarray
    
        :raises ValueError: If the new grid is not strictly increasing.
        """
        if not np.all(np.diff(new_grid) > 0):
            raise ValueError("New energy grid must be strictly increasing.")
        log_energy = np.log10(self.energy)
        log_let = np.log10(self.let)
        log_interp = np.interp(np.log10(new_grid), log_energy, log_let)
        self.energy = new_grid
        self.let = 10 ** log_interp

    def interpolate(self, *, energy: np.ndarray = None, let: np.ndarray = None, loglog: bool = True):
        """
        Interpolate LET or energy values using internal data.
    
        Delegates to `Interpolator` and supports log-log interpolation.
    
        :param energy: Energy values at which to compute LET.
        :type energy: Optional[np.ndarray]
        :param let: LET values at which to compute corresponding energies.
        :type let: Optional[np.ndarray]
        :param loglog: Whether to perform interpolation in log-log space.
        :type loglog: bool
    
        :returns: Interpolated LETs or a dict of energies for each LET.
        :rtype: np.ndarray or dict[float, np.ndarray]
        """
        interpolator = Interpolator(self.energy, self.let, loglog=loglog)
        return interpolator.interpolate(energy=energy, let=let)
    
    @staticmethod
    def from_txt(filepath: str) -> "StoppingPowerTable":
        """
        Create a :class:`StoppingPowerTable` from a .txt file containing header and data.

        The input file must contain a header with required fields and a data section
        with energy and LET values. The header must include the ``Ion``, ``AtomicNumber``,
        ``MassNumber``, ``SourceProgram``, ``IonizationPotential``, and ``Target`` (case-sensitive) fields.

        *Example header*::
           
            SourceProgram=fluka_2020_0
            Target=WATER_LIQUID
            IonizationPotential=77.0
            Ion=H
            AtomicNumber=1
            MassNumber=1

        The data section should contain energy and LET values in two columns, expressed
        in MeV/u and MeV/cm respectively.
        
        :param filepath: Path to the input .txt file.
        :type filepath: str
        
        :returns: :class:`StoppingPowerTable` instance parsed from file.
        :rtype: StoppingPowerTable
        
        :raises ValueError: If required header keys or element definitions are missing or inconsistent.
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            lines = f.readlines()

        header = {line.split('=')[0].strip(): line.split('=')[1].strip()
                  for line in lines if '=' in line}
        
        missing = [key for key in StoppingPowerTable.REQUIRED_HEADER_KEYS if key not in header]
        if missing:
            raise ValueError(f"Missing required header field(s): {', '.join(missing)}")
            
        if header["Target"].upper() != StoppingPowerTable.DEFAULT_TARGET:
            raise ValueError(f"Unsupported target: {header['Target']}. Only {StoppingPowerTable.DEFAULT_TARGET} is supported.")

        ion_symbol = header.get('Ion')
        atomic_number = int(header.get('AtomicNumber'))
        mass_number = int(header.get('MassNumber'))
        source_program = header.get('SourceProgram')
        ionization_potential = float(header["IonizationPotential"])

        lookup = load_lookup_table()
        reverse = {v['symbol']: k for k, v in lookup.items()}
        if ion_symbol not in reverse:
            raise ValueError(f"Ion symbol '{ion_symbol}' is not recognized.")

        expected_atomic_number = lookup[reverse[ion_symbol]]['atomic_number']
        expected_mass_number = lookup[reverse[ion_symbol]]['mass_number']
        if atomic_number != expected_atomic_number or mass_number != expected_mass_number:
            raise ValueError(f"Mismatch in atomic or mass number for ion '{ion_symbol}'.")

        data_lines = [line for line in lines if re.match(r'^\s*\d', line)]
        data = np.array([list(map(float, line.split())) for line in data_lines])
        energy = data[:, 0]
        let = data[:, 1]

        return StoppingPowerTable(
            ion_input=ion_symbol,
            energy=energy,
            let=let,
            mass_number=mass_number,
            source_program=source_program,
            ionization_potential=ionization_potential
        )
