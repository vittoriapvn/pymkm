from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union

from pymkm.io.data_registry import load_lookup_table
from pymkm.utils.interpolation import Interpolator


class StoppingPowerTable:
    """A class for handling stopping power (LET) data for ions in liquid water."""
    
    REQUIRED_HEADER_KEYS = ["Ion", "AtomicNumber", "MassNumber", "SourceProgram", "IonizationPotential", "Target"]
    REQUIRED_DICT_KEYS = ["ion_symbol", "atomic_number", "mass_number", "source_program", "ionization_potential", "target"]
    DEFAULT_TARGET = "WATER_LIQUID" 

    @classmethod
    def get_lookup_table(cls) -> Dict[str, Dict[str, int]]:
        """Returns the ion properties lookup table from JSON."""
        # https://ciaaw.org/atomic-weights.htm
        return load_lookup_table()

    def __init__(self, ion_input: Union[str, int], energy: np.ndarray, let: np.ndarray,
                 mass_number: Optional[int] = None, source_program: Optional[str] = None,
                 ionization_potential: Optional[float] = None):
        """
        Initialize a StoppingPowerTable for a given ion.

        Parameters:
          ion_input (Union[str, int]): The ion identifier, which can be one of the following:
                                       - Full element name (e.g., "Carbon")
                                       - Element symbol (e.g., "C")
                                       - Atomic number (e.g., 6 or "6")
          energy (np.ndarray): Array of energy values.
          let (np.ndarray): Array of corresponding LET values.
          mass_number (Optional[int]): Optional override for the element's mass number.
          source_program (Optional[str]): The data source.
          ionization_potential (Optional[float]): The ionization potential.

        Raises:
          TypeError: if ion_input is not a string.
          ValueError: if the provided identifier is not recognized.
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
        Returns the ion symbol.
        (Since the full name is already used as a master key, only the symbol is stored and exposed.)
        """
        return self.ion_symbol

    @property
    def energy_grid(self) -> np.ndarray:
        return self.energy

    @property
    def stopping_power(self) -> np.ndarray:
        return self.let

    def _validate(self):
        """Run internal consistency checks."""
        if self.energy.shape != self.let.shape:
            raise ValueError("Energy and LET arrays must have the same shape.")
        if len(self.energy) < 150 and self.source_program != 'mstar_3_12':
            raise ValueError("At least 150 data points are required.")
        if not np.all(np.diff(self.energy) > 0):
            raise ValueError("Energy values must be strictly increasing.")

    def to_dict(self) -> Dict:
        """Serialize the table to a dictionary."""
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
        """Create a StoppingPowerTable instance from a dictionary."""
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

    def plot(self, label: Optional[str] = None, show: bool = True, new_figure: bool = True):
        """
        Plot Stopping Power vs. Energy.

        Parameters:
          label: Optional label for the curve.
          show: Whether to display the plot.
          new_figure: If True, creates a new figure and sets a title with the ion symbol;
                      if False, plots on the existing figure and sets a generic title.
        """
        if new_figure:
            plt.figure(figsize=(8, 5))
            plt.title(f'{self.ion_symbol}: Stopping Power vs Energy')
        else:
            if not plt.gca().get_title():
                plt.title("Stopping Power vs Energy")
        plt.plot(self.energy, self.let, label=label or self.ion_symbol, color=self.color, alpha=0.5, linewidth=6)
        plt.xscale('log')
        plt.xlabel('Energy [MeV/u]')
        plt.ylabel('Stopping Power [MeV/cm]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()

    def resample(self, new_grid: np.ndarray):
        """Resample LET on a new energy grid using log-linear interpolation."""
        if not np.all(np.diff(new_grid) > 0):
            raise ValueError("New energy grid must be strictly increasing.")
        log_energy = np.log10(self.energy)
        log_let = np.log10(self.let)
        log_interp = np.interp(np.log10(new_grid), log_energy, log_let)
        self.energy = new_grid
        self.let = 10 ** log_interp

    def interpolate(self, *, energy: np.ndarray = None, let: np.ndarray = None, loglog: bool = True):
        """
        Interpolate LET or energy values using the internal data, delegating to Interpolator.
        
        Parameters:
          energy: Energy values at which to interpolate LET (if given).
          let: LET values at which to interpolate energy (if given).
          loglog: Whether to use log-log interpolation.
        
        Returns:
          Interpolated values or dict of results if LET is provided.
        """
        interpolator = Interpolator(self.energy, self.let, loglog=loglog)
        return interpolator.interpolate(energy=energy, let=let)
    
    @staticmethod
    def from_txt(filepath: str) -> "StoppingPowerTable":
        """Create a StoppingPowerTable instance from a .txt file."""
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
