from typing import Union, Sequence, Dict
import numpy as np

class Interpolator:
    """
    A general-purpose interpolator for LET or specific energy curves,
    supporting non-monotonic data and optional log-log interpolation.
    """

    def __init__(self, energy: np.ndarray, values: np.ndarray, loglog: bool = False):
        """
        Initialize the interpolator.

        Parameters:
          energy: Array of energy values.
          values: Array of LET or specific energy values.
          loglog: Whether to apply log-log interpolation.
        """
        self.energy = np.asarray(energy)
        self.values = np.asarray(values)
        self.loglog = loglog

    def _identify_monotonic_segments(self):
        y_array = self.values
        diff = np.diff(y_array)
        sign_changes = np.where(np.diff(np.sign(diff)))[0] + 1

        segments = []
        start_idx = 0
        for change in sign_changes:
            segments.append((start_idx, change))
            start_idx = change
        segments.append((start_idx, len(y_array) - 1))
        return segments

    def _interpolate_energy_for_let(self, let_input: Union[float, Sequence[float]]) -> Dict[float, np.ndarray]:
        """
        Interpolates energy values corresponding to given LET input(s).

        Parameters:
          let_input: float or array-like LET values.

        Returns:
          Dict[float, np.ndarray]: Each LET key maps to array of energies where it occurs.
        """
        let_input = np.atleast_1d(let_input)
        min_let = self.values.min()
        max_let = self.values.max()

        out_of_bounds = (let_input < min_let) | (let_input > max_let)
        if np.any(out_of_bounds):
            raise ValueError(f"LET input(s) {let_input[out_of_bounds]} are out of bounds: [{min_let}, {max_let}].")

        segments = self._identify_monotonic_segments()
        result = {}

        for single_let in let_input:
            energy_values = []

            for start_idx, end_idx in segments:
                x = self.values[start_idx:end_idx + 1]
                y = self.energy[start_idx:end_idx + 1]

                if len(x) < 2:
                    continue

                if np.all(np.diff(x) > 0):
                    sort_idx = np.argsort(x)
                elif np.all(np.diff(x) < 0):
                    sort_idx = np.argsort(-x)
                # else:
                #     continue

                x = x[sort_idx]
                y = y[sort_idx]

                for i in range(len(x) - 1):
                    x0, x1 = x[i], x[i + 1]
                    y0, y1 = y[i], y[i + 1]

                    if (x0 - single_let) * (x1 - single_let) > 0:
                        continue

                    if self.loglog:
                        if any(val <= 0 for val in (x0, x1, y0, y1, single_let)):
                            raise ValueError("Log-log interpolation requires all inputs > 0.")
                        x0, x1 = np.log10([x0, x1])
                        y0, y1 = np.log10([y0, y1])
                        let_log = np.log10(single_let)
                        e_log = np.interp(let_log, [x0, x1], [y0, y1])
                        energy = 10 ** e_log
                    else:
                        energy = np.interp(single_let, [x0, x1], [y0, y1])

                    energy_values.append(energy)

            result[single_let] = np.array(energy_values)

        return result

    def _interpolate_let_for_energy(self, energy_input: Union[float, Sequence[float]]) -> np.ndarray:
        """
        Interpolates LET values corresponding to given energy input(s).

        Parameters:
          energy_input: float or array-like energy values.

        Returns:
          np.ndarray: Array of interpolated LET values.
        """
        energy_input = np.atleast_1d(energy_input)
        min_e, max_e = self.energy.min(), self.energy.max()

        if np.any(energy_input < min_e) or np.any(energy_input > max_e):
            raise ValueError(f"Energy input(s) {energy_input} out of bounds: [{min_e}, {max_e}].")

        x = self.energy
        y = self.values

        if self.loglog:
            if np.any(x <= 0) or np.any(y <= 0):
                raise ValueError("Log-log interpolation requires all values to be > 0.")
            x_log = np.log10(x)
            y_log = np.log10(y)
            energy_input_log = np.log10(energy_input)
            interp_vals = np.interp(energy_input_log, x_log, y_log)
            result = 10 ** interp_vals
        else:
            result = np.interp(energy_input, x, y)

        return result

    def interpolate(self, *, energy=None, let=None):
        """
        Interpolates LET or energy values depending on the input.

        This is a unified interface to perform:
          - LET interpolation from given energy values (energy → LET)
          - Energy interpolation from given LET values (LET → energy),
            possibly yielding multiple values per LET due to non-monotonicity.

        Parameters:
          energy (float or array-like, optional): Energy value(s) at which to interpolate LET.
          let (float or array-like, optional): LET value(s) at which to interpolate energy.

        Returns:
          np.ndarray or dict[float, np.ndarray]:
            If `energy` is provided, returns interpolated LET values.
            If `let` is provided, returns a dictionary mapping each LET value
            to an array of corresponding energy values (possibly multiple).

        Raises:
          ValueError: If both or neither of `energy` and `let` are provided.
        """
        if energy is not None and let is not None:
            raise ValueError("Provide only one of `energy` or `let`, not both.")
        if energy is not None:
            return self._interpolate_let_for_energy(energy)
        elif let is not None:
            return self._interpolate_energy_for_let(let)
        else:
            raise ValueError("You must provide either `energy` or `let`.")
