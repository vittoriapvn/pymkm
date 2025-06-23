from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import warnings
from tabulate import tabulate

from pymkm.mktable.core import MKTable

@dataclass
class SFTableParameters:
    """
    Configuration container for computing survival fraction (SF) curves
    using the MKM, SMK, or oxygen-effect-incorporated SMK (OSMK) models.

    Attributes
    ----------
    mktable : MKTable
        Precomputed MKTable containing specific energy values.
    alpha0 : float, optional
        Total linear coefficient α_0 in the LQ model [Gy^-1]. Required unless both alphaL and alphaS are provided.
    beta0 : float, optional
        Quadratic coefficient β_0 in the LQ model [Gy^-2]. If not provided, taken from mktable.params.
    dose_grid : np.ndarray, optional
        Array of dose values [Gy] for which to compute survival fractions.
        Defaults to np.arange(0, 15.5, 0.5).

    OSMK-specific parameters (optional, only required if oxygen effect is enabled via pO2):
    alphaL : float, optional
        Linear coefficient for lethal lesions [Gy^-1].
    alphaS : float, optional
        Linear coefficient for sublethal lesions [Gy^-1].
    zR : float, optional
        Parameter for radiation quality dependence of oxygen effect [Gy]. (OSMK 2021 only)
    gamma : float, optional
        Exponent in R_max(zd) expression. (OSMK 2021 only)
    Rm : float, optional
        Minimum value of R_max. (OSMK 2021 only)
    f_rd_max : float, optional
        Maximum domain radius modification factor under hypoxia. (OSMK 2023 only)
    f_z0_max : float, optional
        Maximum saturation parameter modification factor under hypoxia. (OSMK 2023 only)
    Rmax : float, optional
        Maximum value of radioresistance R at pO2 = 0. (OSMK 2023 only)
    K : float, default=3.0
        Partial pressure [mmHg] at which R(pO2) = (1 + R_max)/2. Default from Inaniwa 2021.
    pO2 : float, optional
        Partial oxygen pressure [mmHg] at which to apply the oxygen effect. Enables OSMK mode if set.
    """

    mktable: MKTable
    alpha0: Optional[float] = None
    beta0: Optional[float] = None
    dose_grid: np.ndarray = field(default_factory=lambda: np.arange(0, 15.5, 0.5))

    # Optional parameters for OSMK
    alphaL: Optional[float] = None
    alphaS: Optional[float] = None
    zR: Optional[float] = None
    gamma: Optional[float] = None
    Rm: Optional[float] = None
    f_rd_max: Optional[float] = None
    f_z0_max: Optional[float] = None
    Rmax: Optional[float] = None
    K: float = 3.0  # mmHg, default
    pO2: Optional[float] = None  # mmHg

    def __post_init__(self):
        if not isinstance(self.mktable, MKTable):
            raise TypeError("mktable must be an instance of MKTable")

        if not isinstance(self.dose_grid, np.ndarray):
            self.dose_grid = np.array(self.dose_grid, dtype=float)

        # === Validate beta0 ===
        beta_from_table = self.mktable.params.beta0
        if self.beta0 is None:
            if beta_from_table is not None:
                self.beta0 = beta_from_table
                warnings.warn("beta0 not provided, using value from MKTable.params.")
            else:
                raise ValueError("beta0 must be provided either explicitly or via MKTable.params.")
        elif beta_from_table is not None and abs(beta_from_table - self.beta0) > 1e-6:
            raise ValueError(
                f"Mismatch between provided beta0 ({self.beta0}) and MKTable.params.beta0 ({beta_from_table})."
            )
    
        # === Handle alpha0, alphaL, alphaS consistency only if OSMK is requested ===
        if self.pO2 is not None:
            alphaL, alphaS, alpha0 = self.alphaL, self.alphaS, self.alpha0
            n_provided = sum(v is not None for v in (alpha0, alphaL, alphaS))

            if n_provided < 2:
                raise ValueError("For OSMK (pO2 specified), at least two of alpha0, alphaL, and alphaS must be provided.")

            if alpha0 is None:
                self.alpha0 = alphaL + alphaS
            elif alphaL is None:
                self.alphaL = alpha0 - alphaS
            elif alphaS is None:
                self.alphaS = alpha0 - alphaL
            else:
                if not np.isclose(alphaL + alphaS, alpha0, atol=1e-6):
                    raise ValueError(f"Inconsistent values for OSMK: alpha0={alpha0}, alphaL + alphaS = {alphaL + alphaS}")

            # === Ensure exclusivity between OSMK 2021 and 2023 parameter sets ===
            has_osmk2021 = any(x is not None for x in (self.zR, self.gamma, self.Rm))
            has_osmk2023 = any(x is not None for x in (self.f_rd_max, self.f_z0_max, self.Rmax))
            if has_osmk2021 and has_osmk2023:
                raise ValueError("Cannot mix OSMK 2021 (zR, γ, Rm) and OSMK 2023 (f_rd_max, f_z0_max, Rmax) parameters.")
                
                
    @classmethod
    def from_dict(cls, config: dict) -> "SFTableParameters":
        """
        Create an SFTableParameters instance from a dictionary, warning on unrecognized keys.

        Parameters
        ----------
        config : dict
            Dictionary of parameters with keys matching the field names.

        Returns
        -------
        SFTableParameters
            A populated dataclass instance.

        Raises
        ------
        ValueError
            If unknown keys are found in the config.
        """
        valid_keys = set(cls.__dataclass_fields__.keys())
        incoming_keys = set(config.keys())
        extra_keys = incoming_keys - valid_keys

        if extra_keys:
            raise ValueError(
                f"Unrecognized keys in SFTableParameters config: {sorted(extra_keys)}"
            )

        return cls(**config)


class SFTable:
    def __init__(self, parameters: SFTableParameters):
        """
        Initialize SFTable from given parameters.

        Parameters
        ----------
        parameters : SFTableParameters
            A dataclass with MKTable, alpha0/beta0, and dose grid.
        """
        self.params = parameters
        self.table = None

    def __repr__(self):
        if self.params.alphaL is not None and self.params.alphaS is not None:
            alpha0 = self.params.alphaL + self.params.alphaS
        else:
            alpha0 = self.params.alpha0
        beta0 = self.params.beta0
        return f"<SFTable | α_0 = {alpha0}, β_0 = {beta0}>"

    def summary(self):
        print("\nSFTable Configuration")
        table = [
            ("\u03b1_0 [Gy^-1]", f"{self.params.alpha0:.3f}" if self.params.alpha0 is not None else "None"),
            ("\u03b2_0 [Gy^-2]", f"{self.params.beta0:.3f}" if self.params.beta0 is not None else "None"),
        ]

        if self.params.pO2 is not None:
            table += [
                ("pO2 [mmHg]", f"{self.params.pO2:.2f}"),
                ("α_L [Gy^-1]", f"{self.params.alphaL:.3f}" if self.params.alphaL is not None else "None"),
                ("α_S [Gy^-1]", f"{self.params.alphaS:.3f}" if self.params.alphaS is not None else "None"),
                ("zR (2021) [Gy]", f"{self.params.zR:.2f}" if self.params.zR is not None else "None"),
                ("γ (2021)", f"{self.params.gamma:.2f}" if self.params.gamma is not None else "None"),
                ("Rm (2021)", f"{self.params.Rm:.2f}" if self.params.Rm is not None else "None"),
                ("f_rd_max (2023)", f"{self.params.f_rd_max:.2f}" if self.params.f_rd_max is not None else "None"),
                ("f_z0_max (2023)", f"{self.params.f_z0_max:.2f}" if self.params.f_z0_max is not None else "None"),
                ("Rmax (2023)", f"{self.params.Rmax:.2f}" if self.params.Rmax is not None else "None")
            ]

        print(tabulate(table, headers=["Parameter", "Value"], tablefmt="fancy_grid"))

    def display(self, results: list):
        """
        Displays the survival fraction results in a readable table format.
    
        Parameters
        ----------
        results : list of dict
            Output of the `compute()` method, expected to contain 'params', 'calculation_info', and 'data'.
    
        Raises
        ------
        ValueError
            If the results list is empty.
        """
        if not results:
            raise ValueError("No results to display. Please run 'compute()' first.")
    
        print("\n📈 Survival Fraction Results:")
        for idx, result in enumerate(results):
            params = result.get("params", {})
            calc_info = result.get("calculation_info", "N/A")
            df = result.get("data")
    
            print(f"\n🔹 Result {idx + 1}")
            print(tabulate(params.items(), headers=["Parameter", "Value"], tablefmt="grid"))
            print(f"\nCalculation Info: {calc_info}")
    
            if df is not None and not df.empty:
                print("\nData Table:")
                print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
            else:
                print("No data found in this result.")
            print("-" * 60)
