from dataclasses import asdict
from typing import Optional, Union, Literal
from dataclasses import dataclass
from tabulate import tabulate
from pathlib import Path
import warnings
import pickle
import datetime
import pandas as pd
from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.utils.geometry_tools import GeometryTools


@dataclass
class MKTableParameters:

    @classmethod
    def from_dict(cls, config: dict) -> "MKTableParameters":
        """
        Create an MKTableParameters instance from a dictionary, warning on unrecognized keys.
    
        Parameters
        ----------
        config : dict
            Dictionary of parameters with keys matching the field names.
    
        Returns
        -------
        MKTableParameters
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
                f"Unrecognized keys in MKTableParameters config: {sorted(extra_keys)}"
            )
    
        return cls(**config)

    """
    Configuration container for all MKTable model and geometry parameters.

    This dataclass defines the input parameters required to configure
    a microdosimetric table generation for MKM or SMK models, and optional oxygen-effect
    correction parameters (OSMK 2023), which can be applied transiently during computation.

    Parameters
    ----------
    domain_radius : float
        Radius (in μm) of the sensitive domain where energy deposition is calculated.
    nucleus_radius : float
        Radius (in μm) of the cell nucleus.
    z0 : Optional[float]
        Saturation parameter (Gy). Typically used in SMK. If not provided,
        it can be derived from beta0 in fallback scenarios.
    beta0 : Optional[float]
        Quadratic coefficient β₀ in the LQ model at low LET. Required in MKM,
        and optionally used to derive z0 in SMK if z0 is not given.
    model_name : str
        The track structure model to use ('Kiefer-Chatterjee' or 'Scholz-Kraft').
    core_radius_type : str
        Core radius model ('constant' or 'energy-dependent').
    base_points_b : int
        (default: 150)
        Number of impact parameter sampling points (used in z(b) calculations).
    base_points_r : int
        (default: 150)
        Number of radial sampling points (used in dose integration).
    use_stochastic_model : bool
        Whether to compute stochastic (SMK) quantities. If False, classic MKM is used.
    pO2 : Optional[float]
        Partial pressure of oxygen [mmHg]. Enables the OSMK 2023 correction if set.
    f_rd_max : Optional[float]
        Maximum scaling factor for domain radius under full hypoxia (pO2 = 0).
    f_z0_max : Optional[float]
        Maximum scaling factor for z0 under full hypoxia (pO2 = 0).
    Rmax : Optional[float]
        Maximum radioresistance ratio at pO2 = 0.
    K : float, default=3.0
        Half-effect oxygen pressure [mmHg] where R(pO2) = (1 + Rmax)/2.
    """
    domain_radius: float
    nucleus_radius: float
    z0: Optional[float] = None
    beta0: Optional[float] = None

    model_name: str = "Kiefer-Chatterjee"
    core_radius_type: str = "energy-dependent"
    base_points_b: int = GeometryTools.generate_default_radii.__defaults__[1]
    base_points_r: int = GeometryTools.generate_default_radii.__defaults__[1]

    use_stochastic_model: bool = False
    
    # --- OSMK 2023 Correction Parameters (optional) ---
    pO2: Optional[float] = None        # Oxygen partial pressure [mmHg]
    f_rd_max: Optional[float] = None   # Max scaling for domain radius
    f_z0_max: Optional[float] = None   # Max scaling for z0
    Rmax: Optional[float] = None       # Maximum R at pO2 = 0
    K: float = 3.0                     # Half-effect oxygen pressure [mmHg]
    
    apply_oxygen_effect: bool = False  # Enables correction if True and parameters are present


class MKTable:
    def __repr__(self):
        return (f"<MKTable model={self.model_version}, r_d={self.params.domain_radius}, "
                f"R_n={self.params.nucleus_radius}>")

    @property
    def model_version(self) -> str:
        return "stochastic" if self.params.use_stochastic_model else "classic"
    
    def _default_filename(self, extension: str = ".pkl") -> Path:
        """
        Generate a descriptive filename for the MKTable output based on the model configuration.

        The filename includes:
        - model type (mkm or smk)
        - stopping power source (from sp_table_set)
        - track structure model abbreviation (kc or sk)
        - core radius type (const or ed)
        - domain radius, nucleus radius, z0 and beta0 values

        The file is saved in the appropriate subfolder of ~/.pyMKM based on extension.

        Parameters
        ----------
        extension : str, default ".pkl"
            File extension used to determine the output format and destination folder.

        Returns
        -------
        Path
            Full path where the output file should be saved.
        """
        root = Path.home() / ".pyMKM" / extension.strip(".")
        root.mkdir(parents=True, exist_ok=True)
        suffix = extension if extension.startswith(".") else f".{extension}"

        s = self.sp_table_set.source_info.replace(" ", "_").replace("/", "-")
        r_d = f"rd{self.params.domain_radius:.2f}"
        r_n = f"rn{self.params.nucleus_radius:.1f}"
        z0 = f"z0{self.params.z0:.0f}" if self.params.z0 is not None else "z0None"
        b0 = f"b0{self.params.beta0:.3f}" if self.params.beta0 is not None else "b0None"

        prefix = "smk" if self.params.use_stochastic_model else "mkm"
        model_abbr = "kc" if self.params.model_name.lower().startswith("kiefer") else "sk"
        core_abbr = "const" if self.params.core_radius_type == "constant" else "ed"
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{s}_{model_abbr}_{core_abbr}_rd{r_d}_Rn{r_n}_z0{z0}_b0{b0}_{timestamp}{suffix}"
        return root / filename
    
    def save(self, filename: Optional[Union[str, Path]] = None):
        """
        Save the computed MKTable results to a pickle file.
        
        If no filename is provided, a default name is generated based on model version,
        geometry parameters, and stopping power source. The file is saved under ~/.pyMKM/pkl/.
        
        Parameters
        ----------
        filename : str or Path, optional
            Destination path for the pickle file. If None, a default path is used.
        
        Raises
        ------
        ValueError
            If no data is available in the table (i.e. compute() was not called).
        """
        if not self.table:
            raise ValueError("Cannot save: MKTable has not been computed yet. Run 'compute()' first.")
        path = Path(filename) if filename else self._default_filename(".pkl")
        with open(path, "wb") as f:
            pickle.dump(self.table, f)
        print(f"✅ Table saved to: {path}")
        
    def load(self, filename: Union[str, Path]):
        """
        Load previously saved MKTable results from a pickle file.
        
        Parameters
        ----------
        filename : str or Path
            Full path to the .pkl file to be loaded.
        
        Raises
        ------
        FileNotFoundError
            If the provided path does not exist.
        """
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            self.table = pickle.load(f)
        print(f"📂 Table loaded from: {path}")

    def summary(self, verbose: bool = False):
        """
        Print a formatted summary of the current MKTable configuration using a table.
    
        Parameters
        ----------
        verbose : bool, optional
            If True, includes detailed technical settings. Default is False.
        """
        param_dict = asdict(self.params)
    
        # Main physical parameters with symbols and units
        main_parameters = [
            ("r_d [μm]", param_dict["domain_radius"]),
            ("R_n [μm]", param_dict["nucleus_radius"]),
            ("z₀ [Gy]", param_dict["z0"]),
            ("β₀ [Gy⁻²]", param_dict["beta0"]),
        ]
    
        # Technical parameters
        technical_parameters = [
            ("Track structure model", param_dict["model_name"]),
            ("Track core type", param_dict["core_radius_type"]),
            ("Sampling points for b", param_dict["base_points_b"]),
            ("Sampling points for r", param_dict["base_points_r"]),
        ]
    
        print("\nMKTable Configuration")
        print(f"\nModel version: {self.model_version}")
        print(f"\nStopping power source: {self.sp_table_set.source_info}")
    
        print(tabulate(main_parameters, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
    
        if verbose:
            ions = self.sp_table_set.get_available_ions()
            if ions:
                print("\nAvailable ions:")
                print(", ".join(ions))
            print()
            print(tabulate(technical_parameters, headers=["Setting", "Value"], tablefmt="fancy_grid"))
            print("\nNote: Sampling points refer to base values before internal refinement.")


    """
    MKTable handles the generation and management of microdosimetric tables used in
    MKM and SMK radiobiological models.

    The class stores ion-specific calculations for domain/nucleus geometry, stopping
    power interpolation, and dose-averaged quantities based on the selected physical
    model.

    Parameters
    ----------
    parameters : MKTableParameters
        A dataclass containing all geometric, numerical and model options.
    sp_table_set : Optional[StoppingPowerTableSet]
        If provided, uses the given set of stopping power tables. Otherwise, loads
        the default source ("fluka_2020_0").

    Attributes
    ----------
    params : MKTableParameters
        Input configuration and geometry.
    sp_table_set : StoppingPowerTableSet
        Loaded tables of energy vs LET for each ion.
    table : dict
        Placeholder dictionary for storing results after computation.
    """

    def __init__(self, parameters: MKTableParameters, sp_table_set: Optional[StoppingPowerTableSet] = None):
        self.params = parameters
        self.sp_table_set = sp_table_set or StoppingPowerTableSet.from_default_source("fluka_2020_0")
        self.table = {}
        self._validate_parameters()

    def _validate_parameters(self):
        p = self.params

        # --- Base validation: MKM / SMK logic ---
        if p.z0 is None and p.beta0 is None:
            raise ValueError(
                "Both z0 and beta0 are missing. MKM typically requires beta0; SMK typically requires z0.\n"
                "At least one must be provided to proceed."
            )

        if p.use_stochastic_model:
            if p.z0 is None and p.beta0 is not None:
                warnings.warn("z0 not provided. It will be derived from beta0, which is not standard for SMK.")
            if p.z0 is not None and p.beta0 is not None:
                warnings.warn("Both z0 and beta0 provided. z0 will be used for SMK; beta0 retained only for post-processing.")
        else:
            if p.z0 is not None and p.beta0 is None:
                warnings.warn("z0 provided but beta0 is missing. Will use z0 directly, which is not standard for MKM.")
            if p.z0 is not None and p.beta0 is not None:
                warnings.warn("Both z0 and beta0 provided. In MKM, beta0 will be used to compute z0; the user-provided z0 will be discarded.")
                p.z0 = None # force recalculation from beta0
        
        # --- OSMK 2023 correction validation ---
        if p.apply_oxygen_effect:
            if not p.use_stochastic_model:
                raise ValueError("apply_oxygen_effect=True requires use_stochastic_model=True.")
            required_osmk = ("pO2", "f_rd_max", "f_z0_max", "Rmax")
            missing = [k for k in required_osmk if not hasattr(p, k) or getattr(p, k) is None]
            if missing:
                raise ValueError(f"apply_oxygen_effect=True but missing OSMK 2023 parameters: {missing}")

    def _refresh_parameters(self, original_params: Optional[MKTableParameters] = None) -> None:
        """
        Refresh self.params by reassigning all its fields based on their current values.
    
        If any field is changed compared to original_params, prints the changes and re-displays the summary.
        """
        updated_fields = {}
        current = self.params
        updated = MKTableParameters.from_dict(asdict(current))
    
        reference = original_params or updated  # fallback: compare to self (no-op)
    
        for field_name in current.__dataclass_fields__:
            old_value = getattr(reference, field_name)
            new_value = getattr(updated, field_name)
            if old_value != new_value:
                setattr(self.params, field_name, new_value)
                updated_fields[field_name] = (old_value, new_value)
    
        if updated_fields:
            print("\nMKTableParameters updated:")
            for k, (old, new) in updated_fields.items():
                print(f" - {k}: {old} → {new}")
            print("\nUpdated configuration summary:")
            self.summary(verbose=False)

    def get_table(self, ion: Union[str, int]) -> pd.DataFrame:
        """
        Retrieve the computed microdosimetric table for a specific ion.

        Parameters
        ----------
        ion : str or int
            The ion to retrieve. Can be atomic number (e.g., 6), symbol ('C'), or full name ('Carbon').

        Returns
        -------
        pandas.DataFrame
            The computed table with columns like 'energy', 'let', 'z_bar_star_domain', etc.

        Raises
        ------
        ValueError
            If no results are available or the ion is not found.
        """
        if not self.table:
            raise ValueError("No computed results found. Run 'compute()' first.")

        ion_key = self.sp_table_set._map_to_fullname(ion)

        if ion_key not in self.table:
            raise ValueError(f"Ion '{ion}' not found in computed table.")

        return self.table[ion_key]["data"]

    def display(self, preview_rows: int = 5):
        """
        Displays the calculated results for each ion stored in mk_table in a formatted output.
        
        The method first checks if there is any calculation data available. 
        If data exists, it prints a summary of properties for each ion, 
        followed by a preview of the first and last rows of the associated pandas DataFrame.
        
        Parameters
        ----------
        preview_rows : int, optional
            Number of top and bottom rows to display from each ion’s results table (default is 5).
        
        Raises
        ------
        ValueError
            If no calculation data is found, prompting the user to run 'compute()' first.
        """
        if not self.table:
            raise ValueError("No computed results found. Please run 'compute()' first.")
    
        print("\n📊 Computed Microdosimetric Tables:")
        for ion_key, result in self.table.items():
            print(f"\n🔹 Ion: {ion_key}")
            
            sp_info_table = [(k, v) for k, v in result["stopping_power_info"].items()]
            print(tabulate(sp_info_table, headers=["Stopping Power Info", "Value"], tablefmt="grid"))

            param_table = [(k, v) for k, v in result["params"].items()]
            print(tabulate(param_table, headers=["Parameter", "Value"], tablefmt="grid"))
       
            df = result["data"]
            top = df.head(preview_rows)
            bottom = df.tail(preview_rows)
            
            print(f"\nTop {preview_rows} rows:")
            print(tabulate(top, headers="keys", tablefmt="fancy_grid", showindex=False))
    
            print(f"\nBottom {preview_rows} rows:")
            print(tabulate(bottom, headers="keys", tablefmt="fancy_grid", showindex=False))
            print("-" * 60)
    
    def write_txt(
        self,
        *,
        params: dict,
        filename: Union[str, Path] = None,
        model: Literal["classic", "stochastic"] = None,
        max_atomic_number: int
    ):
        """
        Export the computed MKTable to a .txt file compatible with external tools.
    
        Parameters
        ----------
        params : dict
            Dictionary containing required metadata fields.
            For model="classic", must include:
                - 'CellType': Name of the cell line or type
                - 'Alpha_0': Linear coefficient alpha_0 [Gy^-1]
                - Optional 'Beta': Quadratic coefficient beta0 [Gy^-2] (must match self.params.beta0 if both given)
    
            For model="stochastic", must include:
                - 'CellType': Name of the cell line or type
                - 'Alpha_ref': Reference alpha [Gy^-1]
                - 'Beta_ref': Reference beta [Gy^-2]
                - Optional 'scale_factor': Defaults to 1.0 if not provided (with warning)
                - 'Alpha0': Linear coefficient for SMK (same as Alpha_0)
                - Optional 'Beta0': Quadratic coefficient for SMK (must match self.params.beta0 if both given)
        filename : str or Path, optional
            Destination file path. If None, a default name is generated.
        model : {"classic", "stochastic"}, optional
            Specifies which model version to export. If None, uses the current model.
        max_atomic_number : int
            Maximum atomic number (inclusive) up to which ions will be written.
    
        Raises
        ------
        ValueError
            If the table is empty, required parameters are missing, or there is a mismatch in Beta.
        KeyError
            If unexpected parameters are passed.
        """
        if not self.table:
            raise ValueError("Cannot write: MKTable has not been computed yet. Run 'compute()' first.")
    
        model = model or self.model_version
    
        if model == "stochastic" and not self.params.use_stochastic_model:
            raise ValueError("Stochastic output requested but MKTable was computed in classic mode.")
    
        if model == "classic":
            allowed_keys = {"CellType", "Alpha_0", "Beta"}
            required_keys = {"CellType", "Alpha_0"}
        else:
            allowed_keys = {"CellType", "Alpha_ref", "Beta_ref", "scale_factor", "Alpha0", "Beta0"}
            required_keys = {"CellType", "Alpha_ref", "Beta_ref", "Alpha0"}
    
        incoming_keys = set(params.keys())
    
        if not required_keys.issubset(incoming_keys):
            missing = required_keys - incoming_keys
            raise KeyError(f"Missing required keys in 'params': {missing}")
    
        extra = incoming_keys - allowed_keys
        if extra:
            raise KeyError(f"Unexpected keys in 'params': {extra}")
        
        # Determine maximum available Z from stopping_power_info
        available_Z = [self.table[k]["stopping_power_info"]["atomic_number"] for k in self.table]
        max_Z_table = max(available_Z)

        if max_atomic_number > max_Z_table:
            raise ValueError(f"Requested max_atomic_number={max_atomic_number} exceeds computed table max Z={max_Z_table}.")
    
        # Common header
        path = Path(filename) if filename else self._default_filename(".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
    
        with open(path, "w") as f:
            f.write(f"CellType  {params['CellType']}\n\n")
    
            if model == "classic":
                f.write(f"Parameter Alpha_0 {params['Alpha_0']:.3f}\n")
                beta_param = params.get("Beta")
                beta_obj = self.params.beta0
    
                if beta_obj is None and beta_param is None:
                    raise ValueError("Beta must be defined either in params or in self.params.beta0.")
                if beta_obj is not None and beta_param is not None:
                    if abs(beta_obj - beta_param) > 1e-6:
                        raise ValueError(
                            f"Mismatch between beta0 in params ({beta_param}) and self.params ({beta_obj})"
                        )
                beta = beta_obj if beta_obj is not None else beta_param
                f.write(f"Parameter Beta {beta:.3f}\n")
                f.write(f"Parameter DomainRadius {self.params.domain_radius:.3f}\n")
                f.write(f"Parameter NucleusRadius {self.params.nucleus_radius:.3f}\n\n")
    
            else:  # stochastic
                f.write(f"Parameter Alpha_ref {params['Alpha_ref']:.3f}\n")
                f.write(f"Parameter Beta_ref {params['Beta_ref']:.3f}\n")
                scale = params.get("scale_factor", 1.0)
                if 'scale_factor' not in params:
                    warnings.warn("'scale_factor' not provided, defaulting to 1.00")
                f.write(f"Parameter scale_factor {scale:.2f}\n")
                f.write(f"Parameter Alpha0 {params['Alpha0']:.3f}\n")
    
                beta_param = params.get("Beta0")
                beta_obj = self.params.beta0
                if beta_obj is None and beta_param is None:
                    raise ValueError("Beta0 must be defined either in params or in self.params.beta0.")
                if beta_obj is not None and beta_param is not None:
                    if abs(beta_obj - beta_param) > 1e-6:
                        raise ValueError(
                            f"Mismatch between Beta0 in params ({beta_param}) and self.params ({beta_obj})"
                        )
                beta = beta_obj if beta_obj is not None else beta_param
                f.write(f"Parameter Beta0 {beta:.3f}\n\n")
    
            for ion_key, result in self.table.items():
                Z = self.table[ion_key]["stopping_power_info"].get("atomic_number")
                # if Z is None:
                #     continue
                if Z > max_atomic_number:
                    continue
                
                df = result["data"]
                f.write(f"Fragment {ion_key}\n")
    
                if model == "classic":
                    if "z_bar_star_domain" not in df.columns:
                        raise KeyError(f"Missing expected column 'z_bar_star_domain' for ion {ion_key}.")
                    for _, row in df.iterrows():
                        f.write(f"{row['energy']:.5e} {row['z_bar_star_domain']:.5e}\n")
                else:
                    expected_cols = ["z_bar_domain", "z_bar_star_domain", "z_bar_nucleus"]
                    for col in expected_cols:
                        if col not in df.columns:
                            raise KeyError(f"Missing expected column '{col}' for ion {ion_key}.")
                    for _, row in df.iterrows():
                        f.write(
                            f"{row['energy']:.5e} {row['z_bar_domain']:.5e} {row['z_bar_star_domain']:.5e} {row['z_bar_nucleus']:.5e}\n"
                        )
                f.write("\n")
    
        print(f"📝 Table written to: {path}")
    
        
        
