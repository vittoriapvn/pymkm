from typing import List, Optional, Union
import matplotlib.pyplot as plt
plt.rcParams.update({
    "axes.linewidth": 1.2,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "legend.fontsize": 14
})
import numpy as np

from .core import MKTable

def _validate_plot_columns(x: str, y: str):
    allowed_x = {"energy", "let"}
    allowed_y = {"z_bar_star_domain", "z_bar_domain", "z_bar_nucleus"}
    if x not in allowed_x:
        raise ValueError(f"Invalid x-axis: '{x}'. Allowed values are: {sorted(allowed_x)}")
    if y not in allowed_y:
        raise ValueError(f"Invalid y-axis: '{y}'. Allowed values are: {sorted(allowed_y)}")


def plot(self: MKTable, 
         ions: Optional[List[Union[str, int]]] = None, *, 
         x: str = "energy", 
         y: str = "z_bar_star_domain", 
         verbose: bool = False):
    """
    Plot microdosimetric quantities from the MKTable for one or more ions.

    Parameters
    ----------
    ions : list of str or int, optional
        List of ions to include in the plot. Can be names (e.g. 'Carbon'), symbols ('C'), or atomic numbers (6).
        If None, all available ions in the table are plotted.
    x : str, default="energy"
        The x-axis variable. Must be either "energy" or "let".
    y : str, default="z_bar_star_domain"
        The y-axis variable. Must be one of "z_bar_star_domain", "z_bar_domain", or "z_bar_nucleus".
    verbose : bool, default=False
        If True, displays a text box with model configuration parameters on the plot.

    Raises
    ------
    RuntimeError
        If the table has not been computed yet.
    ValueError
        If invalid column names are provided for x or y.
    """
    if not self.table:
        raise RuntimeError("No computed results found. Please run 'compute()' before plotting.")

    ions = ions or self.sp_table_set.get_available_ions()
    ions = [self.sp_table_set._map_to_fullname(ion) for ion in ions]
    _validate_plot_columns(x, y)

    y_label_map = {
        "z_bar_star_domain": "$\\bar{z}^{*}$ [Gy]",
        "z_bar_domain": "$\\bar{z}_{d}$ [Gy]",
        "z_bar_nucleus": "$\\bar{z}_{n}$ [Gy]"
    }
    x_label_map = {
        "energy": "Energy [MeV/u]",
        "let": "LET [MeV/cm]"
    }

    x_min, x_max = np.inf, -np.inf
    y_max = -np.inf
    for ion in ions:
        df = self.table[ion]["data"]
        x_vals = df[x].values
        y_vals = df[y].values
        x_min = min(x_min, np.min(x_vals))
        x_max = max(x_max, np.max(x_vals))
        y_max = max(y_max, np.max(y_vals))

    plt.figure(figsize=(12, 8))
    for ion in ions:
        df = self.table[ion]["data"]
        ion_symbol = self.sp_table_set.get(ion).ion_symbol
        color = self.sp_table_set.get(ion).color
        plt.plot(df[x], df[y], label=ion_symbol, color=color, alpha=0.5, linewidth=6)

    plt.xlabel(x_label_map.get(x, x.capitalize()))
    plt.ylabel(y_label_map.get(y, y.replace('_', ' ').capitalize()))
    plt.xscale("log" if x == "energy" else "linear")
    plt.ylim(0, y_max * 1.05)
    plt.xlim(x_min, x_max)
    plt.grid(True, which='both', linestyle='--', alpha=0.1)
    plt.legend()

    if verbose:
        param_dict = vars(self.params)
        main_parameters = [
            (r"$r_d$", param_dict["domain_radius"], "μm"),
            (r"$R_n$", param_dict["nucleus_radius"], "μm"),
        ]
    
        if param_dict.get("z0") is not None:
            main_parameters.append((r"$z_0$", param_dict["z0"], "Gy"))
        if param_dict.get("beta0") is not None:
            main_parameters.append((r"$\beta_0$", param_dict["beta0"], "Gy⁻²"))
            
        model_version = f"Model: {self.model_version}"
        info_lines = [model_version] + [f"{k}: {v} {unit}" for k, v, unit in main_parameters]
        info_text = "\n".join(info_lines)
        ax = plt.gca()
        ax.text(0.05, 0.05, info_text, transform=ax.transAxes,
                fontsize=14, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round'))
    plt.tight_layout()    
    plt.show(block=False)
    plt.pause(0.1)

MKTable.plot = plot
