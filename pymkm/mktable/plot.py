"""
Plotting utilities for MKTable results.

This module defines the `plot()` method for the MKTable class, which allows visualization
of microdosimetric quantities such as z̄*, z̄_d, and z̄_n as functions of energy or LET
for different ions.

Plots include optional display of model configuration and geometry.
"""

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
    "legend.fontsize": 14,
    "axes.titlesize": 16
})
import numpy as np

from .core import MKTable

def _validate_plot_columns(x: str, y: str):
    """
    Validate that x and y are valid MKTable data columns.
    
    :param x: Name of the x-axis variable ('energy' or 'let').
    :type x: str
    :param y: Name of the y-axis variable ('z_bar_star_domain', 'z_bar_domain', or 'z_bar_nucleus').
    :type y: str
    
    :raises ValueError: If x or y are not among the allowed options.
    """
    allowed_x = {"energy", "let"}
    allowed_y = {"z_bar_star_domain", "z_bar_domain", "z_bar_nucleus"}
    if x not in allowed_x:
        raise ValueError(f"Invalid x-axis: '{x}'. Allowed values are: {sorted(allowed_x)}")
    if y not in allowed_y:
        raise ValueError(f"Invalid y-axis: '{y}'. Allowed values are: {sorted(allowed_y)}")


def plot(
    self: MKTable,
    ions: Optional[List[Union[str, int]]] = None,
    *,
    x: str = "energy",
    y: str = "z_bar_star_domain",
    verbose: bool = False,
    ax: Optional[plt.Axes] = None,
    show: Optional[bool] = True
):
    """
    Plot microdosimetric quantities from the MKTable.

    :param ions: List of ions to plot. If None, all computed ions are used.
    :type ions: list[str or int], optional
    :param x: x-axis variable ('energy' or 'let').
    :type x: str
    :param y: y-axis variable ('z_bar_star_domain', 'z_bar_domain', 'z_bar_nucleus').
    :type y: str
    :param verbose: Show model configuration in the plot.
    :type verbose: bool
    :param ax: Matplotlib Axes object to draw on. If None, a new figure is created.
    :type ax: Optional[matplotlib.axes.Axes]
    :param show: If True, displays the plot. Set False when embedding or scripting.
    :type show: Optional[bool]

    :raises RuntimeError: If table is empty.
    :raises ValueError: If x or y are invalid.
    """
    if not self.table:
        raise RuntimeError("No computed results found. Run `compute()` before plotting.")

    ions = ions or self.sp_table_set.get_available_ions()
    ions = [self.sp_table_set._map_to_fullname(ion) for ion in ions]
    _validate_plot_columns(x, y)

    x_label_map = {"energy": "Energy [MeV/u]", "let": "LET [MeV/cm]"}
    y_label_map = {
        "z_bar_star_domain": r"$\bar{z}^{*}$ [Gy]",
        "z_bar_domain": r"$\bar{z}_d$ [Gy]",
        "z_bar_nucleus": r"$\bar{z}_n$ [Gy]"
    }

    # Create figure/axes if not provided
    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
        created_fig = True

    x_min, x_max, y_max = np.inf, -np.inf, -np.inf
    for ion in ions:
        df = self.table[ion]["data"]
        x_vals = df[x].values
        y_vals = df[y].values
        x_min = min(x_min, x_vals.min())
        x_max = max(x_max, x_vals.max())
        y_max = max(y_max, y_vals.max())

    for ion in ions:
        df = self.table[ion]["data"]
        ion_symbol = self.sp_table_set.get(ion).ion_symbol
        color = self.sp_table_set.get(ion).color
        ax.plot(df[x], df[y], label=ion_symbol, color=color, alpha=0.5, linewidth=6)

    plot_title = f"Source: {self.sp_table_set.source_info}, Track model: {self.params.model_name} (Core: {self.params.core_radius_type})"

    ax.set_xlabel(x_label_map.get(x, x.capitalize()))
    ax.set_ylabel(y_label_map.get(y, y.replace('_', ' ').capitalize()))
    ax.set_title(plot_title, wrap=True)
    ax.set_xscale("log" if x == "energy" else "linear")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max * 1.05)
    ax.grid(True, which='both', linestyle='--', alpha=0.1)
    ax.legend()

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

        info_lines = [f"Model: {self.model_version}"] + [
            f"{k}: {v:.3f} {unit}" for k, v, unit in main_parameters
        ]
        info_text = "\n".join(info_lines)

        ax.text(0.05, 0.05, info_text, transform=ax.transAxes,
                fontsize=14, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round'))

    if show and created_fig:
        plt.tight_layout()
        plt.show()


MKTable.plot = plot
