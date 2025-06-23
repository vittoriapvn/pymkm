#!/usr/bin/env python
"""
Example usage of StoppingPowerTableSet using real Fluka stopping power tables.

This script demonstrates:
  • Loading stopping power tables from the default Fluka source ("fluka_2020_0").
  • Displaying the total number of loaded tables and the available ion keys.
  • Showing how filtering works using either a symbol or full name.
  • Retrieving a table and displaying key properties (energy grid and LET values).
  • Resampling and interpolating the tables on a new energy grid.
  • Serializing the table set to a JSON file and loading it back.
  • Plotting the stopping power curves (both in a single plot and separately).
"""

import numpy as np
import matplotlib.pyplot as plt

from pymkm.io.table_set import StoppingPowerTableSet

def main():
    # --- Load Tables from the Default Fluka Source ---
    source = "fluka_2020_0"
    print(f"Loading stopping power tables from default Fluka source '{source}'...")
    table_set = StoppingPowerTableSet.from_default_source(source)
    print(f"Total number of loaded tables: {len(table_set)}")
    
    # Display available ions (the keys are the full names as inferred from the lookup)
    available_ions = table_set.get_available_ions()
    print("Available ions:")
    for ion in available_ions:
        print(f"  - {ion}")

    # --- Filtering Examples ---
    # Filter by full name (e.g. "Carbon") and by symbol (e.g. "O" should be resolved to "Oxygen")
    filtered_full = table_set.filter_by_ions(["Carbon"])
    filtered_symbol = table_set.filter_by_ions(["O"])  # Assuming "O" is the symbol for Oxygen
    print("\nFiltered table set by full name 'Carbon':", filtered_full.get_available_ions())
    print("Filtered table set by symbol 'O':", filtered_symbol.get_available_ions())
    
    # --- Common Energy Range ---
    common_range = table_set.get_common_energy_range()
    if common_range:
        print(f"\nCommon energy range across loaded tables: {common_range[0]:.2f} - {common_range[1]:.2f} MeV/u")
    else:
        print("\nNo common energy range found.")
    
    # --- Retrieve and Display a Specific Table ---
    # Let's work with the Carbon table.
    carbon_table = table_set.get("C")  # Using symbol; should resolve to "Carbon"
    if carbon_table is not None:
        print("\nCarbon table retrieved:")
        print("  - First 5 energy grid values:", carbon_table.energy_grid[:5])
        print("  - First 5 stopping power values:", carbon_table.stopping_power[:5])
    else:
        print("\nCarbon table not found.")

    # --- Resample and Interpolate ---
    new_grid = np.logspace(0, 3, 200)  # New energy grid: 1 to 1000 MeV/u with 200 points.
    table_set.resample_all(new_grid)
    print("\nResampled all tables on a new energy grid.")

    # Interpolate stopping power values at specific energies.
    energies_to_interp = np.array([10, 50, 100, 500])
    interpolated_values = table_set.interpolate_all(energy=energies_to_interp)
    print("Interpolated stopping power values:")
    for ion, values in interpolated_values.items():
        print(f"  {ion}: {values}")

    # --- Serialization ---
    json_filename = "fluka_stopping_power_tables.json"
    table_set.save(json_filename)
    print(f"\nSaved table set to JSON file: {json_filename}")
    reloaded_set = StoppingPowerTableSet.load(json_filename)
    print("Reloaded table set from JSON. Available ions:", reloaded_set.get_available_ions())

    # --- Plotting ---
    # Plot all tables in a single figure.
    print("\nPlotting all stopping power curves in a single figure...")
    reloaded_set.plot(single_plot=True)
    plt.show()
    
    # Plot each table in separate figures.
    print("Plotting each stopping power curve in separate figures...")
    table_set.plot(single_plot=False)
    plt.show()


if __name__ == "__main__":
    main()
