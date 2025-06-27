import numpy as np
import matplotlib.pyplot as plt
from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.io.table_set import StoppingPowerTableSet

def main():

    # Select input parameters for rbe tables generation
    atomic_numbers = [2, 6, 8] # He, C, O
    source = "fluka_2020_0" # Source code used to generate stopping power tables (available with pymkm: fluka_2020_0, geant4_11_3_0 or mstar_3_12)
    model_name = "Kiefer-Chatterjee" # Amorphous track structure model (Kiefer-Chatterjee or Scholz-Kraft)
    core_type = "energy-dependent" # Core radius model ('constant' or 'energy-dependent')
    domain_radius = 0.32 # μm
    nucleus_radius = 3.9 # μm
    beta0 = 0.0615 # 1/Gy^2

    title = f"Source: {source}, Track model: {model_name} (Core: {core_type})"

    print(f"\nGenerating stopping power tables for ion Z = {atomic_numbers} (using source '{source}')...")
    sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions(atomic_numbers)

    # Store input parameters
    params = MKTableParameters(
        domain_radius=domain_radius,
        nucleus_radius=nucleus_radius,
        beta0=beta0,
        model_name=model_name,
        core_radius_type=core_type,
    )

    # Generate rbe table
    print(f"\nGenerating MKM tables for ion Z = {atomic_numbers} (using source '{source}')...")
    mk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
    mk_table.compute(ions=atomic_numbers, parallel=True)

    # Plot model result using built-in method
    mk_table.plot(ions=atomic_numbers, x="energy", y="z_bar_star_domain", verbose=True)

    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.pause(0.1)

    # Write to .txt file 
    path = "./MKM_table.txt"
    params = {
        "CellType": "T",
        "Alpha_0": 0.12,
        "Beta": 0.0615
    }
    mk_table.write_txt(params=params, filename=path, max_atomic_number=8)


if __name__ == "__main__":
    main()
