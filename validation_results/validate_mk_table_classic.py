import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Access to local modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from validation_utils.loader import load_validation_file
from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.io.table_set import StoppingPowerTableSet

def validate_mk_table_classic(source: str = "fluka_2020_0"):
    """
    Validate the dose-mean saturation-corrected specific energy z̄* computed by MKTable
    against reference datasets from Inaniwa et al. (2010), for various ions and energies.
    """
    data_dir = Path(__file__).resolve().parent / "mk_table_classic" / "reference_data"
    figure_dir = Path(__file__).resolve().parent / "mk_table_classic" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.txt"))

    for file in files:
        metadata, df_ref = load_validation_file(file)

        label = f"{metadata.get('Reference', 'Unknown')}"
        atomic_number = int(metadata["Atomic_Number"])
        model_name = metadata["Model_Name"]
        core_type = metadata["Core_Radius_Type"]
        domain_radius = float(metadata["Domain_Radius_um"])
        nucleus_radius = float(metadata["Nucleus_Radius_um"])
        beta0 = float(metadata["Beta0_Gy-2"])
        
        title = f"Source: {source}, Track model: {model_name} (Core: {core_type})"

        
        print(f"\nValidating ion Z = {atomic_number} using source '{source}'...")
        sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions([atomic_number])

        params = MKTableParameters(
            domain_radius=domain_radius,
            nucleus_radius=nucleus_radius,
            beta0=beta0,
            model_name=model_name,
            core_radius_type=core_type,
        )

        mk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
        mk_table.compute(ions=[atomic_number])

        # Plot model result using built-in method
        fig, ax = plt.subplots(figsize=(10, 6))
        mk_table.plot(
            ions=[atomic_number],
            x="energy",
            y="z_bar_star_domain",
            verbose=True,
            ax=ax,
            show=False
            )

        ax.plot(
            df_ref['x'], df_ref['y'],
            marker='o', linestyle='None',
            markersize=9, markeredgewidth=0.8,
            markeredgecolor='black', alpha=0.7,
            label=label,
            color=sp_table_set.get(atomic_number).color
        )
        
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=12)
        fig.tight_layout()
        
        plot_file = figure_dir / f"z_bar_star_domain_Z{atomic_number}_{source}.png"
        fig.savefig(plot_file, dpi=300)
        plt.close(fig)
                
if __name__ == "__main__":
    validate_mk_table_classic(source="mstar_3_12")
    # validate_mk_table_classic(source="fluka_2020_0")
    # validate_mk_table_classic(source="geant4_11_3_0")
