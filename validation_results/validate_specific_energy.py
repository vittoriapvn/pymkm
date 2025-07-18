import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Access to local modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from validation_utils.loader import load_validation_file
from validation_utils.layout import choose_horizontal_subplot_layout
from pymkm.physics.particle_track import ParticleTrack
from pymkm.physics.specific_energy import SpecificEnergy

def validate_specific_energy():
    """
    Validate the single-event specific energy z(b) computed by pyMKM
    against reference datasets for various ion species and energies.

    This script loads validation files containing z(b) values,
    compares them with the output of pyMKM's `SpecificEnergy.single_event_specific_energy()`,
    and generates plots for visual inspection.

    Expected data:
    - ./specific_energy/reference_data/*.txt
    - Tabular data with columns: x (impact parameter [μm]), y (z(b) [Gy])
    - Metadata header with:
        * Energy_MeV_u
        * Atomic_Number
        * LET_MeV_cm
        * Model_Name
        * Core_Radius_Type
        * Region_Radius_um (mandatory)
    """
    data_dir = Path(__file__).resolve().parent / "specific_energy" / "reference_data"
    figure_dir = Path(__file__).resolve().parent / "specific_energy" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.txt"))
    layouts = choose_horizontal_subplot_layout(len(files), max_cols_per_fig=3)

    file_idx = 0
    for fig_idx, (n_rows, n_cols) in enumerate(layouts):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4))
        import numpy as np
        axes = np.atleast_1d(axes).flatten()

        for i in range(n_cols):
            if file_idx >= len(files):
                break

            file = files[file_idx]
            metadata, df = load_validation_file(file)
            file_idx += 1

            # Metadata
            label = f"{metadata.get('Reference', 'Unknown')}"
            energy = float(metadata["Energy_MeV_u"])
            atomic_number = int(metadata["Atomic_Number"])
            let = float(metadata["LET_MeV_cm"])
            model_name = metadata["Model_Name"]
            core_type = metadata["Core_Radius_Type"]
            region_radius = float(metadata["Region_Radius_um"])
            
            title = f"Track model: {model_name} (Core: {core_type})"

            # Setup models
            pt = ParticleTrack(model_name=model_name,
                               core_radius_type=core_type,
                               energy=energy,
                               atomic_number=atomic_number,
                               let=let)
            se = SpecificEnergy(pt, region_radius)

            # Compute model prediction
            model_z, model_b, elapsed = se.single_event_specific_energy(parallel=False, return_time=True)
            print(f"Execution time: {elapsed:.4f} seconds")

            # Plot
            ax = axes[i] if n_cols > 1 else axes[0]
            ax.loglog(df['x'], df['y'], alpha=0.5, linewidth=5, color='gray', label=label)
            ax.loglog(model_b, model_z, alpha=0.8, linewidth=2.5, linestyle='--', color='black', label='pyMKM')

            data_units = metadata.get("Data_Units").split(',')
            x_label = data_units[0].strip()
            y_label = data_units[1].strip() if len(data_units) > 1 else ""

            info_text = (
                f"Z: {atomic_number}\n"
                f"E: {energy} MeV/u\n"
                f"LET: {let} MeV/cm\n"
                f"R: {region_radius} μm"
            )

            ax.set_title(title, fontsize=10)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, which='both', linestyle='--', alpha=0.1)
            ax.text(0.05, 0.05, info_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round'))
            ax.legend(fontsize=10)

            # Dynamic axis limits
            ax.set_xlim([model_b.min(), model_b.max() * 1.2])
            ax.set_ylim([min(model_z[model_z > 0].min(), df['y'].min()) * 0.02,
                         max(model_z.max(), df['y'].max()) * 20])

        plt.tight_layout()
        fig.savefig(figure_dir / f"specific_energy_subplot_{fig_idx+1}.png", dpi=300)
        plt.show(block=False)
        plt.pause(0.1)

if __name__ == "__main__":
    validate_specific_energy()
