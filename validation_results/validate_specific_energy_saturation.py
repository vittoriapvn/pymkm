import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np

# Access to local modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from validation_utils.loader import load_validation_file
from validation_utils.layout import choose_horizontal_subplot_layout
from pymkm.physics.particle_track import ParticleTrack
from pymkm.physics.specific_energy import SpecificEnergy

def validate_specific_energy_saturation():
    """
    Validate the saturation-corrected specific energy z_sat(b) computed by pyMKM
    against reference datasets for various ion species and energies.

    Expected metadata in file header:
    - Energy_MeV_u
    - Atomic_Number
    - LET_MeV_cm
    - Model_Name
    - Core_Radius_Type
    - Domain_Radius_um        (used as region radius)
    - Nucleus_Radius_um
    - Beta0
    - Saturation_Model        ("square_root" or "quadratic")
    """
    data_dir = Path(__file__).resolve().parent / "specific_energy_saturation" / "reference_data"
    figure_dir = Path(__file__).resolve().parent / "specific_energy_saturation" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.txt"))
    layouts = choose_horizontal_subplot_layout(len(files), max_cols_per_fig=3)

    file_idx = 0
    for fig_idx, (n_rows, n_cols) in enumerate(layouts):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4))
        axes = np.atleast_1d(axes).flatten()

        for i in range(n_cols):
            if file_idx >= len(files):
                break

            file = files[file_idx]
            metadata, df = load_validation_file(file)
            file_idx += 1

            # --- Metadata extraction ---
            label = f"{metadata.get('Reference', 'Unknown')}"
            energy = float(metadata["Energy_MeV_u"])
            atomic_number = int(metadata["Atomic_Number"])
            let = float(metadata["LET_MeV_cm"])
            model_name = metadata["Model_Name"]
            core_type = metadata["Core_Radius_Type"]
            domain_radius = float(metadata["Domain_Radius_um"])
            nucleus_radius = float(metadata["Nucleus_Radius_um"])
            beta0 = float(metadata["Beta0_Gy-2"])
            model_label = metadata.get("Saturation_Model")
            
            title = f"Track model: {model_name} (Core: {core_type})"

            # --- Model setup ---
            pt = ParticleTrack(model_name=model_name,
                               core_radius_type=core_type,
                               energy=energy,
                               atomic_number=atomic_number,
                               let=let)
            se = SpecificEnergy(pt, region_radius=domain_radius)
            z0 = SpecificEnergy.compute_saturation_parameter(domain_radius, nucleus_radius, beta0)

            # --- Compute z(b) base profile ---
            z_array, b_array, elapsed = se.single_event_specific_energy(parallel=False, return_time=True)

            # --- Compute z_sat(b) corrected profile ---
            model_z = se.saturation_corrected_single_event_specific_energy(
                z0=z0,
                z_array=z_array,
                model=model_label
            )

            model_b = b_array
            print(f"Execution time: {elapsed:.4f} seconds")

            # --- Plotting ---
            ax = axes[i] if n_cols > 1 else axes[0]
            ax.loglog(df['x'], df['y'], alpha=0.5, linewidth=5, color='gray', label=label)
            ax.loglog(model_b, model_z, alpha=0.8, linewidth=2.5, linestyle='--', color='black', label='pyMKM')

            label = f"{metadata.get('Reference', 'Unknown')} ({model_label})"
            data_units = metadata.get("Data_Units").split(',')
            x_label = data_units[0].strip()
            y_label = data_units[1].strip() if len(data_units) > 1 else ""

            info_text = (
                f"Z: {atomic_number}\n"
                f"E: {energy} MeV/u\n"
                f"LET: {let} MeV/cm\n"
                f"$r_d$: {domain_radius} μm\n"
                f"$R_n$: {nucleus_radius} μm\n"
                f"β₀: {beta0} Gy⁻²"
            )

            ax.set_title(title, fontsize=10)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, which='both', linestyle='--', alpha=0.1)
            ax.text(0.05, 0.05, info_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round'))
            ax.legend(fontsize=10)

            # Axis limits
            ax.set_xlim([model_b.min(), model_b.max() * 1.2])
            ax.set_ylim([min(model_z[model_z > 0].min(), df['y'].min()) * 0.02,
                         max(model_z.max(), df['y'].max()) * 20])

        plt.tight_layout()
        fig.savefig(figure_dir / f"specific_energy_saturation_subplot_{fig_idx+1}.png", dpi=300)
        plt.show(block=False)
        plt.pause(0.1)

if __name__ == "__main__":
    validate_specific_energy_saturation()