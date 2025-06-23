import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the parent directory to sys.path to access validation_utils and physics modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from validation_utils.loader import load_validation_file
from validation_utils.layout import choose_horizontal_subplot_layout
from pymkm.physics.particle_track import ParticleTrack

def validate_initial_local_dose():
    """
    Validate the initial radial dose distribution D(r) computed by pyMKM
    against reference datasets for various ion species and energies.

    This script loads validation files containing radial dose profiles,
    compares them with the output of pyMKM's `ParticleTrack.initial_local_dose()`,
    and generates log-log plots for visual inspection.

    Each validation file must contain:
    - A metadata header with keys such as:
        * Energy_MeV_u: ion energy per nucleon [MeV/u]
        * Atomic_Number: atomic number Z
        * LET_MeV_cm: unrestricted LET [MeV/cm]
        * Model_Name: radial dose model (e.g., 'Kiefer-Chatterjee' or 'Scholz-Kraft')
        * Core_Radius_Type: 'constant' or 'energy-dependent'
    - Tabular data with columns:
        * x: radial distance from the ion track [µm]
        * y: reference dose [Gy]

    Generated figures are saved in:
    - ./initial_local_dose/figures/
    """
    data_dir = Path(__file__).resolve().parent / "initial_local_dose" / "reference_data"
    figure_dir = Path(__file__).resolve().parent / "initial_local_dose" / "figures"
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

            # Extract metadata for labels and model configuration
            label = f"{metadata.get('Reference', 'Unknown')}"
            data_units = metadata.get("Data_Units").split(',')
            x_label = data_units[0].strip()
            y_label = data_units[1].strip() if len(data_units) > 1 else ""

            # Physical parameters for the ParticleTrack
            energy = float(metadata.get("Energy_MeV_u"))
            atomic_number = int(metadata.get("Atomic_Number"))
            let = float(metadata.get("LET_MeV_cm"))
            model_name = metadata.get("Model_Name")
            core_type = metadata.get("Core_Radius_Type")
            
            title = f"Track model: {model_name} (Core: {core_type})"

            # Instantiate the ParticleTrack model and compute D(r)
            pt = ParticleTrack(model_name=model_name,
                               core_radius_type=core_type,
                               energy=energy,
                               atomic_number=atomic_number,
                               let=let)
            model_dose, model_radii = pt.initial_local_dose()

            # Display relevant parameters on the plot
            info_text = (
                f"Z: {atomic_number}\n"
                f"E: {energy} MeV/u\n"
                f"LET: {let} MeV/cm"
            )

            # Plot the reference and model radial dose
            ax = axes[i] if n_cols > 1 else axes[0]
            ax.loglog(df['x'], df['y'], alpha=0.4, linewidth=6, color='gray', label=label)
            ax.loglog(model_radii, model_dose, alpha=0.7, linewidth=3, linestyle='--', color='black', label='pyMKM')
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, which='both', linestyle='--', alpha=0.1)
            ax.text(0.05, 0.05, info_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round'))
            ax.legend(fontsize=10)

            # Set axes limits based on model output
            ax.set_xlim([model_radii.min(), model_radii.max() * 1.2])
            ax.set_ylim([model_dose[model_dose > 0].min() * 0.02, model_dose.max() * 20])

        plt.tight_layout()
        fig.savefig(figure_dir / f"initial_local_dose_subplot_{fig_idx+1}.png", dpi=300)
        plt.show(block=False)
        plt.pause(0.1)

if __name__ == "__main__":
    validate_initial_local_dose()
