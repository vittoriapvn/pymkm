import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import defaultdict

# Extend module search path to allow relative imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from validation_utils.layout import choose_horizontal_subplot_layout
from validation_utils.loader import load_validation_file
from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.sftable.core import SFTableParameters, SFTable
from pymkm.io.table_set import StoppingPowerTableSet

def validate_sf_table_stochastic(source: str = "fluka_2020_0"):
    """
    Validate survival fraction curves computed by SFTable (SMK mode)
    against reference datasets for different cell lines and ion species.
    """
    base_dir = Path(__file__).resolve().parent / "sf_table_stochastic"
    ref_dir = base_dir / "reference_data"
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for cell_line in ["HSG", "V79"]:
        folder = ref_dir / cell_line
        grouped_by_Z = defaultdict(list)

        for txt_file in sorted(folder.glob("*.txt")):
            metadata, df_ref = load_validation_file(txt_file)
            Z = int(metadata["Atomic_Number"])
            let_val = float(metadata["LET_MeV_cm"])
            grouped_by_Z[Z].append((let_val, txt_file, metadata, df_ref))

        for Z in sorted(grouped_by_Z):
            entries = grouped_by_Z[Z]
            entries.sort(key=lambda x: x[0])
            layout_list = choose_horizontal_subplot_layout(len(entries), max_cols_per_fig=3)
            entry_idx = 0

            let_val, _, metadata, _ = entries[0]
            cell_type = metadata["Cell_Type"]
            domain_radius = float(metadata["Domain_Radius_um"])
            nucleus_radius = float(metadata["Nucleus_Radius_um"])
            z0 = float(metadata["z0_Gy"])
            beta0 = float(metadata["Beta0_Gy-2"])
            model_name = metadata["Model_Name"]
            core_type = metadata["Core_Radius_Type"]
            atomic_number = int(metadata["Atomic_Number"])

            sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions([Z])
            mk_params = MKTableParameters(
                domain_radius=domain_radius,
                nucleus_radius=nucleus_radius,
                z0=z0,
                beta0=beta0,
                model_name=model_name,
                core_radius_type=core_type,
                use_stochastic_model=True
            )
            mk_table = MKTable(parameters=mk_params, sp_table_set=sp_table_set)

            for fig_idx, (n_rows, n_cols) in enumerate(layout_list):
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
                axs = axs[0]

                for ax_idx, ax in enumerate(axs):
                    if entry_idx >= len(entries):
                        ax.axis("off")
                        continue

                    let_val, txt_file, metadata, df_ref = entries[entry_idx]
                    entry_idx += 1

                    alpha0 = float(metadata["Alpha0_Gy-1"])
                    dose_grid = df_ref["x"].values
                    sf_params = SFTableParameters(mktable=mk_table, alpha0=alpha0, beta0=beta0, dose_grid=dose_grid)
                    sf_table = SFTable(parameters=sf_params)
                    sf_table.compute(ion=atomic_number, let=let_val, force_recompute=True)

                    ref_label = metadata.get("Reference", "Reference")
                    color = sp_table_set.get(atomic_number).color
                    ax.plot(dose_grid, df_ref["y"].values, '--', label=ref_label, linewidth=3, color=color)

                    ion_symbol = sp_table_set.get(atomic_number).ion_symbol
                    for idx, result in enumerate(sf_table.table):
                        model_suffix = "- MKM" if not mk_table.params.use_stochastic_model else "- SMK"
                        params = result["params"]
                        label = f"{ion_symbol} (pyMKM {model_suffix}, E = {params['energy']:.1f} MeV/u)"
                        line_style = '-' if idx == 0 else ':'
                        ax.plot(result["data"]["dose"], result["data"]["survival_fraction"], line_style, label=label, linewidth=6, alpha=0.4, color=color)

                    ax.set_xlim(0, 10)
                    ax.set_ylim(1e-4, 1)
                    ax.set_yscale("log")

                    x_label, y_label = map(str.strip, metadata.get("Data_Units", "Dose [Gy], Survival fraction").split(","))
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_title(f"LET = {let_val:.1f} MeV/cm", fontsize=14)

                    if ax_idx == 0:
                        info_text = (
                            f"{cell_type}\n"
                            f"$\\alpha_0$: {alpha0:.3f} Gy$^{{-1}}$\n"
                            f"$\\beta_0$: {beta0:.4f} Gy$^{{-2}}$\n"
                            f"$r_d$: {domain_radius:.2f} μm\n"
                            f"$R_n$: {nucleus_radius:.2f} μm\n"
                            f"$z_0$: {mk_table.params.z0:.2f} Gy"
                        )
                        ax.text(
                            0.05, 0.05, info_text, transform=ax.transAxes,
                            fontsize=14, verticalalignment='bottom', horizontalalignment='left',
                            bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round')
                        )

                    ax.grid(True, which='both', linestyle='--', alpha=0.3)
                    ax.legend(loc='upper right', fontsize=12)

                title = f"Source: {source}, Track model: {model_name} (Core: {core_type})"
                fig.suptitle(title, fontsize=16)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                fig_path = fig_dir / f"{cell_line}_Z{Z}_{source}.png"
                fig.savefig(fig_path, dpi=300)
                plt.pause(0.1)

if __name__ == "__main__":
    # validate_sf_table_stochastic(source="fluka_2020_0")
    # validate_sf_table_stochastic(source="geant4_11_3_0")
    validate_sf_table_stochastic(source="mstar_3_12")
