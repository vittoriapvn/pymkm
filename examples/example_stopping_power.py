import numpy as np
import matplotlib.pyplot as plt
from pymkm.io.stopping_power import StoppingPowerTable
from pymkm.io.data_registry import get_default_txt_path

# Load stopping power table (Carbon ion from mstar_3_12)
filepath = get_default_txt_path("mstar_3_12", "Z06_A12_Carbon.txt")
spt = StoppingPowerTable.from_txt(filepath)

# Interpolate LET from energy (forward)
min_energy = max(spt.energy.min(), 1e-2)
max_energy = spt.energy.max()
energy_range = np.logspace(np.log10(min_energy), np.log10(max_energy), 6)
let_values = spt.interpolate(energy=energy_range)

# Plot the stopping power curve
spt.plot(label=f"Z={spt.atomic_number} ({spt.source_program})", show=False)

# Overlay interpolated points
plt.scatter(
    energy_range, let_values,
    facecolors='none', edgecolors='black', s=80,
    label="Interpolated LET points"
)

# Inverse interpolation: energy from LET
#let_query = np.array([7000, 8000, 4000])
min_let = max(spt.let.min(), 1e-2)
max_let = spt.let.max()
let_query = np.linspace(min_let, max_let * 0.95, 3)
inverse_interp = spt.interpolate(let=let_query)

# Use different markers and colors for each LET
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', '*', 'X']
colors = plt.cm.tab10.colors

for i, (let_val, energies) in enumerate(inverse_interp.items()):
    marker = markers[i % len(markers)]
    color = colors[i % len(colors)]
    plt.scatter(
        energies, np.full_like(energies, let_val),
        marker=marker, color=color, s=80,
        label=f"LET = {let_val:.0f}"
    )

# Deduplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys())

plt.xlabel("Energy [MeV/u]")
plt.ylabel("LET [MeV/cm]")
plt.title("LET vs Energy with Interpolated and Inverse Points")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print interpolation results
print("Interpolated LET values:")
for e, l in zip(energy_range, let_values):
    print(f"Energy = {e:.2f} MeV/u -> LET = {l:.4f} MeV/cm")

print("\nInverse interpolation (LET → Energy):")
for let_val, energies in inverse_interp.items():
    energy_str = ", ".join(f"{e:.4f}" for e in energies)
    print(f"LET = {let_val:.0f} MeV/cm -> Energies = {energy_str} MeV/u")
