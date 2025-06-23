import subprocess
from pathlib import Path

# Ordered list of validation scripts
script_order = [
    "validate_initial_local_dose.py",
    "validate_specific_energy.py",
    "validate_specific_energy_saturation.py",
    "validate_mk_table_classic.py",
    "validate_mk_table_classic_dose.py",
    "validate_mk_table_stochastic.py",
    "validate_sf_table_classic.py",
    "validate_sf_table_stochastic.py"
]

# Validation script directory
validation_dir = Path(__file__).resolve().parent

print("Starting execution of validation scripts in required order:\n")

for script_name in script_order:
    script_path = validation_dir / script_name
    if not script_path.exists():
        print(f"⚠️ Script not found: {script_name}\n")
        continue

    print(f"==> Running: {script_name}")
    result = subprocess.run(["python", str(script_path)], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Completed: {script_name}\n")
    else:
        print(f"❌ Error in {script_name}:")
        print(result.stderr)
        print()

print("All validation scripts have been processed.")
