import pytest
import numpy as np
import matplotlib.pyplot as plt

from pymkm.io.stopping_power import StoppingPowerTable

def test_valid_construction_with_symbol():
    energy = np.linspace(0.1, 1000, 150)
    let = np.linspace(1.0, 2.0, 150)
    table = StoppingPowerTable("C", energy, let)
    assert table.ion_name == "C"
    assert table.energy.shape == (150,)
    assert table.let.shape == (150,)

def test_valid_construction_with_fullname():
    energy = np.linspace(0.1, 1000, 150)
    let = np.linspace(1.0, 2.0, 150)
    table = StoppingPowerTable("Carbon", energy, let)
    assert table.ion_name == "C"

def test_valid_construction_with_atomic_number():
    energy = np.linspace(0.1, 1000, 150)
    let = np.linspace(1.0, 2.0, 150)
    for input_value in [6, "6"]:
        table = StoppingPowerTable(input_value, energy, let)
        assert table.ion_name == "C"
        assert table.atomic_number == 6

def test_invalid_atomic_number():
    energy = np.linspace(0.1, 1000, 150)
    let = np.linspace(1.0, 2.0, 150)
    with pytest.raises(ValueError):
        StoppingPowerTable(999, energy, let)

def test_invalid_shape():
    energy = np.linspace(1, 100, 150)
    let = np.linspace(1, 2, 149)
    with pytest.raises(ValueError):
        StoppingPowerTable("C", energy, let)

def test_too_few_points():
    energy = np.linspace(1, 10, 100)
    let = np.linspace(1, 2, 100)
    with pytest.raises(ValueError):
        StoppingPowerTable("C", energy, let)

def test_non_increasing_energy():
    energy = np.array([1, 2, 3] + [2] * 147)
    let = np.linspace(1, 2, 150)
    with pytest.raises(ValueError):
        StoppingPowerTable("C", energy, let)

def test_unknown_ion_identifier():
    energy = np.linspace(1, 100, 150)
    let = np.linspace(1, 2, 150)
    with pytest.raises(ValueError):
        StoppingPowerTable("Unobtainium", energy, let)

def test_to_dict_and_from_dict():
    energy = np.linspace(1, 100, 150)
    let = np.linspace(1, 2, 150)
    table = StoppingPowerTable("C", energy, let)
    data = table.to_dict()
    data["target"] = "WATER_LIQUID"
    new_table = StoppingPowerTable.from_dict(data)
    assert new_table.ion_name == table.ion_name
    np.testing.assert_allclose(new_table.energy, table.energy)

def test_from_dict_raises_on_missing_fields():
    # Create a dict missing several required fields
    incomplete_data = {
        "atomic_number": 6,
        "mass_number": 12,
        "energy": [1.0] * 150,
        "let": [0.1] * 150,
        "source_program": "dummy"
        # Missing: ion_symbol, ionization_potential, target
    }

    with pytest.raises(ValueError, match="Missing required field\\(s\\) in dictionary: ion_symbol, ionization_potential, target"):
        _ = StoppingPowerTable.from_dict(incomplete_data)

def test_plot_runs(monkeypatch):
    energy = np.linspace(1, 100, 150)
    let = np.linspace(1, 2, 150)
    table = StoppingPowerTable("C", energy, let)
    show_called = False

    def fake_show():
        nonlocal show_called
        show_called = True

    monkeypatch.setattr(plt, "show", fake_show)
    table.plot(show=True)
    assert show_called is True

def test_plot_title_new_figure(monkeypatch):
    energy = np.linspace(1, 100, 150)
    let = np.linspace(1, 2, 150)
    table = StoppingPowerTable("C", energy, let)
    plt.close("all")
    monkeypatch.setattr(plt, "show", lambda: None)
    table.plot(show=False)
    title = plt.gca().get_title()
    expected = f"{table.ion_name}: Stopping Power vs Energy"
    assert title == expected

def test_plot_title_existing_figure(monkeypatch):
    energy = np.linspace(1, 100, 150)
    let = np.linspace(1, 2, 150)
    table = StoppingPowerTable("C", energy, let)
    plt.close("all")
    plt.figure(figsize=(8,5))
    ax = plt.gca()
    ax.set_title("")
    monkeypatch.setattr(plt, "show", lambda: None)
    table.plot(show=False, ax=ax)
    title = ax.get_title()
    expected = "Stopping Power vs Energy"
    assert title == expected

def test_resample_and_interpolate():
    energy = np.logspace(0, 3, 150)
    let = energy ** -0.5
    table = StoppingPowerTable("C", energy, let)

    # Interpolate LET at a few known points
    query = np.array([1.0, 10.0, 100.0])
    expected = query ** -0.5
    result = table.interpolate(energy=query, loglog=False)
    np.testing.assert_allclose(result, expected, rtol=5e-4)

    # Now test resampling
    new_grid = np.logspace(1, 2, 50)
    table.resample(new_grid)
    assert np.allclose(table.energy, new_grid)

def test_from_txt_valid_complete(tmp_path):
    f = tmp_path / "valid.txt"
    header = """Ion=C
AtomicNumber=6
MassNumber=12
SourceProgram=mock
IonizationPotential=75.0
Target=WATER_LIQUID"""
    data_lines = "\n".join(f"{1.0 + i*0.1:.2f} {10.0 - i*0.01:.2f}" for i in range(150))
    content = f"{header}\n{data_lines}\n"
    f.write_text(content)
    table = StoppingPowerTable.from_txt(str(f))
    assert table.ion_symbol == "C"
    assert table.atomic_number == 6
    assert table.mass_number == 12
    assert table.ionization_potential == 75.0
    assert len(table.energy) == 150
    assert len(table.let) == 150

def test_from_txt_missing_target(tmp_path):
    f = tmp_path / "missing.txt"
    f.write_text("""Ion=H
AtomicNumber=1
MassNumber=1
SourceProgram=mock
IonizationPotential=75.0
1.0 10.0
1.1 9.0
""")
    with pytest.raises(ValueError, match="Missing required header"):
        StoppingPowerTable.from_txt(str(f))

def test_from_txt_invalid_target(tmp_path):
    f = tmp_path / "invalid.txt"
    f.write_text("""Ion=H
AtomicNumber=1
MassNumber=1
SourceProgram=mock
IonizationPotential=75.0
Target=AIR
1.0 10.0
1.1 9.0
""")
    with pytest.raises(ValueError, match="Unsupported target"):
        StoppingPowerTable.from_txt(str(f))

def test_from_txt_raises_on_unknown_ion(tmp_path):
    file = tmp_path / "bad_ion.txt"
    file.write_text(
        "SourceProgram=fluka\n"
        "Ion=Unobtanium\n"
        "AtomicNumber=999\n"
        "MassNumber=999\n"
        "IonizationPotential=10\n"
        "Target=WATER_LIQUID\n"
        "E [MeV/u]\tdEdx [MeV/cm]\n"
        + "\n".join("1.0\t1.0" for _ in range(150))
    )
    with pytest.raises(ValueError, match="Ion symbol 'Unobtanium' is not recognized."):
        StoppingPowerTable.from_txt(str(file))

def test_from_txt_raises_on_mismatch_atomic_mass(tmp_path):
    file = tmp_path / "mismatch.txt"
    file.write_text(
        "SourceProgram=fluka\n"
        "Ion=C\n"
        "AtomicNumber=7\n"  # <-- mismatch!
        "MassNumber=12\n"
        "IonizationPotential=10\n"
        "Target=WATER_LIQUID\n"
        "E [MeV/u]\tdEdx [MeV/cm]\n"
        + "\n".join("1.0\t1.0" for _ in range(150))
    )
    with pytest.raises(ValueError, match="Mismatch in atomic or mass number for ion 'C'"):
        StoppingPowerTable.from_txt(str(file))

def test_energy_out_of_validated_range_warns():
    energy = np.linspace(0.01, 10, 150)  # min < 0.1
    let = np.linspace(1.0, 2.0, 150)
    with pytest.warns(UserWarning, match="outside validated range"):
        StoppingPowerTable("C", energy, let)

    energy = np.linspace(10, 2000, 150)  # max > 1000
    let = np.linspace(1.0, 2.0, 150)
    with pytest.warns(UserWarning, match="outside validated range"):
        StoppingPowerTable("C", energy, let)
        
def test_non_finite_energy_or_let_raises():
    # Caso con NaN in energy
    energy = np.linspace(0.1, 1000, 150)
    energy[10] = np.nan
    let = np.linspace(1.0, 2.0, 150)
    with pytest.raises(ValueError, match="finite values"):
        StoppingPowerTable("C", energy, let)

    # Caso con inf in LET
    energy = np.linspace(0.1, 1000, 150)
    let = np.linspace(1.0, 2.0, 150)
    let[20] = np.inf
    with pytest.raises(ValueError, match="finite values"):
        StoppingPowerTable("C", energy, let)