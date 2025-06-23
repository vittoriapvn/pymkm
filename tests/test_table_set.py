import json
import numpy as np
import matplotlib.pyplot as plt
import pytest

from pymkm.io import data_registry
from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.io.stopping_power import StoppingPowerTable

def create_dummy_table(ion_input: str, energy_grid: np.ndarray = None) -> StoppingPowerTable:
    if energy_grid is None:
        energy = np.logspace(0, 3, 150)
    else:
        energy = energy_grid
    let = 1e-3 * energy**(-0.5)
    table = StoppingPowerTable(
        ion_input=ion_input,
        energy=energy,
        let=let,
        mass_number=12,
        source_program="dummy",
        ionization_potential=10.0
    )
    table.color = "blue"
    table.target = "WATER_LIQUID"  # Required for serialization
    return table

# Loading
@pytest.fixture
def dummy_dict():
    table = create_dummy_table("Carbon")
    return {
        "Carbon": table.to_dict()
    }

def test_source_info_from_dict(dummy_dict):
    s = StoppingPowerTableSet.from_dict(dummy_dict)
    assert s.source_info == "dict"

def test_source_info_from_json(dummy_dict):
    json_str = json.dumps(dummy_dict)
    s = StoppingPowerTableSet.from_json(json_str)
    assert s.source_info == "json"

def test_source_info_from_load(tmp_path, dummy_dict):
    path = tmp_path / "tables.json"
    with open(path, "w") as f:
        json.dump(dummy_dict, f)
    s = StoppingPowerTableSet.load(str(path))
    assert s.source_info == f"loaded:{str(path)}"

def test_source_info_from_default(monkeypatch):
    # Preparazione dummy table
    table = create_dummy_table("Carbon")

    # Monkeypatch registry
    def fake_list_available_defaults(source):
        return ["C.txt"]

    def fake_get_default_txt_path(source, filename):
        return f"/fake/path/{filename}"

    monkeypatch.setattr(data_registry, "list_available_defaults", fake_list_available_defaults)
    monkeypatch.setattr(data_registry, "get_default_txt_path", fake_get_default_txt_path)

    monkeypatch.setattr(StoppingPowerTable, "from_txt", lambda path: table)

    s = StoppingPowerTableSet.from_default_source("fluka_2020_0")
    assert s.source_info == "default:fluka_2020_0"

# Basic operations
def test_add_remove_get_len():
    ts = StoppingPowerTableSet()
    table_c = create_dummy_table("C")
    table_o = create_dummy_table("O")
    ts.add("C", table_c)
    ts.add("O", table_o)
    assert len(ts) == 2
    assert ts.get("Carbon") == table_c
    assert ts["Oxygen"] == table_o
    assert ts.get("C") == table_c
    assert "C" in ts
    assert ts.get(6) == table_c
    assert ts.get("6") == table_c
    ts.remove("Carbon")
    assert len(ts) == 1
    assert ts.get("Carbon") is None

def test_add_overwrite():
    ts = StoppingPowerTableSet()
    table1 = create_dummy_table("C")
    table2 = create_dummy_table("C")
    ts.add("C", table1)
    ts.add("C", table2)
    assert ts.get("C") == table2

def test_remove_non_existing():
    ts = StoppingPowerTableSet()
    ts.remove("Unobtanium")
    assert len(ts) == 0

def test_get_invalid_input():
    ts = StoppingPowerTableSet()
    assert ts.get("UnknownIon") is None
    assert ts.get(999) is None

def test_getitem_unknown():
    ts = StoppingPowerTableSet()
    with pytest.raises(KeyError, match="No table found for ion: UnknownIon"):
        _ = ts["UnknownIon"]

def test_iter_keys_values_items():
    ts = StoppingPowerTableSet()
    table_c = create_dummy_table("C")
    table_o = create_dummy_table("O")
    ts.add("Carbon", table_c)
    ts.add("Oxygen", table_o)
    keys = list(ts.keys())
    values = list(ts.values())
    items = list(ts.items())
    assert set(keys) == {"Carbon", "Oxygen"}
    assert set(values) == {table_c, table_o}
    assert set(items) == {("Carbon", table_c), ("Oxygen", table_o)}
    for key, value in ts:
        assert key in {"Carbon", "Oxygen"}
        assert value in {table_c, table_o}

def test_from_default_source_handles_failure(monkeypatch, capsys):
    monkeypatch.setattr("pymkm.io.table_set.list_available_defaults", lambda source: ["bad.txt"])
    monkeypatch.setattr("pymkm.io.table_set.get_default_txt_path", lambda source, filename: "/dummy/path/bad.txt")

    def mock_from_txt(path):
        raise Exception("simulated error")

    monkeypatch.setattr("pymkm.io.stopping_power.StoppingPowerTable.from_txt", mock_from_txt)

    with pytest.raises(RuntimeError, match="Failed to load bad.txt from dummy_source: simulated error"):
        _ = StoppingPowerTableSet.from_default_source("dummy_source")

def test_from_directory_valid(tmp_path):
    file = tmp_path / "carbon.txt"
    energy_lines = "\n".join(f"{e:.5e}\t1.00000e+00" for e in np.linspace(1, 100, 150))
    content = (
        "SourceProgram=fluka\n"
        "Ion=C\n"
        "AtomicNumber=6\n"
        "MassNumber=12\n"
        "IonizationPotential=10\n"
        "Target=WATER_LIQUID\n"
        "E [MeV/u]\tdEdx [MeV/cm]\n" + energy_lines
    )
    file.write_text(content)
    ts = StoppingPowerTableSet.from_directory(str(tmp_path))
    assert "Carbon" in ts
    assert len(ts) == 1

def test_from_directory_inconsistent_source_program(tmp_path, capsys):
    energy_lines = "\n".join(f"{e:.5e}\t1.00000e+00" for e in np.linspace(1, 100, 150))
    (tmp_path / "c.txt").write_text(
        "SourceProgram=fluka\nIon=C\nAtomicNumber=6\nMassNumber=12\nIonizationPotential=10\nTarget=WATER_LIQUID\nE [MeV/u]\tdEdx [MeV/cm]\n" + energy_lines
    )
    (tmp_path / "o.txt").write_text(
        "SourceProgram=geant4\nIon=O\nAtomicNumber=8\nMassNumber=16\nIonizationPotential=10\nTarget=WATER_LIQUID\nE [MeV/u]\tdEdx [MeV/cm]\n" + energy_lines
    )
    _ = StoppingPowerTableSet.from_directory(str(tmp_path))
    captured = capsys.readouterr()
    assert "Inconsistent source_program" in captured.out

def test_from_directory_inconsistent_ionization_potential(tmp_path, capsys):
    energy_lines = "\n".join(f"{e:.5e}\t1.00000e+00" for e in np.linspace(1, 100, 150))
    (tmp_path / "c.txt").write_text(
        "SourceProgram=fluka\nIon=C\nAtomicNumber=6\nMassNumber=12\nIonizationPotential=10\nTarget=WATER_LIQUID\nE [MeV/u]\tdEdx [MeV/cm]\n" + energy_lines
    )
    (tmp_path / "o.txt").write_text(
        "SourceProgram=fluka\nIon=O\nAtomicNumber=8\nMassNumber=16\nIonizationPotential=99.9\nTarget=WATER_LIQUID\nE [MeV/u]\tdEdx [MeV/cm]\n" + energy_lines
    )
    _ = StoppingPowerTableSet.from_directory(str(tmp_path))
    captured = capsys.readouterr()
    assert "Inconsistent ionization_potential" in captured.out

# Serialization
def test_to_dict_from_dict():
    ts = StoppingPowerTableSet()
    table_c = create_dummy_table("C")
    ts.add("Carbon", table_c)
    d = ts.to_dict()
    assert "Carbon" in d
    ts2 = StoppingPowerTableSet.from_dict(d)
    np.testing.assert_allclose(ts2.get("Carbon").energy, table_c.energy)

def test_to_json_from_json():
    ts = StoppingPowerTableSet()
    table_c = create_dummy_table("C")
    ts.add("Carbon", table_c)
    json_str = ts.to_json()
    ts2 = StoppingPowerTableSet.from_json(json_str)
    assert "Carbon" in ts2.tables

def test_save_load(tmp_path):
    ts = StoppingPowerTableSet()
    table_c = create_dummy_table("C")
    ts.add("Carbon", table_c)
    file_path = tmp_path / "tableset.json"
    ts.save(str(file_path))
    ts_loaded = StoppingPowerTableSet.load(str(file_path))
    assert "Carbon" in ts_loaded
    
def test_from_dict_missing_fields():
    incomplete = {
        "Carbon": {
            "atomic_number": 6,
            "mass_number": 12,
            "energy": [1.0] * 150,
            "let": [0.1] * 150,
            "source_program": "dummy"
        }
    }
    with pytest.raises(ValueError, match="Missing required field"):
        StoppingPowerTableSet.from_dict(incomplete)

def test_from_dict_inconsistent_source_program():
    t1 = create_dummy_table("C", energy_grid=np.logspace(0, 3, 150))
    t2 = create_dummy_table("O", energy_grid=np.logspace(0, 3, 150))
    t2.source_program = "different_source"
    data = {
        "Carbon": t1.to_dict(),
        "Oxygen": t2.to_dict()
    }
    with pytest.raises(ValueError, match="Inconsistent source_program"):
        StoppingPowerTableSet.from_dict(data)

def test_from_dict_inconsistent_ionization_potential():
    t1 = create_dummy_table("C", energy_grid=np.logspace(0, 3, 150))
    t2 = create_dummy_table("O", energy_grid=np.logspace(0, 3, 150))
    t2.ionization_potential = 99.9
    data = {
        "Carbon": t1.to_dict(),
        "Oxygen": t2.to_dict()
    }
    with pytest.raises(ValueError, match="Inconsistent ionization_potential"):
        StoppingPowerTableSet.from_dict(data)

def test_from_txt_unrecognized_ion(tmp_path):
    file = tmp_path / "bad_ion.txt"
    file.write_text(
        """SourceProgram=fluka
Ion=Unobtanium
AtomicNumber=999
MassNumber=999
IonizationPotential=10
Target=WATER_LIQUID
E [MeV/u]\tdEdx [MeV/cm]
""" + "\n".join("1.0\t1.0" for _ in range(150))
    )
    with pytest.raises(ValueError, match="Ion symbol 'Unobtanium' is not recognized."):
        StoppingPowerTable.from_txt(str(file))

def test_from_txt_atomic_mass_mismatch(tmp_path):
    file = tmp_path / "mismatch.txt"
    file.write_text(
        """SourceProgram=fluka
Ion=C
AtomicNumber=7
MassNumber=12
IonizationPotential=10
Target=WATER_LIQUID
E [MeV/u]\tdEdx [MeV/cm]
""" + "\n".join("1.0\t1.0" for _ in range(150))
    )
    with pytest.raises(ValueError, match="Mismatch in atomic or mass number for ion 'C'"):
        StoppingPowerTable.from_txt(str(file))

# Filtering and querying
def test_get_available_ions():
    ts = StoppingPowerTableSet()
    ts.add("Carbon", create_dummy_table("C"))
    ts.add("Oxygen", create_dummy_table("O"))
    ions = ts.get_available_ions()
    assert set(ions) == {"Carbon", "Oxygen"}

def test_filter_by_ions():
    ts = StoppingPowerTableSet()
    ts.add("Carbon", create_dummy_table("C"))
    ts.add("Oxygen", create_dummy_table("O"))
    filtered_symbol = ts.filter_by_ions(["C"])
    assert set(filtered_symbol.keys()) == {"Carbon"}

def test_get_energy_grid():
    table_c = create_dummy_table("C")
    ts = StoppingPowerTableSet()
    ts.add("Carbon", table_c)
    grid = ts.get_energy_grid("C")
    np.testing.assert_allclose(grid, table_c.energy)

def test_get_stopping_power():
    table_c = create_dummy_table("C")
    ts = StoppingPowerTableSet()
    ts.add("Carbon", table_c)
    sp = ts.get_stopping_power("C")
    np.testing.assert_allclose(sp, table_c.let)

def test_get_common_energy_range():
    table1 = create_dummy_table("C")
    table2 = create_dummy_table("O")
    ts = StoppingPowerTableSet()
    ts.add("Carbon", table1)
    ts.add("Oxygen", table2)
    common = ts.get_common_energy_range()
    expected_min = np.min(table1.energy)
    expected_max = np.max(table1.energy)
    assert common == [expected_min, expected_max]

def test_get_common_energy_range_empty():
    ts = StoppingPowerTableSet()
    assert ts.get_common_energy_range() is None

def test_get_common_energy_range_none():
    e1 = np.linspace(50, 100, 150)
    e2 = np.linspace(1, 40, 150)
    t1 = create_dummy_table("C", energy_grid=e1)
    t2 = create_dummy_table("O", energy_grid=e2)
    ts = StoppingPowerTableSet()
    ts.add("Carbon", t1)
    ts.add("Oxygen", t2)
    assert ts.get_common_energy_range() is None

def test_resample_all():
    t1 = create_dummy_table("C")
    t2 = create_dummy_table("O")
    ts = StoppingPowerTableSet()
    ts.add("Carbon", t1)
    ts.add("Oxygen", t2)
    new_grid = np.linspace(5, 50, 100)
    ts.resample_all(new_grid)
    np.testing.assert_allclose(t1.energy, new_grid)
    np.testing.assert_allclose(t2.energy, new_grid)

def test_resample_invalid_grid():
    ts = StoppingPowerTableSet()
    ts.add("C", create_dummy_table("C"))
    with pytest.raises(ValueError, match="strictly increasing"):
        ts.resample_all(np.array([10, 9, 8]))

def test_interpolate_all():
    ts = StoppingPowerTableSet()
    ts.add("C", create_dummy_table("C"))
    ts.add("O", create_dummy_table("O"))
    energy_vals = np.array([1.0, 10.0, 100.0])
    expected_lets = 1e-3 * energy_vals**-0.5
    results_let = ts.interpolate_all(energy=energy_vals)
    for ion in ["Carbon", "Oxygen"]:
        np.testing.assert_allclose(results_let[ion], expected_lets, rtol=1e-4)

def test_interpolate_empty_array():
    ts = StoppingPowerTableSet()
    ts.add("C", create_dummy_table("C"))
    empty_array = np.array([])
    results = ts.interpolate_all(energy=empty_array)
    assert "Carbon" in results
    assert results["Carbon"].size == 0

def test_plot_single(monkeypatch):
    ts = StoppingPowerTableSet()
    ts.add("Carbon", create_dummy_table("C"))
    monkeypatch.setattr(plt, "figure", lambda *a, **kw: None)
    monkeypatch.setattr(plt, "show", lambda: None)
    ts.plot(ions=["Carbon"], show=True, single_plot=True)

def test_plot_multiple(monkeypatch):
    ts = StoppingPowerTableSet()
    ts.add("Carbon", create_dummy_table("C"))
    ts.add("Oxygen", create_dummy_table("O"))
    monkeypatch.setattr(plt, "figure", lambda *a, **kw: None)
    monkeypatch.setattr(plt, "show", lambda: None)
    ts.plot(ions=["Carbon", "Oxygen"], show=True, single_plot=False)

def test_plot_skips_missing(monkeypatch):
    ts = StoppingPowerTableSet()
    ts.add("Carbon", create_dummy_table("C"))
    monkeypatch.setattr(plt, "figure", lambda *a, **kw: None)
    monkeypatch.setattr(plt, "show", lambda: None)
    ts.plot(ions=["Carbon", "Unknown"], show=True, single_plot=True)
