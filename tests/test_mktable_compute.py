import pytest
import numpy as np
import pandas as pd
from pymkm.mktable.core import MKTable, MKTableParameters
from pymkm.mktable.compute import _compute_for_energy_let_pair, _run_energy_let_task, _get_osmk2023_corrected_parameters
from pymkm.io.stopping_power import StoppingPowerTable


def create_dummy_table(ion_input="C"):
    energy = np.logspace(1, 3, 150)
    let = 1e-3 * energy**(-0.5)
    energy = energy[:5]  # Use only 5 energy points to reduce computation time  # Limit for speed
    let = let[:5]
    table = StoppingPowerTable(
        ion_input=ion_input,
        energy=energy,
        let=let,
        mass_number=12,
        source_program="mstar_3_12", # bypasses the >=150 point validation check
        ionization_potential=10.0
    )
    table.color = "blue"
    table.target = "WATER_LIQUID"
    return table

def test__compute_for_energy_let_pair_basic():
    params = dict(
        model_name="Kiefer-Chatterjee",
        core_radius_type="constant",
        domain_radius=0.3,
        nucleus_radius=5.0,
        z0=0.85,
        base_points_b=50,
        base_points_r=50,
        use_stochastic_model=False,
        integration_method="trapz"
    )
    result = _compute_for_energy_let_pair(params, energy=100.0, let=0.01, atomic_number=6)
    assert "z_bar_star_domain" in result
    assert isinstance(result["z_bar_star_domain"], float)

def test__compute_for_energy_let_pair_stochastic():
    params = dict(
        model_name="Kiefer-Chatterjee",
        core_radius_type="constant",
        domain_radius=0.3,
        nucleus_radius=5.0,
        z0=0.85,
        base_points_b=30,
        base_points_r=30,
        use_stochastic_model=True,
        integration_method="trapz"
    )
    result = _compute_for_energy_let_pair(params, energy=100.0, let=0.01, atomic_number=6)
    assert all(k in result for k in ["z_bar_star_domain", "z_bar_domain", "z_bar_nucleus"])

def test__compute_for_ion_serial(monkeypatch):
    # Patch expensive SpecificEnergy methods to speed up test
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(5), np.linspace(0, 1, 5)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05, base_points_b=10, base_points_r=10)
    table = MKTable(parameters=params)  # Initialize with reduced base points for testing
    ion = "Carbon"
    dummy = create_dummy_table(ion)
    table.sp_table_set.add(ion, dummy)

    ion_name, results = table._compute_for_ion(ion, parallel=False)
    assert ion_name == ion
    assert isinstance(results, list)
    assert "z_bar_star_domain" in results[0]

def test_compute_full(monkeypatch):
    # Patch expensive SpecificEnergy methods to speed up test
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(5), np.linspace(0, 1, 5)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    dummy = create_dummy_table("Carbon")
    table.sp_table_set.add("Carbon", dummy)

    table.compute(parallel=False)
    assert "Carbon" in table.table
    df = table.table["Carbon"]["data"]
    assert isinstance(df, pd.DataFrame)
    assert "z_bar_star_domain" in df.columns

def test_compute_z0_fallback(monkeypatch):
    # Patch expensive SpecificEnergy methods to speed up test
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(5), np.linspace(0, 1, 5)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05, z0=None)
    table = MKTable(parameters=params)
    dummy = create_dummy_table("Carbon")
    table.sp_table_set.add("Carbon", dummy)

    assert table.params.z0 is None
    table.compute(parallel=False)
    assert table.params.z0 is not None

def test_run_energy_let_task_executes():
    def mock_func(x, y): return x + y
    result = _run_energy_let_task(mock_func, (2, 3))
    assert result == 5

def test__compute_for_ion_parallel_equivalent(monkeypatch):
    # NOTE: This test runs in serial mode to avoid multiprocessing issues during test
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(3), np.linspace(0, 1, 3)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)
    monkeypatch.setattr("pymkm.utils.parallel.optimal_worker_count", lambda jobs: 1)

    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05, base_points_b=5, base_points_r=5)
    table = MKTable(parameters=params)
    table.sp_table_set.add("Carbon", create_dummy_table("Carbon"))

    ion_name, results = table._compute_for_ion("Carbon", parallel=False)
    assert ion_name == "Carbon"
    assert isinstance(results, list)


def test_compute_with_custom_energy(monkeypatch):
    # Patch SpecificEnergy to avoid full computation
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(3), np.linspace(0, 1, 3)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)

    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    dummy = create_dummy_table("Carbon")
    table.sp_table_set.add("Carbon", dummy)

    custom_energy = [50.0, 100.0]
    table.compute(energy=custom_energy, parallel=False)

    assert "Carbon" in table.table


def test__compute_for_ion_parallel_flag_coverage(monkeypatch):
    # Simulate the parallel=True block using a fake executor to avoid true multiprocessing
    class FakeExecutor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, func, jobs): return [func(job) for job in jobs]

    monkeypatch.setattr("pymkm.mktable.compute.ProcessPoolExecutor", lambda *a, **kw: FakeExecutor())
    monkeypatch.setattr("pymkm.mktable.compute.tqdm", lambda *a, **kw: iter([]))
    monkeypatch.setattr("pymkm.utils.parallel.optimal_worker_count", lambda jobs: 1)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(3), np.linspace(0, 1, 3)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)

    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05, base_points_b=5, base_points_r=5)
    table = MKTable(parameters=params)
    table.sp_table_set.add("C", create_dummy_table("C"))

    # This covers the parallel=True branch structurally using fake executor
    table._compute_for_ion("C", parallel=True)


def test_summary_verbose_outputs(capsys):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    table = MKTable(parameters=params)
    table.sp_table_set.add("Carbon", create_dummy_table("Carbon"))

    table.summary(verbose=True)
    out = capsys.readouterr().out

    assert "MKTable Configuration" in out
    assert "Model version" in out
    assert "Stopping power source" in out
    assert "Carbon" in out
    assert "Track structure model" in out


def test_compute_runtime_error():
    table = MKTable.__new__(MKTable)  # Directly allocate MKTable without calling __init__  # bypass init
    table.sp_table_set = None
    table.params = None

    with pytest.raises(RuntimeError, match="MKTable is not properly initialized"):
        compute = getattr(MKTable, "compute")
        compute(table)

@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided.*")
def test__get_osmk2023_corrected_parameters_returns_expected_values():
    # Minimal MKTableParams with oxygen effect
    params = MKTableParameters(
        domain_radius=0.3,
        nucleus_radius=5.0,
        z0=0.85,
        beta0=0.05,
        apply_oxygen_effect=True,
        use_stochastic_model=True,
        pO2=5.0,
        K=3.0,
        f_rd_max=1.5,
        f_z0_max=2.0,
        Rmax=2.0
    )
    table = MKTable(parameters=params)
    table.sp_table_set.add("C", create_dummy_table("C"))

    rd_eff, z0_eff = _get_osmk2023_corrected_parameters(table)

    # Basic assertions to ensure it runs and returns floats
    assert isinstance(rd_eff, float)
    assert isinstance(z0_eff, float)
    assert rd_eff != params.domain_radius
    assert z0_eff != params.z0

@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided.*")
def test_compute_for_ion_with_oxygen_effect(capsys):
    params = MKTableParameters(
        domain_radius=0.3,
        nucleus_radius=5.0,
        z0=1.0,
        beta0=0.05,
        use_stochastic_model=True,
        apply_oxygen_effect=True,
        pO2=5.0,
        K=3.0,
        f_rd_max=1.5,
        f_z0_max=2.0,
        Rmax=2.0
    )
    table = MKTable(parameters=params)
    table.sp_table_set.add("C", create_dummy_table("C"))

    ion, result = table._compute_for_ion("C", parallel=False)

    captured = capsys.readouterr()
    assert "✔ Using OSMK2023-corrected values:" in captured.out
    assert ion == "C"
    assert isinstance(result, list)

@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided.*")
def test_compute_for_ion_without_oxygen_effect(capsys):
    params = MKTableParameters(
        domain_radius=0.3,
        nucleus_radius=5.0,
        z0=1.0,
        beta0=0.05,
        use_stochastic_model=True,
        apply_oxygen_effect=False  # No oxygen effect
    )
    table = MKTable(parameters=params)
    table.sp_table_set.add("C", create_dummy_table("C"))

    ion, result = table._compute_for_ion("C", parallel=False)

    captured = capsys.readouterr()
    assert "✔ Using OSMK2023-corrected values:" not in captured.out
    assert ion == "C"
    assert isinstance(result, list)

def test__compute_for_ion_respects_number_of_workers(monkeypatch):
    # This test verifies that the number_of_workers argument is correctly passed to the parallel executor in _compute_for_ion.
    called_workers = {} # Track the number of workers passed to FakeExecutor

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            called_workers['count'] = kwargs.get('max_workers', None)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, func, jobs): return [func(job) for job in jobs]

    monkeypatch.setattr("pymkm.mktable.compute.ProcessPoolExecutor", lambda *a, **kw: FakeExecutor(*a, **kw))
    monkeypatch.setattr("pymkm.mktable.compute.tqdm", lambda *a, **kw: iter([]))
    monkeypatch.setattr("pymkm.utils.parallel.optimal_worker_count", lambda jobs: 1)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
                        lambda self, **kwargs: (np.ones(3), np.linspace(0, 1, 3)))
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
                        lambda self, **kwargs: 0.5)
    monkeypatch.setattr("pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
                        lambda self, z0, z_array: z_array)

    params = MKTableParameters(domain_radius=0.3,
                               nucleus_radius=5.0,
                               beta0=0.05, 
                               base_points_b=5, 
                               base_points_r=5)
    table = MKTable(parameters=params)
    table.sp_table_set.add("C", create_dummy_table("C"))

    table._compute_for_ion("C", parallel=True, number_of_workers=1)

    assert called_workers['count'] == 1

