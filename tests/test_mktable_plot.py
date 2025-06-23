# Suppress expected warnings before any import
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*non-interactive.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*z0 provided but beta0 is missing.*")

import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np

from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.io.stopping_power import StoppingPowerTable
from pymkm.io.table_set import StoppingPowerTableSet
import matplotlib.pyplot as plt


def create_dummy_table():
    """Return a dummy stopping power table for Carbon with two energy points."""
    energy = np.array([90.0, 110.0])
    let = np.array([0.01, 0.02])
    table = StoppingPowerTable(
        ion_input="C",
        energy=energy,
        let=let,
        mass_number=12,
        source_program="mstar_3_12",
        ionization_potential=10.0
    )
    table.color = "blue"
    table.target = "WATER_LIQUID"
    return table


@pytest.fixture
def fast_computed_mktable(monkeypatch):
    """Return an MKTable with monkeypatched fast compute and valid data."""
    monkeypatch.setattr(
        "pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
        lambda self, **kwargs: (np.ones(2), np.array([0.0, 1.0]))
    )
    monkeypatch.setattr(
        "pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
        lambda self, **kwargs: 1.0
    )
    monkeypatch.setattr(
        "pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
        lambda self, z0, z_array: z_array
    )

    params = MKTableParameters(
        domain_radius=0.3,
        nucleus_radius=5.0,
        z0=1.0,
        use_stochastic_model=True
    )

    mk_table = MKTable(parameters=params)
    sp_set = StoppingPowerTableSet()
    sp_set.add("Carbon", create_dummy_table())
    mk_table.sp_table_set = sp_set
    mk_table.compute(parallel=False, ions=["Carbon"])
    return mk_table


@pytest.mark.parametrize("x,y", [
    ("energy", "z_bar_star_domain"),
    ("energy", "z_bar_domain"),
    ("let", "z_bar_nucleus"),
])
@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive.*")
def test_plot_valid_inputs(fast_computed_mktable, x, y, verbose):
    """Test that MKTable.plot executes without errors for valid input."""
    fast_computed_mktable.plot(x=x, y=y, verbose=verbose)
    plt.close()


def test_plot_invalid_x_axis_raises(fast_computed_mktable):
    """Ensure ValueError is raised for invalid x-axis name."""
    with pytest.raises(ValueError, match="Invalid x-axis"):
        fast_computed_mktable.plot(x="invalid_x", y="z_bar_domain")


def test_plot_invalid_y_axis_raises(fast_computed_mktable):
    """Ensure ValueError is raised for invalid y-axis name."""
    with pytest.raises(ValueError, match="Invalid y-axis"):
        fast_computed_mktable.plot(x="energy", y="invalid_y")

@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided.*")
def test_plot_before_compute_raises():
    """Ensure RuntimeError is raised when plot is called before compute()."""
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, z0=1.0, beta0=0.05)
    mk_table = MKTable(parameters=params)
    warnings.filterwarnings("ignore", category=UserWarning, message=".*z0.*beta0.*discarded.*")
    sp_set = StoppingPowerTableSet()
    sp_set.add("Carbon", create_dummy_table())
    mk_table.sp_table_set = sp_set

    with pytest.raises(RuntimeError, match="No computed results found"):
        mk_table.plot()

@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive.*")
@pytest.mark.filterwarnings("ignore:Both z0 and beta0 provided.*")
def test_plot_verbose_box_display(monkeypatch):
    monkeypatch.setattr(
        "pymkm.physics.specific_energy.SpecificEnergy.single_event_specific_energy",
        lambda self, **kwargs: (np.ones(2), np.array([0.0, 1.0]))
    )
    monkeypatch.setattr(
        "pymkm.physics.specific_energy.SpecificEnergy.dose_averaged_specific_energy",
        lambda self, **kwargs: 1.0
    )
    monkeypatch.setattr(
        "pymkm.physics.specific_energy.SpecificEnergy.saturation_corrected_single_event_specific_energy",
        lambda self, z0, z_array: z_array
    )

    params = MKTableParameters(
        domain_radius=0.5,
        nucleus_radius=6.0,
        z0=1.2,
        beta0=0.05,
        use_stochastic_model=True
    )

    mk_table = MKTable(parameters=params)
    sp_set = StoppingPowerTableSet()
    sp_set.add("C", create_dummy_table())
    mk_table.sp_table_set = sp_set
    mk_table.compute(ions=["C"], parallel=False)

    mk_table.plot(x="energy", y="z_bar_star_domain", verbose=True)
    plt.close()