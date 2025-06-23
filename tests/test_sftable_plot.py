import pytest
import matplotlib.pyplot as plt
import pandas as pd
from types import SimpleNamespace
from unittest.mock import MagicMock

from pymkm.sftable.core import SFTable, SFTableParameters
from pymkm.mktable.core import MKTable, MKTableParameters

@pytest.fixture
def dummy_sftable():
    """Create a dummy SFTable with mocked data for plotting tests."""
    mk_params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    mk_table = MKTable(parameters=mk_params)

    # Add dummy color and ion mapping
    dummy_sp = MagicMock()
    dummy_sp.color = "blue"
    mk_table.sp_table_set.get = lambda ion: dummy_sp

    params = SFTableParameters(mktable=mk_table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)

    df = pd.DataFrame({"dose": [0, 2, 4], "survival_fraction": [1.0, 0.8, 0.5]})
    sft.table = [{
        "params": {"ion": "C", "energy": 100.0, "let": 0.01, "model": "classic"},
        "calculation_info": "test",
        "data": df
    }]
    return sft

@pytest.mark.filterwarnings("ignore:.*non-interactive.*")
def test_plot_executes_without_errors(dummy_sftable):
    dummy_sftable.plot()
    plt.close()

@pytest.mark.filterwarnings("ignore:.*non-interactive.*")
def test_plot_verbose_executes(dummy_sftable):
    dummy_sftable.plot(verbose=True)
    plt.close()

def test_plot_missing_table_raises():
    mk_params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    mk_table = MKTable(parameters=mk_params)
    params = SFTableParameters(mktable=mk_table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)

    with pytest.raises(ValueError, match="No survival data available"):
        sft.plot()

def test_plot_invalid_let_filter_raises(dummy_sftable):
    with pytest.raises(ValueError, match="No results found for LET"):
        dummy_sftable.plot(let=999.0)

@pytest.mark.filterwarnings("ignore:.*non-interactive.*")
def test_plot_valid_let_filter(dummy_sftable):
    dummy_sftable.plot(let=0.01)
    plt.close()

@pytest.mark.filterwarnings("ignore:.*non-interactive.*")
def test_plot_empty_dataframe_skips(capfd):
    mk_params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    mk_table = MKTable(parameters=mk_params)

    dummy_sp = MagicMock()
    dummy_sp.color = "red"
    mk_table.sp_table_set.get = lambda ion: dummy_sp

    params = SFTableParameters(mktable=mk_table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.table = [{
        "params": {"ion": "C", "energy": 100.0, "let": 0.01, "model": "classic"},
        "calculation_info": "test",
        "data": pd.DataFrame()  # Empty DataFrame
    }]

    sft.plot()
    out = capfd.readouterr().out
    assert "No data in result 1" in out

@pytest.mark.filterwarnings("ignore:.*non-interactive.*")
def test_plot_verbose_with_osmk_version():

    # Create MKTable mock with color accessor
    mk_params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    mk_table = MKTable(parameters=mk_params)
    mk_table.sp_table_set.get = lambda ion: SimpleNamespace(color="green")

    # Set alpha0 + alphaL to satisfy OSMK check
    params = SFTableParameters(
        mktable=mk_table,
        alpha0=0.1,
        alphaL=0.06,
        beta0=0.05,
        pO2=5.0
    )

    sf = SFTable(parameters=params)

    df = pd.DataFrame({"dose": [0, 2, 4], "survival_fraction": [1.0, 0.8, 0.5]})
    sf.table = [{
        "params": {
            "ion": "C",
            "energy": 100.0,
            "let": 0.01,
            "model": "stochastic",
            "osmk_version": "2023"
        },
        "calculation_info": "computed",
        "data": df
    }]

    sf.plot(verbose=True)
    plt.close()