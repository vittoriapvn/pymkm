import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from pymkm.mktable.core import MKTable, MKTableParameters
from pymkm.sftable.core import SFTable, SFTableParameters


def make_mock_table(stochastic=False, interpolate_multiple=False):
    params = MKTableParameters(domain_radius=0.3, nucleus_radius=5.0, beta0=0.05)
    if stochastic:
        params.use_stochastic_model = True
    table = MKTable(parameters=params)

    # Mock required methods and attributes
    mock_sp = MagicMock()
    mock_sp.atomic_number = 6
    if interpolate_multiple:
        mock_sp.interpolate.side_effect = lambda energy=None, let=None: (
            [0.01] if energy is not None else {float(let): [100.0, 110.0]}
        )
    else:
        mock_sp.interpolate.side_effect = lambda energy=None, let=None: (
            [0.01] if energy is not None else {float(let): [100.0]}
        )
    table.sp_table_set.get = MagicMock(return_value=mock_sp)
    table.sp_table_set._map_to_fullname = lambda ion: "Carbon"
    return table

@pytest.mark.filterwarnings("ignore:No precomputed data found for ion.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_sftable_compute_classic(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="classic")
    results = sft.table
    assert isinstance(results, list)
    assert len(results) == 1
    df = results[0]["data"]
    assert "dose" in df.columns and "survival_fraction" in df.columns

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
@pytest.mark.filterwarnings("ignore:No precomputed data found for ion.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_sftable_compute_stochastic(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table(stochastic=True)
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="stochastic")
    results = sft.table
    df = results[0]["data"]
    assert "survival_fraction" in df.columns

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_stochastic_model_mismatch_raises(mock_compute):
    table = make_mock_table(stochastic=False)
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    with pytest.raises(ValueError, match="Stochastic output requested"):
        sft.compute(ion="C", energy=100.0, model="stochastic")

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_interpolated_energy_only(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    df = pd.DataFrame({"energy": [100.0], "let": [0.01]})
    table.table = {"Carbon": {"data": df}}
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="classic", force_recompute=False)
    results = sft.table
    assert results[0]["calculation_info"] == "interpolated"

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_energy_only_interpolated_from_existing_table(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    df = pd.DataFrame({"energy": [100.0, 100.0], "let": [0.01, 0.02]})
    table.table = {"Carbon": {"data": df}}
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="classic", force_recompute=False)
    results = sft.table   
    assert len(results) == 2
    assert all(r["params"]["energy"] == 100.0 for r in results)

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_let_only_interpolated_from_existing_table(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    df = pd.DataFrame({"energy": [100.0], "let": [0.01]})
    table.table = {"Carbon": {"data": df}}
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", let=0.01, model="classic", force_recompute=False)
    results = sft.table  
    assert len(results) == 1
    assert results[0]["params"]["let"] == 0.01

@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_energy_and_let_both_present(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    df = pd.DataFrame({"energy": [100.0], "let": [0.01]})
    table.table = {"Carbon": {"data": df}}
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, let=0.01, model="classic", force_recompute=False)
    results = sft.table
    assert results[0]["calculation_info"] == "interpolated"

@pytest.mark.filterwarnings("ignore:No precomputed data found for ion.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_missing_energy_and_let_raises(mock_compute):
    table = make_mock_table()
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    with pytest.raises(ValueError, match="At least one of 'energy' or 'let' must be specified"):
        sft.compute(ion="C")

@pytest.mark.filterwarnings("ignore:No precomputed data found for ion.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_let_only_multiple_energies(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table(interpolate_multiple=True)
    # Do not preload table.table so that ion_data is None and interpolation is triggered
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", let=0.01, model="classic", force_recompute=False)
    results = sft.table
    assert len(results) == 2
    assert all(r["params"]["let"] == 0.01 for r in results)
    assert set(r["params"]["energy"] for r in results) == {100.0, 110.0}

@pytest.mark.filterwarnings("ignore:No precomputed data found for ion.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_energy_only_infers_let(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    table.table = {}  # Ensure ion_data is None so it triggers interpolation
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="classic", force_recompute=False)
    results = sft.table
    assert len(results) == 1
    assert results[0]["params"]["energy"] == 100.0
    assert results[0]["params"]["let"] == 0.01
    assert results[0]["calculation_info"] == "computed"

@pytest.mark.filterwarnings("ignore:No precomputed data found for ion.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_energy_and_let_both_given_no_data(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    table.table = {}  # Force use of final else branch
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, let=0.01, model="classic", force_recompute=False)
    results = sft.table
    assert len(results) == 1
    assert results[0]["params"]["energy"] == 100.0
    assert results[0]["params"]["let"] == 0.01
    assert results[0]["calculation_info"] == "computed"

@pytest.mark.filterwarnings("ignore:No precomputed data found for ion.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_energy_not_found_triggers_interpolation(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table()
    df = pd.DataFrame({"energy": [90.0], "let": [0.03]})
    table.table = {"Carbon": {"data": df}}
    params = SFTableParameters(mktable=table, alpha0=0.1, beta0=0.05)
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="classic", force_recompute=False)
    results = sft.table
    assert len(results) == 1
    assert results[0]["params"]["energy"] == 100.0
    assert results[0]["params"]["let"] == 0.01
    assert results[0]["calculation_info"] == "computed"

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
@pytest.mark.filterwarnings("ignore:No precomputed data found for ion.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_osmk2023_path(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1,
        "z_bar_domain": 0.05,
        "z_bar_nucleus": 0.2
    }
    table = make_mock_table(stochastic=True)
    params = SFTableParameters(
        mktable=table,
        alphaL=0.03,
        alphaS=0.07,
        beta0=0.05,
        pO2=5.0,
        f_rd_max=1.5,
        f_z0_max=2.0,
        Rmax=3.0
    )
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="stochastic", apply_oxygen_effect=True)
    results = sft.table
    assert results[0]["params"]["osmk_version"] == "2023"
    assert "survival_fraction" in results[0]["data"].columns

def test_osmk_rejected_for_classic_model():
    table = make_mock_table(stochastic=False)
    params = SFTableParameters(
        mktable=table,
        alphaL=0.03, alphaS=0.07, beta0=0.05,
        pO2=5.0, f_rd_max=1.2, f_z0_max=1.8, Rmax=3.0
    )
    sft = SFTable(parameters=params)
    with pytest.raises(ValueError, match="Oxygen effect.*only be applied with model='stochastic'"):
        sft.compute(ion="C", energy=100.0, model="classic", apply_oxygen_effect=True)

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
@pytest.mark.filterwarnings("ignore:No precomputed data found for ion.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_osmk_2021_path(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1, "z_bar_domain": 0.05, "z_bar_nucleus": 0.2
    }
    table = make_mock_table(stochastic=True)
    params = SFTableParameters(
        mktable=table,
        alphaL=0.03, alphaS=0.07, beta0=0.05,
        pO2=5.0, zR=1.5, gamma=2.0, Rm=1.2
    )
    sft = SFTable(parameters=params)
    sft.compute(ion="C", energy=100.0, model="stochastic", apply_oxygen_effect=True)
    assert sft.table[0]["params"]["osmk_version"] == "2021"

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
def test_osmk_raises_if_both_versions_specified():
    table = make_mock_table(stochastic=True)
    with pytest.raises(ValueError, match="Cannot mix OSMK 2021.*2023"):
        SFTableParameters(
            mktable=table,
            alphaL=0.03, alphaS=0.07, beta0=0.05,
            pO2=5.0,
            zR=1.5, gamma=2.0, Rm=1.2,      # OSMK 2021
            f_rd_max=1.5, f_z0_max=2.0, Rmax=3.0  # OSMK 2023
        )

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
def test_osmk_raises_if_missing_all_parameters():
    table = make_mock_table(stochastic=True)
    params = SFTableParameters(
        mktable=table,
        alphaL=0.03, alphaS=0.07, beta0=0.05,
        pO2=5.0  # No zR/gamma/Rm nor f_rd_max/f_z0_max/Rmax
    )
    sft = SFTable(parameters=params)
    with pytest.raises(ValueError, match="required parameters are missing"):
        sft.compute(ion="C", energy=100.0, model="stochastic", apply_oxygen_effect=True)

@pytest.mark.filterwarnings("ignore:z0 not provided.*")
@patch("pymkm.sftable.compute._compute_for_energy_let_pair")
def test_compute_osmk_inconsistent_versions_detected_in_compute(mock_compute):
    mock_compute.return_value = {
        "z_bar_star_domain": 0.1, "z_bar_domain": 0.05, "z_bar_nucleus": 0.2
    }
    table = make_mock_table(stochastic=True)

    # Costruisci con SOLO parametri 2021
    params = SFTableParameters(
        mktable=table,
        alphaL=0.03, alphaS=0.07, beta0=0.05,
        pO2=5.0,
        zR=1.5, gamma=2.0, Rm=1.2
    )
    # Assegna *dopo* i parametri 2023 (forzando inconsistenza in compute)
    params.f_rd_max = 1.5
    params.f_z0_max = 2.0
    params.Rmax = 3.0

    sft = SFTable(parameters=params)
    with pytest.raises(ValueError, match="cannot provide both 2021 and 2023"):
        sft.compute(ion="C", energy=100.0, model="stochastic", apply_oxygen_effect=True)
