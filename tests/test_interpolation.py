import numpy as np
import pytest
from pymkm.utils.interpolation import Interpolator


@pytest.fixture
def sample_data():
    # Non-monotonic segment
    energy = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
    values = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
    return energy, values


def test_interpolate_let_linear(sample_data):
    energy, values = sample_data
    interp = Interpolator(energy, values, loglog=False)
    test_e = [1.5, 2.5]
    expected = np.interp(test_e, energy, values)
    result = interp.interpolate(energy=test_e)
    np.testing.assert_allclose(result, expected)


def test_interpolate_let_loglog():
    energy = np.array([1, 10, 100])
    let = np.array([10, 100, 1000])
    interp = Interpolator(energy, let, loglog=True)
    result = interp.interpolate(energy=[10])
    assert np.allclose(result, [100], rtol=1e-4)


def test_interpolate_energy_for_let_linear(sample_data):
    energy, values = sample_data
    interp = Interpolator(energy, values, loglog=False)
    result = interp.interpolate(let=[20.0])
    # It should return multiple energies where the LET value 20 is reached
    assert isinstance(result, dict)
    assert 20.0 in result
    energies = result[20.0]
    assert len(energies) >= 2
    assert np.all((energies >= min(energy)) & (energies <= max(energy)))


def test_out_of_bounds_energy():
    energy = np.array([1, 2, 3])
    values = np.array([10, 20, 30])
    interp = Interpolator(energy, values)
    with pytest.raises(ValueError, match="out of bounds"):
        interp.interpolate(energy=[0.5, 3.5])


def test_out_of_bounds_let():
    energy = np.array([1, 2, 3])
    values = np.array([10, 20, 30])
    interp = Interpolator(energy, values)
    with pytest.raises(ValueError, match="out of bounds"):
        interp.interpolate(let=[5.0, 40.0])


def test_loglog_invalid_values():
    energy = np.array([0.0, 1.0, 10.0])
    values = np.array([0.0, 10.0, 100.0])
    interp = Interpolator(energy, values, loglog=True)
    with pytest.raises(ValueError, match="requires all values to be > 0"):
        interp.interpolate(energy=[5.0])


def test_invalid_input_combinations(sample_data):
    energy, values = sample_data
    interp = Interpolator(energy, values)
    with pytest.raises(ValueError, match="only one of"):
        interp.interpolate(energy=2.0, let=10.0)
    with pytest.raises(ValueError, match="must provide either"):
        interp.interpolate()


def test_interpolate_energy_with_short_segment():
    energy = np.array([1.0, 2.0, 3.0])
    values = np.array([10.0, 20.0, 30.0])
    interp = Interpolator(energy, values)
    interp._identify_monotonic_segments = lambda: [(0, 0), (0, 2)]  # Inject 1-point segment
    result = interp.interpolate(let=[20.0])
    assert 20.0 in result


def test_interpolate_energy_with_flat_segment():
    energy = np.array([1.0, 2.0, 1.5])
    values = np.array([10.0, 20.0, 15.0])  # Segment 10→20→15 is not monotonic
    interp = Interpolator(energy, values)
    result = interp.interpolate(let=[15.0])
    assert isinstance(result, dict)


def test_loglog_local_zero_check_inside_loop():
    energy = np.array([1.0, 2.0, 3.0])
    values = np.array([10.0, 0.0, 30.0])
    interp = Interpolator(energy, values, loglog=True)
    with pytest.raises(ValueError, match="requires all inputs > 0"):
        interp.interpolate(let=[20.0])


def test_scalar_output_converted_to_array():
    energy = np.array([1.0, 2.0])
    values = np.array([10.0, 20.0])
    interp = Interpolator(energy, values)
    result = interp.interpolate(energy=1.5)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)


def test_segment_not_monotonic_skipped():
    energy = np.array([1.0, 2.0, 2.5, 3.0])
    values = np.array([10.0, 20.0, 15.0, 25.0])  # Mixed monotonicity
    interp = Interpolator(energy, values)
    result = interp.interpolate(let=[17.0])
    assert isinstance(result[17.0], np.ndarray)


def test_loglog_interpolation_inside_loop():
    energy = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
    values = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
    interp = Interpolator(energy, values, loglog=True)
    result = interp.interpolate(let=[20.0])
    assert 20.0 in result
    assert isinstance(result[20.0], np.ndarray)
    assert np.all(result[20.0] > 0)


def test_scalar_output_loglog():
    energy = np.array([1.0, 10.0])
    values = np.array([10.0, 100.0])
    interp = Interpolator(energy, values, loglog=True)
    result = interp.interpolate(energy=3.1623)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)

   
def test_scalar_let_output_is_array():
    energy = np.array([1.0, 2.0, 3.0])
    values = np.array([10.0, 20.0, 30.0])
    interp = Interpolator(energy, values)
    result = interp.interpolate(let=15.0)
    assert isinstance(result, dict)
    assert isinstance(result[15.0], np.ndarray)
    assert result[15.0].shape == (1,)