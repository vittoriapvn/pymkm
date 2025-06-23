import numpy as np
import pytest
from types import SimpleNamespace
from pymkm.biology.oxygen_effect import (
    compute_relative_radioresistance,
    _compute_scaling_factor_component,
    compute_scaling_factors,
    compute_osmk_radioresistance,
    apply_oxygen_correction_alpha,
    apply_oxygen_correction_beta    
)

def test_compute_relative_radioresistance_scalar():
    K = 3.0
    pO2 = 5.0
    K_mult = np.array([2.0])  # Scalar in array
    R = compute_relative_radioresistance(K, pO2, K_mult)
    expected = (5.0 + 3.0) / (5.0 + 3.0 * 2.0)
    assert np.allclose(R, expected)

def test_compute_relative_radioresistance_vector():
    K = 3.0
    pO2 = 5.0
    K_mult = np.array([1.0, 2.0, 3.0])
    R = compute_relative_radioresistance(K, pO2, K_mult)
    expected = (5 + K) / (5 + K * K_mult)
    assert np.allclose(R, expected)

def test_compute_scaling_factor_component():
    R = np.array([1.0, 1.5, 2.0])
    f_max = 1.6
    Rmax = 2.0
    scale = _compute_scaling_factor_component(R, f_max, Rmax)
    expected = 1 + (R - 1) * (f_max - 1) / (Rmax - 1)
    assert np.allclose(scale, expected)

def test_compute_scaling_factors():
    R = np.array([1.0, 1.5, 2.0])
    f_rd_max = 1.2
    f_z0_max = 2.0
    Rmax = 2.0
    f_rd, f_z0 = compute_scaling_factors(R, f_rd_max, f_z0_max, Rmax)
    expected_rd = _compute_scaling_factor_component(R, f_rd_max, Rmax)
    expected_z0 = _compute_scaling_factor_component(R, f_z0_max, Rmax) ** 2
    assert np.allclose(f_rd, expected_rd)
    assert np.allclose(f_z0, expected_z0)

def test_compute_osmk_radioresistance_2021():
    params = SimpleNamespace(
        K=3.0,
        pO2=5.0,
        zR=0.5,
        gamma=2.0,
        Rm=2.0
    )
    z_bar_domain = 1.0
    R, f_rd, f_z0 = compute_osmk_radioresistance("2021", z_bar_domain, params)

    assert isinstance(R, float)
    assert f_rd is None
    assert f_z0 is None

def test_compute_osmk_radioresistance_2023():
    params = SimpleNamespace(
        K=3.0,
        pO2=5.0,
        Rmax=2.0,
        f_rd_max=1.5,
        f_z0_max=2.0
    )
    z_bar_domain = 1.0
    R, f_rd, f_z0 = compute_osmk_radioresistance("2023", z_bar_domain, params)

    assert isinstance(R, float)
    assert isinstance(f_rd, float)
    assert isinstance(f_z0, float)

def test_compute_osmk_radioresistance_invalid_version():
    with pytest.raises(ValueError, match="Unsupported OSMK version: dummy"):
        compute_osmk_radioresistance("dummy", np.array([1.0]), SimpleNamespace())

def test_apply_oxygen_correction_alpha():
    z_bar_star = 2.0
    R = 2.0
    params = SimpleNamespace(alphaL=0.02, alphaS=0.06, beta0=0.05)

    result = apply_oxygen_correction_alpha(z_bar_star, R, params)
    expected = 0.02 + (0.06 / R) + (0.05 * z_bar_star / (R**2))
    assert abs(result - expected) < 1e-10

def test_apply_oxygen_correction_beta():
    z_bar_star = 3.0
    z_bar = 1.5
    R = 2.0
    params = SimpleNamespace(beta0=0.04)

    result = apply_oxygen_correction_beta(z_bar_star, z_bar, R, params)
    expected = (z_bar_star / z_bar) * 0.04 / (R**2)
    assert abs(result - expected) < 1e-10
