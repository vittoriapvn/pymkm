import numpy as np
import pytest

from pymkm.physics.particle_track import ParticleTrack
from pymkm.physics.specific_energy import SpecificEnergy

# --- Tests for Initialization ---

def test_invalid_region_radius():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=100, atomic_number=1, let=10)
    with pytest.raises(ValueError, match="region_radius must be positive"):
        SpecificEnergy(pt, region_radius=0.0)

def test_invalid_particle_track_type():
    with pytest.raises(TypeError, match="particle_track must be an instance of ParticleTrack"):
        SpecificEnergy(particle_track="not_a_track", region_radius=1.0)

def test_valid_initialization():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=100, atomic_number=1, let=10)
    se = SpecificEnergy(pt, region_radius=0.5)
    assert isinstance(se, SpecificEnergy)
    
# --- Tests for single_event_specific_energy ---

def test_single_event_specific_energy_with_manual_b():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=100, atomic_number=1, let=10)
    se = SpecificEnergy(pt, region_radius=0.5)

    b_values = np.linspace(0.01, 3.0, 10)
    z_array, b_array = se.single_event_specific_energy(impact_parameters=b_values)

    assert z_array.shape == b_array.shape
    assert np.all(np.diff(b_array) >= 0)
    assert np.all(np.isfinite(z_array))
    assert np.any(z_array > 0)

def test_single_event_specific_energy_with_default_b():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=80, atomic_number=1, let=8)
    se = SpecificEnergy(pt, region_radius=0.4)

    z_array, b_array = se.single_event_specific_energy()
    assert z_array.shape == b_array.shape
    assert np.all(z_array >= 0)
    assert np.all(np.isfinite(z_array))

def test_custom_sampling_points_for_b_and_r():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=150, atomic_number=6, let=50)
    se = SpecificEnergy(pt, region_radius=1.0)

    impact_parameters = np.linspace(0.01, se.region_radius + se.penumbra_radius, 5)
    z_array, b_array = se.single_event_specific_energy(
        impact_parameters=impact_parameters,
        base_points_r=20
    )

    assert len(z_array) == 5
    assert z_array.shape == b_array.shape
    assert np.all(np.isfinite(z_array))
    assert np.all(z_array >= 0)

def test_internal_zb_behaviour_against_manual_loop():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=60, atomic_number=1, let=12)
    se = SpecificEnergy(pt, region_radius=0.3)

    b_array = np.linspace(0.05, 1.0, 6)
    z_bulk, b_bulk = se.single_event_specific_energy(impact_parameters=b_array)
    z_manual = [se._compute_z_single_b(b, None) for b in b_array]

    np.testing.assert_allclose(z_bulk, z_manual, rtol=1e-4)

def test_single_event_specific_energy_zero_result():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=5.0, atomic_number=1, let=0.5)
    se = SpecificEnergy(pt, region_radius=0.1)

    b_array = np.linspace(10, 20, 5)  # Outside interaction range
    z_array, _ = se.single_event_specific_energy(impact_parameters=b_array)
    assert np.allclose(z_array, 0.0)

def test_compute_z_single_b_direct_call():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=100, atomic_number=1, let=10)
    se = SpecificEnergy(pt, region_radius=0.4)
    z_b = se._compute_z_single_b(b=0.5)
    assert isinstance(z_b, float)
    assert z_b >= 0

def test_single_event_specific_energy_parallel():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=150, atomic_number=6, let=20)
    se = SpecificEnergy(pt, region_radius=0.3)

    b_values = np.linspace(0.01, 2.0, 10)
    z_array, b_array = se.single_event_specific_energy(impact_parameters=b_values, parallel=True)

    assert z_array.shape == b_array.shape
    assert np.all(np.isfinite(z_array))

# --- Saturation parameter tests ---

def test_compute_saturation_parameter_valid():
    z0 = SpecificEnergy.compute_saturation_parameter(6.0, 0.3, 0.3)
    assert isinstance(z0, float)
    assert z0 > 0

def test_compute_saturation_parameter_invalid():
    with pytest.raises(ValueError):
        SpecificEnergy.compute_saturation_parameter(0.0, 0.3, 0.3)
    with pytest.raises(ValueError):
        SpecificEnergy.compute_saturation_parameter(6.0, -0.3, 0.3)
    with pytest.raises(ValueError):
        SpecificEnergy.compute_saturation_parameter(6.0, 0.3, 0.0)

# --- Tests for saturation corrected specific energy ---

@pytest.mark.parametrize("model", ["square_root", "quadratic"])
def test_saturation_correction_models(model):
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=100, atomic_number=6, let=20)
    se = SpecificEnergy(pt, region_radius=0.3)

    z0 = se.compute_saturation_parameter(6.0, 0.3, 0.3)
    z_array, b_array = se.single_event_specific_energy()

    z_corrected = se.saturation_corrected_single_event_specific_energy(
        z0=z0, z_array=z_array, model=model
    )

    assert z_corrected.shape == z_array.shape
    assert np.all(np.isfinite(z_corrected))
    assert np.all(z_corrected >= 0)

def test_saturation_model_invalid():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=100, atomic_number=6, let=20)
    se = SpecificEnergy(pt, region_radius=0.3)
    z0 = se.compute_saturation_parameter(6.0, 0.3, 0.3)
    z_array, b_array = se.single_event_specific_energy()

    with pytest.raises(ValueError, match="Unsupported model"):
        se.saturation_corrected_single_event_specific_energy(
            z0=z0, z_array=z_array, model="unsupported"
        )

# --- Dose-averaged z tests ---

def test_dose_averaged_specific_energy_z():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=100, atomic_number=1, let=10)
    se = SpecificEnergy(pt, region_radius=0.3)
    z_array, b_array = se.single_event_specific_energy()
    z_avg = se.dose_averaged_specific_energy(z_array, b_array)

    assert isinstance(z_avg, float)
    assert z_avg > 0

def test_dose_averaged_specific_energy_z_prime():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=100, atomic_number=6, let=50)
    se = SpecificEnergy(pt, region_radius=0.3)

    z0 = se.compute_saturation_parameter(6.0, 0.3, 0.3)
    z_array, b_array = se.single_event_specific_energy()
    z_corrected = se.saturation_corrected_single_event_specific_energy(
        z0=z0,
        z_array=z_array,
        model="square_root"
    )

    z_avg = se.dose_averaged_specific_energy(z_array, b_array, z_corrected, model="square_root")

    assert isinstance(z_avg, float)
    assert z_avg > 0

def test_dose_averaged_specific_energy_z_sat():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=100, atomic_number=6, let=50)
    se = SpecificEnergy(pt, region_radius=0.3)

    z0 = se.compute_saturation_parameter(6.0, 0.3, 0.3)
    z_array, b_array = se.single_event_specific_energy()
    z_corrected = se.saturation_corrected_single_event_specific_energy(
        z0=z0,
        z_array=z_array,
        model="quadratic"
    )

    z_avg = se.dose_averaged_specific_energy(z_array, b_array, z_corrected, model="quadratic")

    assert isinstance(z_avg, float)
    assert z_avg > 0

def test_dose_averaged_specific_energy_zero_denominator():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=100, atomic_number=1, let=10)
    se = SpecificEnergy(pt, region_radius=0.3)
    z_array = np.zeros(5)
    b_array = np.linspace(0.1, 1.0, 5)
    z_avg = se.dose_averaged_specific_energy(z_array, b_array)
    assert z_avg == 0.0
    
def test_dose_averaged_specific_energy_invalid_model():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=100, atomic_number=6, let=50)
    se = SpecificEnergy(pt, region_radius=0.3)
    z_array = np.array([0.1, 0.2, 0.3])
    b_array = np.array([0.01, 0.02, 0.03])

    with pytest.raises(ValueError, match="Model must be 'square_root' or 'quadratic' when providing z_corrected"):
        se.dose_averaged_specific_energy(z_array, b_array, z_corrected=z_array, model="invalid")

@pytest.mark.parametrize("method", ["trapz", "simps", "quad"])
def test_dose_averaged_specific_energy_with_all_methods(method):
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=120, atomic_number=6, let=30)
    se = SpecificEnergy(pt, region_radius=0.4)
    z_array, b_array = se.single_event_specific_energy()

    z_avg = se.dose_averaged_specific_energy(z_array=z_array, b_array=b_array, integration_method=method)

    assert isinstance(z_avg, float)
    assert z_avg >= 0

def test_dose_averaged_specific_energy_invalid_integration_method():
    pt = ParticleTrack(model_name="Kiefer-Chatterjee", core_radius_type="energy-dependent",
                       energy=100, atomic_number=1, let=10)
    se = SpecificEnergy(pt, region_radius=0.3)
    z_array, b_array = se.single_event_specific_energy()

    with pytest.raises(ValueError, match="Unsupported integration method"):
        se.dose_averaged_specific_energy(z_array, b_array, integration_method="invalid_method")

