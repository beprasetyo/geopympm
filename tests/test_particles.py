import numpy as np
import pytest

from geopympm.core.particles import ParticleSet


def test_2d_particle_initialization():
    position = np.array([[0.125, 0.125], [0.375, 0.125]])
    volume = np.array([0.0625, 0.0625])  # 0.25 m x 0.25 m x 1 m
    particles = ParticleSet(
        position=position,
        mass=1800.0 * volume,
        volume=volume,
        domain_size=np.full((2, 2), 0.25),
        material_id=[0, 1],
    )

    assert particles.n_particles == 2
    assert particles.ndim == 2
    assert particles.position.shape == (2, 2)
    assert particles.stress.shape == (2, 6)
    assert particles.deformation_gradient.shape == (2, 2, 2)
    assert np.allclose(particles.density, 1800.0)
    assert np.allclose(particles.velocity, 0.0)
    assert np.allclose(particles.deformation_gradient, np.eye(2))
    assert particles.has_gimp_domains
    assert particles.position.flags.c_contiguous
    assert particles.stress.flags.c_contiguous
    assert particles.material_id.dtype == np.int32


def test_3d_and_initial_stress():
    stress = np.array([[-100.0, -80.0, -60.0, 5.0, 0.0, 0.0]])
    particles = ParticleSet(
        position=[[0.0, 0.0, 0.0]], mass=[2.0], volume=[0.001], stress=stress
    )

    assert particles.ndim == 3
    assert particles.deformation_gradient.shape == (1, 3, 3)
    assert np.array_equal(particles.stress, stress)
    assert not particles.has_gimp_domains


def test_input_arrays_are_copied_and_displacement_updates():
    position = np.array([[1.0, 2.0]])
    particles = ParticleSet(position=position, mass=[1.0], volume=[0.5])
    position[0, 0] = 999.0
    assert particles.position[0, 0] == 1.0
    particles.position[0, 0] += 0.1
    assert np.allclose(particles.displacement, [[0.1, 0.0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": [[0.0]], "mass": [1.0], "volume": [1.0]},
        {"position": [[0.0, 0.0]], "mass": [0.0], "volume": [1.0]},
        {"position": [[0.0, 0.0]], "mass": [1.0], "volume": [-1.0]},
        {"position": [[0.0, np.nan]], "mass": [1.0], "volume": [1.0]},
        {"position": [[0.0, 0.0]], "mass": [1.0], "volume": [1.0],
         "material_id": [0.5]},
        {"position": [[0.0, 0.0]], "mass": [1.0], "volume": [1.0],
         "domain_size": [[0.5, 0.0]]},
        {"position": [[0.0, 0.0]], "mass": [1.0], "volume": [1.0],
         "stress": [[0.0, 0.0, 0.0]]},
    ],
)
def test_invalid_inputs_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        ParticleSet(**kwargs)
