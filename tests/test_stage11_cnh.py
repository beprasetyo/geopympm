"""Stage 11: constitutive, solver and checkpoint regression tests for UCNH elasticity."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import inspect_checkpoint, load_checkpoint, write_checkpoint
from geopympm.materials import (
    LinearElastic, CompressibleNeoHookean, evaluate_neo_hookean,
    update_neo_hookean_stress, neo_hookean_strain_energy,
)
from geopympm.solvers import ExplicitMPMSolver


def particle_at_F(F):
    p = ParticleSet([[0.45, 0.45]], [0.1], [0.0001])
    p.deformation_gradient[:] = np.asarray(F, dtype=float)
    p.volume[:] = p.volume0 * np.linalg.det(F)
    return p


def cnh():
    return CompressibleNeoHookean.from_young_poisson(1000.0, 0.3)


@pytest.mark.parametrize("K,G", [(0, 1), (1, 0), (-1, 1), (1, -1),
                                   (float('nan'), 2), (2, float('inf')), (True, 1)])
def test_invalid_parameters(K, G):
    with pytest.raises(ValueError):
        CompressibleNeoHookean(K, G)


def test_small_strain_moduli_agree():
    m = cnh()
    l = LinearElastic(1000., 0.3)
    assert m.bulk_modulus == pytest.approx(l.bulk_modulus)
    assert m.shear_modulus == pytest.approx(l.shear_modulus)
    assert m.lame_lambda == pytest.approx(l.lame_lambda)
    assert m.young_modulus == pytest.approx(1000.)
    assert m.poisson_ratio == pytest.approx(0.3)


def test_F_identity_has_no_stress_energy():
    p = particle_at_F(np.eye(2))
    stress, energy = evaluate_neo_hookean(p, cnh())
    np.testing.assert_allclose(stress, 0.0, atol=1e-12)
    np.testing.assert_allclose(energy, 0.0, atol=1e-12)
    assert neo_hookean_strain_energy(p, cnh()) == pytest.approx(0., abs=1e-15)


def independent_3d_reference(F2, K, G):
    F = np.eye(3)
    F[:2, :2] = F2
    J = np.linalg.det(F)
    bbar = (F @ F.T) / (J ** (2/3))
    psi = 0.5*K*(0.5*(J*J-1)-np.log(J)) + 0.5*G*(np.trace(bbar)-3)
    sigma = 0.5*K*(J-1/J)*np.eye(3) + G/J*(bbar-np.trace(bbar)/3*np.eye(3))
    return sigma, psi


@pytest.mark.parametrize("F", [
    [[0.8, 0], [0, 1]], [[1.2, 0], [0, 0.9]],
    [[1, 0.4], [0, 1]], [[0.95, 0.1], [0.05, 1.1]],
])
def test_stress_energy_match_3d_reference(F):
    p = particle_at_F(F)
    m = cnh()
    sigma, energy = evaluate_neo_hookean(p, m)
    exact, psi = independent_3d_reference(np.array(F), m.bulk_modulus, m.shear_modulus)
    np.testing.assert_allclose(sigma[0], [exact[0,0], exact[1,1], exact[2,2],
                                           exact[0,1], exact[1,2], exact[0,2]], rtol=1e-12, atol=1e-12)
    assert energy[0] == pytest.approx(psi, rel=1e-12, abs=1e-12)
    assert neo_hookean_strain_energy(p, m) == pytest.approx(psi * p.volume0[0])


def test_rigid_rotation_objective_and_energy_free():
    phi = math.radians(73.0)
    rot = [[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]]
    p = particle_at_F(rot)
    sigma, psi = evaluate_neo_hookean(p, cnh())
    np.testing.assert_allclose(sigma, 0., atol=1e-12)
    np.testing.assert_allclose(psi, 0., atol=1e-12)


def test_objective_rotation_of_stressed_state():
    m = cnh()
    base = np.array([[1.12, 0.1], [0., 0.92]])
    angle = 0.73
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    p = particle_at_F(base)
    s, e = evaluate_neo_hookean(p, m)
    q = particle_at_F(R @ base)
    t, er = evaluate_neo_hookean(q, m)
    s2 = np.array([[s[0,0], s[0,3]], [s[0,3], s[0,1]]])
    t2 = np.array([[t[0,0], t[0,3]], [t[0,3], t[0,1]]])
    np.testing.assert_allclose(t2, R @ s2 @ R.T, rtol=1e-12, atol=1e-12)
    assert t[0,2] == pytest.approx(s[0,2])
    assert er[0] == pytest.approx(e[0], abs=1e-12)


def test_strain_energy_derivative_matches_first_piola():
    F = np.array([[1.05, 0.2], [0.0, 0.9]])
    p = particle_at_F(F)
    m = cnh()
    sigma, _ = evaluate_neo_hookean(p, m)
    sig = np.array([[sigma[0,0], sigma[0,3]], [sigma[0,3], sigma[0,1]]])
    P = np.linalg.det(F) * sig @ np.linalg.inv(F).T
    h = 1e-6
    for i, j in ((0,0), (0,1), (1,0), (1,1)):
        Fp, Fm = F.copy(), F.copy()
        Fp[i,j] += h
        Fm[i,j] -= h
        eplus = evaluate_neo_hookean(particle_at_F(Fp), m)[1][0]
        eminus = evaluate_neo_hookean(particle_at_F(Fm), m)[1][0]
        assert (eplus-eminus)/(2*h) == pytest.approx(P[i,j], rel=1e-7, abs=1e-6)


def test_small_strain_matches_linear_elastic():
    m = cnh()
    strain = 1e-7
    p = particle_at_F([[1+strain, 0], [0, 1]])
    sigma, _ = evaluate_neo_hookean(p, m)
    lam, shear = m.lame_lambda, m.shear_modulus
    assert sigma[0,0] == pytest.approx((lam+2*shear)*strain, rel=1e-5)
    assert sigma[0,1] == pytest.approx(lam*strain, rel=1e-5)
    assert sigma[0,2] == pytest.approx(lam*strain, rel=1e-5)


def test_invalid_J_or_volume_is_atomic():
    p = particle_at_F(np.eye(2))
    p.stress[:] = 4.0
    before = p.stress.copy()
    p.deformation_gradient[0,0,0] = -1.0
    with pytest.raises(ValueError, match=r"det\(F\)"):
        update_neo_hookean_stress(p, cnh())
    np.testing.assert_array_equal(p.stress, before)
    p.deformation_gradient[:] = np.eye(2)
    p.volume[:] = 2*p.volume0
    with pytest.raises(ValueError, match="volume"):
        update_neo_hookean_stress(p, cnh())
    np.testing.assert_array_equal(p.stress, before)


def test_mapped_material_moduli_and_missing_id():
    p = ParticleSet([[0.25, 0.25], [0.75, 0.75]], [1., 1.], [1., 1.],
                    material_id=[0,1])
    p.deformation_gradient[:,0,0] = 0.9
    p.volume[:] = 0.9*p.volume0
    s, _ = evaluate_neo_hookean(p, {0: CompressibleNeoHookean(100., 50.),
                                     1: CompressibleNeoHookean(200., 100.)})
    np.testing.assert_allclose(s[1], 2*s[0], rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="material_id=1"):
        evaluate_neo_hookean(p, {0: cnh()})


def make_solver(*, material=None, stress_update="hyperelastic", interpolation="mpm"):
    if material is None:
        material = cnh()
    p = ParticleSet([[0.35,0.45],[0.65,0.55]], [0.1, 0.1], [0.001, 0.001],
                    velocity=[[0.01,0.0],[-0.01,0.0]],
                    material_id=[0,1], domain_size=[[0.1,0.1],[0.1,0.1]])
    g = BackgroundGrid((0,0),(0.1,0.1),(10,10))
    if isinstance(material, CompressibleNeoHookean):
        matmap = {0:material, 1:material}
    else:
        matmap = {0:material, 1:material}
    return ExplicitMPMSolver(p, g, matmap, dt=0.0005, pic_fraction=0.,
                             cfl=0.25, stress_update=stress_update,
                             kinematics_update="exponential", interpolation=interpolation)


def test_solver_10_steps_cnh_mass_momentum_energy():
    s = make_solver()
    p0 = (s.particles.mass[:,None] * s.particles.velocity).sum(axis=0)
    results = s.run(10)
    assert results[-1].step == 10
    assert s.time == pytest.approx(0.005)
    p1 = (s.particles.mass[:,None] * s.particles.velocity).sum(axis=0)
    np.testing.assert_allclose(p0, p1, atol=1e-12)
    assert neo_hookean_strain_energy(s.particles, s.materials) >= -1e-12
    assert np.all(np.isfinite(s.particles.stress))


@pytest.mark.parametrize("stress_update", ["incremental", "corotational"])
def test_solver_rejects_mismatched_update(stress_update):
    with pytest.raises(ValueError, match="requires"):
        make_solver(stress_update=stress_update)


def test_solver_rejects_linear_hyperelastic():
    with pytest.raises(ValueError, match="LinearElastic"):
        make_solver(material=LinearElastic(1000.,0.3))


def test_solver_rejects_mixed_constitutive_families():
    s = make_solver()
    with pytest.raises(TypeError, match="mixed families|homogeneous"):
        ExplicitMPMSolver(s.particles, s.grid, {0:LinearElastic(1000.,0.3),1:cnh()},
                          dt=s.dt, stress_update="hyperelastic")


@pytest.mark.parametrize("interpolation", ["mpm", "gimp", "cpgimp", "cpdi"])
def test_cnh_all_interpolation_modes_one_step(interpolation):
    s = make_solver(interpolation=interpolation)
    diag = s.step()
    assert diag.step == 1
    assert np.all(np.isfinite(s.particles.stress))


def test_checkpoint_restart_chained_identical(tmp_path):
    direct = make_solver()
    direct.run(14)
    chain = make_solver()
    chain.run(5)
    record = write_checkpoint(chain, tmp_path)
    meta = inspect_checkpoint(record.manifest_path)
    assert meta["schema_version"] == 2
    assert meta["materials"]["items"][0]["model"] == "CompressibleNeoHookean"
    restored = load_checkpoint(record.manifest_path)
    restored.run(9)
    assert restored.step_count == direct.step_count
    assert restored.time == direct.time
    for field in ("position","velocity","stress","strain","volume","deformation_gradient"):
        np.testing.assert_array_equal(getattr(restored.particles, field),
                                      getattr(direct.particles, field))


def test_checkpoint_schema_cross_version_rejected(tmp_path):
    solver = make_solver()
    record = write_checkpoint(solver, tmp_path)
    meta = inspect_checkpoint(record.manifest_path)
    meta["schema_version"] = 1
    record.manifest_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(record.manifest_path)


def test_linear_elastic_baseline_unchanged():
    s = make_solver(material=LinearElastic(1000.,0.3), stress_update="corotational")
    assert s.step().step == 1


def test_stage11_example_sulsky_smoke(tmp_path):
    """Check standalone script without requiring optional matplotlib/VTK."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    example = root / "examples" / "24_sulsky_colliding_disks_cnh.py"
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, str(example), "--steps", "50", "--no-vtu",
         "--output-dir", str(tmp_path / "run")], env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path/"run"/"summary.json").read_text())
    assert data["steps"] == 50
    assert data["final_time_s"] == pytest.approx(0.025)
    assert data["initial_total_energy_j"] == pytest.approx(0.13)
    assert data["max_total_momentum_drift"] < 1e-11


def test_cnh_output_checkpoint_integrity_and_recovery(tmp_path):
    """Stage 10G-D2/D3 output and recovery remain compatible with schema 2."""
    from geopympm.io import OutputManager, restore_output_manager, recover_latest_checkpoint
    solver = make_solver()
    manager = OutputManager(tmp_path/"visualization", every_n_steps=2,
                            particles=True, grid=False, cpdi_domains=False)
    manager.write_initial(solver)
    for _ in range(4):
        d = solver.step()
        manager.on_step_completed(solver, d)
    record = write_checkpoint(solver, tmp_path/"checkpoints", output_manager=manager)
    loaded = load_checkpoint(record.manifest_path)
    restored = restore_output_manager(record.manifest_path, loaded)
    assert loaded.step_count == 4
    assert [x.step for x in restored.frames['particles']] == [0,2,4]
    recovery = recover_latest_checkpoint(tmp_path/"checkpoints")
    assert recovery.step == 4
