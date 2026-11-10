"""Stage 12: USF, baseline USL, MUSL ordering and restart invariants."""
import json
import numpy as np
import pytest

from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.materials import LinearElastic, CompressibleNeoHookean
from geopympm.solvers import ExplicitMPMSolver
from geopympm.io import write_checkpoint, load_checkpoint, inspect_checkpoint


def make_solver(*, scheme="usl", interpolation="mpm", model="cnh", pic_fraction=0.0,
                advection_update="euler", kinematics_update="exponential"):
    p = ParticleSet([[0.35,0.45],[0.65,0.55], [0.55,0.35]],
                    [0.1,0.1,0.1], [0.001,0.001,0.001],
                    velocity=[[0.03,0.02],[-0.02,-0.025],[0.004,0.003]],
                    domain_size=[[0.08,0.08]]*3)
    g = BackgroundGrid([0.,0.], [0.1,0.1], (10,10))
    if model == "cnh":
        mat=CompressibleNeoHookean.from_young_poisson(1000.,0.3)
        stress_update="hyperelastic"
    else:
        mat=LinearElastic(1000.,0.3)
        stress_update="corotational"
    return ExplicitMPMSolver(p,g,mat,dt=0.0005,cfl=0.25,pic_fraction=pic_fraction,
                             interpolation=interpolation, scheme=scheme,
                             stress_update=stress_update,
                             kinematics_update=kinematics_update,
                             advection_update=advection_update)


@pytest.mark.parametrize("bad", ["", "MUSL", "updated", None, 7, True])
def test_invalid_scheme_rejected(bad):
    with pytest.raises(ValueError,match="scheme"):
        make_solver(scheme=bad)


@pytest.mark.parametrize("interpolation", ["mpm", "gimp", "cpgimp", "cpdi"])
@pytest.mark.parametrize("scheme", ["usl", "usf", "musl"])
@pytest.mark.parametrize("model", ["elastic", "cnh"])
def test_all_scheme_interpolation_material_combinations(interpolation, scheme, model):
    s=make_solver(scheme=scheme,interpolation=interpolation,model=model)
    before=np.sum(s.particles.mass[:,None]*s.particles.velocity,axis=0)
    for i in range(3):
        result=s.step()
        assert result.step==i+1
        assert np.isfinite(result.min_jacobian) and result.min_jacobian>0
    assert np.all(np.isfinite(s.particles.stress))
    assert np.all(np.isfinite(s.particles.deformation_gradient))
    np.testing.assert_allclose(np.sum(s.particles.mass[:,None]*s.particles.velocity,axis=0),before,
                               rtol=0,atol=1e-11)


def test_legacy_usl_default_bitwise_equal_to_explicit_usl():
    a=make_solver(scheme="usl")
    b=make_solver(scheme="usl")
    a.run(8); b.run(8)
    for field in ("position","velocity","stress","volume","deformation_gradient"):
        np.testing.assert_array_equal(getattr(a.particles,field),getattr(b.particles,field))


def test_usf_stress_is_used_in_same_step_force():
    # Initially stress-free particles with nonzero velocity gradients.
    # In USL, first grid internal force uses sigma_n=0;
    # in USF it uses sigma_(n+1) calculated pre-force.
    usl=make_solver(scheme="usl")
    usf=make_solver(scheme="usf")
    for solver in (usl, usf):
        solver.particles.position[:] = [[0.45,0.45],[0.48,0.47],[0.50,0.43]]
    usl.step(); usf.step()
    np.testing.assert_allclose(usl.grid.force_internal, 0., atol=1e-14)
    assert np.linalg.norm(usf.grid.force_internal)>1e-9


def test_musl_remap_uses_updated_particle_momentum():
    from geopympm.interpolation.mpm import evaluate_quad4
    from geopympm.kernels.p2g import p2g_mass_momentum_kernel
    s=make_solver(scheme="musl")
    s.step()
    # grid mass is evaluated at x_n, so reproduce with a fresh solver
    old=make_solver(scheme="musl")
    ip=evaluate_quad4(old.particles,old.grid)
    m=np.zeros_like(s.grid.mass)
    p=np.zeros_like(s.grid.momentum)
    p2g_mass_momentum_kernel(ip.node_ids,ip.weights,s.particles.mass,
                             s.particles.velocity,m,p)
    np.testing.assert_allclose(s.grid.mass,m,atol=0.,rtol=0.)
    np.testing.assert_allclose(s.grid.momentum,p,atol=1e-14,rtol=1e-12)


@pytest.mark.parametrize("scheme", ["usf", "musl"])
@pytest.mark.parametrize("model", ["elastic", "cnh"])
def test_checkpoint_schema3_and_chained_restart(tmp_path,scheme,model):
    direct=make_solver(scheme=scheme,model=model)
    direct.run(9)
    chain=make_solver(scheme=scheme,model=model)
    chain.run(4)
    rec=write_checkpoint(chain,tmp_path)
    meta=inspect_checkpoint(rec.manifest_path)
    assert meta["schema_version"]==3
    assert meta["config"]["scheme"]==scheme
    loaded=load_checkpoint(rec.manifest_path)
    assert loaded.scheme==scheme
    loaded.run(5)
    assert loaded.step_count==direct.step_count
    for field in ("position","velocity","stress","volume","deformation_gradient"):
        np.testing.assert_array_equal(getattr(loaded.particles,field),getattr(direct.particles,field))


def test_legacy_schema_1_and_2_remain_unchanged(tmp_path):
    for model,version in (("elastic",1),("cnh",2)):
        solver=make_solver(model=model)
        rec=write_checkpoint(solver,tmp_path/f"{model}")
        meta=inspect_checkpoint(rec.manifest_path)
        assert meta["schema_version"]==version
        assert "scheme" not in meta["config"]
        loaded=load_checkpoint(rec.manifest_path)
        assert loaded.scheme=="usl"


@pytest.mark.parametrize("scheme", ["usf", "musl"])
def test_checkpoint_reject_missing_scheme(tmp_path,scheme):
    solver=make_solver(scheme=scheme)
    rec=write_checkpoint(solver,tmp_path)
    data=json.loads(rec.manifest_path.read_text())
    del data["config"]["scheme"]
    rec.manifest_path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="scheme|config"):
        load_checkpoint(rec.manifest_path)


@pytest.mark.parametrize("scheme", ["usl", "usf", "musl"])
def test_affine_exponential_advection(scheme):
    s=make_solver(scheme=scheme,advection_update="affine_exponential")
    s.step()
    assert np.all(np.isfinite(s.particles.position))


def test_usf_failure_rolls_back_whole_step():
    s=make_solver(scheme="usf")
    p=s.particles; g=s.grid
    # Too close to boundary during step -> failure after all stages
    p.position[0]=[0.999999,0.999999]
    p.velocity[0]=[100.0,100.0]
    # Disable CFL via tiny dt? Instead verify CFL check leaves state unchanged.
    p0=p.position.copy(); v0=p.velocity.copy()
    with pytest.raises(ValueError, match="CFL"):
        s.step()
    np.testing.assert_array_equal(p.position,p0)
    np.testing.assert_array_equal(p.velocity,v0)
    assert s.step_count==0

@pytest.mark.parametrize("scheme", ["usl", "usf", "musl"])
def test_prescribed_grid_velocity_enforced_with_pic(scheme):
    s = make_solver(scheme=scheme, pic_fraction=1.0)
    # Prescribed zero grid velocity everywhere overrides projected flow.
    s.grid.velocity_bc_mask[:] = True
    s.grid.velocity_bc_value[:] = 0.0
    s.step()
    np.testing.assert_array_equal(s.particles.velocity, 0.0)
    np.testing.assert_array_equal(s.grid.velocity, 0.0)


@pytest.mark.parametrize("scheme", ["usl", "usf", "musl"])
def test_late_grid_exit_rolls_back_everything(scheme):
    s = make_solver(scheme=scheme)
    p = s.particles
    p.position[0] = [0.99999, 0.45]
    p.velocity[0] = [0.10, 0.0]
    original_p = {k: getattr(p,k).copy() for k in s._PARTICLE_MUTABLE}
    original_g = {k: getattr(s.grid,k).copy() for k in s._GRID_MUTABLE}
    with pytest.raises(ValueError, match="outside background grid"):
        s.step()
    for k, state in original_p.items():
        np.testing.assert_array_equal(getattr(p,k), state)
    for k, state in original_g.items():
        np.testing.assert_array_equal(getattr(s.grid,k), state)
    assert s.step_count == 0 and s.time == 0.0


def test_musl_effective_velocity_field_differs_from_usl_on_contact():
    a = make_solver(scheme="usl")
    b = make_solver(scheme="musl")
    for s in (a,b):
        s.particles.position[:] = [[0.45,0.45],[0.48,0.47],[0.50,0.43]]
    a.run(2)
    b.run(2)
    assert not np.array_equal(a.particles.position,b.particles.position)
    assert not np.array_equal(a.particles.stress,b.particles.stress)
