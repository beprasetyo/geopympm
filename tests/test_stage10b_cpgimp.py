"""Stage 10B: exact axis-aligned cpGIMP kinematics, integration, failure safety."""
import math

import numpy as np
import pytest

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import (evaluate_gimp, candidate_cpgimp_domain,
                                    update_cpgimp_domain)
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def make_particle(x=(2.5, 2.5), length=(0.4, 0.6), grid=None):
    g = BackgroundGrid((0, 0), (1, 1), (6, 6)) if grid is None else grid
    p = ParticleSet([x], [1.0], [0.3], domain_size=[length])
    return p, g


@pytest.mark.parametrize('Fx,Fy', [(1., 1.), (1.4, .75), (.5, 1.6)])
def test_cp_domain_stretch_and_constitutive_volume_separate(Fx, Fy):
    p, g = make_particle()
    p.deformation_gradient[0] = np.diag([Fx, Fy])
    p.volume[:] = p.volume0*Fx*Fy
    got = update_cpgimp_domain(p, g)
    np.testing.assert_allclose(got, [[.4*Fx, .6*Fy]], atol=1e-15)
    assert p.volume[0] == pytest.approx(.3*Fx*Fy)
    # Domain has a different area from the physical volume, by design.
    assert np.prod(got) == pytest.approx(.24*Fx*Fy)
    ip = evaluate_gimp(p, g)
    np.testing.assert_allclose(ip.weights.sum(axis=1), 1., atol=1e-13)
    np.testing.assert_allclose(ip.gradients.sum(axis=1), 0., atol=1e-13)
    coords = g.node_coordinates[ip.node_ids]
    np.testing.assert_allclose(np.einsum('pa,pad->pd', ip.weights, coords), p.position,
                               atol=1e-13)
    np.testing.assert_allclose(np.einsum('pai,paj->pij', coords, ip.gradients),
                               np.eye(2)[None], atol=1e-13)


def test_cpgimp_uses_cumulative_F_not_previous_domain_and_stored_reference():
    p,g=make_particle()
    p.deformation_gradient[0] = np.diag([1.2, 0.9])
    update_cpgimp_domain(p,g)
    np.testing.assert_allclose(p.domain_size, [[.48,.54]], atol=1e-15)
    p.deformation_gradient[0] = np.diag([.8,1.1])
    update_cpgimp_domain(p,g)
    np.testing.assert_allclose(p.domain_size, [[.32,.66]], atol=1e-15)
    np.testing.assert_array_equal(p.domain_size0, [[.4,.6]])


def test_pure_shear_not_represented_and_rotation_is_not_objective():
    p,g=make_particle()
    p.deformation_gradient[0] = [[1., .7], [0., 1.]]
    update_cpgimp_domain(p,g)
    np.testing.assert_allclose(p.domain_size, [[.4,.6]], atol=1e-15)
    angle=math.pi/4
    p.deformation_gradient[0] = [[math.cos(angle), -math.sin(angle)],
                                  [math.sin(angle), math.cos(angle)]]
    update_cpgimp_domain(p,g)
    # Documented limitation: cpGIMP shrinks despite det(F)=1 under rotation.
    np.testing.assert_allclose(p.domain_size, [[.4*math.cos(angle),
                                                .6*math.cos(angle)]], atol=1e-15)


@pytest.mark.parametrize('F,reason', [
    ([[0., -1.], [1., 0.]], 'F_xx>0'),
    ([[-.2, 0.], [0., -1.]], 'F_xx>0'),
    ([[1., 0.], [0., 0.]], 'det\\(F\\)'),
    ([[float('nan'), 0.], [0., 1.]], 'finite'),
])
def test_bad_F_fails_without_domain_mutation(F,reason):
    p,g=make_particle()
    old=p.domain_size.copy()
    p.deformation_gradient[0] = F
    with pytest.raises(ValueError,match=reason):
        update_cpgimp_domain(p,g)
    np.testing.assert_array_equal(p.domain_size,old)


def test_too_large_or_outside_domain_fails_without_mutation():
    p,g=make_particle(length=(.8,.8))
    old=p.domain_size.copy()
    p.deformation_gradient[0] = np.diag([1.5,1.0])
    with pytest.raises(ValueError,match='domain_size <='):
        update_cpgimp_domain(p,g)
    np.testing.assert_array_equal(p.domain_size,old)
    p.position[0] = (.43, 2.5)
    p.deformation_gradient[0] = np.diag([1.1,1.0])
    with pytest.raises(ValueError,match='extends beyond'):
        update_cpgimp_domain(p,g)
    np.testing.assert_array_equal(p.domain_size,old)


def test_missing_reference_domain_rejected():
    p=ParticleSet([[2.5,2.5]],[1.],[.25])
    g=BackgroundGrid((0,0),(1,1),(6,6))
    with pytest.raises(ValueError, match='domain_size0'):
        candidate_cpgimp_domain(p,g)


def test_solver_rigid_translation_unchanged_uGIMP_and_cpGIMP():
    for mode in ('gimp','cpgimp'):
        p,g=make_particle(x=(2.1,2.2), length=(.5,.5))
        p.velocity[:]=(.3,-.2)
        s=ExplicitMPMSolver(p,g,LinearElastic(20., .25),dt=.01,
                            interpolation=mode,pic_fraction=0.,
                            stress_update='corotational',
                            kinematics_update='exponential',
                            advection_update='affine_exponential')
        s.run(10)
        np.testing.assert_allclose(p.position, [[2.13,2.18]],atol=2e-13)
        np.testing.assert_allclose(p.deformation_gradient,np.eye(2)[None],atol=2e-13)
        np.testing.assert_array_equal(p.domain_size,p.domain_size0)


@pytest.mark.parametrize('ax,ay', [(0.2,-0.3), (-.1,.15)])
def test_solver_prescribed_affine_field_deforms_domain_exactly(ax,ay):
    p,g=make_particle(x=(2.25,2.75),length=(.5,.4))
    # All nodal velocity DOFs are prescribed by a linear affine field.
    # This isolates the kinematics path; the stress state can evolve but
    # internal force cannot disturb imposed affine velocity gradient.
    g.velocity_bc_mask[:] = True
    g.velocity_bc_value[:] = g.node_coordinates*np.array([ax,ay])
    material=LinearElastic(10.,.25)
    s=ExplicitMPMSolver(p,g,material,dt=.01,
                        interpolation='cpgimp',pic_fraction=1.0,
                        kinematics_update='exponential',
                        advection_update='affine_exponential')
    for k in range(1,6):
        s.step()
        fx=math.exp(ax*.01*k)
        fy=math.exp(ay*.01*k)
        np.testing.assert_allclose(p.deformation_gradient,[np.diag([fx,fy])],
                                   rtol=3e-13,atol=2e-14)
        np.testing.assert_allclose(p.domain_size,[[.5*fx,.4*fy]],
                                   rtol=3e-13,atol=2e-14)
        np.testing.assert_allclose(p.volume, p.volume0*fx*fy,
                                   rtol=3e-13,atol=2e-14)
        np.testing.assert_allclose(p.position,p.position0*np.array([fx,fy]),
                                   rtol=3e-13,atol=2e-14)
        assert s.step_count==k


def test_solver_rejects_inconsistent_initial_domains():
    p,g=make_particle()
    p.deformation_gradient[0]=np.diag([1.1,1.0]);p.volume*=1.1
    with pytest.raises(ValueError,match='initial domain_size'):
        ExplicitMPMSolver(p,g,LinearElastic(20.,.25),dt=.01,
                          interpolation='cpgimp')


def test_solver_cpgimp_oversized_stretch_rolls_back_all_state():
    p,g=make_particle(length=(.95,.5))
    g.velocity_bc_mask[:]=True
    g.velocity_bc_value[:]=g.node_coordinates*np.array([2.,0.])
    s=ExplicitMPMSolver(p,g,LinearElastic(.5,.25),dt=.05,cfl=.9,
                        interpolation='cpgimp',kinematics_update='exponential')
    pre_particle={name:getattr(p,name).copy() for name in s._PARTICLE_MUTABLE}
    pre_grid={name:getattr(g,name).copy() for name in s._GRID_MUTABLE}
    with pytest.raises(ValueError,match='domain_size <='):
        s.step()
    for name,values in pre_particle.items():
        np.testing.assert_array_equal(getattr(p,name),values)
    for name,values in pre_grid.items():
        np.testing.assert_array_equal(getattr(g,name),values)
    assert s.time==0.0 and s.step_count==0


def test_cpgimp_is_opt_in_and_mpm_gimp_defaults_preserved():
    p,g=make_particle()
    material=LinearElastic(100.,.2)
    assert ExplicitMPMSolver(p,g,material,dt=.001).interpolation=='mpm'
    assert ExplicitMPMSolver(p,g,material,dt=.001,interpolation='gimp').interpolation=='gimp'
    assert ExplicitMPMSolver(p,g,material,dt=.001,interpolation='cpgimp').interpolation=='cpgimp'
    with pytest.raises(ValueError,match='interpolation'):
        ExplicitMPMSolver(p,g,material,dt=.001,interpolation='not_a_method')
