"""Stage 9: compare old vs new particle advection for prescribed rigid rotation.

Run from the repository root:
    python examples/10_affine_exponential_advection.py
"""
from validation.stage9.trajectory import rigid_rotation_comparison, affine_flow


def main():
    spin=rigid_rotation_comparison()
    print('Rigid rotation: 0.3 rad, 100 time steps')
    print('Euler trajectory error:',spin.metrics['euler_position_error_end_m'],'m')
    print('Affine exponential trajectory error:',
          spin.metrics['affine_exponential_position_error_end_m'],'m')
    print('Exponential F Jacobian error:',spin.metrics['exp_j_error_max'])
    affine=affine_flow()
    print('General affine field position error:',affine.metrics['position_error_m'],'m')
    assert spin.passed and affine.passed


if __name__=='__main__':
    main()
