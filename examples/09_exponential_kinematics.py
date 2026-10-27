"""Stage 8: compare Euler vs exponential-map F in the full rigid-spin benchmark.

Run: python examples/09_exponential_kinematics.py
"""
from validation.stage8.finite_rotation import rotation_comparison


def main():
    comparison = rotation_comparison()
    for key, value in comparison.metrics.items():
        print(f'{key}: {value:.10g}')
    print('PASS' if comparison.passed else 'FAIL')
    if not comparison.passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
