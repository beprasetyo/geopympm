"""Checks for examples/23_sulsky_colliding_disks.py (linear-elastic surrogate)."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest


_EXAMPLE = (Path(__file__).resolve().parents[1]
            / "examples" / "23_sulsky_colliding_disks.py")
_spec = importlib.util.spec_from_file_location("sulsky_disks_example", _EXAMPLE)
assert _spec is not None and _spec.loader is not None
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


def test_seed_matches_uintah_geometry_density_and_velocity():
    p = example.seed_disks()
    assert p.n_particles == 416
    assert np.count_nonzero(p.material_id == 0) == 208
    assert np.count_nonzero(p.material_id == 1) == 208
    np.testing.assert_allclose(p.volume, example.THICKNESS * (example.CELL / 2)**2)
    np.testing.assert_allclose(p.mass / p.volume, example.RHO)
    np.testing.assert_allclose(p.velocity[:208], np.tile([0.1, 0.1], (208, 1)))
    np.testing.assert_allclose(p.velocity[208:], np.tile([-0.1, -0.1], (208, 1)))
    np.testing.assert_allclose(p.position[:208].mean(axis=0), example.CENTER_A, atol=1e-15)
    np.testing.assert_allclose(p.position[208:].mean(axis=0), example.CENTER_B, atol=1e-15)
    assert not np.any(np.linalg.norm(p.position[:208] - example.CENTER_A, axis=1) > .2 + 1e-14)
    assert not np.any(np.linalg.norm(p.position[208:] - example.CENTER_B, axis=1) > .2 + 1e-14)


def test_initial_kinetic_energy_total_momentum_and_grid_bc():
    solver = example.make_solver()
    p, g = solver.particles, solver.grid
    assert g.n_cells == (20, 20)
    assert g.n_nodes == 441
    assert g.velocity_bc_mask.sum() == 80 * 2  # 80 boundary nodes x 2 components
    assert not np.any(g.velocity_bc_mask[np.logical_and.reduce((
        g.node_coordinates[:, 0] > 0, g.node_coordinates[:, 0] < 1,
        g.node_coordinates[:, 1] > 0, g.node_coordinates[:, 1] < 1))])
    initial = example.snapshot(solver)
    assert initial["kinetic_energy_j"] == pytest.approx(0.13)
    assert initial["p_total_x_kg_m_s"] == pytest.approx(0.0, abs=1e-14)
    assert initial["p_total_y_kg_m_s"] == pytest.approx(0.0, abs=1e-14)
    assert solver.stable_dt_estimate() > solver.dt
    assert solver.materials[0].shear_modulus == pytest.approx(384.6153846)
    assert solver.materials[0].bulk_modulus == pytest.approx(833.3333333)


def test_near_free_flight_and_momentum_conservation():
    solver = example.make_solver()
    initial = example.snapshot(solver)
    for _ in range(30):
        solver.step()
    final = example.snapshot(solver)
    assert final["p_total_x_kg_m_s"] == pytest.approx(0.0, abs=1e-10)
    assert final["p_total_y_kg_m_s"] == pytest.approx(0.0, abs=1e-10)
    assert final["com_distance_m"] < initial["com_distance_m"]
    assert final["kinetic_energy_j"] == pytest.approx(initial["kinetic_energy_j"], rel=0.001)
    assert np.all(np.isfinite(solver.particles.stress))


def test_export_csv_summary_and_vtu_time_series(tmp_path):
    folder = tmp_path / "run"
    summary = example.run_case(output_dir=folder, steps=6, output_every=3, make_plot=False)
    assert summary["steps"] == 6
    assert summary["final_time_s"] == pytest.approx(0.006)
    assert summary["particles_a"] == summary["particles_b"] == 208
    with (folder / "diagnostics.csv").open(newline="") as fd:
        rows = list(csv.DictReader(fd))
    assert len(rows) == 7
    assert float(rows[-1]["time_s"]) == pytest.approx(0.006)
    saved = json.loads((folder / "summary.json").read_text())
    assert "NOT the compressible Neo-Hookean" in saved["fidelity"]
    vis = folder / "visualization"
    assert len(list(vis.glob("particles_*.vtu"))) == 3  # t=0,3,6
    ET.parse(vis / "particles_000006.vtu")
    assert (vis / "particles.pvd").is_file()
    assert (vis / "particles.visit").is_file()


def test_gimp_optional_smoke(tmp_path):
    summary = example.run_case(output_dir=tmp_path / "gimp", steps=5,
                               interpolation="gimp", write_vtu=False)
    assert summary["interpolation"] == "gimp"
    assert np.isfinite(summary["final_kinetic_energy_j"])
    assert not (tmp_path / "gimp" / "visualization").exists()


def test_output_directory_never_overwrites_existing_run(tmp_path):
    folder = tmp_path / "case"
    folder.mkdir()
    (folder / "user_notes.txt").write_text("KEEP")
    with pytest.raises(FileExistsError):
        example.run_case(output_dir=folder, steps=0)
    assert (folder / "user_notes.txt").read_text() == "KEEP"


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_particle_seeding(bad):
    with pytest.raises(ValueError):
        example.seed_disks(particles_per_cell_axis=bad)
