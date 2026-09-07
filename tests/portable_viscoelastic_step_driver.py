"""Write generalized-Maxwell state with one rank count and resume with another."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import amplitudes, constitutive, fields, mesh, models, solvers, studies


def _step():
    domain = dolfinx_mesh.create_box(
        MPI.COMM_WORLD,
        [np.zeros(3), np.asarray((1.0, 0.2, 0.2))],
        [2, 1, 1],
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    model = models.create(
        study=studies.viscoelastic_solid(dimension=3),
        mesh=domain,
        name="portable_viscoelastic_step",
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.isotropic_generalized_maxwell(
            instantaneous_young_modulus=1000.0,
            instantaneous_poisson_ratio=0.2,
            shear_relaxation_ratios=(0.35,),
            bulk_relaxation_ratios=(0.15,),
            relaxation_times=(0.8,),
        )
    )
    model.fix(displacement, on=mesh.face(domain, axis="x", value=0.0), component=0)
    model.fix(displacement, on=mesh.face(domain, axis="y", value=0.0), component=1)
    model.fix(displacement, on=mesh.face(domain, axis="z", value=0.0), component=2)
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=1.0),
        component=0,
        value=0.01,
    )
    return model.step(
        target=displacement,
        material=material,
        duration=2.0,
        steps=4,
        amplitude=amplitudes.tabular(
            (0.0, 1.0, 2.0),
            (0.0, 1.0, 1.0),
        ),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-11,
            maximum_iterations=6,
        ),
        progress=False,
        name="portable_viscoelastic_step",
    )


def _physical_state(step):
    cell_map = step.state.domain.topology.index_map(step.state.domain.topology.dim)
    owned = int(cell_map.size_local)
    points = len(step.state.stress.points)
    keys = np.asarray(step.state.domain.topology.original_cell_index[:owned])
    values = step.state.state.committed_state_vectors()[: owned * points]
    values = values.reshape((owned, points, -1))
    return dict(zip(keys.tolist(), values.tolist(), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "read"))
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()

    if arguments.action == "write":
        step = _step()
        step.solve(until=1.0)
        manifest = step.save_checkpoint(arguments.root, portable=True)
        if MPI.COMM_WORLD.rank == 0:
            assert manifest.is_file()
        return

    reference = _step()
    reference.solve()
    restarted = _step()
    restarted.load_checkpoint(arguments.root)
    assert restarted.accepted_time == 1.0
    restarted.solve()
    assert restarted.last_solve_info.completed_step
    np.testing.assert_allclose(
        restarted.solution.x.array,
        reference.solution.x.array,
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        restarted.state.stress.values,
        reference.state.stress.values,
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    expected = _physical_state(reference)
    actual = _physical_state(restarted)
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=2.0e-9, atol=2.0e-11
        )
    np.testing.assert_allclose(
        [item.constitutive_energy_residual for item in restarted.energy_history],
        [item.constitutive_energy_residual for item in reference.energy_history],
        rtol=2.0e-9,
        atol=2.0e-11,
    )


if __name__ == "__main__":
    main()
