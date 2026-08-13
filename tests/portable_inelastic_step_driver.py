"""Write nonlinear Steps with two ranks and resume them with one rank."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import constitutive, fields, mechanics, mesh, models, solvers, steps, studies


def _step(kind: str):
    domain = dolfinx_mesh.create_box(
        MPI.COMM_WORLD,
        [np.zeros(3), np.asarray((1.0, 0.2, 0.2))],
        [2, 1, 1],
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    model = models.create(
        study=(
            studies.creep_solid()
            if kind == "creep"
            else studies.nonlinear_static(
                physics="solid_mechanics", dimension=3
            )
        ),
        mesh=domain,
        name=f"portable_{kind}_step",
    )
    displacement = model.field(fields.displacement(domain))
    material = (
        constitutive.isotropic_power_law(
            young=1000.0,
            poisson=0.3,
            density=1.0,
            coefficient=2.0e-5,
            stress_exponent=2.0,
            reference_stress=1.0,
        )
        if kind == "creep"
        else constitutive.J2LinearIsotropicHardening(
            young=1000.0,
            poisson=0.3,
            yield_stress=0.4,
            hardening_modulus=100.0,
        )
    )
    model.material(material)
    model.fix(displacement, on=mesh.face(domain, axis="x", value=0.0), component=0)
    model.fix(displacement, on=mesh.face(domain, axis="y", value=0.0), component=1)
    model.fix(displacement, on=mesh.face(domain, axis="z", value=0.0), component=2)
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=1.0),
        component=0,
        value=0.002,
    )
    common = dict(
        displacement=displacement,
        material=material,
        external_force=None,
        constraints=model.constraints,
        study=model.study,
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(
            maximum_iterations=18, line_search="backtracking"
        ),
        progress=False,
        _experimental_distributed=True,
        name=f"portable_{kind}_step",
    )
    return (
        mechanics.implicit_creep_step(duration=0.2, **common)
        if kind == "creep"
        else mechanics.j2_plasticity_step(**common)
    )


def _state(step, kind: str):
    selected = (
        step.state.equivalent_creep_strain
        if kind == "creep"
        else step.state.equivalent_plastic_strain
    )
    cell_map = step.state.domain.topology.index_map(step.state.domain.topology.dim)
    owned = int(cell_map.size_local)
    points = len(step.state.stress.points)
    keys = np.asarray(step.state.domain.topology.original_cell_index[:owned])
    values = selected.values.reshape(-1)[: owned * points].reshape((owned, points))
    return dict(zip(keys.tolist(), values.tolist(), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "read"))
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()

    for kind in ("j2", "creep"):
        path = arguments.root.with_name(arguments.root.name + f"-{kind}")
        if arguments.action == "write":
            step = _step(kind)
            step.solve(until=0.1 if kind == "creep" else 0.5)
            manifest = step.save_checkpoint(path, portable=True)
            if MPI.COMM_WORLD.rank == 0:
                assert manifest.is_file()
            continue

        reference = _step(kind)
        reference.solve()
        restarted = _step(kind)
        restarted.load_checkpoint(path)
        restarted.solve()
        assert restarted.last_solve_info.completed_step
        np.testing.assert_allclose(
            restarted.solution.x.array,
            reference.solution.x.array,
            rtol=2.0e-8,
            atol=2.0e-10,
        )
        expected = _state(reference, kind)
        actual = _state(restarted, kind)
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=2.0e-8, atol=2.0e-10)


if __name__ == "__main__":
    main()
