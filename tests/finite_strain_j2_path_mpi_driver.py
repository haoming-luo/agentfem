"""Two-rank public J2 solve along an unload and non-proportional Fbar path."""

from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np
from mpi4py import MPI

import agentfem
from agentfem import (
    constraints,
    constitutive,
    fields,
    models,
    solvers,
    steps,
    studies,
)

from periodic_cube_fixture import periodic_unit_cube


def _path():
    coordinates = (0.0, 0.35, 0.65, 1.0)
    loaded = np.diag((1.04, 1.0 / np.sqrt(1.04), 1.0 / np.sqrt(1.04)))
    unloaded = np.diag((1.01, 1.0 / np.sqrt(1.01), 1.0 / np.sqrt(1.01)))
    final = unloaded.copy()
    final[0, 1] = 0.03
    return coordinates, (np.eye(3), loaded, unloaded, final)


def _material_reference(material, coordinates, gradients):
    state = material.state_schema.initial_state()
    old_gradient = gradients[0]
    response = None
    first_piola_history = [np.zeros((3, 3), dtype=float)]
    for start, target, new_gradient in zip(
        coordinates[:-1], coordinates[1:], gradients[1:]
    ):
        response = material.update(
            constitutive.MaterialPointInput(
                deformation_gradient_old=old_gradient,
                deformation_gradient_new=new_gradient,
                time=target,
                time_increment=target - start,
                properties=[],
                state_old=state,
                state_schema=material.state_schema,
            )
        )
        state = response.state_new.copy()
        old_gradient = new_gradient
        first_piola_history.append(
            np.linalg.det(new_gradient)
            * response.cauchy_stress
            @ np.linalg.inv(new_gradient).T
        )
    return state, first_piola_history[-1], tuple(first_piola_history)


def main() -> None:
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        raise RuntimeError(
            "finite_strain_j2_path_mpi_driver.py is a two-rank evidence test."
        )
    expected_version = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]["version"]
    if agentfem.__version__ != expected_version:
        raise RuntimeError(
            "MPI evidence imported a different AgentFEM version: "
            f"expected {expected_version}, got {agentfem.__version__} from "
            f"{Path(agentfem.__file__).resolve()}."
        )
    fixture = periodic_unit_cube(comm)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="finite_strain_j2_nonproportional_path_mpi",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = constitutive.finite_strain_j2_logarithmic(
        young=200_000.0,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2_000.0,
    )
    material_record = model.material(material)
    coordinates, gradients = _path()
    macro_path = constraints.deformation_gradient_path(
        coordinates,
        gradients,
        name="tension_unload_shear",
    )
    periodicity = model.constraint(
        constraints.abaqus_periodic_cell(
            displacement,
            nodes=fixture.nodes,
            equations=fixture.equations,
            deformation_gradient_path=macro_path,
            anchor_node=fixture.anchor_node,
            reference_nodes=fixture.reference_nodes,
        )
    )
    step = model.step(
        target=displacement,
        material=material_record,
        constraints=periodicity,
        incrementation=steps.at(*coordinates[1:]),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        output_every=1,
        progress=False,
    )
    result = step.solve_result()
    expected_state, expected_first_piola, piola_history = _material_reference(
        material, coordinates, gradients
    )
    state = step.response.state.committed_state_vectors()
    points_per_cell = len(step.response.state.reference_field.points)
    cell_map = step.response.domain.topology.index_map(
        step.response.domain.topology.dim
    )
    owned_points = int(cell_map.size_local) * points_per_cell
    local_state_error = float(
        np.max(np.abs(state[:owned_points] - expected_state), initial=0.0)
    )
    local_stress_error = float(
        np.max(
            np.abs(
                step.response.first_piola_stress.owned_values
                - expected_first_piola
            ),
            initial=0.0,
        )
    )
    state_error = float(comm.allreduce(local_state_error, op=MPI.MAX))
    stress_error = float(comm.allreduce(local_stress_error, op=MPI.MAX))

    if result.status != "completed" or not step.last_solve_info.converged:
        raise RuntimeError("Distributed non-proportional J2 path did not complete.")
    if [item.load_factor for item in step.accepted_increments] != list(
        coordinates[1:]
    ):
        raise RuntimeError("Distributed non-proportional path lost a knot.")
    if state_error > 5.0e-8 or stress_error > 2.0e-5:
        raise RuntimeError(
            "Distributed non-proportional path differs from its material-point "
            f"oracle: state={state_error:.6g}, stress={stress_error:.6g}."
        )
    if periodicity.mismatch() > 1.0e-10:
        raise RuntimeError("Distributed non-proportional periodicity failed.")
    if not np.allclose(
        periodicity.measured_deformation_gradient(displacement),
        gradients[-1],
        rtol=0.0,
        atol=2.0e-10,
    ):
        raise RuntimeError("Distributed final macroscopic gradient is incorrect.")
    identity = periodicity.scientific_identity()
    if identity["deformation_gradient_path"]["fingerprint"] != macro_path.summary()[
        "fingerprint"
    ]:
        raise RuntimeError("Distributed path identity was not preserved.")
    final_tangent = (gradients[-1] - gradients[-2]) / (
        coordinates[-1] - coordinates[-2]
    )
    expected_reaction = periodicity.reference_cell_volume * float(
        np.sum(expected_first_piola * final_tangent)
    )
    actual_reaction = result.quantity("affine_path_generalized_reaction")
    if not np.isclose(actual_reaction, expected_reaction, rtol=8.0e-6, atol=8.0e-6):
        raise RuntimeError(
            "Distributed affine generalized reaction differs from "
            "V*P:dF/dlambda."
        )
    if not np.allclose(
        result.quantity("affine_constraint_force_resultant"),
        np.zeros(3),
        rtol=0.0,
        atol=2.0e-7,
    ):
        raise RuntimeError("Distributed affine physical resultant is not zero.")
    expected_work = periodicity.reference_cell_volume * sum(
        0.5 * float(np.sum((left_P + right_P) * (right_F - left_F)))
        for left_P, right_P, left_F, right_F in zip(
            piola_history[:-1],
            piola_history[1:],
            gradients[:-1],
            gradients[1:],
        )
    )
    if not np.isclose(
        result.quantity("affine_constraint_path_work"),
        expected_work,
        rtol=8.0e-6,
        atol=8.0e-6,
    ):
        raise RuntimeError(
            "Distributed affine path work differs from material-point work."
        )


if __name__ == "__main__":
    main()
