"""Write and resume the public affine J2 periodic cell across MPI sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import (
    constitutive,
    fields,
    mesh,
    models,
    solvers,
    steps,
    studies,
)

from periodic_cube_fixture import periodic_unit_cube


class _FailingAcceptedObserver:
    def __init__(
        self,
        *,
        fail_rank: int | None = None,
        fail_snapshot_rank: int | None = None,
    ):
        self.accepted = 0
        self.fail_rank = fail_rank
        self.fail_snapshot_rank = fail_snapshot_rank
        self.snapshot_calls = 0

    def snapshot_runtime_state(self):
        self.snapshot_calls += 1
        if (
            self.fail_snapshot_rank is not None
            and MPI.COMM_WORLD.rank == self.fail_snapshot_rank
            and self.snapshot_calls == 2
        ):
            raise RuntimeError("injected distributed observer snapshot failure")
        return {"accepted": self.accepted}

    def restore_runtime_state(self, state) -> None:
        self.accepted = int(state["accepted"])

    def reset(self, _snapshot) -> None:
        self.accepted = 0

    def accept(self, _snapshot) -> None:
        self.accepted += 1
        if self.fail_rank is None or MPI.COMM_WORLD.rank == self.fail_rank:
            raise RuntimeError("injected distributed observer failure")


def _step(
    comm,
    *,
    two_phase: bool = False,
    mutate_material: bool = False,
    swap_regions: bool = False,
):
    fixture = periodic_unit_cube(comm)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="portable_affine_j2_periodic",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    first = constitutive.finite_strain_j2_logarithmic(
        young=200_000.0,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2_000.0,
    )
    if two_phase:
        regions = mesh.partition_cells(
            fixture.domain,
            left=lambda x: x[0] <= 0.5,
            right=lambda x: x[0] > 0.5,
        )
        second = constitutive.finite_strain_j2_logarithmic(
            young=120_000.0,
            poisson=0.3,
            yield_stress=121.0 if mutate_material else 120.0,
            hardening_modulus=1_000.0,
        )
        model.material(
            first,
            region=regions.right if swap_regions else regions.left,
        )
        model.material(
            second,
            region=regions.left if swap_regions else regions.right,
        )
        material = None
    else:
        material = model.material(first)
    periodicity = model.constraint(fixture.constraint(displacement))
    step = model.step(
        target=displacement,
        material=material,
        constraints=periodicity,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )
    return step, fixture, periodicity


def _material_point_reference(material, deformation_gradient):
    state = material.state_schema.initial_state()
    old_gradient = np.eye(3)
    response = None
    for index in range(1, 5):
        factor = index / 4.0
        new_gradient = np.eye(3) + factor * (
            deformation_gradient - np.eye(3)
        )
        response = material.update(
            constitutive.MaterialPointInput(
                deformation_gradient_old=old_gradient,
                deformation_gradient_new=new_gradient,
                time=factor,
                time_increment=0.25,
                properties=[],
                state_old=state,
                state_schema=material.state_schema,
            )
        )
        state = response.state_new.copy()
        old_gradient = new_gradient
    expected_first_piola = (
        np.linalg.det(deformation_gradient)
        * response.cauchy_stress
        @ np.linalg.inv(deformation_gradient).T
    )
    return response.state_new, expected_first_piola


def _verify_completed(step, fixture, periodicity) -> None:
    comm = step.solution.function_space.mesh.comm
    state = step.response.state.committed_state_vectors()
    points_per_cell = len(step.response.state.reference_field.points)
    cell_map = step.response.domain.topology.index_map(step.response.domain.topology.dim)
    owned_points = int(cell_map.size_local) * points_per_cell
    peeq = state[:owned_points, -1]
    global_min = comm.allreduce(float(np.min(peeq, initial=np.inf)), op=MPI.MIN)
    global_max = comm.allreduce(float(np.max(peeq, initial=-np.inf)), op=MPI.MAX)
    mismatch = float(periodicity.mismatch())
    measured = periodicity.measured_deformation_gradient(step.solution)
    expected_state, expected_first_piola = _material_point_reference(
        step.state_transaction.material,
        fixture.deformation_gradient,
    )
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
    state_error = comm.allreduce(local_state_error, op=MPI.MAX)
    stress_error = comm.allreduce(local_stress_error, op=MPI.MAX)

    if step.accepted_load_factor != 1.0:
        raise RuntimeError("Restarted public affine J2 path did not complete.")
    if global_min <= 0.0 or global_max - global_min > 1.0e-9:
        raise RuntimeError("Restarted public affine J2 state is inconsistent.")
    if mismatch > 1.0e-10:
        raise RuntimeError("Restarted periodic equations are not satisfied.")
    if state_error > 2.0e-8 or stress_error > 2.0e-6:
        raise RuntimeError(
            "Restarted quadrature state or stress differs from the material path."
        )
    if not np.allclose(
        measured,
        fixture.deformation_gradient,
        rtol=0.0,
        atol=2.0e-10,
    ):
        raise RuntimeError("Restarted macroscopic deformation is incorrect.")
    if [item.load_factor for item in step.accepted_increments] != [
        0.25,
        0.5,
        0.75,
        1.0,
    ]:
        raise RuntimeError("Restarted accepted-increment history is incomplete.")
    if len(step.execution_events) == 0:
        raise RuntimeError("Restarted execution evidence is empty.")


def _verify_two_phase_completed(step, reference, fixture, periodicity) -> None:
    comm = step.solution.function_space.mesh.comm
    if not isinstance(step.material, constitutive.QuadratureMaterialMap):
        raise RuntimeError("Two-phase public Step did not retain its material map.")
    if step.execution_context.material is not step.material:
        raise RuntimeError("Two-phase execution context lost the material map.")
    if step.accepted_load_factor != 1.0:
        raise RuntimeError("Restarted two-phase affine J2 path did not complete.")
    if periodicity.mismatch() > 1.0e-10:
        raise RuntimeError("Restarted two-phase periodic equations are not satisfied.")
    measured = periodicity.measured_deformation_gradient(step.solution)
    if not np.allclose(
        measured,
        fixture.deformation_gradient,
        rtol=0.0,
        atol=2.0e-10,
    ):
        raise RuntimeError("Restarted two-phase macroscopic deformation is wrong.")
    local_solution_error = float(
        np.max(
            np.abs(step.solution.x.array - reference.solution.x.array),
            initial=0.0,
        )
    )
    local_state_error = float(
        np.max(
            np.abs(
                step.response.state.committed_state_vectors()
                - reference.response.state.committed_state_vectors()
            ),
            initial=0.0,
        )
    )
    local_stress_error = float(
        np.max(
            np.abs(
                step.response.first_piola_stress.values
                - reference.response.first_piola_stress.values
            ),
            initial=0.0,
        )
    )
    errors = tuple(
        comm.allreduce(value, op=MPI.MAX)
        for value in (
            local_solution_error,
            local_state_error,
            local_stress_error,
        )
    )
    if errors[0] > 2.0e-9 or errors[1] > 2.0e-8 or errors[2] > 2.0e-5:
        raise RuntimeError(
            "Restarted two-phase solution differs from the uninterrupted path: "
            f"errors={errors}."
        )
    if [item.load_factor for item in step.accepted_increments] != [
        0.25,
        0.5,
        0.75,
        1.0,
    ]:
        raise RuntimeError("Two-phase accepted-increment history is incomplete.")


def _require_atomic_identity_rejection(step, manifest: Path) -> None:
    solution = step.solution.x.array.copy()
    state = step.response.state.committed_state_vectors().copy()
    try:
        step.load_checkpoint(manifest)
    except ValueError as exc:
        messages = step.solution.function_space.mesh.comm.allgather(str(exc))
        if len(set(messages)) != 1 or "material" not in messages[0]:
            raise RuntimeError(
                "Two-phase checkpoint identity rejection was inconsistent."
            ) from exc
    else:
        raise RuntimeError("A changed two-phase material identity was restored.")
    np.testing.assert_array_equal(step.solution.x.array, solution)
    np.testing.assert_array_equal(
        step.response.state.committed_state_vectors(),
        state,
    )
    if step.accepted_load_factor != 0.0:
        raise RuntimeError("Rejected checkpoint changed the accepted coordinate.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "write",
            "read",
            "reject-inconsistent",
            "reject-material",
            "reject-regions",
            "fail-observer",
            "fail-observer-rank-zero",
            "fail-observer-snapshot-rank-zero",
        ),
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--two-phase", action="store_true")
    arguments = parser.parse_args()

    comm = MPI.COMM_WORLD
    step, fixture, periodicity = _step(
        comm,
        two_phase=arguments.two_phase,
        mutate_material=arguments.action == "reject-material",
        swap_regions=arguments.action == "reject-regions",
    )
    if arguments.action in {"reject-material", "reject-regions"}:
        if not arguments.two_phase:
            raise RuntimeError("Regional identity checks require --two-phase.")
        _require_atomic_identity_rejection(step, arguments.root)
        return
    if arguments.action in {
        "fail-observer",
        "fail-observer-rank-zero",
        "fail-observer-snapshot-rank-zero",
    }:
        observer = _FailingAcceptedObserver(
            fail_rank=(
                0 if arguments.action == "fail-observer-rank-zero" else None
            ),
            fail_snapshot_rank=(
                0
                if arguments.action == "fail-observer-snapshot-rank-zero"
                else None
            ),
        )
        step.accepted_observers = (observer,)
        initial_solution = step.solution.x.array.copy()
        initial_state = step.response.state.committed_state_vectors().copy()
        try:
            step.solve()
        except RuntimeError as exc:
            messages = comm.allgather(str(exc))
            if (
                len(set(messages)) != 1
                or "observer" not in messages[0]
                or "failure" not in messages[0]
            ):
                raise RuntimeError(
                    "Distributed observer failure was not rank-consistent."
                ) from exc
        else:
            raise RuntimeError("Distributed observer failure did not propagate.")
        np.testing.assert_array_equal(step.solution.x.array, initial_solution)
        np.testing.assert_array_equal(
            step.response.state.committed_state_vectors(),
            initial_state,
        )
        if (
            observer.accepted != 0
            or step.accepted_load_factor != 0.0
            or step.state_transaction.accepted_factor != 0.0
            or step.accepted_increments
            or step.attempted_increments
        ):
            raise RuntimeError(
                "Distributed observer failure split the accepted lifecycle."
            )
        return
    if arguments.action == "reject-inconsistent":
        step.solve(until=0.5)
        if comm.rank == 0:
            step.solution.x.array[0] += 1.0e-4
        try:
            step.save_checkpoint(arguments.root)
        except RuntimeError as exc:
            messages = comm.allgather(str(exc))
            if len(set(messages)) != 1 or "U equals U_ACCEPTED" not in messages[0]:
                raise RuntimeError(
                    "Collective affine checkpoint rejection was inconsistent."
                ) from exc
            return
        raise RuntimeError(
            "An inconsistent rank-local affine state was checkpointed."
        )
    if arguments.action == "write":
        step.solve(until=0.5)
        manifest = step.save_checkpoint(arguments.root)
        if comm.rank == 0:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload["writer_rank_count"] != comm.size:
                raise RuntimeError("Checkpoint writer-rank provenance is incorrect.")
            if not payload["portable"]:
                raise RuntimeError("Public affine checkpoint is not portable.")
        return

    reference = None
    if arguments.two_phase:
        reference, _, _ = _step(comm, two_phase=True)
        reference.solve()
    step.load_checkpoint(arguments.root)
    if step.accepted_load_factor != 0.5:
        raise RuntimeError("Portable affine J2 coordinate was not restored.")
    step.solve()
    if arguments.two_phase:
        _verify_two_phase_completed(step, reference, fixture, periodicity)
    else:
        _verify_completed(step, fixture, periodicity)


if __name__ == "__main__":
    main()
