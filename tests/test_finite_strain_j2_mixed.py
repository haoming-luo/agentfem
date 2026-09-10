"""Mixed displacement-pressure finite-strain J2 contracts."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

from agentfem import (
    constitutive,
    constraints,
    fields,
    mechanics,
    mesh,
    models,
    results,
    solvers,
    steps,
    studies,
)
from agentfem.mechanics.finite_strain_plasticity import (
    _mixed_hencky_j2_response,
)
from agentfem.mesh import abaqus
from agentfem.step_providers import step_capability


def _point(material, deformation_gradient, state):
    return constitutive.MaterialPointInput(
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=deformation_gradient,
        time=1.0,
        time_increment=1.0,
        properties=(),
        state_old=state,
        state_schema=material.state_schema,
    )


def _mixed_response(material, deformation_gradient, pressure, state):
    response = material.update(_point(material, deformation_gradient, state))
    jacobian = float(np.linalg.det(deformation_gradient))
    first_piola = (
        jacobian
        * response.cauchy_stress
        @ np.linalg.inv(deformation_gradient).T
    )
    return _mixed_hencky_j2_response(
        deformation_gradient=np.asarray((deformation_gradient,)),
        pressure=np.asarray((pressure,)),
        inverse_bulk_modulus=np.asarray((1.0 / material.bulk_modulus,)),
        first_piola=np.asarray((first_piola,)),
        cauchy_stress=np.asarray((response.cauchy_stress,)),
        tangent=np.asarray((response.consistent_tangent.reshape(3, 3, 3, 3),)),
        strain_energy_density=np.asarray((response.strain_energy_density,)),
        elastic_energy_density=np.asarray(
            (response.stored_energy_density_components["ELENER"],)
        ),
    )


def _fully_affine_constraint(target, deformation_gradient):
    """Constrain every P2 displacement node to one affine cube field."""

    displacement_space, maps = target.space.sub(0).collapse()
    del maps
    coordinates = np.asarray(
        displacement_space.tabulate_dof_coordinates(),
        dtype=float,
    )
    labels = np.arange(1, len(coordinates) + 1, dtype=np.int64)

    def label_at(point):
        distance = np.linalg.norm(coordinates - np.asarray(point), axis=1)
        index = int(np.argmin(distance))
        assert distance[index] < 1.0e-12
        return int(labels[index])

    anchor = label_at((0.0, 0.0, 0.0))
    references = (
        label_at((1.0, 0.0, 0.0)),
        label_at((0.0, 1.0, 0.0)),
        label_at((0.0, 0.0, 1.0)),
    )
    controls = {anchor, *references}
    equations = []
    for label, coordinate in zip(labels, coordinates, strict=True):
        if int(label) in controls:
            continue
        for component in (1, 2, 3):
            terms = [abaqus.EquationTerm(int(label), component, 1.0)]
            terms.extend(
                abaqus.EquationTerm(reference, component, -float(weight))
                for reference, weight in zip(
                    references,
                    coordinate,
                    strict=True,
                )
                if abs(float(weight)) > 1.0e-15
            )
            anchor_weight = 1.0 - float(np.sum(coordinate))
            if abs(anchor_weight) > 1.0e-15:
                terms.append(
                    abaqus.EquationTerm(
                        anchor,
                        component,
                        -anchor_weight,
                    )
                )
            equations.append(abaqus.LinearEquation(tuple(terms)))
    return constraints.abaqus_periodic_cell(
        target,
        nodes=abaqus.AbaqusNodeTable(labels=labels, coordinates=coordinates),
        equations=abaqus.AbaqusEquationSet(tuple(equations)),
        deformation_gradient=np.asarray(deformation_gradient, dtype=float),
        anchor_node=anchor,
        reference_nodes=references,
        name="fully_affine_p2_dg0_cube",
    )


def _fully_affine_constraint_2d(
    target,
    deformation_gradient,
    *,
    lattice=None,
):
    """Constrain every Q2 displacement node to one affine cell field."""

    displacement_space, maps = target.space.sub(0).collapse()
    del maps
    coordinates = np.asarray(
        displacement_space.tabulate_dof_coordinates(),
        dtype=float,
    )[:, :2]
    labels = np.arange(1, len(coordinates) + 1, dtype=np.int64)

    def label_at(point):
        distance = np.linalg.norm(coordinates - np.asarray(point), axis=1)
        index = int(np.argmin(distance))
        assert distance[index] < 1.0e-12
        return int(labels[index])

    selected_lattice = (
        np.eye(2) if lattice is None else np.asarray(lattice, dtype=float)
    )
    assert selected_lattice.shape == (2, 2)
    inverse_lattice = np.linalg.inv(selected_lattice)
    anchor = label_at((0.0, 0.0))
    references = tuple(
        label_at(selected_lattice[:, column]) for column in range(2)
    )
    controls = {anchor, *references}
    equations = []
    for label, coordinate in zip(labels, coordinates, strict=True):
        if int(label) in controls:
            continue
        lattice_coordinate = inverse_lattice @ coordinate
        for component in (1, 2):
            terms = [abaqus.EquationTerm(int(label), component, 1.0)]
            terms.extend(
                abaqus.EquationTerm(reference, component, -float(weight))
                for reference, weight in zip(
                    references,
                    lattice_coordinate,
                    strict=True,
                )
                if abs(float(weight)) > 1.0e-15
            )
            anchor_weight = 1.0 - float(np.sum(lattice_coordinate))
            if abs(anchor_weight) > 1.0e-15:
                terms.append(
                    abaqus.EquationTerm(
                        anchor,
                        component,
                        -anchor_weight,
                    )
                )
            equations.append(abaqus.LinearEquation(tuple(terms)))
    return constraints.abaqus_periodic_cell(
        target,
        nodes=abaqus.AbaqusNodeTable(labels=labels, coordinates=coordinates),
        equations=abaqus.AbaqusEquationSet(tuple(equations)),
        deformation_gradient=np.asarray(deformation_gradient, dtype=float),
        anchor_node=anchor,
        reference_nodes=references,
        name="fully_affine_q2_dpc1_square",
    )


def _periodic_q2_constraint_2d(target, deformation_gradient):
    """Build exact square periodicity for every Q2 boundary node."""

    displacement_space, maps = target.space.sub(0).collapse()
    del maps
    coordinates = np.asarray(
        displacement_space.tabulate_dof_coordinates(),
        dtype=float,
    )[:, :2]
    labels = np.arange(1, len(coordinates) + 1, dtype=np.int64)
    coordinate_to_label = {
        tuple(np.rint(coordinate / 1.0e-12).astype(np.int64)): int(label)
        for label, coordinate in zip(labels, coordinates, strict=True)
    }

    def label_at(point):
        key = tuple(np.rint(np.asarray(point) / 1.0e-12).astype(np.int64))
        return coordinate_to_label[key]

    anchor = label_at((0.0, 0.0))
    references = (label_at((1.0, 0.0)), label_at((0.0, 1.0)))
    controls = {anchor, *references}
    equations = []
    for label, coordinate in zip(labels, coordinates, strict=True):
        active_axes = tuple(
            axis
            for axis in range(2)
            if abs(float(coordinate[axis]) - 1.0) <= 1.0e-12
        )
        if not active_axes or int(label) in controls:
            continue
        wrapped = coordinate.copy()
        wrapped[list(active_axes)] = 0.0
        base = label_at(wrapped)
        for component in (1, 2):
            terms = [
                abaqus.EquationTerm(int(label), component, 1.0),
                abaqus.EquationTerm(base, component, -1.0),
            ]
            terms.extend(
                abaqus.EquationTerm(references[axis], component, -1.0)
                for axis in active_axes
            )
            terms.append(
                abaqus.EquationTerm(
                    anchor,
                    component,
                    float(len(active_axes)),
                )
            )
            equations.append(abaqus.LinearEquation(tuple(terms)))
    return constraints.abaqus_periodic_cell(
        target,
        nodes=abaqus.AbaqusNodeTable(labels=labels, coordinates=coordinates),
        equations=abaqus.AbaqusEquationSet(tuple(equations)),
        deformation_gradient=np.asarray(deformation_gradient, dtype=float),
        anchor_node=anchor,
        reference_nodes=references,
        name="periodic_q2_dpc1_square",
    )


def _periodic_p2_constraint(target, deformation_gradient):
    """Build exact cube periodicity for every P2 boundary node."""

    displacement_space, maps = target.space.sub(0).collapse()
    del maps
    coordinates = np.asarray(
        displacement_space.tabulate_dof_coordinates(),
        dtype=float,
    )
    labels = np.arange(1, len(coordinates) + 1, dtype=np.int64)
    coordinate_to_label = {
        tuple(np.rint(coordinate / 1.0e-12).astype(np.int64)): int(label)
        for label, coordinate in zip(labels, coordinates, strict=True)
    }

    def label_at(point):
        key = tuple(np.rint(np.asarray(point) / 1.0e-12).astype(np.int64))
        return coordinate_to_label[key]

    anchor = label_at((0.0, 0.0, 0.0))
    references = (
        label_at((1.0, 0.0, 0.0)),
        label_at((0.0, 1.0, 0.0)),
        label_at((0.0, 0.0, 1.0)),
    )
    controls = {anchor, *references}
    equations = []
    for label, coordinate in zip(labels, coordinates, strict=True):
        active_axes = tuple(
            axis
            for axis in range(3)
            if abs(float(coordinate[axis]) - 1.0) <= 1.0e-12
        )
        if not active_axes or int(label) in controls:
            continue
        wrapped = coordinate.copy()
        wrapped[list(active_axes)] = 0.0
        base = label_at(wrapped)
        for component in (1, 2, 3):
            terms = [
                abaqus.EquationTerm(int(label), component, 1.0),
                abaqus.EquationTerm(base, component, -1.0),
            ]
            terms.extend(
                abaqus.EquationTerm(references[axis], component, -1.0)
                for axis in active_axes
            )
            terms.append(
                abaqus.EquationTerm(
                    anchor,
                    component,
                    float(len(active_axes)),
                )
            )
            equations.append(abaqus.LinearEquation(tuple(terms)))
    return constraints.abaqus_periodic_cell(
        target,
        nodes=abaqus.AbaqusNodeTable(labels=labels, coordinates=coordinates),
        equations=abaqus.AbaqusEquationSet(tuple(equations)),
        deformation_gradient=np.asarray(deformation_gradient, dtype=float),
        anchor_node=anchor,
        reference_nodes=references,
        name="periodic_p2_dg0_cube",
    )


def _mixed_cube_step(*, deformation_gradient=None, poisson: float = 0.45):
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="tetrahedron",
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="mixed_finite_strain_j2_affine_cube",
    )
    unknown = model.field(fields.displacement_pressure(domain))
    material = constitutive.finite_strain_j2_logarithmic(
        young=20_000.0,
        poisson=poisson,
        yield_stress=80.0,
        hardening_modulus=200.0,
    )
    material_record = model.material(material)
    final_gradient = (
        np.diag((1.04, 0.99, 0.99))
        if deformation_gradient is None
        else np.asarray(deformation_gradient, dtype=float)
    )
    periodicity = model.constraint(
        _fully_affine_constraint(unknown, final_gradient)
    )
    step = model.step(
        target=unknown,
        material=material_record,
        constraints=periodicity,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-10,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )
    return step, model, unknown, material, material_record, periodicity


def _reduced_mixed_jacobian_direction_errors(
    step,
    periodicity,
    unknown,
    *,
    displacement_step: float = 2.0e-7,
    pressure_step: float = 1.0e-2,
):
    """Compare every reduced mixed block with an independent residual difference."""

    reduction = periodicity.reduction(1.0)
    transformation = reduction.matrix(step.solution.function_space.mesh.comm)
    residual_form = fem.form(step.residual_form)
    jacobian_form = fem.form(step.jacobian_form)
    transaction = step.state_transaction
    base = step.solution.x.array[reduction.independent_full_dofs].copy()
    full_tangent = None
    reduced_tangent = None

    def reduced_residual(values):
        step.solution.x.array[:] = reduction.reconstruct(values)
        step.solution.x.scatter_forward()
        transaction.refresh_trial(start_factor=1.0, target_factor=1.0)
        full = fem_petsc.assemble_vector(residual_form)
        full.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        reduced = transformation.createVecRight()
        transformation.multTranspose(full, reduced)
        selected = reduced.array_r.copy()
        full.destroy()
        reduced.destroy()
        return selected

    try:
        step.solution.x.array[:] = reduction.reconstruct(base)
        step.solution.x.scatter_forward()
        transaction.refresh_trial(start_factor=1.0, target_factor=1.0)
        full_tangent = fem_petsc.assemble_matrix(jacobian_form)
        full_tangent.assemble()
        reduced_tangent = full_tangent.PtAP(transformation)

        _pressure_space, pressure_map = unknown.space.sub(1).collapse()
        del _pressure_space
        pressure_map = np.asarray(
            pressure_map[0]
            if isinstance(pressure_map, (list, tuple))
            else pressure_map,
            dtype=np.int64,
        )
        pressure_full_dofs = set(int(value) for value in pressure_map)
        pressure_mask = np.asarray(
            [
                int(full_dof) in pressure_full_dofs
                for full_dof in reduction.independent_full_dofs
            ],
            dtype=bool,
        )
        assert np.any(pressure_mask)
        assert np.any(~pressure_mask)

        errors = {}
        for column_name, column_mask, finite_difference_step in (
            ("p", pressure_mask, pressure_step),
            ("u", ~pressure_mask, displacement_step),
        ):
            direction = np.zeros(reduction.reduced_size, dtype=float)
            direction[column_mask] = np.linspace(
                0.5,
                1.0,
                np.count_nonzero(column_mask),
            )
            direction /= np.linalg.norm(direction)
            numerical = (
                reduced_residual(base + finite_difference_step * direction)
                - reduced_residual(base - finite_difference_step * direction)
            ) / (2.0 * finite_difference_step)

            direction_vector = transformation.createVecRight()
            direction_vector.array[:] = direction
            tangent_vector = direction_vector.duplicate()
            reduced_tangent.mult(direction_vector, tangent_vector)
            analytical = tangent_vector.array_r.copy()
            direction_vector.destroy()
            tangent_vector.destroy()

            for row_name, row_mask in (
                ("u", ~pressure_mask),
                ("p", pressure_mask),
            ):
                difference = numerical[row_mask] - analytical[row_mask]
                scale = max(
                    np.linalg.norm(numerical[row_mask]),
                    np.linalg.norm(analytical[row_mask]),
                    1.0e-12,
                )
                errors[f"K{row_name}{column_name}"] = float(
                    np.linalg.norm(difference) / scale
                )
        return errors
    finally:
        step.solution.x.array[:] = reduction.reconstruct(base)
        step.solution.x.scatter_forward()
        transaction.prepare_resume()
        if full_tangent is not None:
            full_tangent.destroy()
        if reduced_tangent is not None:
            reduced_tangent.destroy()
        transformation.destroy()


def test_mixed_hencky_j2_point_tangent_matches_independent_difference():
    material = constitutive.finite_strain_j2_logarithmic(
        young=2_000.0,
        poisson=0.3,
        yield_stress=12.0,
        hardening_modulus=25.0,
        tangent_relative_step=5.0e-7,
    )
    state = material.state_schema.initial_state()
    deformation_gradient = np.asarray(
        ((1.06, 0.03, 0.0), (0.01, 0.97, 0.0), (0.0, 0.0, 1.01))
    )
    pressure = 7.5
    baseline = _mixed_response(
        material,
        deformation_gradient,
        pressure,
        state,
    )
    numerical = np.empty((3, 3, 3, 3), dtype=float)
    step = 2.0e-6
    for row in range(3):
        for column in range(3):
            plus = deformation_gradient.copy()
            minus = deformation_gradient.copy()
            plus[row, column] += step
            minus[row, column] -= step
            plus_piola = _mixed_response(material, plus, pressure, state)[
                "first_piola"
            ][0]
            minus_piola = _mixed_response(material, minus, pressure, state)[
                "first_piola"
            ][0]
            numerical[:, :, row, column] = (plus_piola - minus_piola) / (
                2.0 * step
            )
    np.testing.assert_allclose(
        baseline["tangent"][0],
        numerical,
        rtol=3.0e-4,
        atol=3.0e-4,
    )

    equilibrium_pressure = material.bulk_modulus * np.log(
        np.linalg.det(deformation_gradient)
    )
    condensed = _mixed_response(
        material,
        deformation_gradient,
        equilibrium_pressure,
        state,
    )
    standard = material.update(_point(material, deformation_gradient, state))
    np.testing.assert_allclose(
        condensed["cauchy_stress"][0],
        standard.cauchy_stress,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert condensed["strain_energy_density"][0] == pytest.approx(
        standard.strain_energy_density,
        rel=2.0e-12,
        abs=2.0e-12,
    )


def test_mixed_hencky_j2_tangent_remains_accurate_at_conditioning_ceiling():
    bulk_to_shear = 1.0e4
    poisson = (3.0 * bulk_to_shear - 2.0) / (
        6.0 * bulk_to_shear + 2.0
    )
    material = constitutive.finite_strain_j2_logarithmic(
        young=20_000.0,
        poisson=poisson,
        yield_stress=80.0,
        hardening_modulus=200.0,
    )
    assert material.bulk_modulus / material.shear_modulus == pytest.approx(
        bulk_to_shear,
        rel=2.0e-12,
    )
    state = material.state_schema.initial_state()
    # det(F)=1 removes the fixed-pressure geometric term at equilibrium and
    # directly exposes cancellation when the primal volumetric tangent is
    # subtracted from the complete material tangent.
    deformation_gradient = np.asarray(
        ((1.0, 0.08, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    standard = material.update(_point(material, deformation_gradient, state))
    assert material.state_schema.unpack(standard.state_new)[
        "equivalent_plastic_strain"
    ] > 0.0
    pressure = material.bulk_modulus * np.log(np.linalg.det(deformation_gradient))
    baseline = _mixed_response(
        material,
        deformation_gradient,
        pressure,
        state,
    )

    numerical = np.empty((3, 3, 3, 3), dtype=float)
    finite_difference_step = 2.0e-7
    for row in range(3):
        for column in range(3):
            plus = deformation_gradient.copy()
            minus = deformation_gradient.copy()
            plus[row, column] += finite_difference_step
            minus[row, column] -= finite_difference_step
            plus_piola = _mixed_response(material, plus, pressure, state)[
                "first_piola"
            ][0]
            minus_piola = _mixed_response(material, minus, pressure, state)[
                "first_piola"
            ][0]
            numerical[:, :, row, column] = (plus_piola - minus_piola) / (
                2.0 * finite_difference_step
            )

    difference = baseline["tangent"][0] - numerical
    relative_error = np.linalg.norm(difference) / np.linalg.norm(numerical)
    # The reference environment gives about 1.5e-6.  A 2e-5 contract retains
    # an order-of-magnitude platform margin while still detecting loss of the
    # deviatoric tangent through bulk-tangent cancellation.
    assert relative_error < 2.0e-5


def test_mixed_j2_affine_cube_uses_public_step_and_complete_block_system():
    final_gradient = np.diag((1.04, 0.99, 0.99))
    step, model, unknown, material, material_record, periodicity = _mixed_cube_step(
        deformation_gradient=final_gradient,
    )
    capability = step_capability(
        model,
        target=unknown,
        options={
            "material": material_record,
            "constraints": periodicity,
        },
    )
    assert capability["supported"]
    assert (
        capability["provider"]["name"]
        == "finite_strain_j2_mixed_affine_static"
    )

    result = step.solve_result()

    assert step.last_solve_info.converged
    assert step.mixed_formulation["blocks"] == ("Kuu", "Kup", "Kpu", "Kpp")
    assert periodicity.mismatch() < 2.0e-10
    pressure = unknown.collapsed_pressure().x.array
    expected_pressure = material.bulk_modulus * np.log(
        np.linalg.det(final_gradient)
    )
    np.testing.assert_allclose(
        pressure,
        expected_pressure,
        rtol=2.0e-8,
        atol=2.0e-8,
    )

    state = material.state_schema.initial_state()
    old_gradient = np.eye(3)
    expected = None
    for index in range(1, 5):
        factor = index / 4.0
        new_gradient = np.eye(3) + factor * (final_gradient - np.eye(3))
        expected = material.update(
            constitutive.MaterialPointInput(
                deformation_gradient_old=old_gradient,
                deformation_gradient_new=new_gradient,
                time=factor,
                time_increment=0.25,
                properties=(),
                state_old=state,
                state_schema=material.state_schema,
            )
        )
        state = expected.state_new
        old_gradient = new_gradient
    expected_piola = (
        np.linalg.det(final_gradient)
        * expected.cauchy_stress
        @ np.linalg.inv(final_gradient).T
    )
    np.testing.assert_allclose(
        step.response.first_piola_stress.values,
        np.broadcast_to(
            expected_piola,
            step.response.first_piola_stress.values.shape,
        ),
        rtol=3.0e-7,
        atol=3.0e-7,
    )
    assert {
        "U",
        "MEAN_KIRCHHOFF_STRESS",
        "P",
        "S",
        "PEEQ",
        "PDENER",
    } <= set(result.fields)
    state_summary = result.metadata["state"]
    assert state_summary["formulation"] == "mixed_u_p_quadratic_hencky"
    assert (
        state_summary["pressure_measure"]
        == "mean_kirchhoff_stress_positive_in_tension"
    )
    assert step.last_solve_info.increments[-1].checks[
        "pressure_block_residual_norm"
    ] < 1.0e-10
    refreshed = step.state_transaction.refresh_trial(
        start_factor=1.0,
        target_factor=1.0,
    )
    np.testing.assert_allclose(
        refreshed.cauchy_stress,
        step.response.cauchy_stress.values,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        refreshed.consistent_tangent.reshape((-1, 3, 3, 3, 3)),
        step.response.tangent.values,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        refreshed.strain_energy_density,
        step.response.strain_energy_density.values,
        rtol=0.0,
        atol=0.0,
    )


def test_mixed_j2_checkpoint_splits_and_atomically_restores_primary_fields(
    tmp_path,
):
    uninterrupted, _, uninterrupted_unknown, _, _, _ = _mixed_cube_step()
    uninterrupted.solve(until=0.5)
    checkpoint = uninterrupted.save_checkpoint(tmp_path / "mixed_j2")
    assert checkpoint.is_file()
    uninterrupted.solve()
    expected_solution = uninterrupted.solution.x.array.copy()
    expected_state = uninterrupted.response.snapshot()

    restarted, _, restarted_unknown, _, _, _ = _mixed_cube_step()
    restarted.load_checkpoint(checkpoint)
    assert restarted.accepted_load_factor == pytest.approx(0.5)
    np.testing.assert_allclose(
        restarted_unknown.value.x.array,
        restarted.accepted_solution.x.array,
        rtol=0.0,
        atol=0.0,
    )
    restarted.solve()

    np.testing.assert_allclose(
        restarted.solution.x.array,
        expected_solution,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    actual_state = restarted.response.snapshot()
    for name, values in expected_state.items():
        np.testing.assert_allclose(
            actual_state[name],
            values,
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    np.testing.assert_allclose(
        restarted_unknown.collapsed_pressure().x.array,
        uninterrupted_unknown.collapsed_pressure().x.array,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_mixed_j2_periodic_two_phase_cell_solves_free_displacement_fluctuations():
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (2, 2, 2),
        comm=MPI.COMM_SELF,
        cell_type="tetrahedron",
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="mixed_j2_two_phase_periodic_patch",
    )
    unknown = model.field(fields.displacement_pressure(domain))
    regions = mesh.partition_cells(
        domain,
        soft=lambda x: x[0] <= 0.5,
        stiff=lambda x: x[0] > 0.5,
    )
    model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=10_000.0,
            poisson=0.45,
            yield_stress=1.0e8,
        ),
        region=regions.soft,
    )
    model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=40_000.0,
            poisson=0.45,
            yield_stress=1.0e8,
        ),
        region=regions.stiff,
    )
    stretch = 1.02
    lateral = 1.0 / np.sqrt(stretch)
    final_gradient = np.diag((stretch, lateral, lateral))
    periodicity = model.constraint(
        _periodic_p2_constraint(unknown, final_gradient)
    )
    reduction = periodicity.reduction(1.0)
    pressure_dofs = len(unknown.collapsed_pressure().x.array)
    assert reduction.reduced_size > pressure_dofs

    step = model.step(
        target=unknown,
        constraints=periodicity,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
            maximum_iterations=25,
            line_search="backtracking",
        ),
        progress=False,
    )
    result = step.solve_result()

    assert step.last_solve_info.completed_step
    assert periodicity.mismatch() < 1.0e-11
    assert step.last_solve_info.increments[-1].checks[
        "pressure_block_residual_norm"
    ] < 1.0e-9
    displacement = unknown.collapsed_displacement()
    coordinates = np.asarray(
        displacement.function_space.tabulate_dof_coordinates(),
        dtype=float,
    )
    affine = ((final_gradient - np.eye(3)) @ coordinates.T).T
    actual = np.asarray(displacement.x.array).reshape((-1, 3))
    assert np.max(np.abs(actual - affine)) > 1.0e-5
    assert "MIXED_POTENTIAL" in result.fields
    assert "MEAN_KIRCHHOFF_STRESS" in result.fields

    # Verify all four blocks in independent coordinates instead of using the
    # nonlinear solve itself as indirect tangent evidence.
    block_errors = _reduced_mixed_jacobian_direction_errors(
        step,
        periodicity,
        unknown,
    )
    assert set(block_errors) == {"Kuu", "Kup", "Kpu", "Kpp"}
    for block, error in block_errors.items():
        assert error < 2.0e-3, f"{block} relative directional error={error:.6g}"


def test_plane_strain_q2_dpc1_mixed_j2_uses_three_pressure_modes(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=2,
            assumption="plane_strain",
        ),
        mesh=domain,
        name="mixed_j2_q2_dpc1_plane_strain_patch",
    )
    unknown = model.field(
        fields.displacement_pressure(
            domain,
            displacement_degree=2,
            pressure_family="DPC",
            pressure_degree=1,
        )
    )
    material = constitutive.finite_strain_j2_logarithmic(
        young=20_000.0,
        poisson=0.45,
        yield_stress=1.0e8,
    )
    material_record = model.material(material)
    final_gradient = np.asarray(((1.03, 0.02), (0.01, 0.99)))
    periodicity = model.constraint(
        _fully_affine_constraint_2d(unknown, final_gradient)
    )
    capability = step_capability(
        model,
        target=unknown,
        options={
            "material": material_record,
            "constraints": periodicity,
        },
    )
    assert capability["supported"]
    assert (
        capability["provider"]["name"]
        == "finite_strain_j2_mixed_affine_static"
    )

    step = model.step(
        target=unknown,
        material=material_record,
        constraints=periodicity,
        incrementation=steps.fixed(3),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-11,
            maximum_iterations=15,
            line_search="backtracking",
        ),
        progress=False,
    )
    output = tmp_path / "q2_dpc1_plane_strain.xdmf"
    result = step.solve_result(output=output, strict_output=True)

    embedded_gradient = np.eye(3)
    embedded_gradient[:2, :2] = final_gradient
    expected = material.update(
        _point(
            material,
            embedded_gradient,
            material.state_schema.initial_state(),
        )
    )
    expected_piola = (
        np.linalg.det(embedded_gradient)
        * expected.cauchy_stress
        @ np.linalg.inv(embedded_gradient).T
    )
    expected_pressure = material.bulk_modulus * np.log(
        np.linalg.det(embedded_gradient)
    )

    assert unknown.summary()["pressure_unknowns_per_cell"] == 3
    assert step.mixed_formulation["kinematics"] == "2D_plane_strain_F33_equals_1"
    assert periodicity.mismatch() < 1.0e-11
    np.testing.assert_allclose(
        step.state_transaction.mixed_pressure.values,
        expected_pressure,
        rtol=2.0e-9,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        step.response.first_piola_stress.values,
        np.broadcast_to(
            expected_piola,
            step.response.first_piola_stress.values.shape,
        ),
        rtol=4.0e-7,
        atol=4.0e-7,
    )
    energy = results.mixed_j2_elastic_energy_diagnostics(
        deformation_gradient=step.state_transaction.deformation_gradient,
        pressure=step.state_transaction.mixed_pressure,
        inverse_bulk_modulus=step.state_transaction.inverse_bulk_modulus,
        condensed_elastic_energy_density=(
            step.response.stored_energy_density_components["ELENER"]
        ),
        reference_volume=periodicity.reference_cell_volume,
    )
    assert energy.algebraic_identity_verified
    assert energy.maximum_absolute_pressure_constraint_residual < 2.0e-10
    assert energy.rms_pressure_constraint_residual < 2.0e-10
    assert energy.signed_energy_gap_density == pytest.approx(0.0, abs=1.0e-10)
    assert energy.primal_elastic_energy_density == pytest.approx(
        energy.condensed_elastic_energy_density,
        abs=1.0e-10,
    )
    assert "MEAN_KIRCHHOFF_STRESS" in result.fields
    assert "MEAN_KIRCHHOFF_STRESS_CELL" in result.fields
    exact_pressure = result.fields["MEAN_KIRCHHOFF_STRESS"]
    assert exact_pressure.location == "cells"
    assert exact_pressure.processing["representation"] == (
        "discontinuous_cell_moments"
    )
    assert exact_pressure.processing["space_family"] == "DPC"
    assert exact_pressure.processing["space_degree"] == 1
    assert exact_pressure.processing["visualization_requires_cell_recovery"]
    recovered_pressure = result.fields["MEAN_KIRCHHOFF_STRESS_CELL"]
    assert recovered_pressure.location == "cells"
    assert recovered_pressure.processing == {
        "source_position": "discontinuous_finite_element_dofs",
        "source_field": "MEAN_KIRCHHOFF_STRESS",
        "source_space_degree": 1,
        "method": "global_l2_projection",
        "representation": "cell_average",
        "target_space": "DG0",
        "postprocessed": True,
        "nodal_extrapolation": False,
        "interelement_smoothing": False,
        "material_boundary_averaging": False,
    }
    np.testing.assert_allclose(
        recovered_pressure.field.x.array,
        expected_pressure,
        rtol=2.0e-9,
        atol=2.0e-9,
    )
    assert output.is_file()
    assert (
        "MEAN_KIRCHHOFF_STRESS"
        in result.metadata["field_output_fields"]["omitted"]
    )
    assert (
        result.metadata["field_output_fields"]["recoveries"]
        ["MEAN_KIRCHHOFF_STRESS_CELL"]
        == "MEAN_KIRCHHOFF_STRESS"
    )
    with h5py.File(output.with_suffix(".h5"), "r") as h5:
        assert "MEAN_KIRCHHOFF_STRESS" not in h5["Frames/0000/Cell"]
        pressure_values = np.asarray(
            h5["Frames/0000/Cell/MEAN_KIRCHHOFF_STRESS_CELL"]
        )
    np.testing.assert_allclose(
        pressure_values,
        expected_pressure,
        rtol=2.0e-9,
        atol=2.0e-9,
    )
    with pytest.raises(
        ValueError,
        match="Request the explicitly recovered DG0 cell-average fields",
    ):
        results.write_result_fields(
            result,
            tmp_path / "invalid_exact_dpc.xdmf",
            names=("MEAN_KIRCHHOFF_STRESS",),
        )
    selected_output = tmp_path / "selected_pressure_cell.xdmf"
    selected = results.write_result_fields(
        result,
        selected_output,
        names=("U", "MEAN_KIRCHHOFF_STRESS_CELL"),
    )
    assert selected.field_names == ("U", "MEAN_KIRCHHOFF_STRESS_CELL")
    assert selected_output.is_file()
    assert step.last_solve_info.increments[-1].checks[
        "pressure_block_residual_norm"
    ] < 1.0e-10
    expected_solution = step.solution.x.array.copy()
    checkpoint = step.save_checkpoint(tmp_path / "q2_dpc1_plane_strain")
    assert checkpoint.is_file()
    step.solution.x.array[:] = 0.0
    step.accepted_solution.x.array[:] = 0.0
    step.load_checkpoint(checkpoint)
    np.testing.assert_allclose(
        step.solution.x.array,
        expected_solution,
        rtol=0.0,
        atol=0.0,
    )

    untouched = results.SimulationResult("invalid_output_request")
    untouched.add_field(
        exact_pressure.name,
        exact_pressure.field,
        unit=exact_pressure.unit,
        location=exact_pressure.location,
        description=exact_pressure.description,
        processing=exact_pressure.processing,
    )
    with pytest.raises(KeyError, match="UNKNOWN_FIELD"):
        results.write_result_fields(
            untouched,
            tmp_path / "unknown_field.xdmf",
            names=("UNKNOWN_FIELD",),
        )
    assert tuple(untouched.fields) == ("MEAN_KIRCHHOFF_STRESS",)

    recovered_pressure.processing["source_field"] = "UNRELATED_FIELD"
    with pytest.raises(ValueError, match="is not the DG0 cell average"):
        results.write_result_fields(
            result,
            tmp_path / "colliding_pressure_recovery.xdmf",
            names=("MEAN_KIRCHHOFF_STRESS_CELL",),
        )
    recovered_pressure.processing["source_field"] = "MEAN_KIRCHHOFF_STRESS"


def test_plane_strain_q2_dpc1_fresh_step_restart_matches_uninterrupted(tmp_path):
    def build_step():
        domain = mesh.rectangle(
            (0.0, 0.0),
            (1.0, 1.0),
            (1, 1),
            comm=MPI.COMM_SELF,
            cell_type="quadrilateral",
        )
        model = models.create(
            study=studies.nonlinear_static(
                physics="solid_mechanics",
                dimension=2,
                assumption="plane_strain",
            ),
            mesh=domain,
        )
        unknown = model.field(
            fields.displacement_pressure(
                domain,
                displacement_degree=2,
                pressure_family="DPC",
                pressure_degree=1,
            )
        )
        model.material(
            constitutive.finite_strain_j2_logarithmic(
                young=20_000.0,
                poisson=0.45,
                yield_stress=80.0,
                hardening_modulus=200.0,
            )
        )
        periodicity = model.constraint(
            _fully_affine_constraint_2d(
                unknown,
                np.asarray(((1.04, 0.02), (0.0, 0.99))),
            )
        )
        step = model.step(
            target=unknown,
            constraints=periodicity,
            incrementation=steps.fixed(4),
            solver_options=solvers.newton(
                relative_tolerance=1.0e-9,
                absolute_tolerance=1.0e-10,
                maximum_iterations=20,
                line_search="backtracking",
            ),
            progress=False,
        )
        return step, unknown

    uninterrupted, uninterrupted_unknown = build_step()
    uninterrupted.solve(until=0.5)
    checkpoint = uninterrupted.save_checkpoint(tmp_path / "q2_dpc1_fresh_step")
    uninterrupted.solve()

    restarted, restarted_unknown = build_step()
    restarted.load_checkpoint(checkpoint)
    restarted.solve()

    np.testing.assert_allclose(
        restarted.solution.x.array,
        uninterrupted.solution.x.array,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        restarted_unknown.collapsed_pressure().x.array,
        uninterrupted_unknown.collapsed_pressure().x.array,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    for name, expected in uninterrupted.response.snapshot().items():
        np.testing.assert_allclose(
            restarted.response.snapshot()[name],
            expected,
            rtol=1.0e-10,
            atol=1.0e-10,
        )


def test_plane_strain_q2_dpc1_preserves_fluctuation_and_four_block_tangent(
    tmp_path,
):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 2),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=2,
            assumption="plane_strain",
        ),
        mesh=domain,
        name="mixed_j2_q2_dpc1_two_phase_periodic_patch",
    )
    unknown = model.field(
        fields.displacement_pressure(
            domain,
            displacement_degree=2,
            pressure_family="DPC",
            pressure_degree=1,
        )
    )
    regions = mesh.partition_cells(
        domain,
        soft=lambda x: x[0] <= 0.5,
        stiff=lambda x: x[0] > 0.5,
    )
    model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=10_000.0,
            poisson=0.45,
            yield_stress=1.0e8,
        ),
        region=regions.soft,
    )
    model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=40_000.0,
            poisson=0.45,
            yield_stress=1.0e8,
        ),
        region=regions.stiff,
    )
    final_gradient = np.asarray(((1.0, 0.03), (0.0, 1.0)))
    periodicity = model.constraint(
        _periodic_q2_constraint_2d(unknown, final_gradient)
    )
    reduction = periodicity.reduction(1.0)
    pressure_dofs = len(unknown.collapsed_pressure().x.array)
    assert reduction.reduced_size > pressure_dofs

    output = results.output_plan(
        tmp_path / "plane_strain_periodic_history",
        requests=(results.periodic_cell_history(periodicity),),
        presentation=None,
        basename="plane_strain_periodic_history",
    )
    step = model.step(
        target=unknown,
        constraints=periodicity,
        # Keep the heterogeneous mixed path deterministic while avoiding a
        # platform-dependent backtracking branch at the coarse 1/3 load
        # increment.  The final four-block tangent is checked independently
        # below, so smaller physical increments strengthen rather than relax
        # the equilibrium contract.
        incrementation=steps.fixed(6),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-10,
            maximum_iterations=30,
            line_search="backtracking",
        ),
        output=output,
        progress=False,
    )
    result = step.solve_result()

    assert step.last_solve_info.completed_step
    assert periodicity.mismatch() < 1.0e-11
    assert step.last_solve_info.increments[-1].checks[
        "pressure_block_residual_norm"
    ] < 1.0e-9
    displacement = unknown.collapsed_displacement()
    coordinates = np.asarray(
        displacement.function_space.tabulate_dof_coordinates(),
        dtype=float,
    )[:, :2]
    affine = ((final_gradient - np.eye(2)) @ coordinates.T).T
    actual = np.asarray(displacement.x.array).reshape((-1, 2))
    assert np.max(np.abs(actual - affine)) > 1.0e-6
    assert np.ptp(step.state_transaction.mixed_pressure.values) > 1.0e-4
    energy = results.mixed_j2_elastic_energy_diagnostics(
        deformation_gradient=step.state_transaction.deformation_gradient,
        pressure=step.state_transaction.mixed_pressure,
        inverse_bulk_modulus=step.state_transaction.inverse_bulk_modulus,
        condensed_elastic_energy_density=(
            step.response.stored_energy_density_components["ELENER"]
        ),
        reference_volume=periodicity.reference_cell_volume,
    )
    assert energy.algebraic_identity_verified
    assert energy.pressure_constraint_defect_energy_density > 0.0
    assert abs(energy.pressure_orthogonality_density) < 2.0e-8
    assert energy.signed_energy_gap_density == pytest.approx(
        energy.pressure_constraint_defect_energy_density,
        abs=2.0e-8,
    )
    energy_scale = max(
        1.0,
        abs(energy.primal_elastic_energy_density),
        abs(energy.condensed_elastic_energy_density),
    )
    assert energy.primal_elastic_energy_density >= (
        energy.condensed_elastic_energy_density - 1.0e-12 * energy_scale
    )
    assert "MEAN_KIRCHHOFF_STRESS" in result.fields
    assert result.metadata["problem"]["numerical_formulation"]["kinematics"] == (
        "2D_plane_strain_F33_equals_1"
    )
    recorder = step.accepted_history_recorders["homogenized_history"]
    assert len(recorder.frames) == len(step.last_solve_info.increments) + 1
    assert recorder.frames[-1].elastic_energy_density == pytest.approx(
        energy.condensed_elastic_energy_density,
        rel=2.0e-13,
        abs=2.0e-13,
    )
    embedded_macro_gradient = recorder.frames[-1].deformation_gradient
    assert embedded_macro_gradient.shape == (3, 3)
    np.testing.assert_allclose(
        embedded_macro_gradient[:2, :2],
        final_gradient,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_array_equal(embedded_macro_gradient[2], (0.0, 0.0, 1.0))
    assert "homogenized_first_piola_stress" in result.histories
    block_errors = _reduced_mixed_jacobian_direction_errors(
        step,
        periodicity,
        unknown,
    )
    assert set(block_errors) == {"Kuu", "Kup", "Kpu", "Kpp"}
    for block, error in block_errors.items():
        assert error < 2.0e-3, f"{block} relative directional error={error:.6g}"


def test_plane_strain_homogenized_tangent_matches_hencky_elasticity(tmp_path):
    lattice = np.asarray(((1.0, 0.35), (0.0, 1.0)))
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    domain.geometry.x[:, 0] += lattice[0, 1] * domain.geometry.x[:, 1]
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=2,
            assumption="plane_strain",
        ),
        mesh=domain,
        name="homogeneous_q2_dpc1_effective_tangent",
    )
    unknown = model.field(
        fields.displacement_pressure(
            domain,
            displacement_degree=2,
            pressure_family="DPC",
            pressure_degree=1,
        )
    )
    material = constitutive.finite_strain_j2_logarithmic(
        young=1_000.0,
        poisson=0.3,
        yield_stress=1.0e8,
    )
    material_record = model.material(material)
    periodicity = model.constraint(
        _fully_affine_constraint_2d(
            unknown,
            np.eye(2),
            lattice=lattice,
        )
    )
    step = model.step(
        target=unknown,
        material=material_record,
        constraints=periodicity,
        incrementation=steps.fixed(1),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            maximum_iterations=10,
        ),
        progress=False,
    )
    simulation = step.solve_result()
    assert simulation.status == "completed"

    lift = periodicity.macro_gradient_lift()
    assert lift.component_order == ("11", "21", "12", "22")
    assert lift.values.shape == (len(step.solution.x.array), 4)
    tangent = results.homogenized_algorithmic_tangent(step, periodicity)

    shear = material.shear_modulus
    lame = material.bulk_modulus - 2.0 * shear / 3.0
    expected = np.asarray(
        (
            (lame + 2.0 * shear, 0.0, 0.0, lame),
            (0.0, shear, shear, 0.0),
            (0.0, shear, shear, 0.0),
            (lame, 0.0, 0.0, lame + 2.0 * shear),
        )
    )
    assert tangent.component_order == lift.component_order
    np.testing.assert_allclose(tangent.values, expected, rtol=2.0e-5, atol=2.0e-5)
    assert tangent.start_load_factor == pytest.approx(0.0)
    assert tangent.load_factor == pytest.approx(1.0)
    assert tangent.constraint_fingerprint == periodicity.scientific_identity()[
        "fingerprint"
    ]
    assert tangent.full_dofs == lift.values.shape[0]
    assert 0 < tangent.reduced_dofs <= tangent.full_dofs
    assert tangent.linear_solver == step.solver_options.linear_solver.summary()
    assert tangent.linearization_state_basis == "pre_increment_committed_state"
    assert tangent.response_generation > 0
    assert tangent.linearization_origin == pytest.approx((0.0, 0.0, 1.0))
    assert tangent.as_dict()["response_generation"] == tangent.response_generation
    assert tangent.as_dict()["linearization_origin"] == pytest.approx(
        tangent.linearization_origin
    )
    assert tangent.maximum_relative_equilibrium_sensitivity < 1.0e-12
    assert tangent.relative_major_symmetry_error < 1.0e-10

    free_macro = constraints.abaqus_periodic_cell(
        unknown,
        nodes=periodicity.nodes,
        equations=periodicity.equations,
        control_displacements=((0.0, None), (0.0, 0.0)),
        anchor_node=periodicity.anchor_node,
        reference_nodes=periodicity.reference_nodes,
        name="mixed_control_affine_cell",
    )
    with pytest.raises(NotImplementedError, match="every macroscopic control DOF"):
        free_macro.macro_gradient_lift()

    retained_runtime = step.state_transaction.snapshot_runtime_state()
    step.state_transaction.rollback_increment(
        accepted_factor=step.accepted_load_factor,
    )
    with pytest.raises(RuntimeError, match="not the retained linearization"):
        results.homogenized_algorithmic_tangent(step, periodicity)

    step.state_transaction.restore_runtime_state(retained_runtime)
    restored_tangent = results.homogenized_algorithmic_tangent(step, periodicity)
    np.testing.assert_array_equal(restored_tangent.values, tangent.values)

    checkpoint = step.save_checkpoint(tmp_path / "homogenized_tangent")
    step.load_checkpoint(checkpoint)
    with pytest.raises(RuntimeError, match="not the retained linearization"):
        results.homogenized_algorithmic_tangent(step, periodicity)


def test_three_dimensional_homogenized_tangent_matches_hencky_elasticity():
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="tetrahedron",
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="homogeneous_p2_dg0_effective_tangent",
    )
    unknown = model.field(
        fields.displacement_pressure(
            domain,
            displacement_degree=2,
            pressure_degree=0,
        )
    )
    material = constitutive.finite_strain_j2_logarithmic(
        young=1_000.0,
        poisson=0.3,
        yield_stress=1.0e8,
    )
    model.material(material)
    periodicity = model.constraint(_fully_affine_constraint(unknown, np.eye(3)))
    step = model.step(
        target=unknown,
        constraints=periodicity,
        incrementation=steps.fixed(1),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            maximum_iterations=10,
        ),
        progress=False,
    )
    step.solve()

    tangent = results.homogenized_algorithmic_tangent(step, periodicity)
    order = tuple((row, column) for column in range(3) for row in range(3))
    shear = material.shear_modulus
    lame = material.bulk_modulus - 2.0 * shear / 3.0
    expected = np.empty((9, 9), dtype=float)
    for a, (i, J) in enumerate(order):
        for b, (k, L) in enumerate(order):
            expected[a, b] = (
                lame * float(i == J) * float(k == L)
                + shear
                * (
                    float(i == k) * float(J == L)
                    + float(i == L) * float(J == k)
                )
            )

    assert tangent.component_order == (
        "11",
        "21",
        "31",
        "12",
        "22",
        "32",
        "13",
        "23",
        "33",
    )
    np.testing.assert_allclose(tangent.values, expected, rtol=3.0e-5, atol=3.0e-5)
    assert tangent.maximum_relative_equilibrium_sensitivity < 1.0e-11
    assert tangent.relative_major_symmetry_error < 1.0e-10


def test_mixed_j2_rejects_unverified_interpolation_and_parallel_mpc():
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="tetrahedron",
    )
    wrong = fields.displacement_pressure(
        domain,
        displacement_degree=1,
        pressure_degree=0,
    )
    material = constitutive.finite_strain_j2_logarithmic(
        young=1_000.0,
        poisson=0.3,
        yield_stress=10.0,
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
    )
    model.field(wrong)
    model.material(material)
    constraint = model.constraint(_fully_affine_constraint(wrong, np.eye(3)))
    capability = step_capability(
        model,
        target=wrong,
        options={"constraints": constraint},
    )
    assert not capability["supported"]

    hexa_domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="hexahedron",
    )
    hexa_model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=hexa_domain,
    )
    hexa_unknown = hexa_model.field(fields.displacement_pressure(hexa_domain))
    hexa_material = hexa_model.material(material)
    hexa_constraint = hexa_model.constraint(
        _fully_affine_constraint(hexa_unknown, np.eye(3))
    )
    hexa_capability = step_capability(
        hexa_model,
        target=hexa_unknown,
        options={
            "material": hexa_material,
            "constraints": hexa_constraint,
        },
    )
    assert not hexa_capability["supported"]
    with pytest.raises(ValueError, match="formulation evidence only for tetrahedron"):
        mechanics.finite_strain_j2_mixed_affine_problem(
            target=hexa_unknown,
            material=material,
            constraint=hexa_constraint,
        )


def test_mixed_j2_fails_closed_beyond_temporary_conditioning_guard():
    with pytest.raises(
        NotImplementedError,
        match="numerical-conditioning boundary",
    ):
        _mixed_cube_step(poisson=0.49999)

    # The guard is specific to the mixed tangent extraction and must not
    # narrow the underlying constitutive material's public parameter range.
    material = constitutive.finite_strain_j2_logarithmic(
        young=20_000.0,
        poisson=0.49999,
        yield_stress=80.0,
        hardening_modulus=200.0,
    )
    material.update(
        _point(
            material,
            np.diag((1.001, 1.0, 1.0)),
            material.state_schema.initial_state(),
        )
    )


def test_plane_strain_macro_embedding_requires_provider_owned_3d_tensors():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    displacement = fields.displacement(domain)
    with pytest.raises(ValueError, match="provider-owned plane-strain"):
        results.homogenize_periodic_cell(
            displacement,
            None,
            macro_deformation_gradient=np.eye(2),
            cell_reference_volume=1.0,
            load_factor=0.0,
        )
    invalid = np.eye(3)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite positive-J"):
        results.homogenize_periodic_cell(
            displacement,
            None,
            macro_deformation_gradient=invalid,
            cell_reference_volume=1.0,
            load_factor=0.0,
        )
