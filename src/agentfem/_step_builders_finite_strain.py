# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Finite-kinematics static solid and membrane Step builders."""

from __future__ import annotations

import numpy as np

from . import constraints as constraint_api
from . import loads as load_api


def fabric_membrane(
    model,
    *,
    target,
    material=None,
    constraints=None,
    solver_options=None,
    measure=None,
    name: str = "fabric_membrane",
    petsc_options_prefix: str = "agentfem_fabric_membrane_",
    incrementation=None,
    increments: int | None = None,
    load_factors=None,
    output=None,
    output_every: int | None = None,
    progress=True,
    status_file=None,
):
    """Build finite-kinematics in-plane equilibrium for woven membranes.

    This provider consumes only the tension and trellising channels. A
    nonzero bending law is rejected rather than silently discarded; bending
    belongs to a future director-shell discretization.
    """

    import ufl
    from dolfinx import fem
    from petsc4py import PETSc

    from . import problems, results
    from .constitutive.fabric import (
        DecoupledFabricSurface,
        fabric_membrane_internal_virtual_work,
    )

    if hasattr(model.mesh, "require_formulation"):
        model.mesh.require_formulation(
            "displacement",
            operation="model.step with finite-kinematics fabric membrane",
        )
    model.check(target=target, step_options={"material": material})
    if hasattr(model.study, "require"):
        model.study.require(analysis="nonlinear_static", physics="solid_mechanics")
    _reject_unconsumed_boundary_models(model, provider="fabric membrane")
    if (
        getattr(model.study, "dimension", None) != 2
        or getattr(model.study, "assumption", None) != "membrane"
    ):
        raise ValueError(
            "The fabric membrane provider requires studies.static_membrane()."
        )
    if tuple(getattr(target.value, "ufl_shape", ())) != (2,):
        raise ValueError(
            "The fabric membrane provider requires a 2D displacement field."
        )
    record = (
        _single_material(model, "model.step with a fabric membrane")
        if material is None
        else model._material_record(material)
    )
    properties = record.item
    if not isinstance(properties, DecoupledFabricSurface):
        raise TypeError("The fabric membrane provider requires DecoupledFabricSurface.")
    if np.any(np.abs(properties.bending_stiffness) > 1.0e-14):
        raise NotImplementedError(
            "The in-plane membrane provider cannot consume bending stiffness. "
            "Use a zero bending matrix; finite-rotation shell bending remains "
            "experimental."
        )

    selected_measure = measure
    if selected_measure is None:
        selected_measure = (
            record.region.measure if record.region is not None else ufl.dx
        )
    internal_residual = fabric_membrane_internal_virtual_work(
        target.value,
        target.test,
        properties,
        measure=selected_measure,
    )
    selected_constraints = _as_tuple(
        model.constraints if constraints is None else constraints
    )
    if any(
        isinstance(item, constraint_api.AbaqusPeriodicConstraint)
        for item in selected_constraints
    ):
        raise NotImplementedError(
            "Affine-periodic fabric membranes require a dedicated homogenization "
            "provider; ordinary strong constraints are supported here."
        )
    selected_output_every = _output_interval(output, output_every, default=1)
    selected_incrementation = _normalize_load_path(
        incrementation,
        increments=increments,
        load_factors=load_factors,
    )
    load_factor = fem.Constant(_domain(model.mesh), PETSc.ScalarType(0.0))
    residual = _load_path_residual(model, target, internal_residual, load_factor)
    jacobian = ufl.derivative(residual, target.value, target.trial)

    def membrane_acceptance():
        diagnostics = results.finite_strain_diagnostics(
            target,
            quadrature_degree=3,
        )
        minimum_j = float(diagnostics["minimum_quadrature_J"])
        energy = float(
            results.integral(
                properties.membrane_energy_ufl(target.value),
                measure=selected_measure,
                comm=_domain(model.mesh).comm,
            )
        )
        accepted = minimum_j > 0.0 and energy >= -1.0e-12
        return {
            "accepted": bool(accepted),
            "minimum_quadrature_J": minimum_j,
            "maximum_quadrature_J": float(diagnostics["maximum_quadrature_J"]),
            "maximum_displacement": float(diagnostics["maximum_displacement"]),
            "membrane_energy": energy,
            "message": (
                "fabric membrane deformation or stored energy is non-physical"
                if not accepted
                else ""
            ),
        }

    problem = problems.incremental_nonlinear(
        residual,
        target.value,
        factor=load_factor,
        value_path=constraint_api.prescribed_value_path(selected_constraints),
        update_load=model._time_update_callback(include_constraints=False),
        acceptance_check=membrane_acceptance,
        jacobian=jacobian,
        incrementation=selected_incrementation,
        constraints=selected_constraints,
        solver_options=solver_options,
        output_every=selected_output_every,
        progress=progress,
        status_file=status_file,
        name=name,
        petsc_options_prefix=petsc_options_prefix,
    )
    problem.material = properties
    problem.primary_fields = {
        "U": target,
        "FABRIC_GENERALIZED_STRAIN": "cell",
        "FABRIC_GENERALIZED_RESULTANT": "cell",
        "FABRIC_WARP_DIRECTION": "cell",
        "FABRIC_WEFT_DIRECTION": "cell",
        "SENER": "cell",
    }
    problem.result_field_role = "constitutive_projection"
    if output is not None and hasattr(output, "finalize"):
        # The declarative output plan recovers its requested fields for every
        # saved frame and registers the final frame in SimulationResult. Do not
        # project the same final fabric fields once before finalization.
        problem.result_field_factory = lambda: (target.value,)
    else:
        problem.result_field_factory = lambda: (
            target.value,
            *results.fabric_membrane_cell_fields(target, properties),
        )
    problem.result_field_recovery = lambda snapshot, *, variables: (
        results.fabric_membrane_cell_fields(
            snapshot.solution,
            properties,
            variables=variables,
        )
    )
    if output is not None and hasattr(output, "bind"):
        output.bind(problem, properties, has_external_power=bool(model.loads))
    return model.add_step(problem)


def hyperelastic(
    model,
    *,
    target,
    material=None,
    constraints=None,
    solver_options=None,
    measure=None,
    name: str = "hyperelastic",
    petsc_options_prefix: str = "agentfem_hyperelastic_",
    incrementation=None,
    increments: int | None = None,
    load_factors=None,
    output=None,
    output_every: int | None = None,
    progress=True,
    status_file=None,
):
    """Build displacement-based finite-strain equilibrium."""

    import ufl
    from dolfinx import fem
    from petsc4py import PETSc

    from . import problems
    from .constitutive import hyperelasticity

    if hasattr(model.mesh, "require_formulation"):
        model.mesh.require_formulation(
            "displacement",
            operation="model.step with finite-strain hyperelasticity",
        )
    model.check(target=target, step_options={"material": material})
    _reject_unconsumed_boundary_models(model, provider="hyperelastic")
    if hasattr(model.study, "require"):
        model.study.require(analysis="nonlinear_static", physics="solid_mechanics")
    record = (
        _single_material(model, "model.step with finite-strain hyperelasticity")
        if material is None
        else model._material_record(material)
    )
    properties = record.item
    if not hyperelasticity.is_finite_strain_hyperelastic(properties):
        raise TypeError("model.step requires a supported hyperelastic material here.")
    if not hyperelasticity.supports_hyperelastic_study(
        properties,
        dimension=getattr(model.study, "dimension", 0),
        assumption=getattr(model.study, "assumption", None),
    ):
        raise ValueError(
            "The Study dimension/assumption has no formulation for the "
            f"selected hyperelastic material {properties.name!r}."
        )
    selected_measure = measure
    if selected_measure is None:
        selected_measure = (
            record.region.measure if record.region is not None else ufl.dx
        )
    internal_residual = hyperelasticity.internal_virtual_work(
        target.value,
        target.test,
        properties,
        measure=selected_measure,
    )
    selected_constraints = _as_tuple(
        model.constraints if constraints is None else constraints
    )
    affine_constraints = [
        item
        for item in selected_constraints
        if isinstance(item, constraint_api.AbaqusPeriodicConstraint)
    ]
    selected_output_every = _output_interval(output, output_every, default=1)
    selected_incrementation = _normalize_load_path(
        incrementation,
        increments=increments,
        load_factors=load_factors,
    )
    if affine_constraints:
        if any(isinstance(item, load_api.AmplitudeLoad) for item in model.loads):
            raise NotImplementedError(
                "Amplitude-driven natural loads are not yet supported by the "
                "affine nonlinear path. Prescribed affine loading remains supported."
            )
        if len(affine_constraints) != 1 or len(selected_constraints) != 1:
            raise ValueError(
                "The affine hyperelastic path currently requires exactly one "
                "AbaqusPeriodicConstraint and no separate Dirichlet constraints."
            )
        residual = internal_residual
        if model.loads:
            residual -= model.external_force(target).expression
        jacobian = hyperelasticity.tangent(residual, target.value, target.trial)
        output_factors = (
            output.required_factors()
            if output is not None and hasattr(output, "required_factors")
            else ()
        )

        def finite_strain_acceptance():
            return _finite_strain_acceptance(
                model,
                target,
                properties,
                selected_measure,
                hyperelasticity,
            )

        problem = problems.affine_nonlinear(
            residual,
            target.value,
            jacobian=jacobian,
            constraint=affine_constraints[0],
            incrementation=selected_incrementation,
            solver_options=solver_options,
            output_every=selected_output_every,
            output_factors=output_factors,
            acceptance_check=finite_strain_acceptance,
            progress=progress,
            status_file=status_file,
            name=name,
        )
    else:
        load_factor = fem.Constant(_domain(model.mesh), PETSc.ScalarType(0.0))
        residual = _load_path_residual(model, target, internal_residual, load_factor)
        jacobian = hyperelasticity.tangent(residual, target.value, target.trial)

        def finite_strain_acceptance():
            return _finite_strain_acceptance(
                model,
                target,
                properties,
                selected_measure,
                hyperelasticity,
            )

        problem = problems.incremental_nonlinear(
            residual,
            target.value,
            factor=load_factor,
            value_path=constraint_api.prescribed_value_path(selected_constraints),
            update_load=model._time_update_callback(include_constraints=False),
            acceptance_check=finite_strain_acceptance,
            jacobian=jacobian,
            incrementation=selected_incrementation,
            constraints=selected_constraints,
            solver_options=solver_options,
            output_every=selected_output_every,
            progress=progress,
            status_file=status_file,
            name=name,
            petsc_options_prefix=petsc_options_prefix,
        )
    if output is not None and hasattr(output, "bind"):
        output.bind(problem, properties, has_external_power=bool(model.loads))
    return model.add_step(problem)


def mixed_hyperelastic(
    model,
    *,
    target,
    material=None,
    constraints=None,
    solver_options=None,
    measure=None,
    name: str = "mixed_hyperelastic",
    petsc_options_prefix: str = "agentfem_mixed_hyperelastic_",
    incrementation=None,
    increments: int | None = None,
    load_factors=None,
    output=None,
    output_every: int | None = None,
    progress=True,
    status_file=None,
):
    """Build verified P2-displacement/DG0-pressure equilibrium."""

    import ufl
    from dolfinx import fem
    from petsc4py import PETSc

    from . import problems
    from .constitutive import hyperelasticity

    if getattr(target, "kind", None) != "displacement_pressure":
        raise TypeError(
            "model.step with mixed hyperelasticity requires "
            "fields.displacement_pressure(...)."
        )
    if (
        int(getattr(target, "displacement_degree", -1)) != 2
        or int(getattr(target, "pressure_degree", -1)) != 0
    ):
        raise ValueError("The verified constant-pressure hybrid route requires P2/DG0.")
    if hasattr(model.mesh, "require_formulation"):
        model.mesh.require_formulation(
            "hybrid",
            operation="model.step with mixed hyperelasticity",
        )
    if hasattr(model.study, "require"):
        model.study.require(analysis="nonlinear_static", physics="solid_mechanics")
    _reject_unconsumed_boundary_models(model, provider="mixed hyperelastic")
    if (
        getattr(model.study, "dimension", None) == 2
        and getattr(model.study, "assumption", None) != "plane_strain"
    ):
        raise NotImplementedError(
            "The mixed Neo-Hookean 2D route currently represents plane strain."
        )
    record = (
        _single_material(model, "model.step with mixed hyperelasticity")
        if material is None
        else model._material_record(material)
    )
    properties = record.item
    if not isinstance(properties, hyperelasticity.MixedNeoHookeanProperties):
        raise TypeError("model.step requires MixedNeoHookeanProperties here.")
    selected_measure = measure or (
        record.region.measure if record.region is not None else ufl.dx
    )
    w = target.value
    displacement, pressure = ufl.split(w)
    test = ufl.TestFunction(target.space)
    trial = ufl.TrialFunction(target.space)
    internal_energy = (
        hyperelasticity.mixed_strain_energy_density(
            displacement,
            pressure,
            properties,
        )
        * selected_measure
    )
    residual = ufl.derivative(internal_energy, w, test)
    load_factor = fem.Constant(_domain(model.mesh), PETSc.ScalarType(0.0))
    residual = _load_path_residual(
        model,
        target.displacement,
        residual,
        load_factor,
    )
    jacobian = ufl.derivative(residual, w, trial)
    selected_constraints = _as_tuple(
        model.constraints if constraints is None else constraints
    )
    affine_constraints = [
        item
        for item in selected_constraints
        if isinstance(item, constraint_api.AbaqusPeriodicConstraint)
    ]
    selected_output_every = _output_interval(output, output_every, default=1)
    selected_incrementation = _normalize_load_path(
        incrementation,
        increments=increments,
        load_factors=load_factors,
    )

    def mixed_acceptance():
        from .results import finite_strain_diagnostics, integral

        displacement_field = target.collapsed_displacement()
        diagnostics = finite_strain_diagnostics(
            displacement_field,
            quadrature_degree=3,
        )
        minimum_j = float(diagnostics["minimum_quadrature_J"])
        return {
            "accepted": bool(minimum_j > 0.0),
            "minimum_quadrature_J": minimum_j,
            "maximum_quadrature_J": float(diagnostics["maximum_quadrature_J"]),
            "mixed_potential": float(
                integral(
                    hyperelasticity.mixed_strain_energy_density(
                        displacement,
                        pressure,
                        properties,
                    ),
                    measure=selected_measure,
                    comm=_domain(model.mesh).comm,
                )
            ),
            "message": (
                "deformation Jacobian became non-positive" if minimum_j <= 0.0 else ""
            ),
        }

    if affine_constraints:
        if len(affine_constraints) != 1 or len(selected_constraints) != 1:
            raise ValueError(
                "The mixed affine hyperelastic path requires exactly one "
                "AbaqusPeriodicConstraint and no separate Dirichlet constraints."
            )
        if model.loads:
            raise NotImplementedError(
                "Natural loads are not yet combined with the mixed affine "
                "periodic path; prescribe the macroscopic deformation gradient."
            )
        output_factors = (
            output.required_factors()
            if output is not None and hasattr(output, "required_factors")
            else ()
        )
        problem = problems.affine_nonlinear(
            residual,
            w,
            jacobian=jacobian,
            constraint=affine_constraints[0],
            incrementation=selected_incrementation,
            solver_options=solver_options,
            output_every=selected_output_every,
            output_factors=output_factors,
            acceptance_check=mixed_acceptance,
            progress=progress,
            status_file=status_file,
            name=name,
        )
    else:
        problem = problems.incremental_nonlinear(
            residual,
            w,
            factor=load_factor,
            value_path=constraint_api.prescribed_value_path(selected_constraints),
            update_load=model._time_update_callback(include_constraints=False),
            acceptance_check=mixed_acceptance,
            jacobian=jacobian,
            incrementation=selected_incrementation,
            constraints=selected_constraints,
            solver_options=solver_options,
            output_every=selected_output_every,
            progress=progress,
            status_file=status_file,
            name=name,
            petsc_options_prefix=petsc_options_prefix,
        )
    problem.primary_fields = {"U": target.displacement, "PRESSURE": target.pressure}
    problem.result_field_factory = lambda: (
        target.collapsed_displacement(name="U"),
        target.collapsed_pressure(name="PRESSURE"),
    )
    problem.snapshot_field_factory = lambda: (
        target.collapsed_displacement(name="U"),
        {"PRESSURE": target.collapsed_pressure(name="PRESSURE")},
    )
    if output is not None and hasattr(output, "bind"):
        output.bind(problem, properties, has_external_power=bool(model.loads))
    return model.add_step(problem)


def _finite_strain_acceptance(
    model,
    target,
    properties,
    measure,
    hyperelasticity,
):
    from .results import finite_strain_diagnostics, integral

    diagnostics = finite_strain_diagnostics(target, quadrature_degree=2)
    minimum_j = float(diagnostics["minimum_quadrature_J"])
    return {
        "accepted": bool(minimum_j > 0.0),
        "minimum_quadrature_J": minimum_j,
        "maximum_quadrature_J": float(diagnostics["maximum_quadrature_J"]),
        "recoverable_strain_energy": float(
            integral(
                hyperelasticity.strain_energy_density(target.value, properties),
                measure=measure,
                comm=_domain(model.mesh).comm,
            )
        ),
        "message": (
            "deformation Jacobian became non-positive" if minimum_j <= 0.0 else ""
        ),
    }


def _output_interval(output, output_every, *, default: int | None) -> int | None:
    """Resolve one nonlinear output cadence without duplicating API policy."""

    if output is not None and output_every is not None:
        raise ValueError("Pass output=... or output_every=..., not both.")
    selected = getattr(output, "every", None) if output is not None else output_every
    if selected is None:
        return default
    selected = int(selected)
    if selected <= 0:
        raise ValueError("Nonlinear output cadence must be a positive integer.")
    return selected


def _normalize_load_path(incrementation, *, increments=None, load_factors=None):
    """Normalize the shared nonlinear-static load-path contract."""

    from . import steps as step_api

    return step_api.normalize(
        incrementation,
        increments=increments,
        load_factors=load_factors,
    )


def _load_path_residual(model, target, internal_residual, load_factor):
    """Add proportional and explicitly amplitude-driven natural loads.

    Ordinary loads follow the accepted nonlinear load factor. Loads carrying
    their own amplitude are updated by the model callback and must not be
    multiplied by the same factor a second time.
    """

    if not model.loads:
        return internal_residual
    proportional = tuple(
        item for item in model.loads if not isinstance(item, load_api.AmplitudeLoad)
    )
    driven = tuple(
        item for item in model.loads if isinstance(item, load_api.AmplitudeLoad)
    )
    residual = internal_residual
    if proportional:
        residual -= (
            load_factor
            * model.external_force(
                target,
                loads=proportional,
            ).expression
        )
    if driven:
        residual -= model.external_force(target, loads=driven).expression
    return residual


def _single_material(model, caller: str):
    if len(model.materials) != 1:
        raise ValueError(f"{caller} requires material=... or exactly one material.")
    return model.materials[0]


def _reject_unconsumed_boundary_models(model, *, provider: str) -> None:
    """Prevent a nonlinear provider from silently dropping weak physics."""

    selected = tuple(getattr(model, "boundary_models", ()))
    if not selected:
        return
    names = tuple(getattr(item, "name", type(item).__name__) for item in selected)
    raise ValueError(
        f"The {provider} provider cannot consume boundary models {names}. "
        "Use a provider that explicitly owns their residual, tangent, dual "
        "reaction, work and energy semantics."
    )


def _as_tuple(item) -> tuple:
    if item is None:
        return ()
    if isinstance(item, tuple):
        return item
    if isinstance(item, list):
        return tuple(item)
    return (item,)


def _domain(mesh):
    if mesh is None:
        return None
    return getattr(mesh, "domain", mesh)


__all__ = ()
