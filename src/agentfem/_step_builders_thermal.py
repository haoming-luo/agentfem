"""Linear and thermal scientific Step builders.\n\nThis private family module keeps steady and transient heat-transfer lowering\nindependent from nonlinear solid, inelastic, modal, and dynamics builders.\n"""

from __future__ import annotations

import numpy as np

from . import constraints as constraint_api
from . import loads as load_api
from .materials.properties import constant_volumetric_heat_capacity


def linear_static(
    model,
    *,
    target,
    K=None,
    F=None,
    constraints=None,
    solver_options=None,
    name: str = "linear_static",
):
    """Build and register a linear static or steady conduction Step."""

    from . import operators, problems

    selected_constraints = model.constraints if constraints is None else constraints
    model.check(
        target=target,
        step_options={
            "K": K,
            "F": F,
            "constraints": selected_constraints,
        },
    )
    update_at_step_end = model._time_update_callback()
    if update_at_step_end is not None:
        update_at_step_end(1.0)

    if getattr(model.study, "is_heat_transfer", False):
        boundary_stiffness, boundary_source = _thermal_boundary_terms(model, target)
        if K is None:
            K = model.conduction(target)
        if boundary_stiffness:
            K = operators.combine(
                K,
                *boundary_stiffness,
                name="K_thermal",
                kind="conduction_and_exchange",
            )
        if F is None:
            sources = []
            if model.loads:
                sources.append(model.external_force(target))
            sources.extend(boundary_source)
            F = (
                operators.combine(*sources, name="Q", kind="thermal_source")
                if sources
                else operators.heat_source_vector(0.0, target)
            )
    else:
        K = K if K is not None else model.stiffness(target)
        foundation_models = tuple(
            item
            for item in model.boundary_models
            if item.__class__.__name__ == "ElasticFoundation"
        )
        foundation_terms = tuple(item.operator(target) for item in foundation_models)
        if foundation_terms:
            K = operators.combine(
                K,
                *foundation_terms,
                name="K_with_foundation",
                kind="solid_and_foundation_stiffness",
            )
        if F is None:
            if model.loads:
                F = model.external_force(target)
            else:
                value_shape = tuple(getattr(target.value, "ufl_shape", ()))
                zero = (
                    0.0
                    if not value_shape
                    else tuple(0.0 for _ in range(value_shape[0]))
                )
                F = model.external_force(
                    target,
                    load=load_api.body_force(
                        zero,
                        target=target,
                        name="zero_external_force",
                    ),
                )

    result_field_factory = None
    if (
        getattr(model.study, "is_solid_mechanics", False)
        and model.materials
        and all(_thermal_expansion_is_zero(record.item) for record in model.materials)
    ):
        assignments = tuple(model.materials)

        def result_field_factory(requested=None):
            from . import results

            isotropic = all(
                hasattr(record.item, "young") and hasattr(record.item, "poisson")
                for record in assignments
            )
            defaults = results.preselected_fields(
                physics="solid_mechanics",
                finite_strain=False,
            )[1:]
            variables = tuple(defaults if requested is None else requested)
            if (
                not isotropic
                and requested is None
                and getattr(model.study, "assumption", None) == "plane_strain"
            ):
                variables = tuple(item for item in variables if item != "MISES")
            return results.small_strain_partition_fields(
                target,
                assignments,
                study=model.study,
                variables=variables,
            )

    step = problems.linear_static(
        K,
        F,
        study=model.study,
        unknown=target,
        constraints=selected_constraints,
        solver_options=solver_options,
        result_field_factory=result_field_factory,
        name=name,
    )
    if not getattr(model.study, "is_heat_transfer", False):
        step.constraint_assets = (
            *constraint_api.constraint_assets(selected_constraints),
            *foundation_models,
        )
    return model.add_step(step)


def heat_transfer(
    model,
    *,
    target,
    dt: float,
    steps: int,
    material=None,
    C=None,
    K=None,
    Q=None,
    constraints=None,
    solver_options=None,
    update_load=None,
    save_every: int | None = None,
    print_every: int | None = None,
    progress=True,
    status_file=None,
    checkpoint=None,
    name: str = "transient_heat",
):
    """Build and register one implicit-Euler heat-transfer Step."""

    import ufl
    from dolfinx import fem

    from . import operators, problems

    selected_constraints = model.constraints if constraints is None else constraints
    model.check(
        target=target,
        step_options={
            "material": material,
            "K": K,
            "F": Q,
            "dt": dt,
            "steps": steps,
            "constraints": selected_constraints,
        },
    )
    if hasattr(model.study, "require"):
        model.study.require(
            analysis="first_order_transient",
            physics="heat_transfer",
        )
    records = (
        (model._material_record(material),)
        if material is not None
        else tuple(model.materials)
    )
    if not records:
        raise ValueError("A transient heat Step requires at least one material.")
    previous = fem.Function(target.space, name="TemperaturePrevious")
    previous.x.array[:] = target.value.x.array
    previous.x.scatter_forward()
    if any(
        bool(getattr(record.item, "state_dependent_heat_transfer", False))
        for record in records
    ):
        if C is not None or K is not None:
            raise ValueError(
                "Temperature-dependent heat transfer builds one consistent "
                "enthalpy/conduction residual. Do not also pass C= or K=."
            )
        return model.add_step(
            _nonlinear_heat_transfer(
                model,
                target=target,
                previous=previous,
                records=records,
                dt=dt,
                steps=steps,
                Q=Q,
                constraints=constraints,
                solver_options=solver_options,
                update_load=update_load,
                save_every=save_every,
                print_every=print_every,
                progress=progress,
                status_file=status_file,
                checkpoint=checkpoint,
                name=name,
            )
        )
    capacity = model.heat_capacity(target, material) if C is None else C
    stiffness = model.conduction(target, material) if K is None else K
    boundary_stiffness, boundary_source = _thermal_boundary_terms(model, target)
    if boundary_stiffness:
        stiffness = operators.combine(
            stiffness,
            *boundary_stiffness,
            name="K_thermal",
            kind="conduction_and_exchange",
        )
    source = Q
    if source is None and model.loads:
        source = model.external_force(target)
    if boundary_source:
        source = operators.combine(
            *((() if source is None else (source,)) + tuple(boundary_source)),
            name="Q_thermal",
            kind="thermal_source",
        )
    history_parts = []
    for index, record in enumerate(records):
        if len(records) > 1 and record.region is None:
            raise ValueError(
                "Multiple-material heat history requires a region for every material."
            )
        history_parts.append(
            operators.heat_capacity_vector(
                previous,
                target,
                constant_volumetric_heat_capacity(record.item),
                measure=record.region.measure if record.region is not None else ufl.dx,
            ).renamed(
                "Q_capacity_history"
                if len(records) == 1
                else f"Q_capacity_{getattr(record.region, 'name', index)}"
            )
        )
    history = (
        history_parts[0]
        if len(history_parts) == 1
        else operators.combine(
            *history_parts,
            name="Q_capacity_history",
            kind="partitioned_heat_capacity_history",
        )
    )
    return model.add_step(
        problems.first_order_transient_run(
            capacity=capacity,
            stiffness=stiffness,
            history=history,
            source=source,
            current=target.value,
            previous=previous,
            dt=dt,
            steps=steps,
            study=model.study,
            constraints=selected_constraints,
            solver_options=solver_options,
            update_load=model._time_update_callback(update_load),
            save_every=save_every,
            print_every=print_every,
            progress=progress,
            status_file=status_file,
            checkpoint_policy=checkpoint,
            name=name,
        )
    )


def _nonlinear_heat_transfer(
    model,
    *,
    target,
    previous,
    records,
    dt,
    steps,
    Q,
    constraints,
    solver_options,
    update_load,
    save_every,
    print_every,
    progress,
    status_file,
    checkpoint,
    name,
):
    """Build conservative ``k(T), c_p(T)`` implicit heat transfer."""

    import ufl

    from . import operators, problems
    from .diagnostics import StateDependentThermalBalanceMonitor

    temperature = target.value
    test = ufl.TestFunction(target.space)
    direction = ufl.TrialFunction(target.space)
    residual = 0
    content_form = 0
    for record in records:
        if len(records) > 1 and record.region is None:
            raise ValueError(
                "Multiple-material nonlinear heat transfer requires a region "
                "for every material."
            )
        selected = record.item
        measure = record.region.measure if record.region is not None else ufl.dx
        if not hasattr(selected, "conductivity") or not hasattr(
            selected, "specific_heat"
        ):
            raise ValueError(
                f"Material {_describe(selected)!r} does not define conductivity "
                "and specific heat."
            )
        conductivity = (
            selected.conductivity_at(temperature)
            if hasattr(selected, "conductivity_at")
            else selected.conductivity
        )
        if hasattr(selected, "volumetric_enthalpy"):
            current_enthalpy = selected.volumetric_enthalpy(temperature)
            previous_enthalpy = selected.volumetric_enthalpy(previous)
        else:
            capacity = selected.density * selected.specific_heat
            current_enthalpy = capacity * temperature
            previous_enthalpy = capacity * previous
        residual += (
            (current_enthalpy - previous_enthalpy) / float(dt) * test
            + conductivity * ufl.dot(ufl.grad(temperature), ufl.grad(test))
        ) * measure
        content_form += current_enthalpy * measure

    outward_forms = []
    boundary_sources = []
    unsupported = []
    for boundary in model.boundary_models:
        if hasattr(boundary, "residual"):
            residual += boundary.residual(temperature, test)
            if hasattr(boundary, "outward_heat_rate_form"):
                outward_forms.append(boundary.outward_heat_rate_form(temperature))
            if hasattr(boundary, "source"):
                boundary_sources.append(boundary.source(target))
        else:
            unsupported.append(getattr(boundary, "name", type(boundary).__name__))
    if unsupported:
        raise ValueError(
            "Nonlinear heat transfer cannot consume these boundary models: "
            f"{unsupported}."
        )

    source = Q
    if source is None and model.loads:
        source = model.external_force(target)
    if source is not None:
        residual -= getattr(source, "expression", source)
    monitor_source = source
    if boundary_sources:
        monitor_source = operators.combine(
            *((() if source is None else (source,)) + tuple(boundary_sources)),
            name="Q_thermal_ledger",
            kind="thermal_source",
        )
    jacobian = ufl.derivative(residual, temperature, direction)
    return problems.nonlinear_first_order_transient_run(
        residual=residual,
        jacobian=jacobian,
        current=temperature,
        previous=previous,
        dt=dt,
        steps=steps,
        study=model.study,
        constraints=model.constraints if constraints is None else constraints,
        solver_options=solver_options,
        update_load=model._time_update_callback(update_load),
        save_every=save_every,
        print_every=print_every,
        progress=progress,
        status_file=status_file,
        checkpoint_policy=checkpoint,
        history_monitor=StateDependentThermalBalanceMonitor(
            content_form=content_form,
            source=monitor_source,
            dt=float(dt),
            outward_forms=tuple(outward_forms),
        ),
        name=name,
    )


def _thermal_boundary_terms(model, target):
    stiffness = []
    source = []
    unsupported = []
    for item in model.boundary_models:
        if hasattr(item, "operator") and hasattr(item, "source"):
            stiffness.append(item.operator(target))
            source.append(item.source(target))
        else:
            unsupported.append(getattr(item, "name", type(item).__name__))
    if unsupported and getattr(model.study, "is_heat_transfer", False):
        raise ValueError(
            f"Heat-transfer steps cannot consume these boundary models: {unsupported}."
        )
    return tuple(stiffness), tuple(source)


def _thermal_expansion_is_zero(material) -> bool:
    selected = getattr(material, "thermal_expansion", 0.0)
    if hasattr(selected, "values"):
        return bool(np.all(np.asarray(selected.values, dtype=float) == 0.0))
    return float(selected or 0.0) == 0.0


def _describe(item):
    if item is None:
        return None
    if hasattr(item, "summary"):
        return item.summary()
    if hasattr(item, "as_dict"):
        return item.as_dict()
    return getattr(item, "name", repr(item))


__all__ = ()
