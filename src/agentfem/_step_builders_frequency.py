# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Frequency-domain Step builders."""

from __future__ import annotations

import numpy as np

from . import loads as load_api


def direct_harmonic(
    model,
    *,
    target,
    K,
    F,
    M=None,
    C=None,
    K_loss=None,
    frequency: float | None = None,
    angular_frequency: float | None = None,
    frequencies=None,
    angular_frequencies=None,
    responses=(),
    execution_order: str = "forward",
    constraints=None,
    load_phase: float = 0.0,
    solver_options=None,
    status_file=None,
    name: str = "direct_harmonic",
):
    """Build and register a direct harmonic Step from explicit operators."""

    from . import mechanics, operators

    model.study.require(analysis="frequency_domain", physics="solid_mechanics")
    system = operators.direct_harmonic_system(
        K,
        F,
        M=M,
        C=C,
        K_loss=K_loss,
        name=f"{name}_system",
    )
    selected = tuple(
        value
        for value in (
            frequency,
            angular_frequency,
            frequencies,
            angular_frequencies,
        )
        if value is not None
    )
    if len(selected) != 1:
        raise ValueError(
            "Specify exactly one of frequency, angular_frequency, frequencies, "
            "or angular_frequencies."
        )
    sweep_frequencies = None
    point_frequency = frequency
    point_angular_frequency = angular_frequency
    if frequencies is not None:
        sweep_frequencies = tuple(float(value) for value in frequencies)
        if not sweep_frequencies:
            raise ValueError("frequencies must contain at least one point.")
        point_frequency = sweep_frequencies[0]
    elif angular_frequencies is not None:
        angular_axis = tuple(float(value) for value in angular_frequencies)
        if not angular_axis:
            raise ValueError("angular_frequencies must contain at least one point.")
        sweep_frequencies = tuple(value / (2.0 * np.pi) for value in angular_axis)
        point_angular_frequency = angular_axis[0]

    selected_constraints = model.constraints if constraints is None else constraints
    point_step = mechanics.direct_harmonic_step(
        displacement=target,
        system=system,
        frequency=point_frequency,
        angular_frequency=point_angular_frequency,
        constraints=selected_constraints,
        load_phase=load_phase,
        study=model.study,
        solver_options=solver_options,
        name=name,
    )
    if sweep_frequencies is None:
        return model.add_step(point_step)
    sweep = mechanics.harmonic_frequency_sweep_step(
        point_step,
        frequencies=sweep_frequencies,
        responses=responses,
        execution_order=execution_order,
        scientific_assets={
            "study": model.study,
            "materials": tuple(model.materials),
            "loads": tuple(model.loads),
            "constraints": tuple(selected_constraints),
            "boundary_models": tuple(model.boundary_models),
        },
        status_file=status_file,
        name=name,
    )
    return model.add_step(sweep)


def harmonic_viscoelastic(
    model,
    *,
    target,
    frequency: float | None = None,
    angular_frequency: float | None = None,
    material=None,
    constraints=None,
    density: float | None = None,
    load_phase: float = 0.0,
    temperature: float | None = None,
    solver_options=None,
    name: str = "harmonic_viscoelastic",
):
    """Build and register one direct generalized-Maxwell harmonic Step."""

    from . import mechanics
    from .constitutive.viscoelasticity import IsotropicGeneralizedMaxwell

    model.check(
        target=target,
        step_options={
            "material": material,
            "frequency": frequency,
            "angular_frequency": angular_frequency,
        },
    )
    model.study.require(analysis="frequency_domain", physics="solid_mechanics")
    properties = (
        model._material_record(material).item
        if material is not None
        else _single_material(model, "model.step harmonic response").item
    )
    if not isinstance(properties, IsotropicGeneralizedMaxwell):
        raise TypeError(
            "A harmonic generalized-Maxwell Step requires one "
            "IsotropicGeneralizedMaxwell material."
        )
    selected_loads = load_api.load_assets(model.loads)
    if any(isinstance(item, load_api.AmplitudeLoad) for item in selected_loads):
        raise ValueError(
            "Frequency-domain loads use load_phase on the Step; time-domain "
            "AmplitudeLoad assets are not valid in a harmonic Study."
        )
    step = mechanics.harmonic_viscoelastic_step(
        displacement=target,
        material=properties,
        frequency=frequency,
        angular_frequency=angular_frequency,
        external_force=(
            model.external_force(target, loads=selected_loads)
            if selected_loads
            else None
        ),
        constraints=model.constraints if constraints is None else constraints,
        density=density,
        load_phase=load_phase,
        temperature=temperature,
        study=model.study,
        solver_options=solver_options,
        name=name,
    )
    return model.add_step(step)


def _single_material(model, caller: str):
    if len(model.materials) != 1:
        raise ValueError(f"{caller} requires material=... or exactly one material.")
    return model.materials[0]


__all__ = ()
