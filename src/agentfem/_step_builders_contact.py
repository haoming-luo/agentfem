# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded nonlinear contact Step builders.

The first route is intentionally narrow: small-strain linear-elastic bulk,
one fixed rigid plane, frictionless penalty contact, and ordinary strong
constraints.  It establishes residual/tangent/dual ownership before surface
search, friction, and deformable-to-deformable contact are introduced.
"""

from __future__ import annotations


def rigid_obstacle_contact(
    model,
    *,
    target,
    constraints=None,
    incrementation=None,
    solver_options=None,
    checkpoint=None,
    progress=True,
    status_file=None,
    name: str = "rigid_obstacle_contact",
):
    """Build small-strain equilibrium with one rigid-plane contact provider."""

    import ufl
    from dolfinx import fem
    from petsc4py import PETSc

    from . import constraints as constraint_api
    from . import problems
    from .boundary_models import RigidObstaclePenaltyContact
    from .checkpointing import _partition_neutral_identity

    model.check(target=target)
    if hasattr(model.study, "require"):
        model.study.require(analysis="nonlinear_static", physics="solid_mechanics")
    contacts = tuple(
        item
        for item in model.boundary_models
        if isinstance(item, RigidObstaclePenaltyContact)
    )
    unsupported_boundary_models = tuple(
        getattr(item, "name", type(item).__name__)
        for item in model.boundary_models
        if not isinstance(item, RigidObstaclePenaltyContact)
    )
    if len(contacts) != 1 or unsupported_boundary_models:
        raise ValueError(
            "The first rigid-obstacle contact Step requires exactly one "
            "RigidObstaclePenaltyContact and no other boundary models; "
            f"contacts={len(contacts)}, unsupported={unsupported_boundary_models}."
        )
    contact = contacts[0]
    selected_constraints = (
        tuple(model.constraints) if constraints is None else tuple(constraints)
    )
    concrete_constraints = constraint_api.constraint_assets(selected_constraints)
    unsupported_constraints = tuple(
        type(item).__name__
        for item in concrete_constraints
        if not isinstance(
            item,
            (
                constraint_api.DirichletConstraint,
                constraint_api.RemoteDisplacementConstraint,
            ),
        )
    )
    if unsupported_constraints:
        raise NotImplementedError(
            "Rigid-obstacle penalty contact currently accepts ordinary strong "
            f"displacement constraints; unsupported={unsupported_constraints}."
        )

    displacement = target.value
    test = target.test
    trial = target.trial
    internal = model.internal_force(displacement, test)
    contact_residual = contact.form(displacement, test)
    load_factor = fem.Constant(
        displacement.function_space.mesh,
        PETSc.ScalarType(0.0),
    )
    external = model.external_force(target) if model.loads else None
    residual = internal.expression + contact_residual
    if external is not None:
        residual -= load_factor * external.expression
    jacobian = ufl.derivative(residual, displacement, trial)
    model_summary = model.summary()
    checkpoint_identity = _partition_neutral_identity(
        {
            key: model_summary[key]
            for key in (
                "study",
                "unit_system",
                "mesh",
                "fields",
                "materials",
                "eigenstrains",
                "constraints",
                "loads",
                "boundary_models",
                "regions",
            )
        }
    )

    problem = problems.incremental_nonlinear(
        residual,
        displacement,
        factor=load_factor,
        value_path=constraint_api.prescribed_value_path(selected_constraints),
        update_load=model._time_update_callback(include_constraints=False),
        jacobian=jacobian,
        incrementation=incrementation,
        constraints=selected_constraints,
        constraint_assets=(*concrete_constraints, contact),
        solver_options=solver_options,
        checkpoint_policy=checkpoint,
        checkpoint_identity=checkpoint_identity,
        progress=progress,
        status_file=status_file,
        name=name,
        petsc_options_prefix="agentfem_rigid_contact_",
    )
    problem.external_force_operator = external
    problem.contact_provider = contact
    problem.primary_fields = {"U": target}
    problem.result_field_factory = lambda: (displacement,)
    return model.add_step(problem)


__all__ = ("rigid_obstacle_contact",)
