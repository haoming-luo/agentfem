# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Built-in Step provider predicates, lowerers, and declarations.

The public provider protocol and deterministic registry live in
`step_providers`; this catalog owns only AgentFEM's built-in physics routes.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ._step_provider_support import (
    COMMON_STEP_OPTIONS as _COMMON_STEP_OPTIONS,
    all_materials_support as _all_materials_support,
    has_complete_linear_system as _has_complete_linear_system,
    is_scalar_target as _is_scalar_target,
    is_vector_target as _is_vector_target,
    normalize as _normalize,
    procedure_method as _procedure_method,
    registered_materials as _registered_materials,
    selected_material as _selected_material,
    supports_axisymmetric_elasticity as _supports_axisymmetric_elasticity,
    supports_conduction as _supports_conduction,
    supports_dynamics as _supports_dynamics,
    supports_elasticity as _supports_elasticity,
    supports_heat_capacity as _supports_heat_capacity,
    target_shape as _target_shape,
)
from .step_providers import (
    StepOptionContract,
    StepProvider,
    StepRequest,
    register_step_provider,
)


def _accept_elastic_interface(model, request: StepRequest) -> bool:
    from ._nonmatching_force import NonmatchingCohesiveForce
    return (getattr(getattr(model, "study", None), "physics", None) == "solid_mechanics"
            and isinstance(request.option("cohesive_force"), NonmatchingCohesiveForce)
            and _is_vector_target(request.target)
            and _all_materials_support(model, request, _supports_elasticity))


def _accept_uniform_hex(model, request):
    from .elements import UniformStrainHex8

    method = _procedure_method(model, request)
    return (isinstance(request.option("element_policy"), UniformStrainHex8)
            and request.option("element_policy").kinematics == "small_strain"
            and _is_vector_target(request.target)
            and (method is None or _normalize(method) in {"central_difference", "explicit_dynamics"}))


def _lower_uniform_hex(model, request):
    from ._step_uniform_hex import lower

    return lower(model, request)


def _accept_finite_uniform_hex(model, request):
    from .elements import UniformStrainHex8
    policy = request.option("element_policy")
    method = _procedure_method(model, request)
    return (isinstance(policy, UniformStrainHex8) and policy.kinematics == "finite_strain"
            and _is_vector_target(request.target)
            and (method is None or _normalize(method) in {"central_difference", "explicit_dynamics"}))


def _lower_finite_uniform_hex(model, request):
    from ._step_finite_uniform_hex import lower
    return lower(model, request)


def _accept_response_uniform_hex(model, request):
    from .constitutive import ExplicitUserMaterial
    return (_accept_finite_uniform_hex(model, request)
            and any(isinstance(record.item, ExplicitUserMaterial)
                    for record in model.materials))


def _lower_response_uniform_hex(model, request):
    from ._step_response_hex import lower
    return lower(model, request)


def _lower_elastic_interface(model, request):
    from ._step_nonmatching import lower
    return lower(model, request)


def _accept_linear_static(model, request: StepRequest) -> bool:
    study = getattr(model, "study", None)
    if request.target is None:
        return False
    physics = getattr(study, "physics", None)
    if physics == "solid_mechanics":
        material_predicate = (
            _supports_axisymmetric_elasticity
            if getattr(study, "assumption", None) == "axisymmetric"
            else _supports_elasticity
        )
        return _is_vector_target(request.target) and (
            _has_complete_linear_system(request)
            or _all_materials_support(model, request, material_predicate)
        )
    if physics == "heat_transfer":
        return _is_scalar_target(request.target) and (
            _has_complete_linear_system(request)
            or _all_materials_support(model, request, _supports_conduction)
        )
    return False


def _accept_rigid_obstacle_contact(model, request: StepRequest) -> bool:
    from .boundary_models import RigidObstaclePenaltyContact
    from .constitutive.elasticity import ElasticIsotropicProperties

    study = getattr(model, "study", None)
    contacts = tuple(
        item
        for item in getattr(model, "boundary_models", ())
        if isinstance(item, RigidObstaclePenaltyContact)
    )
    return (
        getattr(study, "analysis", None) == "nonlinear_static"
        and getattr(study, "physics", None) == "solid_mechanics"
        and _is_vector_target(request.target)
        and len(contacts) == 1
        and len(tuple(getattr(model, "boundary_models", ()))) == 1
        # This Step assembles a *linear small-strain* bulk form. A generic
        # young/poisson duck type also matches finite-strain J2 and must not
        # steal its stateful provider merely because contact was declared.
        and _all_materials_support(
            model,
            request,
            lambda item: type(item) is ElasticIsotropicProperties,
        )
    )


def _lower_rigid_obstacle_contact(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    for key in ("K", "F", "material", "method", "output", "history"):
        options.pop(key, None)
    name = options.pop("name", None) or "rigid_obstacle_contact"
    return _step_builders.rigid_obstacle_contact(
        model,
        target=request.target,
        name=name,
        **options,
    )


def _lower_linear_static(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    options.pop("material", None)
    # Completion belongs to solve_result(), not the numerical constructor.
    # The registry binds the original request in StepExecutionContext.
    options.pop("output", None)
    name = options.pop("name", None) or "linear_static"
    return _step_builders.linear_static(
        model,
        target=request.target,
        name=name,
        **options,
    )


def _accept_coupled_thermoelastic(model, request: StepRequest) -> bool:
    study = getattr(model, "study", None)
    return (
        getattr(study, "physics", None) == "multiphysics"
        and getattr(study, "assumption", None) == "small_strain_thermoelasticity"
        and getattr(study, "dimension", None) == 3
        and _is_vector_target(request.target)
        and _procedure_method(model, request) == "staggered_implicit_euler"
    )


def _lower_coupled_thermoelastic(model, request):
    from ._step_builders_coupled import thermoelastic
    return thermoelastic(model, request)


def _accept_transient_heat(model, request: StepRequest) -> bool:
    return (
        request.target is not None
        and _is_scalar_target(request.target)
        and getattr(getattr(model, "study", None), "physics", None) == "heat_transfer"
        and _all_materials_support(model, request, _supports_conduction)
        and _all_materials_support(model, request, _supports_heat_capacity)
    )


def _lower_transient_heat(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    options.pop("K", None)
    options.pop("F", None)
    material = options.pop("material", None)
    options.pop("output", None)
    options.pop("history", None)
    name = options.pop("name", None) or "transient_heat"
    return _step_builders.heat_transfer(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _accept_neo_hookean(model, request: StepRequest) -> bool:
    from .constitutive import hyperelasticity

    study = getattr(model, "study", None)
    material = _selected_material(model, request)
    supported_kinematics = hyperelasticity.supports_hyperelastic_study(
        material,
        dimension=getattr(study, "dimension", 0),
        assumption=getattr(study, "assumption", None),
    )
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and _is_vector_target(request.target)
        and supported_kinematics
        and hyperelasticity.is_finite_strain_hyperelastic(material)
    )


def _accept_geometric_elasticity(model, request: StepRequest) -> bool:
    from .constitutive.geometric_elasticity import SaintVenantKirchhoffProperties

    return isinstance(
        _selected_material(model, request), SaintVenantKirchhoffProperties
    ) and _accept_neo_hookean(model, request)


def _accept_fabric_membrane(model, request: StepRequest) -> bool:
    from .constitutive.fabric import DecoupledFabricSurface, FabricStack

    study = getattr(model, "study", None)
    return (
        getattr(study, "analysis", None) == "nonlinear_static"
        and getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "dimension", None) == 2
        and getattr(study, "assumption", None) == "membrane"
        and _target_shape(request.target) == (2,)
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, (DecoupledFabricSurface, FabricStack)),
        )
    )


def _accept_mixed_neo_hookean(model, request: StepRequest) -> bool:
    from .constitutive.hyperelasticity import MixedNeoHookeanProperties

    study = getattr(model, "study", None)
    supported_kinematics = getattr(study, "dimension", None) == 3 or (
        getattr(study, "dimension", None) == 2
        and getattr(study, "assumption", None) == "plane_strain"
    )
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(request.target, "kind", None) == "displacement_pressure"
        and supported_kinematics
        and isinstance(
            _selected_material(model, request),
            MixedNeoHookeanProperties,
        )
    )


def _accept_j2(model, request: StepRequest) -> bool:
    from .constitutive.plasticity import (
        ChabocheCombinedHardening,
        J2LinearIsotropicHardening,
    )

    study = getattr(model, "study", None)
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and _is_vector_target(request.target)
        and (
            getattr(study, "dimension", None) == 3
            or (
                getattr(study, "dimension", None) == 2
                and getattr(study, "assumption", None) == "axisymmetric"
            )
        )
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(
                item,
                (J2LinearIsotropicHardening, ChabocheCombinedHardening),
            ),
        )
    )


def _accept_small_strain_material(model, request: StepRequest) -> bool:
    from .constitutive.small_strain_user_material import SmallStrainUserMaterial

    study = getattr(model, "study", None)
    material = _selected_material(model, request)
    return (
        getattr(study, "analysis", None) == "nonlinear_static"
        and getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "dimension", None) == 3
        and _is_vector_target(request.target)
        and isinstance(material, SmallStrainUserMaterial)
    )


def _accept_finite_strain_j2_affine(model, request: StepRequest) -> bool:
    from . import loads as load_api
    from .boundary_models.implicit_admission import affine_rigid_contact_supported
    from .constitutive import FiniteStrainJ2Logarithmic
    from .constraints import AbaqusPeriodicConstraint, constraint_assets

    study = getattr(model, "study", None)
    selected_constraints = request.option("constraints")
    if selected_constraints is None:
        selected_constraints = tuple(getattr(model, "constraints", ()))
    selected_constraints = constraint_assets(selected_constraints)
    physical_loads = load_api.load_assets(
        getattr(model, "loads", ()),
        unwrap_amplitudes=True,
    )
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "dimension", None) == 3
        and getattr(request.target, "kind", None) != "displacement_pressure"
        and _is_vector_target(request.target)
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, FiniteStrainJ2Logarithmic),
        )
        and len(selected_constraints) == 1
        and isinstance(selected_constraints[0], AbaqusPeriodicConstraint)
        and not physical_loads
        and affine_rigid_contact_supported(model)
        and _finite_j2_bulk_quadrature_rejection_reason(request) is None
        and _contact_integration_rejection_reason(model, request) is None
        and _initial_contact_bound_rejection_reason(model, request) is None
    )


def _accept_finite_strain_j2_mixed_affine(
    model,
    request: StepRequest,
) -> bool:
    from . import loads as load_api
    from .constitutive import FiniteStrainJ2Logarithmic
    from .constraints import AbaqusPeriodicConstraint, constraint_assets

    study = getattr(model, "study", None)
    selected_constraints = request.option("constraints")
    if selected_constraints is None:
        selected_constraints = tuple(getattr(model, "constraints", ()))
    selected_constraints = constraint_assets(selected_constraints)
    physical_loads = load_api.load_assets(
        getattr(model, "loads", ()),
        unwrap_amplitudes=True,
    )
    dimension = getattr(study, "dimension", None)
    pressure_family = str(getattr(request.target, "pressure_family", "DG")).upper()
    domain = getattr(getattr(request.target, "space", None), "mesh", None)
    cell_name = None if domain is None else str(domain.topology.cell_name())
    interpolation_supported = (
        dimension == 3
        and cell_name == "tetrahedron"
        and int(getattr(request.target, "displacement_degree", -1)) == 2
        and pressure_family == "DG"
        and int(getattr(request.target, "pressure_degree", -1)) == 0
    ) or (
        dimension == 2
        and cell_name == "quadrilateral"
        and getattr(study, "assumption", None) == "plane_strain"
        and int(getattr(request.target, "displacement_degree", -1)) == 2
        and pressure_family == "DPC"
        and int(getattr(request.target, "pressure_degree", -1)) == 1
    )
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and interpolation_supported
        and domain is not None
        and getattr(request.target, "kind", None) == "displacement_pressure"
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, FiniteStrainJ2Logarithmic),
        )
        and len(selected_constraints) == 1
        and isinstance(selected_constraints[0], AbaqusPeriodicConstraint)
        and not physical_loads
        and not getattr(model, "boundary_models", ())
    )


def _finite_strain_j2_affine_boundary_rejection(model, request: StepRequest) -> str | None:
    """Reject any unreviewed affine/contact Operator composition."""

    from .boundary_models.implicit_admission import affine_rigid_contact_supported
    from .constitutive import FiniteStrainJ2Logarithmic
    from .constraints import AbaqusPeriodicConstraint, constraint_assets

    selected_constraints = request.option("constraints")
    if selected_constraints is None:
        selected_constraints = tuple(getattr(model, "constraints", ()))
    selected_constraints = constraint_assets(selected_constraints)
    if (
        len(selected_constraints) != 1
        or not isinstance(selected_constraints[0], AbaqusPeriodicConstraint)
        or not _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, FiniteStrainJ2Logarithmic),
        )
    ):
        return None
    quadrature_problem = _finite_j2_bulk_quadrature_rejection_reason(request)
    if quadrature_problem is not None:
        return quadrature_problem
    integration_problem = _contact_integration_rejection_reason(model, request)
    if integration_problem is not None:
        return integration_problem
    initial_bound_problem = _initial_contact_bound_rejection_reason(model, request)
    if initial_bound_problem is not None:
        return initial_bound_problem
    if not getattr(model, "boundary_models", ()) or affine_rigid_contact_supported(model):
        return None
    return (
        "AFM-J2-CONTACT-MPC: affine finite-strain J2 admits serial, "
        "named rigid pairs with reviewed geometry and fixed or "
        "normalized prescribed motion; mixed, distributed, and other weak "
        "Operators remain unsupported."
    )


def _finite_strain_j2_strong_preconditions(model, request: StepRequest) -> bool:
    from .constitutive import FiniteStrainJ2Logarithmic
    from .constraints import (
        DirichletConstraint,
        RemoteDisplacementConstraint,
        constraint_assets,
    )

    study = getattr(model, "study", None)
    selected_constraints = request.option("constraints")
    if selected_constraints is None:
        selected_constraints = tuple(getattr(model, "constraints", ()))
    selected_constraints = constraint_assets(selected_constraints)
    ordinary_strong = bool(selected_constraints) and all(
        isinstance(item, (DirichletConstraint, RemoteDisplacementConstraint))
        for item in selected_constraints
    )
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "dimension", None) == 3
        and getattr(request.target, "kind", None) != "displacement_pressure"
        and _is_vector_target(request.target)
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, FiniteStrainJ2Logarithmic),
        )
        and ordinary_strong
    )


def _implicit_planar_friction_geometry_supported(pair) -> bool:
    """Admit reviewed planar-branch geometry; never infer a smoothed normal."""

    from .boundary_models import (
        ClosedTriangulatedRigidSolid, ConvexPolyhedralRigidSurface, RigidPlaneSurface,
        TriangulatedRigidSurface,
    )

    if not isinstance(
        pair.rigid_body.surface, (
            RigidPlaneSurface, ConvexPolyhedralRigidSurface,
            TriangulatedRigidSurface, ClosedTriangulatedRigidSolid,
        ),
    ):
        return False
    return True


def _accept_finite_strain_j2_strong(model, request: StepRequest) -> bool:
    from .boundary_models import (
        ClosedTriangulatedRigidSolid,
        ConvexPolyhedralRigidSurface,
        RigidContactPair,
        RigidObstaclePenaltyContact,
        SmoothRigidSurface,
        TriangulatedRigidSurface,
    )

    if not _finite_strain_j2_strong_preconditions(model, request):
        return False
    if _finite_j2_bulk_quadrature_rejection_reason(request) is not None:
        return False
    if _contact_quadrature_rejection_reason(model, request) is not None:
        return False
    if _contact_integration_rejection_reason(model, request) is not None:
        return False
    if _initial_contact_bound_rejection_reason(model, request) is not None:
        return False
    boundary_models = tuple(getattr(model, "boundary_models", ()))
    if not boundary_models:
        return True
    target_mesh = getattr(getattr(request.target, "space", None), "mesh", None)
    if not boundary_models or getattr(getattr(target_mesh, "comm", None), "size", 1) != 1:
        return False
    if len(boundary_models) == 1 and isinstance(
        boundary_models[0], RigidObstaclePenaltyContact
    ):
        return True
    if len({item.name for item in boundary_models}) != len(boundary_models):
        return False
    return all(
        isinstance(item, RigidContactPair)
        and isinstance(item.rigid_body.surface, (
            TriangulatedRigidSurface, SmoothRigidSurface,
            ConvexPolyhedralRigidSurface,
            ClosedTriangulatedRigidSolid,
        ))
        and (
            item.friction is None
            or _implicit_planar_friction_geometry_supported(item)
        )
        and item.law.invalid_policy == "reject"
        and (
            item.rigid_body.motion_schedule is None
            or (
                item.rigid_body.motion_schedule.start_time == 0.0
                and item.rigid_body.motion_schedule.end_time == 1.0
                and item.rigid_body.motion_schedule.amplitude is None
            )
        )
        for item in boundary_models
    )


def _contact_quadrature_rejection_reason(model, request: StepRequest) -> str | None:
    """Keep a deliberately bounded contact integration option fail-closed."""

    selected = request.options.get("contact_quadrature_degree")
    if selected is None:
        return None
    from .boundary_models import RigidContactPair

    if (
        isinstance(selected, (bool, np.bool_))
        or not isinstance(selected, (int, np.integer))
        or int(selected) not in (5, 7, 9, 11)
    ):
        return "AFM-CONTACT-QUADRATURE: degree must be 5, 7, 9 or 11."
    pairs = tuple(getattr(model, "boundary_models", ()))
    if not pairs or not all(isinstance(item, RigidContactPair) for item in pairs):
        return (
            "AFM-CONTACT-QUADRATURE: explicit degree requires registered "
            "RigidContactPair assets, not a legacy or absent contact law."
        )
    target_value = getattr(request.target, "value", None)
    function_space = getattr(target_value, "function_space", None)
    domain = getattr(function_space, "mesh", None)
    if (
        domain is None
        or str(domain.topology.cell_name()) not in {"tetrahedron", "hexahedron"}
        or int(function_space.element.basix_element.degree) not in (1, 2)
        or int(domain.geometry.cmaps[0].degree) not in (1, 2)
    ):
        return (
            "AFM-CONTACT-QUADRATURE: explicit degree currently requires "
            "a CG1/CG2 tetrahedral or hexahedral displacement and "
            "first- or second-order geometry."
        )
    return None


def _initial_contact_bound_rejection_reason(model, request: StepRequest) -> str | None:
    """Keep a requested *certified* reference overlap bound out of sampled paths.

    The lower-level contact provider checks the actual nodal extrema. This
    inexpensive admission check mirrors its geometric proof prerequisites,
    without projecting or assembling during a capability probe.
    """

    from .boundary_models import RigidContactPair
    from .boundary_models.contact_integration_modes import (
        resolve_contact_integration_modes,
    )
    from .boundary_models.initial_gap_bound import initial_gap_bound_method

    bounded = tuple(
        pair for pair in getattr(model, "boundary_models", ())
        if isinstance(pair, RigidContactPair)
        and pair.initial_penetration_limit is not None
    )
    if not bounded:
        return None
    pairs = tuple(
        pair for pair in getattr(model, "boundary_models", ())
        if isinstance(pair, RigidContactPair)
    )
    try:
        modes = dict(zip(
            (pair.name for pair in pairs),
            resolve_contact_integration_modes(
                pairs, request.options.get("contact_integration", "gauss"),
            ),
            strict=True,
        ))
    except ValueError as exc:
        return str(exc)
    value = getattr(request.target, "value", None)
    space = getattr(value, "function_space", None)
    domain = getattr(space, "mesh", None)
    degree = getattr(getattr(space, "element", None), "basix_element", None)
    cell_name = None if domain is None else str(domain.topology.cell_name())
    facet_topology = {
        "tetrahedron": "triangle",
        "hexahedron": "quadrilateral",
    }.get(cell_name, "unsupported")
    if (
        domain is None
        or degree is None
        or any(
            initial_gap_bound_method(
                pair.rigid_body.surface,
                interpolation_degree=int(degree.degree),
                geometry_degree=int(domain.geometry.cmaps[0].degree),
                facet_topology=facet_topology,
                contact_integration=modes[pair.name],
            ) is None
            for pair in bounded
        )
    ):
        return (
            "AFM-CONTACT-INITIAL-BOUND: initial_penetration_limit requires "
            "first-order slave interpolation and geometry with an analytical "
            "plane, an analytical radial partition, reviewed closed-solid "
            "triangular signed distance, or an open sheet admitted by "
            "reviewed_planar_branch_partition; "
            "other sampled gaps are not a certified face-wise bound."
        )
    return None


def _contact_integration_rejection_reason(model, request: StepRequest) -> str | None:
    """Admit the changing trial partition only where its affine proof applies."""

    selected = request.options.get("contact_integration", "gauss")
    from .boundary_models.contact_integration_modes import (
        CONTACT_INTEGRATION_MODES, resolve_contact_integration_modes,
    )

    if isinstance(selected, str) and selected == "gauss":
        return None
    if isinstance(selected, str) and selected not in CONTACT_INTEGRATION_MODES:
        return "AFM-CONTACT-PARTITION: unknown contact_integration option."
    from .boundary_models import (
        ClosedTriangulatedRigidSolid, ConvexPolyhedralRigidSurface,
        RigidContactPair, RigidCylinderSurface, RigidPlaneSurface,
        RigidSphereSurface, TriangulatedRigidSurface,
    )

    pairs = tuple(getattr(model, "boundary_models", ()))
    if not pairs or not all(isinstance(item, RigidContactPair) for item in pairs):
        return (
            "AFM-CONTACT-PARTITION: named rigid contact pairs are required "
            "for trial active integration."
        )
    try:
        modes = resolve_contact_integration_modes(pairs, selected)
    except ValueError as exc:
        return str(exc)
    required_surface = {
        "planar_active_partition": RigidPlaneSurface,
        "convex_feature_partition": ConvexPolyhedralRigidSurface,
        "closed_planar_branch_partition": ClosedTriangulatedRigidSolid,
        "reviewed_planar_branch_partition": (
            ClosedTriangulatedRigidSolid, TriangulatedRigidSurface,
        ),
        "radial_active_partition": (RigidSphereSurface, RigidCylinderSurface),
    }

    if (any(
        mode != "gauss" and (
            not isinstance(item.rigid_body.surface, required_surface[mode])
            or (mode == "radial_active_partition"
                and item.rigid_body.surface.dimension != 3)
            or item.friction is not None
            or np.asarray(item.law.penalty).ndim != 0
        ) for item, mode in zip(pairs, modes, strict=True)
    ) or (any(mode != "gauss" for mode in modes)
          and request.options.get("contact_quadrature_degree") is not None)):
        return (
            "AFM-CONTACT-PARTITION: trial active partition requires "
            "frictionless analytical-plane, analytical radial, certified "
            "convex, or reviewed per-face planar-branch rigid pairs "
            "(matching the selected mode), scalar penalty and the "
            "default contact quadrature rule."
        )
    if all(mode == "gauss" for mode in modes):
        return None
    value = getattr(request.target, "value", None)
    space = getattr(value, "function_space", None)
    domain = getattr(space, "mesh", None)
    if (domain is None or str(domain.topology.cell_name()) != "tetrahedron"
            or int(space.element.basix_element.degree) != 1
            or int(domain.geometry.cmaps[0].degree) != 1):
        return (
            "AFM-CONTACT-PARTITION: only CG1, first-order tetrahedral "
            "slave triangles have the bounded active partition."
        )
    return None


def _finite_j2_bulk_quadrature_rejection_reason(request: StepRequest) -> str | None:
    """Reject a known underintegrated Q2-hex J2 request before Step creation."""

    selected = request.options.get("quadrature_degree")
    if selected is None:
        return None  # The builder selects the safe ordinary Q2-hex default.
    target_value = getattr(request.target, "value", None)
    function_space = getattr(target_value, "function_space", None)
    domain = getattr(function_space, "mesh", None)
    element = getattr(function_space, "element", None)
    basix_element = getattr(element, "basix_element", None)
    if (
        domain is not None
        and str(domain.topology.cell_name()) == "hexahedron"
        and basix_element is not None
        and int(basix_element.degree) == 2
        and isinstance(selected, (int, np.integer))
        and not isinstance(selected, (bool, np.bool_))
        and int(selected) < 4
    ):
        return (
            "AFM-J2-QUADRATURE: CG2 hexahedral finite-strain J2 requires "
            "bulk quadrature degree >= 4; lower-order integration can leave "
            "uncontrolled modes."
        )
    return None


def _finite_strain_j2_strong_rejection_reason(model, request: StepRequest) -> str | None:
    """Explain the bounded contact/MPI near-match without claiming support."""

    from .boundary_models import (
        RigidContactPair,
        RigidObstaclePenaltyContact,
    )

    if not _finite_strain_j2_strong_preconditions(model, request):
        return None
    bulk_quadrature_problem = _finite_j2_bulk_quadrature_rejection_reason(request)
    if bulk_quadrature_problem is not None:
        return bulk_quadrature_problem
    quadrature_problem = _contact_quadrature_rejection_reason(model, request)
    if quadrature_problem is not None:
        return quadrature_problem
    integration_problem = _contact_integration_rejection_reason(model, request)
    if integration_problem is not None:
        return integration_problem
    initial_bound_problem = _initial_contact_bound_rejection_reason(model, request)
    if initial_bound_problem is not None:
        return initial_bound_problem
    boundary_models = tuple(getattr(model, "boundary_models", ()))
    if (
        len(boundary_models) > 1
        and all(isinstance(item, RigidContactPair) for item in boundary_models)
    ):
        if len({item.name for item in boundary_models}) != len(boundary_models):
            return (
                "AFM-IMPLICIT-CONTACT-MULTI: rigid contact pair names must be "
                "unique for independent force, work and checkpoint identity."
            )
        if any(
            item.friction is not None
            and not _implicit_planar_friction_geometry_supported(item)
            for item in boundary_models
        ):
            return (
                "AFM-IMPLICIT-CONTACT-FRICTION: each frictional rigid pair "
                "must have a reviewed locally planar tangent; curved tools "
                "remain unsupported. Active noncoplanar facet edges and "
                "vertices are rejected during the Newton trial."
            )
        if any(
            item.law.invalid_policy != "reject" for item in boundary_models
        ):
            return (
                "AFM-IMPLICIT-CONTACT-MULTI: every rigid pair must reject "
                "invalid or ambiguous current-geometry projections."
            )
        if any(
            item.rigid_body.motion_schedule is not None
            and (
                item.rigid_body.motion_schedule.start_time != 0.0
                or item.rigid_body.motion_schedule.end_time != 1.0
                or item.rigid_body.motion_schedule.amplitude is not None
            )
            for item in boundary_models
        ):
            return (
                "AFM-IMPLICIT-CONTACT-MOTION: each rigid tool must use the "
                "Step-owned normalized 0-to-1 motion coordinate."
            )
    if len(boundary_models) != 1 or not isinstance(
        boundary_models[0], (RigidObstaclePenaltyContact, RigidContactPair)
    ):
        return None
    target_mesh = getattr(getattr(request.target, "space", None), "mesh", None)
    if getattr(getattr(target_mesh, "comm", None), "size", 1) > 1:
        return (
            "AFM-J2-CONTACT-MPI: finite-strain J2 plus rigid contact "
            "is currently serial; standalone MPI J2/contact evidence does not "
            "certify their composition."
        )
    selected = boundary_models[0]
    if isinstance(selected, RigidContactPair):
        if selected.friction is not None and (
            not _implicit_planar_friction_geometry_supported(selected)
        ):
            return (
                "AFM-IMPLICIT-CONTACT-FRICTION: implicit Coulomb contact is "
                "limited to reviewed locally planar branches in a serial strong-BC "
                "finite-strain J2 Step; curved tools remain unsupported and "
                "active noncoplanar facet edges or vertices are rejected."
            )
        schedule = selected.rigid_body.motion_schedule
        if schedule is not None and (
            schedule.start_time != 0.0
            or schedule.end_time != 1.0
            or schedule.amplitude is not None
        ):
            return (
                "AFM-IMPLICIT-CONTACT-MOTION: static J2 contact requires a "
                "normalized 0-to-1 rigid-motion schedule without a separate "
                "physical-time amplitude; the Step owns the load amplitude."
            )
    return None


def _lower_finite_strain_j2(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = _selected_material(model, request)
    options.pop("material", None)
    for legacy_name in ("K", "F"):
        legacy_value = options.pop(legacy_name, None)
        if legacy_value is not None:
            raise ValueError(
                f"Finite-strain J2 does not consume legacy {legacy_name}=; "
                "declare material, constraints and solver options explicitly."
            )
    name = options.pop("name", None) or "finite_strain_j2"
    return _step_builders.finite_strain_j2(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _lower_j2(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = request.material
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    options.pop("output", None)
    name = options.pop("name", None) or "j2_plasticity"
    return _step_builders.j2_plasticity(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _lower_small_strain_material(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = _selected_material(model, request)
    options.pop("material", None)
    for legacy_name in ("K", "F", "output"):
        options.pop(legacy_name, None)
    name = options.pop("name", None) or "small_strain_material"
    return _step_builders.small_strain_material(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _accept_implicit_creep(model, request: StepRequest) -> bool:
    from .constitutive.creep import IsotropicPowerLawCreepMaterial

    study = getattr(model, "study", None)
    method = _procedure_method(model, request)
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "analysis", None) == "nonlinear_transient"
        and (
            getattr(study, "dimension", None) == 3
            or (
                getattr(study, "dimension", None) == 2
                and getattr(study, "assumption", None) == "axisymmetric"
            )
        )
        and _is_vector_target(request.target)
        and _normalize(method or "implicit_creep")
        in {"implicit_creep", "backward_euler", "backward_euler_newton"}
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, IsotropicPowerLawCreepMaterial),
        )
    )


def _lower_implicit_creep(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = request.material
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    options.pop("method", None)
    options.pop("output", None)
    name = options.pop("name", None) or "implicit_creep"
    return _step_builders.creep(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _accept_quasistatic_viscoelasticity(model, request: StepRequest) -> bool:
    from .constitutive.viscoelasticity import IsotropicGeneralizedMaxwell

    study = getattr(model, "study", None)
    method = _procedure_method(model, request)
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "analysis", None) == "first_order_transient"
        and getattr(study, "dimension", None) == 3
        and _is_vector_target(request.target)
        and _normalize(method or "exact_generalized_maxwell_equilibrium")
        in {
            "quasistatic_viscoelasticity",
            "generalized_maxwell",
            "exact_generalized_maxwell_equilibrium",
        }
        and _all_materials_support(
            model,
            request,
            lambda item: isinstance(item, IsotropicGeneralizedMaxwell),
        )
    )


def _accept_harmonic_viscoelasticity(model, request: StepRequest) -> bool:
    from .constitutive.viscoelasticity import IsotropicGeneralizedMaxwell

    study = getattr(model, "study", None)
    method = _procedure_method(model, request)
    materials = _registered_materials(model, request)
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "analysis", None) == "frequency_domain"
        and getattr(study, "dimension", None) == 3
        and _is_vector_target(request.target)
        and _normalize(method or "direct_harmonic")
        in {"direct_harmonic", "harmonic", "real_block_complex_harmonic"}
        # The current harmonic builder assembles one homogeneous constitutive
        # operator.  Do not advertise a regional multi-material model unless
        # the request explicitly selects the one material to lower.
        and len(materials) == 1
        and isinstance(materials[0], IsotropicGeneralizedMaxwell)
    )


def _accept_direct_harmonic_system(model, request: StepRequest) -> bool:
    study = getattr(model, "study", None)
    method = _procedure_method(model, request)
    return (
        getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "analysis", None) == "frequency_domain"
        and _is_vector_target(request.target)
        and request.option("K") is not None
        and request.option("F") is not None
        and _normalize(method or "direct_harmonic")
        in {
            "direct_harmonic",
            "harmonic",
            "real_block_complex_harmonic",
            "direct_harmonic_sweep",
            "harmonic_sweep",
            "real_block_complex_harmonic_sweep",
        }
    )


def _lower_direct_harmonic_system(model, request: StepRequest):
    from . import _step_builders

    _validate_direct_harmonic_axis(request)
    options = request.lowering_options(
        "method",
        "output",
        "history",
        "progress",
        "checkpoint",
        "material",
    )
    name = options.pop("name", None) or "direct_harmonic"
    return _step_builders.direct_harmonic(
        model,
        target=request.target,
        name=name,
        **options,
    )


def _validate_direct_harmonic_axis(request: StepRequest) -> None:
    """Keep the selected Procedure consistent with scalar or sweep input."""

    plural = any(
        request.option(name) is not None
        for name in ("frequencies", "angular_frequencies")
    )
    method = _normalize(_procedure_method(None, request) or "direct_harmonic")
    sweep = method in {
        "direct_harmonic_sweep",
        "harmonic_sweep",
        "real_block_complex_harmonic_sweep",
    }
    if plural != sweep:
        expected = "a sweep procedure" if plural else "a single-frequency procedure"
        raise ValueError(
            "Direct harmonic frequency coordinates and procedure disagree: "
            f"the request supplies {'a frequency axis' if plural else 'one frequency'} "
            f"and therefore requires {expected}."
        )
    if not plural and request.option("checkpoint") is not None:
        raise ValueError(
            "Direct harmonic checkpointing is defined for a frequency sweep; "
            "one independent frequency has no partial lifecycle to resume."
        )
    if not plural and request.option("status_file") is not None:
        raise ValueError(
            "Direct harmonic status_file is defined for a frequency sweep."
        )


def _lower_harmonic_viscoelasticity(model, request: StepRequest):
    from . import _step_builders

    options = request.lowering_options(
        "K",
        "F",
        "method",
        "output",
        "history",
        "progress",
        "checkpoint",
    )
    name = options.pop("name", None) or "harmonic_viscoelastic"
    return _step_builders.harmonic_viscoelastic(
        model,
        target=request.target,
        name=name,
        **options,
    )


def _lower_quasistatic_viscoelasticity(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = request.material
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    options.pop("method", None)
    options.pop("output", None)
    name = options.pop("name", None) or "viscoelastic"
    return _step_builders.viscoelastic(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _lower_neo_hookean(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = _selected_material(model, request)
    options.pop("material", None)
    name = options.pop("name", None) or "finite_strain_static"
    options.pop("K", None)
    options.pop("F", None)
    return _step_builders.hyperelastic(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _lower_fabric_membrane(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = _selected_material(model, request)
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    name = options.pop("name", None) or "fabric_membrane"
    return _step_builders.fabric_membrane(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _lower_mixed_neo_hookean(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = _selected_material(model, request)
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    name = options.pop("name", None) or "mixed_finite_strain_static"
    return _step_builders.mixed_hyperelastic(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _accept_explicit_dynamics(model, request: StepRequest) -> bool:
    from .constitutive import hyperelasticity

    method = _procedure_method(model, request)
    materials = _registered_materials(model, request)
    explicit_residual = request.options.get("residual") is not None
    return (
        request.target is not None
        and _is_vector_target(request.target)
        and getattr(getattr(model, "study", None), "physics", None) == "solid_mechanics"
        and (
            (
                request.options.get("mass") is not None
                and request.options.get("residual") is not None
            )
            or _all_materials_support(model, request, _supports_dynamics)
        )
        and (
            explicit_residual
            or not any(
                hyperelasticity.is_finite_strain_hyperelastic(item)
                for item in materials
            )
        )
        and (
            method is None
            or _normalize(method) in {"explicit_dynamics", "central_difference"}
        )
    )


def _accept_finite_strain_explicit_dynamics(model, request: StepRequest) -> bool:
    from .constitutive import hyperelasticity

    study = getattr(model, "study", None)
    method = _procedure_method(model, request)
    materials = _registered_materials(model, request)
    return (
        request.target is not None
        and _is_vector_target(request.target)
        and getattr(study, "physics", None) == "solid_mechanics"
        and getattr(study, "analysis", None) == "second_order_dynamics"
        and bool(materials)
        and all(
            hyperelasticity.is_finite_strain_hyperelastic(material)
            and hyperelasticity.supports_hyperelastic_study(
                material,
                dimension=getattr(study, "dimension", 0),
                assumption=getattr(study, "assumption", None),
            )
            and material.density is not None
            for material in materials
        )
        and _normalize(method or "central_difference")
        in {"explicit_dynamics", "central_difference"}
        and request.options.get("residual") is None
    )


def _lower_finite_strain_explicit_dynamics(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    material = _selected_material(model, request)
    for key in ("K", "F", "solver_options", "method", "residual"):
        options.pop(key, None)
    options.pop("material", None)
    options.pop("output", None)
    options.pop("history", None)
    name = options.pop("name", None) or "finite_strain_explicit_dynamics"
    return _step_builders.finite_strain_explicit_dynamics(
        model,
        target=request.target,
        material=material,
        name=name,
        **options,
    )


def _lower_explicit_dynamics(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    options.pop("material", None)
    options.pop("K", None)
    options.pop("F", None)
    options.pop("solver_options", None)
    options.pop("method", None)
    options.pop("output", None)
    options.pop("history", None)
    name = options.pop("name", None) or "explicit_dynamics"
    return _step_builders.explicit_dynamics(
        model,
        target=request.target,
        name=name,
        **options,
    )


def _accept_implicit_dynamics(model, request: StepRequest) -> bool:
    from .constitutive import hyperelasticity

    method = _procedure_method(model, request)
    complete_system = all(
        request.options.get(item) is not None for item in ("M", "K", "F")
    )
    selected_material = _selected_material(model, request)
    return (
        request.target is not None
        and _is_vector_target(request.target)
        and getattr(getattr(model, "study", None), "physics", None) == "solid_mechanics"
        and (
            complete_system
            or _all_materials_support(model, request, _supports_dynamics)
        )
        and (
            complete_system
            or not hyperelasticity.is_finite_strain_hyperelastic(selected_material)
        )
        and _normalize(method or "") in {"newmark", "generalized_alpha"}
    )


def _lower_implicit_dynamics(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    options.pop("material", None)
    options.pop("method", None)
    options.pop("output", None)
    options.pop("history", None)
    method = _procedure_method(model, request)
    name = options.pop("name", None) or f"{method}_dynamics"
    return _step_builders.implicit_dynamics(
        model,
        target=request.target,
        method=method,
        name=name,
        **options,
    )


def _accept_modal(model, request: StepRequest) -> bool:
    complete_system = all(request.options.get(item) is not None for item in ("M", "K"))
    return (
        request.target is not None
        and _is_vector_target(request.target)
        and getattr(getattr(model, "study", None), "physics", None) == "solid_mechanics"
        and getattr(getattr(model, "study", None), "analysis", None) == "modal"
        and (
            complete_system
            or _all_materials_support(model, request, _supports_dynamics)
        )
    )


def _lower_modal(model, request: StepRequest):
    from . import _step_builders

    options = dict(request.options)
    for key in (
        "material",
        "F",
        "solver_options",
        "output",
        "history",
        "progress",
        "checkpoint",
    ):
        options.pop(key, None)
    name = options.pop("name", None) or "modal_analysis"
    return _step_builders.modal(model, target=request.target, name=name, **options)


def _option_contract(
    *extra: str,
    required=(),
    exactly_one_of=(),
) -> StepOptionContract:
    """Build a deterministic built-in contract without repeated core names."""

    return StepOptionContract(
        accepted=tuple(dict.fromkeys((*_COMMON_STEP_OPTIONS, *extra))),
        required=tuple(required),
        exactly_one_of=tuple(tuple(group) for group in exactly_one_of),
    )


register_step_provider(StepProvider(
    name="elastic_nonmatching_interface",
    analyses=("linear_static",),
    accepts=_accept_elastic_interface,
    lower=_lower_elastic_interface,
    priority=150,
    description="Experimental serial P1/Q1 elastic solid with a fixed nonmatching elastic interface.",
    procedure="standard/linear_static",
    option_contract=_option_contract("cohesive_force", required=("cohesive_force",)),
))

register_step_provider(StepProvider(
    name="uniform_strain_hex8_explicit",
    analyses=("explicit_dynamics", "second_order_dynamics"),
    accepts=_accept_uniform_hex,
    lower=_lower_uniform_hex,
    priority=180,
    description="Experimental small-strain elastic Hex8 with owned-cell MPI assembly; nonmatching interfaces remain serial.",
    procedure="explicit/central_difference",
    option_contract=_option_contract("element_policy", "cohesive_force", "dt", "steps", "save_every", "print_every",
        "history_every", "progress", "status_file", "checkpoint", required=("element_policy", "dt", "steps")),
))

register_step_provider(StepProvider(
    name="response_material_hex8_explicit",
    analyses=("explicit_dynamics", "second_order_dynamics"),
    accepts=_accept_response_uniform_hex,
    lower=_lower_response_uniform_hex,
    priority=182,
    description="Experimental serial response-only Hex8 with frictionless translating rigid tools; no implicit tangent or automatic stability claim.",
    procedure="explicit/central_difference",
    option_contract=_option_contract("element_policy", "dt", "steps", "time_increment_ceiling",
        "stress_free_initial_state", "bulk_viscosity", "mass_scaling_factor", "mass_scaling", "execution_kernel",
        "cohesive_force", "bond_contact_policy", "contact_trace_policy", "substep_policy", "material_storage", "contact_execution", "operator_increment_ceilings", "save_every", "print_every", "history_every", "progress",
        "status_file", "checkpoint", "history", required=("element_policy", "dt", "steps",
        "time_increment_ceiling", "stress_free_initial_state")),
))

register_step_provider(StepProvider(
    name="finite_uniform_strain_hex8_explicit",
    analyses=("explicit_dynamics", "second_order_dynamics"),
    accepts=_accept_finite_uniform_hex,
    lower=_lower_finite_uniform_hex,
    priority=181,
    description="Experimental finite Hex8 with one history material, declared spectral ceiling and same-partition restart.",
    procedure="explicit/central_difference",
    option_contract=_option_contract("element_policy", "cohesive_force", "dt", "steps", "save_every", "print_every",
        "history_every", "progress", "status_file", "checkpoint", "omega_squared_bound", "maximum_negative_growth_per_increment",
        required=("element_policy", "dt", "steps")),
))


def _accept_callable_neural_field(_model, request) -> bool:
    from .learning import NeuralFieldSpec

    executor = request.option("executor")
    return isinstance(request.target, NeuralFieldSpec) and (
        callable(executor) or callable(getattr(executor, "solve", None))
    )


def _lower_callable_neural_field(model, request):
    from .learning.execution import (
        CallableNeuralFieldStep,
        NeuralFieldExecutionRequest,
        executor_identity,
    )

    unsupported = {
        name: request.option(name)
        for name in ("K", "F", "constraints", "solver_options")
        if request.option(name) is not None
    }
    if unsupported:
        raise TypeError(
            "A neural-field executor consumes NeuralFieldSpec objectives and "
            f"conditions, not assembled FEM options: {tuple(unsupported)!r}."
        )
    executor = request.option("executor")
    executor_name, executor_version = executor_identity(
        executor,
        name=request.option("executor_name"),
        version=request.option("executor_version"),
    )
    executor_options = request.option("executor_options") or {}
    if not isinstance(executor_options, Mapping):
        raise TypeError("executor_options must be a mapping.")
    execution_request = NeuralFieldExecutionRequest(
        specification=request.target,
        model=model,
        analysis=request.analysis,
        name=request.option("name") or "neural_field",
        output_directory=request.option("output"),
        options=executor_options,
    )
    step = CallableNeuralFieldStep(
        execution_request,
        executor,
        executor_name=executor_name,
        executor_version=executor_version,
    )
    return model.add_step(step)


register_step_provider(
    StepProvider(
        name="rigid_obstacle_penalty_contact_static",
        analyses=("nonlinear_static",),
        accepts=_accept_rigid_obstacle_contact,
        lower=_lower_rigid_obstacle_contact,
        priority=140,
        description=(
            "Lower a small-strain elastic solid with one frictionless rigid-"
            "plane penalty contact boundary to incremental Newton equilibrium."
        ),
        procedure="standard/newton/rigid_obstacle_penalty_contact",
        option_contract=_option_contract(
            "incrementation",
            "checkpoint",
            "progress",
            "status_file",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="small_strain_user_material_static",
        analyses=("nonlinear_static",),
        accepts=_accept_small_strain_material,
        lower=_lower_small_strain_material,
        priority=115,
        description=(
            "Lower a provider-neutral small-strain material to batched "
            "quadrature state and implicit Newton equilibrium."
        ),
        procedure="standard/newton/stateful",
        option_contract=_option_contract(
            "incrementation",
            "quadrature_degree",
            "progress",
            "status_file",
            "amplitude",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="callable_neural_field",
        analyses=(
            "linear_static",
            "nonlinear_static",
            "first_order_transient",
            "nonlinear_transient",
            "second_order_dynamics",
            "explicit_dynamics",
            "modal",
        ),
        accepts=_accept_callable_neural_field,
        lower=_lower_callable_neural_field,
        priority=450,
        description=(
            "Execute a NeuralFieldSpec through a user-owned callable while "
            "retaining the common Step and SimulationResult lifecycle."
        ),
        procedure="learning/neural_field/user_executor",
        option_contract=_option_contract(
            "executor",
            "executor_name",
            "executor_version",
            "executor_options",
            required=("executor",),
        ),
    )
)


register_step_provider(
    StepProvider(
        name="neo_hookean_finite_strain_explicit_dynamics",
        analyses=("explicit_dynamics", "second_order_dynamics"),
        accepts=_accept_finite_strain_explicit_dynamics,
        lower=_lower_finite_strain_explicit_dynamics,
        priority=120,
        description=(
            "Lower a complete finite-strain hyperelastic material partition to a "
            "current-state Total-Lagrangian residual and explicit central "
            "difference."
        ),
        procedure="explicit/central_difference/total_lagrangian",
        option_contract=_option_contract(
            "dt",
            "steps",
            "method",
            "residual",
            "state",
            "mass",
            "cohesive_force",
            "contact_pairs",
            "contact_projection_options",
            "update_load",
            "save_every",
            "print_every",
            "history_every",
            "progress",
            "status_file",
            "checkpoint",
            "history",
            "stability_safety",
            "mass_damping",
            required=("steps",),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="implicit_power_law_creep",
        analyses=("nonlinear_transient",),
        accepts=_accept_implicit_creep,
        lower=_lower_implicit_creep,
        priority=120,
        description=(
            "Lower isotropic power-law creep to backward-Euler quadrature "
            "state, consistent-tangent Newton equilibrium, and cutback."
        ),
        procedure="standard/backward_euler/stateful",
        option_contract=_option_contract(
            "duration",
            "method",
            "incrementation",
            "quadrature_degree",
            "creep_strain_error_tolerance",
            "progress",
            "status_file",
            "amplitude",
            "temperature",
            required=("duration",),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="direct_harmonic_system",
        analyses=("frequency_domain",),
        accepts=_accept_direct_harmonic_system,
        lower=_lower_direct_harmonic_system,
        priority=140,
        description=(
            "Lower explicit storage, loss, mass, viscous damping and force "
            "operators to a real-block direct harmonic finite-element system."
        ),
        procedure="standard/direct_harmonic/real_block_complex",
        option_contract=_option_contract(
            "M",
            "C",
            "K_loss",
            "frequency",
            "angular_frequency",
            "frequencies",
            "angular_frequencies",
            "method",
            "load_phase",
            "responses",
            "execution_order",
            "solver_options",
            "output",
            "progress",
            "checkpoint",
            "status_file",
            required=("K", "F"),
            exactly_one_of=(
                (
                    "frequency",
                    "angular_frequency",
                    "frequencies",
                    "angular_frequencies",
                ),
            ),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="harmonic_generalized_maxwell",
        analyses=("frequency_domain",),
        accepts=_accept_harmonic_viscoelasticity,
        lower=_lower_harmonic_viscoelasticity,
        priority=120,
        description=(
            "Lower isotropic generalized-Maxwell storage/loss moduli to a "
            "real-block direct harmonic finite-element system."
        ),
        procedure="standard/direct_harmonic/real_block_complex",
        option_contract=_option_contract(
            "frequency",
            "angular_frequency",
            "method",
            "density",
            "load_phase",
            "temperature",
            "solver_options",
            "output",
            exactly_one_of=(("frequency", "angular_frequency"),),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="quasistatic_generalized_maxwell",
        analyses=("first_order_transient",),
        accepts=_accept_quasistatic_viscoelasticity,
        lower=_lower_quasistatic_viscoelasticity,
        priority=120,
        description=(
            "Lower an isotropic generalized-Maxwell solid to exact quadrature "
            "history and incremental global equilibrium."
        ),
        procedure="standard/exact_generalized_maxwell/stateful",
        option_contract=_option_contract(
            "duration",
            "steps",
            "time_points",
            "incrementation",
            "time_error_tolerance",
            "method",
            "quadrature_degree",
            "progress",
            "status_file",
            "checkpoint",
            "amplitude",
            "temperature",
            required=("duration",),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="staggered_thermoelastic",
        analyses=("first_order_transient",),
        accepts=_accept_coupled_thermoelastic,
        lower=_lower_coupled_thermoelastic,
        priority=100,
        description="Experimental shared-mesh 3D linear thermoelasticity; SI, strong boundaries, final snapshot.",
        procedure="standard/staggered_implicit_euler",
        option_contract=_option_contract(
            "progress", "print_every", "checkpoint",
            "dt", "steps", "temperature_departure", "heat_loads", "mechanical_loads",
            "relaxation", "rtol", "max_iterations", "displacement_atol", "temperature_atol",
            required=("dt", "steps", "temperature_departure"),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="implicit_euler_heat_transfer",
        analyses=("first_order_transient",),
        accepts=_accept_transient_heat,
        lower=_lower_transient_heat,
        priority=100,
        description="Lower heat capacity/conduction/source to implicit Euler.",
        procedure="standard/implicit_euler",
        option_contract=_option_contract(
            "dt",
            "steps",
            "C",
            "Q",
            "update_load",
            "save_every",
            "print_every",
            "progress",
            "status_file",
            "checkpoint",
            "history",
            "operator_policy",
            required=("dt", "steps"),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="linear_static_operators",
        analyses=("linear_static",),
        accepts=_accept_linear_static,
        lower=_lower_linear_static,
        priority=100,
        description="Lower K/F engineering operators to a linear static solve.",
        procedure="standard/linear",
        option_contract=_option_contract(),
    )
)
register_step_provider(
    StepProvider(
        name="finite_strain_j2_mixed_affine_static",
        analyses=("nonlinear_static",),
        accepts=_accept_finite_strain_j2_mixed_affine,
        rejection_reason=_finite_strain_j2_affine_boundary_rejection,
        lower=_lower_finite_strain_j2,
        priority=135,
        description=(
            "Lower logarithmic finite-strain J2 to P2/DG0 3D or Q2/DPC1 "
            "plane-strain mixed equilibrium under exact affine kinematics."
        ),
        procedure=("standard/newton/stateful/mixed_mean_kirchhoff_stress/affine_mpc"),
        option_contract=_option_contract(
            "incrementation",
            "quadrature_degree",
            "output_every",
            "progress",
            "status_file",
            "checkpoint",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="finite_strain_j2_affine_static",
        analyses=("nonlinear_static",),
        accepts=_accept_finite_strain_j2_affine,
        rejection_reason=_finite_strain_j2_affine_boundary_rejection,
        lower=_lower_finite_strain_j2,
        priority=130,
        description=(
            "Lower logarithmic finite-strain J2 to provider-owned quadrature "
            "state and exact affine/MPC Newton equilibrium."
        ),
        procedure="standard/newton/stateful/affine_mpc",
        option_contract=_option_contract(
            "incrementation",
            "quadrature_degree",
            "contact_integration",
            "output_every",
            "progress",
            "status_file",
            "checkpoint",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="finite_strain_j2_strong_static",
        analyses=("nonlinear_static",),
        accepts=_accept_finite_strain_j2_strong,
        rejection_reason=_finite_strain_j2_strong_rejection_reason,
        lower=_lower_finite_strain_j2,
        priority=125,
        description=(
            "Lower logarithmic finite-strain J2 to provider-owned quadrature "
            "state and strong-boundary Newton equilibrium, optionally with "
            "serial rigid-tool penalty contact: multiple frictionless pairs "
            "and prescribed-moving analytical-plane Coulomb pairs within "
            "their reviewed bounds."
        ),
        procedure="standard/newton/stateful",
        option_contract=_option_contract(
            "incrementation",
            "quadrature_degree",
            "contact_quadrature_degree",
            "contact_integration",
            "amplitude",
            "output_every",
            "progress",
            "status_file",
            "checkpoint",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="j2_small_strain_static",
        analyses=("nonlinear_static",),
        accepts=_accept_j2,
        lower=_lower_j2,
        priority=110,
        description=(
            "Lower J2 plasticity to quadrature-state Newton equilibrium "
            "with algorithmic tangent and cutback."
        ),
        procedure="standard/newton/stateful",
        option_contract=_option_contract(
            "incrementation",
            "quadrature_degree",
            "progress",
            "status_file",
            "amplitude",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="mixed_neo_hookean_constant_pressure",
        analyses=("nonlinear_static",),
        accepts=_accept_mixed_neo_hookean,
        lower=_lower_mixed_neo_hookean,
        priority=115,
        description=(
            "Lower P2 displacement and DG0 pressure to mixed finite-strain "
            "hyperelastic equilibrium."
        ),
        procedure="standard/newton/mixed_constant_pressure",
        option_contract=_option_contract(
            "measure",
            "petsc_options_prefix",
            "incrementation",
            "increments",
            "load_factors",
            "output_every",
            "progress",
            "status_file",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="decoupled_fabric_membrane_static",
        analyses=("nonlinear_static",),
        accepts=_accept_fabric_membrane,
        lower=_lower_fabric_membrane,
        priority=130,
        description=(
            "Lower one or more named woven layers with independent yarn-tension "
            "and trellising-shear channels to finite-kinematics in-plane "
            "membrane equilibrium."
        ),
        procedure="standard/newton/fabric_membrane",
        option_contract=_option_contract(
            "measure",
            "petsc_options_prefix",
            "incrementation",
            "increments",
            "load_factors",
            "output_every",
            "progress",
            "status_file",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="saint_venant_kirchhoff_finite_strain_static",
        analyses=("nonlinear_static",),
        accepts=_accept_geometric_elasticity,
        lower=_lower_neo_hookean,
        priority=110,
        description="Total-Lagrangian elastic solids with exact plane-stress condensation, finite rotations and consistent Newton tangent.",
        procedure="standard/newton",
        option_contract=_option_contract(
            "measure",
            "petsc_options_prefix",
            "incrementation",
            "increments",
            "load_factors",
            "output_every",
            "progress",
            "status_file",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="neo_hookean_finite_strain_static",
        analyses=("nonlinear_static",),
        accepts=_accept_neo_hookean,
        lower=_lower_neo_hookean,
        priority=100,
        description=(
            "Lower a supported displacement-based hyperelastic material to "
            "total-Lagrangian equilibrium."
        ),
        procedure="standard/newton",
        option_contract=_option_contract(
            "measure",
            "petsc_options_prefix",
            "incrementation",
            "increments",
            "load_factors",
            "output_every",
            "progress",
            "status_file",
        ),
    )
)
register_step_provider(
    StepProvider(
        name="linear_structural_modes",
        analyses=("modal",),
        accepts=_accept_modal,
        lower=_lower_modal,
        priority=110,
        description="Lower K/M operators to a constrained SLEPc modal solve.",
        procedure="standard/generalized_hermitian_eigenproblem",
        option_contract=_option_contract(
            "M",
            "modes",
            "target_frequency",
            "tolerance",
            "maximum_iterations",
            "rigid_mode_tolerance",
            required=("modes",),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="implicit_structural_dynamics",
        analyses=("second_order_dynamics",),
        accepts=_accept_implicit_dynamics,
        lower=_lower_implicit_dynamics,
        priority=110,
        description="Lower second-order operators to Newmark/generalized-alpha.",
        procedure="standard/newmark_or_generalized_alpha",
        option_contract=_option_contract(
            "dt",
            "steps",
            "method",
            "spectral_radius",
            "M",
            "C",
            "state",
            "update_load",
            "progress",
            "status_file",
            "checkpoint",
            "history",
            "save_every",
            "print_every",
            "operator_policy",
            required=("dt", "steps"),
        ),
    )
)
register_step_provider(
    StepProvider(
        name="central_difference_explicit_dynamics",
        analyses=("explicit_dynamics", "second_order_dynamics"),
        accepts=_accept_explicit_dynamics,
        lower=_lower_explicit_dynamics,
        priority=100,
        description="Lower a second-order study to explicit central difference.",
        procedure="explicit/central_difference",
        option_contract=_option_contract(
            "dt",
            "steps",
            "method",
            "residual",
            "state",
            "mass",
            "cohesive_force",
            "prescribed",
            "update_load",
            "save_every",
            "print_every",
            "progress",
            "status_file",
            "checkpoint",
            "history",
            required=("dt", "steps"),
        ),
    )
)


def _accept_buckling(model, request):
    return (_is_vector_target(request.target)
            and getattr(model.study, "physics", None) == "solid_mechanics"
            and _all_materials_support(model, request, _supports_elasticity))


def _lower_buckling(model, request):
    from .mechanics.buckling import lower_buckling
    return lower_buckling(model, request)


register_step_provider(StepProvider(
    name="linear_solid_buckling", analyses=("linear_buckling",),
    accepts=_accept_buckling, lower=_lower_buckling, priority=110,
    description="Experimental conservative initial-stress solid buckling with an indefinite geometric pencil.",
    procedure="standard/generalized_nonhermitian_eigenproblem",
    option_contract=StepOptionContract(
        accepted=("K", "F", "solver_options", "reference_displacement", "reference_name", "base_displacement", "base_name",
                  "constraints", "modes", "tolerance", "maximum_iterations", "name", "output"),
        required=("reference_displacement", "reference_name")),
))


def _accept_arc_length(model, request):
    return getattr(request.procedure, "algorithm", None) == "spherical_arc_length"


def _lower_arc_length(model, request):
    from .mechanics.continuation import lower_arc_length

    return lower_arc_length(model, request)


register_step_provider(
    StepProvider(
        name="hyperelastic_spherical_arc_length",
        analyses=("nonlinear_static",),
        accepts=_accept_arc_length,
        lower=_lower_arc_length,
        priority=1000,
        description="Serial sparse continuation adapter for conservative hyperelastic solids.",
        procedure="standard/spherical_arc_length",
        option_contract=StepOptionContract(
            accepted=(
                "K",
                "F",
                "solver_options",
                "material",
                "constraints",
                "arc_options",
                "increments",
                "name",
                "output",
            ),
            required=("increments",),
        ),
    )
)
