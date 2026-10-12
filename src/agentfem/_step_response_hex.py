# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded ordinary-Step admission for response-only user materials.

No automatic stability estimate or synthetic material tangent is introduced.
The initial promotion is serial, single material with frictionless rigid tools; unsupported
model assets are rejected before any numerical state is allocated.
"""
from dataclasses import fields as dataclass_fields
import math

from ._transient_problems import ExplicitDynamicsStep


class ResponseMaterialHexStep(ExplicitDynamicsStep):
    def run(self, *args, **kwargs):
        if self.contact_execution == 'overlap':
            with self.deformable_residual.overlap_execution():
                return super().run(*args, **kwargs)
        return super().run(*args, **kwargs)

    def accepted_material_fields(self):
        """Detached accepted cell records for export, never trial data.

        One quadrature point per cell; no smoothing, energy relabelling or
        source-law-specific variable names are introduced by this observer.
        """
        import numpy as np
        body = self.material_residual
        if body._trial is not None:
            raise RuntimeError("Material export requires an accepted station")
        response = body.internal.response
        if response._owned is not None and response._owned.trial is not None:
            raise RuntimeError("Material export requires an accepted station")
        response.publish_accepted()
        def frozen(value):
            array = np.ascontiguousarray(value)
            return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
        return {"time": body.accepted_time, "state_schema": body.material.state_schema.summary(),
                "cell_keys": frozen(response.state.reference_field.owned_cell_keys),
                "state": frozen(response.state.committed_state_vectors()),
                "cauchy_stress": frozen(response.response_transaction.committed['S'].values),
                "deformation_gradient": frozen(body.accepted_gradient),
                "location": "one_material_point_per_uniform_strain_hex8_cell",
                "stress_measure": "cauchy"}

    def _advance_one(self, t, *, increment_size=None):
        if self.substep_policy is None or increment_size is not None:
            return super()._advance_one(t, increment_size=increment_size)
        return self._substeps.advance(t)

    def summary(self):
        return {**super().summary(), "material": self.material_residual.material.summary(),
                "element_policy": self.element_policy.summary(),
                "execution_scope": "experimental_serial_response_only_hex8",
                "material_storage": self.material_storage,
                "contact_execution": self.contact_execution,
                "operator_increment_ceilings": dict(self.operator_increment_ceilings),
                "stability_scope": "caller_increment_ceiling_not_stability_proof",
                "numerical_inertia": None if self.numerical_inertia is None else self.numerical_inertia.summary(),
                "execution_kernel": None if self.execution_kernel is None else self.execution_kernel.summary(),
                "substeps": None if self.substep_policy is None else self._substeps.summary(),
                "energy_scope": "accepted_internal_work_not_stored_energy"}


def lower(model, request):
    import numpy as np
    from . import boundary_models, fracture, problems, state, time
    from .constitutive import ExplicitUserMaterial
    from .constitutive._explicit_material_driver import ExplicitQuadratureResponse
    from .elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual
    from .mechanics._response_hex_explicit import ResponseHexExplicitResidual, ResponseHexWorkMonitor

    options = dict(request.options)
    policy = options.pop("element_policy")
    domain = request.target.value.function_space.mesh
    model.study.require(analysis="second_order_dynamics", physics="solid_mechanics")
    if domain.comm.size != 1 or model.study.dimension != 3:
        raise NotImplementedError("Response-only Hex8 currently requires serial 3D dynamics")
    if len(model.materials) != 1 or model.materials[0].region is not None:
        raise NotImplementedError("Response-only Hex8 requires one full-domain material")
    material = model.materials[0].item
    if not isinstance(material, ExplicitUserMaterial) or material.density is None:
        raise ValueError("Register an ExplicitUserMaterial with positive density")
    # Fail closed instead of silently dropping an asset not yet lowered here.
    if model.loads or model.constraints or model.eigenstrains:
        raise NotImplementedError("Response-only ordinary Step does not yet admit loads, constraints or eigenstrains")
    deformable = tuple(pair for pair in model.boundary_models
                       if isinstance(pair, boundary_models.DeformableContactPair))
    if len(deformable) > 1:
        raise NotImplementedError("Response-only Hex8 currently admits one deformable contact pair")
    contact_execution = options.pop('contact_execution', 'serial')
    if contact_execution not in ('serial', 'overlap'):
        raise ValueError('contact_execution must be serial or overlap')
    if contact_execution == 'overlap' and not deformable:
        raise ValueError('Contact overlap requires a deformable contact pair')
    pairs = tuple(pair for pair in model.boundary_models
                  if not isinstance(pair, boundary_models.DeformableContactPair))
    names = set()
    for pair in pairs:
        if (not isinstance(pair, boundary_models.RigidContactPair)
                or pair.friction is not None
                or not isinstance(pair.rigid_body.surface, (
                    boundary_models.RigidPlaneSurface, boundary_models.RigidCylinderSurface))):
            raise NotImplementedError("Response-only Hex8 admits frictionless planes and cylinders only")
        if pair.slave_boundary.domain is not domain:
            raise ValueError("Contact boundary must belong to the displacement mesh")
        if pair.name in names:
            raise ValueError("Contact pair names must be unique")
        names.add(pair.name)
        schedule = pair.rigid_body.motion_schedule
        if schedule is not None and np.any(np.asarray(schedule.motion.rotation) != 0):
            raise NotImplementedError("Response-only Hex8 rotating tools are not admitted")
    for name in ("K", "F", "material", "solver_options", "constraints"):
        if options.pop(name, None) is not None:
            raise ValueError(f"Response-only Hex8 does not accept {name}")
    for name in ("output", "history"):
        if options.pop(name, None) is not None:
            raise NotImplementedError(
                f"Response-only Hex8 does not yet lower declarative {name}; "
                "use explicit accepted-state export and the built-in energy ledger")
    dt = options.pop("dt")
    if isinstance(dt, (str, bool)):
        raise ValueError("Response-only Hex8 needs a numerical dt, never dt='auto'")
    dt = float(dt)
    ceiling = float(options.pop("time_increment_ceiling"))
    if not math.isfinite(dt) or not math.isfinite(ceiling) or not 0 < dt <= ceiling:
        raise ValueError("Require 0 < dt <= finite caller time_increment_ceiling")
    from collections.abc import Mapping
    overrides = options.pop('operator_increment_ceilings', {})
    roles = {'body': ceiling, 'rigid_contact': ceiling, 'deformable_contact': ceiling}
    if not isinstance(overrides, Mapping) or set(overrides)-roles.keys():
        raise ValueError('Operator increment ceilings accept body, rigid_contact and deformable_contact only')
    if ('rigid_contact' in overrides and not pairs
            or 'deformable_contact' in overrides and not deformable):
        raise ValueError('An operator increment ceiling requires the corresponding contact asset')
    for role, value in overrides.items():
        if (isinstance(value, (str, bool, np.bool_, complex)) or not np.isscalar(value)
                or not np.isfinite(value) or not dt <= value):
            raise ValueError('Operator increment ceilings must be finite scalars no smaller than dt')
        roles[role] = float(value)
    stress_free = options.pop("stress_free_initial_state")
    if stress_free is not True:
        raise ValueError("Explicit stress-free initial-state declaration is required")
    viscosity = options.pop("bulk_viscosity", None)
    inertia_factor = options.pop("mass_scaling_factor", None)
    execution_kernel = options.pop("execution_kernel", None)
    storage = options.pop("material_storage", "fields")
    if storage not in ("fields", "owned_arrays"):
        raise ValueError("material_storage must be fields or owned_arrays")
    substeps = options.pop("substep_policy", None)
    if substeps is not None and not isinstance(substeps, time.ExplicitSubstepPolicy):
        raise TypeError("substep_policy requires ExplicitSubstepPolicy")
    trace_policy = options.pop("contact_trace_policy", boundary_models.RigidContactTracePolicy())
    if not isinstance(trace_policy, boundary_models.RigidContactTracePolicy):
        raise TypeError("contact_trace_policy requires RigidContactTracePolicy")
    cohesive = options.pop("cohesive_force", None)
    overlap_policy = options.pop("bond_contact_policy", "disjoint")
    if overlap_policy not in ("disjoint", "independent"):
        raise ValueError("bond_contact_policy must be disjoint or independent")
    if cohesive is not None:
        from ._nonmatching_force import NonmatchingCohesiveForce
        if not isinstance(cohesive, NonmatchingCohesiveForce):
            raise TypeError("cohesive_force requires a nonmatching interface force")
        if overlap_policy == "disjoint":
            from dolfinx import fem
            bonded = np.union1d(cohesive.negative_dofs, cohesive.positive_dofs)
            for pair in deformable:
                nodes = np.union1d(pair.secondary_dofs[pair.candidate_nodes], pair.primary_dofs)
                if np.intersect1d(nodes, bonded).size:
                    raise ValueError("Bond/contact traces overlap; declare independent contributions explicitly")
            for pair in pairs:
                region = pair.slave_boundary
                nodes = fem.locate_dofs_topological(request.target.value.function_space,
                    domain.topology.dim - 1, region.facet_tags.find(region.tag))
                if np.intersect1d(nodes, bonded).size:
                    raise ValueError("Bond/contact traces overlap; declare independent contributions explicitly")
    if viscosity is not None:
        from .elements import BulkViscosityPolicy
        if not isinstance(viscosity, BulkViscosityPolicy):
            raise TypeError("bulk_viscosity requires BulkViscosityPolicy")
        if viscosity.reference_density != material.density:
            raise ValueError("Bulk viscosity must use the material's physical reference density")
    with state.field_transaction(displacement=request.target):
        history = state.second_order_state(request.target)
        response = ExplicitQuadratureResponse.create(domain, material.state_schema)
        internal = FiniteUniformHexResidual(history.u, response, density=material.density,
            hourglass_modulus=policy.hourglass_modulus, hourglass_scale=policy.hourglass_scale,
            chunk_size=policy.chunk_size, execution_kernel=execution_kernel)
        numerical_inertia = None
        if inertia_factor is not None:
            from .time._scaled_inertia import FixedScaledInertia
            numerical_inertia = FixedScaledInertia(internal.mass_diagonal, factor=inertia_factor)
            numerical_inertia.validate_state(history)
        inertia = internal if numerical_inertia is None else numerical_inertia
        residual = ResponseHexExplicitResidual(internal, material,
            time_increment_ceiling=roles['body'], material_identity={**material.summary(),
                "operator_increment_ceilings": roles,
                "execution_kernel": None if execution_kernel is None else execution_kernel.summary(),
                "material_storage": storage,
                "substep_policy": None if substeps is None else substeps.summary(),
                "contact_trace_policy": trace_policy.summary(), "bond_contact_policy": overlap_policy},
            stress_free_initial_state=True, bulk_viscosity=viscosity,
            numerical_inertia=numerical_inertia, cohesive=cohesive)
        material_residual = residual
        if storage == "owned_arrays":
            response.enable_owned_arrays()
        for pair in deformable:
            from ._deformable_q1_contact import DeformableQ1Penalty
            from .mechanics._deformable_contact_explicit import DeformableQ1ContactResidual
            if pair.name in names:
                raise ValueError("Contact pair names must be unique")
            operator = DeformableQ1Penalty(pair.secondary, pair.primary,
                candidate_nodes=pair.candidate_nodes, penalty=pair.penalty,
                search_radius=pair.search_radius, geometry_policy=pair.geometry_policy,
                execution=pair.execution)
            residual = DeformableQ1ContactResidual(residual, operator=operator,
                displacement=history.u.value, secondary_dofs=pair.secondary_dofs,
                primary_dofs=pair.primary_dofs, time_increment_ceiling=roles['deformable_contact'], name=pair.name)
        deformable_residual = residual if deformable else None
        # Share only reference geometry for the identical boundary object in
        # this lowering call. Tool-specific filters and physical histories stay
        # independent; no cache survives a mesh/Step lifecycle.
        reference_traces = {}
        for pair in pairs:
            key = id(pair.slave_boundary)
            if key not in reference_traces:
                reference_traces[key] = boundary_models.dolfinx_boundary_region_contact_trace(
                    pair.slave_boundary, history.u.value.function_space, integration=trace_policy.integration)
            adapter = reference_traces[key]
            adapter = trace_policy.apply(adapter, pair)
            residual = boundary_models.dolfinx_explicit_contact_residual(
                residual, adapter=adapter, displacement=history.u.value,
                contact_pair=pair, maximum_stable_time_increment=roles['rigid_contact'],
                name=pair.name)
        options["checkpoint_policy"] = options.pop("checkpoint", None)
        options["name"] = options.get("name") or "response_material_hex8_explicit"
        base = problems.explicit_dynamics(state=history,
            integrator=time.explicit.central_difference(state=history, mass=inertia),
            residual=residual, dt=dt, study=model.study,
            history_monitor=fracture.DynamicEnergyLedger(energy=ResponseHexWorkMonitor(
                material_residual, numerical_inertia=numerical_inertia),
                state=history, mass=inertia.mass_diagonal, residual=residual), **options)
        step = ResponseMaterialHexStep(**{f.name: getattr(base, f.name)
                                         for f in dataclass_fields(base) if f.init})
        step.element_policy = policy
        step.material_residual = material_residual
        step.numerical_inertia = numerical_inertia
        step.execution_kernel = execution_kernel
        step.material_storage = storage
        step.contact_execution = contact_execution
        step.deformable_residual = deformable_residual
        from types import MappingProxyType
        step.operator_increment_ceilings = MappingProxyType(roles)
        step.substep_policy = substeps
        if substeps is not None:
            from .time.substeps import _ExplicitSubsteps
            step._substeps = _ExplicitSubsteps(step, super(ResponseMaterialHexStep, step)._advance_one, substeps)
    return model.add_step(step)
