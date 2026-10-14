# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded ordinary-Step admission for response-only user materials.

No automatic stability estimate or synthetic material tangent is introduced.
The initial promotion is serial, single material with frictionless rigid tools; unsupported
model assets are rejected before any numerical state is allocated.
"""
from dataclasses import fields as dataclass_fields
import math
from os import PathLike

from ._transient_problems import ExplicitDynamicsStep


class ResponseMaterialHexStep(ExplicitDynamicsStep):
    def run(self, *args, **kwargs):
        if self.contact_execution == 'overlap':
            with self.deformable_residual.overlap_execution():
                return super().run(*args, **kwargs)
        return super().run(*args, **kwargs)

    def _require_accepted_observation(self):
        import numpy as np
        body = self.material_residual
        if (body._trial is not None or body.time != body.accepted_time
                or not np.array_equal(body.internal.displacement.x.array,
                                      body.accepted_displacement)):
            raise RuntimeError("Observation requires the accepted displacement and time")

    def accepted_contact_evidence(self, *, name=None):
        """Detached per-pair forces, motion and quality at an accepted station.

        An optional exact pair name avoids evaluating unrelated observations.
        Results are plain JSON-compatible records; observation never reassembles
        contact or advances a material. Force and moment conventions are supplied
        by the owning contact operator, not inferred from an energy value.
        """
        self._require_accepted_observation()
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("Contact observation name must be a nonempty string")
        records, visited = [], set()
        operator = self.residual
        while operator is not None:
            if id(operator) in visited:
                raise RuntimeError("Contact observation encountered a residual cycle")
            visited.add(id(operator))
            observe = getattr(operator, "accepted_contact_observation", None)
            if callable(observe) and (name is None or operator.name == name):
                records.append(observe())
            operator = getattr(operator, "base", None)
        if name is not None and not records:
            raise KeyError(f"No contact pair named {name!r}")
        return tuple(records)

    def accepted_material_fields(self):
        """Detached accepted cell records for export, never trial data.

        One quadrature point per cell; no smoothing, energy relabelling or
        source-law-specific variable names are introduced by this observer.
        """
        import numpy as np
        self._require_accepted_observation()
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
        schema = body.material.state_schema
        state_values = frozen(response.state.committed_state_vectors())
        variables, offset = {}, 0
        for variable in schema.variables:
            variables[variable.name] = state_values[:, offset:offset + variable.size].reshape(
                (len(state_values), *variable.shape))
            offset += variable.size
        return {"time": body.accepted_time, "state_schema": schema.summary(),
                "cell_keys": frozen(response.state.reference_field.owned_cell_keys),
                "state": state_values, "state_variables": variables,
                "cauchy_stress": frozen(response.response_transaction.committed['S'].values),
                "deformation_gradient": frozen(body.accepted_gradient),
                "location": "one_material_point_per_uniform_strain_hex8_cell",
                "stress_measure": "cauchy"}

    def accepted_interface_evidence(self):
        """Observe the bounded elastic bond at committed body configuration.

        Evaluates the elastic interface at fixed history, without beginning a
        trial or reintegrating the body material. Opening/slip use the declared
        reference normal; they are not finite-rotation fracture mode measures.
        Returns None when the model has no bonded interface.
        """
        import numpy as np
        self._require_accepted_observation()
        body = self.material_residual
        force = body.cohesive
        if force is None:
            return None
        if force.assembler._trial is not None:
            raise RuntimeError("Interface observation requires an accepted station")
        pair = force.assembler.pairing
        response = force.evaluate()
        un, up = force._values()
        physical = (-response.negative_residual, -response.positive_residual)
        resultants = [value.sum(axis=0) for value in physical]
        moments = [np.cross(x, f).sum(axis=0) for x, f in zip(
            (pair.negative.vertices+un, pair.positive.vertices+up), physical, strict=True)]
        normal = np.einsum('ij,ij->i', response.jump, pair.normals)
        slip = np.linalg.norm(response.jump-normal[:, None]*pair.normals, axis=1)
        return {"schema": "agentfem.accepted_interface_observation.v1",
                "time": body.accepted_time, "pairing": pair.summary(),
                "negative_force": resultants[0].tolist(), "positive_force": resultants[1].tolist(),
                "negative_moment": moments[0].tolist(), "positive_moment": moments[1].tolist(),
                "moment_reference_point": [0., 0., 0.],
                "force_balance_residual": (resultants[0]+resultants[1]).tolist(),
                "moment_balance_residual": (moments[0]+moments[1]).tolist(),
                "maximum_reference_normal_opening": float(normal.max(initial=0)),
                "maximum_reference_normal_compression": float((-normal).max(initial=0)),
                "maximum_reference_tangential_slip": float(slip.max(initial=0)),
                "stored_energy": response.stored_energy,
                "dissipated_energy": response.dissipated_energy,
                "force_convention": "physical_force_opposite_potential_gradient",
                "jump_convention": "positive_displacement_minus_negative_displacement",
                "scientifically_validated": False}

    def _advance_one(self, t, *, increment_size=None):
        if self.substep_policy is None or increment_size is not None:
            return super()._advance_one(t, increment_size=increment_size)
        return self._substeps.advance(t)

    def summary(self):
        from .elements._hex_validity import affine_screen_backend
        return {**super().summary(), "material": self.material_residual.material.summary(),
                "element_policy": self.element_policy.summary(),
                "execution_scope": "experimental_serial_response_only_hex8",
                "material_storage": self.material_storage,
                "contact_execution": self.contact_execution,
                "affine_geometry_screen": affine_screen_backend(),
                "operator_increment_ceilings": dict(self.operator_increment_ceilings),
                "stability_scope": "caller_increment_ceiling_not_stability_proof",
                "numerical_inertia": None if self.numerical_inertia is None else self.numerical_inertia.summary(),
                "mass_scaling_policy": None if self.mass_scaling_policy is None else self.mass_scaling_policy.summary(),
                "mass_scaling_preparation": self.mass_scaling_preparation,
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
    # Common completion owns output/history; never create a second writer here.
    output = options.pop("output", None)
    if output is not None and not isinstance(output, (str, PathLike)):
        raise TypeError("Transient output must be an XDMF path")
    options.pop("history", None)  # retained by the shared execution context
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
    scaling_policy = options.pop("mass_scaling", None)
    if scaling_policy is not None:
        if not isinstance(scaling_policy, (time.FixedMassScaling, time.InitialMassScaling,
                                           time.TargetIncrementMassScaling)):
            raise TypeError("mass_scaling requires a declared mass-scaling policy")
        if inertia_factor is not None:
            raise ValueError("Use mass_scaling or mass_scaling_factor, not both")
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
        # One reference trace per boundary and one filtered adapter per tool.
        # The same adapters supply initial mass preparation and actual forces.
        reference_traces, rigid_adapters = {}, []
        for pair in pairs:
            key = id(pair.slave_boundary)
            if key not in reference_traces:
                reference_traces[key] = boundary_models.dolfinx_boundary_region_contact_trace(
                    pair.slave_boundary, history.u.value.function_space, integration=trace_policy.integration)
            rigid_adapters.append((pair, trace_policy.apply(reference_traces[key], pair)))
        numerical_inertia = None
        mass_preparation = None
        if inertia_factor is not None:
            from .time._scaled_inertia import FixedScaledInertia
            numerical_inertia = FixedScaledInertia(internal.mass_diagonal, factor=inertia_factor)
            numerical_inertia.validate_state(history)
        if isinstance(scaling_policy, time.InitialMassScaling):
            from .time._initial_mass_scaling import lower_initial_mass_scaling
            numerical_inertia, mass_preparation = lower_initial_mass_scaling(
                scaling_policy, internal, history, dt=dt, cohesive=cohesive,
                rigid_adapters=rigid_adapters, deformable=deformable, viscosity=viscosity)
            numerical_inertia.validate_state(history)
        elif scaling_policy is not None:
            from .time.mass_scaling import _lower_mass_scaling
            numerical_inertia = _lower_mass_scaling(
                scaling_policy, internal, history, dt=dt,
                stiffness_names=("body", "hourglass")
                    + (("cohesive",) if cohesive is not None else ())
                    + tuple(f"rigid_contact:{pair.name}" for pair in pairs)
                    + tuple(f"deformable_contact:{pair.name}" for pair in deformable),
                damping_names=("bulk_viscosity",) if viscosity is not None else ())
            numerical_inertia.validate_state(history)
        inertia = internal if numerical_inertia is None else numerical_inertia
        residual = ResponseHexExplicitResidual(internal, material,
            time_increment_ceiling=roles['body'], material_identity={**material.summary(),
                "operator_increment_ceilings": roles,
                "execution_kernel": None if execution_kernel is None else execution_kernel.summary(),
                "material_storage": storage,
                **({"mass_scaling_policy": scaling_policy.summary()} if scaling_policy is not None else {}),
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
        for pair, adapter in rigid_adapters:
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
                state=history, mass=(inertia if hasattr(inertia, "prepare_increment")
                                     else inertia.mass_diagonal), residual=residual), **options)
        step = ResponseMaterialHexStep(**{f.name: getattr(base, f.name)
                                         for f in dataclass_fields(base) if f.init})
        step.element_policy = policy
        step.material_residual = material_residual
        step.numerical_inertia = numerical_inertia
        step.mass_scaling_policy = scaling_policy
        step.mass_scaling_preparation = mass_preparation
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
