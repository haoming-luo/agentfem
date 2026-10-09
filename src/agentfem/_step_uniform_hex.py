# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded Hex8 lowering to the existing explicit Procedure and result owner."""

from dataclasses import fields as dataclass_fields, replace
from hashlib import sha256

import numpy as np

from ._transient_problems import ExplicitDynamicsStep
from .constitutive.elasticity import _constant_stiffness_matrix_3d


def _material_partition(model, domain):
    count = domain.topology.index_map(3).size_local
    if count == 0:
        raise ValueError("Uniform Hex8 requires nonempty cells.")
    matrices = np.empty((count, 6, 6))
    density = np.empty(count)
    coverage = np.zeros(count, dtype=np.int32)
    for record in model.materials:
        region = record.region
        if region is None:
            indices = np.arange(count)
        else:
            if region.domain is not domain:
                raise ValueError("Material region belongs to another mesh.")
            indices = region.cell_tags.find(region.tag)
        if not len(indices):
            raise ValueError("Uniform Hex8 material region is empty.")
        matrices[indices] = _constant_stiffness_matrix_3d(record.item)
        rho = getattr(record.item, "density", None)
        if rho is None or not np.isfinite(rho) or rho <= 0:
            raise ValueError("Every Hex8 material needs positive finite density.")
        density[indices] = rho
        coverage[indices] += 1
    if np.any(coverage != 1):
        raise ValueError("Hex8 material partition must cover each cell exactly once.")
    if np.all(matrices == matrices[0]):
        matrices = matrices[0].copy()
    return matrices, density


class _Residual:
    def __init__(self, internal, external, cohesive=None):
        self.internal, self.external = internal, external
        self.cohesive = cohesive
        digest = sha256()
        for array in (
            internal.cell_nodes,
            internal.cells.stiffness,
            internal.cells.average_gradient,
            internal.cells.hourglass_modes,
            internal.cells.hourglass_coefficient,
            internal.mass_diagonal,
        ):
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        self.identity = digest.hexdigest()

    def bind_initial_inputs(self, bcs):
        from . import operators
        from dolfinx.fem import petsc
        from .time.explicit import _owned_dirichlet_dofs

        digest = sha256(self.identity.encode())
        if self.external is not None:
            vector = operators.assemble_vector(self.external)
            try:
                digest.update(vector.array.tobytes())
            finally:
                vector.destroy()
        vector = self.internal.displacement.x.petsc_vec.duplicate()
        try:
            vector.set(0)
            petsc.set_bc(vector, bcs)
            digest.update(_owned_dirichlet_dofs(bcs).tobytes())
            digest.update(vector.array.tobytes())
        finally:
            vector.destroy()
        self.identity = digest.hexdigest()

    def assemble_vector(self):
        from . import operators

        vector = self.internal.assemble_vector()
        try:
            if self.cohesive is not None:
                self.cohesive.add_to_vector(vector)
            if self.external is not None:
                force = operators.assemble_vector(self.external)
                try:
                    vector.axpy(-1, force)
                finally:
                    force.destroy()
            if not np.all(np.isfinite(vector.array)):
                raise ValueError("Non-finite Hex8 residual or external load.")
            return vector
        except Exception:
            vector.destroy()
            raise

    def snapshot(self):
        return {
            "schema": "agentfem.uniform-hex-elastic.v1",
            "operator_identity": self.identity,
            "cohesive": None if self.cohesive is None else self.cohesive.snapshot(),
        }

    def restore(self, record):
        if record != self.snapshot():
            raise ValueError("Uniform Hex8 checkpoint operator identity mismatch.")
        if self.cohesive is not None:
            self.cohesive.rollback()
            self.cohesive.restore(record["cohesive"])

    def commit(self):
        if self.cohesive is not None:
            self.cohesive.commit()

    def rollback(self):
        if self.cohesive is not None:
            self.cohesive.rollback()


class _Energy:
    def __init__(self, internal, cohesive=None):
        self.internal = internal
        self.cohesive = cohesive

    def evaluate(self, *, displacement, velocity):
        from .fields import unwrap

        values = self.internal.energies()
        kinetic = float(
            0.5 * np.sum(self.internal.mass_diagonal * unwrap(velocity).x.array ** 2)
        )
        physical = values["strain_energy"] + kinetic
        if self.cohesive is not None:
            interface = self.cohesive.evaluate().stored_energy
            values["cohesive_stored_energy"] = interface
            physical += interface
        return {
            **values,
            "kinetic_energy": kinetic,
            "total_mechanical_energy": physical,
            "total_discrete_energy": physical + values["hourglass_energy"],
            "accounted_internal_kinetic_energy": physical + values["hourglass_energy"],
        }


class UniformHexStep(ExplicitDynamicsStep):
    def checkpoint_capabilities(self):
        return replace(
            super().checkpoint_capabilities(),
            rank_count_portability="unsupported",
            evidence=("serial fixed-operator restart",),
            limitations=("serial only; no cross-partition restore",),
        )

    def save_checkpoint(self, path, *, portable=False):
        if portable:
            raise NotImplementedError("Uniform Hex8 portable restart is not verified.")
        return super().save_checkpoint(path, portable=False)

    def summary(self):
        return {
            **super().summary(),
            "element_policy": self.element_policy.summary(),
            "energy_balance_scope": "accepted_path_work_with_explicit_artificial_energy",
            "execution_scope": "serial_small_strain_elastic",
            "geometry_admission": "bounded_bernstein_with_floating_point_margin",
            "interface": None
            if self.residual.cohesive is None
            else self.residual.cohesive.summary(),
        }


def lower(model, request):
    from . import constraints, fracture, problems, state, time
    from .elements import UniformStrainHex8
    from .elements._uniform_hex_dolfinx import UniformHexResidual

    options = dict(request.options)
    policy = options.pop("element_policy")
    if not isinstance(policy, UniformStrainHex8):
        raise TypeError("Expected UniformStrainHex8 element policy.")
    if model.study.dimension != 3 or model.study.physics != "solid_mechanics":
        raise ValueError("Uniform Hex8 requires a 3D solid dynamics Study.")
    for key in ("K", "F", "material", "solver_options"):
        if options.pop(key, None) is not None:
            raise ValueError(
                f"Uniform Hex8 does not accept {key}; use registered model inputs."
            )
    options.pop("output", None)
    options.pop("history", None)
    steps = options.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("Uniform Hex8 steps must be a positive integer.")
    if model.boundary_models or model.eigenstrains:
        raise NotImplementedError(
            "Uniform Hex8 boundary models and eigenstrains are not yet supported."
        )
    selected_constraints = options.pop("constraints", None)
    assets = tuple(
        model.constraints if selected_constraints is None else selected_constraints
    )
    if any(
        not isinstance(
            item, (constraints.DirichletConstraint, constraints.TimeDependentDirichlet)
        )
        for item in constraints.constraint_assets(assets)
    ):
        raise NotImplementedError(
            "Uniform Hex8 currently accepts strong Dirichlet constraints only."
        )
    model.check(target=request.target)
    history = state.second_order_state(request.target)
    domain = history.u.value.function_space.mesh
    if domain.comm.size != 1:
        raise NotImplementedError("Uniform Hex8 MPI is not yet verified.")
    matrices, density = _material_partition(model, domain)
    internal = UniformHexResidual(
        history.u,
        matrices,
        density=density,
        hourglass_modulus=policy.hourglass_modulus,
        hourglass_scale=policy.hourglass_scale,
        chunk_size=policy.chunk_size,
    )
    external = model.external_force(request.target) if model.loads else None
    cohesive = options.pop("cohesive_force", None)
    if cohesive is not None:
        from ._nonmatching_force import NonmatchingCohesiveForce
        from ._elastic_cohesive import ElasticCohesiveLaw

        if not isinstance(cohesive, NonmatchingCohesiveForce) or not isinstance(
            cohesive.assembler.law, ElasticCohesiveLaw
        ):
            raise NotImplementedError(
                "Uniform Hex8 currently composes only fixed nonmatching elastic interfaces."
            )
        if cohesive.displacement is not history.u.value:
            raise ValueError("Interface force must use the same displacement field.")
        if cohesive.assembler.pairing.method != "coplanar-affine-q1-common-refinement":
            raise ValueError("Uniform Hex8 needs original Q1 common-refinement traces.")
    residual = _Residual(internal, external, cohesive)
    dt = options.pop("dt")
    interface_bound = (
        0
        if cohesive is None
        else cohesive.elastic_stability_bound(internal.mass_diagonal)
    )
    bulk_bound = internal.cells.stability_bound()
    stable = 0.8 * 2 / np.sqrt(bulk_bound + interface_bound)
    dt = stable if dt == "auto" else float(dt)
    if not np.isfinite(dt) or dt <= 0 or dt > stable:
        raise ValueError(
            f"Hex8 dt must be positive and at most the conservative bound {stable:g}."
        )
    update = model._time_update_callback()
    if time.input_summary(update)["changes_operator"]:
        raise NotImplementedError(
            "Uniform Hex8 requires fixed material/operator inputs."
        )
    if update is not None:
        update(0.0)
    prescribed = tuple(constraints.dirichlet_constraints(assets))
    bcs = [item.bc for item in prescribed]
    residual.bind_initial_inputs(bcs)
    constraints.apply_dirichlet_bcs(history.u, bcs)
    initial = residual.assemble_vector()
    try:
        history.a.value.x.array[:] = -initial.array * internal.inv_mass
        from .time.explicit import _assign_prescribed_component, _prescribed_kinematics

        kinematics = _prescribed_kinematics(prescribed, time=0.0, dt=dt)
        _assign_prescribed_component(history.a, kinematics, component=1)
        _assign_prescribed_component(history.v, kinematics, component=2)
        residual.commit()
    finally:
        initial.destroy()
    options["checkpoint_policy"] = options.pop("checkpoint", None)
    options["name"] = options.get("name") or "uniform_strain_hex8_explicit"
    base = problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=internal),
        residual=residual,
        study=model.study,
        dt=dt,
        prescribed=prescribed,
        constraints=assets,
        update_load=update,
        history_monitor=fracture.DynamicEnergyLedger(
            energy=_Energy(internal, cohesive),
            state=history,
            mass=internal.mass_diagonal,
            residual=residual,
            natural_force=external,
            prescribed=prescribed,
        ),
        stability={
            "dt_limit": stable,
            "method": "positive_cell_mass_rayleigh_bound",
            "scope": "fixed_elastic_bulk_hourglass_and_optional_interface",
            "bulk_omega_squared_bound": bulk_bound,
            "interface_omega_squared_bound": interface_bound,
        },
        **options,
    )
    step = UniformHexStep(
        **{
            item.name: getattr(base, item.name)
            for item in dataclass_fields(base)
            if item.init
        }
    )
    step.element_policy = policy
    return model.add_step(step)
