# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded Hex8 lowering to the existing explicit Procedure and result owner."""

from dataclasses import fields as dataclass_fields, replace
from hashlib import sha256

import numpy as np

from ._transient_problems import ExplicitDynamicsStep


def _material_matrix(properties):
    from .materials.properties import (
        ElasticIsotropicProperties,
        ElasticAnisotropic3DProperties,
    )

    frame = getattr(properties, "orientation", None)
    if frame is not None and frame.evolution != "fixed":
        raise NotImplementedError("Uniform Hex8 requires fixed material orientation.")
    raw = getattr(properties, "material", properties)
    if type(raw) is ElasticIsotropicProperties:
        e, nu = float(raw.young), float(raw.poisson)
        mu = e / (2 * (1 + nu))
        lame = e * nu / ((1 + nu) * (1 - 2 * nu))
        matrix = np.diag([2 * mu] * 3 + [mu] * 3)
        matrix[:3, :3] += lame
    elif type(raw) is ElasticAnisotropic3DProperties:
        matrix = np.asarray(raw.stiffness_voigt, dtype=float)
    else:
        raise TypeError("Uniform Hex8 supports constant 3D elastic materials only.")
    if frame is None:
        return matrix
    basis = np.asarray(frame.basis, dtype=float)
    result = np.empty((6, 6))
    order = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    for column, (i, j) in enumerate(order):
        strain = np.zeros((3, 3))
        strain[i, j] = strain[j, i] = 1 if i == j else 0.5
        local = basis.T @ strain @ basis
        vector = np.array([local[a, b] * (1 if a == b else 2) for a, b in order])
        stress = matrix @ vector
        tensor = stress[np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])]
        global_stress = basis @ tensor @ basis.T
        result[:, column] = [global_stress[a, b] for a, b in order]
    return result


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
        matrices[indices] = _material_matrix(record.item)
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
    def __init__(self, internal, external):
        self.internal, self.external = internal, external
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
        }

    def restore(self, record):
        if record != self.snapshot():
            raise ValueError("Uniform Hex8 checkpoint operator identity mismatch.")


class _Energy:
    def __init__(self, internal):
        self.internal = internal

    def evaluate(self, *, displacement, velocity):
        from .fields import unwrap

        values = self.internal.energies()
        kinetic = float(
            0.5 * np.sum(self.internal.mass_diagonal * unwrap(velocity).x.array ** 2)
        )
        physical = values["strain_energy"] + kinetic
        return {
            **values,
            "kinetic_energy": kinetic,
            "total_mechanical_energy": physical,
            "total_discrete_energy": physical + values["hourglass_energy"],
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
            "energy_balance_scope": "components_only_no_external_work_closure",
            "execution_scope": "serial_small_strain_elastic",
        }


def lower(model, request):
    from . import constraints, problems, state, time
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
        not isinstance(item, constraints.DirichletConstraint)
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
    residual = _Residual(internal, external)
    dt = options.pop("dt")
    stable = internal.stable_dt()
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
        from .time.explicit import _owned_dirichlet_dofs

        history.a.value.x.array[_owned_dirichlet_dofs(bcs)] = 0
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
        history_monitor=_Energy(internal),
        stability={
            "dt_limit": stable,
            "method": "positive_cell_mass_rayleigh_bound",
            "scope": "fixed_elastic_bulk_and_hourglass_only",
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
