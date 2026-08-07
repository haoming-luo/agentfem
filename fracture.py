"""Experimental finite-strain dynamic-fracture building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite, sqrt

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from . import fields as field_api
from . import interfaces as interface_api
from . import operators
from .constitutive import hyperelasticity
from .operators.core import OperatorForm
from .kernel import dofs


def finite_strain_internal_force(
    displacement,
    test_function,
    material: hyperelasticity.NeoHookeanProperties,
    *,
    measure=ufl.dx,
    name: str = "F_internal_finite_strain",
) -> OperatorForm:
    """Return the current Total-Lagrangian Neo-Hookean internal force."""

    if not isinstance(material, hyperelasticity.NeoHookeanProperties):
        raise TypeError("finite_strain_internal_force requires NeoHookeanProperties.")
    expression = hyperelasticity.internal_virtual_work(
        displacement,
        test_function,
        material,
        measure=measure,
    )
    return OperatorForm(
        name=name,
        expression=expression,
        kind="finite_strain_internal_force",
        role="vector",
        family="total_lagrangian_neo_hookean",
        metadata={
            "kinematics": "finite_strain",
            "configuration": "reference",
            "stress_measure": "first_piola",
            "maturity": "experimental_explicit_consumer",
        },
    )


@dataclass
class FiniteStrainEnergyMonitor:
    """Accepted-frame kinetic and Neo-Hookean bulk energy monitor."""

    mass: object
    material: hyperelasticity.NeoHookeanProperties
    measure: object = ufl.dx

    def evaluate(self, *, displacement, velocity) -> dict[str, float]:
        from . import operators

        selected = field_api.unwrap(displacement)
        kinetic = 0.5 * operators.quadratic_form(self.mass, velocity)
        density = hyperelasticity.strain_energy_density(selected, self.material)
        local = fem.assemble_scalar(fem.form(density * self.measure))
        bulk = selected.function_space.mesh.comm.allreduce(local, op=MPI.SUM)
        return {
            "kinetic_energy": float(kinetic),
            "bulk_strain_energy": float(bulk),
            "total_mechanical_energy": float(kinetic + bulk),
        }


class DofMappedCohesiveForce:
    """Map the serial 2D cohesive facet kernel to vector finite-element dofs."""

    def __init__(self, assembler, displacement, *, node_to_block_dof):
        if not isinstance(assembler, interface_api.ModeICohesiveFacetAssembler):
            raise TypeError(
                "DofMappedCohesiveForce requires ModeICohesiveFacetAssembler."
            )
        function = field_api.unwrap(displacement)
        if function.function_space.mesh.comm.size != 1:
            raise NotImplementedError(
                "The first cohesive dof adapter is serial-only; MPI ownership "
                "identity must be verified before distributed use."
            )
        block_size = int(function.function_space.dofmap.index_map_bs)
        mapping = np.asarray(node_to_block_dof, dtype=int)
        number_of_blocks = function.x.array.size // block_size
        if mapping.shape != (assembler.number_of_nodes,):
            raise ValueError(
                "node_to_block_dof must map every cohesive assembly node."
            )
        if np.any(mapping < 0) or np.any(mapping >= number_of_blocks):
            raise ValueError("node_to_block_dof contains an invalid block dof.")
        if np.unique(mapping).size != mapping.size:
            raise ValueError("Every cohesive assembly node needs an independent dof.")
        if block_size != assembler.topology.normals.shape[1]:
            raise ValueError("Displacement block size and interface dimension differ.")
        self.assembler = assembler
        self.displacement = function
        self.node_to_block_dof = mapping
        self.block_size = block_size

    def for_displacement(self, displacement) -> "DofMappedCohesiveForce":
        """Bind the same interface state and identity to an active state field."""

        selected = field_api.unwrap(displacement)
        current = self.displacement
        if selected.function_space.mesh is not current.function_space.mesh:
            raise ValueError(
                "A cohesive force can only be rebound within its original mesh."
            )
        return DofMappedCohesiveForce(
            self.assembler,
            displacement,
            node_to_block_dof=self.node_to_block_dof,
        )

    def begin(self):
        values = self.displacement.x.array.reshape((-1, self.block_size))
        return self.assembler.begin(values[self.node_to_block_dof])

    def add_to_vector(self, vector) -> None:
        response = self.begin()
        array = vector.array.reshape((-1, self.block_size))
        np.add.at(
            array,
            self.node_to_block_dof,
            response.internal_force,
        )

    def commit(self) -> None:
        self.assembler.commit()

    def rollback(self) -> None:
        self.assembler.rollback()

    def current_response(self):
        if self.assembler.last_committed_response is not None:
            return self.assembler.last_committed_response
        response = self.begin()
        self.rollback()
        return response

    def summary(self) -> dict[str, object]:
        return {
            "kind": "dof_mapped_cohesive_force",
            "interface": self.assembler.topology.summary(),
            "law": self.assembler.law.summary(),
            "parallel_scope": "serial_experimental",
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "agentfem.dof-mapped-cohesive-force.v1",
            "node_to_block_dof": self.node_to_block_dof.tolist(),
            "negative_nodes": self.assembler.topology.negative_nodes.tolist(),
            "positive_nodes": self.assembler.topology.positive_nodes.tolist(),
            "law": self.assembler.law.summary(),
            "state": self.assembler.state.snapshot(),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("schema") != "agentfem.dof-mapped-cohesive-force.v1":
            raise ValueError("Unsupported cohesive dof-state schema.")
        checks = {
            "node-to-dof map": (
                snapshot.get("node_to_block_dof"),
                self.node_to_block_dof.tolist(),
            ),
            "negative facet topology": (
                snapshot.get("negative_nodes"),
                self.assembler.topology.negative_nodes.tolist(),
            ),
            "positive facet topology": (
                snapshot.get("positive_nodes"),
                self.assembler.topology.positive_nodes.tolist(),
            ),
            "cohesive law": (snapshot.get("law"), self.assembler.law.summary()),
        }
        for label, (stored, current) in checks.items():
            if stored != current:
                raise ValueError(
                    f"Cohesive checkpoint {label} differs: "
                    f"stored={stored!r}, current={current!r}."
                )
        self.assembler.state.restore(snapshot["state"])
        self.assembler.last_committed_response = None

    def stability_inputs(self, mass) -> dict[str, float]:
        """Return a conservative interface-oscillator screening tuple."""

        diagonal = np.asarray(
            mass.mass if hasattr(mass, "mass") else mass,
            dtype=float,
        ).reshape((-1, self.block_size))
        nodal_mass = diagonal[self.node_to_block_dof, 0]
        topology = self.assembler.topology
        negative = nodal_mass[topology.negative_nodes]
        positive = nodal_mass[topology.positive_nodes]
        return {
            "interface_stiffness": float(self.assembler.law.initial_stiffness),
            "interface_area": float(
                np.min(topology.lengths) * self.assembler.thickness
            ),
            "negative_mass": float(np.min(negative)),
            "positive_mass": float(np.min(positive)),
        }


def p1_input_node_to_block_dof(displacement, *, number_of_input_nodes: int):
    """Recover audited input-node identity for a first-order vector space.

    Coincident cohesive nodes cannot be matched by coordinates.  DOLFINx
    retains their distinct input indices on the geometry map, so this routine
    walks cell-local geometry and field dofmaps together and verifies that
    every input node resolves to exactly one block dof.
    """

    function = field_api.unwrap(displacement)
    space = function.function_space
    domain = space.mesh
    if domain.comm.size != 1:
        raise NotImplementedError(
            "Automatic cohesive node-to-DOF recovery is serial-only until "
            "distributed input-node ownership is verified."
        )
    if int(space.dofmap.index_map_bs) != int(domain.geometry.dim):
        raise ValueError("Cohesive displacement must be one blocked vector space.")
    requested = int(number_of_input_nodes)
    if requested <= 0:
        raise ValueError("number_of_input_nodes must be positive.")
    geometry_maps = getattr(domain.geometry, "dofmaps", None)
    if geometry_maps is None:
        geometry_dofmap = domain.geometry.dofmap
    else:
        geometry_dofmap = geometry_maps[0]
    input_indices = np.asarray(domain.geometry.input_global_indices, dtype=int)
    mapping = np.full(requested, -1, dtype=int)
    number_of_cells = int(
        domain.topology.index_map(domain.topology.dim).size_local
    )
    for cell in range(number_of_cells):
        geometry_dofs = np.asarray(geometry_dofmap[cell], dtype=int)
        field_dofs = np.asarray(space.dofmap.cell_dofs(cell), dtype=int)
        if geometry_dofs.size != field_dofs.size:
            raise ValueError(
                "Automatic cohesive mapping requires first-order nodal geometry "
                "and a first-order displacement field."
            )
        for geometry_dof, field_dof in zip(
            geometry_dofs, field_dofs, strict=True
        ):
            source_node = int(input_indices[geometry_dof])
            if not 0 <= source_node < requested:
                raise ValueError("DOLFINx geometry contains an unexpected input node id.")
            previous = mapping[source_node]
            if previous not in {-1, int(field_dof)}:
                raise RuntimeError(
                    "One input node resolved to inconsistent displacement block dofs."
                )
            mapping[source_node] = int(field_dof)
    missing = np.flatnonzero(mapping < 0)
    if missing.size:
        raise ValueError(
            "Some split input nodes are absent from the displacement space: "
            f"{missing.tolist()}."
        )
    if np.unique(mapping).size != mapping.size:
        raise RuntimeError(
            "Coincident split nodes lost independent displacement identities."
        )
    return mapping


def mode_i_cohesive_force(
    split: interface_api.SplitInterfaceMesh,
    displacement,
    law: interface_api.BilinearCohesiveLaw,
    *,
    normal_hint,
    thickness: float = 1.0,
    tolerance: float = 1.0e-10,
) -> DofMappedCohesiveForce:
    """Build the executable Mode-I force directly from a split mesh contract."""

    if not isinstance(split, interface_api.SplitInterfaceMesh):
        raise TypeError("mode_i_cohesive_force requires SplitInterfaceMesh.")
    topology = interface_api.pair_coincident_line_facets(
        split.coordinates,
        split.negative_facets,
        split.positive_facets,
        normal_hint=normal_hint,
        tolerance=tolerance,
    )
    assembler = interface_api.ModeICohesiveFacetAssembler(
        topology,
        law,
        number_of_nodes=split.coordinates.shape[0],
        thickness=thickness,
    )
    mapping = p1_input_node_to_block_dof(
        displacement,
        number_of_input_nodes=split.coordinates.shape[0],
    )
    return DofMappedCohesiveForce(
        assembler,
        displacement,
        node_to_block_dof=mapping,
    )


class FiniteStrainCohesiveResidual:
    """Assemble bulk UFL and paired-facet interface forces into one residual."""

    def __init__(self, bulk, cohesive: DofMappedCohesiveForce):
        self.bulk = bulk
        self.cohesive = cohesive

    def assemble_vector(self):
        vector = operators.assemble_vector(self.bulk)
        try:
            self.cohesive.add_to_vector(vector)
        except Exception:
            vector.destroy()
            self.cohesive.rollback()
            raise
        return vector

    def commit(self) -> None:
        self.cohesive.commit()

    def rollback(self) -> None:
        self.cohesive.rollback()

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "agentfem.finite-strain-cohesive-residual.v1",
            "cohesive": self.cohesive.snapshot(),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("schema") != "agentfem.finite-strain-cohesive-residual.v1":
            raise ValueError("Unsupported finite-strain cohesive residual schema.")
        self.cohesive.restore(snapshot["cohesive"])

    def summary(self) -> dict[str, object]:
        return {
            "name": "R_bulk_plus_cohesive",
            "kind": "finite_strain_cohesive_residual",
            "role": "residual",
            "family": "total_lagrangian_neo_hookean+cohesive_interface",
            "parts": (
                self.bulk.summary() if hasattr(self.bulk, "summary") else repr(self.bulk),
                self.cohesive.summary(),
            ),
            "maturity": "experimental_serial_global_consumer",
        }


class MassProportionalDampingResidual:
    """Add ``alpha M v_mid`` with transactional dissipation accounting."""

    def __init__(self, base, *, mass, velocity, coefficient: float, dt: float):
        selected = float(coefficient)
        selected_dt = float(dt)
        if not isfinite(selected) or selected < 0.0:
            raise ValueError("Mass-proportional damping must be finite and nonnegative.")
        if not isfinite(selected_dt) or selected_dt <= 0.0:
            raise ValueError("Damping residual requires dt > 0.")
        self.base = base
        self.mass = mass
        self.velocity = field_api.unwrap(velocity)
        self.coefficient = selected
        self.dt = selected_dt
        self.dissipated_energy = 0.0
        self._trial_dissipation: float | None = None

    def assemble_vector(self):
        vector = operators.assemble_vector(self.base)
        diagonal = np.asarray(
            self.mass.mass if hasattr(self.mass, "mass") else self.mass,
            dtype=float,
        )
        velocity = np.asarray(self.velocity.x.array, dtype=float)
        if diagonal.shape != velocity.shape or vector.array.shape != velocity.shape:
            vector.destroy()
            raise ValueError("Damping mass, velocity, and residual layouts differ.")
        vector.array[:] += self.coefficient * diagonal * velocity
        dofmap = self.velocity.function_space.dofmap
        owned = int(dofmap.index_map.size_local * dofmap.index_map_bs)
        local_power = self.coefficient * float(
            np.dot(diagonal[:owned] * velocity[:owned], velocity[:owned])
        )
        power = self.velocity.function_space.mesh.comm.allreduce(
            local_power, op=MPI.SUM
        )
        self._trial_dissipation = self.dt * float(power)
        return vector

    def commit(self) -> None:
        if hasattr(self.base, "commit"):
            self.base.commit()
        if self._trial_dissipation is None:
            raise RuntimeError("No damping trial is available to commit.")
        self.dissipated_energy += self._trial_dissipation
        self._trial_dissipation = None

    def rollback(self) -> None:
        if hasattr(self.base, "rollback"):
            self.base.rollback()
        self._trial_dissipation = None

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "agentfem.mass-proportional-damping-residual.v1",
            "coefficient": self.coefficient,
            "dt": self.dt,
            "dissipated_energy": self.dissipated_energy,
            "base": (
                self.base.snapshot() if hasattr(self.base, "snapshot") else None
            ),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("schema") != "agentfem.mass-proportional-damping-residual.v1":
            raise ValueError("Unsupported damping-residual checkpoint schema.")
        if not np.isclose(float(snapshot["coefficient"]), self.coefficient):
            raise ValueError("Checkpoint damping coefficient differs.")
        if not np.isclose(float(snapshot["dt"]), self.dt):
            raise ValueError("Checkpoint damping time increment differs.")
        base = snapshot.get("base")
        if base is not None:
            if not hasattr(self.base, "restore"):
                raise ValueError("Checkpoint contains base state without a consumer.")
            self.base.restore(base)
        elif hasattr(self.base, "restore"):
            raise ValueError("Checkpoint lacks required base residual state.")
        self.dissipated_energy = float(snapshot["dissipated_energy"])
        self._trial_dissipation = None

    def summary(self) -> dict[str, object]:
        return {
            "name": "R_plus_mass_proportional_damping",
            "kind": "mass_proportional_damping_residual",
            "coefficient": self.coefficient,
            "dt": self.dt,
            "dissipated_energy": self.dissipated_energy,
            "base": self.base.summary() if hasattr(self.base, "summary") else repr(self.base),
        }


@dataclass
class DampingEnergyMonitor:
    """Add accepted viscous dissipation to an existing mechanical monitor."""

    energy: object
    damping: MassProportionalDampingResidual

    def restore(self, history_record: dict[str, float]) -> None:
        if hasattr(self.energy, "restore"):
            self.energy.restore(history_record)

    def evaluate(self, *, displacement, velocity) -> dict[str, float]:
        values = self.energy.evaluate(
            displacement=displacement,
            velocity=velocity,
        )
        dissipated = float(self.damping.dissipated_energy)
        values["numerical_damping_dissipation"] = dissipated
        if "accounted_internal_kinetic_energy" in values:
            values["accounted_internal_kinetic_energy"] += dissipated
        elif "total_mechanical_energy" in values:
            values["total_mechanical_energy"] += dissipated
        return values

@dataclass
class FiniteStrainCohesiveEnergyMonitor:
    """Typed accepted-frame energy for bulk plus cohesive dynamics."""

    bulk: FiniteStrainEnergyMonitor
    cohesive: DofMappedCohesiveForce
    _initial_dissipation: float | None = None

    def restore(self, history_record: dict[str, float]) -> None:
        """Restore the energy reference retained in transient history."""

        if "initial_cohesive_dissipation" not in history_record:
            raise ValueError(
                "Cohesive restart history lacks initial_cohesive_dissipation."
            )
        self._initial_dissipation = float(
            history_record["initial_cohesive_dissipation"]
        )

    def evaluate(self, *, displacement, velocity) -> dict[str, float]:
        values = self.bulk.evaluate(displacement=displacement, velocity=velocity)
        response = self.cohesive.current_response()
        if self._initial_dissipation is None:
            self._initial_dissipation = float(response.dissipated_energy)
        fracture = float(response.dissipated_energy) - self._initial_dissipation
        stored = float(response.stored_energy)
        accounted = (
            values["kinetic_energy"]
            + values["bulk_strain_energy"]
            + stored
            + fracture
        )
        values.update(
            {
                "cohesive_stored_energy": stored,
                "cohesive_fracture_dissipation": fracture,
                "initial_cohesive_dissipation": self._initial_dissipation,
                "accounted_internal_kinetic_energy": accounted,
            }
        )
        # The bulk-only total is intentionally removed: it is incomplete once
        # an interface consumer is active.
        values.pop("total_mechanical_energy", None)
        return values


@dataclass
class DynamicEnergyLedger:
    """Accepted-frame external work and mechanical-energy closure.

    Natural-load work and strong prescribed-motion work are integrated with
    the trapezoidal rule between accepted configurations.  The latter uses
    the constrained generalized force ``M a + R``.  MPC, contact, and weak
    constraints require their own dual variables and are not silently folded
    into this first ledger.
    """

    energy: object
    state: object
    mass: object
    residual: object
    natural_force: object | None = None
    prescribed: tuple[object, ...] = ()
    _previous_displacement: np.ndarray | None = None
    _previous_natural_force: np.ndarray | None = None
    _previous_prescribed_force: np.ndarray | None = None
    _natural_work: float = 0.0
    _prescribed_work: float = 0.0
    _initial_accounted_energy: float | None = None

    def _owned_size(self, displacement) -> int:
        function = field_api.unwrap(displacement)
        dofmap = function.function_space.dofmap
        return int(dofmap.index_map.size_local * dofmap.index_map_bs)

    def _assemble_owned(self, operator, displacement) -> np.ndarray:
        owned = self._owned_size(displacement)
        if operator is None:
            return np.zeros(owned, dtype=float)
        vector = operators.assemble_vector(operator)
        try:
            return np.asarray(vector.array[:owned], dtype=float).copy()
        finally:
            vector.destroy()

    def _prescribed_dofs(self, displacement) -> np.ndarray:
        from . import constraints as constraint_api

        owned = self._owned_size(displacement)
        selected = []
        unsupported = []
        for item in constraint_api.dirichlet_constraints(self.prescribed):
            bc = getattr(item, "bc", None)
            if bc is None:
                unsupported.append(type(item).__name__)
                continue
            indices, first_ghost = bc.dof_indices()
            selected.extend(
                int(value)
                for value in np.asarray(indices[:first_ghost], dtype=np.int64)
                if int(value) < owned
            )
        if unsupported:
            raise NotImplementedError(
                "Dynamic prescribed-motion work requires inspectable strong "
                f"Dirichlet constraints; unsupported={tuple(unsupported)}."
            )
        return (
            np.empty(0, dtype=np.int64)
            if not selected
            else np.unique(np.asarray(selected, dtype=np.int64))
        )

    def _sample(self, displacement, acceleration):
        function = field_api.unwrap(displacement)
        acceleration_function = field_api.unwrap(acceleration)
        owned = self._owned_size(displacement)
        current = np.asarray(function.x.array[:owned], dtype=float).copy()
        natural = self._assemble_owned(self.natural_force, displacement)
        prescribed_force = np.zeros(owned, dtype=float)
        constrained = self._prescribed_dofs(displacement)
        if constrained.size:
            try:
                residual = self._assemble_owned(self.residual, displacement)
            finally:
                if hasattr(self.residual, "rollback"):
                    self.residual.rollback()
            diagonal = np.asarray(
                self.mass.mass if hasattr(self.mass, "mass") else self.mass,
                dtype=float,
            )[:owned]
            inertia = diagonal * np.asarray(
                acceleration_function.x.array[:owned], dtype=float
            )
            prescribed_force[constrained] = (
                residual[constrained] + inertia[constrained]
            )
        return current, natural, prescribed_force

    @staticmethod
    def _global_dot(displacement, left, right) -> float:
        function = field_api.unwrap(displacement)
        local = float(np.dot(left, right))
        return float(function.function_space.mesh.comm.allreduce(local, op=MPI.SUM))

    @staticmethod
    def _accounted(values: dict[str, float]) -> float:
        if "accounted_internal_kinetic_energy" in values:
            return float(values["accounted_internal_kinetic_energy"])
        return float(values["total_mechanical_energy"])

    def restore(self, history_record: dict[str, float]) -> None:
        required = {
            "natural_load_work",
            "prescribed_motion_work",
            "initial_accounted_energy",
        }
        missing = sorted(required.difference(history_record))
        if missing:
            raise ValueError(
                "Dynamic energy restart lacks channels: " + ", ".join(missing)
            )
        if hasattr(self.energy, "restore"):
            self.energy.restore(history_record)
        self._natural_work = float(history_record["natural_load_work"])
        self._prescribed_work = float(history_record["prescribed_motion_work"])
        self._initial_accounted_energy = float(
            history_record["initial_accounted_energy"]
        )
        (
            self._previous_displacement,
            self._previous_natural_force,
            self._previous_prescribed_force,
        ) = self._sample(self.state.u, self.state.a)

    def evaluate(self, *, displacement, velocity) -> dict[str, float]:
        values = self.energy.evaluate(
            displacement=displacement,
            velocity=velocity,
        )
        current, natural, prescribed_force = self._sample(
            displacement,
            self.state.a,
        )
        if self._previous_displacement is not None:
            increment = current - self._previous_displacement
            self._natural_work += 0.5 * self._global_dot(
                displacement,
                self._previous_natural_force + natural,
                increment,
            )
            self._prescribed_work += 0.5 * self._global_dot(
                displacement,
                self._previous_prescribed_force + prescribed_force,
                increment,
            )
        self._previous_displacement = current
        self._previous_natural_force = natural
        self._previous_prescribed_force = prescribed_force
        accounted = self._accounted(values)
        if self._initial_accounted_energy is None:
            self._initial_accounted_energy = accounted
        external = self._natural_work + self._prescribed_work
        balance = self._initial_accounted_energy + external - accounted
        scale = max(
            abs(self._initial_accounted_energy),
            abs(external),
            abs(accounted),
            np.finfo(float).eps,
        )
        values.update(
            {
                "natural_load_work": self._natural_work,
                "prescribed_motion_work": self._prescribed_work,
                "external_work": external,
                "initial_accounted_energy": self._initial_accounted_energy,
                "total_accounted_energy": accounted,
                "energy_balance_error": balance,
                "relative_energy_balance_error": abs(balance) / scale,
            }
        )
        return values


@dataclass(frozen=True)
class IsotropicWaveSpeeds:
    """Reference small-on-zero wave speeds for one isotropic material."""

    pressure: float
    shear: float
    rayleigh: float
    configuration: str = "unstretched_reference"

    def summary(self) -> dict[str, object]:
        return {
            "pressure_wave_speed": self.pressure,
            "shear_wave_speed": self.shear,
            "rayleigh_wave_speed": self.rayleigh,
            "configuration": self.configuration,
        }


@dataclass(frozen=True)
class IncrementalWaveSpeeds:
    """Small-on-large bulk-wave modes about one homogeneous deformation."""

    speeds: np.ndarray
    reference_speeds: np.ndarray
    polarizations: np.ndarray
    acoustic_tensor: np.ndarray
    reference_direction: np.ndarray
    current_direction: np.ndarray
    direction_configuration: str
    deformation_jacobian: float

    @property
    def fastest(self) -> float:
        return float(self.speeds[-1])

    @property
    def slowest(self) -> float:
        return float(self.speeds[0])

    def summary(self) -> dict[str, object]:
        return {
            "kind": "small_on_large_bulk_wave_speeds",
            "speeds": self.speeds.tolist(),
            "reference_speeds": self.reference_speeds.tolist(),
            "reference_direction": self.reference_direction.tolist(),
            "current_direction": self.current_direction.tolist(),
            "direction_configuration": self.direction_configuration,
            "deformation_jacobian": self.deformation_jacobian,
            "density_measure": "reference",
            "rayleigh_speed": None,
        }


def neo_hookean_material_tangent(
    deformation_gradient,
    material: hyperelasticity.NeoHookeanProperties,
) -> np.ndarray:
    """Return ``A[i,J,k,L] = dP[i,J]/dF[k,L]`` for the declared energy."""

    if not isinstance(material, hyperelasticity.NeoHookeanProperties):
        raise TypeError("neo_hookean_material_tangent requires NeoHookeanProperties.")
    F = np.asarray(deformation_gradient, dtype=float)
    if F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape[0] not in {2, 3}:
        raise ValueError("deformation_gradient must be one finite 2x2 or 3x3 array.")
    if not np.all(np.isfinite(F)):
        raise ValueError("deformation_gradient must contain only finite values.")
    J = float(np.linalg.det(F))
    if J <= 0.0:
        raise ValueError("Incremental material tangent requires det(F) > 0.")
    inverse_transpose = np.linalg.inv(F).T
    dimension = F.shape[0]
    identity = np.eye(dimension)
    mu = float(material.mu)
    lam = float(material.lambda_)
    volumetric = lam * np.log(J) - mu
    tangent = np.empty((dimension, dimension, dimension, dimension), dtype=float)
    for i in range(dimension):
        for J_index in range(dimension):
            for k in range(dimension):
                for L in range(dimension):
                    tangent[i, J_index, k, L] = (
                        mu * identity[i, k] * identity[J_index, L]
                        + lam
                        * inverse_transpose[i, J_index]
                        * inverse_transpose[k, L]
                        - volumetric
                        * inverse_transpose[i, L]
                        * inverse_transpose[k, J_index]
                    )
    return tangent


def incremental_wave_speeds(
    deformation_gradient,
    direction,
    material: hyperelasticity.NeoHookeanProperties,
    *,
    direction_configuration: str = "current",
) -> IncrementalWaveSpeeds:
    """Return homogeneous small-on-large bulk-wave speeds.

    The material acoustic tensor is contracted in the reference
    configuration with ``rho0``.  When a current direction is supplied, it is
    pulled back to the reference body and the resulting phase speeds are
    pushed forward.  This distinction is essential after prestrain.  The
    function does not infer a prestrained Rayleigh speed; that requires the
    appropriate surface-wave secular problem and boundary orientation.
    """

    if material.density is None:
        raise ValueError("Incremental wave speeds require material density.")
    F = np.asarray(deformation_gradient, dtype=float)
    tangent = neo_hookean_material_tangent(F, material)
    selected = np.asarray(direction, dtype=float)
    if selected.shape != (F.shape[0],) or not np.all(np.isfinite(selected)):
        raise ValueError("direction must be one finite vector matching F.")
    norm = float(np.linalg.norm(selected))
    if norm <= 0.0:
        raise ValueError("direction must be nonzero.")
    selected /= norm
    configuration = str(direction_configuration).strip().lower()
    if configuration == "current":
        current_direction = selected
        pulled = F.T @ current_direction
        stretch = float(np.linalg.norm(pulled))
        reference_direction = pulled / stretch
        speed_scale = stretch
    elif configuration == "reference":
        reference_direction = selected
        pushed_wave_vector = np.linalg.solve(F.T, reference_direction)
        inverse_stretch = float(np.linalg.norm(pushed_wave_vector))
        current_direction = pushed_wave_vector / inverse_stretch
        speed_scale = 1.0 / inverse_stretch
    else:
        raise ValueError("direction_configuration must be 'reference' or 'current'.")
    acoustic = np.einsum(
        "iJkL,J,L->ik",
        tangent,
        reference_direction,
        reference_direction,
    )
    acoustic = 0.5 * (acoustic + acoustic.T)
    eigenvalues, polarizations = np.linalg.eigh(acoustic)
    tolerance = np.finfo(float).eps * max(1.0, float(np.linalg.norm(acoustic))) * 100.0
    if np.min(eigenvalues) <= tolerance:
        raise ValueError(
            "The acoustic tensor is not positive definite in this direction; "
            "the homogeneous prestrained state has lost strong ellipticity or "
            "is too close to the numerical tolerance."
        )
    reference_speeds = np.sqrt(eigenvalues / float(material.density))
    return IncrementalWaveSpeeds(
        speeds=np.asarray(reference_speeds * speed_scale, dtype=float),
        reference_speeds=np.asarray(reference_speeds, dtype=float),
        polarizations=np.asarray(polarizations, dtype=float),
        acoustic_tensor=np.asarray(acoustic, dtype=float),
        reference_direction=np.asarray(reference_direction, dtype=float),
        current_direction=np.asarray(current_direction, dtype=float),
        direction_configuration=configuration,
        deformation_jacobian=float(np.linalg.det(F)),
    )


def isotropic_reference_wave_speeds(
    material: hyperelasticity.NeoHookeanProperties,
) -> IsotropicWaveSpeeds:
    """Return unstretched 3D isotropic ``c_d``, ``c_s``, and ``c_R``.

    The Rayleigh value is the positive root of the classical isotropic secular
    polynomial.  These values are V1 reference checks, not prestrained crack
    regime classifiers.
    """

    if material.density is None:
        raise ValueError("Wave speeds require a material density.")
    rho = float(material.density)
    cs = sqrt(material.mu / rho)
    cp = sqrt((material.lambda_ + 2.0 * material.mu) / rho)
    ratio = (cs / cp) ** 2
    # Polynomial in xi=(c_R/c_s)^2:
    # xi^3 - 8 xi^2 + (24 - 16 beta) xi - 16 (1 - beta) = 0.
    roots = np.roots([1.0, -8.0, 24.0 - 16.0 * ratio, -16.0 * (1.0 - ratio)])
    candidates = [
        float(root.real)
        for root in roots
        if abs(float(root.imag)) < 1.0e-10 and 0.0 < float(root.real) < 1.0
    ]
    if len(candidates) != 1:
        raise RuntimeError("Could not identify the physical Rayleigh-wave root.")
    return IsotropicWaveSpeeds(
        pressure=cp,
        shear=cs,
        rayleigh=cs * sqrt(candidates[0]),
    )


@dataclass(frozen=True)
class StableTimeIncrement:
    """Visible body/interface estimate for central difference."""

    selected: float
    body_limit: float
    interface_limit: float | None
    safety_factor: float
    controller: str

    def summary(self) -> dict[str, object]:
        return {
            "selected": self.selected,
            "body_limit": self.body_limit,
            "interface_limit": self.interface_limit,
            "safety_factor": self.safety_factor,
            "controller": self.controller,
            "maturity": "screening_estimate",
        }


@dataclass(frozen=True)
class CohesiveCrackHistory:
    """Crack-front position and window-fitted speed on a fixed path."""

    time: np.ndarray
    position: np.ndarray
    speed: np.ndarray
    damage_threshold: float
    fit_window: int
    direction: str

    def summary(self) -> dict[str, object]:
        finite_speed = self.speed[np.isfinite(self.speed)]
        return {
            "kind": "cohesive_crack_history",
            "frames": int(self.time.size),
            "damage_threshold": self.damage_threshold,
            "fit_window": self.fit_window,
            "direction": self.direction,
            "maximum_speed": (
                None if finite_speed.size == 0 else float(np.max(finite_speed))
            ),
            "method": "threshold_interpolation_then_local_linear_fit",
        }


@dataclass(frozen=True)
class PreloadTransferReport:
    """Evidence for a quasi-static displacement to Explicit state transfer."""

    mode: str
    residual_force_norm: float
    acceleration_norm: float
    force_tolerance: float
    equilibrium_accepted: bool
    initial_velocity: str

    def summary(self) -> dict[str, object]:
        return {
            "kind": "preload_to_explicit_state_transfer",
            "mode": self.mode,
            "residual_force_norm": self.residual_force_norm,
            "acceleration_norm": self.acceleration_norm,
            "force_tolerance": self.force_tolerance,
            "equilibrium_accepted": self.equilibrium_accepted,
            "initial_velocity": self.initial_velocity,
        }


def transfer_preload_to_explicit(
    preload_displacement,
    *,
    state,
    mass,
    residual,
    initial_velocity=None,
    mode: str = "equilibrium",
    force_tolerance: float = 1.0e-8,
    acceleration_projection=None,
) -> PreloadTransferReport:
    """Initialize ``u/v/a`` consistently from a quasi-static preload state.

    ``mode='equilibrium'`` rejects a force imbalance above the declared
    tolerance.  ``mode='release'`` retains the computed initial acceleration
    as the physical release/impact condition.  A cohesive residual is rolled
    back after evaluation, so transfer does not advance irreversible damage.
    """

    selected_mode = str(mode).strip().lower()
    if selected_mode not in {"equilibrium", "release"}:
        raise ValueError("Preload transfer mode must be 'equilibrium' or 'release'.")
    tolerance = float(force_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("force_tolerance must be finite and nonnegative.")
    state.u.assign(preload_displacement)
    state.u_next.assign(preload_displacement)
    if initial_velocity is None:
        for item in (state.v, state.v_mid, state.v_next):
            function = field_api.unwrap(item)
            function.x.array[:] = 0.0
            function.x.scatter_forward()
        velocity_label = "zero"
    else:
        for item in (state.v, state.v_mid, state.v_next):
            item.assign(initial_velocity)
        velocity_label = "transferred"

    vector = operators.assemble_vector(residual)
    try:
        force_norm = float(vector.norm())
        inverse = mass.inv_mass if hasattr(mass, "inv_mass") else np.asarray(mass)
        dofs.assign_owned(state.a_next, -vector.array * inverse)
    finally:
        vector.destroy()
        if hasattr(residual, "rollback"):
            residual.rollback()
    if acceleration_projection is not None:
        acceleration_projection(state.a_next)
    state.a.assign(state.a_next)
    acceleration = field_api.unwrap(state.a)
    local_squared = float(np.dot(dofs.owned_array(acceleration), dofs.owned_array(acceleration)))
    acceleration_norm = sqrt(
        acceleration.function_space.mesh.comm.allreduce(local_squared, op=MPI.SUM)
    )
    accepted = force_norm <= tolerance
    if selected_mode == "equilibrium" and not accepted:
        raise RuntimeError(
            "Preload-to-Explicit transfer is not in equilibrium: residual force "
            f"norm {force_norm:.6e} exceeds tolerance {tolerance:.6e}. "
            "Use mode='release' only when this imbalance is the declared loading event."
        )
    return PreloadTransferReport(
        mode=selected_mode,
        residual_force_norm=force_norm,
        acceleration_norm=acceleration_norm,
        force_tolerance=tolerance,
        equilibrium_accepted=accepted,
        initial_velocity=velocity_label,
    )


def cohesive_crack_tip(
    path_coordinate,
    damage,
    *,
    threshold: float = 0.95,
    direction: str = "increasing",
) -> float:
    """Locate the contiguous crack front by interpolating a damage threshold."""

    coordinate = np.asarray(path_coordinate, dtype=float)
    values = np.asarray(damage, dtype=float)
    if coordinate.ndim != 1 or values.shape != coordinate.shape or coordinate.size < 2:
        raise ValueError("path_coordinate and damage must be equal 1D arrays of size >= 2.")
    if np.any(~np.isfinite(coordinate)) or np.any(~np.isfinite(values)):
        raise ValueError("Crack-front inputs must be finite.")
    selected_threshold = float(threshold)
    if not 0.0 < selected_threshold < 1.0:
        raise ValueError("damage threshold must lie strictly between zero and one.")
    order = np.argsort(coordinate)
    if direction == "decreasing":
        order = order[::-1]
    elif direction != "increasing":
        raise ValueError("direction must be 'increasing' or 'decreasing'.")
    x = coordinate[order]
    d = values[order]
    active = d >= selected_threshold
    if not active[0]:
        return float("nan")
    inactive = np.flatnonzero(~active)
    if inactive.size == 0:
        return float(x[-1])
    right = int(inactive[0])
    left = right - 1
    denominator = d[right] - d[left]
    if abs(denominator) <= np.finfo(float).eps:
        return float(0.5 * (x[left] + x[right]))
    fraction = (selected_threshold - d[left]) / denominator
    return float(x[left] + fraction * (x[right] - x[left]))


def crack_tip_history(
    time_values,
    path_coordinate,
    damage_frames,
    *,
    threshold: float = 0.95,
    fit_window: int = 5,
    direction: str = "increasing",
) -> CohesiveCrackHistory:
    """Build a crack history without single-failed-element speed spikes."""

    times = np.asarray(time_values, dtype=float)
    frames = np.asarray(damage_frames, dtype=float)
    coordinate = np.asarray(path_coordinate, dtype=float)
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("time_values must be a strictly increasing 1D array.")
    if frames.shape != (times.size, coordinate.size):
        raise ValueError("damage_frames must have shape (frames, path_points).")
    window = int(fit_window)
    if window < 3 or window % 2 == 0:
        raise ValueError("fit_window must be an odd integer of at least three.")
    position = np.asarray(
        [
            cohesive_crack_tip(
                coordinate,
                frame,
                threshold=threshold,
                direction=direction,
            )
            for frame in frames
        ],
        dtype=float,
    )
    speed = np.full(times.shape, np.nan, dtype=float)
    half = window // 2
    for index in range(times.size):
        start = max(0, index - half)
        stop = min(times.size, index + half + 1)
        valid = np.isfinite(position[start:stop])
        if np.count_nonzero(valid) < 2:
            continue
        t = times[start:stop][valid]
        x = position[start:stop][valid]
        speed[index] = float(np.polyfit(t - np.mean(t), x, 1)[0])
    return CohesiveCrackHistory(
        time=times.copy(),
        position=position,
        speed=speed,
        damage_threshold=float(threshold),
        fit_window=window,
        direction=direction,
    )


def mach_cone_angle(*, crack_speed: float, shear_wave_speed: float) -> float:
    """Return the ideal Mach angle ``asin(c_s / v)`` in radians."""

    speed = float(crack_speed)
    shear = float(shear_wave_speed)
    if not isfinite(speed) or not isfinite(shear) or shear <= 0.0:
        raise ValueError("Crack and shear-wave speeds must be finite and positive.")
    if speed <= shear:
        raise ValueError("A shear Mach cone requires crack_speed > shear_wave_speed.")
    return float(np.arcsin(shear / speed))


def separation_regime(
    *,
    crack_speed: float,
    rayleigh_wave_speed: float,
    shear_wave_speed: float,
    failed_fraction: float,
    simultaneous_failed_fraction: float,
    spall_fraction: float = 0.8,
) -> str:
    """Classify one frame with explicit crack-speed and spall evidence."""

    speed = float(crack_speed)
    rayleigh = float(rayleigh_wave_speed)
    shear = float(shear_wave_speed)
    failed = float(failed_fraction)
    simultaneous = float(simultaneous_failed_fraction)
    threshold = float(spall_fraction)
    if not 0.0 < rayleigh < shear:
        raise ValueError("Wave speeds must satisfy 0 < c_R < c_s.")
    if not all(0.0 <= value <= 1.0 for value in (failed, simultaneous, threshold)):
        raise ValueError("Failure fractions must lie in [0, 1].")
    if failed >= threshold and simultaneous >= threshold:
        return "spall_like"
    if not isfinite(speed):
        return "unresolved"
    if speed <= rayleigh:
        return "sub_rayleigh_crack_like"
    if speed <= shear:
        return "trans_rayleigh"
    return "supershear"


def estimate_stable_time_increment(
    *,
    characteristic_length,
    dilatational_speed: float,
    safety_factor: float = 0.8,
    interface_stiffness: float | None = None,
    interface_area: float | None = None,
    negative_mass: float | None = None,
    positive_mass: float | None = None,
) -> StableTimeIncrement:
    """Estimate explicit stability from body transit and interface oscillator.

    This helper is intentionally labelled a screening estimate.  Element- and
    formulation-specific spectral estimates must replace it before V1 release
    validation.
    """

    lengths = np.asarray(characteristic_length, dtype=float)
    if lengths.size == 0 or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("characteristic_length must contain positive values.")
    speed = float(dilatational_speed)
    factor = float(safety_factor)
    if not isfinite(speed) or speed <= 0.0:
        raise ValueError("dilatational_speed must be finite and positive.")
    if not isfinite(factor) or not (0.0 < factor <= 1.0):
        raise ValueError("safety_factor must satisfy 0 < value <= 1.")
    body = float(np.min(lengths) / speed)

    optional = (
        interface_stiffness,
        interface_area,
        negative_mass,
        positive_mass,
    )
    if all(value is None for value in optional):
        interface = None
    elif any(value is None for value in optional):
        raise ValueError(
            "Interface stability requires stiffness, area, and both side masses."
        )
    else:
        stiffness, area, minus, plus = (float(value) for value in optional)
        if any(not isfinite(value) or value <= 0.0 for value in (stiffness, area, minus, plus)):
            raise ValueError("Interface stability inputs must be finite and positive.")
        reduced_mass = minus * plus / (minus + plus)
        interface = float(2.0 * sqrt(reduced_mass / (stiffness * area)))
    controller = "body" if interface is None or body <= interface else "interface"
    selected_limit = body if interface is None else min(body, interface)
    return StableTimeIncrement(
        selected=factor * selected_limit,
        body_limit=body,
        interface_limit=interface,
        safety_factor=factor,
        controller=controller,
    )


def minimum_cell_nodal_spacing(domain) -> float:
    """Return an MPI-global conservative spacing from cell geometry nodes."""

    geometry_dofmaps = getattr(domain.geometry, "dofmaps", None)
    dofmap = np.asarray(
        geometry_dofmaps[0] if geometry_dofmaps is not None else domain.geometry.dofmap,
        dtype=int,
    )
    owned = int(domain.topology.index_map(domain.topology.dim).size_local)
    local = np.inf
    for cell in range(owned):
        nodes = np.asarray(dofmap[cell], dtype=int).reshape(-1)
        points = np.asarray(domain.geometry.x[nodes, : domain.geometry.dim], dtype=float)
        for left, right in combinations(range(points.shape[0]), 2):
            distance = float(np.linalg.norm(points[right] - points[left]))
            if distance > np.finfo(float).eps:
                local = min(local, distance)
    global_minimum = float(domain.comm.allreduce(local, op=MPI.MIN))
    if not isfinite(global_minimum) or global_minimum <= 0.0:
        raise ValueError("Could not determine a positive cell nodal spacing.")
    return global_minimum


__all__ = [
    "DofMappedCohesiveForce",
    "CohesiveCrackHistory",
    "PreloadTransferReport",
    "FiniteStrainEnergyMonitor",
    "FiniteStrainCohesiveEnergyMonitor",
    "FiniteStrainCohesiveResidual",
    "IsotropicWaveSpeeds",
    "StableTimeIncrement",
    "estimate_stable_time_increment",
    "cohesive_crack_tip",
    "crack_tip_history",
    "finite_strain_internal_force",
    "isotropic_reference_wave_speeds",
    "minimum_cell_nodal_spacing",
    "mach_cone_angle",
    "separation_regime",
    "transfer_preload_to_explicit",
]
