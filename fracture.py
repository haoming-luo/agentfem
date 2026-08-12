"""Experimental finite-strain dynamic-fracture building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from math import isfinite, sqrt
from pathlib import Path
from time import perf_counter

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from . import fields as field_api
from . import interfaces as interface_api
from . import operators
from .constitutive import hyperelasticity
from .operators.core import OperatorForm
from .kernel import dofs
from .fracture_evidence import DynamicFractureEvidenceBundle


_COHESIVE_PETSC_EVENTS = {
    "constitutive": "AgentFEMCohesiveConstitutive",
    "vector": "AgentFEMCohesiveVectorAssembly",
    "matrix": "AgentFEMCohesiveMatrixAssembly",
}


def _cohesive_state_by_key(state, topology) -> dict[str, dict[str, list[float]]]:
    """Encode every cohesive state field by durable physical facet key."""

    factory = getattr(state, "state_arrays", None)
    arrays = (
        factory()
        if callable(factory)
        else {"maximum_opening": np.asarray(state.committed_maximum, dtype=float)}
    )
    points = int(topology.quadrature_points_per_facet)
    records = {}
    for name, values in sorted(arrays.items()):
        selected = np.asarray(values, dtype=float)
        if selected.shape != (int(topology.number_of_points),):
            raise ValueError(f"Cohesive state field {name!r} has an invalid shape.")
        by_facet = selected.reshape((-1, points))
        records[name] = {
            key: by_facet[index].tolist()
            for index, key in enumerate(topology.facet_keys)
        }
    return records


def _restore_cohesive_state_by_key(state, topology, records) -> None:
    """Restore all state fields while allowing execution-local facet order."""

    expected_fields = set(state.state_arrays())
    if set(records) != expected_fields:
        raise ValueError("Cohesive checkpoint state fields differ.")
    expected_keys = set(topology.facet_keys)
    arrays = {}
    for name, by_key in records.items():
        if set(by_key) != expected_keys:
            raise ValueError(
                f"Cohesive checkpoint facet keys differ for state field {name!r}."
            )
        arrays[name] = np.asarray(
            [by_key[key] for key in topology.facet_keys], dtype=float
        ).reshape(-1)
    state.restore_state_arrays(arrays)


def finite_strain_internal_force(
    displacement,
    test_function,
    material,
    *,
    measure=ufl.dx,
    name: str = "F_internal_finite_strain",
) -> OperatorForm:
    """Return the current Total-Lagrangian hyperelastic internal force."""

    if not hyperelasticity.is_finite_strain_hyperelastic(material):
        raise TypeError("finite_strain_internal_force requires a supported material.")
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
        family=(
            "total_lagrangian_neo_hookean"
            if isinstance(material, hyperelasticity.NeoHookeanProperties)
            else "total_lagrangian_mooney_rivlin"
        ),
        metadata={
            "kinematics": "finite_strain",
            "configuration": "reference",
            "stress_measure": "first_piola",
            "maturity": "experimental_explicit_consumer",
        },
    )


@dataclass
class FiniteStrainEnergyMonitor:
    """Accepted-frame kinetic and hyperelastic bulk energy monitor."""

    mass: object
    material: object
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
    """Map a serial cohesive facet kernel to vector finite-element dofs."""

    def __init__(self, assembler, displacement, *, node_to_block_dof):
        supported = (
            interface_api.ModeICohesiveFacetAssembler,
            interface_api.ModeICohesiveSurfaceAssembler,
        )
        if not isinstance(assembler, supported):
            raise TypeError(
                "DofMappedCohesiveForce requires a supported cohesive assembler."
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
        # The constructor proves this map is one-to-one. Direct indexed
        # addition avoids the duplicate-index machinery in ``np.add.at``.
        array[self.node_to_block_dof] += response.internal_force

    def add_to_matrix(self, matrix) -> None:
        """Add the consistent interface tangent to a serial PETSc matrix."""

        values = self.displacement.x.array.reshape((-1, self.block_size))
        elements = self.assembler.tangent_elements(
            values[self.node_to_block_dof]
        )
        matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        components = np.arange(self.block_size, dtype=np.int32)
        for nodes, element in zip(
            elements.nodes, elements.matrices, strict=True
        ):
            blocks = self.node_to_block_dof[nodes]
            scalar_dofs = (
                blocks[:, None] * self.block_size + components[None, :]
            ).reshape(-1).astype(np.int32)
            matrix.setValuesLocal(
                scalar_dofs,
                scalar_dofs,
                element,
                addv=PETSc.InsertMode.ADD_VALUES,
            )

    def commit(self) -> None:
        self.assembler.commit()

    def rollback(self) -> None:
        self.assembler.rollback()

    def initialize_precrack(self, facets) -> None:
        """Initialize selected interface facets before the first increment."""

        self.assembler.initialize_precrack(facets)

    def save_portable_state(self, path):
        """Save physical-keyed interface state for a serial execution."""

        return interface_api.save_portable_cohesive_state(
            path,
            self.assembler.topology,
            self.assembler.state,
            comm=self.displacement.function_space.mesh.comm,
        )

    def load_portable_state(self, path):
        """Restore physical-keyed interface state for a serial execution."""

        return interface_api.load_portable_cohesive_state(
            path,
            self.assembler.topology,
            self.assembler.state,
            comm=self.displacement.function_space.mesh.comm,
        )

    def current_response(self):
        if self.assembler.last_committed_response is not None:
            return self.assembler.last_committed_response
        response = self.begin()
        self.rollback()
        return response

    def cycle_opening(self) -> np.ndarray:
        """Evaluate the current opening without advancing interface history."""

        response = self.begin()
        try:
            return np.asarray(response.opening, dtype=float).reshape(-1).copy()
        finally:
            self.rollback()

    def material_point_response(self):
        """Evaluate local cohesive state at the active displacement field."""

        response = self.begin()
        try:
            return self.assembler.material_point_response(response)
        finally:
            self.rollback()

    def interface_quantities(self) -> dict[str, np.ndarray]:
        """Return standard local interface fields at cohesive quadrature points."""

        response = self.current_response()
        return {
            "JUMP_N": np.asarray(response.opening, dtype=float).copy(),
            "JUMP_T": np.asarray(response.tangential_jump, dtype=float).copy(),
            "TRACTION_N": np.asarray(response.traction, dtype=float).copy(),
            "TRACTION_T": np.asarray(response.tangential_traction, dtype=float).copy(),
            "DAMAGE": np.asarray(response.damage, dtype=float).copy(),
            "MODE_MIXITY": np.asarray(response.mode_mixity, dtype=float).copy(),
        }

    def audit_mode_i(self, **options):
        """Audit accepted kinematics without advancing interface state."""

        return interface_api.audit_mode_i_kinematics(
            self.current_response(), **options
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "dof_mapped_cohesive_force",
            "interface": self.assembler.topology.summary(),
            "law": self.assembler.law.summary(),
            "interface_kinematics": self.assembler.tangential,
            "tangential_stiffness": self.assembler.tangential_stiffness,
            "parallel_scope": "serial_experimental",
            "restart_identity": self.assembler.topology.identity(),
        }

    def snapshot(self) -> dict[str, object]:
        state_records = _cohesive_state_by_key(
            self.assembler.state,
            self.assembler.topology,
        )
        snapshot = {
            "schema": "agentfem.dof-mapped-cohesive-force.v5",
            "node_to_block_dof": self.node_to_block_dof.tolist(),
            "negative_nodes": self.assembler.topology.negative_nodes.tolist(),
            "positive_nodes": self.assembler.topology.positive_nodes.tolist(),
            "interface_identity": self.assembler.topology.identity(),
            "dof_map_role": "execution_local_not_state_identity",
            "law": self.assembler.law.summary(),
            "interface_kinematics": self.assembler.tangential,
            "tangential_stiffness": self.assembler.tangential_stiffness,
            "state_by_field_and_key": state_records,
            "state_identity": "ordered_physical_facet_and_quadrature",
        }
        if "maximum_opening" in state_records:
            snapshot["maximum_opening_by_key"] = state_records["maximum_opening"]
        return snapshot

    def restore(self, snapshot: dict[str, object]) -> None:
        schema = snapshot.get("schema")
        if schema not in {
            "agentfem.dof-mapped-cohesive-force.v1",
            "agentfem.dof-mapped-cohesive-force.v2",
            "agentfem.dof-mapped-cohesive-force.v3",
            "agentfem.dof-mapped-cohesive-force.v4",
            "agentfem.dof-mapped-cohesive-force.v5",
        }:
            raise ValueError("Unsupported cohesive dof-state schema.")
        if schema != "agentfem.dof-mapped-cohesive-force.v5" and (
            self.assembler.tangential != "free"
        ):
            raise ValueError(
                "Legacy cohesive checkpoints do not record interface "
                "kinematics and can only be restored into the historical "
                "free-slip Mode-I consumer. Recreate or explicitly migrate "
                "the checkpoint before using tie, degraded, or mixed behavior."
            )
        if schema == "agentfem.dof-mapped-cohesive-force.v1":
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
        else:
            # Dof numbers are execution-local.  Irreversible state follows the
            # ordered physical interface identity and cohesive law instead.
            checks = {
                "physical interface identity": (
                    snapshot.get("interface_identity"),
                    self.assembler.topology.identity(),
                ),
                "cohesive law": (snapshot.get("law"), self.assembler.law.summary()),
            }
            if schema == "agentfem.dof-mapped-cohesive-force.v5":
                checks.update(
                    {
                        "interface kinematics": (
                            snapshot.get("interface_kinematics"),
                            self.assembler.tangential,
                        ),
                        "tangential stiffness": (
                            snapshot.get("tangential_stiffness"),
                            self.assembler.tangential_stiffness,
                        ),
                    }
                )
        for label, (stored, current) in checks.items():
            if stored != current:
                raise ValueError(
                    f"Cohesive checkpoint {label} differs: "
                    f"stored={stored!r}, current={current!r}."
                )
        if schema in {
            "agentfem.dof-mapped-cohesive-force.v1",
            "agentfem.dof-mapped-cohesive-force.v2",
        }:
            self.assembler.state.restore(snapshot["state"])
        elif schema == "agentfem.dof-mapped-cohesive-force.v3":
            records = snapshot.get("maximum_opening_by_key", {})
            if set(records) != set(self.assembler.topology.facet_keys):
                raise ValueError("Cohesive checkpoint physical facet keys differ.")
            values = np.asarray(
                [records[key] for key in self.assembler.topology.facet_keys],
                dtype=float,
            )
            self.assembler.state.initialize(values.reshape(-1))
        else:
            _restore_cohesive_state_by_key(
                self.assembler.state,
                self.assembler.topology,
                snapshot.get("state_by_field_and_key", {}),
            )
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
        normal_stiffness = float(
            self.assembler.law.initial_stiffness
            if hasattr(self.assembler.law, "initial_stiffness")
            else self.assembler.law.normal_stiffness
        )
        return {
            "interface_stiffness": max(
                normal_stiffness,
                float(self.assembler.tangential_stiffness),
            ),
            "interface_area": float(
                np.min(topology.measures) * self.assembler.thickness
            ),
            "negative_mass": float(np.min(negative)),
            "positive_mass": float(np.min(positive)),
        }


@dataclass(frozen=True)
class NamedCohesiveResponse:
    """Responses and aggregate energy from several named interfaces."""

    by_name: dict[str, object]

    @property
    def stored_energy(self) -> float:
        return float(sum(item.stored_energy for item in self.by_name.values()))

    @property
    def dissipated_energy(self) -> float:
        return float(sum(item.dissipated_energy for item in self.by_name.values()))


class CohesiveForceCollection:
    """Atomically compose independent named cohesive-interface forces.

    Each interface retains its own topology, law, precrack, state, energy and
    restart identity.  The collection is itself a valid cohesive-force
    consumer for the existing bulk-plus-interface residual and energy ledger.
    """

    def __init__(self, interfaces):
        if isinstance(interfaces, dict):
            records = tuple(interfaces.items())
        else:
            records = tuple(interfaces)
        if not records:
            raise ValueError("A cohesive-force collection cannot be empty.")
        names = [str(name).strip() for name, _force in records]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Named cohesive interfaces require unique nonempty names.")
        self._forces = {
            name: force for name, (_declared, force) in zip(names, records, strict=True)
        }
        required = (
            "add_to_vector",
            "commit",
            "rollback",
            "snapshot",
            "restore",
            "summary",
            "current_response",
            "for_displacement",
            "stability_inputs",
        )
        for name, force in self._forces.items():
            missing = [item for item in required if not callable(getattr(force, item, None))]
            if missing:
                raise TypeError(
                    f"Cohesive interface {name!r} is missing operations {missing}."
                )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._forces)

    @property
    def displacement(self):
        selected = tuple(force.displacement for force in self._forces.values())
        first = selected[0]
        if any(value is not first for value in selected[1:]):
            raise RuntimeError(
                "Named cohesive interfaces must share one displacement field."
            )
        return first

    def __getitem__(self, name: str):
        return self._forces[str(name)]

    def for_displacement(self, displacement) -> "CohesiveForceCollection":
        return CohesiveForceCollection(
            {
                name: force.for_displacement(displacement)
                for name, force in self._forces.items()
            }
        )

    def add_to_vector(self, vector) -> None:
        for force in self._forces.values():
            force.add_to_vector(vector)

    def add_to_matrix(self, matrix) -> None:
        for name, force in self._forces.items():
            add = getattr(force, "add_to_matrix", None)
            if not callable(add):
                raise TypeError(
                    f"Cohesive interface {name!r} has no consistent tangent consumer."
                )
            add(matrix)

    def commit(self) -> None:
        for force in self._forces.values():
            force.commit()

    def rollback(self) -> None:
        for force in self._forces.values():
            force.rollback()

    def initialize_precrack(self, by_name: dict[str, object]) -> None:
        unknown = set(by_name) - set(self._forces)
        if unknown:
            raise KeyError(f"Unknown cohesive interface names: {sorted(unknown)}.")
        for name, facets in by_name.items():
            self._forces[name].initialize_precrack(facets)

    def begin_cycle(self, minimum_by_name, maximum_by_name, *, cycles: int = 1):
        """Begin one replaceable cycle block across all named interfaces."""

        expected = set(self._forces)
        if set(minimum_by_name) != expected or set(maximum_by_name) != expected:
            raise ValueError("Cycle extrema must be supplied for every named interface.")
        responses = {}
        try:
            for name, force in self._forces.items():
                transaction = force.assembler.state
                begin = getattr(transaction, "begin_cycle", None)
                if not callable(begin):
                    raise TypeError(
                        f"Cohesive interface {name!r} does not use a cyclic law."
                    )
                responses[name] = begin(
                    minimum_by_name[name],
                    maximum_by_name[name],
                    cycles=cycles,
                )
        except Exception:
            self.rollback()
            raise
        return responses

    def commit_cycle(self) -> None:
        for name, force in self._forces.items():
            commit = getattr(force.assembler.state, "commit_cycle", None)
            if not callable(commit):
                self.rollback()
                raise TypeError(f"Cohesive interface {name!r} has no cycle transaction.")
        for force in self._forces.values():
            force.assembler.state.commit_cycle()

    def current_response(self) -> NamedCohesiveResponse:
        return NamedCohesiveResponse(
            {name: force.current_response() for name, force in self._forces.items()}
        )

    def cycle_openings(self) -> dict[str, np.ndarray]:
        """Return named quadrature openings for a global cycle controller."""

        return {
            name: force.cycle_opening()
            for name, force in self._forces.items()
        }

    def material_point_responses(self) -> dict[str, object]:
        """Return named local responses without committing trial history."""

        return {
            name: force.material_point_response()
            for name, force in self._forces.items()
        }

    def interface_quantities(self) -> dict[str, dict[str, np.ndarray]]:
        """Return standard fields grouped by durable interface name."""

        return {
            name: force.interface_quantities()
            for name, force in self._forces.items()
        }

    def audit_mode_i(self, **options) -> dict[str, object]:
        return {
            name: force.audit_mode_i(**options)
            for name, force in self._forces.items()
        }

    def stability_inputs(self, mass) -> dict[str, float]:
        candidates = [force.stability_inputs(mass) for force in self._forces.values()]

        def limit(record):
            reduced = (
                record["negative_mass"]
                * record["positive_mass"]
                / (record["negative_mass"] + record["positive_mass"])
            )
            return 2.0 * sqrt(
                reduced
                / (record["interface_stiffness"] * record["interface_area"])
            )

        return min(candidates, key=limit)

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "agentfem.named-cohesive-forces.v1",
            "interfaces": {
                name: force.snapshot() for name, force in self._forces.items()
            },
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("schema") != "agentfem.named-cohesive-forces.v1":
            raise ValueError("Unsupported named cohesive-force schema.")
        records = snapshot.get("interfaces", {})
        if set(records) != set(self._forces):
            raise ValueError("Named cohesive interfaces differ from checkpoint.")
        for name, force in self._forces.items():
            force.restore(records[name])

    def summary(self) -> dict[str, object]:
        return {
            "kind": "named_cohesive_force_collection",
            "interfaces": {
                name: force.summary() for name, force in self._forces.items()
            },
            "state": "independent_per_interface",
            "commit": "atomic_collection_lifecycle",
        }

    @property
    def distributed(self) -> bool:
        return any(
            isinstance(force, DistributedDofMappedCohesiveForce)
            for force in self._forces.values()
        )


def named_cohesive_forces(**interfaces) -> CohesiveForceCollection:
    """Create an atomically managed collection from named cohesive forces."""

    return CohesiveForceCollection(interfaces)


def named_mode_i_cohesive_forces(
    split,
    displacement,
    *,
    laws,
    normal_hints,
    thicknesses=None,
    tangential="free",
    tangential_stiffness=None,
    tolerance: float = 1.0e-10,
) -> CohesiveForceCollection:
    """Build independent named forces on one atomically split solver mesh."""

    if not isinstance(split, interface_api.NamedSplitInterfaceMesh):
        raise TypeError(
            "named_mode_i_cohesive_forces requires NamedSplitInterfaceMesh."
        )
    selected_laws = dict(laws)
    selected_normals = dict(normal_hints)
    selected_thicknesses = (
        {name: 1.0 for name in split.names}
        if thicknesses is None
        else dict(thicknesses)
    )
    selected_tangential = (
        {name: tangential for name in split.names}
        if isinstance(tangential, str)
        else dict(tangential)
    )
    selected_tangential_stiffness = (
        {name: tangential_stiffness for name in split.names}
        if tangential_stiffness is None or np.isscalar(tangential_stiffness)
        else dict(tangential_stiffness)
    )
    expected = set(split.names)
    for label, values in (
        ("laws", selected_laws),
        ("normal_hints", selected_normals),
        ("thicknesses", selected_thicknesses),
        ("tangential", selected_tangential),
        ("tangential_stiffness", selected_tangential_stiffness),
    ):
        if set(values) != expected:
            raise ValueError(
                f"Named cohesive {label} must match interfaces {sorted(expected)}."
            )
    forces = {
        name: mode_i_cohesive_force(
            split[name],
            displacement,
            selected_laws[name],
            normal_hint=selected_normals[name],
            thickness=float(selected_thicknesses[name]),
            tangential=selected_tangential[name],
            tangential_stiffness=selected_tangential_stiffness[name],
            tolerance=tolerance,
        )
        for name in split.names
    }
    return CohesiveForceCollection(forces)


def cohesive_forces(
    split,
    displacement,
    *,
    laws,
    normal_hints,
    thicknesses=None,
    tangential=None,
    tangential_stiffness=None,
    tolerance: float = 1.0e-10,
) -> CohesiveForceCollection:
    """Build a recommended force for every named split interface.

    Each interface independently selects the conservative scalar Mode-I tie or
    the full mixed-mode response from its law. ``tangential`` may be one mode
    for every interface or a name-to-mode mapping when a model deliberately
    combines different interface assumptions.
    """

    if not isinstance(split, interface_api.NamedSplitInterfaceMesh):
        raise TypeError("cohesive_forces requires NamedSplitInterfaceMesh.")
    expected = set(split.names)
    selected_laws = dict(laws)
    selected_normals = dict(normal_hints)
    selected_thicknesses = (
        {name: 1.0 for name in split.names}
        if thicknesses is None
        else dict(thicknesses)
    )
    selected_tangential = (
        {name: None for name in split.names}
        if tangential is None
        else (
            {name: tangential for name in split.names}
            if isinstance(tangential, str)
            else dict(tangential)
        )
    )
    selected_stiffness = (
        {name: tangential_stiffness for name in split.names}
        if tangential_stiffness is None or np.isscalar(tangential_stiffness)
        else dict(tangential_stiffness)
    )
    for label, values in (
        ("laws", selected_laws),
        ("normal_hints", selected_normals),
        ("thicknesses", selected_thicknesses),
        ("tangential", selected_tangential),
        ("tangential_stiffness", selected_stiffness),
    ):
        if set(values) != expected:
            raise ValueError(
                f"Named cohesive {label} must match interfaces {sorted(expected)}."
            )
    return CohesiveForceCollection(
        {
            name: cohesive_force(
                split[name],
                displacement,
                selected_laws[name],
                normal_hint=selected_normals[name],
                tangential=selected_tangential[name],
                tangential_stiffness=selected_stiffness[name],
                thickness=float(selected_thicknesses[name]),
                tolerance=tolerance,
            )
            for name in split.names
        }
    )


class _SparseCohesiveExchange:
    """One-time owner schedule for sparse interface trace and force exchange."""

    def __init__(
        self,
        *,
        comm,
        input_node_to_block_dof,
        input_node_owned,
        required_nodes,
        owned_interface_nodes,
        block_size: int,
    ):
        self.comm = comm
        self.mapping = np.asarray(input_node_to_block_dof, dtype=int)
        self.owned = np.asarray(input_node_owned, dtype=bool)
        self.required_nodes = np.asarray(required_nodes, dtype=int)
        self.owned_interface_nodes = np.asarray(owned_interface_nodes, dtype=int)
        self.block_size = int(block_size)
        number_of_nodes = int(self.mapping.size)
        if self.owned.shape != (number_of_nodes,):
            raise ValueError("Sparse cohesive ownership has an incompatible shape.")
        if np.any(self.required_nodes < 0) or np.any(
            self.required_nodes >= number_of_nodes
        ):
            raise ValueError("Sparse cohesive schedule contains an invalid node id.")

        owner_by_node = np.full(number_of_nodes, -1, dtype=int)
        for rank, nodes in enumerate(
            comm.allgather(np.flatnonzero(self.owned).astype(int).tolist())
        ):
            selected = np.asarray(nodes, dtype=int)
            if np.any(owner_by_node[selected] >= 0):
                raise RuntimeError("A cohesive input node has multiple MPI owners.")
            owner_by_node[selected] = int(rank)
        if np.any(owner_by_node[self.required_nodes] < 0):
            raise RuntimeError("A required cohesive trace node has no MPI owner.")
        self.owner_by_node = owner_by_node

        requested_by_owner = []
        for rank in range(comm.size):
            if rank == comm.rank:
                requested_by_owner.append(np.empty(0, dtype=int))
            else:
                requested_by_owner.append(
                    self.required_nodes[owner_by_node[self.required_nodes] == rank]
                )
        incoming = comm.alltoall(
            [nodes.astype(int).tolist() for nodes in requested_by_owner]
        )
        self.receive_nodes_by_owner = tuple(
            np.asarray(nodes, dtype=int) for nodes in requested_by_owner
        )
        self.send_nodes_by_requester = tuple(
            np.asarray(nodes, dtype=int) for nodes in incoming
        )
        for nodes in self.send_nodes_by_requester:
            if nodes.size and (
                np.any(~self.owned[nodes]) or np.any(self.mapping[nodes] < 0)
            ):
                raise RuntimeError(
                    "Sparse cohesive schedule requested a node not owned locally."
                )
        self.local_required_owned = self.required_nodes[
            owner_by_node[self.required_nodes] == int(comm.rank)
        ]
        position = np.full(number_of_nodes, -1, dtype=int)
        position[self.owned_interface_nodes] = np.arange(
            self.owned_interface_nodes.size, dtype=int
        )
        self.owned_position = position
        required_position = np.full(number_of_nodes, -1, dtype=int)
        required_position[self.required_nodes] = np.arange(
            self.required_nodes.size, dtype=int
        )
        self.required_position = required_position

    @staticmethod
    def _counts(groups, components: int) -> tuple[np.ndarray, np.ndarray]:
        counts = np.asarray(
            [int(nodes.size) * int(components) for nodes in groups],
            dtype=np.int32,
        )
        offsets = np.zeros_like(counts)
        if counts.size > 1:
            offsets[1:] = np.cumsum(counts[:-1], dtype=np.int32)
        return counts, offsets

    def _exchange(
        self,
        send_parts,
        *,
        send_groups,
        receive_groups,
        components: int,
    ) -> np.ndarray:
        selected_parts = [np.asarray(part, dtype=float).reshape(-1) for part in send_parts]
        if len(selected_parts) != len(send_groups):
            raise ValueError("Sparse cohesive send parts do not match the schedule.")
        send_buffer = (
            np.concatenate(selected_parts)
            if any(part.size for part in selected_parts)
            else np.empty(0, dtype=float)
        )
        send_counts, send_offsets = self._counts(send_groups, components)
        if send_buffer.size != int(np.sum(send_counts)):
            raise ValueError("Sparse cohesive send payload has the wrong size.")
        receive_counts, receive_offsets = self._counts(receive_groups, components)
        receive_buffer = np.empty(int(np.sum(receive_counts)), dtype=float)
        self.comm.Alltoallv(
            [send_buffer, send_counts, send_offsets, MPI.DOUBLE],
            [receive_buffer, receive_counts, receive_offsets, MPI.DOUBLE],
        )
        return receive_buffer

    def gather_owned_dof_values(self, dof_values) -> np.ndarray:
        """Return only the global-node rows required by locally owned facets."""

        values = np.asarray(dof_values, dtype=float)
        if values.ndim != 2:
            raise ValueError("Owned DOF values must be a 2D blocked array.")
        result = np.zeros((self.required_nodes.size, values.shape[1]), dtype=float)
        local = self.local_required_owned
        result[self.required_position[local]] = values[self.mapping[local]]
        send_parts = [
            values[self.mapping[nodes]].reshape(-1)
            if nodes.size
            else np.empty(0, dtype=float)
            for nodes in self.send_nodes_by_requester
        ]
        received = self._exchange(
            send_parts,
            send_groups=self.send_nodes_by_requester,
            receive_groups=self.receive_nodes_by_owner,
            components=int(values.shape[1]),
        )
        cursor = 0
        components = int(values.shape[1])
        for nodes in self.receive_nodes_by_owner:
            count = int(nodes.size) * components
            if count:
                result[self.required_position[nodes]] = received[
                    cursor : cursor + count
                ].reshape((-1, components))
            cursor += count
        return result

    def accumulate_to_owners(self, local_values) -> np.ndarray:
        """Sum local facet contributions on the MPI owner of every node."""

        values = np.asarray(local_values, dtype=float)
        if values.ndim != 2 or values.shape[0] != self.required_nodes.size:
            raise ValueError("Local cohesive force has an incompatible node layout.")
        components = int(values.shape[1])
        result = np.zeros(
            (self.owned_interface_nodes.size, components), dtype=float
        )
        local = self.local_required_owned
        np.add.at(
            result,
            self.owned_position[local],
            values[self.required_position[local]],
        )
        send_parts = [
            values[self.required_position[nodes]].reshape(-1)
            if nodes.size
            else np.empty(0, dtype=float)
            for nodes in self.receive_nodes_by_owner
        ]
        received = self._exchange(
            send_parts,
            send_groups=self.receive_nodes_by_owner,
            receive_groups=self.send_nodes_by_requester,
            components=components,
        )
        cursor = 0
        for nodes in self.send_nodes_by_requester:
            count = int(nodes.size) * components
            if count:
                contribution = received[cursor : cursor + count].reshape(
                    (-1, components)
                )
                np.add.at(result, self.owned_position[nodes], contribution)
            cursor += count
        return result

    def summary(self) -> dict[str, int | str]:
        remote = int(sum(nodes.size for nodes in self.receive_nodes_by_owner))
        requesters = int(
            sum(bool(nodes.size) for nodes in self.send_nodes_by_requester)
        )
        owners = int(sum(bool(nodes.size) for nodes in self.receive_nodes_by_owner))
        return {
            "kind": "sparse_cohesive_owner_schedule",
            "rank_required_nodes": int(self.required_nodes.size),
            "rank_remote_trace_nodes": remote,
            "rank_owner_peers": owners,
            "rank_requester_peers": requesters,
            "rank_trace_values_per_exchange": remote * self.block_size,
            "rank_force_values_per_exchange": remote * self.block_size,
        }


class DistributedDofMappedCohesiveForce:
    """MPI assembler for a physical-keyed split interface.

    Split cohesive sides are topologically disconnected, so ordinary DOLFINx
    ghost cells do not necessarily expose both traces on one rank.  This
    adapter exchanges only the input-node traces required by locally owned
    facets, evaluates each physical facet on exactly one deterministic rank,
    and returns sparse force contributions to their node owners before writing
    owned displacement entries.  The owner schedule is built once; time-step
    exchange uses numeric ``MPI_Alltoallv`` payloads rather than dense vectors
    proportional to all volume nodes.
    """

    def __init__(
        self,
        assembler,
        displacement,
        *,
        input_node_to_block_dof,
        input_node_owned,
        global_topology,
        global_facet_indices,
        local_input_nodes=None,
    ):
        supported = (
            interface_api.ModeICohesiveFacetAssembler,
            interface_api.ModeICohesiveSurfaceAssembler,
        )
        if not isinstance(assembler, supported):
            raise TypeError(
                "DistributedDofMappedCohesiveForce requires "
                "a supported Mode-I cohesive assembler."
            )
        function = field_api.unwrap(displacement)
        comm = function.function_space.mesh.comm
        if comm.size < 2:
            raise ValueError("The distributed cohesive adapter requires MPI size > 1.")
        block_size = int(function.function_space.dofmap.index_map_bs)
        mapping = np.asarray(input_node_to_block_dof, dtype=int)
        owned = np.asarray(input_node_owned, dtype=bool)
        local_nodes = (
            np.arange(assembler.number_of_nodes, dtype=int)
            if local_input_nodes is None
            else np.asarray(local_input_nodes, dtype=int)
        )
        if owned.shape != mapping.shape:
            raise ValueError("Distributed input-node maps have an incompatible shape.")
        if local_nodes.shape != (assembler.number_of_nodes,):
            raise ValueError("Local cohesive node layout and assembler differ.")
        if np.any(local_nodes < 0) or np.any(local_nodes >= mapping.size):
            raise ValueError("Local cohesive layout contains an invalid input node.")
        if np.unique(local_nodes).size != local_nodes.size:
            raise ValueError("Local cohesive input nodes must be unique.")
        if np.any(owned & (mapping < 0)):
            raise ValueError("Every owned input node must map to one local block dof.")
        if block_size != assembler.topology.normals.shape[1]:
            raise ValueError("Displacement block size and interface dimension differ.")
        selected = np.asarray(global_facet_indices, dtype=int)
        if selected.shape != (assembler.topology.number_of_facets,):
            raise ValueError("Local facet indices do not match the local assembler.")
        self.assembler = assembler
        self.displacement = function
        self.input_node_to_block_dof = mapping
        self.input_node_owned = owned
        self.local_input_nodes = local_nodes
        self.global_topology = global_topology
        self.global_facet_indices = selected
        self.block_size = block_size
        self.comm = comm
        self.interface_nodes = np.unique(
            np.concatenate(
                (
                    self.global_topology.negative_nodes.reshape(-1),
                    self.global_topology.positive_nodes.reshape(-1),
                )
            )
        ).astype(int)
        self.local_owned_interface_nodes = np.asarray(
            [node for node in self.interface_nodes if self.input_node_owned[node]],
            dtype=int,
        )
        self.local_owned_interface_dofs = self.input_node_to_block_dof[
            self.local_owned_interface_nodes
        ].astype(int)
        self.exchange = _SparseCohesiveExchange(
            comm=self.comm,
            input_node_to_block_dof=self.input_node_to_block_dof,
            input_node_owned=self.input_node_owned,
            required_nodes=self.local_input_nodes,
            owned_interface_nodes=self.local_owned_interface_nodes,
            block_size=self.block_size,
        )
        index_map = function.function_space.dofmap.index_map
        local_global = index_map.local_to_global(
            self.local_owned_interface_dofs.astype(np.int32)
        )
        owned_records = dict(
            zip(
                self.local_owned_interface_nodes.tolist(),
                np.asarray(local_global, dtype=int).tolist(),
                strict=True,
            )
        )
        self.input_node_to_global_block_dof = np.full(mapping.size, -1, dtype=int)
        for records in self.comm.allgather(owned_records):
            for node, global_dof in records.items():
                previous = self.input_node_to_global_block_dof[int(node)]
                if previous >= 0 and previous != int(global_dof):
                    raise RuntimeError(
                        "A cohesive input node has inconsistent global dof identity."
                    )
                self.input_node_to_global_block_dof[int(node)] = int(global_dof)
        if np.any(self.input_node_to_global_block_dof[self.interface_nodes] < 0):
            raise RuntimeError("A cohesive interface node has no global PETSc dof.")

    @property
    def node_to_block_dof(self) -> np.ndarray:
        """Execution-local map; absent input nodes retain ``-1``."""

        return self.input_node_to_block_dof

    def for_displacement(self, displacement):
        selected = field_api.unwrap(displacement)
        if selected.function_space.mesh is not self.displacement.function_space.mesh:
            raise ValueError(
                "A cohesive force can only be rebound within its original mesh."
            )
        return DistributedDofMappedCohesiveForce(
            self.assembler,
            selected,
            input_node_to_block_dof=self.input_node_to_block_dof,
            input_node_owned=self.input_node_owned,
            local_input_nodes=self.local_input_nodes,
            global_topology=self.global_topology,
            global_facet_indices=self.global_facet_indices,
        )

    def _required_displacement(self) -> np.ndarray:
        values = self.displacement.x.array.reshape((-1, self.block_size))
        return self.exchange.gather_owned_dof_values(values)

    def _globalize_energy(self, local) -> interface_api.CohesiveFacetResponse:
        return interface_api.CohesiveFacetResponse(
            internal_force=local.internal_force,
            opening=local.opening,
            traction=local.traction,
            damage=local.damage,
            stored_energy=float(
                self.comm.allreduce(local.stored_energy, op=MPI.SUM)
            ),
            dissipated_energy=float(
                self.comm.allreduce(local.dissipated_energy, op=MPI.SUM)
            ),
            jump=local.jump,
            tangential_jump=local.tangential_jump,
            traction_vector=local.traction_vector,
            tangential_traction=local.tangential_traction,
            mode_mixity=local.mode_mixity,
        )

    @PETSc.Log.EventDecorator(_COHESIVE_PETSC_EVENTS["constitutive"])
    def begin(self):
        local = self.assembler.begin(self._required_displacement())
        return self._globalize_energy(local)

    @PETSc.Log.EventDecorator(_COHESIVE_PETSC_EVENTS["vector"])
    def add_to_vector(self, vector) -> None:
        response = self.begin()
        array = vector.array.reshape((-1, self.block_size))
        array[self.local_owned_interface_dofs] += self.exchange.accumulate_to_owners(
            response.internal_force
        )

    @PETSc.Log.EventDecorator(_COHESIVE_PETSC_EVENTS["matrix"])
    def add_to_matrix(self, matrix) -> None:
        """Add locally owned facets using global PETSc dof identities."""

        elements = self.assembler.tangent_elements(
            self._required_displacement()
        )
        matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        components = np.arange(self.block_size, dtype=PETSc.IntType)
        for compact_nodes, element in zip(
            elements.nodes, elements.matrices, strict=True
        ):
            input_nodes = self.local_input_nodes[compact_nodes]
            blocks = self.input_node_to_global_block_dof[input_nodes]
            scalar_dofs = (
                blocks[:, None] * self.block_size + components[None, :]
            ).reshape(-1).astype(PETSc.IntType)
            matrix.setValues(
                scalar_dofs,
                scalar_dofs,
                element,
                addv=PETSc.InsertMode.ADD_VALUES,
            )

    def commit(self) -> None:
        self.assembler.commit()

    def rollback(self) -> None:
        self.assembler.rollback()

    def initialize_precrack(self, facets) -> None:
        selected = np.asarray(facets)
        if selected.dtype == bool:
            if selected.shape != (self.global_topology.number_of_facets,):
                raise ValueError("Boolean precrack mask has the wrong global shape.")
            global_mask = selected
        else:
            global_mask = np.zeros(self.global_topology.number_of_facets, dtype=bool)
            indices = np.asarray(selected, dtype=int)
            if np.any(indices < 0) or np.any(indices >= global_mask.size):
                raise ValueError("Precrack facet index is out of range.")
            global_mask[indices] = True
        self.assembler.initialize_precrack(global_mask[self.global_facet_indices])

    def current_response(self):
        if self.assembler.last_committed_response is not None:
            return self._globalize_energy(self.assembler.last_committed_response)
        response = self.begin()
        self.rollback()
        return response

    def cycle_opening(self) -> np.ndarray:
        """Evaluate local-owner openings without advancing interface history."""

        response = self.begin()
        try:
            return np.asarray(response.opening, dtype=float).reshape(-1).copy()
        finally:
            self.rollback()

    def material_point_response(self):
        """Evaluate the local-owner constitutive response at active displacement."""

        local = self.assembler.begin(self._required_displacement())
        try:
            return self.assembler.material_point_response(local)
        finally:
            self.assembler.rollback()

    def interface_quantities(self) -> dict[str, np.ndarray]:
        """Return this MPI rank's physical-facet-owner interface fields."""

        response = self.current_response()
        return {
            "JUMP_N": np.asarray(response.opening, dtype=float).copy(),
            "JUMP_T": np.asarray(response.tangential_jump, dtype=float).copy(),
            "TRACTION_N": np.asarray(response.traction, dtype=float).copy(),
            "TRACTION_T": np.asarray(response.tangential_traction, dtype=float).copy(),
            "DAMAGE": np.asarray(response.damage, dtype=float).copy(),
            "MODE_MIXITY": np.asarray(response.mode_mixity, dtype=float).copy(),
        }

    def audit_mode_i(self, **options):
        return interface_api.audit_mode_i_kinematics(
            self.current_response(), **options
        )

    def summary(self) -> dict[str, object]:
        local = int(self.assembler.topology.number_of_facets)
        global_count = int(self.comm.allreduce(local, op=MPI.SUM))
        return {
            "kind": "distributed_dof_mapped_cohesive_force",
            "interface": self.global_topology.summary(),
            "law": self.assembler.law.summary(),
            "interface_kinematics": self.assembler.tangential,
            "tangential_stiffness": self.assembler.tangential_stiffness,
            "parallel_scope": "mpi_sparse_owner_exchange",
            "rank_count": int(self.comm.size),
            "local_owned_facets": local,
            "global_facets": global_count,
            "force_exchange": "physical_input_nodes+sparse_alltoallv",
            "local_assembly_layout": (
                "interface_compact"
                if self.local_input_nodes.size < self.input_node_to_block_dof.size
                else "input_node_complete"
            ),
            "communication": self.exchange.summary(),
            "restart_identity": self.global_topology.identity(),
            "maturity": "experimental_sparse_mpi_consumer",
        }

    def performance_profile(self) -> dict[str, object]:
        """Return deterministic MPI workload evidence for this interface.

        Counts are suitable for regression tests and decomposition studies;
        wall-clock timings remain PETSc profiling data because CI timing is too
        noisy to be a scientific performance threshold. Run with PETSc
        ``-log_view`` to inspect the named constitutive/vector/matrix events.
        """

        communication = self.exchange.summary()
        local = {
            "rank": int(self.comm.rank),
            "owned_facets": int(self.assembler.topology.number_of_facets),
            "required_nodes": int(communication["rank_required_nodes"]),
            "remote_trace_nodes": int(
                communication["rank_remote_trace_nodes"]
            ),
            "trace_values_per_exchange": int(
                communication["rank_trace_values_per_exchange"]
            ),
            "force_values_per_exchange": int(
                communication["rank_force_values_per_exchange"]
            ),
        }
        ranks = tuple(self.comm.allgather(local))
        facet_counts = np.asarray(
            [item["owned_facets"] for item in ranks], dtype=float
        )
        average = float(np.mean(facet_counts)) if facet_counts.size else 0.0
        imbalance = (
            float(np.max(facet_counts) / average) if average > 0.0 else 0.0
        )
        return {
            "schema": "agentfem.cohesive-mpi-profile.v1",
            "rank_count": int(self.comm.size),
            "global_facets": int(np.sum(facet_counts)),
            "global_remote_trace_nodes": int(
                sum(item["remote_trace_nodes"] for item in ranks)
            ),
            "global_trace_values_per_exchange": int(
                sum(item["trace_values_per_exchange"] for item in ranks)
            ),
            "global_force_values_per_exchange": int(
                sum(item["force_values_per_exchange"] for item in ranks)
            ),
            "maximum_facet_imbalance": imbalance,
            "petsc_events": dict(_COHESIVE_PETSC_EVENTS),
            "ranks": ranks,
        }

    def snapshot(self) -> dict[str, object]:
        local_state = _cohesive_state_by_key(
            self.assembler.state,
            self.assembler.topology,
        )
        gathered = self.comm.allgather(
            (
                tuple(self.assembler.topology.facet_keys),
                local_state,
            )
        )
        field_names = tuple(sorted(local_state))
        records = {name: {} for name in field_names}
        for keys, fields in gathered:
            if tuple(sorted(fields)) != field_names:
                raise RuntimeError("Distributed cohesive state fields differ by rank.")
            for name in field_names:
                for key in keys:
                    if key in records[name]:
                        raise RuntimeError(
                            f"Cohesive facet {key} has multiple MPI owners."
                        )
                    records[name][key] = fields[name][key]
        expected = set(self.global_topology.facet_keys)
        if any(set(records[name]) != expected for name in field_names):
            raise RuntimeError("Distributed cohesive snapshot lacks global facets.")
        snapshot = {
            "schema": "agentfem.dof-mapped-cohesive-force.v5",
            "interface_identity": self.global_topology.identity(),
            "law": self.assembler.law.summary(),
            "interface_kinematics": self.assembler.tangential,
            "tangential_stiffness": self.assembler.tangential_stiffness,
            "state_by_field_and_key": records,
            "state_identity": "ordered_physical_facet_and_quadrature",
        }
        if "maximum_opening" in records:
            snapshot["maximum_opening_by_key"] = records["maximum_opening"]
        return snapshot

    def restore(self, snapshot: dict[str, object]) -> None:
        schema = snapshot.get("schema")
        if schema not in {
            "agentfem.dof-mapped-cohesive-force.v3",
            "agentfem.dof-mapped-cohesive-force.v4",
            "agentfem.dof-mapped-cohesive-force.v5",
        }:
            raise ValueError("Unsupported distributed cohesive-state schema.")
        if schema != "agentfem.dof-mapped-cohesive-force.v5" and (
            self.assembler.tangential != "free"
        ):
            raise ValueError(
                "Legacy distributed cohesive checkpoints do not record "
                "interface kinematics and can only restore free-slip Mode-I."
            )
        if snapshot.get("interface_identity") != self.global_topology.identity():
            raise ValueError("Distributed cohesive physical interface differs.")
        if snapshot.get("law") != self.assembler.law.summary():
            raise ValueError("Distributed cohesive law differs.")
        if schema == "agentfem.dof-mapped-cohesive-force.v5" and (
            snapshot.get("interface_kinematics") != self.assembler.tangential
            or snapshot.get("tangential_stiffness")
            != self.assembler.tangential_stiffness
        ):
            raise ValueError("Distributed cohesive interface kinematics differ.")
        if schema == "agentfem.dof-mapped-cohesive-force.v3":
            records = snapshot.get("maximum_opening_by_key", {})
            if set(records) != set(self.global_topology.facet_keys):
                raise ValueError("Distributed cohesive snapshot facet keys differ.")
            local = np.asarray(
                [records[key] for key in self.assembler.topology.facet_keys],
                dtype=float,
            )
            self.assembler.state.initialize(local.reshape(-1))
        else:
            global_records = snapshot.get("state_by_field_and_key", {})
            local_records = {
                name: {
                    key: records[key]
                    for key in self.assembler.topology.facet_keys
                }
                for name, records in global_records.items()
            }
            _restore_cohesive_state_by_key(
                self.assembler.state,
                self.assembler.topology,
                local_records,
            )
        self.assembler.last_committed_response = None

    def save_portable_state(self, path):
        return interface_api.save_portable_cohesive_state(
            path,
            self.assembler.topology,
            self.assembler.state,
            comm=self.comm,
        )

    def load_portable_state(self, path):
        return interface_api.load_portable_cohesive_state(
            path,
            self.assembler.topology,
            self.assembler.state,
            comm=self.comm,
        )

    def stability_inputs(self, mass) -> dict[str, float]:
        diagonal = np.asarray(
            mass.mass if hasattr(mass, "mass") else mass,
            dtype=float,
        ).reshape((-1, self.block_size))
        masses = self.exchange.gather_owned_dof_values(diagonal[:, :1]).reshape(-1)
        if np.any(masses <= 0.0):
            raise RuntimeError("Distributed cohesive stability lacks nodal masses.")
        negative = np.asarray(
            masses[self.assembler.topology.negative_nodes.reshape(-1)]
        )
        positive = np.asarray(
            masses[self.assembler.topology.positive_nodes.reshape(-1)]
        )
        normal_stiffness = float(
            self.assembler.law.initial_stiffness
            if hasattr(self.assembler.law, "initial_stiffness")
            else self.assembler.law.normal_stiffness
        )
        return {
            "interface_stiffness": max(
                normal_stiffness,
                float(self.assembler.tangential_stiffness),
            ),
            "interface_area": float(
                np.min(self.global_topology.measures) * self.assembler.thickness
            ),
            "negative_mass": float(self.comm.allreduce(np.min(negative), op=MPI.MIN)),
            "positive_mass": float(self.comm.allreduce(np.min(positive), op=MPI.MIN)),
        }


def _p1_input_node_layout(displacement, *, number_of_input_nodes: int):
    """Recover local DOFs and ownership for durable input-node identities.

    Coincident cohesive nodes cannot be matched by coordinates.  DOLFINx
    retains their distinct input indices on the geometry map, so this routine
    walks cell-local geometry and field dofmaps together and verifies that
    every input node resolves to exactly one block dof.
    """

    function = field_api.unwrap(displacement)
    space = function.function_space
    domain = space.mesh
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
    owned = np.zeros(requested, dtype=bool)
    cell_map = domain.topology.index_map(domain.topology.dim)
    number_of_cells = int(cell_map.size_local + cell_map.num_ghosts)
    owned_blocks = int(space.dofmap.index_map.size_local)
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
            if int(field_dof) < owned_blocks:
                owned[source_node] = True
    owned_by_rank = domain.comm.allgather(
        np.flatnonzero(owned).astype(int).tolist()
    )
    counts = np.zeros(requested, dtype=int)
    for nodes in owned_by_rank:
        counts[np.asarray(nodes, dtype=int)] += 1
    invalid = np.flatnonzero(counts != 1)
    if invalid.size:
        raise RuntimeError(
            "Every split input node must have exactly one owning MPI rank; "
            f"invalid={invalid.tolist()}, counts={counts[invalid].tolist()}."
        )
    return mapping, owned


def p1_input_node_to_block_dof(displacement, *, number_of_input_nodes: int):
    """Recover the complete serial input-node to block-DOF map.

    Distributed cohesive execution uses the ownership-aware internal layout;
    this compatibility helper remains strict because one NumPy array cannot
    represent remote DOFs.
    """

    function = field_api.unwrap(displacement)
    if function.function_space.mesh.comm.size != 1:
        raise NotImplementedError(
            "Use mode_i_cohesive_force for distributed input-node ownership."
        )
    mapping, _ = _p1_input_node_layout(
        displacement,
        number_of_input_nodes=number_of_input_nodes,
    )
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
    law,
    *,
    normal_hint,
    thickness: float = 1.0,
    tolerance: float = 1.0e-10,
    tangential: str = "free",
    tangential_stiffness: float | None = None,
) -> DofMappedCohesiveForce | DistributedDofMappedCohesiveForce:
    """Build a fixed-path cohesive force from a split mesh contract.

    ``tangential='free'`` preserves the original normal-only interface.
    ``'tie'`` supplies a strict Mode-I penalty constraint that does not
    degrade. ``'degraded'`` transfers shear while the normal-driven interface
    is intact and releases it with normal damage. A
    :class:`MixedModeBilinearCohesiveLaw` requires ``'mixed'``.
    """

    if not isinstance(split, interface_api.SplitInterfaceMesh):
        raise TypeError("mode_i_cohesive_force requires SplitInterfaceMesh.")
    dimension = int(split.coordinates.shape[1])
    if dimension == 2:
        topology = interface_api.pair_coincident_line_facets(
            split.coordinates,
            split.negative_facets,
            split.positive_facets,
            normal_hint=normal_hint,
            tolerance=tolerance,
        )
        assembler_type = interface_api.ModeICohesiveFacetAssembler
    elif dimension == 3:
        if float(thickness) != 1.0:
            raise ValueError(
                "A 3D cohesive surface integrates physical area directly; "
                "thickness must remain 1.0."
            )
        topology = interface_api.pair_coincident_surface_facets(
            split.coordinates,
            split.negative_facets,
            split.positive_facets,
            normal_hint=normal_hint,
            tolerance=tolerance,
        )
        assembler_type = interface_api.ModeICohesiveSurfaceAssembler
    else:
        raise ValueError("Mode-I cohesive force requires a 2D or 3D split mesh.")
    function = field_api.unwrap(displacement)
    comm = function.function_space.mesh.comm
    contracts = comm.allgather(
        {
            "split": split.identity(),
            "interface": topology.identity(),
            "law": law.summary(),
            "thickness": float(thickness),
            "tangential": str(tangential),
            "tangential_stiffness": (
                None if tangential_stiffness is None else float(tangential_stiffness)
            ),
        }
    )
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError(
            "Every MPI rank must declare the same split interface, normal, "
            "cohesive law, and thickness."
        )
    if comm.size == 1:
        assembler_options = {"number_of_nodes": split.coordinates.shape[0]}
        if dimension == 2:
            assembler_options["thickness"] = thickness
        assembler_options["tangential"] = tangential
        assembler_options["tangential_stiffness"] = tangential_stiffness
        assembler = assembler_type(topology, law, **assembler_options)
        mapping = p1_input_node_to_block_dof(
            displacement,
            number_of_input_nodes=split.coordinates.shape[0],
        )
        return DofMappedCohesiveForce(
            assembler,
            displacement,
            node_to_block_dof=mapping,
        )
    mapping, owned = _p1_input_node_layout(
        displacement,
        number_of_input_nodes=split.coordinates.shape[0],
    )
    if topology.number_of_facets < int(comm.size):
        raise ValueError(
            "Distributed cohesive execution currently requires at least one "
            "physical interface facet per MPI rank."
        )
    owner_by_key = {
        key: index % int(comm.size)
        for index, key in enumerate(sorted(topology.facet_keys))
    }
    owners = np.asarray(
        [owner_by_key[key] for key in topology.facet_keys], dtype=int
    )
    selected = np.flatnonzero(owners == int(comm.rank))
    global_negative = topology.negative_nodes[selected]
    global_positive = topology.positive_nodes[selected]
    local_input_nodes = np.unique(
        np.concatenate((global_negative.reshape(-1), global_positive.reshape(-1)))
    ).astype(int)
    global_to_local = np.full(split.coordinates.shape[0], -1, dtype=int)
    global_to_local[local_input_nodes] = np.arange(local_input_nodes.size, dtype=int)
    topology_options = {
        "negative_nodes": global_to_local[global_negative],
        "positive_nodes": global_to_local[global_positive],
        "normals": topology.normals[selected],
        "tolerance": topology.tolerance,
        "facet_keys": tuple(topology.facet_keys[index] for index in selected),
    }
    if dimension == 2:
        topology_options["lengths"] = topology.lengths[selected]
        local_topology = interface_api.PairedLineFacets(**topology_options)
        assembler = assembler_type(
            local_topology,
            law,
            number_of_nodes=local_input_nodes.size,
            thickness=thickness,
            tangential=tangential,
            tangential_stiffness=tangential_stiffness,
        )
    else:
        topology_options["areas"] = topology.areas[selected]
        local_topology = interface_api.PairedSurfaceFacets(**topology_options)
        assembler = assembler_type(
            local_topology,
            law,
            number_of_nodes=local_input_nodes.size,
            tangential=tangential,
            tangential_stiffness=tangential_stiffness,
        )
    return DistributedDofMappedCohesiveForce(
        assembler,
        displacement,
        input_node_to_block_dof=mapping,
        input_node_owned=owned,
        local_input_nodes=local_input_nodes,
        global_topology=topology,
        global_facet_indices=selected,
    )


def cohesive_force(
    split: interface_api.SplitInterfaceMesh,
    displacement,
    law,
    *,
    normal_hint,
    tangential: str | None = None,
    tangential_stiffness: float | None = None,
    thickness: float = 1.0,
    tolerance: float = 1.0e-10,
):
    """Build the recommended full-vector fixed-path interface consumer.

    A scalar Mode-I law defaults to a nondegrading tangential penalty tie:
    that law declares no shear strength or Mode-II fracture energy from which
    AgentFEM could infer tangential damage.  A mixed-mode law selects its full
    vector damage response.  Damage-compatible scalar shear transfer remains
    available through the explicit ``tangential='degraded'`` option.
    """

    selected = tangential
    if selected is None:
        selected = (
            "mixed"
            if isinstance(law, interface_api.MixedModeBilinearCohesiveLaw)
            else "tie"
        )
    return mode_i_cohesive_force(
        split,
        displacement,
        law,
        normal_hint=normal_hint,
        thickness=thickness,
        tolerance=tolerance,
        tangential=selected,
        tangential_stiffness=tangential_stiffness,
    )


class FiniteStrainCohesiveResidual:
    """Assemble bulk UFL and paired-facet interface forces into one residual."""

    def __init__(
        self,
        bulk,
        cohesive,
    ):
        self.bulk = bulk
        self.cohesive = cohesive

    @property
    def displacement(self):
        return self.cohesive.displacement

    def assemble_vector(self, *, bcs=None):
        ledger = getattr(self, "performance", None)
        bulk_started = perf_counter()
        vector = operators.assemble_vector(self.bulk)
        if ledger is not None:
            ledger.add("bulk_residual_assembly", perf_counter() - bulk_started)
        try:
            cohesive_started = perf_counter()
            self.cohesive.add_to_vector(vector)
            if ledger is not None:
                ledger.add(
                    "cohesive_force_assembly",
                    perf_counter() - cohesive_started,
                )
        except Exception:
            vector.destroy()
            self.cohesive.rollback()
            raise
        if bcs:
            _zero_owned_bc_entries(vector, bcs)
        return vector

    def assemble_matrix(self, tangent, *, bcs=None):
        """Assemble the bulk and interface algorithmic tangent together.

        Strong constraints are eliminated only after both contributions have
        entered the matrix.  This ordering is essential because duplicated
        cohesive traces introduce couplings absent from the bulk sparsity
        graph.
        """

        ledger = getattr(self, "performance", None)
        started = perf_counter()
        matrix = operators.assemble_matrix(tangent)
        try:
            self.cohesive.add_to_matrix(matrix)
            matrix.assemble()
            if bcs:
                constrained = _owned_bc_global_dofs(self.displacement, bcs)
                matrix.zeroRowsColumns(constrained, diag=1.0)
                matrix.assemble()
            if ledger is not None:
                ledger.add("bulk_cohesive_tangent_assembly", perf_counter() - started)
            return matrix
        except Exception:
            matrix.destroy()
            raise

    def reaction_field(self, *, name: str = "RF"):
        """Return the unconstrained bulk-plus-interface algebraic residual.

        At a converged equilibrium its free entries vanish and strong-
        constraint entries are nodal reactions.  The definition intentionally
        excludes MPC, weak, contact and multiplier reactions.
        """

        vector = self.assemble_vector()
        reaction = fem.Function(self.displacement.function_space, name=name)
        try:
            values = vector.array_r
            reaction.x.array[: len(values)] = values
            reaction.x.scatter_forward()
        finally:
            vector.destroy()
            self.rollback()
        return reaction

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
        parallel = isinstance(
            self.cohesive, DistributedDofMappedCohesiveForce
        ) or bool(getattr(self.cohesive, "distributed", False))
        bulk_family = getattr(self.bulk, "family", "finite_strain_bulk")
        return {
            "name": "R_bulk_plus_cohesive",
            "kind": "finite_strain_cohesive_residual",
            "role": "residual",
            "family": f"{bulk_family}+cohesive_interface",
            "parts": (
                self.bulk.summary() if hasattr(self.bulk, "summary") else repr(self.bulk),
                self.cohesive.summary(),
            ),
            "maturity": (
                "experimental_mpi_sparse_consumer"
                if parallel
                else "experimental_serial_global_consumer"
            ),
        }


def _owned_bc_local_dofs(bcs) -> np.ndarray:
    records = []
    for bc in bcs:
        dofs, first_ghost = bc.dof_indices()
        records.append(np.asarray(dofs[:first_ghost], dtype=np.int32))
    if not records:
        return np.empty(0, dtype=np.int32)
    return np.unique(np.concatenate(records)).astype(np.int32)


def _zero_owned_bc_entries(vector, bcs) -> None:
    selected = _owned_bc_local_dofs(bcs)
    if selected.size:
        with vector.localForm() as local:
            local.array_w[selected] = 0.0


def _owned_bc_global_dofs(displacement, bcs) -> np.ndarray:
    local_scalar = _owned_bc_local_dofs(bcs)
    if local_scalar.size == 0:
        return np.empty(0, dtype=PETSc.IntType)
    space = displacement.function_space
    block_size = int(space.dofmap.index_map_bs)
    local_blocks = (local_scalar // block_size).astype(np.int32)
    components = local_scalar % block_size
    global_blocks = space.dofmap.index_map.local_to_global(local_blocks)
    return (
        np.asarray(global_blocks, dtype=PETSc.IntType) * block_size
        + components.astype(PETSc.IntType)
    )


@dataclass(frozen=True)
class CohesiveNewtonSolveInfo:
    """Convergence evidence for one native bulk-plus-interface equilibrium."""

    converged: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    accepted_step_lengths: tuple[float, ...]
    linear_converged_reasons: tuple[int, ...]
    message: str = ""

    def summary(self) -> dict[str, object]:
        return {
            "kind": "finite_strain_cohesive_newton_info",
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_residual_norm": self.initial_residual_norm,
            "residual_norm": self.residual_norm,
            "accepted_step_lengths": self.accepted_step_lengths,
            "linear_converged_reasons": self.linear_converged_reasons,
            "message": self.message,
        }


@dataclass(frozen=True)
class ArcLengthOptions:
    """Crisfield-style spherical continuation controls.

    ``load_scale`` maps one physical load unit into displacement-norm units;
    making it explicit prevents a numerically convenient but dimensionally
    hidden arc-length equation.
    """

    radius: float
    load_scale: float
    maximum_iterations: int = 30
    relative_tolerance: float = 1.0e-7
    absolute_tolerance: float = 1.0e-9
    constraint_tolerance: float = 1.0e-8
    load_derivative_step: float = 1.0e-6

    def __post_init__(self) -> None:
        for name in (
            "radius",
            "load_scale",
            "relative_tolerance",
            "absolute_tolerance",
            "constraint_tolerance",
            "load_derivative_step",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"Arc-length {name} must be finite and positive.")
        if int(self.maximum_iterations) < 1:
            raise ValueError("Arc-length maximum_iterations must be positive.")

    def summary(self) -> dict[str, object]:
        return {
            "kind": "spherical_arc_length_options",
            "radius": self.radius,
            "load_scale": self.load_scale,
            "maximum_iterations": self.maximum_iterations,
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "constraint_tolerance": self.constraint_tolerance,
            "load_derivative_step": self.load_derivative_step,
        }


@dataclass(frozen=True)
class ArcLengthSolveInfo:
    converged: bool
    iterations: int
    load: float
    load_increment: float
    residual_norm: float
    constraint_residual: float
    linear_converged_reasons: tuple[int, ...]
    message: str = ""

    def summary(self) -> dict[str, object]:
        return {
            "kind": "cohesive_arc_length_solve_info",
            "converged": self.converged,
            "iterations": self.iterations,
            "load": self.load,
            "load_increment": self.load_increment,
            "residual_norm": self.residual_norm,
            "constraint_residual": self.constraint_residual,
            "linear_converged_reasons": self.linear_converged_reasons,
            "message": self.message,
        }


class FiniteStrainCohesiveEquilibrium:
    """Native Newton consumer for UFL bulk and zero-thickness interfaces.

    A scalar physical load is supplied through ``set_load``.  The class is
    directly callable with the ``GlobalCyclicFatigueStep`` equilibrium
    protocol, but it remains useful for monotonic quasi-static paths.
    Cohesive history is never committed inside Newton; the owning step keeps
    begin/commit/rollback authority.
    """

    def __init__(
        self,
        residual: FiniteStrainCohesiveResidual,
        tangent,
        displacement,
        *,
        set_load=None,
        load_parameter=None,
        reference_load: float = 1.0,
        bcs=(),
        solver_options=None,
        control_displacement=None,
        reaction=None,
        bulk_strain_energy=None,
    ):
        if not isinstance(residual, FiniteStrainCohesiveResidual):
            raise TypeError("Native cohesive equilibrium requires its residual type.")
        if (set_load is None) == (load_parameter is None):
            raise ValueError(
                "Provide exactly one of set_load or load_parameter."
            )
        selected_reference = float(reference_load)
        if not isfinite(selected_reference) or selected_reference == 0.0:
            raise ValueError("reference_load must be finite and nonzero.")
        if set_load is None:
            if not hasattr(load_parameter, "value"):
                raise TypeError("load_parameter must expose a writable value.")

            def set_load(value):
                load_parameter.value = float(value) / selected_reference

        elif not callable(set_load):
            raise TypeError("set_load must accept one physical scalar load.")
        for label, callback in (
            ("control_displacement", control_displacement),
            ("reaction", reaction),
            ("bulk_strain_energy", bulk_strain_energy),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{label} must be callable when supplied.")
        from . import solvers

        self.residual = residual
        self.tangent = tangent
        self.displacement = field_api.unwrap(displacement)
        if self.displacement is not residual.displacement:
            raise ValueError("Residual and equilibrium must share one displacement field.")
        self.set_load = set_load
        self.reference_load = selected_reference
        self.bcs = tuple(bcs)
        selected_options = solver_options or solvers.newton()
        if not isinstance(selected_options, solvers.NewtonSolverOptions):
            raise TypeError(
                "FiniteStrainCohesiveEquilibrium requires NewtonSolverOptions."
            )
        self.solver_options = selected_options
        self.control_displacement = control_displacement
        self.reaction = reaction
        self.bulk_strain_energy = bulk_strain_energy
        self.last_info: CohesiveNewtonSolveInfo | None = None

    def _residual_vector(self):
        return self.residual.assemble_vector(bcs=self.bcs)

    def solve(self, load: float) -> CohesiveNewtonSolveInfo:
        from dolfinx.fem import petsc as fem_petsc
        from . import solvers

        selected_load = float(load)
        if not isfinite(selected_load):
            raise ValueError("Cohesive equilibrium load must be finite.")
        self.set_load(selected_load)
        function = self.displacement
        rollback = function.x.array.copy()
        fem_petsc.set_bc(function.x.petsc_vec, list(self.bcs))
        function.x.scatter_forward()
        options = self.solver_options
        accepted_steps = []
        linear_reasons = []
        residual_vector = self._residual_vector()
        initial_norm = float(residual_vector.norm())
        current_norm = initial_norm
        residual_vector.destroy()
        threshold = options.absolute_tolerance + options.relative_tolerance * initial_norm
        converged = np.isfinite(current_norm) and current_norm <= threshold
        iteration = 0
        message = "converged at initial state" if converged else ""
        try:
            while not converged and iteration < options.maximum_iterations:
                iteration += 1
                residual_vector = self._residual_vector()
                matrix = self.residual.assemble_matrix(
                    self.tangent, bcs=self.bcs
                )
                right_hand_side = residual_vector.copy()
                right_hand_side.scale(-1.0)
                correction = function.x.petsc_vec.duplicate()
                correction.set(0.0)
                linear = solvers.solve_matrix_system(
                    matrix,
                    right_hand_side,
                    correction,
                    options.linear_solver,
                    raise_on_failure=False,
                )
                linear_reasons.append(linear.converged_reason)
                residual_vector.destroy()
                right_hand_side.destroy()
                matrix.destroy()
                if not linear.converged:
                    correction.destroy()
                    message = (
                        "linear correction failed with PETSc reason "
                        f"{linear.converged_reason}"
                    )
                    break
                correction.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT,
                    mode=PETSc.ScatterMode.FORWARD,
                )
                base = function.x.array.copy()
                direction = correction.array.copy()
                correction.destroy()
                alpha = 1.0
                accepted = False
                minimum = (
                    options.minimum_step_length
                    if options.line_search == "backtracking"
                    else 1.0
                )
                while alpha + 1.0e-15 >= minimum:
                    function.x.array[:] = base + alpha * direction
                    fem_petsc.set_bc(function.x.petsc_vec, list(self.bcs))
                    function.x.scatter_forward()
                    trial = self._residual_vector()
                    trial_norm = float(trial.norm())
                    trial.destroy()
                    decreases = (
                        trial_norm < current_norm
                        or trial_norm <= current_norm * (1.0 - 1.0e-4 * alpha)
                    )
                    if np.isfinite(trial_norm) and (
                        options.line_search != "backtracking" or decreases
                    ):
                        current_norm = trial_norm
                        accepted_steps.append(alpha)
                        accepted = True
                        break
                    self.residual.rollback()
                    alpha *= options.line_search_reduction
                if not accepted:
                    function.x.array[:] = base
                    function.x.scatter_forward()
                    message = "line search did not reduce the residual"
                    break
                converged = current_norm <= threshold
                if converged:
                    message = "converged"
            if not converged and not message:
                message = "maximum Newton iterations reached"
            info = CohesiveNewtonSolveInfo(
                converged=converged,
                iterations=iteration,
                initial_residual_norm=initial_norm,
                residual_norm=current_norm,
                accepted_step_lengths=tuple(accepted_steps),
                linear_converged_reasons=tuple(linear_reasons),
                message=message,
            )
            self.last_info = info
            if not converged:
                function.x.array[:] = rollback
                function.x.scatter_forward()
                self.residual.rollback()
                if options.error_if_not_converged:
                    raise RuntimeError(
                        "Finite-strain cohesive Newton solve failed: " + message
                    )
            return info
        except Exception:
            function.x.array[:] = rollback
            function.x.scatter_forward()
            self.residual.rollback()
            raise

    def __call__(self, *, load: float, branch: str, cycle: int):
        info = self.solve(load)
        evidence = {
            "branch": branch,
            "load": float(load),
            "cycle": int(cycle),
            "converged": info.converged,
            "iterations": info.iterations,
            "metadata": {"newton": info.summary()},
        }
        if self.control_displacement is not None:
            evidence["control_displacement"] = float(
                self.control_displacement(self.displacement)
            )
        if self.reaction is not None:
            evidence["reaction"] = float(self.reaction(self.displacement))
        if self.bulk_strain_energy is not None:
            evidence["bulk_strain_energy"] = float(
                self.bulk_strain_energy(self.displacement)
            )
        return evidence

    def summary(self) -> dict[str, object]:
        return {
            "kind": "finite_strain_cohesive_equilibrium",
            "procedure": "quasi_static_newton",
            "residual": self.residual.summary(),
            "solver": self.solver_options.summary(),
            "load_control": "physical_scalar_callback",
            "reference_load": self.reference_load,
            "cohesive_state_commit": "owned_by_step_lifecycle",
            "maturity": "experimental_native_global_consumer",
        }


class FiniteStrainCohesiveArcLength:
    """Spherical arc-length continuation for cohesive equilibrium paths.

    This consumer reuses the exact bulk-plus-interface residual, tangent,
    boundary elimination and cohesive transaction.  It adds one physical load
    unknown rather than introducing a second cohesive implementation.  A
    converged trial is accepted only by :meth:`commit`; :meth:`advance` is the
    convenient solve-and-commit route.
    """

    def __init__(
        self,
        equilibrium: FiniteStrainCohesiveEquilibrium,
        options: ArcLengthOptions,
        *,
        initial_load: float = 0.0,
    ):
        if not isinstance(equilibrium, FiniteStrainCohesiveEquilibrium):
            raise TypeError("Arc length requires a cohesive equilibrium consumer.")
        if not isinstance(options, ArcLengthOptions):
            raise TypeError("options must be ArcLengthOptions.")
        self.equilibrium = equilibrium
        self.options = options
        self.load = float(initial_load)
        if not isfinite(self.load):
            raise ValueError("initial_load must be finite.")
        self._accepted_displacement = equilibrium.displacement.x.array.copy()
        self._accepted_load = self.load
        self._trial_info: ArcLengthSolveInfo | None = None
        self._previous_increment: tuple[np.ndarray, float] | None = None
        self.history: list[dict[str, object]] = []
        equilibrium.set_load(self.load)

    @property
    def displacement(self):
        return self.equilibrium.displacement

    def _owned_dot(self, left, right) -> float:
        space = self.displacement.function_space
        owned = int(space.dofmap.index_map.size_local * space.dofmap.index_map_bs)
        local = float(np.dot(np.asarray(left)[:owned], np.asarray(right)[:owned]))
        return float(space.mesh.comm.allreduce(local, op=MPI.SUM))

    def _load_derivative(self, load: float):
        step = self.options.load_derivative_step * max(1.0, abs(load))
        plus = None
        minus = None
        try:
            self.equilibrium.set_load(load + step)
            plus = self.equilibrium._residual_vector()
            self.equilibrium.residual.rollback()
            self.equilibrium.set_load(load - step)
            minus = self.equilibrium._residual_vector()
            self.equilibrium.residual.rollback()
            plus.axpy(-1.0, minus)
            plus.scale(0.5 / step)
            derivative = plus
            plus = None
            return derivative
        finally:
            if plus is not None:
                plus.destroy()
            if minus is not None:
                minus.destroy()
            self.equilibrium.residual.rollback()
            self.equilibrium.set_load(load)

    def _linear_solve(self, matrix, right_hand_side):
        from . import solvers

        solution = self.displacement.x.petsc_vec.duplicate()
        solution.set(0.0)
        info = solvers.solve_matrix_system(
            matrix,
            right_hand_side,
            solution,
            self.equilibrium.solver_options.linear_solver,
            raise_on_failure=False,
        )
        if not info.converged:
            solution.destroy()
            raise RuntimeError(
                "Arc-length linear solve failed with PETSc reason "
                f"{info.converged_reason}."
            )
        solution.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD,
        )
        return solution, int(info.converged_reason)

    def begin_step(self, *, direction: float = 1.0) -> ArcLengthSolveInfo:
        """Solve one replaceable continuation increment."""

        if self._trial_info is not None:
            raise RuntimeError("Commit or rollback the active arc-length trial.")
        from dolfinx.fem import petsc as fem_petsc

        sign = float(np.sign(direction))
        if sign == 0.0:
            raise ValueError("Arc-length direction cannot be zero.")
        function = self.displacement
        function.x.array[:] = self._accepted_displacement
        fem_petsc.set_bc(function.x.petsc_vec, list(self.equilibrium.bcs))
        function.x.scatter_forward()
        self.equilibrium.set_load(self._accepted_load)
        matrix = self.equilibrium.residual.assemble_matrix(
            self.equilibrium.tangent, bcs=self.equilibrium.bcs
        )
        derivative = self._load_derivative(self._accepted_load)
        derivative.scale(-1.0)
        predictor, reason = self._linear_solve(matrix, derivative)
        matrix.destroy()
        derivative.destroy()
        predictor_array = predictor.array.copy()
        predictor.destroy()
        if self._previous_increment is not None:
            previous_u, previous_load = self._previous_increment
            orientation = self._owned_dot(predictor_array, previous_u) + (
                self.options.load_scale**2 * previous_load
            )
            if orientation < 0.0:
                sign *= -1.0
        norm = sqrt(
            self._owned_dot(predictor_array, predictor_array)
            + self.options.load_scale**2
        )
        load_increment = sign * self.options.radius / norm
        function.x.array[:] = (
            self._accepted_displacement + load_increment * predictor_array
        )
        fem_petsc.set_bc(function.x.petsc_vec, list(self.equilibrium.bcs))
        function.x.scatter_forward()
        selected_load = self._accepted_load + load_increment
        self.equilibrium.set_load(selected_load)

        linear_reasons = [reason]
        initial_norm = None
        residual_norm = float("inf")
        constraint = float("inf")
        converged = False
        message = ""
        try:
            for iteration in range(1, self.options.maximum_iterations + 1):
                residual = self.equilibrium._residual_vector()
                residual_norm = float(residual.norm())
                if initial_norm is None:
                    initial_norm = residual_norm
                increment_u = function.x.array - self._accepted_displacement
                increment_load = selected_load - self._accepted_load
                constraint = (
                    self._owned_dot(increment_u, increment_u)
                    + (self.options.load_scale * increment_load) ** 2
                    - self.options.radius**2
                )
                residual_limit = (
                    self.options.absolute_tolerance
                    + self.options.relative_tolerance * initial_norm
                )
                constraint_limit = (
                    self.options.constraint_tolerance * self.options.radius**2
                )
                if residual_norm <= residual_limit and abs(constraint) <= constraint_limit:
                    residual.destroy()
                    converged = True
                    message = "converged"
                    break
                matrix = self.equilibrium.residual.assemble_matrix(
                    self.equilibrium.tangent, bcs=self.equilibrium.bcs
                )
                residual.scale(-1.0)
                correction, correction_reason = self._linear_solve(matrix, residual)
                residual.destroy()
                load_derivative = self._load_derivative(selected_load)
                load_derivative.scale(-1.0)
                load_direction, load_reason = self._linear_solve(
                    matrix, load_derivative
                )
                load_derivative.destroy()
                matrix.destroy()
                linear_reasons.extend((correction_reason, load_reason))
                correction_array = correction.array.copy()
                direction_array = load_direction.array.copy()
                correction.destroy()
                load_direction.destroy()
                denominator = (
                    self._owned_dot(increment_u, direction_array)
                    + self.options.load_scale**2 * increment_load
                )
                if abs(denominator) <= np.finfo(float).eps:
                    message = "arc-length corrector denominator vanished"
                    break
                correction_load = (
                    -0.5 * constraint
                    - self._owned_dot(increment_u, correction_array)
                ) / denominator
                function.x.array[:] += correction_array + correction_load * direction_array
                selected_load += correction_load
                fem_petsc.set_bc(function.x.petsc_vec, list(self.equilibrium.bcs))
                function.x.scatter_forward()
                self.equilibrium.set_load(selected_load)
            else:
                iteration = self.options.maximum_iterations
                message = "maximum arc-length iterations reached"
            info = ArcLengthSolveInfo(
                converged=converged,
                iterations=iteration,
                load=float(selected_load),
                load_increment=float(selected_load - self._accepted_load),
                residual_norm=residual_norm,
                constraint_residual=float(constraint),
                linear_converged_reasons=tuple(linear_reasons),
                message=message,
            )
            if not converged:
                function.x.array[:] = self._accepted_displacement
                function.x.scatter_forward()
                self.equilibrium.set_load(self._accepted_load)
                self.equilibrium.residual.rollback()
                raise RuntimeError("Cohesive arc-length step failed: " + message)
            self.load = float(selected_load)
            self._trial_info = info
            return info
        except Exception:
            if self._trial_info is None:
                function.x.array[:] = self._accepted_displacement
                function.x.scatter_forward()
                self.load = self._accepted_load
                self.equilibrium.set_load(self.load)
                self.equilibrium.residual.rollback()
            raise

    def commit(self) -> None:
        if self._trial_info is None:
            raise RuntimeError("No converged arc-length trial is available.")
        increment = self.displacement.x.array - self._accepted_displacement
        load_increment = self.load - self._accepted_load
        self.equilibrium.residual.commit()
        self._previous_increment = (increment.copy(), float(load_increment))
        self._accepted_displacement = self.displacement.x.array.copy()
        self._accepted_load = float(self.load)
        self.history.append(self._trial_info.summary())
        self._trial_info = None

    def rollback(self) -> None:
        self.equilibrium.residual.rollback()
        self.displacement.x.array[:] = self._accepted_displacement
        self.displacement.x.scatter_forward()
        self.load = self._accepted_load
        self.equilibrium.set_load(self.load)
        self._trial_info = None

    def advance(self, *, direction: float = 1.0) -> ArcLengthSolveInfo:
        info = self.begin_step(direction=direction)
        self.commit()
        return info

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "agentfem.cohesive-arc-length.v1",
            "load": self._accepted_load,
            "displacement": self._accepted_displacement.tolist(),
            "previous_increment": (
                None
                if self._previous_increment is None
                else {
                    "displacement": self._previous_increment[0].tolist(),
                    "load": self._previous_increment[1],
                }
            ),
            "cohesive": self.equilibrium.residual.snapshot(),
            "options": self.options.summary(),
            "history": list(self.history),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("schema") != "agentfem.cohesive-arc-length.v1":
            raise ValueError("Unsupported cohesive arc-length checkpoint schema.")
        if snapshot.get("options") != self.options.summary():
            raise ValueError("Arc-length checkpoint options differ.")
        displacement = np.asarray(snapshot.get("displacement"), dtype=float)
        if displacement.shape != self.displacement.x.array.shape or not np.all(
            np.isfinite(displacement)
        ):
            raise ValueError("Arc-length checkpoint displacement layout differs.")
        load = float(snapshot["load"])
        if not isfinite(load):
            raise ValueError("Arc-length checkpoint load must be finite.")
        previous = snapshot.get("previous_increment")
        if previous is None:
            restored_previous = None
        else:
            previous_displacement = np.asarray(
                previous["displacement"], dtype=float
            )
            previous_load = float(previous["load"])
            if (
                previous_displacement.shape != displacement.shape
                or not np.all(np.isfinite(previous_displacement))
                or not isfinite(previous_load)
            ):
                raise ValueError(
                    "Arc-length checkpoint previous increment is invalid."
                )
            restored_previous = (
                previous_displacement,
                previous_load,
            )
        history = snapshot.get("history", ())
        if not isinstance(history, (list, tuple)) or not all(
            isinstance(record, dict) for record in history
        ):
            raise ValueError("Arc-length checkpoint history is invalid.")
        self.equilibrium.residual.restore(snapshot["cohesive"])
        self._accepted_displacement = displacement.copy()
        self.displacement.x.array[:] = displacement
        self.displacement.x.scatter_forward()
        self._accepted_load = self.load = load
        self.equilibrium.set_load(self.load)
        self._previous_increment = restored_previous
        self.history = [dict(record) for record in history]
        self._trial_info = None

    def summary(self) -> dict[str, object]:
        return {
            "kind": "finite_strain_cohesive_arc_length",
            "procedure": "spherical_arc_length",
            "options": self.options.summary(),
            "equilibrium": self.equilibrium.summary(),
            "commit": "explicit_trial_or_advance",
            "maturity": "experimental_post_peak_consumer",
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
    cohesive: object
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
    accepts_accepted_residual: bool = True

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

    def _sample(self, displacement, acceleration, *, residual_owned=None):
        function = field_api.unwrap(displacement)
        acceleration_function = field_api.unwrap(acceleration)
        owned = self._owned_size(displacement)
        current = np.asarray(function.x.array[:owned], dtype=float).copy()
        natural = self._assemble_owned(self.natural_force, displacement)
        prescribed_force = np.zeros(owned, dtype=float)
        constrained = self._prescribed_dofs(displacement)
        if constrained.size:
            if residual_owned is None:
                try:
                    residual = self._assemble_owned(self.residual, displacement)
                finally:
                    if hasattr(self.residual, "rollback"):
                        self.residual.rollback()
            else:
                residual = np.asarray(residual_owned, dtype=float)
                if residual.shape != (owned,):
                    raise ValueError(
                        "Accepted residual cache does not match the owned "
                        "displacement layout."
                    )
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

    def advance(
        self,
        *,
        displacement,
        velocity=None,
        residual_owned=None,
    ) -> dict[str, float]:
        """Advance accepted-increment work without assembling an energy snapshot.

        External work is path dependent and therefore consumes every accepted
        increment.  Bulk strain and kinetic energies are state functions and
        only need evaluation when a history frame is retained.  Keeping these
        two responsibilities separate preserves cadence-independent work while
        avoiding repeated finite-element energy assembly between saved frames.
        """

        current, natural, prescribed_force = self._sample(
            displacement,
            self.state.a,
            residual_owned=residual_owned,
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
        external = self._natural_work + self._prescribed_work
        return {
            "natural_load_work": self._natural_work,
            "prescribed_motion_work": self._prescribed_work,
            "external_work": external,
        }

    def evaluate(
        self,
        *,
        displacement,
        velocity,
        residual_owned=None,
    ) -> dict[str, float]:
        values = self.energy.evaluate(
            displacement=displacement,
            velocity=velocity,
        )
        work = self.advance(
            displacement=displacement,
            velocity=velocity,
            residual_owned=residual_owned,
        )
        accounted = self._accounted(values)
        if self._initial_accounted_energy is None:
            self._initial_accounted_energy = accounted
        external = work["external_work"]
        balance = self._initial_accounted_energy + external - accounted
        scale = max(
            abs(self._initial_accounted_energy),
            abs(external),
            abs(accounted),
            np.finfo(float).eps,
        )
        values.update(work)
        values.update(
            {
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


@dataclass(frozen=True)
class PrincipalSurfaceWaveSpeed:
    """Reference-coordinate principal surface-wave secular solution."""

    speed: float
    limiting_bulk_speed: float
    speed_ratio: float
    attenuation_roots: np.ndarray
    secular_residual: float
    propagation_axis: int
    depth_axis: int
    configuration: str = "prestrained_reference"

    def summary(self) -> dict[str, object]:
        return {
            "kind": "principal_small_on_large_surface_wave_speed",
            "speed": self.speed,
            "limiting_bulk_speed": self.limiting_bulk_speed,
            "speed_ratio": self.speed_ratio,
            "attenuation_roots": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in self.attenuation_roots
            ],
            "secular_residual": self.secular_residual,
            "propagation_axis": self.propagation_axis,
            "depth_axis": self.depth_axis,
            "configuration": self.configuration,
            "method": "reference_total_lagrangian_stroh_decay",
        }


def neo_hookean_material_tangent(
    deformation_gradient,
    material,
) -> np.ndarray:
    """Return ``A[i,J,k,L] = dP[i,J]/dF[k,L]`` for a supported energy."""

    if not hyperelasticity.is_finite_strain_hyperelastic(material):
        raise TypeError("material tangent requires a supported hyperelastic material.")
    F = np.asarray(deformation_gradient, dtype=float)
    if F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape[0] not in {2, 3}:
        raise ValueError("deformation_gradient must be one finite 2x2 or 3x3 array.")
    if not np.all(np.isfinite(F)):
        raise ValueError("deformation_gradient must contain only finite values.")
    J = float(np.linalg.det(F))
    if J <= 0.0:
        raise ValueError("Incremental material tangent requires det(F) > 0.")
    if isinstance(material, hyperelasticity.MooneyRivlinProperties):
        base_scale = max(1.0, float(np.linalg.norm(F)))
        step = np.cbrt(np.finfo(float).eps) * base_scale
        tangent = np.empty(F.shape + F.shape, dtype=float)
        for k in range(F.shape[0]):
            for L in range(F.shape[1]):
                perturbation = np.zeros_like(F)
                perturbation[k, L] = step
                tangent[:, :, k, L] = (
                    hyperelasticity.mooney_rivlin_first_piola_value(
                        F + perturbation, material
                    )
                    - hyperelasticity.mooney_rivlin_first_piola_value(
                        F - perturbation, material
                    )
                ) / (2.0 * step)
        major = np.transpose(tangent, (2, 3, 0, 1))
        return 0.5 * (tangent + major)
    if isinstance(
        material,
        hyperelasticity.PlaneStressNeoHookeanProperties,
    ):
        if F.shape != (2, 2):
            raise ValueError("Plane-stress material tangent requires one 2x2 F.")
        thickness = hyperelasticity.plane_stress_thickness_stretch_value(
            F,
            material,
        )
        full_gradient = np.eye(3)
        full_gradient[:2, :2] = F
        full_gradient[2, 2] = thickness
        full = _neo_hookean_material_tangent_core(full_gradient, material)
        denominator = float(full[2, 2, 2, 2])
        if denominator <= 0.0:
            raise ValueError(
                "Plane-stress local thickness mode has a non-positive tangent."
            )
        condensed = full[:2, :2, :2, :2].copy()
        condensed -= np.einsum(
            "ij,kl->ijkl",
            full[:2, :2, 2, 2],
            full[2, 2, :2, :2],
        ) / denominator
        return condensed
    return _neo_hookean_material_tangent_core(F, material)


def _neo_hookean_material_tangent_core(
    deformation_gradient,
    material: hyperelasticity.NeoHookeanProperties,
) -> np.ndarray:
    """Return the unconstrained tangent for one 2D or 3D gradient."""

    F = np.asarray(deformation_gradient, dtype=float)
    J = float(np.linalg.det(F))
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
    material,
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
    deformation_jacobian = float(np.linalg.det(F))
    if isinstance(material, hyperelasticity.PlaneStressNeoHookeanProperties):
        deformation_jacobian *= (
            hyperelasticity.plane_stress_thickness_stretch_value(F, material)
        )
    elif (
        isinstance(material, hyperelasticity.MooneyRivlinProperties)
        and material.plane_stress_incompressible
    ):
        deformation_jacobian = 1.0
    return IncrementalWaveSpeeds(
        speeds=np.asarray(reference_speeds * speed_scale, dtype=float),
        reference_speeds=np.asarray(reference_speeds, dtype=float),
        polarizations=np.asarray(polarizations, dtype=float),
        acoustic_tensor=np.asarray(acoustic, dtype=float),
        reference_direction=np.asarray(reference_direction, dtype=float),
        current_direction=np.asarray(current_direction, dtype=float),
        direction_configuration=configuration,
        deformation_jacobian=deformation_jacobian,
    )


def principal_surface_wave_speed(
    deformation_gradient,
    material: hyperelasticity.NeoHookeanProperties,
    *,
    propagation_axis: int = 0,
    scan_points: int = 320,
) -> PrincipalSurfaceWaveSpeed:
    """Solve the 2D small-on-large principal surface-wave secular problem.

    The homogeneous deformation must be diagonal in the supplied coordinate
    system.  A harmonic incremental displacement propagates along one
    principal reference axis and decays into the other.  The admissible speed
    makes the two decaying partial waves satisfy zero incremental nominal
    traction.  The returned speed is measured per reference length and uses
    the reference density, consistently with :func:`incremental_wave_speeds`.

    This is a local half-space oracle.  It does not by itself assert that a
    finite preloaded strip, a newly released crack face, or a thin 3D sheet has
    reached that idealized state.
    """

    if material.density is None:
        raise ValueError("Surface-wave speed requires material density.")
    F = np.asarray(deformation_gradient, dtype=float)
    if F.shape != (2, 2) or not np.all(np.isfinite(F)):
        raise ValueError("principal surface waves require one finite 2x2 F.")
    scale = max(1.0, float(np.linalg.norm(F)))
    if np.linalg.norm(F - np.diag(np.diag(F))) > 1.0e-12 * scale:
        raise ValueError(
            "principal_surface_wave_speed requires a diagonal deformation "
            "gradient; rotate a principal state into its principal basis first."
        )
    axis = int(propagation_axis)
    if axis not in {0, 1}:
        raise ValueError("propagation_axis must be 0 or 1.")
    points = int(scan_points)
    if points < 80:
        raise ValueError("scan_points must be at least 80.")
    depth = 1 - axis
    if isinstance(material, hyperelasticity.PlaneStressNeoHookeanProperties):
        first_piola = hyperelasticity.plane_stress_first_piola_value(F, material)
    elif isinstance(material, hyperelasticity.MooneyRivlinProperties):
        first_piola = hyperelasticity.mooney_rivlin_first_piola_value(F, material)
    else:
        jacobian = float(np.linalg.det(F))
        first_piola = (
            float(material.mu) * F
            + (float(material.lambda_) * np.log(jacobian) - float(material.mu))
            * np.linalg.inv(F).T
        )
    base_traction = np.asarray(first_piola[:, depth], dtype=float)
    traction_scale = max(float(material.young), float(np.linalg.norm(first_piola)))
    if np.linalg.norm(base_traction) > 1.0e-8 * traction_scale:
        raise ValueError(
            "The selected principal half-space surface is not traction-free "
            "in the base state. A loaded-interface incremental-wave problem "
            "requires its own boundary condition and must not be labelled a "
            "Rayleigh surface-wave reference."
        )
    tangent = neo_hookean_material_tangent(F, material)
    Q = tangent[:, axis, :, axis]
    C = tangent[:, axis, :, depth] + tangent[:, depth, :, axis]
    T = tangent[:, depth, :, depth]
    R = tangent[:, depth, :, axis]
    density = float(material.density)
    identity = np.eye(2)

    bulk = incremental_wave_speeds(
        F,
        identity[axis],
        material,
        direction_configuration="reference",
    )
    limiting = float(bulk.slowest)

    def secular(speed: float):
        D = Q - density * float(speed) ** 2 * identity
        companion = np.block(
            [
                [np.zeros((2, 2)), identity],
                [-np.linalg.solve(T, D), -np.linalg.solve(T, C)],
            ]
        )
        roots, vectors = np.linalg.eig(companion)
        tolerance = 1.0e-9 * max(1.0, float(np.max(np.abs(roots))))
        selected = np.flatnonzero(np.imag(roots) > tolerance)
        if selected.size != 2:
            return np.inf, np.empty(0, dtype=complex)
        selected = selected[np.argsort(np.imag(roots[selected]))]
        selected_roots = np.asarray(roots[selected], dtype=complex)
        columns = []
        for index in selected:
            amplitude = vectors[:2, index]
            traction = (R + roots[index] * T) @ amplitude
            columns.append(traction)
        traction_matrix = np.column_stack(columns)
        separation = abs(selected_roots[1] - selected_roots[0])
        scale = (
            float(np.linalg.norm(columns[0]))
            * float(np.linalg.norm(columns[1]))
            * max(separation, np.finfo(float).eps)
        )
        # The ordinary two-partial-wave determinant contains the attenuation
        # root difference as a removable factor. Dividing it out prevents a
        # repeated propagation root from masquerading as a traction-free
        # surface mode; the same regularized secular equation remains valid in
        # the double-root limit.
        determinant = (
            traction_matrix[0, 0] * traction_matrix[1, 1]
            - traction_matrix[0, 1] * traction_matrix[1, 0]
        )
        residual = float(abs(determinant) / max(scale, np.finfo(float).eps))
        return residual, selected_roots

    speeds = np.linspace(0.25 * limiting, 0.999 * limiting, points)
    residuals = np.asarray([secular(value)[0] for value in speeds])
    finite = np.isfinite(residuals)
    if np.count_nonzero(finite) < 3:
        raise RuntimeError("No two decaying partial waves were found below c_s.")
    candidate = int(np.nanargmin(np.where(finite, residuals, np.nan)))
    if candidate == 0 or candidate == points - 1:
        raise RuntimeError(
            "The surface-wave residual minimum lies on the search boundary."
        )
    left = float(speeds[candidate - 1])
    right = float(speeds[candidate + 1])
    ratio = 0.5 * (np.sqrt(5.0) - 1.0)
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = secular(x1)[0]
    f2 = secular(x2)[0]
    for _ in range(80):
        if f1 > f2:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = secular(x2)[0]
        else:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = secular(x1)[0]
    speed = 0.5 * (left + right)
    residual, roots = secular(speed)
    if not np.isfinite(residual) or residual > 1.0e-7:
        raise RuntimeError(
            "The principal surface-wave secular solve did not converge; "
            f"normalized traction residual={residual:.6g}."
        )
    return PrincipalSurfaceWaveSpeed(
        speed=float(speed),
        limiting_bulk_speed=limiting,
        speed_ratio=float(speed / limiting),
        attenuation_roots=roots,
        secular_residual=residual,
        propagation_axis=axis,
        depth_axis=depth,
    )


def isotropic_reference_wave_speeds(
    material,
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
    if isinstance(material, hyperelasticity.MooneyRivlinProperties):
        if material.plane_stress_incompressible:
            raise ValueError(
                "Use incremental_wave_speeds for the reduced plane-stress "
                "Mooney-Rivlin sheet."
            )
        longitudinal_modulus = material.bulk_modulus + 4.0 * material.mu / 3.0
    else:
        longitudinal_modulus = material.lambda_ + 2.0 * material.mu
    cp = sqrt(longitudinal_modulus / rho)
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
class CrackPropagationFit:
    """Representative crack speed fitted across a declared path interval."""

    speed: float
    intercept: float
    r_squared: float
    samples: int
    start_position: float
    end_position: float
    start_time: float
    end_time: float

    def summary(self) -> dict[str, object]:
        return {
            "kind": "crack_propagation_fit",
            "speed": self.speed,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "samples": self.samples,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "method": "least_squares_position_over_declared_path_interval",
            "spatial_configuration": "declared_by_history_consumer",
        }


@dataclass(frozen=True)
class InterfaceFrontHistory:
    """Front position and fitted speed for one declared interface signal."""

    time: np.ndarray
    position: np.ndarray
    speed: np.ndarray
    signal: str
    threshold: float
    fit_window: int
    direction: str

    def summary(self) -> dict[str, object]:
        finite_speed = self.speed[np.isfinite(self.speed)]
        return {
            "kind": "interface_front_history",
            "signal": self.signal,
            "threshold": self.threshold,
            "frames": int(self.time.size),
            "fit_window": self.fit_window,
            "direction": self.direction,
            "maximum_speed": (
                None if finite_speed.size == 0 else float(np.max(finite_speed))
            ),
            "method": "contiguous_threshold_interpolation_then_local_linear_fit",
        }


@dataclass(frozen=True)
class CohesiveFrontEnsemble:
    """Crack-front evidence from multiple thresholds and physical signals."""

    histories: tuple[InterfaceFrontHistory, ...]

    def __post_init__(self) -> None:
        histories = tuple(self.histories)
        if not histories:
            raise ValueError("A cohesive front ensemble requires at least one history.")
        reference = histories[0].time
        if any(
            history.time.shape != reference.shape
            or not np.array_equal(history.time, reference)
            for history in histories[1:]
        ):
            raise ValueError("All cohesive front histories must share exact frame times.")
        object.__setattr__(self, "histories", histories)

    @property
    def time(self) -> np.ndarray:
        return self.histories[0].time

    @property
    def median_position(self) -> np.ndarray:
        return _nan_statistic(
            np.asarray([item.position for item in self.histories]),
            np.nanmedian,
        )

    @property
    def median_speed(self) -> np.ndarray:
        return _nan_statistic(
            np.asarray([item.speed for item in self.histories]),
            np.nanmedian,
        )

    def summary(self) -> dict[str, object]:
        maxima = np.asarray(
            [
                np.nanmax(item.speed)
                if np.any(np.isfinite(item.speed))
                else np.nan
                for item in self.histories
            ],
            dtype=float,
        )
        finite = maxima[np.isfinite(maxima)]
        return {
            "kind": "cohesive_front_ensemble",
            "observers": [item.summary() for item in self.histories],
            "maximum_speed_median": (
                None if finite.size == 0 else float(np.median(finite))
            ),
            "maximum_speed_spread": (
                None if finite.size == 0 else float(np.max(finite) - np.min(finite))
            ),
            "purpose": "observer_sensitivity_not_single_element_speed",
        }


@dataclass(frozen=True)
class CohesiveInterfaceTrace:
    """Portable accepted-frame record on one fixed cohesive interface."""

    time: np.ndarray
    path_coordinate: np.ndarray
    opening: np.ndarray
    traction: np.ndarray
    damage: np.ndarray
    dissipated_energy_density: np.ndarray
    tangential_jump: np.ndarray | None = None
    tangential_traction: np.ndarray | None = None
    mode_mixity: np.ndarray | None = None
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        coordinate = np.asarray(self.path_coordinate, dtype=float)
        if time.ndim != 1 or time.size < 1 or np.any(~np.isfinite(time)):
            raise ValueError("Cohesive trace time must be a finite 1D array.")
        if time.size > 1 and np.any(np.diff(time) <= 0.0):
            raise ValueError("Cohesive trace time must be strictly increasing.")
        if coordinate.ndim != 1 or coordinate.size < 1 or np.any(~np.isfinite(coordinate)):
            raise ValueError("Cohesive trace coordinates must be a finite 1D array.")
        shape = (time.size, coordinate.size)
        arrays = {}
        for name in ("opening", "traction", "damage", "dissipated_energy_density"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != shape or np.any(~np.isfinite(values)):
                raise ValueError(f"Cohesive trace {name} must have shape {shape} and be finite.")
            arrays[name] = values.copy()
        vector_arrays = {}
        for name in ("tangential_jump", "tangential_traction"):
            declared = getattr(self, name)
            if declared is None:
                continue
            values = np.asarray(declared, dtype=float)
            if values.ndim != 3 or values.shape[:2] != shape or np.any(~np.isfinite(values)):
                raise ValueError(
                    f"Cohesive trace {name} must have shape {shape}+(components,) and be finite."
                )
            vector_arrays[name] = values.copy()
        scalar_optional = None
        if self.mode_mixity is not None:
            scalar_optional = np.asarray(self.mode_mixity, dtype=float)
            if scalar_optional.shape != shape or np.any(~np.isfinite(scalar_optional)):
                raise ValueError(
                    f"Cohesive trace mode_mixity must have shape {shape} and be finite."
                )
        object.__setattr__(self, "time", time.copy())
        object.__setattr__(self, "path_coordinate", coordinate.copy())
        for name, values in arrays.items():
            object.__setattr__(self, name, values)
        for name in ("tangential_jump", "tangential_traction"):
            object.__setattr__(self, name, vector_arrays.get(name))
        object.__setattr__(
            self,
            "mode_mixity",
            None if scalar_optional is None else scalar_optional.copy(),
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def write(self, path: str | Path) -> Path:
        """Write a compact, dependency-free NPZ research artifact."""

        location = Path(path)
        if location.suffix.lower() != ".npz":
            location = location.with_suffix(".npz")
        location.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": np.asarray("agentfem.cohesive-interface-trace.v2"),
            "time": self.time,
            "path_coordinate": self.path_coordinate,
            "opening": self.opening,
            "traction": self.traction,
            "damage": self.damage,
            "dissipated_energy_density": self.dissipated_energy_density,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        for name in ("tangential_jump", "tangential_traction", "mode_mixity"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        np.savez_compressed(location, **payload)
        return location

    @classmethod
    def read(cls, path: str | Path) -> "CohesiveInterfaceTrace":
        location = Path(path)
        with np.load(location, allow_pickle=False) as archive:
            schema = str(archive["schema"])
            if schema not in {
                "agentfem.cohesive-interface-trace.v1",
                "agentfem.cohesive-interface-trace.v2",
            }:
                raise ValueError(f"Unsupported cohesive trace schema {schema!r}.")
            return cls(
                time=archive["time"],
                path_coordinate=archive["path_coordinate"],
                opening=archive["opening"],
                traction=archive["traction"],
                damage=archive["damage"],
                dissipated_energy_density=archive["dissipated_energy_density"],
                tangential_jump=(
                    archive["tangential_jump"]
                    if "tangential_jump" in archive.files
                    else None
                ),
                tangential_traction=(
                    archive["tangential_traction"]
                    if "tangential_traction" in archive.files
                    else None
                ),
                mode_mixity=(
                    archive["mode_mixity"]
                    if "mode_mixity" in archive.files
                    else None
                ),
                metadata=json.loads(str(archive["metadata_json"])),
            )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "cohesive_interface_trace",
            "frames": int(self.time.size),
            "interface_points": int(self.path_coordinate.size),
            "time_interval": [float(self.time[0]), float(self.time[-1])],
            "maximum_opening": float(np.max(self.opening)),
            "maximum_damage": float(np.max(self.damage)),
            "vector_kinematics": self.tangential_jump is not None,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ScientificComparison:
    """Common scalar evidence for a simulation-to-observation comparison."""

    kind: str
    samples: int
    root_mean_square_error: float
    normalized_root_mean_square_error: float
    correlation: float | None
    metadata: dict[str, object] | None = None

    def summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "samples": self.samples,
            "root_mean_square_error": self.root_mean_square_error,
            "normalized_root_mean_square_error": self.normalized_root_mean_square_error,
            "correlation": self.correlation,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class PreloadTransferReport:
    """Evidence for a quasi-static displacement to Explicit state transfer."""

    mode: str
    residual_force_norm: float
    total_force_norm: float
    constrained_force_norm: float
    acceleration_norm: float
    force_tolerance: float
    equilibrium_accepted: bool
    initial_velocity: str
    source_step: str | None = None
    destination_step: str | None = None
    source_energy: float | None = None
    destination_energy: float | None = None
    relative_energy_jump: float | None = None

    def summary(self) -> dict[str, object]:
        return {
            "kind": "preload_to_explicit_state_transfer",
            "mode": self.mode,
            "residual_force_norm": self.residual_force_norm,
            "total_force_norm": self.total_force_norm,
            "constrained_force_norm": self.constrained_force_norm,
            "acceleration_norm": self.acceleration_norm,
            "force_tolerance": self.force_tolerance,
            "equilibrium_accepted": self.equilibrium_accepted,
            "initial_velocity": self.initial_velocity,
            "source_step": self.source_step,
            "destination_step": self.destination_step,
            "source_energy": self.source_energy,
            "destination_energy": self.destination_energy,
            "relative_energy_jump": self.relative_energy_jump,
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
    energy_monitor=None,
    source_energy: float | None = None,
    source_step: str | None = None,
    destination_step: str | None = None,
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
        total_force_norm = float(vector.norm())
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
    owned_acceleration = np.asarray(dofs.owned_array(acceleration), dtype=float)
    local_squared = float(np.dot(owned_acceleration, owned_acceleration))
    acceleration_norm = sqrt(
        acceleration.function_space.mesh.comm.allreduce(local_squared, op=MPI.SUM)
    )
    if hasattr(mass, "mass"):
        diagonal = np.asarray(mass.mass, dtype=float)[: owned_acceleration.size]
    else:
        selected_inverse = np.asarray(inverse, dtype=float)[: owned_acceleration.size]
        diagonal = np.divide(
            1.0,
            selected_inverse,
            out=np.zeros_like(selected_inverse),
            where=selected_inverse != 0.0,
        )
    free_force = diagonal * owned_acceleration
    local_force_squared = float(np.dot(free_force, free_force))
    force_norm = sqrt(
        acceleration.function_space.mesh.comm.allreduce(
            local_force_squared,
            op=MPI.SUM,
        )
    )
    constrained_force_norm = sqrt(
        max(0.0, total_force_norm**2 - force_norm**2)
    )
    destination_energy = None
    if energy_monitor is not None:
        energy_values = energy_monitor.evaluate(
            displacement=state.u,
            velocity=state.v,
        )
        destination_energy = _accounted_energy_value(energy_values)
    selected_source_energy = (
        None if source_energy is None else float(source_energy)
    )
    relative_energy_jump = None
    if selected_source_energy is not None and destination_energy is not None:
        scale = max(abs(selected_source_energy), abs(destination_energy), 1.0e-30)
        relative_energy_jump = abs(destination_energy - selected_source_energy) / scale
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
        total_force_norm=total_force_norm,
        constrained_force_norm=constrained_force_norm,
        acceleration_norm=acceleration_norm,
        force_tolerance=tolerance,
        equilibrium_accepted=accepted,
        initial_velocity=velocity_label,
        source_step=None if source_step is None else str(source_step),
        destination_step=(
            None if destination_step is None else str(destination_step)
        ),
        source_energy=selected_source_energy,
        destination_energy=destination_energy,
        relative_energy_jump=relative_energy_jump,
    )


def _accounted_energy_value(values: dict[str, float]) -> float:
    for key in (
        "accounted_internal_kinetic_energy",
        "total_mechanical_energy",
    ):
        if key in values:
            return float(values[key])
    raise ValueError(
        "The preload transfer energy monitor does not expose an accounted "
        "mechanical-energy channel."
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


def fit_crack_propagation_speed(
    history: CohesiveCrackHistory,
    *,
    start_position: float,
    end_position: float,
    minimum_samples: int = 3,
) -> CrackPropagationFit | None:
    """Fit one representative speed over a fixed physical path interval.

    Local window fits are appropriate for identifying a peak-speed regime but
    their maximum is sensitive to crack initiation and short transients. This
    fit instead compares the same propagation distance across mesh and
    time-step refinements. ``None`` is returned when the observed front has not
    crossed enough of the interval; partial propagation is therefore not
    silently reported as a converged speed.
    """

    if not isinstance(history, CohesiveCrackHistory):
        raise TypeError("history must be a CohesiveCrackHistory.")
    start = float(start_position)
    end = float(end_position)
    required = int(minimum_samples)
    if not isfinite(start) or not isfinite(end) or end <= start:
        raise ValueError("end_position must be finite and greater than start_position.")
    if required < 3:
        raise ValueError("minimum_samples must be at least three.")
    valid = (
        np.isfinite(history.time)
        & np.isfinite(history.position)
        & (history.position >= start)
        & (history.position <= end)
    )
    if np.count_nonzero(valid) < required:
        return None
    time = np.asarray(history.time[valid], dtype=float)
    position = np.asarray(history.position[valid], dtype=float)
    if np.unique(position).size < 3 or float(np.ptp(position)) <= 0.0:
        return None
    mean_time = float(np.mean(time))
    speed, centered_intercept = np.polyfit(time - mean_time, position, 1)
    intercept = float(centered_intercept - speed * mean_time)
    predicted = float(speed) * time + intercept
    residual = float(np.sum((position - predicted) ** 2))
    total = float(np.sum((position - np.mean(position)) ** 2))
    r_squared = 1.0 if total <= np.finfo(float).eps else 1.0 - residual / total
    return CrackPropagationFit(
        speed=float(speed),
        intercept=intercept,
        r_squared=float(r_squared),
        samples=int(time.size),
        start_position=start,
        end_position=end,
        start_time=float(time[0]),
        end_time=float(time[-1]),
    )


def interface_front_history(
    time_values,
    path_coordinate,
    signal_frames,
    *,
    signal: str,
    threshold: float,
    fit_window: int = 5,
    direction: str = "increasing",
) -> InterfaceFrontHistory:
    """Track a contiguous interface front from any increasing damage signal.

    Suitable signals include damage, opening, and cumulative cohesive
    dissipation.  The threshold carries the physical units of the signal.
    """

    times = np.asarray(time_values, dtype=float)
    coordinate = np.asarray(path_coordinate, dtype=float)
    frames = np.asarray(signal_frames, dtype=float)
    selected_threshold = float(threshold)
    label = str(signal).strip()
    if not label:
        raise ValueError("Interface front signal needs a nonempty name.")
    if not isfinite(selected_threshold):
        raise ValueError("Interface front threshold must be finite.")
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("time_values must be a strictly increasing 1D array.")
    if coordinate.ndim != 1 or coordinate.size < 2:
        raise ValueError("path_coordinate must be a 1D array of size >= 2.")
    if frames.shape != (times.size, coordinate.size) or np.any(~np.isfinite(frames)):
        raise ValueError("signal_frames must be finite with shape (frames, path_points).")
    window = int(fit_window)
    if window < 3 or window % 2 == 0:
        raise ValueError("fit_window must be an odd integer of at least three.")
    position = np.asarray(
        [
            _contiguous_threshold_front(
                coordinate,
                frame,
                threshold=selected_threshold,
                direction=direction,
            )
            for frame in frames
        ],
        dtype=float,
    )
    return InterfaceFrontHistory(
        time=times.copy(),
        position=position,
        speed=_window_fitted_speed(times, position, window),
        signal=label,
        threshold=selected_threshold,
        fit_window=window,
        direction=direction,
    )


def cohesive_front_ensemble(
    trace: CohesiveInterfaceTrace,
    *,
    damage_thresholds=(0.5, 0.75, 0.95),
    opening_thresholds=(),
    dissipation_thresholds=(),
    fit_window: int = 5,
    direction: str = "increasing",
) -> CohesiveFrontEnsemble:
    """Build observer-sensitivity evidence from a portable interface trace."""

    histories = []
    for threshold in damage_thresholds:
        selected = float(threshold)
        if not 0.0 < selected < 1.0:
            raise ValueError("Damage observer thresholds must lie in (0, 1).")
        histories.append(
            interface_front_history(
                trace.time,
                trace.path_coordinate,
                trace.damage,
                signal="damage",
                threshold=selected,
                fit_window=fit_window,
                direction=direction,
            )
        )
    for name, thresholds, values in (
        ("opening", opening_thresholds, trace.opening),
        ("dissipated_energy_density", dissipation_thresholds, trace.dissipated_energy_density),
    ):
        for threshold in thresholds:
            histories.append(
                interface_front_history(
                    trace.time,
                    trace.path_coordinate,
                    values,
                    signal=name,
                    threshold=float(threshold),
                    fit_window=fit_window,
                    direction=direction,
                )
            )
    return CohesiveFrontEnsemble(tuple(histories))


def compare_curve(
    reference_coordinate,
    reference_values,
    simulation_coordinate,
    simulation_values,
    *,
    coordinate_name: str = "coordinate",
    quantity_name: str = "value",
) -> ScientificComparison:
    """Interpolate a simulated curve onto observed coordinates and compare."""

    reference_x, reference_y = _sorted_curve(
        reference_coordinate, reference_values, name="reference"
    )
    simulation_x, simulation_y = _sorted_curve(
        simulation_coordinate, simulation_values, name="simulation"
    )
    mask = (reference_x >= simulation_x[0]) & (reference_x <= simulation_x[-1])
    if np.count_nonzero(mask) < 2:
        raise ValueError("Curve comparison requires at least two overlapping samples.")
    observed = reference_y[mask]
    predicted = np.interp(reference_x[mask], simulation_x, simulation_y)
    return _scientific_comparison(
        observed,
        predicted,
        kind="curve_comparison",
        metadata={
            "coordinate": str(coordinate_name),
            "quantity": str(quantity_name),
            "interpolation": "simulation_linear_to_reference_coordinates",
            "overlap": [float(reference_x[mask][0]), float(reference_x[mask][-1])],
        },
    )


def compare_mach_cone(
    *,
    crack_speed: float,
    shear_wave_speed: float,
    observed_angle: float,
    unit: str = "radian",
) -> ScientificComparison:
    """Compare an observed Mach angle with ``asin(c_s/v)``."""

    selected_unit = str(unit).strip().lower()
    if selected_unit not in {"radian", "degree"}:
        raise ValueError("Mach-cone angle unit must be 'radian' or 'degree'.")
    predicted = mach_cone_angle(
        crack_speed=crack_speed,
        shear_wave_speed=shear_wave_speed,
    )
    observed = float(observed_angle)
    if selected_unit == "degree":
        predicted = float(np.degrees(predicted))
    if not isfinite(observed):
        raise ValueError("Observed Mach-cone angle must be finite.")
    error = abs(predicted - observed)
    scale = max(abs(observed), np.finfo(float).eps)
    return ScientificComparison(
        kind="mach_cone_comparison",
        samples=1,
        root_mean_square_error=error,
        normalized_root_mean_square_error=error / scale,
        correlation=None,
        metadata={
            "unit": selected_unit,
            "observed_angle": observed,
            "predicted_angle": predicted,
            "relation": "asin(shear_wave_speed/crack_speed)",
        },
    )


def compare_rectilinear_field(
    reference_x,
    reference_y,
    reference_values,
    simulation_x,
    simulation_y,
    simulation_values,
    *,
    quantity_name: str = "field",
    reference_mask=None,
    simulation_mask=None,
) -> ScientificComparison:
    """Compare scalar maps after bilinear interpolation on their overlap.

    Arrays use image-style shape ``(len(y), len(x))``.  This deliberately
    separates interpolation evidence from visualization and avoids a SciPy
    dependency in the base installation.
    """

    rx, ry, observed = _rectilinear_field(
        reference_x, reference_y, reference_values, name="reference"
    )
    sx, sy, simulated = _rectilinear_field(
        simulation_x, simulation_y, simulation_values, name="simulation"
    )
    x_mask = (rx >= sx[0]) & (rx <= sx[-1])
    y_mask = (ry >= sy[0]) & (ry <= sy[-1])
    if np.count_nonzero(x_mask) < 2 or np.count_nonzero(y_mask) < 2:
        raise ValueError("Field comparison requires a two-dimensional overlapping grid.")
    selected_x = rx[x_mask]
    selected_y = ry[y_mask]
    along_x = np.asarray(
        [np.interp(selected_x, sx, row) for row in simulated],
        dtype=float,
    )
    predicted = np.asarray(
        [np.interp(selected_y, sy, along_x[:, index]) for index in range(selected_x.size)],
        dtype=float,
    ).T
    selected_observed = observed[np.ix_(y_mask, x_mask)]
    valid = np.ones(selected_observed.shape, dtype=bool)
    if reference_mask is not None:
        reference_valid = np.asarray(reference_mask, dtype=bool)
        if reference_valid.shape != observed.shape:
            raise ValueError("reference_mask must match reference_values.")
        valid &= reference_valid[np.ix_(y_mask, x_mask)]
    if simulation_mask is not None:
        simulation_valid = np.asarray(simulation_mask, dtype=bool)
        if simulation_valid.shape != simulated.shape:
            raise ValueError("simulation_mask must match simulation_values.")
        mask_along_x = np.asarray(
            [np.interp(selected_x, sx, row.astype(float)) for row in simulation_valid],
            dtype=float,
        )
        mask_on_reference = np.asarray(
            [
                np.interp(selected_y, sy, mask_along_x[:, index])
                for index in range(selected_x.size)
            ],
            dtype=float,
        ).T
        # A comparison point is valid only when every bilinear contributor is
        # in the physical domain.  This prevents void fill values from entering
        # a field error silently.
        valid &= mask_on_reference >= 1.0 - 64.0 * np.finfo(float).eps
    if np.count_nonzero(valid) < 2:
        raise ValueError("Field comparison requires at least two valid samples.")
    return _scientific_comparison(
        selected_observed[valid],
        predicted[valid],
        kind="rectilinear_field_comparison",
        metadata={
            "quantity": str(quantity_name),
            "interpolation": "bilinear_simulation_to_reference_grid",
            "overlap_bounds": [
                [float(selected_x[0]), float(selected_x[-1])],
                [float(selected_y[0]), float(selected_y[-1])],
            ],
            "overlap_shape": [int(selected_y.size), int(selected_x.size)],
            "valid_samples": int(np.count_nonzero(valid)),
            "mask_policy": "all_bilinear_contributors_inside",
        },
    )


def compare_rectilinear_observations(
    reference,
    simulation,
    *,
    quantity_name: str | None = None,
) -> ScientificComparison:
    """Compare two portable rectilinear observations with semantic checks."""

    reference_quantity = str(getattr(reference, "quantity", "field"))
    simulation_quantity = str(getattr(simulation, "quantity", "field"))
    selected_quantity = str(quantity_name or reference_quantity)
    if quantity_name is None and reference_quantity != simulation_quantity:
        raise ValueError(
            "Rectilinear observation quantities differ; pass quantity_name only "
            "after reviewing the intended comparison."
        )
    reference_unit = getattr(reference, "unit", None)
    simulation_unit = getattr(simulation, "unit", None)
    if reference_unit != simulation_unit:
        raise ValueError(
            "Rectilinear observation units differ; convert them explicitly before comparison."
        )
    reference_coordinate_unit = getattr(reference, "coordinate_unit", None)
    simulation_coordinate_unit = getattr(simulation, "coordinate_unit", None)
    if reference_coordinate_unit != simulation_coordinate_unit:
        raise ValueError(
            "Rectilinear observation coordinate units differ; convert coordinates "
            "explicitly before comparison."
        )
    reference_configuration = getattr(reference, "configuration", None)
    simulation_configuration = getattr(simulation, "configuration", None)
    if reference_configuration != simulation_configuration:
        raise ValueError(
            "Rectilinear observation configurations differ; provide an explicit "
            "reference/current coordinate registration before comparison."
        )
    reference_system = getattr(reference, "coordinate_system", None)
    simulation_system = getattr(simulation, "coordinate_system", None)
    if reference_system != simulation_system:
        raise ValueError(
            "Rectilinear observation coordinate systems differ; register both "
            "observations to one reviewed coordinate system before comparison."
        )
    reference_names = tuple(getattr(reference, "coordinate_names", ("x", "y")))
    simulation_names = tuple(getattr(simulation, "coordinate_names", ("x", "y")))
    if reference_names != simulation_names:
        raise ValueError(
            "Rectilinear observation coordinate names differ; reorder the axes explicitly."
        )
    comparison = compare_rectilinear_field(
        reference.x,
        reference.y,
        reference.values,
        simulation.x,
        simulation.y,
        simulation.values,
        quantity_name=selected_quantity,
        reference_mask=getattr(reference, "mask", None),
        simulation_mask=getattr(simulation, "mask", None),
    )
    metadata = dict(comparison.metadata or {})
    metadata.update(
        {
            "unit": reference_unit,
            "coordinate_unit": reference_coordinate_unit,
            "configuration": reference_configuration,
            "coordinate_system": reference_system,
            "coordinate_names": reference_names,
        }
    )
    return ScientificComparison(
        kind=comparison.kind,
        samples=comparison.samples,
        root_mean_square_error=comparison.root_mean_square_error,
        normalized_root_mean_square_error=comparison.normalized_root_mean_square_error,
        correlation=comparison.correlation,
        metadata=metadata,
    )


def _contiguous_threshold_front(coordinate, values, *, threshold, direction):
    order = np.argsort(coordinate)
    if direction == "decreasing":
        order = order[::-1]
    elif direction != "increasing":
        raise ValueError("direction must be 'increasing' or 'decreasing'.")
    x = np.asarray(coordinate, dtype=float)[order]
    signal = np.asarray(values, dtype=float)[order]
    active = signal >= threshold
    if not active[0]:
        return float("nan")
    inactive = np.flatnonzero(~active)
    if inactive.size == 0:
        return float(x[-1])
    right = int(inactive[0])
    left = right - 1
    denominator = signal[right] - signal[left]
    if abs(denominator) <= np.finfo(float).eps:
        return float(0.5 * (x[left] + x[right]))
    fraction = (threshold - signal[left]) / denominator
    return float(x[left] + fraction * (x[right] - x[left]))


def _window_fitted_speed(times, position, window):
    speed = np.full(np.asarray(times).shape, np.nan, dtype=float)
    half = int(window) // 2
    for index in range(len(times)):
        start = max(0, index - half)
        stop = min(len(times), index + half + 1)
        valid = np.isfinite(position[start:stop])
        if np.count_nonzero(valid) < 2:
            continue
        t = times[start:stop][valid]
        x = position[start:stop][valid]
        speed[index] = float(np.polyfit(t - np.mean(t), x, 1)[0])
    return speed


def _nan_statistic(values, statistic):
    with np.errstate(invalid="ignore"):
        return statistic(values, axis=0)


def _sorted_curve(coordinate, values, *, name):
    x = np.asarray(coordinate, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError(f"{name} curve must contain equal 1D arrays of size >= 2.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError(f"{name} curve must be finite.")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if np.any(np.diff(x) <= 0.0):
        raise ValueError(f"{name} curve coordinates must be unique.")
    return x, y


def _rectilinear_field(x, y, values, *, name):
    selected_x = np.asarray(x, dtype=float)
    selected_y = np.asarray(y, dtype=float)
    field = np.asarray(values, dtype=float)
    if (
        selected_x.ndim != 1
        or selected_y.ndim != 1
        or selected_x.size < 2
        or selected_y.size < 2
        or field.shape != (selected_y.size, selected_x.size)
    ):
        raise ValueError(
            f"{name} field must have shape (len(y), len(x)) on 1D axes."
        )
    if np.any(~np.isfinite(selected_x)) or np.any(~np.isfinite(selected_y)) or np.any(~np.isfinite(field)):
        raise ValueError(f"{name} field and axes must be finite.")
    x_order = np.argsort(selected_x)
    y_order = np.argsort(selected_y)
    selected_x = selected_x[x_order]
    selected_y = selected_y[y_order]
    if np.any(np.diff(selected_x) <= 0.0) or np.any(np.diff(selected_y) <= 0.0):
        raise ValueError(f"{name} field axes must be unique.")
    return selected_x, selected_y, field[np.ix_(y_order, x_order)]


def _scientific_comparison(observed, predicted, *, kind, metadata):
    observed_values = np.asarray(observed, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    difference = predicted_values - observed_values
    rmse = float(np.sqrt(np.mean(difference**2)))
    span = float(np.max(observed_values) - np.min(observed_values))
    scale = span if span > np.finfo(float).eps else max(
        float(np.sqrt(np.mean(observed_values**2))), np.finfo(float).eps
    )
    correlation = None
    if observed_values.size >= 2 and np.std(observed_values) > 0.0 and np.std(predicted_values) > 0.0:
        correlation = float(np.corrcoef(observed_values, predicted_values)[0, 1])
    return ScientificComparison(
        kind=kind,
        samples=int(observed_values.size),
        root_mean_square_error=rmse,
        normalized_root_mean_square_error=rmse / scale,
        correlation=correlation,
        metadata=metadata,
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
    rapid_failed_fraction: float | None = None,
    ligament_traction_ratio: float | None = None,
    pressure_wave_speed: float | None = None,
) -> str:
    """Classify one frame with explicit crack-speed and spall evidence."""

    speed = float(crack_speed)
    rayleigh = float(rayleigh_wave_speed)
    shear = float(shear_wave_speed)
    failed = float(failed_fraction)
    simultaneous = float(simultaneous_failed_fraction)
    rapid = simultaneous if rapid_failed_fraction is None else float(rapid_failed_fraction)
    traction_ratio = (
        None if ligament_traction_ratio is None else float(ligament_traction_ratio)
    )
    pressure = None if pressure_wave_speed is None else float(pressure_wave_speed)
    threshold = float(spall_fraction)
    if not 0.0 < rayleigh < shear:
        raise ValueError("Wave speeds must satisfy 0 < c_R < c_s.")
    if not all(
        0.0 <= value <= 1.0
        for value in (failed, simultaneous, rapid, threshold)
    ):
        raise ValueError("Failure fractions must lie in [0, 1].")
    if traction_ratio is not None and (
        not isfinite(traction_ratio) or traction_ratio < 0.0
    ):
        raise ValueError("ligament_traction_ratio must be finite and nonnegative.")
    if pressure is not None and (not isfinite(pressure) or pressure <= shear):
        raise ValueError("pressure_wave_speed must be finite and greater than c_s.")
    traction_reached = traction_ratio is None or traction_ratio >= 0.95
    if failed >= threshold and rapid >= threshold and traction_reached:
        return "spall_like"
    if not isfinite(speed):
        return "unresolved"
    if speed <= rayleigh:
        return "sub_rayleigh_crack_like"
    if speed <= shear:
        return "trans_rayleigh"
    if (
        pressure is not None
        and speed > pressure
        and failed >= threshold
        and rapid >= 0.5 * threshold
        and traction_reached
    ):
        return "spall_like"
    if pressure is not None and speed > pressure:
        return "unresolved_discrete_failure"
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
    "CohesiveForceCollection",
    "DynamicFractureEvidenceBundle",
    "DofMappedCohesiveForce",
    "DistributedDofMappedCohesiveForce",
    "CohesiveFrontEnsemble",
    "CohesiveInterfaceTrace",
    "CohesiveCrackHistory",
    "CrackPropagationFit",
    "InterfaceFrontHistory",
    "NamedCohesiveResponse",
    "PreloadTransferReport",
    "FiniteStrainEnergyMonitor",
    "FiniteStrainCohesiveEnergyMonitor",
    "CohesiveNewtonSolveInfo",
    "ArcLengthOptions",
    "ArcLengthSolveInfo",
    "FiniteStrainCohesiveArcLength",
    "FiniteStrainCohesiveEquilibrium",
    "FiniteStrainCohesiveResidual",
    "IsotropicWaveSpeeds",
    "PrincipalSurfaceWaveSpeed",
    "StableTimeIncrement",
    "ScientificComparison",
    "cohesive_front_ensemble",
    "compare_curve",
    "compare_mach_cone",
    "compare_rectilinear_field",
    "compare_rectilinear_observations",
    "estimate_stable_time_increment",
    "cohesive_crack_tip",
    "crack_tip_history",
    "fit_crack_propagation_speed",
    "interface_front_history",
    "finite_strain_internal_force",
    "isotropic_reference_wave_speeds",
    "principal_surface_wave_speed",
    "minimum_cell_nodal_spacing",
    "mach_cone_angle",
    "cohesive_force",
    "cohesive_forces",
    "mode_i_cohesive_force",
    "named_cohesive_forces",
    "named_mode_i_cohesive_forces",
    "p1_input_node_to_block_dof",
    "separation_regime",
    "transfer_preload_to_explicit",
]
