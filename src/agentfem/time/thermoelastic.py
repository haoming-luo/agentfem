# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Private bounded thermoelastic Procedure, before public Step promotion.

The benchmark owns its independent oracle. This execution object owns only
the physical participants and one accepted time coordinate. No mesh creation
or reference solution belongs here.
"""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from math import isfinite
from hashlib import sha256
import json
import numpy as np

from .. import checkpointing, results, solvers, state
from ..constraints import TimeDependentDirichlet
from ..operators.thermoelastic import _thermoelastic_blocks
from ..provenance import collective_call, collective_canonical_record
from ._staggered import StaggeredFieldIteration
from ._thermoelastic_energy import ThermoelasticEnergyEvidence


class _ThermoelasticStep:
    """Same-mesh 3D homogeneous small-strain physics with SI numeric inputs.

    Temperature is explicitly a departure from material reference temperature.
    Registered time-dependent strong values modify only the RHS. Joint restart
    includes their accepted time and conjugate force history. Public execution
    and output policy still need integration before built-in Step promotion.
    """

    def __init__(
        self,
        displacement,
        temperature_departure,
        *,
        material,
        dt,
        steps,
        mechanical_bcs=(),
        thermal_bcs=(),
        heat_load=None,
        mechanical_load=None,
        solver_options=None,
        input_identity=None,
        name="coupled_thermoelastic",
    ):
        self.u, self.theta = displacement, temperature_departure
        self.comm = self.u.space.mesh.comm
        configuration = (dt, steps)
        if any(item != configuration for item in self.comm.allgather(configuration)):
            raise ValueError(
                "AFM-COUPLING-003: time configuration differs across ranks."
            )
        if not isfinite(dt) or dt <= 0 or type(steps) is not int or steps < 1:
            raise ValueError("Positive finite dt and positive integer steps required.")
        self.dt, self.total_steps, self.completed_steps = float(dt), steps, 0
        self.name, self.material = name, material
        mechanical_assets, thermal_assets = tuple(mechanical_bcs), tuple(thermal_bcs)
        self._time_boundaries = tuple(
            item
            for item in (*mechanical_assets, *thermal_assets)
            if isinstance(item, TimeDependentDirichlet)
        )
        mechanical_bcs = tuple(getattr(item, "bc", item) for item in mechanical_assets)
        thermal_bcs = tuple(getattr(item, "bc", item) for item in thermal_assets)
        self.input_identity = collective_canonical_record(
            {
                "declared_inputs": input_identity,
                "material": material.as_dict(),
                "boundary_histories": [
                    item.summary() for item in self._time_boundaries
                ],
            },
            comm=self.comm,
            label="coupled scientific input identity",
        )
        self.restart_source = None
        self.u_old = self.u.value.copy()
        self.theta_old = self.theta.value.copy()
        self.blocks = _thermoelastic_blocks(
            self.u,
            self.theta,
            old_displacement=self.u_old,
            old_temperature_departure=self.theta_old,
            material=material,
            dt=dt,
            heat_load=heat_load,
            mechanical_load=mechanical_load,
        )
        self._iteration = StaggeredFieldIteration(
            {
                "displacement": self.u.value,
                "temperature": self.theta.value,
            }
        )
        self.history = []
        self._active = False
        self._closed = False
        self._resources = ExitStack()
        mechanical_bcs, thermal_bcs = tuple(mechanical_bcs), tuple(thermal_bcs)
        initial_boundary_values = [
            np.asarray(item.constant.value).copy() for item in self._time_boundaries
        ]
        try:
            with self._boundary_time(0.0):
                pass
            self.thermal = self._resources.enter_context(
                solvers.prepare_linear_problem(
                    self.blocks.thermal_matrix,
                    self.blocks.thermal_rhs,
                    self.theta.value,
                    bcs=thermal_bcs,
                    options=solver_options,
                )
            )
            self.mechanical = self._resources.enter_context(
                solvers.prepare_linear_problem(
                    self.blocks.mechanical_matrix,
                    self.blocks.mechanical_rhs,
                    self.u.value,
                    bcs=mechanical_bcs,
                    options=solver_options,
                )
            )
            self.energy = ThermoelasticEnergyEvidence(
                self, mechanical_bcs, thermal_bcs, heat_load, mechanical_load
            )
        except BaseException:
            for item, value in zip(self._time_boundaries, initial_boundary_values):
                item.constant.value = value
            self.close()
            raise

    @property
    def time(self):
        return self.completed_steps * self.dt

    @contextmanager
    def _boundary_time(self, time):
        previous = [
            np.asarray(item.constant.value).copy() for item in self._time_boundaries
        ]
        try:

            def update():
                for item in self._time_boundaries:
                    item.update(time)
                    if not np.all(np.isfinite(item.constant.value)):
                        raise ValueError(
                            "AFM-COUPLING-002: non-finite prescribed boundary."
                        )

            collective_call(
                update, comm=self.comm, label="update coupled boundary time"
            )
            yield
        except BaseException:
            for item, value in zip(self._time_boundaries, previous):
                item.constant.value = value
            raise

    def advance(
        self,
        *,
        relaxation=1.0,
        max_iterations=200,
        rtol=1e-9,
        displacement_atol=1e-13,
        temperature_atol=1e-11,
        acceptance_check=None,
    ):
        declaration = (
            self.completed_steps,
            self._active,
            self._closed,
            acceptance_check is not None,
        )
        if any(item != declaration for item in self.comm.allgather(declaration)):
            raise ValueError(
                "AFM-COUPLING-003: accepted boundary differs across ranks."
            )
        if self._active or self._closed or self.completed_steps >= self.total_steps:
            raise RuntimeError("AFM-COUPLING-004: no new coupled window is available.")
        self._active = True
        try:
            with (
                self._boundary_time((self.completed_steps + 1) * self.dt),
                state.field_transaction(displacement=self.u, temperature=self.theta),
            ):
                residuals = self._iteration.run(
                    (self.thermal.solve, self.mechanical.solve),
                    absolute_tolerances={
                        "displacement": displacement_atol,
                        "temperature": temperature_atol,
                    },
                    rtol=rtol,
                    relaxation=relaxation,
                    max_iterations=max_iterations,
                    decoupled=self.blocks.beta == 0,
                )
                record = {
                    "step": self.completed_steps + 1,
                    "time": (self.completed_steps + 1) * self.dt,
                    "outer_iterations": len(residuals),
                    "residuals": residuals,
                    "iteration_policy": {
                        "relaxation": relaxation,
                        "rtol": rtol,
                        "max_iterations": max_iterations,
                        "displacement_atol": displacement_atol,
                        "temperature_atol": temperature_atol,
                    },
                }
                evidence, reaction, external = self.energy.evaluate()
                record.update(evidence)
                if acceptance_check is not None:
                    collective_call(
                        lambda: acceptance_check(deepcopy(record)),
                        comm=self.comm,
                        label="accept coupled window",
                    )
            for previous, current in (
                (self.u_old, self.u.value),
                (self.theta_old, self.theta.value),
            ):
                previous.x.array[:] = current.x.array
                previous.x.scatter_forward()
            self.completed_steps += 1
            self.energy.commit(reaction, external)
            self.history.append(record)
            return deepcopy(record)
        finally:
            self._active = False

    def run(self, **iteration_options):
        while self.completed_steps < self.total_steps:
            self.advance(**iteration_options)
        return self

    def result(self):
        if not self.history:
            raise RuntimeError("No accepted coupled window is available.")
        result = results.SimulationResult(self.name)
        result.add_field("Displacement", self.u.value.copy(), unit="m")
        result.add_field("TemperatureDeparture", self.theta.value.copy(), unit="K")
        for key in ("linearized_heat_balance_absolute", "quadratic_balance_absolute"):
            result.add_quantity(
                "maximum_" + key,
                max(record[key] for record in self.history),
                unit="J",
                kind="diagnostic",
            )
        for key in (
            "natural_heat_input",
            "prescribed_temperature_heat_input",
            "prescribed_motion_path_work",
            "natural_load_path_work",
        ):
            result.add_quantity(
                "cumulative_" + key,
                sum(record[key] for record in self.history),
                unit="J",
                kind="diagnostic",
            )
        result.metadata.update(
            {
                "maturity": "private_thermoelastic_procedure",
                "unit_system": "SI",
                "time": self.time,
                "accepted_steps": self.completed_steps,
                "reference_temperature": self.material.reference_temperature,
                "energy_semantics": "linearized_heat_and_quadratic_identity_not_general_first_law",
                "restart_source": deepcopy(self.restart_source),
                "history": deepcopy(self.history),
                "matrix_assemblies": {
                    "thermal": self.thermal.matrix_assembly_count,
                    "mechanical": self.mechanical.matrix_assembly_count,
                },
                "solve_counts": {
                    "thermal": self.thermal.solve_count,
                    "mechanical": self.mechanical.solve_count,
                },
                "limitations": [
                    "no_public_step_provider",
                    "no_general_energy_verification",
                    "constant_material_and_natural_loads",
                ],
            }
        )
        result.add_scientific_inputs(
            model=self.input_identity,
            time_path={"dt": self.dt, "steps": self.total_steps},
            iteration_policies=[record["iteration_policy"] for record in self.history],
        )
        return result

    @staticmethod
    def _history_digest(history):
        return sha256(
            json.dumps(history, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def checkpoint_capabilities(self):
        return checkpointing.CheckpointCapabilities(
            schemas=(checkpointing.TRANSIENT_CHECKPOINT_SCHEMA,),
            boundary="accepted_step",
            payload_scope="full_restart_state",
            state_components=(
                "displacement",
                "temperature_departure",
                "mechanical_reaction",
                "external_force",
                "accepted_time",
                "history",
            ),
            atomic_publication=True,
            rank_count_portability="requires_portable_policy",
            identity_scope=(
                "mesh",
                "element",
                "material",
                "declared_inputs",
                "boundary_histories",
                "time_path",
            ),
            limitations=(
                "Explicit scientific input identity is required.",
                "Constant material and natural loads; fixed strong-boundary DOF sets.",
                "Not a public coupled Step yet.",
            ),
        )

    def _checkpoint_arguments(self):
        boundary = (self.completed_steps, self._active, self._closed)
        if any(item != boundary for item in self.comm.allgather(boundary)):
            raise ValueError(
                "AFM-COUPLING-003: checkpoint boundary differs across ranks."
            )
        if self._active or self._closed:
            raise RuntimeError(
                "AFM-COUPLING-004: checkpoint requires an open accepted boundary."
            )
        if not self.input_identity["declared_inputs"]:
            raise ValueError(
                "AFM-COUPLING-006: checkpoint requires explicit scientific input identity."
            )
        return dict(
            step_kind="coupled_thermoelastic",
            step_name=self.name,
            procedure={"method": "staggered_backward_euler", "schema": 1},
            dt=self.dt,
            total_steps=self.total_steps,
            time_inputs={
                "kind": "time_input_plan",
                "restart_identity_bound": True,
                "identity": self.input_identity,
            },
        )

    def save_checkpoint(self, path, *, portable=True):
        return checkpointing.save_transient_checkpoint(
            path,
            **self._checkpoint_arguments(),
            completed_steps=self.completed_steps,
            state={
                "displacement": self.u_old,
                "temperature_departure": self.theta_old,
                "mechanical_reaction": self.energy.old_reaction,
                "external_force": self.energy.old_external,
            },
            accepted_times=[record["time"] for record in self.history],
            history_records=self.history,
            auxiliary_state={"history_sha256": self._history_digest(self.history)},
            portable=portable,
        )

    def load_checkpoint(self, path):
        staged = {
            "displacement": self.u_old.copy(),
            "temperature_departure": self.theta_old.copy(),
            "mechanical_reaction": self.energy.old_reaction.copy(),
            "external_force": self.energy.old_external.copy(),
        }
        metadata = checkpointing.load_transient_checkpoint(
            path, **self._checkpoint_arguments(), state=staged
        )
        count, history = metadata["completed_steps"], metadata["history_records"]
        if (
            type(count) is not int
            or not 0 <= count <= self.total_steps
            or len(history) != count
            or metadata["time"] != count * self.dt
            or metadata["accepted_times"] != [(i + 1) * self.dt for i in range(count)]
            or (metadata.get("auxiliary_state") or {}).get("history_sha256")
            != self._history_digest(history)
        ):
            raise ValueError(
                "AFM-COUPLING-005: inconsistent accepted checkpoint history."
            )
        for i, record in enumerate(history):
            if record["step"] != i + 1 or record["time"] != (i + 1) * self.dt:
                raise ValueError("AFM-COUPLING-005: inconsistent checkpoint time.")
        with (
            self._boundary_time(count * self.dt),
            state.field_transaction(
                u=self.u,
                t=self.theta,
                u_old=self.u_old,
                t_old=self.theta_old,
                reaction=self.energy.old_reaction,
                external=self.energy.old_external,
            ),
        ):
            for target, key in (
                (self.u.value, "displacement"),
                (self.u_old, "displacement"),
                (self.theta.value, "temperature_departure"),
                (self.theta_old, "temperature_departure"),
                (self.energy.old_reaction, "mechanical_reaction"),
                (self.energy.old_external, "external_force"),
            ):
                target.x.array[:] = staged[key].x.array
                target.x.scatter_forward()
        self.completed_steps, self.history = count, deepcopy(history)
        self.restart_source = {
            "manifest": metadata["manifest_path"],
            "mode": metadata["restart_mode"],
            "accepted_step": count,
        }
        return metadata

    def close(self):
        self._resources.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
