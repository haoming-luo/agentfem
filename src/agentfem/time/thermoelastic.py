# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Private bounded thermoelastic Procedure, before public Step promotion.

The benchmark owns its independent oracle. This execution object owns only
the physical participants and one accepted time coordinate. No mesh creation
or reference solution belongs here.
"""

from contextlib import ExitStack
from copy import deepcopy
from math import isfinite
from hashlib import sha256
import json

from .. import checkpointing, results, solvers, state
from ..operators.thermoelastic import _thermoelastic_blocks
from ..provenance import collective_call, collective_canonical_record
from ._staggered import StaggeredFieldIteration


class _ThermoelasticStep:
    """Same-mesh 3D homogeneous small-strain physics with fixed inputs.

    Temperature is explicitly a departure from material reference temperature.
    Joint fixed-input restart is available with explicit scientific identity.
    Boundary/time-input and general work evidence must be completed before
    registering this private executable as a built-in public provider.
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
        self.input_identity = collective_canonical_record(
            {"declared_inputs": input_identity, "material": material.as_dict()},
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
        try:
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
        except BaseException:
            self.close()
            raise

    @property
    def time(self):
        return self.completed_steps * self.dt

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
            with state.field_transaction(displacement=self.u, temperature=self.theta):
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
        result.metadata.update(
            {
                "maturity": "private_fixed_input_procedure",
                "time": self.time,
                "accepted_steps": self.completed_steps,
                "reference_temperature": self.material.reference_temperature,
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
                    "fixed_inputs_only",
                ],
            }
        )
        return result

    @staticmethod
    def _history_digest(history):
        return sha256(
            json.dumps(history, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

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
            state={"displacement": self.u_old, "temperature_departure": self.theta_old},
            accepted_times=[record["time"] for record in self.history],
            history_records=self.history,
            auxiliary_state={"history_sha256": self._history_digest(self.history)},
            portable=portable,
        )

    def load_checkpoint(self, path):
        staged = {
            "displacement": self.u_old.copy(),
            "temperature_departure": self.theta_old.copy(),
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
        with state.field_transaction(
            u=self.u, t=self.theta, u_old=self.u_old, t_old=self.theta_old
        ):
            for target, key in (
                (self.u.value, "displacement"),
                (self.u_old, "displacement"),
                (self.theta.value, "temperature_departure"),
                (self.theta_old, "temperature_departure"),
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
