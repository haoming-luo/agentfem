# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded serial dyadic retries, retaining the existing explicit transaction."""
from dataclasses import dataclass
import copy
import numpy as np


@dataclass(frozen=True)
class ExplicitSubstepPolicy:
    maximum_depth: int = 8
    material_cutback: bool = False

    def __post_init__(self):
        if (isinstance(self.maximum_depth, bool) or not isinstance(self.maximum_depth, int)
                or not 1 <= self.maximum_depth <= 8 or not isinstance(self.material_cutback, bool)):
            raise ValueError("Substeps require depth in [1,8] and an explicit material_cutback bool")

    def summary(self):
        return {"kind": "bounded_dyadic_explicit_retry", "maximum_depth": self.maximum_depth,
                "material_cutback": self.material_cutback, "mpi": False,
                "stability_certificate": False, "inertia_cadence": "once_per_macro_increment"}


class _ExplicitSubsteps:
    def __init__(self, step, advance, policy):
        from .._contact_path_guard import ContactPathResolutionError
        from ..constitutive._explicit_material import ExplicitMaterialIncrementExceeded, ExplicitMaterialCutbackRequest
        from ..fields import unwrap
        if unwrap(step.state.u).function_space.mesh.comm.size != 1:
            raise NotImplementedError("Explicit substeps require serial execution")
        if step.history_every != 1:
            raise NotImplementedError("Substeps require the accepted energy ledger at each macro station")
        if not callable(getattr(step.history_monitor, "restore", None)):
            raise TypeError("Substeps require a restorable energy ledger")
        self.step, self.original, self.policy = step, advance, policy
        self.retry_errors = (ContactPathResolutionError,)
        if policy.material_cutback:
            self.retry_errors += (ExplicitMaterialIncrementExceeded, ExplicitMaterialCutbackRequest)
        self.events, self.rejection_counts = [], {}

    def _reject(self, error):
        name = type(error).__name__
        self.rejection_counts[name] = self.rejection_counts.get(name, 0) + 1

    def advance(self, end):
        from .._transient_problems import _record_transient_history, _explicit_inertia_state_owner
        step = self.step
        try:
            return self.original(end)
        except self.retry_errors as error:
            self._reject(error)
        start = step.completed_steps * step.dt
        state = step.state.snapshot()
        residual = step.residual.transaction_snapshot() if hasattr(step.residual, "transaction_snapshot") else step.residual.snapshot()
        ledger = copy.deepcopy(step.history_records[-1])
        if not np.isclose(ledger["time"], start, rtol=0, atol=4*np.spacing(max(start, step.dt))):
            raise RuntimeError("Substep ledger is not the accepted macro station")
        previous = step.integrator.last_residual_owned
        inertia = _explicit_inertia_state_owner(step)
        inertia_state = None if inertia is None else inertia.snapshot()
        accepted = []

        def interval(a, b, depth):
            try:
                self.original(b, increment_size=b-a)
            except self.retry_errors as error:
                self._reject(error)
                if depth >= self.policy.maximum_depth:
                    raise
                middle = a + (b-a)/2
                interval(a, middle, depth+1)
                interval(middle, b, depth+1)
            else:
                # Only the physical work ledger consumes accepted subincrements.
                # Pure user observations are evaluated by the ordinary Procedure
                # once the complete macro increment has succeeded. A failed
                # branch therefore cannot publish an observation/history row.
                _record_transient_history(step, b, store=False)
                accepted.append({"start": a, "end": b, "dt": b-a, "depth": depth})

        try:
            if inertia is not None:
                inertia.prepare_increment(step.state, increment=step.completed_steps+1, time=start, dt=step.dt)
            middle = start + (end-start)/2
            interval(start, middle, 1)
            interval(middle, end, 1)
        except BaseException:
            if inertia is not None:
                inertia.restore(inertia_state)
            step.state.restore(state)
            step.residual.restore(residual)
            step.integrator.last_residual_owned = previous
            for callback in (step.update_load, getattr(step.residual, "update_time", None)):
                if callable(callback):
                    callback(start)
            for prescribed in step.prescribed:
                if callable(getattr(prescribed, "update", None)):
                    prescribed.update(start)
            step.history_monitor.restore(ledger)
            raise
        self.events.append({"macro_increment": step.completed_steps+1, "subincrements": accepted})

    def summary(self):
        return {**self.policy.summary(), "statistics_scope": "current_process_only",
                "rejection_counts": dict(self.rejection_counts), "events": list(self.events)}
