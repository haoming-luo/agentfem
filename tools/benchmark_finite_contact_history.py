# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Development benchmark: PYTHONPATH=src:tests python tools/benchmark_finite_contact_history.py.

Reference mode reinstates full work-history integration/JSON rollback snapshots;
both modes use identical material/mesh/load/integrator and complete history.
"""

import json
from contextlib import ExitStack
from time import perf_counter
from unittest.mock import patch

import numpy as np

from agentfem.boundary_models.contact_work import PrescribedContactWorkState
from agentfem.boundary_models.dolfinx_explicit_contact import DolfinxExplicitContactResidual
from test_finite_hex_contact_composition import prepare_contact


def run(reference):
    step = prepare_contact(public=True, duration=0.1)
    with ExitStack() as context:
        if reference:
            context.enter_context(patch.object(
                PrescribedContactWorkState, "path_work",
                property(lambda self: self._integrated_work(self.accepted)),
            ))
            context.enter_context(patch.object(
                DolfinxExplicitContactResidual, "transaction_snapshot",
                DolfinxExplicitContactResidual.checkpoint_snapshot,
            ))
        start = perf_counter()
        step.run()
        elapsed = perf_counter() - start
    return elapsed, step


def main():
    durations = {"reference": [], "optimized": []}
    reference = None
    for repeat in range(3):
        for legacy in ((True, False) if repeat % 2 == 0 else (False, True)):
            elapsed, step = run(legacy)
            durations["reference" if legacy else "optimized"].append(elapsed)
            if reference is None:
                reference = step
            else:
                np.testing.assert_array_equal(step.state.u.value.x.array, reference.state.u.value.x.array)
                np.testing.assert_array_equal(step.state.v.value.x.array, reference.state.v.value.x.array)
                assert step.material_residual.snapshot() == reference.material_residual.snapshot()
                assert step.history_records == reference.history_records
    old, new = (float(np.median(durations[key])) for key in ("reference", "optimized"))
    print(json.dumps({"increments": 1000, "cells": 12, "seconds": durations,
                      "reference_median_seconds": old, "optimized_median_seconds": new,
                      "speedup": old / new, "elapsed_reduction": 1 - new / old,
                      "compared_arrays_and_histories_identical": True}, indent=2))


if __name__ == "__main__":
    main()
