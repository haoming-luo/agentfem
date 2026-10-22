# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Partition-invariant geometry execution is not a physical model parameter."""
from contextlib import ExitStack

import numpy as np
import pytest

from agentfem.boundary_models import Q1ContactExecution, q1_contact_runtime
from test_deformable_contact_explicit import make_step


@pytest.mark.parametrize("old,new", [(1, 4), (4, 2), (2, 1)])
def test_native_q1_worker_restart_preserves_fields_history_and_provenance(tmp_path, old, new):
    with ExitStack() as owners:
        a = owners.enter_context(q1_contact_runtime(backend="native", projection_workers=old))
        b = owners.enter_context(q1_contact_runtime(backend="native", projection_workers=new))
        assert a.execution.summary() != b.execution.summary()
        assert a.execution.compatible_checkpoint_summary() == b.execution.compatible_checkpoint_summary()
        continuous = make_step(closed=True, execution=a.execution)
        continuous.run()
        partial = make_step(closed=True, execution=a.execution)
        partial.run(until_step=8)
        path = partial.save_checkpoint(tmp_path / "worker-portable")
        target = make_step(closed=True, execution=b.execution)
        assert target.residual.identity != partial.residual.identity
        target.load_checkpoint(path)
        target.run()
        for name in ("u", "v", "a"):
            np.testing.assert_array_equal(getattr(target.state, name).value.x.array,
                                          getattr(continuous.state, name).value.x.array)
        assert target.history_records == continuous.history_records
        assert target.residual.accepted[2].potential_energy > 0


def test_legacy_q1_checkpoint_requires_original_workers_then_explicit_resave(tmp_path, monkeypatch):
    with ExitStack() as owners:
        a = owners.enter_context(q1_contact_runtime(backend="native", projection_workers=1))
        b = owners.enter_context(q1_contact_runtime(backend="native", projection_workers=2))
        original = make_step(closed=True, execution=a.execution)
        original.run(until_step=8)
        old = original.residual.checkpoint_snapshot()
        old["schema"] = "agentfem.private-deformable-q1-contact.v1"
        old.pop("restart_identity")
        with monkeypatch.context() as patch:
            patch.setattr(original.residual, "checkpoint_snapshot", lambda: old)
            path = original.save_checkpoint(tmp_path / "legacy")
        changed = make_step(closed=True, execution=b.execution)
        with pytest.raises(ValueError, match="identity"):
            changed.load_checkpoint(path)
        assert changed.completed_steps == 0
        np.testing.assert_array_equal(changed.state.u.value.x.array, 0.)
        same = make_step(closed=True, execution=a.execution)
        same.load_checkpoint(path)
        upgraded = same.save_checkpoint(tmp_path / "upgraded")
        changed.load_checkpoint(upgraded)
        same.run()
        changed.run()
        np.testing.assert_array_equal(changed.state.u.value.x.array, same.state.u.value.x.array)
        assert changed.history_records == same.history_records


def test_q1_compatibility_declaration_is_opt_in_and_frozen():
    declaration = {"algorithm": {"revision": 1}}
    execution = Q1ContactExecution({"provider": "external", "workers": 1},
                                    restart_identity=declaration)
    expected = execution.compatible_checkpoint_summary()
    declaration["algorithm"]["revision"] = 2
    assert execution.compatible_checkpoint_summary() == expected
    strict = Q1ContactExecution({"provider": "external", "workers": 1})
    assert strict.compatible_checkpoint_summary() == strict.checkpoint_summary()
    with pytest.raises(TypeError, match="restart_identity"):
        Q1ContactExecution({"provider": "external"}, restart_identity={})


@pytest.mark.parametrize("change", ["provider", "revision", "arithmetic", "reuse"])
def test_q1_restart_rejects_changed_semantics_atomically(tmp_path, change):
    with q1_contact_runtime(backend="native") as runtime:
        old = make_step(closed=True, execution=runtime.execution)
        old.run(until_step=2)
        path = old.save_checkpoint(tmp_path / "accepted")
        from dataclasses import replace
        identity = runtime.execution.compatible_checkpoint_summary()["identity"]
        options = {}
        if change == "reuse":
            options["separation_reuse"] = "anchored"
        else:
            identity[change] = "different"
        execution = replace(runtime.execution, restart_identity=identity, **options)
        target = make_step(closed=True, execution=execution)
        before = target.residual.snapshot()
        with pytest.raises(ValueError, match="identity"):
            target.load_checkpoint(path)
        assert target.completed_steps == 0
        assert target.residual.snapshot() == before


@pytest.mark.parametrize("bad", [{"schema": "unknown"}, {"identity": "bad"},
                                  {"restart_identity": "0" * 64}])
def test_malformed_compatible_record_is_rejected_without_mutation(bad):
    step = make_step(closed=True)
    before = step.residual.snapshot()
    with pytest.raises(ValueError, match="identity"):
        step.residual.restore({**before, **bad})
    assert step.residual.snapshot() == before
