# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Contact composition preserves numeric State without promising new physics."""

import copy
import json

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import boundary_models, checkpointing, time
from test_dolfinx_explicit_contact import _contact, _left_region
from test_finite_hex_material_envelope_step import prepare


def wrap(base, displacement, name):
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(displacement.function_space.mesh), displacement.function_space,
    )
    return boundary_models.dolfinx_explicit_contact_residual(
        base, adapter=adapter, displacement=displacement,
        projector=boundary_models.rigid_plane(point=(-1, 0, 0), normal=(1, 0, 0)),
        penalty=200, maximum_stable_time_increment=1e-4, name=name,
    )


def test_real_material_numeric_state_survives_two_contact_wrappers(tmp_path):
    step = prepare()
    step.run(until_step=3)
    bulk = step.residual
    residual = wrap(wrap(bulk, step.state.u.value, "inner"), step.state.u.value, "outer")
    record = residual.checkpoint_snapshot()
    numeric = record["base_state"]["base_state"]
    assert isinstance(numeric["gradient"], np.ndarray)
    assert all(isinstance(value, np.ndarray) for value in numeric["fields"].values())
    assert residual.checkpoint_capabilities().rank_count_portability == "unsupported"
    # The common archive, not the contact layer, owns numeric encoding.
    nodal = {"displacement": step.state.u, "velocity": step.state.v,
             "acceleration": step.state.a}
    arguments = dict(step_kind="explicit_dynamics_step", step_name="nested_state",
                     procedure=step.procedure, dt=step.dt, total_steps=step.steps,
                     state=nodal, time_inputs=time.input_summary(step.update_load))
    path = checkpointing.save_transient_checkpoint(
        tmp_path / "numeric", **arguments, completed_steps=3,
        auxiliary_state={"residual": record},
    )
    assert json.loads(path.read_text())["schema"] == "agentfem.transient-checkpoint.v6"
    restored = checkpointing.load_transient_checkpoint(path, **arguments)
    before = bulk.snapshot()
    bulk.accepted_gradient[:] = 42
    residual.restore(restored["auxiliary_state"]["residual"])
    assert bulk.snapshot() == before
    with pytest.raises(ValueError, match="same-partition"):
        checkpointing.save_transient_checkpoint(
            tmp_path / "portable", **arguments, completed_steps=3,
            auxiliary_state={"residual": record}, portable=True,
        )


def test_numeric_transaction_does_not_call_legacy_bulk_json_snapshot(monkeypatch):
    step = prepare()
    residual = wrap(step.residual, step.state.u.value, "contact")

    def reject():
        raise AssertionError("numeric material State must not expand to JSON")

    monkeypatch.setattr(step.residual, "snapshot", reject)
    for method in (residual.transaction_snapshot, residual.checkpoint_snapshot):
        assert isinstance(method()["base_state"]["gradient"], np.ndarray)


def test_outer_corruption_is_rejected_before_bulk_state_changes():
    step = prepare()
    residual = wrap(step.residual, step.state.u.value, "contact")
    record = residual.transaction_snapshot()
    before = step.residual.snapshot()
    corrupt = copy.deepcopy(record)
    corrupt["name"] = "another-contact"
    corrupt["base_state"]["gradient"][:] = 2
    with pytest.raises(ValueError, match="identity"):
        residual.restore(corrupt)
    assert step.residual.snapshot() == before


def test_stateless_contact_keeps_legacy_snapshot_and_portable_contract():
    _, _, residual = _contact(MPI.COMM_SELF)
    assert residual.checkpoint_snapshot() == residual.snapshot()
    assert residual.transaction_snapshot() == residual.snapshot()
    assert residual.checkpoint_capabilities().rank_count_portability == "requires_portable_policy"


def test_undeclared_numeric_provider_cannot_inherit_contact_portability():
    _, _, residual = _contact(MPI.COMM_SELF)

    class Undeclared:
        def checkpoint_snapshot(self):
            return {"history": np.zeros(1)}

    residual.base = Undeclared()
    with pytest.raises(ValueError, match="explicit checkpoint capability"):
        residual.checkpoint_capabilities()
