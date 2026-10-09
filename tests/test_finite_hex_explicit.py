# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private serial lifecycle acceptance; not a public finite-strain capability."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from agentfem import problems, state, time
from agentfem.diagnostics import MechanicalEnergyMonitor
from agentfem.mechanics._finite_hex_explicit import FiniteHexExplicitResidual
from test_finite_uniform_hex_dolfinx import setup


def make_step(bound=1e8):
    u, law, response, internal = setup()
    history = state.second_order_state(u)
    internal.displacement = history.u.value
    residual = FiniteHexExplicitResidual(internal, law, omega_squared_bound=bound)
    x = u.function_space.tabulate_dof_coordinates()
    history.v.value.x.array[:] = (1e-3 * x).ravel()
    step = problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=internal),
        residual=residual,
        history_monitor=MechanicalEnergyMonitor(mass=internal.mass_diagonal),
        dt=1e-5,
        steps=4,
        progress=False,
    )
    return step


def test_private_finite_hex_advances_existing_procedure_and_restarts(tmp_path):
    reference = make_step()
    reference.run()
    partial = make_step()
    partial.run(until_step=2)
    checkpoint = partial.save_checkpoint(tmp_path / "finite")
    resumed = make_step()
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    for name in ("u", "v", "a"):
        np.testing.assert_allclose(
            getattr(resumed.state, name).value.x.array,
            getattr(reference.state, name).value.x.array,
            rtol=0,
            atol=1e-15,
        )
    assert resumed.residual.snapshot() == reference.residual.snapshot()
    assert resumed.residual.accepted_time == pytest.approx(4e-5)
    assert resumed.residual.last_bound > 0


def test_spectral_rejection_restores_nodal_and_material_state():
    step = make_step(bound=1)
    before = step.state.snapshot()
    material_before = step.residual.snapshot()
    with pytest.raises(ValueError, match="spectral ceiling"):
        step.run()
    for name, value in before["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][name], value)
    assert step.residual.snapshot() == material_before
    assert step.completed_steps == 0


def test_trial_cannot_be_committed_at_another_time():
    step = make_step()
    residual = step.residual
    saved = residual.snapshot()
    residual.update_time(1e-5)
    vector = residual.assemble_vector()
    vector.destroy()
    residual.update_time(2e-5)
    with pytest.raises(RuntimeError, match="Time changed"):
        residual.commit()
    residual.restore(saved)
    assert residual.snapshot() == saved


def test_explicit_increment_uses_array_snapshot_not_json_lists(monkeypatch):
    step = make_step()

    def forbid_json_snapshot():
        raise AssertionError("Checkpoint JSON snapshot used during an increment")

    monkeypatch.setattr(step.residual, "snapshot", forbid_json_snapshot)
    step.run()
    record = step.residual.transaction_snapshot()
    assert isinstance(record["gradient"], np.ndarray)
    assert all(isinstance(value, np.ndarray) for value in record["fields"].values())


@pytest.mark.parametrize("corrupt", [False, True])
def test_restart_rollback_uses_array_snapshot(tmp_path, monkeypatch, corrupt):
    import json

    source = make_step()
    source.run(until_step=2)
    path = source.save_checkpoint(tmp_path / "array-rollback")
    if corrupt:
        record = json.loads(path.read_text())
        record["auxiliary_arrays"]["sha256"] = "0" * 64
        path.write_text(json.dumps(record))
    target = make_step()
    before = target.residual.snapshot()
    nodal = target.state.snapshot()
    original = target.residual.snapshot

    def forbid_json_snapshot():
        raise AssertionError("Restart rollback allocated a JSON snapshot")

    monkeypatch.setattr(target.residual, "snapshot", forbid_json_snapshot)
    if corrupt:
        with pytest.raises(ValueError):
            target.load_checkpoint(path)
        assert original() == before
        for name, value in nodal["fields"].items():
            np.testing.assert_array_equal(
                target.state.snapshot()["fields"][name], value
            )
    else:
        target.load_checkpoint(path)
        assert original() == source.residual.snapshot()
        assert target.completed_steps == 2


def test_material_cutback_request_rejects_increment_and_restores_state(monkeypatch):
    step = make_step()
    saved = step.residual.snapshot()
    original = step.residual.internal.response.update

    def request_cutback(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, suggested_time_scale=np.full(result.point_count, 0.5))

    monkeypatch.setattr(step.residual.internal.response, "update", request_cutback)
    with pytest.raises(ValueError, match="requested increment reduction"):
        step.run()
    assert step.completed_steps == 0
    assert step.residual.snapshot() == saved


def test_corrupt_auxiliary_record_is_rejected_before_assignment():
    step = make_step()
    step.run(until_step=2)
    saved = step.residual.snapshot()
    corrupt = deepcopy(saved)
    corrupt["fields"][next(iter(corrupt["fields"]))][0] = 99
    corrupt["fields"][next(reversed(corrupt["fields"]))] = [float("nan")]
    with pytest.raises(ValueError, match="checkpoint field"):
        step.residual.restore(corrupt)
    assert step.residual.snapshot() == saved


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_failure_after_material_commit_restores_entire_accepted_station(monkeypatch, exception):
    step = make_step()
    step.run(until_step=2)
    nodal = step.state.snapshot()
    material = step.residual.snapshot()
    original = step.residual.commit

    def fail_after_commit():
        original()
        raise exception("injected post-commit failure")

    monkeypatch.setattr(step.residual, "commit", fail_after_commit)
    with pytest.raises(exception, match="post-commit"):
        step.run()
    assert step.completed_steps == 2
    assert step.residual.snapshot() == material
    for name, value in nodal["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][name], value)
    monkeypatch.setattr(step.residual, "commit", original)
    step.run()
    reference = make_step()
    reference.run()
    assert step.residual.snapshot() == reference.residual.snapshot()
    np.testing.assert_array_equal(
        step.state.u.value.x.array, reference.state.u.value.x.array
    )


def test_changed_stability_identity_rejects_restart_without_modification(tmp_path):
    source = make_step()
    source.run(until_step=2)
    path = source.save_checkpoint(tmp_path / "identity")
    target = make_step(bound=2e8)
    before = target.state.snapshot()
    material = target.residual.snapshot()
    with pytest.raises(ValueError, match="identity"):
        target.load_checkpoint(path)
    assert target.residual.snapshot() == material
    for name, value in before["fields"].items():
        np.testing.assert_array_equal(target.state.snapshot()["fields"][name], value)


def test_binary_checkpoint_does_not_expand_material_arrays_to_json(
    tmp_path, monkeypatch
):
    import json
    from agentfem import checkpointing
    from mpi4py import MPI

    source = make_step()
    source.run(until_step=2)
    monkeypatch.setattr(
        source.residual, "snapshot", lambda: pytest.fail("JSON expansion")
    )
    path = source.save_checkpoint(tmp_path / "binary")
    metadata = json.loads(path.read_text())
    assert metadata["schema"] == "agentfem.transient-checkpoint.v6"
    assert source.checkpoints[-1].schema == metadata["schema"]
    payload = path.parent / metadata["auxiliary_arrays"]["path"]
    assert payload.is_file()
    assert source.checkpoint_capabilities().rank_count_portability == "unsupported"
    target = make_step()
    target.load_checkpoint(path)
    assert target.completed_steps == 2
    checkpointing._remove_transient_checkpoint(path, comm=MPI.COMM_SELF)
    assert not path.exists() and not payload.exists()


def test_existing_json_auxiliary_checkpoint_remains_readable(tmp_path, monkeypatch):
    import json

    source = make_step()
    source.run(until_step=2)
    monkeypatch.setattr(
        source.residual, "checkpoint_snapshot", source.residual.snapshot
    )
    path = source.save_checkpoint(tmp_path / "legacy-json")
    assert json.loads(path.read_text())["schema"] == "agentfem.transient-checkpoint.v5"
    target = make_step()
    target.load_checkpoint(path)
    assert target.residual.snapshot() == source.residual.snapshot()


def test_binary_checkpoint_rejects_portability_claim_before_publication(tmp_path):
    source = make_step()
    with pytest.raises(ValueError, match="serial same-partition"):
        source.save_checkpoint(tmp_path / "portable", portable=True)
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["shape", "nan", "path", "missing"])
def test_binary_auxiliary_rejection_preserves_all_state(tmp_path, failure):
    import json
    from hashlib import sha256

    source = make_step()
    source.run(until_step=2)
    path = source.save_checkpoint(tmp_path / "damaged")
    metadata = json.loads(path.read_text())
    descriptor = metadata["auxiliary_arrays"]
    payload = path.parent / descriptor["path"]
    if failure in {"shape", "nan"}:
        with np.load(payload, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data}
        key = next(iter(arrays))
        if failure == "shape":
            arrays[key] = arrays[key].reshape(-1)
        else:
            arrays[key].flat[0] = np.nan
        np.savez(payload, **arrays)
        descriptor["size"] = payload.stat().st_size
        descriptor["sha256"] = sha256(payload.read_bytes()).hexdigest()
    elif failure == "path":
        descriptor["path"] = "../outside.npz"
    else:
        descriptor["path"] = "damaged.missing.auxiliary.npz"
    path.write_text(json.dumps(metadata))
    target = make_step()
    before = target.state.snapshot()
    material = target.residual.snapshot()
    with pytest.raises((ValueError, FileNotFoundError)):
        target.load_checkpoint(path)
    assert target.completed_steps == 0
    assert target.residual.snapshot() == material
    for name, values in before["fields"].items():
        np.testing.assert_array_equal(target.state.snapshot()["fields"][name], values)


@pytest.mark.parametrize("where", ["auxiliary", "manifest"])
def test_failed_binary_publication_preserves_previous_checkpoint(
    tmp_path, monkeypatch, where
):
    from agentfem import checkpointing

    source = make_step()
    source.run(until_step=2)
    path = source.save_checkpoint(tmp_path / "atomic")
    before = path.read_bytes()
    files = set(tmp_path.iterdir())
    if where == "auxiliary":
        original = checkpointing.atomic_savez

        def fail(selected, **arrays):
            if str(selected).endswith(".auxiliary.npz"):
                raise OSError("injected auxiliary write failure")
            return original(selected, **arrays)

        monkeypatch.setattr(checkpointing, "atomic_savez", fail)
    else:

        def fail(*args, **kwargs):
            raise OSError("injected manifest publication failure")

        monkeypatch.setattr(checkpointing, "atomic_write_text", fail)
    with pytest.raises(RuntimeError, match="injected"):
        source.save_checkpoint(path)
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == files
    restored = make_step()
    restored.load_checkpoint(path)
    assert restored.residual.snapshot() == source.residual.snapshot()


def test_nonlinear_bar_matches_independent_ode_with_time_refinement():
    """Constrained one-cell bar: m_eff*u'' + C*log(1+u)/(1+u)=0."""
    from scipy.integrate import solve_ivp
    from mpi4py import MPI
    from dolfinx import mesh as dm
    from agentfem import constitutive, fields, models, studies
    from agentfem.constitutive.material_driver import MaterialQuadratureResponse
    from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual

    young, poisson, density = 100.0, 0.3, 2.0
    modulus = young * (1 - poisson) / ((1 + poisson) * (1 - 2 * poisson))
    effective_mass = density / 2
    duration = 0.1
    oracle = solve_ivp(
        lambda t, y: (y[1], -modulus * np.log1p(y[0]) / (1 + y[0]) / effective_mass),
        (0, duration),
        (0, 2),
        rtol=1e-12,
        atol=1e-13,
    )
    errors = []
    for dt in (0.002, 0.001, 0.0005):
        domain = dm.create_box(
            MPI.COMM_SELF,
            [[0, 0, 0], [1, 1, 1]],
            [1, 1, 1],
            cell_type=dm.CellType.hexahedron,
        )
        model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
        u = model.field(fields.displacement(domain))
        model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2))
        model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
        law = constitutive.finite_strain_j2_logarithmic(
            young=young, poisson=poisson, yield_stress=1e9
        )
        response = MaterialQuadratureResponse.create(
            domain,
            law.state_schema,
            degree=1,
            stored_energy_component_names=law.stored_energy_component_names,
        )
        history = state.second_order_state(u)
        internal = FiniteUniformHexResidual(
            history.u,
            response,
            density=density,
            hourglass_modulus=40,
            hourglass_scale=0.1,
        )
        residual = FiniteHexExplicitResidual(internal, law, omega_squared_bound=10000)
        x = u.value.function_space.tabulate_dof_coordinates()
        history.v.value.x.array[::3] = 2 * x[:, 0]
        step = problems.explicit_dynamics(
            state=history,
            integrator=time.explicit.central_difference(state=history, mass=internal),
            residual=residual,
            dt=dt,
            steps=round(duration / dt),
            prescribed=tuple(model.constraints),
            history_monitor=MechanicalEnergyMonitor(mass=internal.mass_diagonal),
            progress=False,
        )
        step.run()
        right = np.isclose(x[:, 0], 1)
        value = history.u.value.x.array[::3][right]
        np.testing.assert_allclose(value, value[0], atol=1e-13)
        errors.append(abs(value[0] - oracle.y[0, -1]))
        assert value[0] > 0.1  # finite, not an infinitesimal deformation example
    assert all(fine < coarse / 3.8 for coarse, fine in zip(errors, errors[1:]))
    assert errors[-1] < 2e-6
