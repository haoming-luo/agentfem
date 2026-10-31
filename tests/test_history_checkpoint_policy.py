"""Automatic restart belongs to accepted structural history, not project glue."""

import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from agentfem import checkpointing, steps
from test_finite_strain_material_step import build


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_policy_retains_only_own_scheduled_states_and_restores(tmp_path, mechanism):
    policy = checkpointing.every(1, directory=tmp_path, keep_last=2)
    step, _ = build(mechanism=mechanism, checkpoint=policy)
    step.solve(until=0.25)
    manual = step.save_checkpoint(tmp_path / "manual")
    unrelated = tmp_path / "another-run-inc-00000001.npz"
    unrelated.write_bytes(b"unrelated")
    result = step.solve_result()
    scheduled = [
        r for r in step.checkpoints if r.metadata.get("role") == "scheduled_checkpoint"
    ]
    assert [r.metadata["accepted_increment"] for r in scheduled] == [3, 4]
    assert manual.exists() and unrelated.read_bytes() == b"unrelated"
    assert result.checkpoint_contract["status"] == "accepted"
    assert result.checkpoint_contract["policy"]["keep_last"] == 2
    for record in scheduled:
        assert record.path.exists()
        manifest = record.path.with_suffix(".npz.checkpoint.json")
        assert (
            json.loads(manifest.read_text())["metadata"]["role"]
            == "scheduled_checkpoint"
        )
    restored, _ = build(mechanism=mechanism)
    restored.load_checkpoint(scheduled[0].path)
    restored.solve()
    np.testing.assert_array_equal(restored.solution.x.array, step.solution.x.array)
    np.testing.assert_array_equal(
        restored.state.committed_state_vectors(), step.state.committed_state_vectors()
    )


def test_pause_final_and_restart_cadence_use_global_accepted_count(tmp_path):
    first, _ = build(checkpoint=checkpointing.every(3, directory=tmp_path / "first"))
    first.solve(until=0.5)
    assert [r.metadata["accepted_increment"] for r in first.checkpoints] == [2]
    saved = first.checkpoints[-1].path
    second, _ = build(checkpoint=checkpointing.every(3, directory=tmp_path / "second"))
    second.load_checkpoint(saved)
    second.solve()
    assert [r.metadata["accepted_increment"] for r in second.checkpoints] == [3, 4]
    assert saved.exists()


def test_final_false_does_not_save_unscheduled_pause(tmp_path):
    step, _ = build(checkpoint=checkpointing.every(3, directory=tmp_path, final=False))
    step.solve(until=0.5)
    assert not step.checkpoints and not list(tmp_path.iterdir())
    step.solve()
    assert [r.metadata["accepted_increment"] for r in step.checkpoints] == [3]


def test_failed_write_keeps_last_good_checkpoint_and_accepted_state(
    tmp_path, monkeypatch
):
    step, _ = build(checkpoint=checkpointing.every(1, directory=tmp_path, keep_last=1))
    original = checkpointing._atomic_savez_exclusive

    def fail_second(path, **arrays):
        if "00000002" in str(path):
            raise OSError("injected checkpoint disk full")
        return original(path, **arrays)

    monkeypatch.setattr(checkpointing, "_atomic_savez_exclusive", fail_second)
    with pytest.raises(OSError, match="disk full"):
        step.solve()
    assert step.accepted_load_factor == 0.5
    assert len(step.attempted_increments) == 2
    assert all(r["accepted"] for r in step.attempted_increments)
    assert len(step.checkpoints) == 1 and step.checkpoints[0].path.exists()
    restarted, _ = build()
    restarted.load_checkpoint(step.checkpoints[0].path)
    restarted.solve()
    whole, _ = build()
    whole.solve()
    np.testing.assert_array_equal(
        restarted.state.committed_state_vectors(), whole.state.committed_state_vectors()
    )


def test_output_failure_has_already_saved_same_accepted_state(tmp_path, monkeypatch):
    from agentfem.results._live_xdmf import LiveXDMFTimeSeries

    def fail(*args, **kwargs):
        raise OSError("injected field output failure")

    step, _ = build(checkpoint=checkpointing.every(1, directory=tmp_path / "restart"))
    monkeypatch.setattr(LiveXDMFTimeSeries, "_store_frame", fail)
    with pytest.raises(OSError, match="field output"):
        step.solve_result(output=tmp_path / "results.xdmf")
    assert step.accepted_load_factor == 0.25
    restored, _ = build()
    restored.load_checkpoint(step.checkpoints[-1].path)
    assert restored.accepted_load_factor == 0.25


def test_scheduled_collision_never_overwrites_existing_payload(tmp_path):
    policy = checkpointing.every(1, directory=tmp_path)
    step, _ = build(checkpoint=policy)
    path = policy.path(step_name=step.name, increment=1).with_suffix(".npz")
    path.write_bytes(b"existing scientific evidence")
    with pytest.raises(FileExistsError, match="checkpoint"):
        step.solve()
    assert path.read_bytes() == b"existing scientific evidence"
    assert step.accepted_load_factor == 0.25


def test_portable_policy_is_rejected_before_execution(tmp_path):
    with pytest.raises((ValueError, NotImplementedError), match="portable|partition"):
        build(checkpoint=checkpointing.every(1, directory=tmp_path, portable=True))
    assert not list(tmp_path.iterdir())


def test_adaptive_checkpoint_failures_do_not_cut_back(tmp_path, monkeypatch):
    step, _ = build(
        checkpoint=checkpointing.every(1, directory=tmp_path),
        incrementation=steps.automatic(initial=0.25, minimum=0.01, maximum=0.25),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("injected archive failure")

    monkeypatch.setattr(checkpointing, "_atomic_savez_exclusive", fail)
    with pytest.raises(RuntimeError, match="archive failure"):
        step.solve()
    assert len(step.attempted_increments) == 1
    assert step.attempted_increments[0]["accepted"]
    assert step.accepted_load_factor == 0.25


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_automatic_plane_stress_restart_preserves_thickness_and_history(
    tmp_path, mechanism
):
    from test_history_plane_stress import structure

    step, _ = structure(
        mechanism=mechanism,
        checkpoint=checkpointing.every(3, directory=tmp_path, keep_last=2),
    )
    step.solve(until=0.5)
    saved = step.checkpoints[-1].path
    resumed, _ = structure(mechanism=mechanism)
    resumed.load_checkpoint(saved)
    np.testing.assert_array_equal(resumed._accepted_F, step._accepted_F)
    resumed.solve()
    whole, _ = structure(mechanism=mechanism)
    whole.solve()
    np.testing.assert_array_equal(resumed._accepted_F, whole._accepted_F)
    np.testing.assert_array_equal(
        resumed.state.committed_state_vectors(), whole.state.committed_state_vectors()
    )


def test_sidecar_failure_preserves_payload_and_does_not_prune(tmp_path, monkeypatch):
    step, _ = build(checkpoint=checkpointing.every(1, directory=tmp_path, keep_last=1))
    step.solve(until=0.25)
    good = step.checkpoints[-1].path
    original = checkpointing.atomic_write_text

    def fail(path, text):
        if "00000002" in str(path):
            raise OSError("injected sidecar failure")
        return original(path, text)

    monkeypatch.setattr(checkpointing, "atomic_write_text", fail)
    with pytest.raises(OSError, match="sidecar failure"):
        step.solve()
    assert good.exists()
    assert len(step.checkpoints) == 2
    last = step.checkpoints[-1]
    assert last.metadata.get("role") != "scheduled_checkpoint"
    restored, _ = build()
    restored.load_checkpoint(last.path)
    assert restored.accepted_load_factor == 0.5
    # Failed discovery metadata must not poison a later retention attempt.
    monkeypatch.setattr(checkpointing, "atomic_write_text", original)
    step.solve()
    assert last.path.exists()


def test_numerical_rejection_never_schedules_trial_state(tmp_path):
    step, material = build(checkpoint=checkpointing.every(1, directory=tmp_path))
    step.solve(until=0.5)
    before = tuple(tmp_path.iterdir())

    def fail(point):
        raise RuntimeError("injected local rejection")

    material.update = fail
    with pytest.raises(RuntimeError, match="rejected increment"):
        step.solve()
    assert tuple(tmp_path.iterdir()) == before
    assert step.accepted_load_factor == 0.5
    assert len(step.checkpoints) == 2
    restored, _ = build()
    restored.load_checkpoint(step.checkpoints[-1].path)
    assert restored.accepted_load_factor == 0.5


def test_checkpoint_precedes_failed_reporter(tmp_path):
    class BrokenReporter:
        def emit(self, event):
            if event.kind == "increment_converged":
                raise OSError("injected reporter failure")

    step, _ = build(
        checkpoint=checkpointing.every(1, directory=tmp_path), progress=BrokenReporter()
    )
    with pytest.raises(OSError, match="reporter failure"):
        step.solve()
    restored, _ = build()
    restored.load_checkpoint(step.checkpoints[-1].path)
    assert restored.accepted_load_factor == 0.25


def test_retention_failure_never_destroys_new_checkpoint(tmp_path, monkeypatch):
    step, _ = build(checkpoint=checkpointing.every(1, directory=tmp_path, keep_last=1))
    step.solve(until=0.25)
    old = step.checkpoints[-1].path

    def fail(*args, **kwargs):
        raise OSError("injected retention permission failure")

    monkeypatch.setattr(checkpointing, "remove_serial_checkpoint", fail)
    with pytest.raises(OSError, match="retention permission"):
        step.solve()
    assert old.exists() and len(step.checkpoints) == 2
    restored, _ = build()
    restored.load_checkpoint(step.checkpoints[-1].path)
    assert restored.accepted_load_factor == 0.5
    assert len(step.attempted_increments) == 2


def test_material_cutback_counts_only_accepted_increments(tmp_path):
    from dataclasses import replace
    from test_finite_strain_material_step import HistoryMaterial

    class Limited(HistoryMaterial):
        def update(self, point):
            response = super().update(point)
            return replace(response, suggested_time_scale=0.5 if point.time_increment > 0.3 else 1.0)

    control = steps.automatic(initial=0.5, minimum=0.001, maximum=0.5)
    step, _ = build(material=Limited(), incrementation=control,
                    checkpoint=checkpointing.every(2, directory=tmp_path))
    step.solve()
    assert any(not r["accepted"] for r in step.attempted_increments)
    for record in step.checkpoints:
        with np.load(record.path) as data:
            ledger = json.loads(str(data["ledger"].item()))
        count = len(ledger["accepted"])
        assert record.metadata["accepted_increment"] == count
        assert count % 2 == 0 or ledger["coordinate"] == 1.0
        assert ledger["accepted"][-1] == ledger["attempted"][-1]
    reference, _ = build(material=Limited(), incrementation=control)
    reference.solve()
    assert reference.accepted_increments == step.accepted_increments
    np.testing.assert_array_equal(reference.state.committed_state_vectors(), step.state.committed_state_vectors())


def test_dotted_prefix_keeps_increment_identity(tmp_path):
    step, _ = build(
        checkpoint=checkpointing.every(1, directory=tmp_path, prefix="plate.v2")
    )
    step.solve()
    assert [record.path.name for record in step.checkpoints] == [
        f"plate.v2-inc-{i:08d}.npz" for i in range(1, 5)
    ]


@pytest.mark.parametrize("name", ["every", "keep_last"])
@pytest.mark.parametrize(
    "value",
    [True, False, np.bool_(True), 1.5, 2.0, "2", 0, -2, float("nan"), float("inf")],
)
def test_policy_counts_never_silently_truncate(name, value, tmp_path):
    kwargs = {"every": 2, "directory": tmp_path, name: value}
    with pytest.raises(ValueError, match="positive integer"):
        checkpointing.CheckpointPolicy(**kwargs)


@pytest.mark.parametrize(
    "prefix", ["", " ", ".", "..", "../run", "a/b", "a\\b", "a\x00b"]
)
def test_policy_prefix_is_not_a_path(prefix, tmp_path):
    with pytest.raises(ValueError, match="filename prefix"):
        checkpointing.every(2, directory=tmp_path, prefix=prefix)


@pytest.mark.parametrize("name", ["final", "portable"])
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_policy_flags_are_explicit_booleans(name, value, tmp_path):
    with pytest.raises(ValueError, match="boolean"):
        checkpointing.every(2, directory=tmp_path, **{name: value})


def test_numpy_integer_cadence_is_supported(tmp_path):
    policy = checkpointing.every(np.int64(3), directory=tmp_path, keep_last=np.int32(2))
    assert policy.every == 3 and policy.keep_last == 2


def test_exclusive_archive_publication_refuses_racing_writer(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    gate = Barrier(2)
    path = tmp_path / "state.npz"

    def publish(value):
        gate.wait(timeout=5)
        try:
            checkpointing._atomic_savez_exclusive(path, state=np.full(100, value))
        except FileExistsError:
            return None
        return value

    with ThreadPoolExecutor(max_workers=2) as pool:
        winners = [value for value in pool.map(publish, [1, 2]) if value is not None]
    assert len(winners) == 1
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive["state"], np.full(100, winners[0]))
    assert list(tmp_path.iterdir()) == [path]


def test_exclusive_publication_failure_preserves_archive_and_cleans_temporary(tmp_path):
    path = tmp_path / "state.npz"
    checkpointing.atomic_savez(path, value=np.array([7]))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        checkpointing._atomic_savez_exclusive(path, value=np.array([8]))
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_manual_atomic_archive_retains_replacement_and_payload_keyword_compatibility(tmp_path):
    path = tmp_path / "manual.npz"
    checkpointing.atomic_savez(path, overwrite=np.array([1]))
    checkpointing.atomic_savez(path, overwrite=np.array([2]))
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive["overwrite"], [2])


def test_bound_policy_does_not_drift_from_execution_contract(tmp_path):
    policy = checkpointing.every(2, directory=tmp_path)
    step, _ = build(checkpoint=policy)
    assert step.checkpoint_policy == step.execution_context.policy.checkpoint
    with pytest.raises(AttributeError):
        step.checkpoint_policy = checkpointing.every(1, directory=tmp_path)


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_restart_from_older_checkpoint_checks_overlapping_output(tmp_path, mechanism):
    from agentfem import results

    first, _ = build(mechanism=mechanism, checkpoint=checkpointing.every(
        2, directory=tmp_path / "checkpoints", final=False))
    a = tmp_path / "first.xdmf"
    first.solve_result(until=0.75, output=a)
    saved = first.checkpoints[-1]
    assert saved.coordinate_value == 1.0  # output has already reached 1.5 s
    resumed, _ = build(mechanism=mechanism)
    resumed.load_checkpoint(saved.path)
    b = tmp_path / "second.xdmf"
    resumed.solve_result(output=b)
    with pytest.raises(ValueError, match="overlap"):
        results.join_xdmf_segments([a, b], tmp_path / "strict.xdmf")
    joined = results.join_xdmf_segments([a, b], tmp_path / "joined.xdmf",
                                      overlap="identical", expected_times=[0.5, 1., 1.5, 2.])
    assert [float(t.get("Value")) for t in ET.parse(joined).findall(".//Time")] == [0.5, 1., 1.5, 2.]
    first.solve()
    np.testing.assert_array_equal(first.state.committed_state_vectors(), resumed.state.committed_state_vectors())


def test_example_preflight_rejects_old_class_without_checkpoint_keyword(
    tmp_path, monkeypatch
):
    from agentfem import mechanics, project
    import shutil

    source = Path(__file__).resolve().parents[1] / "examples/history_material"
    for name in ("case.py", "material.py", "README.md", "agentfem.toml"):
        shutil.copy2(source / name, tmp_path / name)

    class OldStep:
        def __init__(self, *, displacement, material):
            pass

    monkeypatch.setattr(mechanics, "FiniteStrainMaterialStep", OldStep)
    config = project.ProjectConfig.load(tmp_path)
    assert any("checkpoint" in error for error in config.check())


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_process_exit_between_checkpoint_and_field_publication_is_recoverable(
    tmp_path, mechanism
):
    from agentfem import results

    # This child exits without context-manager cleanup, after durable acceptance
    # but before that station's field output. It is not a graceful pause test.
    program = """
import os
from agentfem import checkpointing
from test_finite_strain_material_step import build
class ExitAtSecondAccepted:
    def emit(self, event):
        if event.kind == 'increment_converged' and event.increment == 2:
            os._exit(23)
step, _ = build(mechanism=MECHANISM,
    checkpoint=checkpointing.every(2, directory='checkpoints'),
    progress=ExitAtSecondAccepted())
step.solve_result(output='interrupted.xdmf')
""".replace("MECHANISM", repr(mechanism))
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root / "tests"))),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    child = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert child.returncode == 23, child.stdout + child.stderr
    interrupted = tmp_path / "interrupted.xdmf"
    assert [
        float(t.get("Value")) for t in ET.parse(interrupted).findall(".//Time")
    ] == [0.5]
    archives = list((tmp_path / "checkpoints").glob("*.npz"))
    assert len(archives) == 1
    resumed, _ = build(mechanism=mechanism)
    resumed.load_checkpoint(archives[0])
    second = tmp_path / "resumed.xdmf"
    resumed.solve_result(output=second)
    assert [float(t.get("Value")) for t in ET.parse(second).findall(".//Time")] == [
        1.0,
        1.5,
        2.0,
    ]
    whole, _ = build(mechanism=mechanism)
    full = tmp_path / "whole.xdmf"
    whole.solve_result(output=full)
    np.testing.assert_array_equal(
        resumed.state.committed_state_vectors(), whole.state.committed_state_vectors()
    )
    joined = results.join_xdmf_segments(
        [interrupted, second],
        tmp_path / "joined.xdmf",
        expected_times=[0.5, 1.0, 1.5, 2.0],
    )
    assert len(ET.parse(joined).findall(".//Time")) == 4
    import h5py

    def arrays(path):
        for frame in ET.parse(path).findall(".//Grid[@GridType='Uniform']"):
            for attribute in frame.findall("Attribute"):
                filename, dataset = (
                    attribute.find("DataItem").text.strip().split(":", 1)
                )
                with h5py.File(path.parent / filename) as data:
                    yield attribute.get("Name"), data[dataset][:]

    for (name, actual), (reference_name, expected) in zip(
        arrays(joined), arrays(full), strict=True
    ):
        assert name == reference_name
        np.testing.assert_array_equal(actual, expected)
