"""Metadata publication must neither silently lose keys nor recurse on NumPy scalars."""

import json
from pathlib import Path

import numpy as np
import pytest

from agentfem.results import CheckpointRecord, SimulationResult


@pytest.mark.parametrize("container", [lambda x: x, lambda x: [x], lambda x: np.array([x])])
@pytest.mark.parametrize("value", ["1.25", "nan", "inf"])
def test_extended_scalar_has_addressable_conversion_failure(container, value):
    scalar = np.longdouble(value)
    if not isinstance(scalar.item(), type(scalar)):
        pytest.skip("This NumPy platform supplies a Python scalar conversion.")
    result = SimulationResult("extended", metadata={"measurement": container(scalar)})
    with pytest.raises(ValueError, match="no lossless JSON conversion"):
        result.manifest()


def test_metadata_key_collision_is_not_silent_data_loss():
    result = SimulationResult("collision", metadata={"labels": {1: "first", "1": "second"}})
    with pytest.raises(ValueError, match="keys collide"):
        result.manifest()


def test_supported_numpy_metadata_preserves_numeric_json_values():
    result = SimulationResult("numpy", metadata={
        "integer": np.int64(7), "real": np.float32(1.25), "flag": np.bool_(True),
        "array": np.array([[1, 2]], dtype=np.int32),
    })
    metadata = json.loads(json.dumps(result.manifest(), allow_nan=False))["metadata"]
    assert metadata == {"integer": 7, "real": 1.25, "flag": True, "array": [[1, 2]]}


def test_invalid_metadata_preserves_existing_manifest(tmp_path):
    path = tmp_path / "result.json"
    result = SimulationResult("valid", metadata={"measurement": 1.25})
    result.write_manifest(path)
    previous = path.read_bytes()
    result.metadata["labels"] = {1: "first", "1": "second"}
    with pytest.raises(ValueError, match="keys collide"):
        result.write_manifest(path)
    assert path.read_bytes() == previous
    assert not path.with_suffix(".json.tmp").exists()


def test_checkpoint_sidecar_failure_keeps_previous_and_cleans_temporary(tmp_path, monkeypatch):
    checkpoint = CheckpointRecord(
        name="accepted", path=tmp_path / "state.npz", schema="test.state.v1",
        step_name="loading", coordinate_name="time", coordinate_value=1.0,
    )
    path = checkpoint.write_manifest()
    original = path.read_bytes()
    replace = Path.replace

    def fail_replace(source, target):
        if Path(target) == path:
            raise PermissionError("injected sidecar replacement failure")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError, match="sidecar replacement"):
        checkpoint.write_manifest()
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
