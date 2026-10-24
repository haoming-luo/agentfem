# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from agentfem import results, provenance


def histories():
    simulation = results.SimulationResult("history-archive")
    times = np.array([0., 1., np.nextafter(1., 2.), 2.])
    simulation.add_history("energy", times, [0., -0., 3., 5.], unit="J")
    simulation.add_history("force", times, np.arange(12.).reshape(4, 3), unit="N")
    simulation.add_history("empty", [], np.empty((0, 2)), unit="m")
    return simulation


def assert_same(left, right):
    assert left.as_dict(include_values=False) == right.as_dict(include_values=False)
    for name in ("abscissa", "values"):
        a, b = getattr(left, name), getattr(right, name)
        assert a.shape == b.shape
        assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("storage", ["inline", "npz"])
def test_lossless_roundtrip_and_integrity(tmp_path, storage):
    simulation = histories()
    path = simulation.write_manifest(tmp_path / "result.json", include_histories=True,
                                     history_storage=storage)
    loaded = results.read_histories(path)
    for original, restored in zip(simulation.histories.values(), loaded, strict=True):
        assert_same(original, restored)
    assert provenance.verify_manifest(path).verified
    assert "history_archive" not in simulation.artifacts
    record = json.loads(path.read_text())
    if storage == "npz":
        assert "values" not in record["history_records"][0]
        axis = [item["storage"]["abscissa"] for item in record["history_records"][:2]]
        assert axis[0] == axis[1]
        assert record["history_archive"]["array_count"] == 5
        assert set(record["artifacts"]) == {"history_archive"}


def test_old_manifest_survives_failed_republication(tmp_path, monkeypatch):
    simulation = histories()
    path = simulation.write_manifest(tmp_path / "result.json", include_histories=True,
                                     history_storage="npz")
    original = path.read_bytes()
    old_histories = results.read_histories(path)
    simulation.histories["energy"].values[:] = 42.
    replace = Path.replace

    def fail_manifest(source, destination):
        if source == path.with_suffix(".json.tmp"):
            raise OSError("injected publication failure")
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_manifest)
    with pytest.raises(RuntimeError, match="injected publication failure"):
        simulation.write_manifest(path, include_histories=True, history_storage="npz")
    assert path.read_bytes() == original
    assert provenance.verify_manifest(path).verified
    for a, b in zip(old_histories, results.read_histories(path), strict=True):
        assert_same(a, b)


@pytest.mark.parametrize("change", ["bytes", "missing", "shape", "traversal", "duplicate", "version"])
def test_bad_archive_is_rejected(tmp_path, change):
    path = histories().write_manifest(tmp_path / "result.json", include_histories=True,
                                      history_storage="npz")
    record = json.loads(path.read_text())
    companion = tmp_path / record["history_archive"]["artifact"]
    if change == "bytes":
        with companion.open("ab") as stream:
            stream.write(b"corrupt")
    elif change == "missing":
        companion.unlink()
    elif change == "shape":
        record["history_records"][0]["sample_count"] = 5
    elif change == "traversal":
        record["history_archive"]["artifact"] = "../elsewhere.npz"
    elif change == "duplicate":
        record["history_records"][1]["name"] = "energy"
    else:
        record["history_archive"]["schema_version"] = "99"
    # Exercise individual schema gates, not just the outer manifest seal.
    record.pop("provenance_seal")
    path.write_text(json.dumps(record))
    with pytest.raises((ValueError, FileNotFoundError)):
        results.read_histories(path)


@pytest.mark.parametrize("change", ["dtype", "object", "nonfinite", "extra", "reference", "invalid_zip", "fingerprint"])
def test_malformed_payload_with_matching_file_checksum_is_rejected(tmp_path, change):
    from agentfem.results.history_archive import _file_digest
    path = histories().write_manifest(tmp_path / "result.json", include_histories=True,
                                      history_storage="npz")
    record = json.loads(path.read_text())
    record.pop("provenance_seal")
    companion = tmp_path / record["history_archive"]["artifact"]
    with np.load(companion, allow_pickle=False) as stored:
        arrays = dict(stored)
    key = record["history_records"][0]["storage"]["values"]
    if change == "dtype":
        arrays[key] = arrays[key].astype(np.float32)
    elif change == "object":
        arrays[key] = np.array([{"not": "numeric"}], dtype=object)
    elif change == "nonfinite":
        arrays[key][0] = np.nan
    elif change == "fingerprint":
        arrays[key][0] = 100.
    elif change == "extra":
        arrays["unused"] = np.ones(1)
    elif change == "reference":
        record["history_records"][0]["storage"]["values"] = "missing"
    np.savez(companion, **arrays)
    if change == "invalid_zip":
        companion.write_bytes(b"not an array archive")
    record["history_archive"].update(sha256=_file_digest(companion), bytes=companion.stat().st_size)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        results.read_histories(path)


def test_sealed_unit_metadata_cannot_silently_change(tmp_path):
    path = histories().write_manifest(tmp_path / "result.json", include_histories=True,
                                      history_storage="npz")
    record = json.loads(path.read_text())
    record["history_records"][0]["unit"] = "kJ"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="metadata changed"):
        results.read_histories(path)


def test_metadata_only_is_not_treated_as_empty_history(tmp_path):
    path = histories().write_manifest(tmp_path / "result.json")
    with pytest.raises(ValueError, match="values were not published"):
        results.read_histories(path)
    with pytest.raises(ValueError, match="requires include_histories"):
        histories().write_manifest(path, history_storage="npz")


@pytest.mark.parametrize("bad", [True, 1.5, -1, "2"])
def test_invalid_declared_shape_is_rejected(tmp_path, bad):
    path = histories().write_manifest(tmp_path / "result.json", include_histories=True)
    record = json.loads(path.read_text())
    record.pop("provenance_seal")
    record["history_records"][-1]["value_shape"] = [bad]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="invalid declared history shape"):
        results.read_histories(path)


@pytest.mark.parametrize("component", ["quantity", "axis", "values"])
def test_real_result_channels_never_silently_discard_imaginary_parts(component):
    if component == "quantity":
        with pytest.raises(ValueError, match="requires real values"):
            results.ResultQuantity("q", np.array([1. + 2.j]))
    else:
        axis = np.array([0., 1.])
        values = np.array([2., 3.])
        if component == "axis":
            axis = axis + 1.j
        else:
            values = values + 1.j
        with pytest.raises(ValueError, match="requires real values"):
            results.HistoryResult("q", axis, values)


def test_shared_axes_reduce_storage_without_decimating(tmp_path):
    simulation = results.SimulationResult("shared-axis")
    time = np.arange(1001) * 1e-6
    for index in range(8):
        simulation.add_history(f"channel-{index}", time, np.sin(time + index), unit="1")
    inline = simulation.write_manifest(tmp_path / "inline.json", include_histories=True)
    binary = simulation.write_manifest(tmp_path / "binary.json", include_histories=True,
                                       history_storage="npz")
    record = json.loads(binary.read_text())
    assert record["history_archive"]["array_count"] == 9
    size = binary.stat().st_size + (tmp_path / record["history_archive"]["artifact"]).stat().st_size
    assert size < inline.stat().st_size / 2
    assert all(item.abscissa.size == 1001 for item in results.read_histories(binary))


def test_read_histories_without_finite_element_runtime(tmp_path):
    path = histories().write_manifest(tmp_path / "result.json", include_histories=True,
                                      history_storage="npz")
    code = '''
import importlib.abc, sys
class NoSolver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'dolfinx', 'petsc4py', 'mpi4py', 'ufl', 'basix'}:
            raise RuntimeError('unexpected solver import: ' + fullname)
sys.meta_path.insert(0, NoSolver())
from agentfem import results
assert len(results.read_histories(sys.argv[1])) == 3
assert results.HistoryResult.__module__ == 'agentfem.results.core'
assert 'read_histories' in dir(results)
assert not {'dolfinx', 'petsc4py', 'mpi4py', 'ufl', 'basix'} & set(sys.modules)
'''
    completed = subprocess.run([sys.executable, "-c", code, str(path)],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


def test_public_exports_resolve_to_their_owning_implementation():
    import importlib
    assert set(results.__all__) == set(results._LAZY_EXPORTS)
    for name, module in results._LAZY_EXPORTS.items():
        assert getattr(results, name) is getattr(importlib.import_module(f"agentfem.results.{module}"), name)


def test_cli_compares_inline_and_binary_histories(tmp_path, capsys):
    from agentfem.cli import main
    simulation = histories()
    left = simulation.write_manifest(tmp_path / "left.json", include_histories=True)
    right = simulation.write_manifest(tmp_path / "right.json", include_histories=True,
                                      history_storage="npz")
    original = right.read_bytes()
    assert main(["compare", str(left), str(right), "--history", "energy", "--atol", "0", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "within_tolerance"
    assert right.read_bytes() == original


def test_cli_returns_structured_error_for_malformed_archive(tmp_path, capsys):
    from agentfem.cli import main
    from agentfem.results.history_archive import _file_digest
    left = histories().write_manifest(tmp_path / "left.json", include_histories=True)
    right = histories().write_manifest(tmp_path / "right.json", include_histories=True, history_storage="npz")
    record = json.loads(right.read_text())
    record.pop("provenance_seal")
    companion = tmp_path / record["history_archive"]["artifact"]
    companion.write_bytes(b"not an array archive")
    record["history_archive"].update(bytes=companion.stat().st_size, sha256=_file_digest(companion))
    right.write_text(json.dumps(record))
    assert main(["compare", str(left), str(right), "--history", "energy", "--atol", "0", "--json"]) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["schema"] == "agentfem.cli-error"
    assert "AFM-HISTORY-ARCHIVE" in error["error"]["message"]


def test_collective_archive_rejects_rank_local_values(tmp_path):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires multiple MPI ranks")
    path = Path(comm.bcast(str(tmp_path / "result.json") if comm.rank == 0 else None, root=0))
    simulation = histories()
    simulation.write_manifest(path, include_histories=True, history_storage="npz", comm=comm)
    assert all(comm.allgather(provenance.verify_manifest(path).verified))
    original = path.read_bytes()
    simulation.histories["force"].values[0, 0] += comm.rank
    with pytest.raises(RuntimeError, match="History archive data differs"):
        simulation.write_manifest(path, include_histories=True, history_storage="npz", comm=comm)
    assert all(comm.allgather(path.read_bytes() == original))
