# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless history artifacts; no solver, pickle, resampling or trust promotion."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import uuid
import zipfile

import numpy as np

from .core import HistoryResult


_SCHEMA = "agentfem.history-arrays"
_VERSION = "0.1.0"


def _array_key(array):
    digest = sha256(json.dumps(list(array.shape)).encode())
    if array.size:
        digest.update(memoryview(array).cast("B"))
    return "a_" + digest.hexdigest()


def _prepare(histories):
    """Snapshot and validate histories before collective/publication decisions."""
    arrays, records = {}, []
    for history in histories:
        selected = HistoryResult(
            history.name, history.abscissa, history.values, unit=history.unit,
            abscissa_name=history.abscissa_name,
            abscissa_unit=history.abscissa_unit, description=history.description,
        )
        keys = {}
        for label, value in (("abscissa", selected.abscissa), ("values", selected.values)):
            array = np.asarray(value, dtype="<f8", order="C")
            key = _array_key(array)
            arrays.setdefault(key, array)
            keys[label] = key
        records.append({**selected.as_dict(include_values=False), "storage": keys})
    if len({item["name"] for item in records}) != len(records):
        raise ValueError("AFM-HISTORY-ARCHIVE: duplicate history names")
    return records, arrays


def _file_digest(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(manifest, records, arrays):
    """Publish a unique immutable companion before the JSON commit point."""
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    destination = manifest.with_name(
        f"{manifest.stem}-histories-{uuid.uuid4().hex}.npz"
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=manifest.parent, prefix=".history-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            # Uncompressed NPY members avoid text expansion/compression cost.
            # Identical binary64 arrays (typically all time axes) are stored once.
            np.savez(stream, **arrays)
        temporary.replace(destination)
        return {
            "schema": _SCHEMA, "schema_version": _VERSION,
            "artifact": destination.name,
            "sha256": _file_digest(destination),
            "bytes": destination.stat().st_size,
            "array_count": len(arrays),
            "history_count": len(records),
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_histories(manifest_path: str | Path) -> tuple[HistoryResult, ...]:
    """Read complete histories from an inline or NPZ result manifest.

    Requires only NumPy, not FEniCSx. External arrays are loaded only from the
    explicitly selected manifest's directory, after SHA-256 and member checks.
    No pickle, unit conversion, interpolation or scientific validation occurs.
    Metadata-only manifests cannot be used as though they contain curve data.
    The original manifest and its provenance seal are never rewritten.
    """
    path = Path(manifest_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema") != "agentfem.simulation-result"
            or record.get("schema_version") != "0.1.0"):
        raise ValueError("AFM-HISTORY-ARCHIVE: unsupported result schema")
    if "provenance_seal" in record:
        from ..provenance import content_fingerprint

        seal = record["provenance_seal"]
        content = {key: value for key, value in record.items() if key != "provenance_seal"}
        if (not isinstance(seal, dict)
                or content_fingerprint(content) != "sha256:" + str(seal.get("manifest_sha256"))):
            raise ValueError("AFM-HISTORY-ARCHIVE: sealed history metadata changed")
    histories = record.get("history_records")
    if not isinstance(histories, list) or any(not isinstance(item, dict) for item in histories):
        raise ValueError("AFM-HISTORY-ARCHIVE: missing history records")
    archive = record.get("history_archive")
    arrays = None
    try:
        if archive is not None:
            if (not isinstance(archive, dict) or archive.get("schema") != _SCHEMA
                    or archive.get("schema_version") != _VERSION):
                raise ValueError("AFM-HISTORY-ARCHIVE: unsupported array schema")
            checksum = archive.get("sha256")
            if (any(type(archive.get(key)) is not int or archive[key] < 0
                    for key in ("bytes", "array_count", "history_count"))
                    or not isinstance(checksum, str) or len(checksum) != 64
                    or any(character not in "0123456789abcdef" for character in checksum)):
                raise ValueError("AFM-HISTORY-ARCHIVE: malformed array inventory")
            name = archive.get("artifact")
            if (not isinstance(name, str) or not name.endswith(".npz")
                    or Path(name).name != name or "/" in name or "\\" in name):
                raise ValueError("AFM-HISTORY-ARCHIVE: companion must be a local filename")
            source = path.parent / name
            if source.resolve().parent != path.parent.resolve():
                raise ValueError("AFM-HISTORY-ARCHIVE: companion escapes result directory")
            if (source.stat().st_size != archive.get("bytes")
                    or _file_digest(source) != archive.get("sha256")):
                raise ValueError("AFM-HISTORY-ARCHIVE: companion checksum mismatch")
            required = set()
            for item in histories:
                storage = item.get("storage")
                if (not isinstance(storage, dict)
                        or set(storage) != {"abscissa", "values"}
                        or any(not isinstance(value, str) for value in storage.values())
                        or "abscissa" in item or "values" in item):
                    raise ValueError("AFM-HISTORY-ARCHIVE: malformed array references")
                required.update(storage.values())
            with zipfile.ZipFile(source) as container:
                members = container.namelist()
                if (len(members) != len(set(members))
                        or set(members) != {key + ".npy" for key in required}
                        or len(required) != archive.get("array_count")
                        or len(histories) != archive.get("history_count")):
                    raise ValueError("AFM-HISTORY-ARCHIVE: array inventory mismatch")
            arrays = np.load(source, allow_pickle=False)
        result, loaded = [], {}
        for item in histories:
            if not {"name", "unit", "abscissa_name", "abscissa_unit", "sample_count", "value_shape"} <= item.keys():
                raise ValueError("AFM-HISTORY-ARCHIVE: malformed history metadata")
            count, shape = item["sample_count"], item["value_shape"]
            if (type(count) is not int or count < 0 or not isinstance(shape, list)
                    or any(type(size) is not int or size < 0 for size in shape)):
                raise ValueError("AFM-HISTORY-ARCHIVE: invalid declared history shape")
            if arrays is None:
                if "storage" in item or "abscissa" not in item or "values" not in item:
                    raise ValueError("AFM-HISTORY-ARCHIVE: history values were not published")
                x, y = item["abscissa"], item["values"]
                # JSON [] cannot retain the trailing shape of an empty vector
                # history. Recover only that unambiguous case from the schema;
                # never reshape nonempty/malformed data to fit its declaration.
                if count == 0 and isinstance(y, list) and not y:
                    y = np.empty((0, *shape), dtype=float)
            else:
                for key in item["storage"].values():
                    if key not in loaded:
                        value = arrays[key]
                        if value.dtype.str != "<f8" or not value.flags.c_contiguous:
                            raise ValueError("AFM-HISTORY-ARCHIVE: expected contiguous binary64 arrays")
                        if _array_key(value) != key:
                            raise ValueError("AFM-HISTORY-ARCHIVE: array fingerprint mismatch")
                        loaded[key] = value
                x, y = (loaded[item["storage"][key]] for key in ("abscissa", "values"))
            history = HistoryResult(
                name=item["name"], abscissa=x, values=y, unit=item["unit"],
                abscissa_name=item["abscissa_name"], abscissa_unit=item["abscissa_unit"],
                description=item.get("description", ""),
            )
            if (history.abscissa.size != item.get("sample_count")
                    or list(history.value_shape) != item.get("value_shape")):
                raise ValueError("AFM-HISTORY-ARCHIVE: declared history shape mismatch")
            result.append(history)
        if len({item.name for item in result}) != len(result):
            raise ValueError("AFM-HISTORY-ARCHIVE: duplicate history names")
        return tuple(result)
    except (zipfile.BadZipFile, EOFError, KeyError, TypeError) as error:
        raise ValueError(f"AFM-HISTORY-ARCHIVE: malformed history payload: {error}") from error
    finally:
        if arrays is not None:
            arrays.close()
