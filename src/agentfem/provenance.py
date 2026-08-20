"""Lightweight provenance seals for AgentFEM result manifests.

The seal is an integrity record, not a DRM device and not a scientific
verification claim.  It binds a canonical result manifest to the byte content
of its registered artifacts using only the Python standard library.  A future
signature service can sign the stable ``seal_sha256`` without changing the
result contract or the local workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import inspect
import json
from math import isfinite
from pathlib import Path
from typing import Mapping
import warnings

import numpy as np


SEAL_SCHEMA = "agentfem.provenance-seal"
SEAL_SCHEMA_VERSION = "0.1.0"
ALGORITHM = "sha256"
RUNTIME_SCHEMA = "agentfem.runtime-lock"
RUNTIME_SCHEMA_VERSION = "0.1.0"
_CHUNK_SIZE = 1024 * 1024
ORIGIN = {
    "project": "AgentFEM",
    "initiated_by": "Haoming Luo",
    "open_sourced": "2026-07",
    "repository": "https://github.com/haoming-luo/agentfem",
    "license": "Apache-2.0",
    "citation_file": "CITATION.cff",
}


def _canonical_bytes(record: object) -> bytes:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(record: object) -> str:
    return sha256(_canonical_bytes(record)).hexdigest()


def content_fingerprint(record: object) -> str:
    """Return a canonical content identity for one JSON-safe scientific record."""

    return f"sha256:{_digest(record)}"


def scientific_input_manifest(
    value: object,
    *,
    label: str = "scientific_inputs",
    require_nonempty: bool = False,
) -> dict[str, object]:
    """Describe and fingerprint scientific inputs without hiding opaque parts."""

    missing: list[dict[str, str]] = []
    if require_nonempty and isinstance(value, Mapping) and not value:
        missing.append(
            {"path": str(label), "reason": "no_scientific_inputs_declared"}
        )
    record = _scientific_input_record(
        value,
        path=str(label),
        missing=missing,
        seen=set(),
    )
    identity = {
        "label": str(label),
        "record": record,
    }
    return {
        "schema": "agentfem.scientific-input-manifest",
        "schema_version": "0.1.0",
        "label": str(label),
        "complete": not missing,
        "missing": tuple(missing),
        "record": record,
        "fingerprint": content_fingerprint(identity),
    }


def _scientific_input_record(value, *, path: str, missing, seen):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _scientific_input_record(
            value.item(),
            path=path,
            missing=missing,
            seen=seen,
        )
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Path):
        return _scientific_file_record(value, path=path, missing=missing)
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "kind": "array",
            "dtype": contiguous.dtype.str,
            "shape": contiguous.shape,
            "sha256": sha256(contiguous.tobytes(order="C")).hexdigest(),
        }
    identity = id(value)
    if identity in seen:
        missing.append({"path": path, "reason": "cyclic_object_graph"})
        return _opaque_input(value)
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            return {
                str(key): _scientific_input_record(
                    value[key],
                    path=f"{path}.{key}",
                    missing=missing,
                    seen=seen,
                )
                for key in sorted(value, key=lambda item: str(item))
            }
        if isinstance(value, (tuple, list, set, frozenset)):
            selected = tuple(value)
            if isinstance(value, (set, frozenset)):
                selected = tuple(sorted(value, key=lambda item: str(item)))
            return [
                _scientific_input_record(
                    item,
                    path=f"{path}[{index}]",
                    missing=missing,
                    seen=seen,
                )
                for index, item in enumerate(selected)
            ]
        for method_name in ("to_ir", "as_dict", "summary"):
            method = getattr(value, method_name, None)
            if not callable(method):
                continue
            try:
                converted = method()
            except (TypeError, ValueError, RuntimeError):
                continue
            if converted is value:
                continue
            return {
                "kind": "declared_scientific_object",
                "python_type": _python_type(value),
                "contract": method_name,
                "value": _scientific_input_record(
                    converted,
                    path=f"{path}.{method_name}",
                    missing=missing,
                    seen=seen,
                ),
            }
        if callable(value):
            return _scientific_callable_record(
                value,
                path=path,
                missing=missing,
                seen=seen,
            )
    finally:
        seen.remove(identity)
    missing.append({"path": path, "reason": "no_scientific_identity_contract"})
    return _opaque_input(value)


def _scientific_file_record(value: Path, *, path: str, missing) -> dict[str, object]:
    selected = value.expanduser()
    record: dict[str, object] = {
        "kind": "file",
        "name": selected.name,
    }
    if not selected.exists():
        record["status"] = "missing"
        missing.append({"path": path, "reason": "file_missing"})
    elif not selected.is_file():
        record["status"] = "not_a_file"
        missing.append({"path": path, "reason": "path_is_not_a_file"})
    else:
        record.update(
            {
                "status": "hashed",
                "size_bytes": selected.stat().st_size,
                "sha256": _file_digest(selected),
            }
        )
    return record


def _scientific_callable_record(
    value,
    *,
    path: str,
    missing,
    seen,
) -> dict[str, object]:
    record = {
        "kind": "callable",
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", getattr(value, "__name__", None)),
    }
    try:
        source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError):
        source = None
    if source is None:
        missing.append({"path": path, "reason": "callable_source_unavailable"})
        record["source_sha256"] = None
    else:
        record["source_sha256"] = sha256(source).hexdigest()
    try:
        source_file = inspect.getsourcefile(value)
    except TypeError:
        source_file = None
    if source_file is None and not inspect.isroutine(value):
        try:
            source_file = inspect.getsourcefile(type(value))
        except TypeError:
            source_file = None
    source_path = None if source_file is None else Path(source_file)
    if source_path is None or not source_path.is_file():
        missing.append({"path": path, "reason": "callable_source_file_unavailable"})
        record["source_file_sha256"] = None
    else:
        record["source_file_name"] = source_path.name
        record["source_file_sha256"] = _file_digest(source_path)
    defaults = getattr(value, "__defaults__", None)
    if defaults:
        record["defaults"] = _scientific_input_record(
            defaults,
            path=f"{path}.__defaults__",
            missing=missing,
            seen=seen,
        )
    keyword_defaults = getattr(value, "__kwdefaults__", None)
    if keyword_defaults:
        record["keyword_defaults"] = _scientific_input_record(
            keyword_defaults,
            path=f"{path}.__kwdefaults__",
            missing=missing,
            seen=seen,
        )
    closure = getattr(value, "__closure__", None)
    if closure:
        closure_values = []
        for index, cell in enumerate(closure):
            try:
                contents = cell.cell_contents
            except ValueError:
                contents = None
            closure_values.append(
                _scientific_input_record(
                    contents,
                    path=f"{path}.__closure__[{index}]",
                    missing=missing,
                    seen=seen,
                )
            )
        record["closure"] = closure_values
    bound = getattr(value, "__self__", None)
    if bound is not None and not inspect.isclass(bound):
        record["bound_state"] = _scientific_input_record(
            bound,
            path=f"{path}.__self__",
            missing=missing,
            seen=seen,
        )
    if isinstance(value, partial):
        record["partial"] = {
            "function": _scientific_input_record(
                value.func,
                path=f"{path}.func",
                missing=missing,
                seen=seen,
            ),
            "args": _scientific_input_record(
                value.args,
                path=f"{path}.args",
                missing=missing,
                seen=seen,
            ),
            "keywords": _scientific_input_record(
                value.keywords or {},
                path=f"{path}.keywords",
                missing=missing,
                seen=seen,
            ),
        }
    elif not inspect.isroutine(value):
        state = getattr(value, "__dict__", None)
        if isinstance(state, Mapping) and state:
            record["callable_state"] = _scientific_input_record(
                state,
                path=f"{path}.__dict__",
                missing=missing,
                seen=seen,
            )
        else:
            missing.append(
                {"path": path, "reason": "callable_object_state_unavailable"}
            )
    return record


def _python_type(value) -> str:
    selected = type(value)
    return f"{selected.__module__}.{selected.__qualname__}"


def _opaque_input(value) -> dict[str, object]:
    return {
        "kind": "opaque_scientific_input",
        "python_type": _python_type(value),
        "fingerprinted": False,
    }


def _file_digest(path: Path) -> str:
    checksum = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            checksum.update(chunk)
    return checksum.hexdigest()


def _artifact_path(value: object, *, base: Path) -> Path:
    selected = Path(str(value)).expanduser()
    return selected if selected.is_absolute() else base / selected


def _artifact_inventory(
    artifacts: Mapping[str, object],
    *,
    base: Path,
) -> tuple[dict[str, object], ...]:
    records = []
    for name, value in sorted(artifacts.items()):
        path = _artifact_path(value, base=base)
        record: dict[str, object] = {
            "name": str(name),
            "path": str(value),
        }
        if not path.exists():
            record["status"] = "missing"
        elif not path.is_file():
            record["status"] = "not_a_file"
        else:
            record.update(
                {
                    "status": "hashed",
                    "size_bytes": path.stat().st_size,
                    "sha256": _file_digest(path),
                }
            )
        records.append(record)
    return tuple(records)


def seal_manifest(
    manifest: Mapping[str, object],
    *,
    base: str | Path,
    producer_version: str,
) -> dict[str, object]:
    """Return a deterministic integrity seal for an unsealed manifest."""

    content = dict(manifest)
    content.pop("provenance_seal", None)
    artifacts = _artifact_inventory(
        content.get("artifacts", {}),
        base=Path(base),
    )
    complete = all(item["status"] == "hashed" for item in artifacts)
    payload: dict[str, object] = {
        "schema": SEAL_SCHEMA,
        "schema_version": SEAL_SCHEMA_VERSION,
        "producer": "AgentFEM",
        "producer_version": str(producer_version),
        "origin": ORIGIN,
        "algorithm": ALGORITHM,
        "completeness": "complete" if complete else "incomplete",
        "manifest_sha256": _digest(content),
        "artifacts": artifacts,
    }
    payload["seal_sha256"] = _digest(payload)
    payload["seal_id"] = f"sha256:{payload['seal_sha256']}"
    return payload


def runtime_manifest() -> dict[str, object]:
    """Capture runtime evidence and a stable compatibility identity.

    Paths and the working directory remain visible in ``report`` but are not
    runtime-lock gates. The identity contains the versions, scalar contract,
    MPI size/vendor, and source or installed-distribution evidence that can
    change numerical behavior or the code being executed.
    """

    from .platforms import runtime_report

    report = runtime_report().summary()
    execution = dict(report["execution"])
    source = execution.get("source")
    distribution = execution.get("distribution")
    direct_vcs = None
    if isinstance(distribution, Mapping):
        direct_url = distribution.get("direct_url")
        if isinstance(direct_url, Mapping):
            direct_vcs = direct_url.get("vcs_info")
    source_mode = execution.get("mode") == "source_checkout"
    identity = {
        "platform": {
            "system": report["platform"]["system"],
            "route": report["platform"]["route"],
        },
        "python": report["python"],
        "machine": report["machine"],
        "operating_system": report["operating_system"],
        "packages": report["packages"],
        "mpi": {
            "vendor": report["mpi"]["vendor"],
            "rank_count": report["mpi"]["rank_count"],
        },
        "numerics": report["numerics"],
        "execution": {
            "mode": execution.get("mode"),
            "source": source if source_mode else None,
            "distribution": (
                None
                if source_mode or not isinstance(distribution, Mapping)
                else {
                    "version": distribution.get("version"),
                    "record_sha256": distribution.get("record_sha256"),
                    "vcs_info": direct_vcs,
                }
            ),
        },
    }
    return {
        "schema": RUNTIME_SCHEMA,
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "fingerprint": content_fingerprint(identity),
        "identity": identity,
        "report": report,
    }


def freeze_runtime(path: str | Path) -> Path:
    """Atomically write the current runtime lock for a frozen campaign."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            runtime_manifest(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


@dataclass(frozen=True)
class RuntimeComparison:
    """Compatibility decision between a frozen and current runtime."""

    expected_fingerprint: str
    actual_fingerprint: str
    mismatches: tuple[dict[str, object], ...] = ()

    @property
    def compatible(self) -> bool:
        return not self.mismatches

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.runtime-comparison",
            "schema_version": "0.1.0",
            "compatible": self.compatible,
            "expected_fingerprint": self.expected_fingerprint,
            "actual_fingerprint": self.actual_fingerprint,
            "mismatches": self.mismatches,
        }

    def format(self) -> str:
        heading = "AgentFEM runtime: compatible" if self.compatible else "AgentFEM runtime: mismatch"
        lines = [heading]
        lines.extend(
            f"  {item['path']}: expected {item['expected']!r}, got {item['actual']!r}"
            for item in self.mismatches
        )
        return "\n".join(lines)


def compare_runtime(
    expected: str | Path | Mapping[str, object],
    *,
    actual: Mapping[str, object] | None = None,
) -> RuntimeComparison:
    """Compare a stored runtime identity with the current or supplied one."""

    frozen = _load_runtime_record(expected)
    current = runtime_manifest() if actual is None else dict(actual)
    if frozen.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("Expected an AgentFEM runtime-lock manifest.")
    if current.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("Actual runtime record is not an AgentFEM runtime lock.")
    if frozen.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise ValueError("Unsupported frozen AgentFEM runtime-lock schema version.")
    if current.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise ValueError("Unsupported current AgentFEM runtime-lock schema version.")
    expected_identity = frozen.get("identity")
    actual_identity = current.get("identity")
    if not isinstance(expected_identity, Mapping) or not isinstance(actual_identity, Mapping):
        raise ValueError("Runtime locks require mapping-valued identity records.")
    if frozen.get("fingerprint") != content_fingerprint(expected_identity):
        raise ValueError("Frozen runtime-lock fingerprint does not match its identity.")
    if current.get("fingerprint") != content_fingerprint(actual_identity):
        raise ValueError("Current runtime-lock fingerprint does not match its identity.")
    mismatches: list[dict[str, object]] = []
    _identity_differences(expected_identity, actual_identity, path="", output=mismatches)
    return RuntimeComparison(
        expected_fingerprint=str(frozen.get("fingerprint")),
        actual_fingerprint=str(current.get("fingerprint")),
        mismatches=tuple(mismatches),
    )


def require_runtime(
    expected: str | Path | Mapping[str, object],
    *,
    policy: str = "error",
) -> RuntimeComparison:
    """Enforce or warn about a frozen runtime before a scientific campaign."""

    selected = str(policy).strip().lower().replace("-", "_")
    if selected not in {"error", "warn"}:
        raise ValueError("Runtime policy must be 'error' or 'warn'.")
    comparison = compare_runtime(expected)
    if comparison.compatible:
        return comparison
    if selected == "error":
        raise RuntimeError(comparison.format())
    warnings.warn(comparison.format(), RuntimeWarning, stacklevel=2)
    return comparison


def _load_runtime_record(value: str | Path | Mapping[str, object]) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _identity_differences(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
    *,
    path: str,
    output: list[dict[str, object]],
) -> None:
    keys = tuple(sorted(set(expected) | set(actual)))
    for key in keys:
        selected_path = f"{path}.{key}" if path else str(key)
        left = expected.get(key)
        right = actual.get(key)
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            _identity_differences(left, right, path=selected_path, output=output)
        elif left != right:
            output.append(
                {"path": selected_path, "expected": left, "actual": right}
            )


@dataclass(frozen=True)
class SealVerification:
    """Outcome of checking one stored provenance seal."""

    path: Path
    status: str
    seal_id: str | None
    issues: tuple[dict[str, str], ...] = ()

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.provenance-verification",
            "schema_version": "0.1.0",
            "path": str(self.path),
            "status": self.status,
            "verified": self.verified,
            "seal_id": self.seal_id,
            "issues": self.issues,
        }

    def format(self) -> str:
        lines = [
            f"AgentFEM provenance: {self.status}",
            f"  manifest: {self.path}",
        ]
        if self.seal_id:
            lines.append(f"  seal: {self.seal_id}")
        for issue in self.issues:
            lines.append(f"  {issue['code']}: {issue['message']}")
        return "\n".join(lines)


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def verify_manifest(path: str | Path) -> SealVerification:
    """Verify a result manifest and every artifact recorded in its seal."""

    selected = Path(path).expanduser().resolve()
    record = json.loads(selected.read_text(encoding="utf-8"))
    stored = record.pop("provenance_seal", None)
    if not isinstance(stored, dict):
        return SealVerification(
            selected,
            "unsealed",
            None,
            (_issue("AFM-SEAL-001", "Result manifest has no provenance seal."),),
        )

    issues: list[dict[str, str]] = []
    stored_id = stored.get("seal_id")
    stored_digest = stored.get("seal_sha256")
    seal_payload = dict(stored)
    seal_payload.pop("seal_id", None)
    seal_payload.pop("seal_sha256", None)
    expected_seal_digest = _digest(seal_payload)
    if stored_digest != expected_seal_digest:
        issues.append(
            _issue("AFM-SEAL-002", "The stored seal record has been modified.")
        )
    if stored_id != f"sha256:{stored_digest}":
        issues.append(
            _issue("AFM-SEAL-002", "The seal identifier does not match its digest.")
        )

    if stored.get("manifest_sha256") != _digest(record):
        issues.append(
            _issue("AFM-SEAL-003", "The result manifest content has changed.")
        )

    stored_artifacts = tuple(stored.get("artifacts", ()))
    current_artifacts = _artifact_inventory(
        record.get("artifacts", {}),
        base=selected.parent,
    )
    if stored_artifacts != current_artifacts:
        issues.append(
            _issue(
                "AFM-SEAL-004",
                "One or more registered artifacts are missing, replaced, or changed.",
            )
        )

    modified = any(item["code"] != "AFM-SEAL-005" for item in issues)
    incomplete = (
        stored.get("completeness") != "complete"
        or any(item.get("status") != "hashed" for item in current_artifacts)
    )
    if incomplete and not modified:
        issues.append(
            _issue(
                "AFM-SEAL-005",
                "The seal is internally consistent, but not every artifact was available.",
            )
        )
    status = "modified" if modified else ("incomplete" if incomplete else "verified")
    return SealVerification(
        selected,
        status,
        None if stored_id is None else str(stored_id),
        tuple(issues),
    )


__all__ = [
    "ALGORITHM",
    "ORIGIN",
    "RUNTIME_SCHEMA",
    "RUNTIME_SCHEMA_VERSION",
    "SEAL_SCHEMA",
    "SEAL_SCHEMA_VERSION",
    "SealVerification",
    "RuntimeComparison",
    "compare_runtime",
    "content_fingerprint",
    "freeze_runtime",
    "require_runtime",
    "runtime_manifest",
    "scientific_input_manifest",
    "seal_manifest",
    "verify_manifest",
]
