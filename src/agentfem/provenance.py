"""Lightweight provenance seals for AgentFEM result manifests.

The seal is an integrity record, not a DRM device and not a scientific
verification claim.  It binds a canonical result manifest to the byte content
of its registered artifacts using only the Python standard library.  A future
signature service can sign the stable ``seal_sha256`` without changing the
result contract or the local workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping


SEAL_SCHEMA = "agentfem.provenance-seal"
SEAL_SCHEMA_VERSION = "0.1.0"
ALGORITHM = "sha256"
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
    "SEAL_SCHEMA",
    "SEAL_SCHEMA_VERSION",
    "SealVerification",
    "seal_manifest",
    "verify_manifest",
]
