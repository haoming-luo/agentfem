"""Build and inspect AgentFEM's multi-source runtime download contract.

The runtime artifact manifest owns bytes and SHA256 identities.  This module
adds delivery routes without rebuilding, renaming, or otherwise changing an
artifact.  GitHub Releases remains the canonical source; project-operated
mirrors may publish byte-identical copies for users with poor international
connectivity.

The module deliberately has no third-party dependencies so it can be used by
release automation, a small bootstrap program, or an AI agent before the FEM
runtime exists.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote, urlsplit, urlunsplit


SCHEMA = "agentfem.runtime-downloads"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DownloadRoute:
    """One HTTPS location carrying byte-identical release artifacts."""

    name: str
    base_url: str
    audience: str = "global"
    priority: int = 100
    canonical: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower().replace(" ", "-")
        audience = str(self.audience).strip().lower().replace(" ", "-")
        parts = urlsplit(str(self.base_url).strip())
        if not name or not audience:
            raise ValueError("Download routes require a name and audience.")
        if parts.scheme != "https" or not parts.netloc:
            raise ValueError("Runtime download routes must use an absolute HTTPS URL.")
        normalized = urlunsplit(
            (parts.scheme, parts.netloc, parts.path.rstrip("/"), "", "")
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "audience", audience)
        object.__setattr__(self, "base_url", normalized)
        object.__setattr__(self, "priority", int(self.priority))

    def url(self, filename: str) -> str:
        return f"{self.base_url}/{quote(filename)}"

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "audience": self.audience,
            "priority": self.priority,
            "canonical": bool(self.canonical),
        }


def _load_artifact_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "agentfem.runtime-artifacts":
        raise ValueError("Expected an AgentFEM runtime-artifacts manifest.")
    version = str(payload.get("agentfem_version", "")).strip()
    artifacts = payload.get("artifacts")
    if not version or not isinstance(artifacts, list):
        raise ValueError("Runtime artifact manifest is incomplete.")
    names = []
    for record in artifacts:
        if not isinstance(record, dict):
            raise ValueError("Runtime artifact records must be objects.")
        name = str(record.get("filename", "")).strip()
        digest = str(record.get("sha256", "")).lower()
        size = record.get("bytes")
        if (
            not name
            or Path(name).name != name
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError(f"Invalid runtime artifact record: {record!r}.")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("Runtime artifact filenames must be unique.")
    return payload


def build_download_manifest(
    artifact_manifest: Path | str,
    *,
    routes: Iterable[DownloadRoute],
) -> dict[str, object]:
    """Bind immutable runtime artifacts to one or more download routes."""

    source = _load_artifact_manifest(Path(artifact_manifest))
    selected = tuple(sorted(routes, key=lambda item: (item.priority, item.name)))
    if not selected:
        raise ValueError("At least one runtime download route is required.")
    if len({item.name for item in selected}) != len(selected):
        raise ValueError("Runtime download route names must be unique.")
    canonical = tuple(item for item in selected if item.canonical)
    if len(canonical) != 1:
        raise ValueError("Exactly one runtime download route must be canonical.")
    records = []
    for artifact in source["artifacts"]:
        filename = str(artifact["filename"])
        records.append(
            {
                "filename": filename,
                "bytes": int(artifact["bytes"]),
                "sha256": str(artifact["sha256"]),
                "downloads": [
                    {
                        "route": route.name,
                        "url": route.url(filename),
                    }
                    for route in selected
                ],
            }
        )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "agentfem_version": source["agentfem_version"],
        "source": source.get("source"),
        "identity_manifest": "runtime-artifacts.json",
        "selection_policy": (
            "prefer the lowest-priority reachable route; always verify bytes "
            "and sha256 before installation"
        ),
        "routes": [item.summary() for item in selected],
        "artifacts": records,
    }


def write_download_manifest(
    artifact_manifest: Path | str,
    output: Path | str,
    *,
    routes: Iterable[DownloadRoute],
) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_download_manifest(artifact_manifest, routes=routes)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def select_download(
    manifest: dict[str, object],
    filename: str,
    *,
    audience: str = "global",
    reachable: Callable[[str], bool] | None = None,
) -> dict[str, object]:
    """Select one route deterministically, optionally probing reachability."""

    if manifest.get("schema") != SCHEMA:
        raise ValueError("Expected an AgentFEM runtime-downloads manifest.")
    records = {str(item["filename"]): item for item in manifest.get("artifacts", ())}
    if filename not in records:
        raise KeyError(f"Runtime artifact {filename!r} is not in the manifest.")
    routes = {str(item["name"]): item for item in manifest.get("routes", ())}
    preferred_audience = str(audience).strip().lower().replace(" ", "-")
    downloads = list(records[filename]["downloads"])
    downloads.sort(
        key=lambda item: (
            routes[item["route"]].get("audience") != preferred_audience,
            int(routes[item["route"]].get("priority", 100)),
            item["route"],
        )
    )
    predicate = (lambda _url: True) if reachable is None else reachable
    for item in downloads:
        if predicate(str(item["url"])):
            return {
                **item,
                "filename": filename,
                "bytes": records[filename]["bytes"],
                "sha256": records[filename]["sha256"],
            }
    raise ConnectionError(f"No reachable route was found for {filename!r}.")


def verify_artifact(path: Path | str, record: dict[str, object]) -> None:
    """Fail if a downloaded artifact differs from the canonical identity."""

    selected = Path(path)
    if selected.name != record.get("filename"):
        raise ValueError("Downloaded filename differs from the manifest record.")
    size = selected.stat().st_size
    if size != int(record["bytes"]):
        raise ValueError("Downloaded artifact byte count differs from the manifest.")
    digest_state = sha256()
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest_state.update(block)
    digest = digest_state.hexdigest()
    if digest != str(record["sha256"]).lower():
        raise ValueError("Downloaded artifact SHA256 differs from the manifest.")


def _route(value: str, *, canonical: bool = False) -> DownloadRoute:
    try:
        name, base_url = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Routes use NAME=HTTPS_BASE_URL.") from exc
    audience = "mainland-china" if name.strip().lower() == "china" else "global"
    priority = 10 if audience == "mainland-china" else 100
    try:
        return DownloadRoute(
            name=name,
            base_url=base_url,
            audience=audience,
            priority=priority,
            canonical=canonical,
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("artifact_manifest", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--official",
        required=True,
        help="canonical route as NAME=HTTPS_BASE_URL",
    )
    result.add_argument(
        "--mirror",
        action="append",
        default=[],
        help="optional byte-identical mirror as NAME=HTTPS_BASE_URL",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    official = _route(args.official, canonical=True)
    mirrors = tuple(_route(value) for value in args.mirror)
    target = write_download_manifest(
        args.artifact_manifest,
        args.output,
        routes=(official, *mirrors),
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
