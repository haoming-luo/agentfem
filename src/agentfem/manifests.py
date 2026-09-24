# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard-library readers for published AgentFEM scientific manifests.

This module deliberately imports neither DOLFINx nor PETSc.  Data consumers
can inspect result and campaign contracts without loading a solver runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


SUPPORTED_SCHEMAS = {
    "agentfem.simulation-result": {"0.1.0"},
    "agentfem.campaign": {"0.1.0"},
    "agentfem.campaign-audit": {"0.1.0"},
    "agentfem.scientific-dataset": {"0.1.0"},
    "agentfem.scientific-field-dataset": {"0.1.0"},
}


def read(path: str | Path, *, schema: str | None = None) -> dict[str, object]:
    """Read and validate one versioned JSON manifest."""

    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON manifest {source}: {exc}.") from exc
    if not isinstance(record, dict):
        raise ValueError("AgentFEM manifest root must be a JSON object.")
    actual_schema = str(record.get("schema", ""))
    version = str(record.get("schema_version", ""))
    if schema is not None and actual_schema != schema:
        raise ValueError(f"Expected schema {schema!r}, found {actual_schema!r}.")
    supported = SUPPORTED_SCHEMAS.get(actual_schema)
    if supported is None:
        raise ValueError(f"Unsupported AgentFEM manifest schema {actual_schema!r}.")
    if version not in supported:
        raise ValueError(
            f"Unsupported {actual_schema} version {version!r}; "
            f"supported={tuple(sorted(supported))}."
        )
    return record


def result(path: str | Path) -> dict[str, object]:
    """Read a SimulationResult manifest and reject duplicate field names."""

    record = read(path, schema="agentfem.simulation-result")
    fields = tuple(record.get("field_records", ()))
    _require_unique_names(fields, label="field_records")
    quantities = tuple(record.get("quantity_records", ()))
    _require_unique_names(quantities, label="quantity_records")
    return record


def campaign(path: str | Path) -> dict[str, object]:
    """Read a campaign report and reject duplicate case identities."""

    record = read(path, schema="agentfem.campaign")
    cases = tuple(record.get("records", ()))
    identities = tuple(
        str(item.get("case", {}).get("case_id", ""))
        for item in cases
        if isinstance(item, Mapping)
    )
    if not all(identities) or len(set(identities)) != len(identities):
        raise ValueError("Campaign records require unique non-empty case IDs.")
    return record


def _require_unique_names(records, *, label: str) -> None:
    names = tuple(
        str(item.get("name", "")) for item in records if isinstance(item, Mapping)
    )
    if len(names) != len(records) or not all(names) or len(set(names)) != len(names):
        raise ValueError(f"{label} require unique non-empty names.")


__all__ = ("SUPPORTED_SCHEMAS", "campaign", "read", "result")
