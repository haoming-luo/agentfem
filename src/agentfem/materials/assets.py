"""Explicit loaders for project-owned material assets.

TOML can select an asset, but executable constitutive equations remain Python
or compiled code. This module provides the small-project counterpart to an
installed AgentFEM extension: one deliberately loaded Python file, one public
material object, and a retained content fingerprint.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys

from .definitions import (
    MaterialDefinition,
    define,
    load_registered_definition,
    registered_definitions,
)
from .library import load_definition


class MaterialAssetError(RuntimeError):
    """A project material asset could not be loaded unambiguously."""


def load(
    source: str | Path,
    *,
    model: str | None = None,
    symbol: str | None = None,
    role: str = "mechanical",
) -> MaterialDefinition:
    """Load a packaged card by name or an explicitly selected Python asset.

    A path is never interpreted as TOML constitutive code. Python files are
    trusted executable project assets and must publish ``material`` or a
    zero-argument ``create_material`` factory unless ``symbol=`` is supplied.
    """

    selected = Path(source).expanduser()
    if selected.suffix.lower() == ".py" or selected.exists():
        return load_python(selected, symbol=symbol, role=role)
    if symbol is not None:
        raise ValueError("symbol= is valid only when loading a Python file.")
    if str(source) in registered_definitions():
        return load_registered_definition(str(source))
    return load_definition(str(source), model=model)


def load_python(
    path: str | Path,
    *,
    symbol: str | None = None,
    role: str = "mechanical",
) -> MaterialDefinition:
    """Load one trusted project-owned Python material with source provenance."""

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".py":
        raise ValueError("Project material assets must be Python files ending in .py.")
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = source.read_bytes()
    digest = sha256(raw).hexdigest()
    module_name = f"_agentfem_material_{digest[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise MaterialAssetError(f"Could not create a loader for {source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise MaterialAssetError(
            f"Could not execute material asset {source}: {type(exc).__name__}: {exc}"
        ) from exc

    selected_symbol = symbol
    if selected_symbol is None:
        if hasattr(module, "material"):
            selected_symbol = "material"
        elif hasattr(module, "create_material"):
            selected_symbol = "create_material"
        else:
            raise MaterialAssetError(
                f"Material asset {source} must publish `material` or "
                "`create_material()`, or be loaded with symbol=."
            )
    if not hasattr(module, selected_symbol):
        raise MaterialAssetError(
            f"Material asset {source} has no symbol {selected_symbol!r}."
        )
    candidate = getattr(module, selected_symbol)
    if callable(candidate):
        try:
            candidate = candidate()
        except TypeError as exc:
            raise MaterialAssetError(
                f"Material factory {selected_symbol!r} must accept no arguments."
            ) from exc

    provenance = {
        "asset_kind": "project_python_material",
        "source_path": str(source),
        "source_sha256": digest,
        "symbol": selected_symbol,
    }
    if isinstance(candidate, MaterialDefinition):
        return replace(
            candidate,
            source=f"python:{source}",
            metadata={**dict(candidate.metadata), **provenance},
        )
    selected_role = str(role).strip().lower().replace("-", "_")
    if not selected_role:
        raise ValueError("role must be non-empty.")
    name = str(getattr(candidate, "name", source.stem))
    return define(
        name,
        behaviors={selected_role: candidate},
        source=f"python:{source}",
        metadata=provenance,
    )
