# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Small local-only history of explicitly opened result locations."""
import json
from pathlib import Path
from agentfem._atomic_text import atomic_write_text


def local_path(text):
    value = str(text).strip().strip('"\'').rstrip('\\').strip()
    if not value:
        raise ValueError('Paste a folder or result path.')
    return Path(value).expanduser().resolve(strict=True)


def recent_locations(storage, opened=None):
    storage = Path(storage)
    try:
        items = json.loads(storage.read_text())
        if not isinstance(items, list):
            items = []
    except (OSError, ValueError):
        items = []
    items = [p for p in items if isinstance(p, str) and Path(p).exists()]
    if opened is not None:
        value = str(Path(opened).resolve())
        items = [value] + [p for p in items if p != value]
    items = list(dict.fromkeys(items))[:8]
    if opened is not None:
        try:
            storage.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(storage, json.dumps(items))
        except OSError:
            pass  # Optional history must never prevent opening a result.
    return [{'title': Path(p).name or p, 'subtitle': p, 'value': p} for p in items]
