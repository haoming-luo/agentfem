# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Atomic text publication without importing a numerical runtime."""

import os
from pathlib import Path
import tempfile


def atomic_write_text(path, content: str) -> Path:
    selected = Path(path)
    selected.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=selected.parent,
            prefix=f".{selected.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(selected)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return selected
