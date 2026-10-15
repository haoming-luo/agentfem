# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Serial run handoff: reuse and retain only owned, verified checkpoints."""
import json
from pathlib import Path


def _owned(record, root):
    path = Path(record.path)
    return (record.metadata.get("role") != "restart_source" and not path.is_symlink()
            and path.resolve().is_relative_to(Path(root).resolve()))


def _verify(path, *, increment):
    from . import checkpointing as cp
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get('schema') not in {cp.TRANSIENT_CHECKPOINT_SCHEMA,
            cp._TRANSIENT_ARRAY_CHECKPOINT_SCHEMA, cp._TRANSIENT_RANK_ARRAY_CHECKPOINT_SCHEMA}:
        raise ValueError("AFM-RUN-CHECKPOINT: unsupported retention schema")
    if data.get('completed_steps') != increment or data.get('rank_count') != 1:
        raise ValueError("AFM-RUN-CHECKPOINT: accepted station/rank mismatch")
    records = list(data['shards'])
    if data.get('auxiliary_arrays'):
        records.append(data['auxiliary_arrays'])
    records.extend(item['arrays'] for item in data.get('auxiliary_by_rank', ()))
    if data.get('portable_state'):
        records.append(data['portable_state'])
    if not records:
        raise ValueError("AFM-RUN-CHECKPOINT: missing payloads")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("AFM-RUN-CHECKPOINT: payload has no integrity record")
        cp._retention_payload_path(path, record['path'], generation=data.get('generation'))
        cp.validate_checkpoint_record(path.parent, record)


def final_checkpoint(step, output):
    """Reuse an owned scheduled endpoint, otherwise save normally."""
    for record in reversed(getattr(step, 'checkpoints', ())):
        if _owned(record, output) and record.metadata.get('completed_steps') == step.completed_steps:
            _verify(record.path, increment=step.completed_steps)
            return Path(record.path)
    return step.save_checkpoint(Path(output) / 'accepted-final')


def retain_completed(step, output, final, keep, comm):
    """Explicit opt-in; never traverse directories or remove restart sources.

    Called only after final export succeeds. Revalidate final payload checksums
    before touching any old manifest. Deletion is not a claim of restart parity.
    """
    from . import checkpointing as cp
    _verify(final, increment=step.completed_steps)
    selected = [r for r in getattr(step, 'checkpoints', ()) if _owned(r, output)]
    paths = list(dict.fromkeys(Path(r.path).resolve() for r in selected))
    final = Path(final).resolve()
    paths = [p for p in paths if p != final] + [final]
    removed = []
    for path in paths[:-keep]:
        cp._remove_transient_checkpoint(path, comm=comm)
        removed.append(str(path))
    if removed:
        step.checkpoints[:] = [r for r in step.checkpoints if str(Path(r.path).resolve()) not in removed]
    return {"keep": keep, "removed_manifests": removed, "final": str(final),
            "scope": "current_run_owned_transient_checkpoints", "recoverable": False}
