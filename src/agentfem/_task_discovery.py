# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic bilingual task lookup over generated installed knowledge."""
from importlib import resources
import json
import re
from . import __version__


def _normalize(text):
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


def task_record(query):
    tasks = json.loads(resources.files("agentfem.knowledge").joinpath("catalog.json").read_text())["tasks"]
    query = str(query).strip()
    if not query:
        raise ValueError("Task query cannot be empty.")
    exact = query.removeprefix("task:")
    matches = [task for task in tasks if task["id"] == exact]
    if query.startswith("task:") and not matches:
        raise ValueError(f"Unknown task ID {exact!r}; available: {[task['id'] for task in tasks]}")
    if not matches:
        normalized = _normalize(query)
        ranked = []
        for task in tasks:
            terms = [_normalize(term) for term in [task["title"], *task["aliases"]]]
            score = max((len(term) if term and term in normalized else
                len(normalized)/2 if normalized and normalized in term else 0) for term in terms)
            if score:
                ranked.append((score, task))
        matches = [task for _, task in sorted(ranked, key=lambda row: (-row[0], row[1]["id"]))[:3]]
    return {"schema": "agentfem.task-guidance", "schema_version": "0.1.0",
        "agentfem_version": __version__, "query": query, "status": "matched" if matches else "unmatched",
        "matches": matches, "available_tasks": [task["id"] for task in tasks],
        "execution_checked": False, "case_executed": False,
        "next": "Inspect --api for signatures; use models.step_capability for the actual model/option combination. "
            "No search match does not mean a physical capability is unsupported."}


def format_task(record):
    lines = [f"AgentFEM task guidance: {record['status']}"]
    for task in record["matches"]:
        lines += [f"{task['id']}: {task['title']}", f"  requires: {task['prerequisites']}",
            f"  limits: {task['limitations']}", task['snippet']]
    lines.append(record['next'])
    return "\n".join(lines)
