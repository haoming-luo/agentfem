"""Human-facing views of AgentFEM's machine-readable records.

This module owns presentation only.  It does not decide capability, mutate a
result, or weaken verification.  The same records remain available in full to
agents, GUIs, and automation through ``--json``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import textwrap


_CAPABILITY_TOPICS = {
    "materials": "Constitutive models and material-point capabilities",
    "procedures": "Analysis procedures and finite-element step providers",
    "workflow": "Public workflow and project commands",
    "runtime": "Current numerical runtime",
    "extensions": "Installed extension packages",
}


def _compact(text: object, *, width: int = 92) -> str:
    return textwrap.shorten(
        " ".join(str(text).split()),
        width=width,
        placeholder="…",
    )


def _value(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        shown = ", ".join(_value(item) for item in value[:3])
        return f"[{shown}{', …' if len(value) > 3 else ''}]"
    return str(value)


def format_capabilities(record: Mapping[str, object], topic: str | None = None) -> str:
    """Return a progressive human view of the full capability record."""

    selected = None if topic is None else str(topic).strip().lower().replace("-", "_")
    materials = tuple(record.get("constitutive", ()))
    providers = tuple(record.get("step_providers", ()))
    if selected is None:
        maturity = {}
        for item in materials:
            key = str(item.get("maturity", "unspecified"))
            maturity[key] = maturity.get(key, 0) + 1
        stable = sum(
            count
            for key, count in maturity.items()
            if not key.startswith("experimental") and "contract" not in key
        )
        runtime = record.get("runtime", {})
        packages = runtime.get("packages", {}) if isinstance(runtime, Mapping) else {}
        extensions = record.get("extensions", {})
        installed = extensions.get("installed", ()) if isinstance(extensions, Mapping) else ()
        return "\n".join(
            (
                f"AgentFEM {record.get('agentfem_version', '')} capabilities",
                f"  workflow    {len(record.get('templates', ()))} project templates · "
                f"{len(record.get('commands', ()))} CLI commands",
                f"  materials   {len(materials)} declared capabilities · {stable} beyond experimental maturity",
                f"  procedures  {len(providers)} finite-element step providers",
                f"  runtime     DOLFINx {packages.get('fenics-dolfinx') or 'not found'} · "
                f"PETSc {packages.get('petsc4py') or 'not found'}",
                f"  extensions  {len(installed)} installed",
                "",
                "Explore one area:",
                "  agentfem capabilities materials|procedures|workflow|runtime|extensions",
                "  agentfem capabilities <material-name>",
                "Full machine record: agentfem capabilities --json",
            )
        )

    if selected in _CAPABILITY_TOPICS:
        if selected == "materials":
            lines = ["Materials", "  NAME                         MATURITY"]
            for item in materials:
                lines.append(
                    f"  {str(item.get('name', '')):<28} {item.get('maturity', 'unspecified')}"
                )
            lines.append("\nDetails: agentfem capabilities <material-name>")
            return "\n".join(lines)
        if selected == "procedures":
            lines = ["Analysis procedures"]
            for item in providers:
                analyses = ", ".join(item.get("analyses", ())) or "declared analysis"
                lines.append(f"  {item.get('name', '<unnamed>')}: {_compact(analyses, width=68)}")
            return "\n".join(lines)
        if selected == "workflow":
            core = record.get("public_api", {}).get("core", ())
            return "\n".join(
                (
                    "Public workflow",
                    "  Study → Model → scientific assets → model.step() → SimulationResult",
                    f"  core modules: {', '.join(core)}",
                    f"  templates: {', '.join(record.get('templates', ())) or '<none>'}",
                    "  commands: " + ", ".join(record.get("commands", ())),
                )
            )
        if selected == "runtime":
            runtime = record.get("runtime", {})
            platform = runtime.get("platform", {})
            packages = runtime.get("packages", {})
            return "\n".join(
                (
                    "Numerical runtime",
                    f"  platform: {platform.get('route', platform.get('system', 'unknown'))}",
                    f"  support: {platform.get('level', 'unknown')}",
                    f"  Python: {runtime.get('python', 'unknown')}",
                    f"  DOLFINx: {packages.get('fenics-dolfinx') or 'not installed'}",
                    f"  PETSc/petsc4py: {packages.get('petsc4py') or 'not installed'}",
                    "Full diagnostics: agentfem doctor",
                )
            )
        extensions = record.get("extensions", {})
        installed = extensions.get("installed", ()) if isinstance(extensions, Mapping) else ()
        lines = ["Installed extensions"]
        lines.extend(
            f"  {item.get('name', '<unnamed>')} ({item.get('distribution') or 'unknown package'})"
            for item in installed
        )
        if not installed:
            lines.append("  <none>")
        return "\n".join(lines)

    for item in materials:
        if str(item.get("name", "")).lower() != selected:
            continue
        lines = [
            str(item["name"]),
            f"  model: {item.get('model', '')}",
            f"  maturity: {item.get('maturity', 'unspecified')}",
            f"  scope: {_compact(item.get('available_scope', ''), width=100)}",
        ]
        limitations = tuple(item.get("limitations", ()))
        if limitations:
            lines.append("  current boundaries:")
            lines.extend(f"    - {_compact(value, width=94)}" for value in limitations)
        return "\n".join(lines)

    available = ", ".join(sorted(str(item.get("name")) for item in materials))
    raise ValueError(
        f"Unknown capability topic {topic!r}. Use one of "
        f"{tuple(_CAPABILITY_TOPICS)} or a material name: {available}."
    )


def format_record(record: Mapping[str, object], path: Path) -> str:
    """Summarize a result, execution, or pointer without dumping raw JSON."""

    schema = record.get("schema")
    if schema == "agentfem.simulation-result":
        return format_result(record, path)
    lines = [f"AgentFEM record · {schema or 'unknown'}"]
    for key in ("project", "run_name", "run_number", "run_id", "status", "stage"):
        value = record.get(key)
        if value is not None:
            lines.append(f"  {key.replace('_', ' ')}: {_value(value)}")
    error = record.get("error")
    if isinstance(error, Mapping):
        code = error.get("code")
        message = error.get("message")
        if code:
            lines.append(f"  error: {code}")
        if message:
            lines.append(f"  reason: {_compact(message)}")
    lines.append(f"  record: {path}")
    return "\n".join(lines)


def format_result(record: Mapping[str, object], path: Path, *, maximum_quantities: int = 6) -> str:
    """Return the small result view most users need after a run."""

    verification = record.get("verification")
    accepted = None
    policy = None
    if isinstance(verification, Mapping):
        accepted = verification.get("acceptable")
        policy = verification.get("quality_policy")
    lines = [
        f"AgentFEM result · {record.get('name', 'simulation')}",
        f"  status: {record.get('status', 'unknown')}",
        f"  trust: {record.get('trust_level', 'unrated')}",
    ]
    if accepted is not None:
        suffix = f" ({policy})" if policy else ""
        lines.append(f"  verification: {'accepted' if accepted else 'not accepted'}{suffix}")

    quantities = tuple(record.get("quantity_records", ()))
    scalar = [item for item in quantities if item.get("shape") in (None, [], ())]
    priorities = (
        "relative_force_balance_error",
        "relative_energy_balance_error",
        "energy_balance_error",
        "displacement_max_abs",
        "temperature_maximum",
        "temperature_minimum",
        "reaction_force",
        "maximum",
    )
    ordered = sorted(
        scalar,
        key=lambda item: next(
            (index for index, name in enumerate(priorities) if name in str(item.get("name", ""))),
            len(priorities),
        ),
    )[:maximum_quantities]
    if ordered:
        lines.append("  key results:")
        for item in ordered:
            unit = f" {item['unit']}" if item.get("unit") else ""
            lines.append(f"    {item.get('name')}: {_value(item.get('value'))}{unit}")

    fields = tuple(item.get("name") for item in record.get("field_records", ()))
    if fields:
        lines.append("  fields: " + ", ".join(str(item) for item in fields))
    artifacts = record.get("artifacts", {})
    if isinstance(artifacts, Mapping) and artifacts:
        lines.append("  files: " + ", ".join(sorted(str(value) for value in artifacts.values())))
    lines.append(f"  record: {path}")
    return "\n".join(lines)


def result_summary_markdown(record: Mapping[str, object], path: Path) -> str:
    """Create a disposable human index whose source of truth is result.json."""

    text = format_result(record, path)
    return "# AgentFEM run summary\n\n```text\n" + text + "\n```\n\n"


def format_run_table(records: Sequence[Mapping[str, object]]) -> str:
    """Format recent execution records as a compact, stable table."""

    if not records:
        return "No AgentFEM runs found."
    lines = ["Recent AgentFEM runs", "  #    STATUS      NAME                 RUN"]
    for index, item in enumerate(records, start=1):
        name = str(item.get("run_name") or item.get("project") or "run")
        identifier = str(item.get("directory_name") or item.get("run_id") or "unknown")
        lines.append(
            f"  {index:<4} {str(item.get('status', 'unknown')):<11} "
            f"{_compact(name, width=20):<20} {_compact(identifier, width=34)}"
        )
    lines.append("\nInspect the latest result: agentfem show latest")
    return "\n".join(lines)


__all__ = [
    "format_capabilities",
    "format_record",
    "format_result",
    "format_run_table",
    "result_summary_markdown",
]
