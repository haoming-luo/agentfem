"""Validate AgentFEM scientific knowledge assets and build reference outputs.

Scientific Function Cards are the single source of truth for public scientific
semantics.  This script validates those machine-readable cards and generates a
human reference manual plus a compact agent-readable catalog.  Generated files
are committed so the repository, release archives, and documentation site carry
the same reviewed knowledge.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"
CARD_DIR = KNOWLEDGE_DIR / "cards"
BENCHMARK_DIR = KNOWLEDGE_DIR / "benchmarks"
CATALOG_PATH = KNOWLEDGE_DIR / "catalog.json"
REFERENCE_PATH = ROOT / "docs" / "reference" / "scientific_function_reference.md"

CARD_SCHEMA = "agentfem.scientific-function-card"
CARD_SCHEMA_VERSION = "0.1.0"
BENCHMARK_SCHEMA = "agentfem.benchmark-card"
BENCHMARK_SCHEMA_VERSION = "0.1.0"
CATALOG_SCHEMA = "agentfem.scientific-function-catalog"
CATALOG_SCHEMA_VERSION = "0.1.0"

CARD_KINDS = {
    "analysis_step",
    "constraint",
    "data_asset",
    "learned_model",
    "material",
    "operator",
    "workflow",
}
CARD_STATUSES = {"supported", "experimental", "contract_only", "deprecated"}
ID_PATTERN = re.compile(r"^agentfem\.[a-z0-9][a-z0-9_.-]*$")


class KnowledgeValidationError(ValueError):
    """Raised when one or more knowledge assets violate their contract."""


def _read_records(directory: Path) -> list[tuple[Path, dict[str, object]]]:
    records: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise KnowledgeValidationError(f"{path.relative_to(ROOT)}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise KnowledgeValidationError(
                f"{path.relative_to(ROOT)}: top-level value must be an object"
            )
        records.append((path, value))
    if not records:
        raise KnowledgeValidationError(f"{directory.relative_to(ROOT)} contains no JSON assets")
    return records


def _nonempty_string(value, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path} must be a non-empty string")


def _string_list(
    value,
    path: str,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        errors.append(f"{path} must be {qualifier} of strings")
        return
    for index, item in enumerate(value):
        _nonempty_string(item, f"{path}[{index}]", errors)


def _mapping(value, path: str, errors: list[str]) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return None
    return value


def _validate_io_rows(value, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{path} must be a non-empty list")
        return
    for index, row in enumerate(value):
        selected = _mapping(row, f"{path}[{index}]", errors)
        if selected is None:
            continue
        for key in ("name", "type", "unit_role", "description"):
            _nonempty_string(selected.get(key), f"{path}[{index}].{key}", errors)


def _validate_card(
    source: Path,
    card: Mapping[str, object],
    benchmark_ids: set[str],
) -> list[str]:
    label = str(source.relative_to(ROOT))
    errors: list[str] = []
    required = {
        "schema",
        "schema_version",
        "id",
        "title",
        "kind",
        "status",
        "summary",
        "public_api",
        "implementation",
        "science",
        "usage",
        "verification",
        "references",
        "related_cards",
    }
    missing = sorted(required - set(card))
    if missing:
        errors.append(f"{label}: missing fields {missing}")
        return errors
    if card["schema"] != CARD_SCHEMA:
        errors.append(f"{label}.schema must be {CARD_SCHEMA!r}")
    if card["schema_version"] != CARD_SCHEMA_VERSION:
        errors.append(f"{label}.schema_version must be {CARD_SCHEMA_VERSION!r}")
    _nonempty_string(card["id"], f"{label}.id", errors)
    if isinstance(card["id"], str) and not ID_PATTERN.fullmatch(card["id"]):
        errors.append(f"{label}.id is not a stable AgentFEM knowledge ID")
    _nonempty_string(card["title"], f"{label}.title", errors)
    if card["kind"] not in CARD_KINDS:
        errors.append(f"{label}.kind must be one of {sorted(CARD_KINDS)}")
    if card["status"] not in CARD_STATUSES:
        errors.append(f"{label}.status must be one of {sorted(CARD_STATUSES)}")
    _nonempty_string(card["summary"], f"{label}.summary", errors)
    _string_list(card["public_api"], f"{label}.public_api", errors)
    _string_list(card["implementation"], f"{label}.implementation", errors)
    if isinstance(card["implementation"], list):
        for path in card["implementation"]:
            if isinstance(path, str) and not (ROOT / path).is_file():
                errors.append(f"{label}.implementation path does not exist: {path}")

    science = _mapping(card["science"], f"{label}.science", errors)
    if science is not None:
        for key in (
            "statement",
            "equations",
            "inputs",
            "outputs",
            "assumptions",
            "conventions",
            "applicability",
            "limitations",
        ):
            if key not in science:
                errors.append(f"{label}.science missing {key!r}")
        _nonempty_string(science.get("statement"), f"{label}.science.statement", errors)
        equations = science.get("equations")
        if not isinstance(equations, list) or not equations:
            errors.append(f"{label}.science.equations must be a non-empty list")
        else:
            for index, equation in enumerate(equations):
                row = _mapping(
                    equation,
                    f"{label}.science.equations[{index}]",
                    errors,
                )
                if row is not None:
                    for key in ("label", "expression", "description"):
                        _nonempty_string(
                            row.get(key),
                            f"{label}.science.equations[{index}].{key}",
                            errors,
                        )
        _validate_io_rows(science.get("inputs"), f"{label}.science.inputs", errors)
        _validate_io_rows(science.get("outputs"), f"{label}.science.outputs", errors)
        for key in ("assumptions", "conventions", "applicability", "limitations"):
            _string_list(
                science.get(key),
                f"{label}.science.{key}",
                errors,
                allow_empty=(key == "limitations"),
            )

    usage = _mapping(card["usage"], f"{label}.usage", errors)
    if usage is not None:
        _nonempty_string(
            usage.get("minimal_example"),
            f"{label}.usage.minimal_example",
            errors,
        )
        _string_list(usage.get("examples"), f"{label}.usage.examples", errors)
        if isinstance(usage.get("examples"), list):
            for path in usage["examples"]:
                if isinstance(path, str) and not (ROOT / path).is_file():
                    errors.append(f"{label}.usage example does not exist: {path}")

    verification = _mapping(card["verification"], f"{label}.verification", errors)
    if verification is not None:
        _string_list(verification.get("tests"), f"{label}.verification.tests", errors)
        if isinstance(verification.get("tests"), list):
            for path in verification["tests"]:
                if isinstance(path, str) and not (ROOT / path).is_file():
                    errors.append(f"{label}.verification test does not exist: {path}")
        _string_list(
            verification.get("benchmarks"),
            f"{label}.verification.benchmarks",
            errors,
            allow_empty=True,
        )
        if isinstance(verification.get("benchmarks"), list):
            for benchmark_id in verification["benchmarks"]:
                if isinstance(benchmark_id, str) and benchmark_id not in benchmark_ids:
                    errors.append(
                        f"{label}.verification references unknown benchmark {benchmark_id!r}"
                    )
        _string_list(
            verification.get("validation_rules"),
            f"{label}.verification.validation_rules",
            errors,
        )

    references = card["references"]
    if not isinstance(references, list) or not references:
        errors.append(f"{label}.references must be a non-empty list")
    else:
        for index, reference in enumerate(references):
            row = _mapping(reference, f"{label}.references[{index}]", errors)
            if row is None:
                continue
            _nonempty_string(
                row.get("title"),
                f"{label}.references[{index}].title",
                errors,
            )
            _nonempty_string(
                row.get("locator"),
                f"{label}.references[{index}].locator",
                errors,
            )
            locator = row.get("locator")
            if (
                isinstance(locator, str)
                and "://" not in locator
                and not locator.startswith("doi:")
                and not (ROOT / locator).exists()
            ):
                errors.append(f"{label}.references locator does not exist: {locator}")
    _string_list(
        card["related_cards"],
        f"{label}.related_cards",
        errors,
        allow_empty=True,
    )
    return errors


def _validate_benchmark(source: Path, benchmark: Mapping[str, object]) -> list[str]:
    label = str(source.relative_to(ROOT))
    errors: list[str] = []
    required = {
        "schema",
        "schema_version",
        "id",
        "title",
        "status",
        "purpose",
        "physics",
        "execution",
        "observables",
        "acceptance",
        "limitations",
    }
    missing = sorted(required - set(benchmark))
    if missing:
        errors.append(f"{label}: missing fields {missing}")
        return errors
    if benchmark["schema"] != BENCHMARK_SCHEMA:
        errors.append(f"{label}.schema must be {BENCHMARK_SCHEMA!r}")
    if benchmark["schema_version"] != BENCHMARK_SCHEMA_VERSION:
        errors.append(
            f"{label}.schema_version must be {BENCHMARK_SCHEMA_VERSION!r}"
        )
    for key in ("id", "title", "status", "purpose", "physics"):
        _nonempty_string(benchmark[key], f"{label}.{key}", errors)
    execution = _mapping(benchmark["execution"], f"{label}.execution", errors)
    if execution is not None:
        for key in ("command", "example", "evidence"):
            _nonempty_string(execution.get(key), f"{label}.execution.{key}", errors)
        example = execution.get("example")
        if isinstance(example, str) and not (ROOT / example).is_file():
            errors.append(f"{label}.execution example does not exist: {example}")
    _string_list(benchmark["observables"], f"{label}.observables", errors)
    _string_list(benchmark["acceptance"], f"{label}.acceptance", errors)
    _string_list(
        benchmark["limitations"],
        f"{label}.limitations",
        errors,
        allow_empty=True,
    )
    return errors


def _resolve_public_api(path: str):
    parts = path.split(".")
    for pivot in range(len(parts), 0, -1):
        try:
            value = importlib.import_module(".".join(parts[:pivot]))
        except ModuleNotFoundError:
            continue
        for part in parts[pivot:]:
            value = getattr(value, part)
        return value
    raise ImportError(f"cannot resolve public API path {path!r}")


def validate_assets(
    cards: Iterable[tuple[Path, Mapping[str, object]]],
    benchmarks: Iterable[tuple[Path, Mapping[str, object]]],
    *,
    check_imports: bool = False,
) -> None:
    selected_cards = list(cards)
    selected_benchmarks = list(benchmarks)
    benchmark_ids = {
        value.get("id")
        for _, value in selected_benchmarks
        if isinstance(value.get("id"), str)
    }
    errors: list[str] = []
    for source, benchmark in selected_benchmarks:
        errors.extend(_validate_benchmark(source, benchmark))
    for source, card in selected_cards:
        errors.extend(_validate_card(source, card, benchmark_ids))

    card_ids = [card.get("id") for _, card in selected_cards]
    duplicates = sorted(
        {
            card_id
            for card_id in card_ids
            if isinstance(card_id, str) and card_ids.count(card_id) > 1
        }
    )
    if duplicates:
        errors.append(f"duplicate scientific function card IDs: {duplicates}")
    known_cards = {value for value in card_ids if isinstance(value, str)}
    for source, card in selected_cards:
        for related in card.get("related_cards", []):
            if related not in known_cards:
                errors.append(
                    f"{source.relative_to(ROOT)} references unknown related card {related!r}"
                )
        if check_imports:
            for api_path in card.get("public_api", []):
                try:
                    _resolve_public_api(api_path)
                except (AttributeError, ImportError, ModuleNotFoundError) as exc:
                    errors.append(
                        f"{source.relative_to(ROOT)} cannot resolve {api_path!r}: {exc}"
                    )
    if errors:
        raise KnowledgeValidationError(
            "Scientific knowledge validation failed:\n- " + "\n- ".join(errors)
        )


def build_catalog(
    cards: Iterable[tuple[Path, Mapping[str, object]]],
    benchmarks: Iterable[tuple[Path, Mapping[str, object]]],
) -> str:
    card_rows = []
    for source, card in cards:
        card_rows.append(
            {
                "id": card["id"],
                "title": card["title"],
                "kind": card["kind"],
                "status": card["status"],
                "summary": card["summary"],
                "public_api": card["public_api"],
                "source": str(source.relative_to(ROOT)),
                "related_cards": card["related_cards"],
            }
        )
    benchmark_rows = []
    for source, benchmark in benchmarks:
        benchmark_rows.append(
            {
                "id": benchmark["id"],
                "title": benchmark["title"],
                "status": benchmark["status"],
                "physics": benchmark["physics"],
                "source": str(source.relative_to(ROOT)),
            }
        )
    record = {
        "schema": CATALOG_SCHEMA,
        "schema_version": CATALOG_SCHEMA_VERSION,
        "cards": sorted(card_rows, key=lambda item: item["id"]),
        "benchmarks": sorted(benchmark_rows, key=lambda item: item["id"]),
    }
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def _markdown_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    header = rows[0]
    result = [
        "| " + " | ".join(_escape_cell(value) for value in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    result.extend(
        "| " + " | ".join(_escape_cell(value) for value in row) + " |"
        for row in rows[1:]
    )
    return result


def _escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _bullets(values: Iterable[str]) -> list[str]:
    selected = list(values)
    return ["- None declared."] if not selected else [f"- {value}" for value in selected]


def build_reference(
    cards: Iterable[tuple[Path, Mapping[str, object]]],
    benchmarks: Iterable[tuple[Path, Mapping[str, object]]],
) -> str:
    selected_cards = sorted(cards, key=lambda item: item[1]["id"])
    selected_benchmarks = sorted(benchmarks, key=lambda item: item[1]["id"])
    lines = [
        "# AgentFEM Scientific Function Reference",
        "",
        "<!-- Generated by build_knowledge.py; edit knowledge/cards instead. -->",
        "",
        "This reference is generated from versioned Scientific Function Cards. It",
        "is simultaneously a human manual, a review surface, and the source for",
        "the compact machine-readable `knowledge/catalog.json`.",
        "",
        "## Function index",
        "",
    ]
    index_rows = [["Stable ID", "Title", "Kind", "Status"]]
    for _, card in selected_cards:
        index_rows.append(
            [
                f"[`{card['id']}`](#{card['id'].replace('.', '-')})",
                card["title"],
                card["kind"],
                card["status"],
            ]
        )
    lines.extend(_markdown_table(index_rows))
    lines.extend(["", "## Benchmark index", ""])
    benchmark_rows = [["Stable ID", "Title", "Physics", "Status"]]
    for _, benchmark in selected_benchmarks:
        benchmark_rows.append(
            [
                f"`{benchmark['id']}`",
                benchmark["title"],
                benchmark["physics"],
                benchmark["status"],
            ]
        )
    lines.extend(_markdown_table(benchmark_rows))

    for source, card in selected_cards:
        science = card["science"]
        usage = card["usage"]
        verification = card["verification"]
        lines.extend(
            [
                "",
                f"## {card['title']}",
                "",
                f"**Stable ID:** `{card['id']}`  ",
                f"**Kind:** `{card['kind']}`  ",
                f"**Status:** `{card['status']}`  ",
                f"**Source card:** `{source.relative_to(ROOT)}`",
                "",
                card["summary"],
                "",
                "### Public API",
                "",
                *_bullets(f"`{value}`" for value in card["public_api"]),
                "",
                "### Scientific contract",
                "",
                science["statement"],
            ]
        )
        for equation in science["equations"]:
            lines.extend(
                [
                    "",
                    f"**{equation['label']}**",
                    "",
                    "$$",
                    equation["expression"],
                    "$$",
                    "",
                    equation["description"],
                ]
            )
        lines.extend(["", "#### Inputs", ""])
        input_rows = [["Name", "Type", "Unit role", "Meaning"]]
        input_rows.extend(
            [
                row["name"],
                row["type"],
                row["unit_role"],
                row["description"],
            ]
            for row in science["inputs"]
        )
        lines.extend(_markdown_table(input_rows))
        lines.extend(["", "#### Outputs", ""])
        output_rows = [["Name", "Type", "Unit role", "Meaning"]]
        output_rows.extend(
            [
                row["name"],
                row["type"],
                row["unit_role"],
                row["description"],
            ]
            for row in science["outputs"]
        )
        lines.extend(_markdown_table(output_rows))
        for heading, key in (
            ("Assumptions", "assumptions"),
            ("Conventions", "conventions"),
            ("Applicability", "applicability"),
            ("Limitations", "limitations"),
        ):
            lines.extend(["", f"#### {heading}", "", *_bullets(science[key])])
        lines.extend(
            [
                "",
                "### Minimal example",
                "",
                "```python",
                usage["minimal_example"].rstrip(),
                "```",
                "",
                "### Verification",
                "",
                "**Tests**",
                "",
                *_bullets(f"`{value}`" for value in verification["tests"]),
                "",
                "**Benchmarks**",
                "",
                *_bullets(f"`{value}`" for value in verification["benchmarks"]),
                "",
                "**Validation rules**",
                "",
                *_bullets(verification["validation_rules"]),
                "",
                "### References",
                "",
                *[
                    f"- {reference['title']}: `{reference['locator']}`"
                    for reference in card["references"]
                ],
            ]
        )
    lines.append("")
    return "\n".join(lines)


def generated_outputs(*, check_imports: bool = False) -> dict[Path, str]:
    cards = _read_records(CARD_DIR)
    benchmarks = _read_records(BENCHMARK_DIR)
    validate_assets(cards, benchmarks, check_imports=check_imports)
    return {
        CATALOG_PATH: build_catalog(cards, benchmarks),
        REFERENCE_PATH: build_reference(cards, benchmarks),
    }


def write_outputs(outputs: Mapping[Path, str]) -> None:
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def check_outputs(outputs: Mapping[Path, str]) -> None:
    stale = [
        str(path.relative_to(ROOT))
        for path, content in outputs.items()
        if not path.is_file() or path.read_text() != content
    ]
    if stale:
        raise KnowledgeValidationError(
            "Generated knowledge outputs are stale or missing: "
            + ", ".join(stale)
            + ". Run python build_knowledge.py."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when generated knowledge outputs differ from committed files",
    )
    parser.add_argument(
        "--check-imports",
        action="store_true",
        help="also import every declared public API path",
    )
    args = parser.parse_args(argv)
    outputs = generated_outputs(check_imports=args.check_imports)
    if args.check:
        check_outputs(outputs)
        print(f"Validated {len(_read_records(CARD_DIR))} scientific function cards.")
    else:
        write_outputs(outputs)
        print(f"Built AgentFEM scientific reference: {REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
