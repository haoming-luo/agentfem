"""Classify a repository change into the minimum trustworthy CI tier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess


LEVELS = ("docs", "targeted", "core", "release")
_RANK = {level: index for index, level in enumerate(LEVELS)}

_DOCUMENTATION_FILES = {
    "README.md",
    "CONTRIBUTING.md",
    "AGENT_GUIDE.md",
    "WORKFLOW.md",
    "CONCEPTS.md",
    "CITATION.cff",
    "mkdocs.yml",
}
_DOCUMENTATION_PREFIXES = ("docs/", "site/")
_RELEASE_FILES = {
    "pyproject.toml",
    "environment.yml",
    "release_gate.py",
    "promotion_gate.py",
}
_RELEASE_PREFIXES = (
    "packaging/runtime/",
    "src/agentfem/release/",
    ".github/workflows/publish-",
    ".github/workflows/runtime-installers",
    ".github/workflows/platform-acceptance",
)
_TARGETED_SOURCE_FILES = {
    "src/agentfem/campaigns.py",
    "src/agentfem/cli.py",
    "src/agentfem/datasets.py",
    "src/agentfem/feedback.py",
    "src/agentfem/manifests.py",
    "src/agentfem/project.py",
    "src/agentfem/surrogates.py",
    "src/agentfem/upgrades.py",
}
_TARGETED_PREFIXES = (
    ".github/",
    "examples/",
    "services/",
    "src/agentfem/integrations/",
    "src/agentfem/knowledge/",
)
_ML_PREFIXES = (
    "src/agentfem/learning",
    "src/agentfem/datasets",
    "src/agentfem/surrogates",
    "tests/test_datasets.py",
    "tests/test_surrogates.py",
)


@dataclass(frozen=True)
class ValidationScope:
    level: str
    tests: tuple[str, ...] = ()
    ml: bool = False

    @property
    def numerical(self) -> bool:
        return self.level != "docs"

    def outputs(self) -> dict[str, str]:
        return {
            "level": self.level,
            "numerical": str(self.numerical).lower(),
            "ml": str(self.ml).lower(),
            "tests": " ".join(self.tests),
        }


def _at_least(current: str, candidate: str) -> str:
    return candidate if _RANK[candidate] > _RANK[current] else current


def _path_level(path: str) -> str:
    if path in _DOCUMENTATION_FILES or path.startswith(_DOCUMENTATION_PREFIXES):
        return "docs"
    if path in _RELEASE_FILES or path.startswith(_RELEASE_PREFIXES):
        return "release"
    if path.startswith("tests/"):
        filename = path.removeprefix("tests/")
        is_direct_test = (
            "/" not in filename
            and filename.startswith("test_")
            and filename.endswith(".py")
        )
        if is_direct_test and not filename.startswith("test_parallel_"):
            return "targeted"
        # Drivers, fixtures, Goldens and explicitly distributed modules do not
        # provide useful evidence when selected as a serial pytest path.
        return "core"
    if path in _TARGETED_SOURCE_FILES or path.startswith(_TARGETED_PREFIXES):
        return "targeted"
    if path.startswith("src/agentfem/"):
        return "core"
    if path in {"build_docs.py", "build_knowledge.py", "ci_validation_scope.py"}:
        return "targeted"
    # Unknown build or repository files fail safe instead of silently reducing
    # scientific validation.
    return "core"


def classify_changes(
    paths: list[str] | tuple[str, ...],
    *,
    requested: str = "auto",
) -> ValidationScope:
    """Return the minimum CI tier that covers ``paths``.

    Manual requests are authoritative.  Automatic classification is
    conservative for unknown paths, while test-only corrections remain in the
    targeted tier and therefore do not replay the complete MPI evidence chain.
    """

    if requested != "auto":
        if requested not in LEVELS:
            raise ValueError(f"Unsupported validation level: {requested!r}.")
        return ValidationScope(
            requested,
            ml=requested == "release",
        )

    normalized = tuple(
        dict.fromkeys(str(Path(path)).replace("\\", "/") for path in paths if path)
    )
    if not normalized:
        return ValidationScope("core")

    level = "docs"
    for path in normalized:
        level = _at_least(level, _path_level(path))
    tests = tuple(
        sorted(
            path
            for path in normalized
            if path.startswith("tests/test_") and path.endswith(".py")
        )
    )
    ml = level == "release" or any(path.startswith(_ML_PREFIXES) for path in normalized)
    return ValidationScope(level, tests=tests, ml=ml)


def changed_paths(base: str, head: str) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", base, head],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--requested", default="auto")
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()

    if arguments.requested == "auto" and (not arguments.base or not arguments.head):
        scope = ValidationScope("core")
    else:
        paths = (
            changed_paths(arguments.base, arguments.head)
            if arguments.base and arguments.head
            else []
        )
        scope = classify_changes(paths, requested=arguments.requested)

    outputs = scope.outputs()
    text = "\n".join(f"{key}={value}" for key, value in outputs.items()) + "\n"
    if arguments.github_output is None:
        print(text, end="")
    else:
        with arguments.github_output.open("a", encoding="utf-8") as stream:
            stream.write(text)


if __name__ == "__main__":
    main()
