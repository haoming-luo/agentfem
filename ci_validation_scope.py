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
    "src/agentfem/campaigns/",
    "src/agentfem/datasets/",
    "src/agentfem/integrations/",
    "src/agentfem/knowledge/",
    "src/agentfem/surrogates/",
)
_SOURCE_TEST_MAP = {
    "src/agentfem/campaigns/": ("tests/test_campaigns.py",),
    "src/agentfem/datasets/": (
        "tests/test_datasets.py",
        "tests/test_field_datasets.py",
    ),
    "src/agentfem/surrogates/": ("tests/test_surrogates.py",),
    "src/agentfem/feedback.py": ("tests/test_feedback.py",),
    "src/agentfem/community.py": ("tests/test_community.py",),
    "src/agentfem/project.py": ("tests/test_project_cli.py",),
    "src/agentfem/cli.py": (
        "tests/test_project_cli.py",
        "tests/test_common_workflows.py",
    ),
}
_CORE_SOURCE_TEST_MAP = {
    "src/agentfem/_api_contract.py": (
        "tests/test_documentation.py",
        "tests/test_project_cli.py",
    ),
    "src/agentfem/_model_validation.py": (
        "tests/test_element_contracts.py",
        "tests/test_validation.py",
    ),
    "src/agentfem/_solver_lifecycle.py": (
        "tests/test_prepared_linear_problem.py",
        "tests/test_harmonic.py",
    ),
    "src/agentfem/_hybrid_nonlinear.py": (
        "tests/test_hybrid_nonlinear.py",
    ),
    "src/agentfem/_nonlinear_problems.py": (
        "tests/test_common_workflows.py",
        "tests/test_p1_platform.py",
    ),
    "src/agentfem/_transient_problems.py": (
        "tests/test_common_workflows.py",
        "tests/test_transient_heat_decay_workflow.py",
        "tests/test_transient_restart.py",
    ),
    "src/agentfem/constitutive/user_material.py": (
        "tests/test_user_material.py",
    ),
    "src/agentfem/constitutive/__init__.py": (
        "tests/test_constitutive_models.py",
        "tests/test_user_material.py",
    ),
    "src/agentfem/constitutive/material_driver.py": (
        "tests/test_user_material.py",
        "tests/test_finite_strain_plasticity.py",
        "tests/test_finite_strain_j2_material_map.py",
    ),
    "src/agentfem/mechanics/_incremental_runtime.py": (
        "tests/test_incremental_runtime.py",
        "tests/test_small_strain_material_step.py",
        "tests/test_p1_platform.py",
        "tests/test_viscoelasticity.py",
    ),
    "src/agentfem/mechanics/small_strain_material.py": (
        "tests/test_small_strain_material_step.py",
        "tests/test_learned_constitutive.py",
    ),
    "src/agentfem/mechanics/plasticity.py": (
        "tests/test_external_inelastic_benchmark.py",
        "tests/test_p1_platform.py",
    ),
    "src/agentfem/mechanics/creep.py": (
        "tests/test_external_inelastic_benchmark.py",
        "tests/test_p1_platform.py",
    ),
    "src/agentfem/mechanics/viscoelasticity.py": (
        "tests/test_viscoelasticity.py",
    ),
    "src/agentfem/mechanics/finite_strain_plasticity.py": (
        "tests/test_finite_strain_j2_mixed.py",
        "tests/test_finite_strain_j2_periodic.py",
        "tests/test_finite_strain_j2_standard.py",
    ),
    "src/agentfem/mechanics/harmonic.py": (
        "tests/test_harmonic.py",
        "tests/test_harmonic_backend_evidence.py",
    ),
    "src/agentfem/events.py": (
        "tests/test_solvers.py",
        "tests/test_results.py",
        "tests/test_harmonic.py",
    ),
    "src/agentfem/elements/": (
        "tests/test_element_contracts.py",
        "tests/test_validation.py",
    ),
    "src/agentfem/fracture": ("tests/test_fracture_v5.py",),
    "src/agentfem/mesh/": (
        "tests/test_element_contracts.py",
        "tests/test_mesh_formats.py",
        "tests/test_mesh_quality.py",
        "tests/test_mixed_cell_topologies.py",
    ),
    "src/agentfem/operators/": (
        "tests/test_operators.py",
        "tests/test_common_workflows.py",
    ),
    "src/agentfem/problems.py": (
        "tests/test_common_workflows.py",
        "tests/test_results.py",
    ),
    "src/agentfem/results": (
        "tests/test_results.py",
    ),
    "src/agentfem/solvers.py": (
        "tests/test_solvers.py",
        "tests/test_prepared_linear_problem.py",
        "tests/test_common_workflows.py",
    ),
    "src/agentfem/step_providers.py": (
        "tests/test_common_workflows.py",
        "tests/test_validation.py",
    ),
    "src/agentfem/_architecture_contract.py": (
        "tests/test_architecture_contract.py",
    ),
}
_CORE_SOURCE_MPI_TEST_MAP = {
    "src/agentfem/_model_validation.py": (
        "tests/test_element_contracts.py",
    ),
    "src/agentfem/_hybrid_nonlinear.py": (
        "tests/test_parallel_mixed.py",
    ),
    "src/agentfem/_nonlinear_problems.py": (
        "tests/test_parallel_affine.py",
        "tests/test_parallel_inelastic.py",
    ),
    "src/agentfem/_transient_problems.py": (
        "tests/test_parallel_transient.py",
    ),
    "src/agentfem/constitutive/material_driver.py": (
        "tests/test_parallel_inelastic.py",
    ),
    "src/agentfem/mechanics/_incremental_runtime.py": (
        "tests/test_parallel_inelastic.py",
        "tests/test_parallel_viscoelasticity.py",
    ),
    "src/agentfem/mechanics/small_strain_material.py": (
        "tests/test_parallel_learning.py",
        "tests/test_parallel_inelastic.py",
    ),
    "src/agentfem/mechanics/plasticity.py": (
        "tests/test_parallel_inelastic.py",
    ),
    "src/agentfem/mechanics/creep.py": (
        "tests/test_parallel_inelastic.py",
    ),
    "src/agentfem/mechanics/viscoelasticity.py": (
        "tests/test_parallel_viscoelasticity.py",
    ),
    "src/agentfem/mechanics/finite_strain_plasticity.py": (
        "tests/test_parallel_inelastic.py",
        "tests/test_parallel_mixed.py",
    ),
    "src/agentfem/mechanics/harmonic.py": (
        "tests/test_parallel_viscoelasticity.py",
    ),
    "src/agentfem/elements/": ("tests/test_element_contracts.py",),
    "src/agentfem/fracture": ("tests/test_parallel_cohesive.py",),
    "src/agentfem/mesh/": ("tests/test_element_contracts.py",),
    "src/agentfem/solvers.py": (
        "tests/test_parallel_affine.py",
        "tests/test_parallel_results.py",
    ),
    "src/agentfem/problems.py": (
        "tests/test_parallel_results.py",
        "tests/test_parallel_transient.py",
    ),
    "src/agentfem/results": ("tests/test_parallel_results.py",),
}
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
    mpi_tests: tuple[str, ...] = ()
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
            "mpi_tests": " ".join(self.mpi_tests),
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
    selected_tests = {
        path
        for path in normalized
        if path.startswith("tests/test_") and path.endswith(".py")
    }
    mpi_tests: set[str] = set()
    unmapped_core_source = False
    if level in {"targeted", "core"}:
        for path in normalized:
            for prefix, mapped_tests in _SOURCE_TEST_MAP.items():
                if path == prefix or path.startswith(prefix):
                    selected_tests.update(mapped_tests)
            for prefix, mapped_tests in _CORE_SOURCE_TEST_MAP.items():
                if path == prefix or path.startswith(prefix):
                    selected_tests.update(mapped_tests)
            for prefix, mapped_tests in _CORE_SOURCE_MPI_TEST_MAP.items():
                if path == prefix or path.startswith(prefix):
                    mpi_tests.update(mapped_tests)
            if (
                level == "core"
                and path.startswith("src/agentfem/")
                and _path_level(path) == "core"
                and not any(
                    path == prefix or path.startswith(prefix)
                    for prefix in _CORE_SOURCE_TEST_MAP
                )
            ):
                unmapped_core_source = True
    if unmapped_core_source:
        # An unknown core owner cannot be covered by an invented partial set;
        # fail safe to the complete release evidence ladder.
        level = "release"
        selected_tests.clear()
        mpi_tests.clear()
    tests = tuple(sorted(selected_tests))
    ml = level == "release" or any(path.startswith(_ML_PREFIXES) for path in normalized)
    return ValidationScope(
        level,
        tests=tests,
        mpi_tests=tuple(sorted(mpi_tests)),
        ml=ml,
    )


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
