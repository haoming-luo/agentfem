from pathlib import Path

import pytest

from ci_validation_scope import (
    _CORE_SOURCE_MPI_TEST_MAP,
    _CORE_SOURCE_TEST_MAP,
    _SOURCE_TEST_MAP,
    ValidationScope,
    classify_changes,
)


@pytest.mark.parametrize(
    ("paths", "expected"),
    (
        (("README.md", "docs/guide/index.md"), "docs"),
        (("tests/test_models.py",), "targeted"),
        (("src/agentfem/cli.py", "tests/test_project_cli.py"), "targeted"),
        (("src/agentfem/operators/identity.py",), "core"),
        (("src/agentfem/results.py", "docs/guide/results.md"), "core"),
        (("tests/test_parallel_results.py",), "core"),
        (("tests/portable_checkpoint_driver.py",), "core"),
        (("tests/goldens/abaqus_migration_plan.json",), "core"),
        (("pyproject.toml",), "release"),
        (("unclassified-build-input.txt",), "core"),
    ),
)
def test_change_scope_uses_the_minimum_trustworthy_tier(paths, expected):
    assert classify_changes(paths).level == expected


def test_test_only_fix_returns_stable_target_list_without_requesting_full_ci():
    scope = classify_changes(
        [
            "tests/test_p1_platform.py",
            "docs/agentfem.json",
            "tests/test_hybrid_nonlinear.py",
            "tests/test_p1_platform.py",
        ]
    )

    assert scope == ValidationScope(
        "targeted",
        tests=(
            "tests/test_hybrid_nonlinear.py",
            "tests/test_p1_platform.py",
        ),
    )


def test_targeted_source_change_selects_stable_owner_tests():
    scope = classify_changes(
        [
            "src/agentfem/campaigns/core.py",
            "src/agentfem/cli.py",
        ]
    )

    assert scope.level == "targeted"
    assert scope.tests == (
        "tests/test_campaigns.py",
        "tests/test_common_workflows.py",
        "tests/test_project_cli.py",
    )


def test_known_core_change_selects_serial_and_distributed_owner_suites():
    scope = classify_changes(
        [
            "src/agentfem/constitutive/material_driver.py",
            "src/agentfem/mechanics/viscoelasticity.py",
        ]
    )

    assert scope.level == "core"
    assert scope.tests == (
        "tests/test_finite_strain_j2_material_map.py",
        "tests/test_finite_strain_plasticity.py",
        "tests/test_user_material.py",
        "tests/test_viscoelasticity.py",
    )
    assert scope.mpi_tests == (
        "tests/test_parallel_inelastic.py",
        "tests/test_parallel_viscoelasticity.py",
    )


def test_discretization_owners_select_focused_serial_and_mpi_evidence():
    scope = classify_changes(
        [
            "src/agentfem/_api_contract.py",
            "src/agentfem/_model_validation.py",
            "src/agentfem/elements/__init__.py",
            "src/agentfem/mesh/compatibility.py",
        ]
    )

    assert scope.level == "core"
    assert scope.tests == (
        "tests/test_documentation.py",
        "tests/test_element_contracts.py",
        "tests/test_mesh_formats.py",
        "tests/test_mesh_quality.py",
        "tests/test_mixed_cell_topologies.py",
        "tests/test_project_cli.py",
        "tests/test_validation.py",
    )
    assert scope.mpi_tests == ("tests/test_element_contracts.py",)
    assert not scope.ml


def test_unknown_core_change_fails_safe_to_complete_release_validation():
    scope = classify_changes(["src/agentfem/unmapped_core.py"])

    assert scope.level == "release"
    assert scope.tests == ()
    assert scope.mpi_tests == ()


def test_every_declared_owner_test_exists_in_the_repository():
    mappings = (
        _SOURCE_TEST_MAP,
        _CORE_SOURCE_TEST_MAP,
        _CORE_SOURCE_MPI_TEST_MAP,
    )
    selected = {
        path
        for mapping in mappings
        for paths in mapping.values()
        for path in paths
    }

    missing = tuple(path for path in sorted(selected) if not Path(path).is_file())

    assert missing == ()


@pytest.mark.parametrize("level", ("docs", "targeted", "core", "release"))
def test_manual_validation_level_is_authoritative(level):
    scope = classify_changes(["README.md"], requested=level)

    assert scope.level == level
    assert scope.ml is (level == "release")


def test_ml_bridge_runs_only_for_related_or_release_changes():
    assert classify_changes(["src/agentfem/surrogates.py"]).ml
    assert classify_changes(["tests/test_datasets.py"]).ml
    assert not classify_changes(["src/agentfem/operators/identity.py"]).ml


def test_executable_identity_owner_selects_serial_and_mpi_evidence():
    scope = classify_changes(["src/agentfem/operators/identity.py"])

    assert scope.level == "core"
    assert scope.tests == (
        "tests/test_common_workflows.py",
        "tests/test_dynamics.py",
        "tests/test_harmonic.py",
        "tests/test_operators.py",
        "tests/test_provenance.py",
    )
    assert scope.mpi_tests == ("tests/test_parallel_operator_identity.py",)


def test_checkpoint_owner_selects_restart_and_portability_evidence():
    scope = classify_changes(["src/agentfem/checkpointing.py"])

    assert scope.level == "core"
    assert scope.tests == ("tests/test_transient_restart.py",)
    assert scope.mpi_tests == ("tests/test_parallel_transient.py",)


def test_empty_automatic_diff_fails_safe_to_core():
    assert classify_changes([]) == ValidationScope("core")
