import pytest

from ci_validation_scope import ValidationScope, classify_changes


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


@pytest.mark.parametrize("level", ("docs", "targeted", "core", "release"))
def test_manual_validation_level_is_authoritative(level):
    scope = classify_changes(["README.md"], requested=level)

    assert scope.level == level
    assert scope.ml is (level == "release")


def test_ml_bridge_runs_only_for_related_or_release_changes():
    assert classify_changes(["src/agentfem/surrogates.py"]).ml
    assert classify_changes(["tests/test_datasets.py"]).ml
    assert not classify_changes(["src/agentfem/operators/identity.py"]).ml


def test_empty_automatic_diff_fails_safe_to_core():
    assert classify_changes([]) == ValidationScope("core")
