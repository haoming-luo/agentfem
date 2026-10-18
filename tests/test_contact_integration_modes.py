"""Exact named-pair integration policy and fail-closed input checks."""

from types import SimpleNamespace

import pytest

from agentfem.boundary_models.contact_integration_modes import (
    resolve_contact_integration_modes,
)


def _pair(name):
    return SimpleNamespace(name=name)


def test_contact_integration_policy_resolves_named_pairs_without_implicit_fallback():
    pairs = (_pair("plane"), _pair("closed"))
    assert resolve_contact_integration_modes(pairs, "gauss") == ("gauss", "gauss")
    assert resolve_contact_integration_modes(pairs, {
        "closed": "closed_planar_branch_partition",
        "plane": "planar_active_partition",
    }) == ("planar_active_partition", "closed_planar_branch_partition")


def test_reviewed_facet_branch_is_a_named_mode_without_implicit_fallback():
    assert resolve_contact_integration_modes(
        (_pair("open_tool"),),
        {"open_tool": "reviewed_planar_branch_partition"},
    ) == ("reviewed_planar_branch_partition",)


@pytest.mark.parametrize("selected", (
    {"plane": "planar_active_partition"},
    {"plane": "planar_active_partition", "closed": "gauss", "unused": "gauss"},
    {"plane": "planar_active_partition", "closed": "invented"},
    {"plane": "planar_active_partition", "closed": None},
    "invented",
    3,
))
def test_contact_integration_policy_rejects_incomplete_or_unknown_input(selected):
    with pytest.raises(ValueError, match="AFM-CONTACT-INTEGRATION"):
        resolve_contact_integration_modes((_pair("plane"), _pair("closed")), selected)


def test_contact_integration_policy_rejects_duplicate_names():
    with pytest.raises(ValueError, match="must be unique"):
        resolve_contact_integration_modes((_pair("tool"), _pair("tool")), "gauss")
