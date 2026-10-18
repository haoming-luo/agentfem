# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the integration policy of independently named rigid contact pairs.

This is an input contract, not a numerical method. A string preserves the
existing uniform mode. A mapping must cover every pair exactly, so adding a
tool cannot silently cause it to inherit a sampled integration rule.
"""

from __future__ import annotations

from collections.abc import Mapping


_MODES = frozenset({
    "gauss", "planar_active_partition", "convex_feature_partition",
    "closed_planar_branch_partition", "reviewed_planar_branch_partition",
    "radial_active_partition",
})


def resolve_contact_integration_modes(pairs, selected) -> tuple[str, ...]:
    """Return modes in pair order after exact-coverage and spelling checks."""

    names = tuple(getattr(pair, "name", None) for pair in pairs)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("AFM-CONTACT-INTEGRATION: contact pairs need names.")
    if len(names) != len(set(names)):
        raise ValueError("AFM-CONTACT-INTEGRATION: contact pair names must be unique.")
    if isinstance(selected, str):
        if selected not in _MODES:
            raise ValueError("AFM-CONTACT-INTEGRATION: unknown contact integration mode.")
        modes = (selected,) * len(names)
    elif isinstance(selected, Mapping):
        if set(selected) != set(names):
            raise ValueError(
                "AFM-CONTACT-INTEGRATION: per-pair modes must cover exactly "
                "the registered contact pair names."
            )
        modes = tuple(selected[name] for name in names)
    else:
        raise ValueError(
            "AFM-CONTACT-INTEGRATION: expected one mode or a mapping of "
            "contact pair names to modes."
        )
    if any(not isinstance(mode, str) or mode not in _MODES for mode in modes):
        raise ValueError("AFM-CONTACT-INTEGRATION: unknown contact integration mode.")
    return modes


__all__ = ["resolve_contact_integration_modes"]
