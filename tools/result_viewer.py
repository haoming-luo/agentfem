# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility launcher for the optional installed local viewer."""

from agentfem.viewer import build_viewer, main, scalar_choices

__all__ = ["build_viewer", "scalar_choices"]

if __name__ == "__main__":
    main()
