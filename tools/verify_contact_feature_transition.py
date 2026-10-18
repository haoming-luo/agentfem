# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Research oracle: an active sharp-feature boundary needs a line tangent.

For the unit triangle 0<=x, 0<=y, x+y<=1, let the negative contact gap be
g=max(x-c-a, c-x-a)=|x-c|-a.  The kink at x=c is *inside* the active strip,
where g=-a rather than zero.  A fixed-branch bulk Hessian therefore cannot be
the derivative of the integrated force.  This is an analytical illustration,
not a released sharp-tool contact law or a general contact benchmark.
"""

from __future__ import annotations

import json


def active_feature_oracle(*, width: float, center: float, penalty: float = 1.0):
    a = float(width)
    c = float(center)
    k = float(penalty)
    if not 0 < a < c < 1 - a or not k > 0:
        raise ValueError("Require 0 < width < center < 1-width and penalty > 0.")
    # The y-length at x is 1-x.  Odd terms about x=c cancel in the potential.
    energy = k * a**3 * (1 - c) / 3
    force = -k * a**3 / 3
    true_tangent = 0.0
    # Differentiate the two affine feature branches while incorrectly holding
    # their integration domains fixed: this omits the x=c interface line.
    fixed_branch_tangent = 2 * k * a * (1 - c)
    # Distributional derivative of max(g1,g2) contributes k*g*delta(g1-g2)
    # times the squared normal jump; g=-a at the switching interface.
    feature_interface_tangent = -fixed_branch_tangent
    return {
        "scope": "research_only_active_two_plane_feature_on_unit_triangle",
        "potential": energy,
        "generalized_force": force,
        "true_tangent": true_tangent,
        "fixed_branch_bulk_tangent": fixed_branch_tangent,
        "missing_feature_interface_tangent": feature_interface_tangent,
        "balanced_tangent": fixed_branch_tangent + feature_interface_tangent,
    }


def main() -> None:
    print(json.dumps(active_feature_oracle(width=0.2, center=0.5), indent=2))


if __name__ == "__main__":
    main()
