# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Batched material-frame kinematics for explicitly chosen constitutive paths.

``polar_strain_increment`` integrates the rate along a linear right-stretch
path between two total deformation gradients. It returns tensor strain (not
engineering shear) and material-to-spatial endpoint frames. ``exact`` refers
to that declared path integral, not an unknown deformation history or a vendor
VUMAT increment. The optional midpoint rule has a distinct numerical identity.

These utilities own neither material history nor trial/commit/rollback. A
provider must include its integration rule and reference axes in its identity,
return spatial Cauchy stress to the ordinary material protocol, and account for
its own work/energy convention. No tangent or stability bound is implied.

The private module remains the implementation/compatibility location; public
and legacy imports deliberately execute the same checked arithmetic.
"""
from ._corotational import (
    CorotationalIncrement,
    polar_strain_increment,
    push_corotational_stress,
)

__all__ = ["CorotationalIncrement", "polar_strain_increment", "push_corotational_stress"]
