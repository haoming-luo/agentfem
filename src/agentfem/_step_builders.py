# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Internal scientific builders used by public Step providers.

The stable user entry point is :meth:`agentfem.models.Model.step`.  Builders
live here so the Model remains a registry/facade rather than accumulating the
construction logic of every material and solution procedure.  Compatibility
methods on Model delegate to these functions throughout the 0.2.x series.
"""

from ._step_builders_dynamics import (
    explicit_dynamics,
    finite_strain_explicit_dynamics,
    implicit_dynamics,
    modal,
)
from ._step_builders_finite_strain import (
    fabric_membrane,
    hyperelastic,
    mixed_hyperelastic,
)
from ._step_builders_frequency import direct_harmonic, harmonic_viscoelastic
from ._step_builders_inelastic import (
    creep,
    finite_strain_j2,
    j2_plasticity,
    viscoelastic,
)
from ._step_builders_thermal import heat_transfer, linear_static

__all__ = (
    "creep",
    "direct_harmonic",
    "explicit_dynamics",
    "fabric_membrane",
    "finite_strain_explicit_dynamics",
    "finite_strain_j2",
    "harmonic_viscoelastic",
    "heat_transfer",
    "hyperelastic",
    "implicit_dynamics",
    "j2_plasticity",
    "linear_static",
    "mixed_hyperelastic",
    "modal",
    "viscoelastic",
)
