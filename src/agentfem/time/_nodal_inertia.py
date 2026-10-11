# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private operator-row mass planner; State and Procedure remain external.

For symmetric PSD stiffness K and lagged-velocity damping C, a sufficient
fixed-coefficient central-difference condition is h^2 K + 2 h C < 4 M.
Absolute unscaled assembled row bounds give a diagonal majorant. This planner
does not derive or certify these operator bounds, or admit a softening model.
"""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np


@dataclass(frozen=True)
class NodalInertiaPlan:
    nodal_factors: np.ndarray
    physical_nodal_mass: np.ndarray
    numerical_nodal_mass: np.ndarray
    target_increment: float
    safety_factor: float
    stiffness_names: tuple[str, ...]
    damping_names: tuple[str, ...]

    def __post_init__(self):
        for name in ("nodal_factors", "physical_nodal_mass", "numerical_nodal_mass"):
            value = np.array(getattr(self, name), copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    def summary(self):
        return {
            "kind": "combined_operator_row_inertia_plan",
            "target_increment": self.target_increment,
            "safety_factor": self.safety_factor,
            "stiffness_contributions": list(self.stiffness_names),
            "damping_contributions": list(self.damping_names),
            "added_mass_ratio": float(
                self.numerical_nodal_mass.sum() / self.physical_nodal_mass.sum() - 1
            ),
            "maximum_nodal_factor": float(self.nodal_factors.max()),
            "nodes_scaled": int(np.count_nonzero(self.nodal_factors > 1)),
            "scope": "supplied_fixed_symmetric_PSD_operator_diagonal_majorants",
            "nonlinear_path_certified": False,
        }


def plan_nodal_inertia(
    physical_nodal_mass,
    *,
    stiffness_rows,
    damping_rows=None,
    target_increment,
    safety_factor=0.8,
    minimum_nodal_factors=None,
):
    """Plan isotropic nodal inertia from additive *unscaled* operator row bounds.

    Each mapping entry is (nodes, dimension), with stable distinct names and
    nonnegative diagonal majorants of K or C (absolute row sums suffice).
    Operator-derived Loewner diagonal majorants are also admissible. Do not supply
    previously mass-scaled spectral estimates or take maxima across sources.
    Three components share one nodal mass. Existing accepted mass is a lower
    bound; this function never reduces it or mutates a live State. The owning
    VariableScaledInertia separately enforces its added-mass budget and ledger.
    """
    physical = np.asarray(physical_nodal_mass)
    if (
        physical.dtype.kind not in "fiu"
        or physical.ndim != 1
        or not physical.size
        or not np.isfinite(physical).all()
        or np.any(physical <= 0)
    ):
        raise ValueError("Physical nodal masses must be positive finite real values.")
    for value in (target_increment, safety_factor):
        raw = np.asarray(value)
        if (
            raw.ndim != 0
            or raw.dtype.kind not in "fiu"
            or not np.isfinite(raw)
            or raw <= 0
        ):
            raise ValueError("Invalid increment or safety factor.")
    if safety_factor > 1:
        raise ValueError("Safety factor cannot exceed one.")
    groups = [stiffness_rows, {} if damping_rows is None else damping_rows]
    if any(not isinstance(group, Mapping) for group in groups) or not stiffness_rows:
        raise ValueError("Named stiffness row contributions are required.")
    names = [name for group in groups for name in group]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("Operator row contribution names must be nonempty and unique.")
    shape = None
    totals = []
    for group in groups:
        total = None
        for value in group.values():
            row = np.asarray(value)
            if (
                row.dtype.kind not in "fiu"
                or row.ndim != 2
                or row.shape[0] != physical.size
                or row.shape[1] not in (2, 3)
                or not np.isfinite(row).all()
                or np.any(row < 0)
            ):
                raise ValueError(
                    "Operator rows must be finite nonnegative (nodes,dimension)."
                )
            if shape is not None and row.shape != shape:
                raise ValueError("Operator row layouts differ.")
            shape = row.shape
            total = row.astype(float).copy() if total is None else total + row
        totals.append(np.zeros(shape) if total is None else total)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h = float(target_increment) / float(safety_factor)
        needed = np.max((h * h * totals[0] + 2 * h * totals[1]) / 4, axis=1)
        numerical = np.maximum(physical, needed)
    if minimum_nodal_factors is not None:
        previous = np.asarray(minimum_nodal_factors)
        if (
            previous.dtype.kind not in "fiu"
            or previous.shape != physical.shape
            or not np.isfinite(previous).all()
            or np.any(previous < 1)
        ):
            raise ValueError(
                "Previous nodal factors must be finite, compatible and at least one."
            )
        numerical = np.maximum(numerical, physical * previous)
    factors = numerical / physical
    if (
        not np.isfinite(numerical).all()
        or not np.isfinite(factors).all()
        or not np.isfinite(h)
        or h == 0
    ):
        raise ValueError("Combined inertia planning overflowed/underflowed.")
    return NodalInertiaPlan(
        factors,
        physical,
        numerical,
        float(target_increment),
        float(safety_factor),
        tuple(groups[0]),
        tuple(groups[1]),
    )
