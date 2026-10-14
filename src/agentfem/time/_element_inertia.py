# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private fixed/variable policy building block from supplied element bounds.

The operator owns and must justify each element spectral upper bound. This
function only distributes the required numerical inertia; it cannot establish
those bounds from an arbitrary material response or certify contact/damping.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElementInertiaPlan:
    element_factors: np.ndarray
    nodal_factors: np.ndarray
    physical_nodal_mass: np.ndarray
    numerical_nodal_mass: np.ndarray
    target_body_increment: float
    supplied_body_spectral_bound: float

    def __post_init__(self):
        for name in (
            "element_factors",
            "nodal_factors",
            "physical_nodal_mass",
            "numerical_nodal_mass",
        ):
            value = np.array(getattr(self, name), copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    def summary(self):
        return {
            "kind": "element_bound_numerical_inertia_plan",
            "elements_scaled": int(np.count_nonzero(self.element_factors > 1)),
            "maximum_element_factor": float(self.element_factors.max(initial=1)),
            "physical_mass": float(self.physical_nodal_mass.sum()),
            "numerical_mass": float(self.numerical_nodal_mass.sum()),
            "added_mass_ratio": float(
                self.numerical_nodal_mass.sum() / self.physical_nodal_mass.sum() - 1
            ),
            "target_body_increment": self.target_body_increment,
            "supplied_body_spectral_bound": self.supplied_body_spectral_bound,
            "whole_system_stability_certificate": False,
            "scope": "supplied positive-semidefinite element bounds only",
        }


def plan_element_inertia(
    cell_nodes,
    element_nodal_mass,
    spectral_upper_bounds,
    *,
    target_body_increment,
    safety_factor=0.8,
    minimum_nodal_factors=None,
):
    """Scale only elements whose supplied physical-mass bound exceeds budget.

    For each PSD K_e <= lambda_e M_e, choosing f_e >= lambda_e/lambda_target
    gives sum(K_e) <= lambda_target * sum(f_e M_e). Increasing previously
    accepted nodal masses preserves the inequality. Contact and damping are
    not included; their contributions must be admitted separately.
    """
    nodes = np.asarray(cell_nodes)
    raw = [np.asarray(v) for v in (element_nodal_mass, spectral_upper_bounds)]
    if (
        nodes.ndim != 2
        or not nodes.size
        or nodes.dtype.kind not in "iu"
        or np.any(nodes < 0)
    ):
        raise ValueError(
            "Element inertia requires nonnegative integer cell/node indices."
        )
    if any(np.iscomplexobj(v) for v in raw):
        raise ValueError("Element inertia data must be real.")
    mass, bounds = [np.asarray(v, dtype=float) for v in raw]
    if (
        mass.shape != nodes.shape
        or bounds.shape != (len(nodes),)
        or not np.isfinite(mass).all()
        or np.any(mass <= 0)
        or not np.isfinite(bounds).all()
        or np.any(bounds < 0)
    ):
        raise ValueError(
            "Element mass and positive spectral bounds must match cell layout."
        )
    if (
        isinstance(target_body_increment, bool)
        or isinstance(safety_factor, bool)
        or np.iscomplexobj(target_body_increment)
        or np.iscomplexobj(safety_factor)
        or np.ndim(target_body_increment) != 0
        or np.ndim(safety_factor) != 0
        or not np.isfinite(target_body_increment)
        or target_body_increment <= 0
        or not np.isfinite(safety_factor)
        or not 0 < safety_factor <= 1
    ):
        raise ValueError("Invalid body target increment or safety factor.")
    # IDs are compact serial nodal identities. Reject gaps before allocating a
    # potentially huge bincount result for a malformed external index table.
    unique = np.unique(nodes)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("Element inertia requires compact serial nodal identities.")
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        budget = np.square(np.float64(2 * float(safety_factor)) / target_body_increment)
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("Element inertia spectral budget overflow/underflow.")
    factors = np.maximum(1.0, bounds / budget)
    physical = np.bincount(nodes.ravel(), weights=mass.ravel(), minlength=len(unique))
    numerical = np.bincount(
        nodes.ravel(), weights=(mass * factors[:, None]).ravel(), minlength=len(unique)
    )
    if minimum_nodal_factors is not None:
        previous = np.asarray(minimum_nodal_factors)
        if (
            np.iscomplexobj(previous)
            or previous.shape != physical.shape
            or not np.isfinite(previous).all()
            or np.any(previous < 1)
        ):
            raise ValueError(
                "Previous nodal scaling must be finite, at least one and layout-compatible."
            )
        numerical = np.maximum(numerical, physical * previous)
    if not np.isfinite(physical).all() or not np.isfinite(numerical).all():
        raise ValueError("Element inertia mass accumulation overflowed.")
    return ElementInertiaPlan(
        factors,
        numerical / physical,
        physical,
        numerical,
        float(target_body_increment),
        float(np.max(bounds / factors)),
    )
