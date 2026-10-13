# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Linear initial-stress stability through the ordinary Model.step workflow."""

from dataclasses import dataclass
import numpy as np
import ufl
from .. import operators
from ..constitutive import elasticity
from ..backends._buckling import solve_buckling
from .modal import _collect_modal_bcs_collectively, _require_homogeneous_modal_bcs


@dataclass
class LinearBucklingStep:
    target: object
    stiffness: object
    geometric: object
    constraints: tuple
    reference_name: str
    base_name: str | None = None
    modes: int = 3
    tolerance: float = 1.0e-9
    maximum_iterations: int = 1000
    name: str = "linear_buckling"
    procedure: object = None

    def solve(self):
        u = getattr(self.target, "value", self.target)
        bcs = _collect_modal_bcs_collectively(u, constraints=self.constraints)
        _require_homogeneous_modal_bcs(u, bcs)
        self.pairs, self.info = solve_buckling(
            self.stiffness,
            self.geometric,
            self.target,
            bcs,
            modes=self.modes,
            tolerance=self.tolerance,
            maximum_iterations=self.maximum_iterations,
        )
        return tuple(item[1] for item in self.pairs)

    def summary(self):
        return {
            "kind": "linear_initial_stress_buckling",
            "name": self.name,
            "reference_state": self.reference_name,
            "fixed_base_state": self.base_name,
            "equation": "(K + KG_base + lambda KG_reference) phi = 0",
            "scope": "small_displacement_linear_elastic_conservative_loads",
            "modes": self.modes,
            "solve": getattr(self, "info", None),
        }

    def solve_result(self, *, output=None, strict_output=False):
        from ..results import SimulationResult
        from ..results.lifecycle import complete_result

        self.solve()
        result = SimulationResult(name=self.name, metadata={"problem": self.summary()})
        result.add_quantities(
            {
                "load_factors": [p[0] for p in self.pairs],
                "relative_residuals": [p[2] for p in self.pairs],
            },
            kind="buckling",
        )
        for i, (_, mode, _) in enumerate(self.pairs, 1):
            result.add_field(
                f"Buckling_mode_{i}",
                mode,
                description="Relative buckling shape, not a physical displacement or imperfection amplitude.",
            )
        return complete_result(self, result, output=output, strict_output=strict_output)


def lower_buckling(model, request):
    """Build K and KG from registered regions and supplied equilibrium fields."""
    opts = dict(request.options)
    if any(opts.get(key) is not None for key in ("K", "F", "solver_options")):
        raise ValueError(
            "AFM-BUCKLING-010: custom K/F and linear solver options are not supported by this provider."
        )
    target = request.target
    u = target.value
    study = model.study
    if study.dimension not in (2, 3) or study.assumption not in (
        None,
        "plane_stress",
        "plane_strain",
    ):
        raise ValueError(
            "AFM-BUCKLING-005: only plane stress/strain and 3D solids are supported."
        )
    if model.loads or model.boundary_models or model.eigenstrains:
        raise ValueError(
            "AFM-BUCKLING-006: solve loads/eigenstrains in separate reference/base static models; buckling accepts equilibrium displacement fields and homogeneous perturbation supports only."
        )
    modes = opts.get("modes", 3)
    maximum = opts.get("maximum_iterations", 1000)
    tol = float(opts.get("tolerance", 1.0e-9))
    if (
        any(
            isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1
            for x in (modes, maximum)
        )
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError(
            "AFM-BUCKLING-007: positive integer mode/iteration counts and positive finite tolerance required."
        )
    K = model.stiffness(target)

    def geometric(displacement, name):
        value = getattr(displacement, "value", displacement)
        if (
            value.function_space.mesh is not u.function_space.mesh
            or value.ufl_shape != u.ufl_shape
        ):
            raise ValueError(
                "AFM-BUCKLING-008: reference and base fields must share the target mesh and vector shape."
            )
        # Snapshot: later changes of the static solution must not change this pencil.
        value = value.copy()
        parts = []
        for record in model.materials:
            measure = record.region.measure if record.region is not None else ufl.dx
            stress = elasticity.stress(value, record.item, study=study)
            parts.append(
                operators.geometric_stiffness(
                    target, stress, measure=measure, name=name
                )
            )
        return operators.combine(*parts, name=name)

    reference = opts["reference_displacement"]
    KG = geometric(reference, "KG_reference")
    base = opts.get("base_displacement")
    if base is not None:
        K = operators.combine(K, geometric(base, "KG_base"), name="K_base")
    step = LinearBucklingStep(
        target,
        K,
        KG,
        tuple(
            model.constraints
            if opts.get("constraints") is None
            else opts["constraints"]
        ),
        opts["reference_name"],
        opts.get("base_name"),
        modes=modes,
        tolerance=tol,
        maximum_iterations=maximum,
        name=opts.get("name") or "linear_buckling",
        procedure=request.procedure,
    )
    if not step.reference_name or (base is not None and not step.base_name):
        raise ValueError(
            "AFM-BUCKLING-009: name the reference load/state and any fixed base state."
        )
    return model.add_step(step)
