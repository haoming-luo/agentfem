# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Outer normal augmentation inside the existing standard-J2 increment.

There is no material commit, load update, checkpoint or linear solver here.
The enclosing Procedure owns those accepted-boundary operations. Each inner
solve starts from the same committed material state, with the last displacement
as its initial Newton guess and a fixed multiplier throughout line search.
"""

from dataclasses import replace
import time

import numpy as np


def solve_augmented_plane_increment(problem, provider, **coordinates):
    started = time.perf_counter()
    state = provider.normal_multiplier_state
    if state.augmentation_count or provider._outer_in_progress:
        raise RuntimeError("AL increment did not start from an accepted boundary.")
    provider._outer_in_progress = True
    infos = []
    rejection = None
    ready = False
    confirmation_seconds = 0.0
    for _ in range(state.maximum_augmentations + 1):
        info = problem._solve_frozen_increment(**coordinates)
        infos.append(info)
        if not info.converged:
            rejection = info.rejection_reason
            break
        # Reassemble the *total* current residual. No sampled contact-only
        # criterion can license multiplier updates or accepted increments.
        confirmation_started = time.perf_counter()
        rhs, norm = problem._correction_rhs()
        rhs.destroy()
        confirmation_seconds += time.perf_counter() - confirmation_started
        tolerance = (problem.solver_options.absolute_tolerance
                     + problem.solver_options.relative_tolerance * info.initial_residual_norm)
        if not np.isfinite(norm) or norm < 0 or norm > tolerance:
            rejection = "AL frozen-multiplier total equilibrium could not be confirmed"
            break
        ready, diagnostics = provider.finish_frozen_equilibrium(
            residual_norm=norm, residual_tolerance=tolerance,
        )
        infos[-1] = replace(info, residual_norm=norm)
        if ready:
            break
        if state.augmentation_count >= state.maximum_augmentations:
            rejection = (
                "normal augmentation limit reached: penetration="
                f"{diagnostics['maximum_penetration']:.6g}, complementarity="
                f"{diagnostics['maximum_complementarity']:.6g}"
            )
            break
        provider.augment()
    last = infos[-1]
    totals = {
        name: sum(getattr(info, name) for info in infos)
        for name in (
            "iterations", "material_update_seconds", "residual_assembly_seconds",
            "tangent_assembly_seconds", "linear_solve_seconds", "line_search_seconds",
            "linear_solve_calls", "linear_iterations",
            "line_search_evaluations",
        )
    }
    totals["residual_assembly_seconds"] += confirmation_seconds
    return replace(
        last, **totals, converged=ready, rejection_reason=rejection,
        initial_residual_norm=infos[0].initial_residual_norm,
        total_seconds=time.perf_counter() - started,
        normal_augmentations=state.augmentation_count,
        maximum_inner_newton_iterations=max(info.iterations for info in infos),
        linear_converged_reasons=tuple(
            reason for info in infos for reason in info.linear_converged_reasons
        ),
    )
