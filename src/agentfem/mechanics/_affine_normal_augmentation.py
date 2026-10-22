# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded experimental AL/MPC composition with the ordinary affine path.

Admission requires one fixed-normal plane with disjoint reduced normal rows;
this is not a general mortar or spatial stability certificate.
This owner performs outer iterations only: the affine path still owns trial
snapshots, load factors, material/Operator commit, rollback and checkpoints.
"""

from contextlib import nullcontext
from dataclasses import replace

import numpy as np

from .._affine_equilibrium import solve_serial_affine_equilibrium
from ..boundary_models._augmented_plane_trial_provider import (
    DolfinxAugmentedPlaneTrialProvider, validate_affine_plane_normal_rows,
)
from ._normal_multiplier_acceleration import ProjectedMultiplierAitken


class AffinePlaneNormalAugmentation:
    def __init__(self, provider):
        if not isinstance(provider, DolfinxAugmentedPlaneTrialProvider):
            raise TypeError("Affine augmentation requires the fixed-normal plane provider.")
        self.provider = provider

    def summary(self):
        return {
            "kind": "affine_disjoint_normal_augmentation",
            "schema_version": 1,
            "policy": self.provider.policy.summary(),
            "maturity": "experimental",
            "normal_rows": "disjoint_reduced_support",
        }

    def solve(self, residual, jacobian, function, reduction, transformation, values, **options):
        transaction = options.get("transaction")
        scope = getattr(transaction, "_trial_response_reuse_scope", None)
        with scope() if callable(scope) else nullcontext():
            return self._solve(
                residual, jacobian, function, reduction, transformation, values, **options,
            )

    def _solve(self, residual, jacobian, function, reduction, transformation, values, **options):
        from ..solvers import _reduced_residual_norm

        provider = self.provider
        if options["operators"] != (provider,):
            raise ValueError("Affine augmentation requires exactly its own Operator.")
        validate_affine_plane_normal_rows(
            provider.adapter.trace, provider.surface.normal, reduction,
        )
        state = provider.normal_multiplier_state
        if state.augmentation_count or provider._outer_in_progress:
            raise RuntimeError("Affine normal augmentation requires an accepted starting boundary.")
        provider._outer_in_progress = True
        accelerator = (ProjectedMultiplierAitken(provider.adapter.trace.weights)
                       if provider.policy.acceleration == "aitken" else None)
        infos, relaxations = [], []
        diagnostics = {}
        ready = False
        message = ""
        for _ in range(state.maximum_augmentations + 1):
            info = solve_serial_affine_equilibrium(
                residual, jacobian, function, reduction, transformation, values, **options,
            )
            infos.append(info)
            values = info.reduced_values
            if not info.converged:
                message = info.message or "AL frozen affine equilibrium failed"
                break
            norm = _reduced_residual_norm(residual, transformation, (provider,))
            if not np.isfinite(norm) or norm > info.threshold:
                message = "AL total reduced equilibrium could not be confirmed"
                break
            ready, diagnostics = provider.finish_frozen_equilibrium(
                residual_norm=norm, residual_tolerance=info.threshold,
            )
            infos[-1] = replace(info, residual_norm=norm,
                                residual_assemblies=info.residual_assemblies + 1)
            if ready:
                break
            if state.augmentation_count >= state.maximum_augmentations:
                message = (
                    "normal augmentation limit reached: penetration="
                    f"{diagnostics['maximum_penetration']:.6g}, complementarity="
                    f"{diagnostics['maximum_complementarity']:.6g}"
                )
                break
            relaxations.append(provider.augment(accelerator=accelerator))
        return replace(
            infos[-1], converged=ready, message=message,
            initial_norm=infos[0].initial_norm,
            iterations=sum(item.iterations for item in infos),
            adaptation_iterations=max(item.iterations for item in infos),
            step_lengths=tuple(value for item in infos for value in item.step_lengths),
            linear_iterations=tuple(value for item in infos for value in item.linear_iterations),
            linear_reasons=tuple(value for item in infos for value in item.linear_reasons),
            residual_assemblies=sum(item.residual_assemblies for item in infos),
            reused_assemblies=sum(item.reused_assemblies for item in infos),
            procedure_checks={
                "normal_augmentations": state.augmentation_count,
                "maximum_inner_newton_iterations": max(item.iterations for item in infos),
                "normal_relaxation_factors": relaxations,
                "normal_acceleration_resets": 0 if accelerator is None else accelerator.resets,
                "normal_acceptance": diagnostics,
                "normal_augmentation_scope": "bounded_affine_disjoint_normal_rows",
                "material_response_reuses": sum(
                    item.procedure_checks.get("material_response_reuses", 0) for item in infos
                ),
            },
        )
