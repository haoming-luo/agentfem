# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Increment-local acceleration of a fixed-trace projected pressure map.

This is a Procedure helper, not material/contact State or a convergence test.
Dynamic Aitken (Crossed-Secant) is applied in the reference-area metric. We
bound relaxation and discard secant history across active-set changes.
See arXiv:2603.12205, sections 2 and 4; no constant-stiffness or global
convergence result from that work is assumed for finite plasticity.
"""

import numpy as np


class ProjectedMultiplierAitken:
    """One short-lived instance per attempted increment; never checkpointed."""

    def __init__(self, reference_weights):
        weights = np.asarray(reference_weights, dtype=float)
        if (weights.ndim != 1 or not weights.size
                or not np.all(np.isfinite(weights)) or np.any(weights <= 0.)):
            raise ValueError("Multiplier acceleration requires positive finite trace weights.")
        weights = weights / np.max(weights)
        self._weights = weights / np.sum(weights)
        self._previous = None
        self._active = None
        self._omega = 1.
        self.resets = 0

    def relaxation(self, multiplier, projected_map):
        old = np.asarray(multiplier, dtype=float)
        plain = np.asarray(projected_map, dtype=float)
        if (old.shape != self._weights.shape or plain.shape != old.shape
                or not np.all(np.isfinite(old)) or not np.all(np.isfinite(plain))
                or np.any(old < 0.) or np.any(plain < 0.)):
            raise ValueError("Multiplier acceleration requires finite nonnegative fixed-trace arrays.")
        residual = plain - old
        active = plain > 0.
        omega = 1.
        if self._previous is not None:
            if np.array_equal(active, self._active):
                # Normalize first: changing coherent pressure units must not
                # overflow/underflow the dot products or change relaxation.
                scale = max(float(np.max(np.abs(residual))),
                            float(np.max(np.abs(self._previous))))
                if scale > 0.:
                    previous = self._previous / scale
                    current = residual / scale
                    change = current - previous
                    denom = float(np.dot(self._weights * change, change))
                    magnitude = max(float(np.dot(self._weights * current, current)),
                                    float(np.dot(self._weights * previous, previous)))
                    if denom > np.finfo(float).eps * magnitude:
                        raw = -self._omega * float(np.dot(self._weights * previous, change)) / denom
                        if np.isfinite(raw) and raw > 0.:
                            omega = float(np.clip(raw, .1, 10.))
                        else:
                            self.resets += 1
                    else:
                        self.resets += 1
                else:
                    self.resets += 1
            else:
                self.resets += 1
        self._previous = residual.copy()
        self._active = active.copy()
        self._omega = omega
        return omega
