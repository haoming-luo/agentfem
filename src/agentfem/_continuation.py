# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Sparse spherical continuation, independent of any material or FE library."""

from dataclasses import dataclass
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve


@dataclass(frozen=True)
class ArcLengthOptions:
    """Dimensionless RMS displacement/load metric and adaptive path controls."""

    initial: float = 0.05
    minimum: float = 1.0e-5
    maximum: float = 0.1
    displacement_scale: float = 1.0
    load_scale: float = 1.0
    tolerance: float = 1.0e-8
    maximum_iterations: int = 20
    maximum_cutbacks: int = 10

    def __post_init__(self):
        for value in (
            self.initial,
            self.minimum,
            self.maximum,
            self.displacement_scale,
            self.load_scale,
            self.tolerance,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "AFM-ARC-001: scales, tolerances and arc lengths must be positive and finite."
                )
        if not self.minimum <= self.initial <= self.maximum:
            raise ValueError("AFM-ARC-001: require minimum <= initial <= maximum.")
        for value in (self.maximum_iterations, self.maximum_cutbacks):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    "AFM-ARC-001: iteration/cutback limits must be positive integers."
                )


class ArcLengthPath:
    """Trace f_int(u)-dead-lambda*reference=0 using a bordered Newton system.

    ``evaluate(u)`` must be a pure, stateless callback returning internal force
    and its consistent tangent (dense or scipy sparse). It must not commit
    material history. Inputs/returned accepted states are copied. This serial
    algebraic engine does not discretize physics or silently switch branches.
    The initial equilibrium must be supplied; bifurcation branch switching,
    irreversible material transactions and dynamic snap response are not implied.
    """

    def __init__(
        self,
        evaluate,
        reference,
        *,
        initial=None,
        initial_load=0.0,
        dead_load=None,
        options=None,
    ):
        self.options = options or ArcLengthOptions()
        self.evaluate = evaluate
        self.reference = np.asarray(reference, dtype=float).copy()
        if (
            self.reference.ndim != 1
            or not len(self.reference)
            or not np.all(np.isfinite(self.reference))
            or np.linalg.norm(self.reference) == 0
        ):
            raise ValueError("AFM-ARC-002: nonzero finite reference vector required.")
        self._u = (
            np.zeros_like(self.reference)
            if initial is None
            else np.asarray(initial, dtype=float).copy()
        )
        self.dead = (
            np.zeros_like(self.reference)
            if dead_load is None
            else np.asarray(dead_load, dtype=float).copy()
        )
        if (
            self._u.shape != self.reference.shape
            or self.dead.shape != self.reference.shape
            or not np.all(np.isfinite(self._u))
            or not np.all(np.isfinite(self.dead))
            or not np.isfinite(initial_load)
        ):
            raise ValueError("AFM-ARC-002: invalid initial state or fixed load.")
        self._load = float(initial_load)
        self.weights = np.r_[
            np.full(
                len(self._u), 1 / (len(self._u) * self.options.displacement_scale**2)
            ),
            1 / self.options.load_scale**2,
        ]
        self.force_scale = np.linalg.norm(self.reference)
        self.previous = None
        self.next_length = self.options.initial
        self.attempts = []
        self.history = []
        residual, _ = self._equilibrium(self._u, self._load)
        if np.linalg.norm(residual) / self.force_scale > self.options.tolerance:
            raise ValueError("AFM-ARC-003: initial state is not an equilibrium.")

    @property
    def displacement(self):
        return self._u.copy()

    @property
    def load_factor(self):
        return self._load

    def _equilibrium(self, u, load):
        force, tangent = self.evaluate(u.copy())
        force = np.asarray(force, dtype=float)
        tangent = sparse.csr_matrix(tangent, dtype=float)
        n = len(self._u)
        if force.shape != (n,) or tangent.shape != (n, n):
            raise ValueError(
                "AFM-ARC-004: callback force/tangent dimensions do not match."
            )
        if not np.all(np.isfinite(force)) or not np.all(np.isfinite(tangent.data)):
            raise FloatingPointError("Nonfinite equilibrium evaluation.")
        return force - self.dead - load * self.reference, tangent

    @staticmethod
    def _solve(matrix, rhs):
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            out = spsolve(sparse.csc_matrix(matrix), rhs)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("Nonfinite continuation correction.")
        return out

    def _border(self, tangent, row):
        return sparse.bmat(
            [
                [tangent, sparse.csr_matrix(-self.reference[:, None])],
                [sparse.csr_matrix(row[:-1][None, :]), sparse.csr_matrix([[row[-1]]])],
            ],
            format="csc",
        )

    def advance(self):
        """Accept one equilibrium or leave the last accepted state unchanged."""
        o = self.options
        accepted = np.r_[self._u, self._load]
        _, tangent = self._equilibrium(self._u, self._load)
        if self.previous is None:
            direction = np.r_[self._solve(tangent, self.reference), 1.0]
        else:
            direction = self._solve(
                self._border(tangent, self.weights * self.previous),
                np.r_[np.zeros(len(self._u)), 1.0],
            )
        direction /= np.sqrt(np.dot(self.weights * direction, direction))
        if (
            self.previous is not None
            and np.dot(self.weights * direction, self.previous) < 0
        ):
            direction *= -1
        length = self.next_length
        for cutback in range(o.maximum_cutbacks + 1):
            candidate = accepted + length * direction
            reason = "maximum corrector iterations"
            for iteration in range(o.maximum_iterations + 1):
                try:
                    residual, tangent = self._equilibrium(candidate[:-1], candidate[-1])
                    delta = candidate - accepted
                    arc = np.dot(self.weights * delta, delta) - length**2
                    error = np.linalg.norm(residual) / self.force_scale
                    arc_error = abs(arc) / length**2
                    if error <= o.tolerance and arc_error <= o.tolerance:
                        if np.dot(self.weights * delta, direction) <= 0:
                            reason = "corrector reversed continuation direction"
                            break
                        self._u, self._load = (
                            candidate[:-1].copy(),
                            float(candidate[-1]),
                        )
                        self.previous = delta.copy()
                        self.next_length = min(
                            o.maximum, length * (1.25 if iteration <= 4 else 1.0)
                        )
                        record = dict(
                            load_factor=self._load,
                            displacement=self._u.copy(),
                            arc_length=length,
                            residual=error,
                            arc_error=arc_error,
                            iterations=iteration,
                            cutbacks=cutback,
                        )
                        self.history.append(record)
                        self.attempts.append(
                            dict(accepted=True, length=length, iterations=iteration)
                        )
                        return {**record, "displacement": record["displacement"].copy()}
                    if iteration == o.maximum_iterations:
                        break
                    correction = self._solve(
                        self._border(tangent, 2 * self.weights * delta),
                        -np.r_[residual, arc],
                    )
                    candidate += correction
                except (
                    FloatingPointError,
                    MatrixRankWarning,
                    np.linalg.LinAlgError,
                ) as exc:
                    reason = str(exc)
                    break
            self.attempts.append(dict(accepted=False, length=length, reason=reason))
            length *= 0.5
            if length < o.minimum:
                break
        raise RuntimeError(
            "AFM-ARC-005: continuation failed; accepted state unchanged: " + reason
        )

    def run(self, increments, *, stop=None):
        if (
            isinstance(increments, bool)
            or not isinstance(increments, int)
            or increments < 1
        ):
            raise ValueError("increments must be a positive integer.")
        for _ in range(increments):
            record = self.advance()
            if stop is not None and stop(record):
                break
        return tuple(
            {**r, "displacement": r["displacement"].copy()} for r in self.history
        )
