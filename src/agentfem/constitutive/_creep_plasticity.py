# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Experimental simultaneous small-strain creep/plasticity local integration.

The bounded serial 3D consumer reuses the implicit creep Step. Plastic and creep
history are returned together; local trials always use the same accepted state.
"""

from dataclasses import dataclass, field

import numpy as np

from .creep import IsotropicCreepMaterial, ImplicitCreepState, PowerLawCreep, _flow_rate_slope
from .plasticity import (
    ChabocheCombinedHardening, ChabocheState, J2LinearIsotropicHardening,
    J2PlasticState, PlasticEnergyIncrement, _symmetric_tensor,
)
from ..materials.properties import TemperatureDependentThermoElasticProperties


# Orthonormal Mandel basis: Euclidean products equal tensor double contraction.
_BASIS = np.zeros((6, 3, 3))
for _a, (_i, _j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))):
    _BASIS[_a, _i, _j] = _BASIS[_a, _j, _i] = 1. if _i == _j else 1. / np.sqrt(2.)
_IDENTITY = np.array([1., 1., 1., 0., 0., 0.])
_DEV = np.eye(6) - np.outer(_IDENTITY, _IDENTITY) / 3.


def _vector(tensor):
    return np.einsum("aij,ij->a", _BASIS, tensor)


def _tensor(vector):
    return np.einsum("a,aij->ij", vector, _BASIS)


@dataclass(frozen=True)
class CreepPlasticityState:
    creep: ImplicitCreepState = field(default_factory=ImplicitCreepState)
    plastic: J2PlasticState | ChabocheState = field(default_factory=J2PlasticState)


@dataclass(frozen=True)
class CreepPlasticityUpdate:
    stress: np.ndarray
    state: CreepPlasticityState
    algorithmic_tangent: np.ndarray
    creep_increment: float
    plastic_increment: float
    creep_work: float
    plastic_energy: PlasticEnergyIncrement
    local_iterations: int
    residual_norm: float


@dataclass(frozen=True)
class CreepPlasticityMaterial:
    """Compose one creep branch with existing isotropic J2 or Chaboche.

    Solve sigma = P(eps - CE_old - dt flow(sigma), plastic_old), where P is
    the existing complete discrete plastic update. This is a simultaneous
    backward-Euler composition, not a one-pass operator split. If C_p is its
    tangent and L is the creep flow derivative, the consistent tangent is
    (I + dt C_p L)^(-1) C_p. No symmetry or coaxial flow is assumed.

    The first boundary uses constant elastic moduli and optional secant thermal
    strain / Arrhenius power-law rate through the creep material. Temperature-
    dependent plastic parameters, static recovery and finite strain are absent.
    """

    creep: IsotropicCreepMaterial
    plastic: J2LinearIsotropicHardening | ChabocheCombinedHardening
    name: str = "simultaneous creep and plasticity"
    stateful_constitutive = True

    @property
    def requires_temperature(self):
        return self.creep.requires_temperature

    @property
    def young(self):
        return self.creep.young

    @property
    def poisson(self):
        return self.creep.poisson

    @property
    def density(self):
        return self.creep.density

    def thermal_strain(self, temperature):
        return self.creep.thermal_strain(temperature)

    def stress_from_state(self, strain, state, *, temperature=None):
        elastic = (strain - state.creep.creep_strain - state.plastic.plastic_strain
                   - self.thermal_strain(temperature))
        return np.einsum("ijkl,kl->ij", self.plastic.elastic_tangent(), elastic)

    def __post_init__(self):
        if not isinstance(self.creep, IsotropicCreepMaterial):
            raise TypeError("creep must be an IsotropicCreepMaterial.")
        if not isinstance(self.plastic, (J2LinearIsotropicHardening, ChabocheCombinedHardening)):
            raise TypeError("plastic must be an isotropic J2 or Chaboche material.")
        if isinstance(self.creep.elastic, TemperatureDependentThermoElasticProperties):
            raise ValueError("Coupled integration currently requires constant elastic moduli.")
        if isinstance(self.creep.creep, PowerLawCreep) and self.creep.creep.stress_exponent < 1.:
            raise ValueError("Coupled flow requires stress exponent >= 1.")
        if not np.allclose([self.creep.young, self.creep.poisson],
                           [self.plastic.young, self.plastic.poisson], rtol=1.e-13, atol=0.):
            raise ValueError("Creep and plastic branches must share elastic moduli.")

    def initial_state(self):
        plastic = (self.plastic.initial_state() if isinstance(self.plastic, ChabocheCombinedHardening)
                   else J2PlasticState())
        return CreepPlasticityState(plastic=plastic)

    def update(self, total_strain, *, time_start, time_end, state=None,
               temperature=None, tolerance=1.e-10, maximum_iterations=60):
        strain = _symmetric_tensor(total_strain, label="total_strain")
        start, end = float(time_start), float(time_end)
        if not np.isfinite([start, end]).all() or not 0. <= start < end:
            raise ValueError("Require finite 0 <= time_start < time_end.")
        if not np.isfinite(tolerance) or tolerance <= 0. or maximum_iterations < 1:
            raise ValueError("Local tolerance and iteration limit must be positive.")
        old = self.initial_state() if state is None else state
        if not isinstance(old, CreepPlasticityState):
            raise TypeError("Coupled update requires CreepPlasticityState.")
        if isinstance(self.plastic, ChabocheCombinedHardening):
            if not isinstance(old.plastic, ChabocheState) or len(old.plastic.backstresses) != self.plastic.backstress_count:
                raise ValueError("Plastic state does not match Chaboche branch.")
        elif not isinstance(old.plastic, J2PlasticState):
            raise ValueError("Plastic state does not match J2 branch.")
        dt = end - start
        # Also performs common temperature/rate validation. This predictor does
        # not commit either branch and uses the old plastic strain only.
        predictor = self.creep.update(
            strain - old.plastic.plastic_strain, time_start=start, time_end=end,
            state=old.creep, temperature=temperature,
        )
        law = (self.creep.creep if self.creep.temperature_dependence is None
               else self.creep.temperature_dependence.at_temperature(temperature))
        mechanical = strain - self.creep.thermal_strain(temperature)
        stress = _vector(predictor.stress)

        def response(candidate):
            s = _DEV @ candidate
            q = np.sqrt(1.5 * np.dot(s, s))
            rate, slope = _flow_rate_slope(law, q, end)
            if q > 0.:
                direction = 1.5 * s / q
                flow = rate * direction
                derivative = 1.5 * rate / q * _DEV + (slope - rate / q) * np.outer(direction, direction)
            else:
                flow = np.zeros(6)
                derivative = 1.5 * slope * _DEV
            if not np.isfinite(flow).all() or not np.isfinite(derivative).all():
                raise FloatingPointError("Nonfinite creep trial flow.")
            creep_delta = dt * _tensor(flow)
            plastic = self.plastic.update(mechanical - old.creep.creep_strain - creep_delta, old.plastic)
            cp = np.einsum("aij,ijkl,bkl->ab", _BASIS, plastic.algorithmic_tangent, _BASIS)
            residual = candidate - _vector(plastic.stress)
            jacobian = np.eye(6) + dt * cp @ derivative
            return residual, jacobian, cp, plastic, creep_delta, float(dt * rate)

        scale = max(1., self.plastic.yield_stress)
        for iteration in range(1, int(maximum_iterations) + 1):
            residual, jacobian, cp, plastic, delta, increment = response(stress)
            norm = float(np.linalg.norm(residual))
            if norm <= tolerance * scale:
                tangent = np.linalg.solve(jacobian, cp)
                return CreepPlasticityUpdate(
                    stress=_tensor(stress),
                    state=CreepPlasticityState(
                        ImplicitCreepState(old.creep.creep_strain + delta,
                                           old.creep.equivalent_creep_strain + increment), plastic.state),
                    algorithmic_tangent=np.einsum("aij,ab,bkl->ijkl", _BASIS, tangent, _BASIS),
                    creep_increment=increment,
                    plastic_increment=plastic.plastic_multiplier_increment,
                    creep_work=float(np.dot(stress, _vector(delta))),
                    plastic_energy=plastic.energy_increment,
                    local_iterations=iteration, residual_norm=norm,
                )
            correction = np.linalg.solve(jacobian, -residual)
            alpha = 1.
            for _ in range(24):
                trial = stress + alpha * correction
                try:
                    trial_residual = response(trial)[0]
                except (FloatingPointError, OverflowError):
                    trial_residual = np.full(6, np.inf)
                if np.linalg.norm(trial_residual) < (1. - 1.e-4 * alpha) * norm:
                    stress = trial
                    break
                alpha *= 0.5
            else:
                raise RuntimeError("Coupled creep/plasticity local line search failed; reduce the increment.")
        raise RuntimeError("Coupled creep/plasticity did not converge; reduce the increment.")

    def as_dict(self):
        return dict(name=self.name, model="simultaneous_creep_plasticity",
                    creep=self.creep.as_dict(), plastic=self.plastic.as_dict(),
                    integration="simultaneous_backward_euler",
                    kinematics="small_strain", maturity="experimental_global_patch",
                    fem_quadrature_driver=True, global_scope="serial_3d_homogeneous")
