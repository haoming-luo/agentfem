# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit numerical mass policies, independent of material density.

Operators/providers supply reviewed unscaled diagonal majorants. The policy
combines them; the existing inertia State owns accepted factors, rollback and
restart. A bound provider is trusted executable code, not an AI runtime.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from ..provenance import content_fingerprint
from ._nodal_inertia import plan_nodal_inertia
from ._scaled_inertia import FixedScaledInertia
from ._variable_inertia import VariableScaledInertia


def _readonly(value):
    if np.ma.isMaskedArray(value):
        raise ValueError('Masked mass-scaling arrays are not supported')
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu':
        raise ValueError('Mass-scaling arrays must be real numeric values')
    owner = array
    while isinstance(owner, np.ndarray) and owner.base is not None:
        owner = owner.base
    if isinstance(owner, bytes) and array.flags.c_contiguous:
        # Reuse immutable reference geometry/mass between provider evaluations.
        return array
    return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)


def _array_identity(value):
    return {'dtype': value.dtype.str, 'shape': list(value.shape),
            'sha256': sha256(value.tobytes()).hexdigest()}


def _scalar(value, name, *, minimum=0., inclusive=False):
    raw = np.asarray(value)
    if (raw.ndim != 0 or raw.dtype.kind not in 'fiu' or not np.isfinite(raw)
            or (raw < minimum if inclusive else raw <= minimum)):
        raise ValueError(f'{name} must be a finite real scalar '
                         f'{">=" if inclusive else ">"} {minimum}')
    return float(raw)


@dataclass(frozen=True)
class MassScalingRequest:
    """Read-only accepted data in this serial displacement-node ordering.

Reference coordinates/cell nodes are numerical layout, not external mesh IDs.
Required contribution names are an exact inventory, including stabilization
and damping; a provider must explicitly return even a known-zero contribution.
No trial state, material commit operation or mutable FEM object is exposed.
"""
    physical_nodal_mass: np.ndarray
    reference_coordinates: np.ndarray
    cell_nodes: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    current_factors: np.ndarray
    increment: int
    time: float
    time_increment: float
    stiffness_names: tuple[str, ...]
    damping_names: tuple[str, ...]

    def __post_init__(self):
        for name in ('physical_nodal_mass', 'reference_coordinates', 'cell_nodes',
                     'displacement', 'velocity', 'current_factors'):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True)
class MassScalingBounds:
    """Named, unscaled (nodes, components) diagonal majorants of K and C.

The planner assumes symmetric PSD stiffness and lagged-velocity damping.
Provider identity/domain must justify that assumption. Softening/negative
curvature and arbitrary nonlinear path stability are not certified here.
"""
    stiffness: Mapping
    damping: Mapping = field(default_factory=dict)

    def __post_init__(self):
        for name in ('stiffness', 'damping'):
            group = getattr(self, name)
            if not isinstance(group, Mapping):
                raise TypeError('Mass scaling bounds require named mappings')
            object.__setattr__(self, name, MappingProxyType(
                {key: _readonly(value) for key, value in group.items()}))


@dataclass(frozen=True)
class FixedMassScaling:
    """Explicit uniform/nodal factors; never a stable-increment estimator."""
    factor: object
    maximum_added_mass_ratio: float

    def __post_init__(self):
        from ._scaled_inertia import _checked_nodal_factors
        raw = _readonly(self.factor)
        _checked_nodal_factors(raw, raw.size)
        object.__setattr__(self, 'factor', raw)
        object.__setattr__(self, 'maximum_added_mass_ratio', _scalar(
            self.maximum_added_mass_ratio, 'maximum_added_mass_ratio', inclusive=True))

    def summary(self):
        return {'kind': 'fixed_mass_scaling',
                'factor_identity': content_fingerprint(self.factor.tolist()),
                'maximum_added_mass_ratio': self.maximum_added_mass_ratio,
                'material_density_changed': False,
                'automatic_stability_certificate': False}


@dataclass(frozen=True)
class InitialMassScaling:
    """Generate fixed nodal mass from an explicit reference-stiffness model.

    This initial, stress-free estimate is NOT a damaged-path tangent bound.
    ``reference_modulus`` is a nonnegative upper modulus of the declared
    reference dP/dF model (stress units), not density or a frequency. The Step
    supplies mesh, physical mass, hourglass, bonds, contact and initial damping.
    No nodal file, user callback, network or AI participates in planning.
    """
    target_increment: float
    reference_modulus: float
    maximum_added_mass_ratio: float
    safety_factor: float = .8

    def __post_init__(self):
        for name in ('target_increment', 'reference_modulus'):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        object.__setattr__(self, 'maximum_added_mass_ratio', _scalar(
            self.maximum_added_mass_ratio, 'maximum_added_mass_ratio', inclusive=True))
        safety = _scalar(self.safety_factor, 'safety_factor')
        if safety >= 1:
            raise ValueError('safety_factor must be strictly below one')
        object.__setattr__(self, 'safety_factor', safety)

    def summary(self):
        return {'kind': 'initial_operator_mass_scaling',
                'target_increment': self.target_increment,
                'reference_modulus': self.reference_modulus,
                'maximum_added_mass_ratio': self.maximum_added_mass_ratio,
                'safety_factor': self.safety_factor,
                'scope': 'reference_stiffness_surrogate_initial_geometry_zero_velocity',
                'material_density_changed': False,
                'automatic_stability_certificate': False,
                'evolving_contact_and_damage_certified': False}


@dataclass(frozen=True)
class TargetIncrementMassScaling:
    """Calculate nodal factors from a declared bound provider, not an NPZ.

``update_every=None`` plans once at construction. A positive integer requests
updates before macro increments 1, 1+n, 1+2n, ...; dyadic retries do not shift
the cadence. The provider must be deterministic/stateless and cover evolution
between evaluations. The policy never decreases accepted mass or silently
clips a proposal to its budget. This is not an automatic time-step selector.
"""
    target_increment: float
    bounds: object = field(repr=False, compare=False)
    identity: Mapping = field(repr=False, compare=False)
    maximum_added_mass_ratio: float
    safety_factor: float = .8
    update_every: int | None = None
    _identity_json: str = field(init=False, repr=False)

    def __post_init__(self):
        if not callable(self.bounds):
            raise TypeError('bounds must be a deterministic callable')
        if not isinstance(self.identity, Mapping) or not self.identity:
            raise ValueError('A nonempty bound-provider identity is required')
        encoded = json.dumps(dict(self.identity), sort_keys=True, allow_nan=False)
        from ..constitutive.explicit_user_material import _FrozenIdentity
        object.__setattr__(self, '_identity_json', encoded)
        object.__setattr__(self, 'identity', _FrozenIdentity(encoded))
        object.__setattr__(self, 'target_increment', _scalar(self.target_increment, 'target_increment'))
        safety = _scalar(self.safety_factor, 'safety_factor')
        if safety >= 1:
            raise ValueError('safety_factor must be strictly below one')
        object.__setattr__(self, 'safety_factor', safety)
        object.__setattr__(self, 'maximum_added_mass_ratio', _scalar(
            self.maximum_added_mass_ratio, 'maximum_added_mass_ratio', inclusive=True))
        if self.update_every is not None and (
                isinstance(self.update_every, (bool, np.bool_))
                or not isinstance(self.update_every, (int, np.integer))
                or self.update_every < 1):
            raise ValueError('update_every must be a positive integer or None')
        if self.update_every is not None:
            object.__setattr__(self, 'update_every', int(self.update_every))

    def summary(self):
        return {'kind': 'target_increment_mass_scaling',
                'target_increment': self.target_increment,
                'bound_provider': json.loads(self._identity_json),
                'maximum_added_mass_ratio': self.maximum_added_mass_ratio,
                'safety_factor': self.safety_factor, 'update_every': self.update_every,
                'bound_scope': 'provider_supplied_symmetric_PSD_diagonal_majorants',
                'material_density_changed': False,
                'automatic_stability_certificate': False}


def _lower_mass_scaling(policy, internal, history, *, dt, stiffness_names, damping_names):
    """Provider lowering; no new State or Procedure implementation."""
    physical = internal.mass_diagonal
    if isinstance(policy, FixedMassScaling):
        mass = FixedScaledInertia(physical, factor=policy.factor)
        if mass.summary()['added_mass_ratio'] > policy.maximum_added_mass_ratio:
            raise ValueError('Mass scaling exceeds its added-mass budget')
        return mass
    if not isinstance(policy, TargetIncrementMassScaling):
        raise TypeError('mass_scaling requires FixedMassScaling or TargetIncrementMassScaling')
    if dt > policy.target_increment:
        raise ValueError('Step dt must not exceed the mass-scaling target increment')
    coordinates = _readonly(history.u.value.function_space.tabulate_dof_coordinates())
    cell_nodes = _readonly(internal.cell_nodes)
    physical_nodal = _readonly(physical[::3])
    context_identity = content_fingerprint({
        'coordinates': _array_identity(coordinates), 'cell_nodes': _array_identity(cell_nodes),
        'stiffness_names': stiffness_names, 'damping_names': damping_names,
    })

    def propose(**data):
        request = MassScalingRequest(
            physical_nodal, coordinates, cell_nodes,
            data['displacement'].reshape(-1, 3), data['velocity'].reshape(-1, 3),
            data['current_factors'], data['increment'], data['time'], data['dt'],
            tuple(stiffness_names), tuple(damping_names))
        bounds = policy.bounds(request)
        if not isinstance(bounds, MassScalingBounds):
            raise TypeError('Bound provider must return MassScalingBounds')
        if (set(bounds.stiffness) != set(stiffness_names)
                or set(bounds.damping) != set(damping_names)):
            raise ValueError('Mass-scaling operator coverage differs: expected '
                             f'stiffness={stiffness_names}, damping={damping_names}')
        if any(row.shape != (len(physical_nodal), 3)
               for group in (bounds.stiffness, bounds.damping) for row in group.values()):
            raise ValueError('Mass-scaling bounds must match all three displacement components')
        return plan_nodal_inertia(
            physical_nodal, stiffness_rows=bounds.stiffness, damping_rows=bounds.damping,
            target_increment=policy.target_increment, safety_factor=policy.safety_factor,
            minimum_nodal_factors=data['current_factors']).nodal_factors

    if policy.update_every is not None:
        return VariableScaledInertia(
            physical, policy=propose,
            policy_identity={**policy.summary(), 'layout': context_identity},
            maximum_added_mass_ratio=policy.maximum_added_mass_ratio,
            update_every=policy.update_every)
    factors = propose(displacement=history.u.value.x.array, velocity=history.v.value.x.array,
                      current_factors=np.ones(len(physical_nodal)), increment=0, time=0., dt=dt)
    mass = FixedScaledInertia(physical, factor=factors)
    if mass.summary()['added_mass_ratio'] > policy.maximum_added_mass_ratio:
        raise ValueError('Mass scaling exceeds its added-mass budget')
    return mass
