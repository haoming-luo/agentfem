# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private fixed numerical inertia, separate from physical material density.

No automatic stable increment, variable mass update or vendor equivalence is
implied. Fixed scaling is an explicit modeling approximation whose sensitivity
must be evaluated. The immutable identity participates in restart admission.
"""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np


def _checked_nodal_factors(factor, count):
    """Validate without allocating a new physical/numerical diagonal."""
    raw = np.asarray(factor)
    if raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise ValueError("Inertia factors must be real numeric values, not booleans.")
    scale = np.asarray(raw, dtype=float)
    if (
        scale.shape not in ((), (count,))
        or not np.isfinite(scale).all()
        or np.any(scale < 1)
    ):
        raise ValueError(
            "Fixed inertia factors must be finite scalar or nodal values >= 1."
        )
    return np.broadcast_to(scale, (count,))


@dataclass(frozen=True, init=False)
class FixedScaledInertia:
    physical_mass_diagonal: np.ndarray
    mass_diagonal: np.ndarray
    inv_mass: np.ndarray
    factors: np.ndarray
    block_size: int
    identity: str

    def __init__(self, physical_mass_diagonal, *, factor, block_size=3):
        if (
            isinstance(factor, (bool, np.bool_))
            or isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size < 1
            or np.iscomplexobj(physical_mass_diagonal)
            or np.iscomplexobj(factor)
        ):
            raise ValueError(
                "Inertia requires real mass/factors and a positive block size."
            )
        mass = np.asarray(physical_mass_diagonal, dtype=float)
        if (
            mass.ndim != 1
            or not len(mass)
            or len(mass) % block_size
            or not np.isfinite(mass).all()
            or np.any(mass <= 0)
        ):
            raise ValueError("Physical mass must be a positive finite nodal diagonal.")
        blocks = mass.reshape(-1, block_size)
        if not np.array_equal(blocks, np.broadcast_to(blocks[:, :1], blocks.shape)):
            raise ValueError("Physical nodal mass must be isotropic across components.")
        factors = _checked_nodal_factors(factor, len(blocks)).copy()
        numerical = mass * np.repeat(factors, block_size)
        if not np.isfinite(numerical).all() or not np.isfinite(1 / numerical).all():
            raise ValueError("Numerical inertia overflow/underflow.")
        digest = sha256(b"agentfem.fixed-scaled-inertia.v1")
        digest.update(str(block_size).encode())
        digest.update(mass.astype("<f8").tobytes())
        digest.update(factors.astype("<f8").tobytes())
        for name, value in (
            ("physical_mass_diagonal", mass),
            ("mass_diagonal", numerical),
            ("inv_mass", 1 / numerical),
            ("factors", factors),
        ):
            value = np.array(value, copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "block_size", block_size)
        object.__setattr__(self, "identity", digest.hexdigest())

    def checkpoint_identity(self):
        return {
            "schema": "agentfem.fixed-scaled-inertia.v1",
            "identity": self.identity,
            "portability": "same_serial_nodal_order",
            "fixed": True,
        }

    @property
    def mass(self):
        """Existing lumped-mass protocol used by reaction/work consumers."""
        return self.mass_diagonal

    def validate_state(self, state):
        from ..fields import unwrap

        function = unwrap(state.u)
        if function.function_space.mesh.comm.size != 1:
            raise NotImplementedError(
                "Fixed scaled inertia is serial pending distributed admission."
            )
        if function.x.array.shape != self.mass_diagonal.shape:
            raise ValueError("Numerical inertia and nodal state shapes differ.")

    def summary(self):
        physical = float(self.physical_mass_diagonal[:: self.block_size].sum())
        numerical = float(self.mass_diagonal[:: self.block_size].sum())
        return {
            **self.checkpoint_identity(),
            "physical_mass": physical,
            "numerical_mass": numerical,
            "added_mass": numerical - physical,
            "added_mass_ratio": numerical / physical - 1,
            "minimum_factor": float(self.factors.min()),
            "maximum_factor": float(self.factors.max()),
            "automatic_stability_certificate": False,
            "material_density_changed": False,
        }
