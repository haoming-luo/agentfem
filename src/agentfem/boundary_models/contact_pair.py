# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Scientific contact-pair assets independent of search and Procedures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from agentfem.mesh import BoundaryRegion

from .contact_response import (
    FrictionlessPenaltyContactLaw,
    frictionless_penalty_contact_law,
)
from .contact_friction import (
    PenaltyCoulombFrictionLaw,
    penalty_coulomb_friction_law,
)
from .rigid_body import RigidBody


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class RigidContactPair:
    """Bind one slave boundary, rigid body, and local contact law.

    The pair is a Model asset.  It does not own closest-point search,
    finite-element trace adaptation, penalty stability screening, increment
    acceptance, or result storage; those remain Backend, Operator, Procedure,
    State, and Result responsibilities respectively.
    """

    slave_boundary: BoundaryRegion
    rigid_body: RigidBody
    law: FrictionlessPenaltyContactLaw
    name: str = "rigid_contact_pair"
    friction: PenaltyCoulombFrictionLaw | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.slave_boundary, BoundaryRegion):
            raise TypeError("RigidContactPair requires one BoundaryRegion slave.")
        if not isinstance(self.rigid_body, RigidBody):
            raise TypeError("RigidContactPair requires one RigidBody master.")
        if not isinstance(self.law, FrictionlessPenaltyContactLaw):
            raise TypeError(
                "RigidContactPair requires FrictionlessPenaltyContactLaw."
            )
        if self.friction is not None and not isinstance(
            self.friction,
            PenaltyCoulombFrictionLaw,
        ):
            raise TypeError(
                "RigidContactPair friction must be PenaltyCoulombFrictionLaw."
            )
        if np.asarray(self.law.penalty).ndim != 0:
            raise ValueError(
                "The scientific rigid contact-pair contract currently requires "
                "one scalar penalty; pointwise arrays belong to backend lowering."
            )
        dimension = int(self.slave_boundary.domain.geometry.dim)
        if dimension != self.rigid_body.dimension:
            raise ValueError("Contact slave and rigid body dimensions must match.")
        if not str(self.name).strip():
            raise ValueError("RigidContactPair requires a name.")
        object.__setattr__(self, "name", str(self.name))

    @property
    def dimension(self) -> int:
        return self.rigid_body.dimension

    @property
    def scientific_identity(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._identity_payload()).encode("utf-8")
        ).hexdigest()

    def _identity_payload(self) -> dict[str, object]:
        boundary = self.slave_boundary
        return {
            "name": self.name,
            "slave_boundary": {
                "name": boundary.name,
                "tag": int(boundary.tag),
                "selection": boundary.selection,
            },
            "rigid_body_identity": self.rigid_body.scientific_identity,
            "law": self.law.summary(),
            "friction": None if self.friction is None else self.friction.summary(),
        }

    def summary(self) -> dict[str, object]:
        return {
            "kind": "rigid_contact_pair",
            **self._identity_payload(),
            "dimension": self.dimension,
            "scientific_identity": self.scientific_identity,
            "search": "backend_owned",
            "stability": "procedure_owned",
        }


def rigid_contact_pair(
    slave_boundary: BoundaryRegion,
    rigid_body: RigidBody,
    *,
    penalty,
    friction_coefficient: float | None = None,
    tangential_penalty=None,
    invalid_policy: str = "reject",
    name: str = "rigid_contact_pair",
) -> RigidContactPair:
    """Create one solver-neutral rigid pair with optional friction semantics."""

    if (friction_coefficient is None) != (tangential_penalty is None):
        raise ValueError(
            "Friction requires both friction_coefficient and tangential_penalty."
        )
    friction = (
        None
        if friction_coefficient is None
        else penalty_coulomb_friction_law(
            friction_coefficient,
            tangential_penalty,
            name=f"{name}_friction",
        )
    )
    return RigidContactPair(
        slave_boundary=slave_boundary,
        rigid_body=rigid_body,
        law=frictionless_penalty_contact_law(
            penalty,
            invalid_policy=invalid_policy,
            name=f"{name}_law",
        ),
        name=name,
        friction=friction,
    )


__all__ = ["RigidContactPair", "rigid_contact_pair"]
