# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Response-only user material boundary for explicit integration.

This local API does not imply ordinary global-Step admission, an implicit
tangent, a stable increment, or a thermodynamic energy decomposition.
Provider callbacks are trusted executable code. State belongs to the caller.
"""
from dataclasses import dataclass, field
import json
import math
from typing import Callable, Mapping

from .user_material import MaterialStateSchema
from .material_array_batch import MaterialPointArrayBatchInput
from ._explicit_material import (
    ExplicitMaterialBatchOutput,
    ExplicitMaterialIncrementLimit,
    ExplicitMaterialIncrementExceeded,
    validated_explicit_material_update,
)


@dataclass(frozen=True)
class ExplicitUserMaterial:
    """A named, restart-identifiable batch response without synthetic tangent.

    ``update`` must not commit state. Inputs are immutable accepted arrays;
    outputs describe a trial. Work is incremental per reference volume.
    ``identity`` should include the law revision and all fixed material inputs;
    callables are deliberately excluded from the serializable identity.
    """

    state_schema: MaterialStateSchema
    update: Callable = field(repr=False, compare=False)
    identity: Mapping = field(repr=False, compare=False)
    name: str = "user_explicit_material"
    increment_limit: Callable | None = field(default=None, repr=False, compare=False)
    increment_policy_identity: str | None = None
    density: float | None = None
    _identity_json: str = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("Expected MaterialStateSchema")
        if not callable(self.update):
            raise TypeError("update must be a batch callable")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Material name must be nonempty")
        if not isinstance(self.identity, Mapping) or not self.identity:
            raise ValueError("Material identity must be a nonempty JSON mapping")
        encoded = json.dumps(dict(self.identity), sort_keys=True, allow_nan=False)
        object.__setattr__(self, "_identity_json", encoded)
        # Snapshot the source rather than retaining a caller-mutable mapping.
        object.__setattr__(self, "identity", _FrozenIdentity(encoded))
        supplied = self.increment_limit is not None
        if supplied and not callable(self.increment_limit):
            raise TypeError("increment_limit must be callable")
        if supplied != (self.increment_policy_identity is not None):
            raise ValueError("Increment callback and policy identity must be supplied together")
        if supplied and (not isinstance(self.increment_policy_identity, str)
                         or not self.increment_policy_identity.strip()):
            raise ValueError("Increment policy identity must be nonempty")
        if self.density is not None:
            value = float(self.density)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Material density must be positive finite")
            object.__setattr__(self, "density", value)

    @property
    def explicit_increment_limit(self):
        return self.increment_limit

    @property
    def explicit_increment_policy_identity(self):
        return self.increment_policy_identity

    def update_explicit_array_batch(self, request):
        if not isinstance(request, MaterialPointArrayBatchInput):
            raise TypeError("Expected MaterialPointArrayBatchInput")
        if request.state_schema != self.state_schema:
            raise ValueError("Material state schema mismatch")
        return self.update(request)

    def evaluate(self, request):
        """Validate a local trial; never commit it or silently retry."""
        return validated_explicit_material_update(self, request)

    def summary(self):
        return {"name": self.name, "kind": "response_only_user_material",
                "identity": json.loads(self._identity_json),
                "state_schema": self.state_schema.summary(),
                "density": self.density,
                "increment_policy_identity": self.increment_policy_identity,
                "stress_measure": "cauchy", "kinematics": "deformation_gradient",
                "work_convention": "increment_per_reference_volume",
                "tangent_available": False,
                "global_step_admission": "requires_independent_provider_contract"}


class _FrozenIdentity(Mapping):
    """Detached JSON values on read; nested edits cannot alter restart identity."""
    def __init__(self, encoded):
        self._encoded = encoded

    def __getitem__(self, key):
        return json.loads(self._encoded)[key]

    def __iter__(self):
        return iter(json.loads(self._encoded))

    def __len__(self):
        return len(json.loads(self._encoded))


def explicit_user_material(*, state_schema, update, identity, name="user_explicit_material",
                           increment_limit=None, increment_policy_identity=None, density=None):
    return ExplicitUserMaterial(state_schema, update, identity, name,
                                increment_limit, increment_policy_identity, density)


__all__ = ["ExplicitUserMaterial", "ExplicitMaterialBatchOutput",
           "ExplicitMaterialIncrementLimit", "ExplicitMaterialIncrementExceeded",
           "explicit_user_material"]
