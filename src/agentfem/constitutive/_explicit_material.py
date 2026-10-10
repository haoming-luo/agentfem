# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private response-only contract. No implicit tangent or stability is implied.

The input reuses MaterialPointArrayBatchInput. Material history can include
accepted stress/work when its algorithm needs them. Output work is an increment
per REFERENCE volume, never specific work or recoverable stored energy.
"""

from dataclasses import dataclass, replace

import numpy as np

from .material_array_batch import MaterialPointArrayBatchInput, _array
from .user_material import MaterialStateSchema


@dataclass(frozen=True)
class ExplicitMaterialBatchOutput:
    cauchy_stress: np.ndarray
    state_new: np.ndarray
    state_schema: MaterialStateSchema
    internal_work_density_increment: np.ndarray
    stored_energy_density: np.ndarray | None = None
    dissipation_density_increment: np.ndarray | None = None
    requested_time_scale: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("Expected MaterialStateSchema.")
        values = np.asarray(self.cauchy_stress)
        if values.ndim != 3:
            raise ValueError("Stress requires shape (points, 3, 3).")
        count = len(values)
        stress = _array(values, (count, 3, 3), "cauchy_stress")
        scale = np.maximum(np.max(np.abs(stress), axis=(1, 2)), np.finfo(float).tiny)
        if np.any(
            np.max(np.abs(stress - stress.transpose(0, 2, 1)), axis=(1, 2))
            > 1.0e-12 * scale
        ):
            raise ValueError("Cauchy stress must be symmetric.")
        object.__setattr__(self, "cauchy_stress", stress)
        object.__setattr__(
            self,
            "state_new",
            _array(self.state_new, (count, self.state_schema.size), "state_new"),
        )
        object.__setattr__(
            self,
            "internal_work_density_increment",
            _array(
                self.internal_work_density_increment,
                (count,),
                "internal_work_density_increment",
            ),
        )
        for name in ("stored_energy_density", "dissipation_density_increment"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _array(value, (count,), name))
        scales = (
            np.ones(count)
            if self.requested_time_scale is None
            else self.requested_time_scale
        )
        scales = _array(scales, (count,), "requested_time_scale")
        if np.any(scales <= 0) or np.any(scales > 1):
            raise ValueError("Requested time scale must lie in (0, 1].")
        object.__setattr__(self, "requested_time_scale", scales)

    @property
    def point_count(self):
        return len(self.cauchy_stress)

    def energy_contract(self):
        return {
            "volume_configuration": "reference",
            "internal_work": "increment",
            "stored_energy_available": self.stored_energy_density is not None,
            "dissipation_available": self.dissipation_density_increment is not None,
            "thermodynamic_validation": "not_established_by_response",
            "tangent_available": False,
            "explicit_stability": "requires_independent_procedure_contract",
        }


def validated_explicit_material_update(material, request):
    """Pure local provider evaluation: never accept state or fall back silently."""
    if not isinstance(request, MaterialPointArrayBatchInput):
        raise TypeError("Expected MaterialPointArrayBatchInput.")
    if request.state_schema != material.state_schema:
        raise ValueError("Explicit request state schema differs from material.")
    method = getattr(material, "update_explicit_array_batch", None)
    if not callable(method):
        raise TypeError("Material must implement update_explicit_array_batch.")
    response = method(request)
    if not isinstance(response, ExplicitMaterialBatchOutput):
        raise TypeError(
            "Expected ExplicitMaterialBatchOutput; no implicit response coercion."
        )
    response = replace(response)
    if (
        response.state_schema != request.state_schema
        or response.point_count != request.point_count
    ):
        raise ValueError("Explicit response changed point count or state identity.")
    return response
