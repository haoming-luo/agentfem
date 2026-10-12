# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Instance-local trusted execution boundary for response-only Hex8 arithmetic.

Kernels return detached arrays. They do not receive quadrature Fields, PETSc
objects or a commit operation. This is an execution extension, not a new law,
an automatic stability certificate, or isolation from malicious Python code.
"""
from dataclasses import dataclass, field
import json
from typing import Callable, Mapping

import numpy as np

from ..constitutive._explicit_material import ExplicitMaterialBatchOutput
from ..constitutive.material_array_batch import _array


@dataclass(frozen=True)
class ExplicitHexGeometry:
    average_gradient: np.ndarray
    hourglass_modes: np.ndarray
    volume: np.ndarray
    hourglass_coefficient: np.ndarray

    def __post_init__(self):
        n = len(self.volume)
        for name, shape in (("average_gradient", (n, 8, 3)),
                            ("hourglass_modes", (n, 8, 4)),
                            ("volume", (n,)), ("hourglass_coefficient", (n,))):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        if np.any(self.volume <= 0) or np.any(self.hourglass_coefficient < 0):
            raise ValueError("Invalid explicit Hex8 reference geometry")


@dataclass(frozen=True)
class ExplicitHexBodyOutput:
    material_response: ExplicitMaterialBatchOutput
    first_piola_stress: np.ndarray
    internal_force: np.ndarray
    hourglass_energy: np.ndarray

    def __post_init__(self):
        if not isinstance(self.material_response, ExplicitMaterialBatchOutput):
            raise TypeError("Expected ExplicitMaterialBatchOutput")
        n = self.material_response.point_count
        for name, shape in (("first_piola_stress", (n, 3, 3)),
                            ("internal_force", (n, 8, 3)), ("hourglass_energy", (n,))):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        if np.any(self.hourglass_energy < 0):
            raise ValueError("Hourglass stabilization energy must be nonnegative")


@dataclass(frozen=True)
class ExplicitHexBodyKernel:
    """Trusted callback: ``evaluate(material, request, geometry, displacement)``.

    The output includes material response, work-conjugate Piola stress, total
    cell internal force (material plus hourglass), and hourglass energy. The
    callback must preserve the material's declared law and exact fallback rules.
    Identify its pinned implementation in ``identity``. Core checks transport
    and acceptance; scientific equivalence needs separate reference tests.
    """
    evaluate: Callable
    identity: Mapping
    _identity_json: str = field(init=False, repr=False)

    def __post_init__(self):
        if not callable(self.evaluate) or not isinstance(self.identity, Mapping):
            raise TypeError("Kernel requires a callback and identity mapping")
        encoded = json.dumps(dict(self.identity), sort_keys=True, allow_nan=False)
        if not self.identity:
            raise ValueError("Kernel requires a nonempty pinned identity")
        object.__setattr__(self, "_identity_json", encoded)

    def summary(self):
        return {"kind": "trusted_explicit_hex_body_kernel", "identity": json.loads(self._identity_json),
                "state_owner": "core", "scope": "serial_response_only_hex8"}


def prepare_kernel_response(kernel, geometry, owner, material, displacement, gradient,
                            *, deformation_gradient_old, time, time_increment):
    from ..constitutive.material_array_batch import MaterialPointArrayBatchInput
    from ..constitutive._explicit_material import validated_explicit_material_update
    from ..constitutive._explicit_material_driver import ExplicitQuadratureTrial, _immutable_committed_history
    from ._finite_uniform_hex import FiniteUniformHexResponse

    request = MaterialPointArrayBatchInput(deformation_gradient_old, gradient,
        _immutable_committed_history(owner.response.state), material.state_schema, time, time_increment)
    u = _array(displacement, (request.point_count, 8, 3), "cell_displacement")
    outputs = []

    def evaluate(checked):
        output = kernel.evaluate(material, checked, geometry, u)
        if not isinstance(output, ExplicitHexBodyOutput):
            raise TypeError("Body kernel must return ExplicitHexBodyOutput")
        outputs.append(output)
        return output.material_response

    response = validated_explicit_material_update(material, request, response_evaluator=evaluate)
    if np.any(response.requested_time_scale < 1):
        from ..constitutive._explicit_material import ExplicitMaterialCutbackRequest
        raise ExplicitMaterialCutbackRequest("Material requested a smaller increment; trial rejected without state mutation.")
    output = outputs[0]
    # Repeat transport validation at the execution boundary, including mutated
    # NumPy shape metadata, without copying immutable byte-backed payloads.
    output = ExplicitHexBodyOutput(response, output.first_piola_stress,
                                   output.internal_force, output.hourglass_energy)
    energy = None if response.stored_energy_density is None else response.stored_energy_density * geometry.volume
    return ExplicitQuadratureTrial(response, output.first_piola_stress), FiniteUniformHexResponse(
        output.internal_force, energy, output.hourglass_energy)
