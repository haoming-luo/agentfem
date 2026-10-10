# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private serial quadrature lowering for response-only material trials.

Reuse MaterialQuadratureState. No second state store, implicit commit,
synthetic tangent, artificial storage field or global Step registration.
"""

from dataclasses import dataclass

import numpy as np

from ._explicit_material import (
    ExplicitMaterialBatchOutput,
    validated_explicit_material_update,
)
from .material_array_batch import MaterialPointArrayBatchInput, _array
from .material_driver import _cauchy_to_first_piola
from ..state import field_transaction


@dataclass(frozen=True)
class ExplicitQuadratureTrial:
    response: ExplicitMaterialBatchOutput
    first_piola_stress: np.ndarray


@dataclass
class ExplicitQuadratureResponse:
    """Real stress/work fields only; no placeholder DPDF or SENER exists."""

    state: object
    first_piola_stress: object
    cauchy_stress: object
    internal_work_density_increment: object
    response_transaction: object

    @classmethod
    def create(cls, domain, state_schema, *, degree=1):
        from .quadrature import (
            MaterialQuadratureState,
            QuadratureField,
            QuadratureTransaction,
        )

        state = MaterialQuadratureState.create(domain, state_schema, degree=degree)
        shapes = {"P": (3, 3), "S": (3, 3), "WORK": ()}
        trial = {
            name: QuadratureField.create(
                domain, name=name, value_shape=shape, degree=degree
            )
            for name, shape in shapes.items()
        }
        committed = {
            name: QuadratureField.create(
                domain, name=name + "_accepted", value_shape=shape, degree=degree
            )
            for name, shape in shapes.items()
        }
        return cls(
            state,
            trial["P"],
            trial["S"],
            trial["WORK"],
            QuadratureTransaction(committed, trial, "explicit-material-response"),
        )

    @property
    def domain(self):
        return self.state.domain

    def trial_fields(self):
        return {
            "P": self.first_piola_stress.function,
            "S": self.cauchy_stress.function,
            "WORK": self.internal_work_density_increment.function,
            **{
                f"state_{name}": value.function
                for name, value in self.state.trial.items()
            },
        }

    def update(self, material, **options):
        with field_transaction(**self.trial_fields()):
            trial = update_explicit_material_points(material, self.state, **options)
            self.first_piola_stress.assign(trial.first_piola_stress)
            self.cauchy_stress.assign(trial.response.cauchy_stress)
            self.internal_work_density_increment.assign(
                trial.response.internal_work_density_increment
            )
        return trial.response

    def commit(self):
        fields = {
            f"state_{name}": value.function
            for name, value in self.state.committed.items()
        }
        fields.update(
            {
                f"response_{name}": value.function
                for name, value in self.response_transaction.committed.items()
            }
        )
        with field_transaction(**fields):
            self.state.commit()
            self.response_transaction.commit()

    def rollback(self):
        with field_transaction(**self.trial_fields()):
            self.state.rollback()
            self.response_transaction.rollback()


def update_explicit_material_points(
    material,
    state,
    *,
    deformation_gradient_old,
    deformation_gradient_new,
    time,
    time_increment,
    properties=(),
    max_batch_points=1024,
):
    """Prepare all chunks before atomically writing trial fields; caller commits.

    MPI is intentionally refused until assignment/ghost exchange failures have
    their own collective acceptance. A per-point cutback request rejects this
    fixed-increment candidate before touching trial or committed fields.
    """
    if state.domain.comm.size != 1:
        raise NotImplementedError(
            "Response-only material quadrature MPI is not yet admitted."
        )
    if state.state_schema != material.state_schema:
        raise ValueError("Material and quadrature state schemas differ.")
    if (
        isinstance(max_batch_points, bool)
        or not isinstance(max_batch_points, (int, np.integer))
        or max_batch_points < 1
    ):
        raise ValueError("max_batch_points must be a positive integer.")
    accepted = state.committed_state_vectors()
    count = len(accepted)
    if count == 0:
        raise ValueError("Serial material quadrature requires at least one point.")
    old = _array(deformation_gradient_old, (count, 3, 3), "deformation_gradient_old")
    new = _array(deformation_gradient_new, (count, 3, 3), "deformation_gradient_new")
    batches = []
    for start in range(0, count, max_batch_points):
        region = slice(start, start + max_batch_points)
        request = MaterialPointArrayBatchInput(
            old[region],
            new[region],
            accepted[region],
            state.state_schema,
            time,
            time_increment,
            properties=np.asarray(properties),
        )
        batches.append(validated_explicit_material_update(material, request))
    channels = (
        "cauchy_stress",
        "state_new",
        "internal_work_density_increment",
        "stored_energy_density",
        "dissipation_density_increment",
        "requested_time_scale",
    )
    combined = {}
    for name in channels:
        values = [getattr(batch, name) for batch in batches]
        if any(value is None for value in values):
            if not all(value is None for value in values):
                raise ValueError(
                    f"Explicit material changed {name} availability across chunks."
                )
            combined[name] = None
        else:
            combined[name] = np.concatenate(values)
    result = ExplicitMaterialBatchOutput(state_schema=state.state_schema, **combined)
    if np.any(result.requested_time_scale < 1):
        raise ValueError(
            "Material requested a smaller increment; trial rejected without state mutation."
        )
    piola = _cauchy_to_first_piola(new, result.cauchy_stress)
    piola.setflags(write=False)
    # The existing State remains the sole owner; failed late assignment restores
    # all previously visible trial values, not a partly updated batch.
    with field_transaction(
        **{name: value.function for name, value in state.trial.items()}
    ):
        state.assign_trial_state_vectors(result.state_new)
    return ExplicitQuadratureTrial(result, piola)
