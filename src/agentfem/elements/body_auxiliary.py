# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Stateless disjoint-cell adapters; publication follows a complete worker join."""
import numpy as np


def validate_cell_workers(workers):
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ValueError('Cell workers must be an integer in [1,16]')
    return workers


def auxiliary_callbacks(*, workers=1, executor=None):
    from .. import _body_pipeline
    from ._hex_validity import _coordinate_edges
    from ._bulk_viscosity import BulkViscosityResponse
    validate_cell_workers(workers)
    edges = np.ascontiguousarray(_coordinate_edges(), dtype=np.int32)

    def checked(value, shape):
        a = np.asarray(value)
        # The execution boundary has already frozen and finite-checked these
        # arrays. Recheck layout, not every value for a second time per step.
        if a.shape != shape or np.iscomplexobj(a):
            raise ValueError('Native auxiliary array shape contract violated')
        return np.require(a, dtype=float, requirements=('C','A'))

    def kinematics(geometry, displacement):
        n = len(geometry.volume)
        raw = _body_pipeline.auxiliary(0, (
            checked(geometry.coordinates,(n,8,3)), checked(displacement,(n,8,3)),
            checked(geometry.average_gradient,(n,8,3)),
            checked(geometry.reference_perturbation_radius,(n,)), edges), workers, executor)
        return np.frombuffer(raw[0], dtype=float).reshape(n,3,3), np.frombuffer(raw[1], dtype=np.int32).astype(bool)

    def viscosity(geometry, f, velocity, coefficients, reference):
        n = len(geometry.volume)
        if geometry.lumped_mass is None:
            return reference()
        raw = _body_pipeline.auxiliary(1, (
            checked(f,(n,3,3)), checked(geometry.average_gradient,(n,8,3)),
            checked(velocity,(n,8,3)), checked(geometry.lumped_mass,(n,8)),
            checked(geometry.volume,(n,)),
            *[checked(coefficients[name],(n,)) for name in
              ('density','wave_speed','characteristic_length','linear','quadratic')]), workers, executor)
        if np.any(np.frombuffer(raw[-1], dtype=np.int32)):
            return reference()
        return BulkViscosityResponse(*(
            np.frombuffer(value, dtype=float).reshape(shape)
            for value, shape in zip(raw[:-1], ((n,8,3), (n,), (n,), (n,), (n,)), strict=True)))

    return kinematics, viscosity


def gather_callback(*, workers=1, executor=None):
    from .. import _body_pipeline
    validate_cell_workers(workers)

    def gather(values, cell_nodes):
        # Check indices while copying disjoint cells. Never mutate or cache the
        # solver's live nodal data, connectivity, or previous trial's output.
        raw, = _body_pipeline.auxiliary(2, (values, cell_nodes), workers, executor)
        return np.frombuffer(raw, dtype=float).reshape(len(cell_nodes), 8, 3)

    return gather
