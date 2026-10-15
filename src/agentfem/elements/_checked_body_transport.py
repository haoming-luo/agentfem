# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Internal lowering of the exact built-in checked C result, not a provider API.

Only native_material_execution calls this immediately after run_checked. The C
worker checks finite stress/state/work/P/force/energy, symmetric Cauchy stress
and nonnegative hourglass energy before publishing immutable bytes. External
callbacks and ordinary constructors keep full validation. This is trusted
implementation code, not a security boundary against arbitrary Python code.
"""
import numpy as np
from ..constitutive._explicit_material import ExplicitMaterialBatchOutput, _response_transport_stamp
from .explicit_execution import ExplicitHexBodyOutput, _body_transport_stamp


def checked_body_transport(raw, schema, count):
    shapes = ((count, 3, 3), (count, schema.size), (count,),
              (count, 3, 3), (count, 8, 3), (count,))
    if type(raw) is not tuple or len(raw) != len(shapes):
        raise TypeError('Invalid built-in checked body payload')
    values = []
    for data, shape in zip(raw, shapes, strict=True):
        if type(data) is not bytes or len(data) != int(np.prod(shape))*8:
            raise ValueError('Invalid built-in checked body storage')
        values.append(np.frombuffer(data, dtype=float).reshape(shape))
    stress, state, work, piola, force, energy = values
    response = object.__new__(ExplicitMaterialBatchOutput)
    for name, value in dict(cauchy_stress=stress, state_new=state, state_schema=schema,
            internal_work_density_increment=work, stored_energy_density=None,
            dissipation_density_increment=None,
            requested_time_scale=np.frombuffer(np.ones(count).tobytes(), dtype=float)).items():
        object.__setattr__(response, name, value)
    object.__setattr__(response, '_transport_stamp', _response_transport_stamp(response))
    output = object.__new__(ExplicitHexBodyOutput)
    for name, value in dict(material_response=response, first_piola_stress=piola,
                           internal_force=force, hourglass_energy=energy).items():
        object.__setattr__(output, name, value)
    object.__setattr__(output, '_transport_stamp', _body_transport_stamp(output))
    return output
