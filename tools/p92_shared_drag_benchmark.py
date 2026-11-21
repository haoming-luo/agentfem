"""Literature-informed ideal stress-step check, NOT measured P92 validation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from tools import p92_shared_drag_reference as kernel
from tools.p92_measured_fit import write_new


def elastic_stress_step(state, target, material):
    """Ideal instantaneous elastic step; no finite-time flow or history reset."""
    result = np.asarray(state, dtype=float).copy()
    delta = (target-result[0])/material.modulus
    result[8] += (target+result[0])*delta/2
    result[7] += delta
    result[0] = target
    return result


def benchmark(*, method="Radau", rtol=1e-9):
    mat = kernel.unverified_table2_transcription()
    initial = elastic_stress_step(np.zeros(12), 210., mat)
    forward = kernel.integrate(initial, 15*3600., 0., mat, stress_control=True, method=method, rtol=rtol)
    virgin_rate = kernel.rates(initial, 0., mat, stress_control=True)[6]
    records = []
    for reverse_stress in (0., -120., -190.):
        reversed_state = elastic_stress_step(forward.y[:, -1], reverse_stress, mat)
        reverse = kernel.integrate(reversed_state, 3600., 0., mat, stress_control=True, method=method, rtol=rtol)
        reload = elastic_stress_step(reverse.y[:, -1], 210., mat)
        reloaded = kernel.integrate(reload, 15*3600., 0., mat, stress_control=True, method=method, rtol=rtol)
        query = np.array([0., 1., 10., 60., 600., 3600., 15*3600.])
        end = reloaded.y[:, -1]
        records.append(dict(reverse_stress_MPa=reverse_stress,
            reload_initial_flow_per_s=float(kernel.rates(reload, 0., mat, stress_control=True)[6]),
            reload_to_virgin_rate=float(kernel.rates(reload, 0., mat, stress_control=True)[6]/virgin_rate),
            reload_backstress_MPa=float(sum(reload[list(kernel.X_INDICES)])),
            reload_drag_MPa=float(mat.initial_drag*np.exp(reload[5])),
            reload_time_s=query.tolist(), reload_states=reloaded.sol(query).T.tolist(),
            end_balance_MPa=float(end[8]-end[9]-kernel.stored_energy(end, mat))))
    return dict(schema="agentfem.research.shared-drag-ideal-stress-check.v1",
        source_doi="10.1016/j.ijmecsci.2020.106044", original_material="10%Cr at 600 C, not P92 at 620 C",
        model_id=kernel.MODEL_ID, parameters=asdict(mat), method=method, rtol=rtol, records=records,
        table_transcription_verified=False, published_reproduction_verified=False,
        independent_experimental_validation=False, engineering_qualified=False,
        scope="Declared 210 MPa/15 h, reverse/1 h, 210 MPa/15 h ideal-step discriminator; not digitized paper curves")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = benchmark()
    independent = benchmark(method="BDF", rtol=1e-10)
    delta = np.array([np.array(a["reload_states"])-b["reload_states"]
                     for a, b in zip(result["records"], independent["records"], strict=True)])
    errors = dict(stress_MPa=float(abs(delta[:, :, [0, 1, 2, 4]]).max()),
        strain=float(abs(delta[:, :, [3, 6, 7, 10, 11]]).max()),
        log_drag=float(abs(delta[:, :, 5]).max()), energy_MPa=float(abs(delta[:, :, [8, 9]]).max()))
    result["numerical_comparison"] = dict(errors=errors, passed=bool(errors["stress_MPa"] < 1e-4
        and errors["strain"] < 1e-8 and errors["log_drag"] < 1e-8 and errors["energy_MPa"] < 1e-6))
    write_new(args.output, result)
    print(json.dumps(dict(rates=[r["reload_initial_flow_per_s"] for r in result["records"]],
                          numerical=result["numerical_comparison"])))
