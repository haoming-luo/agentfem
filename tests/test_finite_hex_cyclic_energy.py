# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Public finite-Hex loading reversals, independent coaxial history and work."""

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

from agentfem import amplitudes, constitutive, elements, fields, models, studies


def prepare(steps):
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, 1, 1, 1, cell_type=mesh.CellType.hexahedron
    )
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=100,
            poisson=0.3,
            yield_stress=1,
            hardening_modulus=5,
            density=2,
        )
    )
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2))
    model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
    model.fix(
        u,
        on=lambda x: np.isclose(x[0], 1),
        components=0,
        value=amplitudes.Amplitude(
            "cyclic_log_stretch",
            lambda t: np.expm1(0.15 * np.sin(2 * np.pi * t)),
            metadata={"log_amplitude": 0.15, "period": 1.0},
        ),
    )
    return model.step(
        target=u,
        element_policy=elements.uniform_strain_hex8(
            hourglass_modulus=40,
            hourglass_scale=0.1,
            kinematics="finite_strain",
        ),
        omega_squared_bound=1e4,
        maximum_negative_growth_per_increment=0.1,
        dt=2 / steps,
        steps=steps,
        progress=False,
    )


def scalar_reference(steps):
    """Signed coaxial plastic logarithmic strain and accumulated equivalent strain.

    This scalar recurrence does not call the material implementation, SVD or
    spectral return code. Reversal keeps signed plastic strain distinct from PEEQ.
    """
    mu, bulk, hardening = 100 / 2.6, 100 / 1.2, 5.0
    plastic, peeq = 0.0, 0.0
    for t in np.linspace(0, 2, steps + 1)[1:]:
        log_stretch = 0.15 * np.sin(2 * np.pi * t)
        trial = log_stretch - 1.5 * plastic
        increment = max(
            0.0, (2 * mu * abs(trial) - 1 - hardening * peeq) / (3 * mu + hardening)
        )
        plastic += np.sign(trial) * increment
        peeq += increment
    elastic = log_stretch - 1.5 * plastic
    stress = np.diag(
        [
            bulk * log_stretch + 4 * mu / 3 * elastic,
            bulk * log_stretch - 2 * mu / 3 * elastic,
            bulk * log_stretch - 2 * mu / 3 * elastic,
        ]
    ) / np.exp(log_stretch)
    stored = (
        bulk / 2 * log_stretch**2 + 2 * mu / 3 * elastic**2 + hardening / 2 * peeq**2
    )
    return stress, stored, peeq


def test_public_cyclic_plastic_work_refines_and_matches_independent_history():
    errors = []
    for count in (400, 800):
        step = prepare(count)
        step.run()
        stress, stored, peeq = scalar_reference(count)
        response = step.residual.internal.response
        np.testing.assert_allclose(
            response.cauchy_stress.owned_values[0], stress, atol=2e-11
        )
        row = step.history_records[-1]
        assert row["bulk_stored_energy"] == pytest.approx(stored, abs=2e-12)
        assert row["material_dissipation"] == pytest.approx(peeq, abs=2e-12)
        assert row["material_dissipation"] > 0.5
        assert (
            np.min(
                np.diff([item["material_dissipation"] for item in step.history_records])
            )
            >= -1e-14
        )
        assert row["hourglass_energy"] < 1e-24
        errors.append(abs(row["energy_balance_error"]))
    assert errors[1] < 0.4 * errors[0]
    assert errors[1] < 2e-3


def test_cyclic_restart_after_load_reversal_is_exact(tmp_path):
    continuous = prepare(400)
    continuous.run()
    partial = prepare(400)
    partial.run(until_step=163)
    path = partial.save_checkpoint(tmp_path / "cyclic")
    restored = prepare(400)
    restored.load_checkpoint(path)
    restored.run()
    assert restored.history_records == continuous.history_records
    assert restored.residual.snapshot() == continuous.residual.snapshot()
