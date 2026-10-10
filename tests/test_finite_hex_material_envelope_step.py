# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""An independent external material drives the ordinary bounded Explicit Step."""

from dataclasses import replace
import numpy as np
import pytest
from mpi4py import MPI

from agentfem import amplitudes, constitutive, fields, models, studies
from test_finite_hex_external_material import ReferenceNeoHookean
from test_finite_hex_step import problem


class BoundedNeoHookean(ReferenceNeoHookean):
    minimum, maximum = 0.95, 1.05
    source = "compressible Neo-Hookean analytic dP/dF norm envelope v1"

    def explicit_stability_envelope(self):
        mu, lam = 30, 40
        coefficient = max(abs(3 * lam * np.log(s) - mu) for s in (self.minimum, self.maximum))
        return constitutive.FirstPiolaTangentEnvelope(
            positive_modulus=mu + (3 * lam + coefficient) / self.minimum**2,
            negative_modulus=max(0, coefficient / self.minimum**2 - mu),
            minimum_stretch=self.minimum, maximum_stretch=self.maximum, source=self.source,
        )


def prepare(material=None, *, comm=MPI.COMM_SELF, **options):
    prototype, _, policy = problem(comm)
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=prototype.mesh)
    u = model.field(fields.displacement(prototype.mesh))
    model.material(material or BoundedNeoHookean())
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2))
    model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
    model.fix(u, on=lambda x: np.isclose(x[0], 1), components=0,
              value=amplitudes.Amplitude("ramp", lambda t: 100 * t * t,
                                        metadata={"coefficient": 100}))
    return model.step(target=u, element_policy=policy,
                      **{"dt": 1e-4, "steps": 100, "progress": False, **options})


def test_material_envelope_matches_explicit_ceiling_and_restart(tmp_path):
    candidate = prepare()
    ceiling = candidate.residual.bound
    reference = prepare(omega_squared_bound=ceiling)
    candidate.solve_result(field_variables=("S",))
    reference.run()
    np.testing.assert_array_equal(candidate.state.u.value.x.array, reference.state.u.value.x.array)
    assert candidate.history_records[-1] == pytest.approx(reference.history_records[-1])
    partial = prepare()
    partial.run(until_step=37)
    checkpoint = partial.save_checkpoint(tmp_path / "envelope")
    resumed = prepare()
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    np.testing.assert_array_equal(candidate.state.u.value.x.array, resumed.state.u.value.x.array)
    assert "provider_domain_envelope" in candidate.summary()["stability_scope"]
    changed = BoundedNeoHookean()
    changed.source = "different derivation revision"
    with pytest.raises(ValueError, match="identity|differ"):
        prepare(changed).load_checkpoint(checkpoint)


def test_auto_increment_comes_from_material_and_operator_not_a_guessed_wave_speed():
    candidate = prepare(dt="auto")
    assert candidate.dt == pytest.approx(0.8 * 2 / np.sqrt(candidate.residual.bound))
    assert candidate.residual.stability.contributions[0].method == (
        "material_domain_envelope_reference_gram_and_hourglass"
    )


def test_domain_violation_rolls_back_entire_increment():
    material = BoundedNeoHookean()
    material.maximum = 1.00000001
    step = prepare(material)
    before = step.state.snapshot()
    residual = step.residual.snapshot()
    with pytest.raises(ValueError, match="envelope domain"):
        step.run(until_step=1)
    assert step.completed_steps == 0
    assert step.residual.snapshot() == residual
    for name, values in before["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][name], values)


def test_false_initial_material_bound_is_rejected():
    class Incorrect(BoundedNeoHookean):
        def explicit_stability_envelope(self):
            return replace(super().explicit_stability_envelope(), positive_modulus=1)

    with pytest.raises(ValueError, match="violates"):
        prepare(Incorrect())


def test_missing_material_envelope_does_not_silently_enable_auto_stability():
    with pytest.raises(ValueError, match="omega_squared_bound"):
        prepare(ReferenceNeoHookean())
