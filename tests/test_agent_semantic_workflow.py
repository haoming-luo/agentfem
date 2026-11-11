"""Deterministic tool-facing acceptance, not an LLM success-rate benchmark."""

import json

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import fields, mesh, models, results, studies, units
from agentfem.constitutive import elasticity
from agentfem.step_providers import StepOptionError


def test_inspect_repair_solve_uses_one_model_and_one_option_contract():
    domain = mesh.rectangle(
        (0.0, 0.0), (1.0, 0.2), (4, 2), comm=MPI.COMM_SELF, cell_type="quadrilateral"
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        mesh=domain,
        units=units.si(),
        name="semantic-cantilever",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(elasticity.isotropic_elastic(young=210e9, poisson=0.3, density=7800))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1)
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2)
    model.clamp(displacement, on=left)
    model.traction((0.0, -1e6), on=right)
    before = model.to_ir()
    assert before["root"]["validation"]["valid"]
    assert before["root"]["unit_system"]["base"]["length"] == "m"
    options = {"solver_option": None}
    capability = models.step_capability(model, target=displacement, options=options)
    assert capability["supported"] is False
    with pytest.raises(StepOptionError) as caught:
        model.step(target=displacement, **options)
    assert caught.value.details()["issues"] == capability["option_issues"]
    assert model.steps == []
    assert model.to_ir() == before  # rejected request did not modify the model
    # Explicitly approved spelling correction in the test; not an autonomous repair.
    assert caught.value.details()["issues"][0]["suggestions"] == ("solver_options",)
    assert models.step_capability(
        model, target=displacement, options={"solver_options": None}
    )["supported"]
    result = model.step(target=displacement, solver_options=None).solve_result()
    assert len(model.steps) == 1
    assert np.isfinite(displacement.value.x.array).all()
    assert np.max(np.abs(displacement.value.x.array)) > 0
    assert result.status == "completed"
    assert results.static_force_balance(model.steps[-1].problem).relative_error < 1e-10
    json.dumps(model.to_ir(), allow_nan=False)
