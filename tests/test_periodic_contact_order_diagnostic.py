# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""The order diagnostic must not leave quadratic periodic trace DOFs free."""
from pathlib import Path
import runpy

import numpy as np
import pytest


def functions():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]
                             / "tools/diagnose_periodic_contact_order.py"))


@pytest.mark.parametrize("degree", [1, 2])
def test_all_boundary_field_dofs_obey_periodic_motion_and_guides(degree):
    step = functions()["make_step"](degree=degree, increments=4, yield_stress=1e6)
    coordinates = step.solution.function_space.tabulate_dof_coordinates()
    reduction = step.constraint.reduction(.25)
    from scipy.sparse import csr_matrix
    T = csr_matrix((reduction.coefficients, reduction.column_indices, reduction.row_offsets),
                   shape=(reduction.full_size, reduction.reduced_size))
    # An arbitrary reduced vector exercises the constraint null space, not only
    # the trivial affine solution which would conceal missing midpoint rows.
    q = np.random.default_rng(241).normal(size=reduction.reduced_size)
    u = (T @ q + reduction.offset).reshape(-1, 3)
    keys = {tuple(np.rint(x * 1e12).astype(int)): i for i, x in enumerate(coordinates)}
    F = np.diag([1.02, 1.02**-.5, 1.02**-.5])
    checked = 0
    for i, x in enumerate(coordinates):
        axes = np.isclose(x, 1.)
        if np.any(axes):
            wrapped = np.where(axes, 0., x)
            j = keys[tuple(np.rint(wrapped * 1e12).astype(int))]
            np.testing.assert_allclose(u[i] - u[j], (F - np.eye(3)) @ (x - wrapped), atol=2e-14)
            checked += 1
        if np.isclose(x[2], 0.) or np.isclose(x[2], 1.):
            assert u[i, 0] == pytest.approx(.02 * x[0], abs=2e-14)
    assert checked > 50
    if degree == 2:
        assert len(coordinates) > 125
        assert step.contact_provider.adapter.interpolation_degree == 2


def test_quadratic_periodic_gauss_contact_accepts_current_total_residual():
    diagnostic = functions()
    step = diagnostic["make_step"](degree=2, increments=8)
    step.solve(until=.25)
    assert step.constraint.mismatch(.25) < 1e-12
    assert step.contact_provider.evaluate().active_points > 0
    observation = diagnostic["observe_penalty"](step, .25)
    assert observation["peeq_reference_volume_mean"] > 0
    assert "normal_relative_work" not in observation
    assert observation["normal_relative_work_status"] == "unavailable_on_this_diagnostic_route"
    full = step.assemble_full_residual()
    T = step.constraint.reduction(.25).matrix(step.solution.function_space.mesh.comm)
    r = T.createVecRight()
    try:
        T.multTranspose(full, r)
        assert r.norm() < 1e-7
    finally:
        full.destroy()
        r.destroy()
        T.destroy()


def test_duplicate_field_coordinates_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        functions()["periodic_field_equations"](np.zeros((2, 3)))


@pytest.mark.parametrize("limit,measured", [(None, .001), (.000375, .0003), (.000375, .0004)])
def test_diagnostic_records_and_enforces_selected_path_policy(tmp_path, monkeypatch, limit, measured):
    import json
    import sys
    from types import SimpleNamespace
    main = functions()["main"]
    output = tmp_path / "comparison.json"
    arguments = ["diagnostic", "--cells", "4", "--degrees", "2", "--output", str(output)]
    if limit is not None:
        arguments += ["--plastic-increment-limit", str(limit)]
    monkeypatch.setattr(sys, "argv", arguments)
    policies = []
    def make(**kwargs):
        policies.append(kwargs["incrementation"])
        history = []
        def solve(*, until):
            history.append(SimpleNamespace(converged=True, iterations=2,
                checks={"maximum_plastic_increment": measured}))
        return SimpleNamespace(solve=solve, accepted_increments=history, attempted_increments=history,
            solution=SimpleNamespace(x=SimpleNamespace(array=np.zeros(3))))
    monkeypatch.setitem(main.__globals__, "make_step", make)
    monkeypatch.setitem(main.__globals__, "observe_penalty", lambda step, factor: {"factor": factor})
    if limit is not None and measured > limit:
        with pytest.raises(RuntimeError, match="failed; evidence preserved"):
            main()
    else:
        main()
    report = json.loads(output.read_text())
    assert report["incrementation"] == json.loads(json.dumps(policies[0].summary()))
    row = report["records"][0]
    assert row["maximum_accepted_plastic_increment"] == measured
    assert row["rejected_attempts"] == 0 and row["newton_iterations"] == 8
    assert (row["status"] == "failed") == (limit is not None and measured > limit)
    assert report["accuracy_claim"].startswith("none:")
