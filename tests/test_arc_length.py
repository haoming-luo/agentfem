# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Physics-independent continuation and its public FEM adapter."""

import numpy as np
import pytest
from mpi4py import MPI
from agentfem import solvers, mesh, models, studies, fields, constitutive, procedures


def arch_force(u):
    a, h, ea = 1.0, 0.2, 10.0
    y = h - u[0]
    length = np.sqrt(a * a + y * y)
    original = np.sqrt(a * a + h * h)
    force = 2 * ea * (1 / length - 1 / original) * y
    tangent = 2 * ea * (1 / original - a * a / length**3)
    return np.array([force]), np.array([[tangent]])


def arch_path(scale=1.0):
    return solvers.ArcLengthPath(
        lambda u: tuple(scale * x for x in arch_force(u)),
        [0.03 * scale],
        options=solvers.ArcLengthOptions(
            initial=0.06, maximum=0.06, displacement_scale=0.2, load_scale=1.0
        ),
    )


def test_shallow_arch_crosses_both_limit_points_and_load_reversal():
    p = arch_path()
    p.run(100, stop=lambda r: r["displacement"][0] > 0.42)
    q = np.array([r["displacement"][0] for r in p.history])
    load = np.array([r["load_factor"] * 0.03 for r in p.history])
    exact = np.array([arch_force(np.array([v]))[0][0] for v in q])
    np.testing.assert_allclose(load, exact, atol=3.0e-10)
    assert np.all(np.diff(q) > 0)
    assert np.any(np.diff(load) > 0) and np.any(np.diff(load) < 0)
    assert load.min() < 0 and q[-1] > 0.4
    assert max(r["arc_error"] for r in p.history) < 1.0e-8
    scaled = arch_path(1.0e6)
    scaled.run(len(q))
    np.testing.assert_allclose(scaled.displacement, p.displacement, rtol=1.0e-9)
    assert scaled.load_factor == pytest.approx(p.load_factor, rel=1.0e-9)


def test_cutback_and_failure_leave_accepted_state_unchanged():
    state = {"fail": False, "always": False}

    def evaluate(u):
        if state["fail"] and u[0] != 0:
            state["fail"] = state["always"]
            raise FloatingPointError("invalid trial")
        return u.copy(), np.eye(1)

    p = solvers.ArcLengthPath(evaluate, [1.0])
    state["fail"] = True
    record = p.advance()
    assert record["cutbacks"] == 1 and not p.attempts[0]["accepted"]
    p = solvers.ArcLengthPath(evaluate, [1.0])
    state.update(fail=True, always=True)
    with pytest.raises(RuntimeError, match="accepted state unchanged"):
        p.advance()
    np.testing.assert_array_equal(p.displacement, [0.0])
    assert p.load_factor == 0 and not p.history


def test_fixed_load_initial_equilibrium_and_guardrails():
    p = solvers.ArcLengthPath(
        lambda u: (2 * u, 2 * np.eye(2)),
        [1.0, 0.0],
        dead_load=[2.0, 0.0],
        initial=[1.0, 0.0],
    )
    p.run(3)
    np.testing.assert_allclose(
        2 * p.displacement, [2 + p.load_factor, 0.0], atol=1.0e-10
    )
    with pytest.raises(ValueError, match="initial state"):
        solvers.ArcLengthPath(lambda u: (u, np.eye(1)), [1.0], initial=[1.0])
    with pytest.raises(ValueError):
        solvers.ArcLengthOptions(initial=0.0)


def patch(dim=2, comm=MPI.COMM_SELF):
    d = (
        mesh.rectangle(
            (0.0, 0.0),
            (1.0, 0.2),
            (4, 2),
            cell_type="quadrilateral",
            comm=comm,
        )
        if dim == 2
        else mesh.cuboid(
            (0.0, 0.0, 0.0),
            (1.0, 0.2, 0.2),
            (3, 2, 2),
            cell_type="hexahedron",
            comm=comm,
        )
    )
    m = models.create(
        study=studies.static_solid(
            dimension=dim,
            assumption="plane_strain" if dim == 2 else None,
            nonlinear=True,
        ),
        mesh=d,
    )
    u = m.field(fields.displacement(d, degree=2))
    m.material(constitutive.neo_hookean(young=1000.0, poisson=0.0))
    left = mesh.boundary(d, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(d, lambda x: np.isclose(x[0], 1.0), name="right")
    m.clamp(u, on=left)
    m.traction((1.0,) + (0.0,) * (dim - 1), on=right)
    return m, u


@pytest.mark.parametrize("dim", [2, 3])
def test_public_arc_length_provider_matches_hyperelastic_patch(dim):
    m, u = patch(dim)
    step = m.step(
        target=u,
        procedure=procedures.arc_length(),
        increments=4,
        arc_options=solvers.ArcLengthOptions(
            initial=0.1, maximum=0.1, displacement_scale=0.001
        ),
    )
    r = step.solve_result()
    lam = r.quantity("load_factor")
    # nu=0 compressible neo-Hookean nominal P=mu*(stretch-1/stretch).
    stretch = (lam / 500 + np.sqrt((lam / 500) ** 2 + 4)) / 2
    from agentfem import results

    tip = results.probe(u.value, at=(1.0, 0.1) if dim == 2 else (1.0, 0.1, 0.1))
    assert float(np.asarray(tip).reshape(-1)[0]) == pytest.approx(
        stretch - 1, rel=2.0e-5, abs=1.0e-9
    )
    assert max(r.quantity("equilibrium_residuals")) < 1.0e-8


def test_solid_arch_crosses_limit_point_with_step_refinement():
    import runpy
    from pathlib import Path

    if MPI.COMM_WORLD.size != 1:
        pytest.skip("continuum arc-length adapter is explicitly serial")
    solve = runpy.run_path(
        str(Path(__file__).parents[1] / "examples/shallow_arch_arc_length/case.py")
    )["solve_arch"]
    r, u, load = solve(nx=32, arc_size=0.04, increments=80)
    _, fine_u, fine_load = solve(nx=64, arc_size=0.02, increments=160)
    assert np.all(np.diff(-u) > 0) and np.all(np.diff(-fine_u) > 0)
    assert np.any(np.diff(load) < 0) and np.any(np.diff(fine_load) < 0)
    assert max(r.quantity("equilibrium_residuals")) < 1.0e-8
    selected = np.linspace(0.02, 0.32, 30)
    coarse = np.interp(selected, -u, load)
    fine = np.interp(selected, -fine_u, fine_load)
    assert np.max(np.abs(coarse - fine)) / np.max(np.abs(fine)) < 0.05


def test_adapter_rejects_nonzero_displacement_and_custom_solver_options():
    m, u = patch()
    top = mesh.boundary(
        u.value.function_space.mesh, lambda x: np.isclose(x[1], 0.2), name="top"
    )
    m.prescribe(u, 0.01, on=top, component=1)
    with pytest.raises(
        (ValueError, NotImplementedError), match="[Hh]omogeneous|nonzero|non-zero"
    ):
        m.step(target=u, procedure=procedures.arc_length(), increments=3)
    m, u = patch()
    with pytest.raises(ValueError, match="override"):
        m.step(
            target=u,
            procedure=procedures.arc_length(),
            increments=3,
            solver_options=solvers.LinearSolverOptions(),
        )


@pytest.mark.skipif(
    MPI.COMM_WORLD.size == 1, reason="checks explicit multi-rank rejection"
)
def test_distributed_fem_adapter_fails_explicitly():
    m, u = patch(comm=MPI.COMM_WORLD)
    with pytest.raises(NotImplementedError, match="serial"):
        m.step(target=u, procedure=procedures.arc_length(), increments=3)
