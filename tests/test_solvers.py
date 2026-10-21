from __future__ import annotations

import pytest

from agentfem import steps
from agentfem.diagnostics import (
    SolveEventRecorder,
    StandardRunReporter,
    compose_reporters,
)
from agentfem.solvers import (
    LinearSolverOptions,
    NewtonSolverOptions,
    SolveEvent,
    direct_solver,
    newton,
)


def test_solver_options_are_inspectable():
    options = LinearSolverOptions(
        ksp_type="cg",
        pc_type="hypre",
        rtol=1.0e-8,
        max_it=500,
    )

    assert options.summary() == {
        "kind": "linear_solver_options",
        "mpc_assembly": "native",
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1.0e-8,
        "atol": None,
        "max_it": 500,
        "factor_solver_type": None,
        "error_if_not_converged": True,
    }
    assert options.petsc_options() == {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_error_if_not_converged": False,
        "ksp_rtol": 1.0e-8,
        "ksp_max_it": 500,
    }


@pytest.mark.parametrize(
    "kwargs",
    (
        {"rtol": 0.0},
        {"atol": -1.0},
        {"max_it": 0},
        {"pc_type": "gamg", "factor_solver_type": "mumps"},
    ),
)
def test_solver_options_reject_invalid_tolerances(kwargs):
    with pytest.raises(ValueError):
        LinearSolverOptions(**kwargs)


def test_public_newton_policy_adapts_to_snes_and_affine_reduction():
    policy = newton(
        relative_tolerance=1.0e-7,
        absolute_tolerance=1.0e-9,
        maximum_iterations=24,
        linear_solver=direct_solver(package="mumps"),
    )

    assert isinstance(policy, NewtonSolverOptions)
    assert policy.summary()["kind"] == "newton_solver"
    assert policy.summary()["linear_solver"]["factor_solver_type"] == "mumps"
    assert policy.for_snes().petsc_options() == {
        "snes_type": "newtonls",
        "snes_rtol": 1.0e-7,
        "snes_atol": 1.0e-9,
        "snes_max_it": 24,
        "snes_error_if_not_converged": True,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "snes_linesearch_type": "bt",
        "pc_factor_mat_solver_type": "mumps",
    }
    affine = policy.for_affine_reduction()
    assert affine.rtol == pytest.approx(1.0e-7)
    assert affine.max_it == 24
    assert affine.factor_solver_type == "mumps"


def test_newton_preserves_explicit_inner_linear_tolerances_on_both_routes():
    linear = LinearSolverOptions(ksp_type="gmres", pc_type="ilu",
                                 rtol=3e-12, atol=7e-14, max_it=321)
    policy = newton(linear_solver=linear)
    snes = policy.for_snes()
    options = snes.petsc_options()
    assert options["ksp_rtol"] == linear.rtol
    assert options["ksp_atol"] == linear.atol
    assert options["ksp_max_it"] == linear.max_it
    assert policy.for_affine_reduction().linear_options == linear
    assert snes.summary()["ksp_rtol"] == linear.rtol
    # Checkpoint identity must distinguish this corrected policy from an old
    # execution which silently dropped explicitly requested inner tolerances.
    assert policy.summary()["linear_tolerance_transport"] == "explicit.v1"


def test_default_newton_keeps_unspecified_inner_backend_defaults_and_identity():
    policy = newton()
    assert "linear_tolerance_transport" not in policy.summary()
    for key in ("ksp_rtol", "ksp_atol", "ksp_max_it"):
        assert key not in policy.for_snes().petsc_options()
        assert key not in policy.for_snes().summary()


@pytest.mark.parametrize("kwargs", [{"rtol": float("nan")}, {"atol": float("inf")},
                                   {"max_it": 1.5}, {"max_it": True}])
def test_inner_linear_options_reject_nonfinite_or_nonintegral_controls(kwargs):
    with pytest.raises(ValueError):
        LinearSolverOptions(**kwargs)


def test_snes_backend_receives_declared_inner_tolerances(monkeypatch):
    import numpy as np
    import ufl
    from dolfinx import fem, mesh
    import dolfinx.fem.petsc as fem_petsc
    from mpi4py import MPI
    from agentfem import solvers

    domain = mesh.create_unit_interval(MPI.COMM_SELF, 3)
    space = fem.functionspace(domain, ("Lagrange", 1))
    u = fem.Function(space)
    u.x.array[:] = .8
    v = ufl.TestFunction(space)
    residual = (u**3 + u - 2.) * v * ufl.dx
    original = fem_petsc.NonlinearProblem
    captured = []
    def build(*args, **kwargs):
        problem = original(*args, **kwargs)
        captured.append(problem)
        return problem
    monkeypatch.setattr(fem_petsc, "NonlinearProblem", build)
    linear = LinearSolverOptions(ksp_type="gmres", pc_type="ilu",
                                 rtol=3e-12, atol=7e-14, max_it=321)
    _, info = solvers.solve_nonlinear_problem(
        residual, u, options=newton(linear_solver=linear),
        petsc_options_prefix="test_inner_tolerance_transport_",
    )
    assert info.converged
    np.testing.assert_allclose(u.x.array, 1., atol=1e-9, rtol=0.)
    rtol, atol, _, maximum = captured[0].solver.getKSP().getTolerances()
    assert (rtol, atol, maximum) == (linear.rtol, linear.atol, linear.max_it)


@pytest.mark.parametrize("line_search", ("invalid", "cubic"))
def test_public_newton_policy_rejects_unknown_line_search(line_search):
    with pytest.raises(ValueError):
        newton(line_search=line_search)


def test_automatic_incrementation_uses_a_limit_not_a_requested_count():
    control = steps.automatic(
        initial=0.1,
        minimum=1.0e-4,
        maximum=0.5,
        max_increments=10,
    )

    assert control.max_increments == 10
    assert control.after_convergence(0.1, iterations=3) == pytest.approx(0.15)
    assert control.after_convergence(0.4, iterations=3) == pytest.approx(0.5)
    assert control.after_failure(0.2) == pytest.approx(0.05)
    assert control.summary()["kind"] == "automatic_incrementation"


def test_fixed_increments_are_an_explicit_compatibility_mode():
    control = steps.fixed(4)

    assert control.load_factors == (0.25, 0.5, 0.75, 1.0)
    assert control.increments == 4


def test_standard_reporter_flushes_concise_status_and_iteration_text(
    tmp_path,
    capsys,
):
    comm = type("Comm", (), {"rank": 0})()
    status = tmp_path / "job.sta"
    reporter = StandardRunReporter(comm, status_file=status, verbosity=2)
    base = {
        "step_name": "nonlinear_static",
        "step_number": 1,
        "increment": 1,
        "attempt": 1,
        "start_factor": 0.0,
        "target_factor": 0.25,
    }

    reporter.emit(
        SolveEvent(
            "step_started",
            "nonlinear_static",
            incrementation="automatic_incrementation",
        )
    )
    reporter.emit(SolveEvent("increment_started", **base))
    reporter.emit(
        SolveEvent(
            "iteration",
            **base,
            iteration=1,
            residual_norm=1.0e-5,
            step_length=1.0,
        )
    )
    reporter.emit(
        SolveEvent(
            "increment_converged",
            **base,
            iteration=1,
            residual_norm=1.0e-10,
        )
    )

    console = capsys.readouterr().out
    assert "[STEP 1]" in console
    assert "[INC 1 | ATT 1]" in console
    assert "ITER 01" in console
    assert "CONVERGED" in status.read_text(encoding="utf-8")


def test_standard_reporter_default_hides_routine_newton_noise_but_keeps_status(
    tmp_path,
    capsys,
):
    comm = type("Comm", (), {"rank": 0})()
    status = tmp_path / "job.sta"
    reporter = StandardRunReporter(comm, status_file=status, verbosity=0)
    base = {
        "step_name": "nonlinear_static",
        "step_number": 1,
        "increment": 1,
        "attempt": 1,
        "start_factor": 0.0,
        "target_factor": 0.25,
    }

    reporter.emit(SolveEvent("increment_started", **base))
    reporter.emit(
        SolveEvent(
            "iteration",
            **base,
            iteration=1,
            residual_norm=1.0e-5,
        )
    )
    reporter.emit(
        SolveEvent(
            "increment_converged",
            **base,
            iteration=1,
            residual_norm=1.0e-10,
        )
    )

    assert capsys.readouterr().out == ""
    assert "CONVERGED" in status.read_text(encoding="utf-8")


def test_standard_reporter_heartbeat_exposes_long_nonlinear_iteration(
    tmp_path,
    capsys,
):
    comm = type("Comm", (), {"rank": 0})()
    status = tmp_path / "job.sta"
    reporter = StandardRunReporter(
        comm,
        status_file=status,
        verbosity=0,
        heartbeat_seconds=30.0,
    )
    reporter._last_heartbeat -= 31.0

    reporter.emit(
        SolveEvent(
            "iteration",
            "finite_strain_j2",
            step_number=1,
            increment=7,
            attempt=1,
            start_factor=0.30,
            target_factor=0.35,
            iteration=4,
            residual_norm=2.5e-7,
        )
    )

    console = capsys.readouterr().out
    assert "[INC 7 | ITER 4] RUNNING" in console
    assert "progress=35.0%" in console
    assert "load=0.35" in console
    assert "residual=2.500000e-07" in console
    assert "RUNNING" in status.read_text(encoding="utf-8")


def test_hidden_progress_event_is_recorded_but_not_printed(capsys):
    comm = type("Comm", (), {"rank": 0})()
    recorder = SolveEventRecorder()
    reporter = compose_reporters(
        recorder,
        StandardRunReporter(comm, show_iterations=False),
    )
    event = SolveEvent(
        "time_increment",
        "heat",
        increment=2,
        time=0.2,
        total_increments=4,
        display=False,
    )

    reporter.emit(event)

    assert capsys.readouterr().out == ""
    assert recorder.records()[0]["time"] == 0.2
    assert SolveEvent.from_dict(event.as_dict()) == event
    assert SolveEvent(
        "iteration",
        "nonlinear",
        residual_norm=float("inf"),
    ).as_dict()["residual_norm"] is None


def test_transient_reporter_heartbeat_and_event_retention_are_bounded(capsys):
    comm = type("Comm", (), {"rank": 0})()
    reporter = StandardRunReporter(comm, heartbeat_seconds=0.0)
    reporter.emit(
        SolveEvent(
            "time_increment",
            "explicit",
            increment=4,
            time=0.4,
            total_increments=10,
            display=False,
            message="energy_err=1.0e-06",
        )
    )
    console = capsys.readouterr().out
    assert "4/10" in console
    assert "ETA~" in console
    assert "energy_err=1.0e-06" in console

    recorder = SolveEventRecorder(max_events=16)
    for index in range(100):
        recorder.emit(
            SolveEvent(
                "time_increment",
                "explicit",
                increment=index + 1,
                time=float(index + 1),
                total_increments=100,
                display=False,
            )
        )
    recorder.emit(SolveEvent("transient_completed", "explicit", increment=100))
    assert len(recorder.events) == 16
    assert recorder.events[-1].kind == "transient_completed"
    assert recorder.dropped_events > 0

    restored = [
        SolveEvent("time_increment", "explicit", increment=index)
        for index in range(40)
    ]
    restored_recorder = SolveEventRecorder(restored, max_events=16)
    assert len(restored) == 16
    assert restored[0].increment == 0
    assert restored[-1].increment == 39
    assert restored_recorder.dropped_events == 24
