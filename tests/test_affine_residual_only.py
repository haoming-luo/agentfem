"""Optional residual-only Operators must not change Newton or failure semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from agentfem import solvers


class _Vector:
    def __init__(self):
        self.events = []

    def ghostUpdate(self, **kwargs):
        self.events.append(("ghost", kwargs))

    def destroy(self):
        self.events.append("destroy")


@pytest.mark.parametrize("require_tangent", [False, True])
def test_optional_residual_hook_preserves_legacy_operators_and_newton(monkeypatch, require_tangent):
    vector = _Vector()
    monkeypatch.setattr(solvers.fem_petsc, "assemble_vector", lambda form: vector)
    events = []
    adapted = SimpleNamespace(
        evaluate=lambda: events.append("full") or "tangent",
        add_residual=lambda v, c: events.append(("add", v, c)),
        add_current_residual=lambda v: events.append(("current", v)),
    )
    legacy = SimpleNamespace(
        evaluate=lambda: events.append("legacy") or "legacy tangent",
        add_residual=lambda v, c: events.append(("legacy add", v, c)),
    )
    result, contributions = solvers._assemble_affine_full_residual(
        object(), (adapted, legacy), require_tangent=require_tangent,
    )
    assert result is vector
    if require_tangent:
        assert events == ["full", ("add", vector, "tangent"),
                          "legacy", ("legacy add", vector, "legacy tangent")]
        assert contributions == [(adapted, "tangent"), (legacy, "legacy tangent")]
    else:
        assert events == [("current", vector), "legacy", ("legacy add", vector, "legacy tangent")]
        assert contributions == []
    assert vector.events == [("ghost", dict(
        addv=solvers.PETSc.InsertMode.ADD, mode=solvers.PETSc.ScatterMode.REVERSE,
    ))]


@pytest.mark.parametrize("failure", ["current", "ghost"])
def test_residual_only_failure_destroys_vector_without_silent_fallback(monkeypatch, failure):
    vector = _Vector()
    monkeypatch.setattr(solvers.fem_petsc, "assemble_vector", lambda form: vector)

    def fail(*args, **kwargs):
        raise ValueError("invalid current state")

    def forbidden():
        raise AssertionError("A failed current residual must not silently fall back")

    operator = SimpleNamespace(evaluate=forbidden, add_current_residual=fail)
    if failure == "ghost":
        operator.add_current_residual = lambda v: None
        vector.ghostUpdate = fail
    with pytest.raises(ValueError, match="invalid current state"):
        solvers._assemble_affine_full_residual(object(), (operator,), require_tangent=False)
    assert vector.events == ["destroy"]


def test_derived_contact_law_does_not_inherit_penalty_residual_shortcut():
    from agentfem.boundary_models.dolfinx_implicit_facet_contact import DolfinxImplicitRigidContact

    events = []

    class DerivedLaw(DolfinxImplicitRigidContact):
        def evaluate(self):
            events.append("derived law")
            return "derived contribution"

        def add_residual(self, vector, contribution):
            events.append((vector, contribution))

        def _current_point_response(self, **kwargs):
            raise AssertionError("Penalty observation cannot represent a derived law")

    operator = object.__new__(DerivedLaw)
    vector = _Vector()
    operator.add_current_residual(vector)
    assert events == ["derived law", (vector, "derived contribution")]


@pytest.mark.parametrize("factor,expected_calls", [
    (0., 1), (.1, 1), (.25, 2), (.5, 1), (.75, 2), (1., 1),
    (np.nextafter(.25, 0.), 1), (np.nextafter(.25, 1.), 1),
])
def test_affine_dual_pair_reuses_only_identical_path_directions(monkeypatch, factor, expected_calls):
    from agentfem.constraints.affine import AbaqusPeriodicConstraint

    constraint = object.__new__(AbaqusPeriodicConstraint)
    object.__setattr__(constraint, "deformation_gradient_path",
                       SimpleNamespace(coordinates=(0., .25, .75, 1.)))
    calls = []

    def evaluate(self, problem, *, load_factor, path_side="left"):
        calls.append((problem, load_factor, path_side))
        return object()

    monkeypatch.setattr(AbaqusPeriodicConstraint, "dual_evidence", evaluate)
    problem = object()
    a, b = constraint._accepted_dual_evidence_pair(problem, load_factor=factor)
    assert len(calls) == expected_calls
    assert (a is b) == (expected_calls == 1)
    assert calls[0] == (problem, factor, "left")
    if expected_calls == 2:
        assert calls[1] == (problem, factor, "right")


def test_derived_affine_constraint_retains_both_its_own_duals():
    from agentfem.constraints.affine import AbaqusPeriodicConstraint

    class DerivedConstraint(AbaqusPeriodicConstraint):
        def dual_evidence(self, problem, *, load_factor, path_side="left"):
            return path_side

    constraint = object.__new__(DerivedConstraint)
    object.__setattr__(constraint, "deformation_gradient_path", None)
    assert constraint._accepted_dual_evidence_pair(object(), load_factor=.5) == ("left", "right")


def test_legacy_constraint_dual_provider_still_receives_both_directions():
    from agentfem._nonlinear_problems import AffineNonlinearVariationalProblem

    calls, records = [], []
    def evaluate(problem, *, load_factor, path_side="left"):
        calls.append(path_side)
        return path_side
    problem = SimpleNamespace(
        constraint=SimpleNamespace(dual_evidence=evaluate),
        constraint_dual_history=SimpleNamespace(
            append=lambda factor, evidence, **kw: records.append((factor, evidence, kw))),
    )
    AffineNonlinearVariationalProblem._capture_constraint_dual(problem, .5)
    assert calls == ["left", "right"]
    assert records == [(.5, "left", {"outgoing_evidence": "right"})]
