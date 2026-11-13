"""Independent state/work checks for the public cohesive path utility."""
from types import SimpleNamespace

import pytest

from agentfem.cohesive_path import MonotonicCohesivePath


class Equilibrium:
    def __init__(self):
        self.last_info = None
        self.failure = None

    def __call__(self, *, load, branch, cycle):
        assert (branch, cycle) == ("monotonic", 0)
        self.last_info = SimpleNamespace(converged=self.failure is None)
        if self.failure == "exception":
            raise RuntimeError("nonconvergence")
        return dict(converged=self.failure is None, reaction=-2 * load,
                    bulk_strain_energy=load ** 2)


class Cohesive:
    def __init__(self):
        self.dissipation = 10.0
        self.commits = 0

    def begin(self):
        return SimpleNamespace(stored_energy=0.0, dissipated_energy=self.dissipation)

    def commit(self):
        self.commits += 1


def test_work_and_dissipation_advance_only_on_accepted_states():
    eq, state = Equilibrium(), Cohesive()
    path = MonotonicCohesivePath(eq, state, state, observe=lambda _: {"tip": 2.0})
    assert path.try_accept(0)["dissipated"] == 0
    first = path.try_accept(1)
    assert first["tip"] == 2
    assert path.external_work == 1
    for failure in ("result", "exception"):
        eq.failure = failure
        assert path.try_accept(2) is None
        assert (path.coordinate, path.external_work, state.commits) == (1, 1, 2)
    eq.failure = None
    state.dissipation = 10.25
    last = path.try_accept(2)
    assert last["dissipated"] == 0.25
    assert (path.external_work, path.coordinate, state.commits) == (4, 2, 3)
    with pytest.raises(ValueError, match="nondecreasing"):
        path.try_accept(1)


def test_observation_failure_does_not_commit_or_update_ledger():
    eq, state = Equilibrium(), Cohesive()
    path = MonotonicCohesivePath(eq, state, state, observe=lambda _: {"reaction": 123})
    with pytest.raises(ValueError, match="overwrite"):
        path.try_accept(1)
    assert (path.coordinate, path.external_work, state.commits) == (0, 0, 0)


def test_programming_errors_are_not_misclassified_as_cutbacks():
    class Broken:
        last_info = None

        def __call__(self, **kwargs):
            raise RuntimeError("invalid assembly")

    state = Cohesive()
    path = MonotonicCohesivePath(Broken(), state, state, observe=lambda _: {})
    with pytest.raises(RuntimeError, match="invalid assembly"):
        path.try_accept(1)
