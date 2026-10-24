# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""PETSc object ownership survives setup, solve and interruption failures."""
from types import SimpleNamespace

import pytest

from agentfem import solvers


@pytest.mark.parametrize('borrowed', [False, True])
@pytest.mark.parametrize('phase', ['setOperators', 'solve', 'getConvergedReason'])
@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
def test_linear_failure_releases_only_owned_ksp(monkeypatch, borrowed, phase, error):
    destroyed = []
    def fail(*args):
        raise error('injected linear failure')
    solver = SimpleNamespace(
        setOperators=lambda *a: None, solve=lambda *a: None,
        getConvergedReason=lambda: 2, getIterationNumber=lambda: 1,
        getResidualNorm=lambda: 0., destroy=lambda: destroyed.append(True),
    )
    setattr(solver, phase, fail)
    monkeypatch.setattr(solvers, 'create_ksp', lambda *args: solver)
    with pytest.raises(error, match='injected linear failure'):
        solvers.solve_matrix_system(SimpleNamespace(comm=None), None, None,
                                    ksp=solver if borrowed else None)
    assert destroyed == ([] if borrowed else [True])


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
def test_configure_failure_releases_new_ksp(monkeypatch, error):
    destroyed = []
    def fail(*args):
        raise error('injected setup failure')
    solver = SimpleNamespace(setType=fail, destroy=lambda: destroyed.append(True))
    factory = SimpleNamespace(create=lambda comm: solver)
    monkeypatch.setattr(solvers, 'PETSc', SimpleNamespace(KSP=lambda: factory))
    with pytest.raises(error, match='injected setup failure'):
        solvers.create_ksp(None)
    assert destroyed == [True]
