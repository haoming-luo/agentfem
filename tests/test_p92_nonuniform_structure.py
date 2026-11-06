import numpy as np
import pytest
from mpi4py import MPI

from tools.p92_nonuniform_structure import build, observation, run

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.size != 1, reason="Serial research composition")


def test_nonuniform_hold_and_restart(tmp_path):
    report = run(checkpoint_dir=tmp_path / "checkpoint")
    assert report["restart_exact"]
    assert report["accepted_increments"] == 20
    assert report["loaded"]["time_s"] == 4
    assert report["held"]["time_s"] == 64
    assert 0 < report["held"]["tip_reaction_N"] < report["loaded"]["tip_reaction_N"]
    assert report["held"]["maximum_accumulated_inelastic_strain"] > 0
    for state in (report["loaded"], report["held"]):
        assert state["relative_force_balance"] < 1e-7
        assert state["accepted_equilibrium_residual_norm"] < 1e-6
        low, high = state["sampled_axial_stress_range_MPa"]
        assert high > 100 and low < -100
        assert low == pytest.approx(-high, rel=1e-7)
        balance = np.array(state["support_reaction_N"]) + [0, state["tip_reaction_N"], 0]
        assert state["constraint_force_balance_N"] == pytest.approx(balance)
    assert not report["engineering_qualified"]


@pytest.mark.parametrize("options", [{"resolution": 0}, {"resolution": True},
                                     {"refinement": -1}, {"refinement": 1.5}])
def test_invalid_discretizations_fail_before_mesh(options):
    with pytest.raises(ValueError, match="positive integers"):
        build(**options)


def test_internal_force_cancellation_cannot_hide_constraint_imbalance(monkeypatch):
    from types import SimpleNamespace as NS
    from tools import p92_nonuniform_structure as screen

    def reaction(step, *, on=None, component=None):
        return {None: np.zeros(3), "left": np.array([0., -9., 0.]), "right": 10.}[on]

    monkeypatch.setattr(screen.results, "reaction_resultant", reaction)
    step = NS(time=4., response=NS(cauchy_stress=NS(values=np.zeros((1, 3, 3)))),
              state=NS(committed={"P": NS(values=np.zeros(1))}),
              accepted_increments=[NS(residual_norm=1.)])
    report = observation(step, "left", "right")
    assert report["global_force_residual_N"] == [0., 0., 0.]
    assert report["constraint_force_balance_N"] == [0., 1., 0.]
    assert report["relative_force_balance"] == pytest.approx(.1)
    assert report["accepted_equilibrium_residual_norm"] == 1.
