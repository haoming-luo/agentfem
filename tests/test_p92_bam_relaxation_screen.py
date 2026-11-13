import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tools.p92_bam_relaxation_screen import relaxation, screen


@pytest.mark.parametrize("exponent", [1., 1.+1e-10, 5., 20.])
def test_analytic_relaxation_against_independent_ode(exponent):
    t = np.linspace(0, 180, 101)
    ode = solve_ivp(lambda _, y: [-2*(y[0]/200)**exponent], [0, 180], [200.],
                    t_eval=t, method="DOP853", rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(relaxation(t, 200, 2, exponent), ode.y[0], rtol=2e-10)


def test_holdout_cannot_change_fit():
    t = np.linspace(5, 182, 400)
    data = {"protocol": "RF", "time_s": t, "cycle": np.ones(len(t)),
            "stress_Pa": relaxation(t-5, 200, 2, 5)*1e6, "sha256": "synthetic"}
    original = screen(data)
    changed = dict(data, stress_Pa=data["stress_Pa"].copy())
    changed["stress_Pa"][t > 60] *= 1.5
    altered = screen(changed)
    assert original["exponent"] == altered["exponent"]
    assert original["initial_rate_MPa_per_s"] == altered["initial_rate_MPa_per_s"]
    assert original["exponent"] == pytest.approx(5, rel=1e-6)
    assert altered["reports"]["within_path_holdout"]["rmse_MPa"] > 1


def test_later_hold_is_prediction_not_refit():
    t = np.linspace(0, 182, 400)
    first = relaxation(np.maximum(t-5, 0), 200, 2, 5)
    later = relaxation(np.maximum(t-5, 0), 150, 2*(150/200)**5, 5)
    data = {"protocol": "RF", "time_s": np.r_[t, t+368*9],
            "cycle": np.r_[np.ones(len(t)), np.full(len(t), 10)],
            "stress_Pa": np.r_[first, later]*1e6, "sha256": "synthetic"}
    original = screen(data)
    changed = dict(data, stress_Pa=data["stress_Pa"].copy())
    changed["stress_Pa"][len(t)+20:] *= 1.1
    altered = screen(changed)
    assert original["exponent"] == altered["exponent"]
    assert original["initial_rate_MPa_per_s"] == altered["initial_rate_MPa_per_s"]
    assert original["later_conditional_holds"][0]["rmse_MPa"] < 0.02
    assert altered["later_conditional_holds"][0]["rmse_MPa"] > 1
    assert original["later_conditional_holds"][1] == {"cycle": 100, "available": False}
    repeated = {key: np.insert(value, len(t)+30, value[len(t)+30])
                if isinstance(value, np.ndarray) else value for key, value in data.items()}
    assert screen(repeated)["exponent"] == original["exponent"]


@pytest.mark.parametrize("key,value", [("time_s", [1, 0]), ("stress_Pa", [1, np.nan]),
                                       ("cycle", [1, 1.5]), ("stress_Pa", [1])])
def test_malformed_histories_rejected(key, value):
    data = {"protocol": "RF", "time_s": [0, 1], "stress_Pa": [1, 2], "cycle": [1, 1]}
    data[key] = value
    with pytest.raises(ValueError, match="aligned finite"):
        screen(data)
