from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

from tools.p92_recoverable_spectrum import Material as Spectrum
from tools.p92_two_rate_reference import Material, Branch
from tools.p92_spectrum_recovery_joint import decode_history, history_identifiability, decode


def card():
    b = Branch(20., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    return asdict(Spectrum(Material(10000., (b, b)), (20000., 40000.), (.5, 10.)))


def test_substitution_keeps_count_and_freezes_weak_coefficients():
    seed=card(); z=np.zeros(22); z[7]=.1; z[8]=-.1
    candidate=decode_history(seed,z)
    baseline=decode(seed,np.zeros(22),(0.,0.))
    assert candidate.backbone == baseline.backbone
    assert candidate.static_ratios == (1/(1+10**(-.1)),)*2
    assert candidate.static_history_rates == (250*10**(-.1),)*2
    assert candidate.static_history == "shared"
    z[9]=.1
    assert decode_history(seed,z).backbone.branches[1].strength == pytest.approx(20*10**.1)
    assert seed == card()


def test_independent_history_columns_observed_and_confounded_rejected():
    matrix=np.eye(22); basis=matrix[:, :10]
    result=history_identifiability(matrix,basis)
    assert result["passed"]
    matrix[:,8]=matrix[:,7]
    assert not history_identifiability(matrix,basis)["passed"]
    assert not history_identifiability(np.eye(22),np.eye(22)[:,12:22])["passed"]


def test_substitution_rejects_invalid_shape_and_existing_evolution():
    for z in (np.zeros(23),np.full(22,np.nan)):
        with pytest.raises(ValueError):decode_history(card(),z)
    seed=card(); seed['static_ratios']=(.5,.5)
    with pytest.raises(ValueError,match="unevolved"):decode_history(seed,np.zeros(22))


def test_scaled_power_accounting_with_substituted_parameters():
    from tools import p92_recoverable_spectrum as kernel
    m=decode_history(card(),np.zeros(22)); y=np.array([70.,1.,-2.,.01,3.,-1.,.02,.01,0.,0.,-.001,.002])
    v=kernel.rates(y,.003,m); h=1e-5
    energy_rate=(kernel.stored_energy(y+h*v,m)-kernel.stored_energy(y-h*v,m))/(2*h)
    assert v[8] == pytest.approx(energy_rate+v[9],abs=1e-9)
    assert v[9]>=0
    assert replace(m,static_ratios=(1.,1.)).backbone == m.backbone


def test_unresolved_audit_retains_failure_without_optimization(monkeypatch):
    from tools import p92_spectrum_recovery_joint as module
    seed = dict(model_id=module.kernel.MODEL_ID, parameters=card(),
                source_sha256={p: p for p in ("RF", "SLR", "SLC")})
    monkeypatch.setattr(module, "read_source", lambda path, protocol: dict(sha256=protocol))
    def early(source, *, cycles):
        assert cycles == 3
        return source
    monkeypatch.setattr(module, "measured_path", early)
    monkeypatch.setattr(module, "attach_waveforms", lambda path, source: path)
    monkeypatch.setattr(module, "replay", lambda path, mat, **kw:
                        (np.ones(10)*mat.backbone.branches[0].strength, {}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Unresolved audit must not trigger fitting")
    monkeypatch.setattr(module, "least_squares", forbidden)
    result = module.identify(seed, Path("unused"), (0., 0.), shared_history=True)
    assert not result["fit"]["success"]
    assert result["fit"]["residual_evaluations"] == 89
    assert result["training_cycles"] == [1, 2, 3]
    assert not result["engineering_qualified"]


def test_weak_history_is_not_rescued_by_observable_nuisance_directions(monkeypatch):
    from tools import p92_spectrum_recovery_joint as module
    seed = dict(model_id=module.kernel.MODEL_ID, parameters=card(),
                source_sha256={p: p for p in ("RF", "SLR", "SLC")})
    monkeypatch.setattr(module, "read_source", lambda path, protocol: dict(sha256=protocol))
    monkeypatch.setattr(module, "measured_path", lambda source, cycles: source)
    monkeypatch.setattr(module, "attach_waveforms", lambda path, source: path)
    monkeypatch.setattr(module, "decode_history", lambda card, z: z)
    weights = np.arange(22., 0., -1.)
    weights[7:9] = (1e-7, 2e-7)
    monkeypatch.setattr(module, "replay", lambda path, z, **kw: (weights*z, {}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Other observable parameters cannot rescue weak history")
    monkeypatch.setattr(module, "least_squares", forbidden)
    result = module.identify(seed, Path("unused"), (0., 0.), shared_history=True)
    assert not result["history_audit"]["passed"]
    assert not result["fit"]["success"]
    assert "history directions" in result["fit"]["message"]


def test_archived_matched_cards_keep_budget_identity_and_failed_gates():
    root = Path(__file__).resolve().parents[1]/"evidence"/"creep"
    for name in ("baseline", "shared"):
        stem = f"2026-11-21-static-memory-matched-{name}"
        fit = json.loads((root/f"{stem}.json").read_text())
        frozen = json.loads((root/f"{stem}-frozen.json").read_text())
        assert len(fit["coordinates"]) == 22
        assert len(fit["fit"]["coordinates"]) == 10
        assert fit["maximum_log_coordinate_norm"] == .3
        assert fit["training_cycles"] == [1, 2, 3]
        assert fit["fit"]["nfev"] <= 40
        assert frozen["fit_file_sha256"] == fit["archived_source_sha256"]
        assert frozen["parameters"] == fit["fit"]["parameters"]
        assert frozen["source_sha256"] == fit["source_sha256"]
        assert not frozen["engineering_qualified"] and not frozen["life_qualified"]
        for result in frozen["protocols"].values():
            assert result["numerical"]["passed"]
            assert result["response_screen"]["status"] == "failed"


@pytest.mark.parametrize("weak", [False, True])
def test_replacement_space_keeps_budget_and_cannot_rescue_unresolved_axes(monkeypatch, weak):
    from dataclasses import dataclass
    from types import SimpleNamespace
    from tools import p92_spectrum_recovery_joint as module

    @dataclass
    class Point:
        values: object

    seed = dict(model_id=module.kernel.MODEL_ID, parameters=card(),
                source_sha256={p: p for p in ("RF", "SLR", "SLC")})
    monkeypatch.setattr(module, "read_source", lambda path, protocol: dict(sha256=protocol))
    monkeypatch.setattr(module, "measured_path", lambda source, cycles: source)
    monkeypatch.setattr(module, "attach_waveforms", lambda path, source: path)
    monkeypatch.setattr(module, "decode_history", lambda card, z: Point(z.copy()))
    weights = np.arange(22., 0., -1.)
    weights[7:9] = (1e-7, 2e-7) if weak else (.1, .2)
    monkeypatch.setattr(module, "replay", lambda path, point, **kw:
                        (weights*point.values+.01, {}))
    calls = []

    def optimizer(fun, initial, **kwargs):
        assert not weak, "An unresolved replacement must not enter optimization"
        calls.append(True)
        assert initial.shape == (10,)
        assert kwargs["bounds"] == pytest.approx((-.3/np.sqrt(10), .3/np.sqrt(10)))
        assert kwargs["max_nfev"] == 40
        assert kwargs["ftol"] == kwargs["xtol"] == kwargs["gtol"] == 1e-5
        fun(initial)
        return SimpleNamespace(x=initial, success=True, message="synthetic", nfev=1,
                               active_mask=np.zeros(10, dtype=int))

    monkeypatch.setattr(module, "least_squares", optimizer)
    result = module.identify(seed, Path("unused"), (0., 0.), shared_history=True,
                             replacement_space=True)
    assert result["replacement_space"]
    assert result["training_cycles"] == [1, 2, 3]
    assert result["fit"]["success"] is (not weak)
    assert bool(calls) is (not weak)
    if not weak:
        basis = np.asarray(result["basis"])
        np.testing.assert_allclose(basis[[7, 8]]@basis[[7, 8]].T, np.eye(2), atol=1e-12)
        assert result["maximum_log_coordinate_norm"] == .3
        assert len(result["coordinates"]) == 22


def test_replacement_geometry_is_explicit_and_isolates_shared_history():
    from tools.p92_spectrum_recovery_joint import identify
    seed = dict(model_id="research.scalar-two-rate-recoverable-spectrum.v1")
    for kwargs in (dict(replacement_space=1), dict(replacement_space=True),
                   dict(replacement_space=True, recoverable_kinetics=True)):
        with pytest.raises(ValueError, match="replacement"):
            identify(seed, Path("unused"), (0., 0.), **kwargs)
