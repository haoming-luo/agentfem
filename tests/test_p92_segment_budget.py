from copy import deepcopy

import numpy as np
import pytest

from tools.p92_segment_budget import segment_budget


def fixture():
    source = dict(sha256="same", time_s=[0., 1., 2.], cycle=[1, 1, 1],
                  stress_Pa=np.array([0., 10., 20.])*1e6, strain=[0., .01, .02])
    prediction = dict(source_sha256="same", segments=[dict(segment=0, cycle=1,
        stress_control=False, source_indices=[0, 1, 2],
        sampled_stress_MPa=[0., 12., 24.], sampled_strain=[0., .011, .022])])
    return source, prediction


def test_increment_identity_keeps_input_error_separate():
    source, prediction = fixture()
    before = deepcopy(prediction)
    row = segment_budget(source, prediction, 1000.)["records"][0]
    assert row["interval_input_term_MPa"] == pytest.approx(2.)
    assert row["interval_nonelastic_term_MPa"] == pytest.approx(2.)
    assert row["interval_stress_bias_change_MPa"] == pytest.approx(4.)
    assert row["algebraic_closure_MPa"] < 1e-12
    assert prediction == before


@pytest.mark.parametrize("ids", [[0, 0], [2, 1], [-1, 1], [0., 1.], [0, 3], [1]])
def test_reject_ambiguous_observation_identity(ids):
    source, prediction = fixture()
    prediction["segments"][0]["source_indices"] = ids
    with pytest.raises(ValueError, match="identities"):
        segment_budget(source, prediction, 1000.)


def test_no_interpolation_for_missing_prediction():
    source, prediction = fixture()
    prediction["segments"][0]["sampled_stress_MPa"] = [0., 24.]
    with pytest.raises(ValueError, match="aligned"):
        segment_budget(source, prediction, 1000.)


def test_reject_changed_source():
    source, prediction = fixture()
    source["sha256"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        segment_budget(source, prediction, 1000.)


def test_switch_bracket_excludes_ambiguous_samples_without_interpolation():
    source, prediction = fixture()
    path = dict(source_sha256="same", switches=[dict(bracket_s=[2., 2.1])])
    row = segment_budget(source, prediction, 1000., path=path)["records"][0]
    assert row["source_indices"] == [1, 2]
    assert row["excluded_switch_indices"] == [0]
    path["switches"][0]["bracket_s"] = [1.5, 2.1]
    with pytest.raises(ValueError, match="ambiguity"):
        segment_budget(source, prediction, 1000., path=path)


def test_slope_comparison_separates_predicted_and_recorded_strain():
    from tools.p92_reversal_slope_comparison import compare
    source, prediction = fixture()
    observed = dict(source_sha256="same", records=[dict(segment=0, cycle=1,
        source_indices=[0, 1, 2], apparent_slope_MPa=1000.)])
    row = compare(source, prediction, observed)["records"][0]
    assert row["predicted_slope_MPa"] == pytest.approx(24/.022)
    assert row["predicted_on_recorded_strain_slope_MPa"] == pytest.approx(1200.)
    assert row["apparent_slope_MPa"] == 1000.
    prediction["segments"][0]["source_indices"] = [0, 1, 3]
    with pytest.raises(ValueError, match="Missing"):
        compare(source, prediction, observed)


def test_slope_comparison_rejects_changed_source_and_duplicate_samples():
    from tools.p92_reversal_slope_comparison import compare
    source, prediction = fixture()
    observed = dict(source_sha256="wrong", records=[])
    with pytest.raises(ValueError, match="identity"):
        compare(source, prediction, observed)
    observed = dict(source_sha256="same", records=[dict(segment=0, source_indices=[0, 1, 2])])
    prediction["segments"][0]["source_indices"] = [0, 0, 2]
    with pytest.raises(ValueError, match="Duplicate"):
        compare(source, prediction, observed)
