import numpy as np
import pytest

from agentfem import benchmarks


def _curve(*, scale=1.0, source="synthetic contract fixture"):
    crack_length = np.asarray((20.0, 25.0, 30.0, 35.0))
    return benchmarks.MixedModeBendingCurve.create(
        crack_length=crack_length,
        load=scale * np.asarray((100.0, 94.0, 88.0, 82.0)),
        displacement=scale * np.asarray((0.4, 0.5, 0.62, 0.76)),
        mode_i_fraction=np.asarray((0.2, 0.3, 0.4, 0.5)),
        source=source,
    )


def test_external_mixed_mode_curve_contract_reads_and_compares(tmp_path):
    csv = tmp_path / "mmb.csv"
    csv.write_text(
        "crack_length,load,displacement,mode_i_fraction\n"
        "20,100,0.4,0.2\n25,94,0.5,0.3\n"
        "30,88,0.62,0.4\n35,82,0.76,0.5\n",
        encoding="utf-8",
    )
    reference = benchmarks.MixedModeBendingCurve.read_csv(
        csv,
        source="NASA mixed-mode bending reference fixture",
    )
    predicted = _curve()
    report = benchmarks.compare_mixed_mode_bending_curves(
        reference,
        predicted,
        load_relative_tolerance=1.0e-12,
        displacement_relative_tolerance=1.0e-12,
        mode_i_fraction_absolute_tolerance=1.0e-12,
    )

    assert report.accepted
    assert len(reference.identity_sha256) == 64
    assert report.summary()["schema"] == (
        "agentfem.mixed-mode-bending-comparison.v1"
    )


def test_external_mixed_mode_comparison_rejects_bad_curve_and_range():
    reference = _curve()
    report = benchmarks.compare_mixed_mode_bending_curves(
        reference,
        _curve(scale=1.2, source="biased solver fixture"),
        load_relative_tolerance=0.05,
        displacement_relative_tolerance=0.05,
        mode_i_fraction_absolute_tolerance=0.01,
    )
    assert not report.accepted

    short = benchmarks.MixedModeBendingCurve.create(
        crack_length=(21.0, 30.0, 34.0),
        load=(1.0, 1.0, 1.0),
        displacement=(1.0, 1.0, 1.0),
        mode_i_fraction=(0.2, 0.3, 0.4),
        source="short fixture",
    )
    with pytest.raises(ValueError, match="must cover"):
        benchmarks.compare_mixed_mode_bending_curves(
            reference,
            short,
            load_relative_tolerance=0.1,
            displacement_relative_tolerance=0.1,
            mode_i_fraction_absolute_tolerance=0.1,
        )
