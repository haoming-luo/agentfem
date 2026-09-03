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
    assert report.summary()["schema"] == ("agentfem.mixed-mode-bending-comparison.v1")


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


def test_external_curve_identity_includes_units_and_rejects_unit_mismatch():
    units = {
        "crack_length": "mm",
        "load": "N",
        "displacement": "mm",
        "mode_i_fraction": "1",
    }
    reference = benchmarks.MixedModeBendingCurve.create(
        crack_length=(20.0, 25.0),
        load=(100.0, 90.0),
        displacement=(0.4, 0.5),
        mode_i_fraction=(0.2, 0.3),
        source="NASA/CR-2012-217562 unit fixture",
        units=units,
    )
    incompatible = benchmarks.MixedModeBendingCurve.create(
        crack_length=(0.02, 0.025),
        load=(100.0, 90.0),
        displacement=(0.0004, 0.0005),
        mode_i_fraction=(0.2, 0.3),
        source="converted solver fixture",
        units={**units, "crack_length": "m", "displacement": "m"},
    )

    assert reference.units_complete
    assert reference.summary()["units"] == units
    with pytest.raises(ValueError, match="identical units"):
        benchmarks.compare_mixed_mode_bending_curves(
            reference,
            incompatible,
            load_relative_tolerance=1.0,
            displacement_relative_tolerance=1.0,
            mode_i_fraction_absolute_tolerance=1.0,
        )


@pytest.mark.parametrize("kind", ("dcb", "enf"))
def test_delamination_beam_oracles_recover_closed_form_energy_release(kind):
    geometry = {
        "width": 20.0,
        "arm_thickness": 2.0,
        "elastic_modulus": 70_000.0,
        "source": "classical beam-theory verification fixture",
    }
    if kind == "enf":
        geometry["half_span"] = 50.0
    spec = benchmarks.delamination_benchmark_spec(kind, **geometry)
    crack = np.linspace(10.0, 30.0, 101)
    load = np.full_like(crack, 100.0)
    result = benchmarks.beam_theory_energy_release_curve(
        spec, crack_length=crack, load=load
    )
    if kind == "dcb":
        expected = (
            12.0
            * load**2
            * crack**2
            / (spec.elastic_modulus * spec.width**2 * spec.arm_thickness**3)
        )
        np.testing.assert_allclose(result.mode_ii_energy_release_rate, 0.0)
    else:
        expected = (
            9.0
            * load**2
            * crack**2
            / (16.0 * spec.elastic_modulus * spec.width**2 * spec.arm_thickness**3)
        )
        np.testing.assert_allclose(result.mode_i_energy_release_rate, 0.0)
    np.testing.assert_allclose(result.total_energy_release_rate, expected, rtol=3.0e-4)
    assert spec.standard_family.startswith("ASTM")


def test_mmb_requires_declared_partition_and_assessment_checks_guardrails():
    spec = benchmarks.delamination_benchmark_spec(
        "mmb",
        width=20.0,
        arm_thickness=2.0,
        elastic_modulus=70_000.0,
        half_span=50.0,
        source="source-identified MMB compliance fixture",
    )
    crack = np.linspace(10.0, 30.0, 11)
    load = np.full_like(crack, 100.0)
    compliance = 1.0e-4 + 2.0e-8 * crack**3
    with pytest.raises(ValueError, match="mode_i_fraction"):
        benchmarks.compliance_energy_release_curve(
            spec, crack_length=crack, load=load, compliance=compliance
        )
    reference = benchmarks.compliance_energy_release_curve(
        spec,
        crack_length=crack,
        load=load,
        compliance=compliance,
        mode_i_fraction=np.linspace(0.2, 0.6, crack.size),
    )
    np.testing.assert_allclose(
        reference.mode_i_energy_release_rate + reference.mode_ii_energy_release_rate,
        reference.total_energy_release_rate,
    )
    accepted = benchmarks.assess_delamination_benchmark(
        spec,
        reference,
        reference,
        energy_release_relative_tolerance=0.01,
        minimum_process_zone_elements=4.0,
        artificial_dissipation=2.0,
        internal_energy=100.0,
    )
    rejected = benchmarks.assess_delamination_benchmark(
        spec,
        reference,
        reference,
        energy_release_relative_tolerance=0.01,
        minimum_process_zone_elements=2.0,
        artificial_dissipation=10.0,
        internal_energy=100.0,
    )
    assert accepted.accepted
    assert not rejected.accepted


def test_delamination_energy_curve_rejects_malformed_or_inconsistent_channels():
    common = {
        "crack_length": (10.0, 20.0, 30.0),
        "compliance": (1.0, 2.0, 3.0),
        "total_energy_release_rate": (2.0, 3.0, 4.0),
        "mode_i_energy_release_rate": (1.0, 1.5, 2.0),
        "mode_ii_energy_release_rate": (1.0, 1.5, 2.0),
        "source": "traceable fixture",
    }
    valid = benchmarks.DelaminationEnergyReleaseCurve(**common)
    assert not valid.crack_length.flags.writeable
    assert len(valid.identity_sha256) == 64
    assert valid.summary()["identity_sha256"] == valid.identity_sha256
    with pytest.raises(ValueError, match="sum to total G"):
        benchmarks.DelaminationEnergyReleaseCurve(
            **{**common, "mode_ii_energy_release_rate": (0.0, 0.0, 0.0)}
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        benchmarks.DelaminationEnergyReleaseCurve(
            **{**common, "crack_length": (10.0, 10.0, 30.0)}
        )
    with pytest.raises(ValueError, match="identify its source"):
        benchmarks.DelaminationEnergyReleaseCurve(**{**common, "source": " "})


@pytest.mark.parametrize("kind", ("dcb", "enf", "mmb"))
def test_delamination_convergence_certificate_needs_three_improving_levels(kind):
    geometry = {
        "width": 20.0,
        "arm_thickness": 2.0,
        "elastic_modulus": 70_000.0,
        "source": "NASA independent delamination benchmark family",
    }
    if kind in {"enf", "mmb"}:
        geometry["half_span"] = 50.0
    spec = benchmarks.delamination_benchmark_spec(kind, **geometry)
    crack = np.linspace(10.0, 30.0, 41)
    load = np.full_like(crack, 100.0)
    if kind == "mmb":
        compliance = 1.0e-4 + 2.0e-8 * crack**3
        reference = benchmarks.compliance_energy_release_curve(
            spec,
            crack_length=crack,
            load=load,
            compliance=compliance,
            mode_i_fraction=np.linspace(0.2, 0.6, crack.size),
        )
    else:
        reference = benchmarks.beam_theory_energy_release_curve(
            spec,
            crack_length=crack,
            load=load,
        )

    def biased(factor):
        return benchmarks.DelaminationEnergyReleaseCurve(
            crack_length=reference.crack_length.copy(),
            compliance=reference.compliance.copy(),
            total_energy_release_rate=(
                reference.total_energy_release_rate * (1.0 + factor)
            ),
            mode_i_energy_release_rate=(
                reference.mode_i_energy_release_rate * (1.0 + factor)
            ),
            mode_ii_energy_release_rate=(
                reference.mode_ii_energy_release_rate * (1.0 + factor)
            ),
            source=f"computed {kind} level {factor}",
        )

    certificate = benchmarks.certify_delamination_convergence(
        spec,
        (biased(0.08), biased(0.02), biased(0.005)),
        reference,
        element_sizes=(1.0, 0.5, 0.25),
        process_zone_elements=(4.0, 8.0, 16.0),
        artificial_dissipation_fractions=(0.02, 0.01, 0.005),
        reference_relative_tolerance=0.01,
        refinement_relative_tolerance=0.02,
    )

    assert certificate.accepted
    assert certificate.asymptotic_trend
    assert certificate.observed_order == pytest.approx(2.0, rel=0.2)
    assert certificate.mode_i_fraction_maximum_errors[-1] < 1.0e-12
    assert certificate.reference_identity_sha256 == reference.identity_sha256
    assert certificate.curve_identity_sha256 == tuple(
        item.identity_sha256
        for item in (biased(0.08), biased(0.02), biased(0.005))
    )
    assert certificate.summary()["schema"] == (
        "agentfem.delamination-convergence-certificate.v1"
    )

    unresolved = benchmarks.certify_delamination_convergence(
        spec,
        (biased(0.08), biased(0.02), biased(0.005)),
        reference,
        element_sizes=(1.0, 0.5, 0.25),
        process_zone_elements=(2.0, 4.0, 8.0),
        artificial_dissipation_fractions=(0.02, 0.01, 0.005),
        reference_relative_tolerance=0.01,
        refinement_relative_tolerance=0.02,
    )
    assert not unresolved.accepted


def test_delamination_certificate_rejects_mode_partition_and_bad_level_order():
    spec = benchmarks.delamination_benchmark_spec(
        "mmb",
        width=20.0,
        arm_thickness=2.0,
        elastic_modulus=70_000.0,
        half_span=50.0,
        source="source-identified MMB fixture",
    )
    crack = np.linspace(10.0, 30.0, 21)
    load = np.full_like(crack, 100.0)
    reference = benchmarks.compliance_energy_release_curve(
        spec,
        crack_length=crack,
        load=load,
        compliance=1.0e-4 + 2.0e-8 * crack**3,
        mode_i_fraction=np.full_like(crack, 0.4),
    )

    def wrong_partition(total_factor, fraction):
        total = reference.total_energy_release_rate * total_factor
        return benchmarks.DelaminationEnergyReleaseCurve(
            crack_length=crack,
            compliance=reference.compliance,
            total_energy_release_rate=total,
            mode_i_energy_release_rate=total * fraction,
            mode_ii_energy_release_rate=total * (1.0 - fraction),
            source="computed structural level",
        )

    certificate = benchmarks.certify_delamination_convergence(
        spec,
        (
            wrong_partition(1.08, 0.55),
            wrong_partition(1.02, 0.55),
            wrong_partition(1.005, 0.55),
        ),
        reference,
        element_sizes=(1.0, 0.5, 0.25),
        process_zone_elements=(4.0, 8.0, 16.0),
        artificial_dissipation_fractions=(0.02, 0.01, 0.005),
        reference_relative_tolerance=0.01,
        refinement_relative_tolerance=0.02,
        mode_partition_absolute_tolerance=0.02,
    )
    assert not certificate.accepted
    assert certificate.mode_i_fraction_maximum_errors[-1] == pytest.approx(0.15)

    with pytest.raises(ValueError, match="decrease from coarse to fine"):
        benchmarks.certify_delamination_convergence(
            spec,
            (reference, reference, reference),
            reference,
            element_sizes=(1.0, 0.5, 0.75),
            process_zone_elements=(4.0, 8.0, 16.0),
            artificial_dissipation_fractions=(0.0, 0.0, 0.0),
            reference_relative_tolerance=0.01,
            refinement_relative_tolerance=0.02,
        )
