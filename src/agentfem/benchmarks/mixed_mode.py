"""External-data contracts for mixed-mode cohesive structure benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


_DELAMINATION_STANDARDS = {
    "dcb": "ASTM D5528/D5528M (Mode I DCB family)",
    "enf": "ASTM D7905/D7905M (Mode II ENF family)",
    "mmb": "ASTM D6671/D6671M (mixed-mode MMB family)",
}


def _curve_values(values, *, name: str) -> np.ndarray:
    selected = np.asarray(values, dtype=float).reshape(-1)
    if selected.size < 2 or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must contain at least two finite values.")
    return selected


@dataclass(frozen=True)
class MixedModeBendingCurve:
    """One traceable load/displacement/mode-mix curve versus crack length."""

    crack_length: np.ndarray
    load: np.ndarray
    displacement: np.ndarray
    mode_i_fraction: np.ndarray
    source: str
    identity_sha256: str
    units: dict[str, str | None]

    @property
    def units_complete(self) -> bool:
        return all(self.units[name] for name in self.units)

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.mixed-mode-bending-curve.v1",
            "source": self.source,
            "identity_sha256": self.identity_sha256,
            "points": int(self.crack_length.size),
            "units": dict(self.units),
            "units_complete": self.units_complete,
        }

    @classmethod
    def create(
        cls,
        *,
        crack_length,
        load,
        displacement,
        mode_i_fraction,
        source: str,
        units: dict[str, str | None] | None = None,
    ) -> "MixedModeBendingCurve":
        arrays = {
            "crack_length": _curve_values(crack_length, name="crack_length"),
            "load": _curve_values(load, name="load"),
            "displacement": _curve_values(displacement, name="displacement"),
            "mode_i_fraction": _curve_values(mode_i_fraction, name="mode_i_fraction"),
        }
        sizes = {value.size for value in arrays.values()}
        if len(sizes) != 1:
            raise ValueError("Mixed-mode curve columns must have equal length.")
        if np.any(np.diff(arrays["crack_length"]) <= 0.0):
            raise ValueError("crack_length must be strictly increasing.")
        if np.any(
            (arrays["mode_i_fraction"] < 0.0) | (arrays["mode_i_fraction"] > 1.0)
        ):
            raise ValueError("mode_i_fraction must remain within [0, 1].")
        selected_source = str(source).strip()
        if not selected_source:
            raise ValueError("A mixed-mode reference must declare its source.")
        expected_units = {
            "crack_length",
            "load",
            "displacement",
            "mode_i_fraction",
        }
        selected_units = (
            {name: None for name in expected_units}
            if units is None
            else {str(name): (None if value is None else str(value).strip()) for name, value in units.items()}
        )
        if set(selected_units) != expected_units:
            raise ValueError(
                "Mixed-mode units must identify exactly "
                f"{sorted(expected_units)}."
            )
        if selected_units["mode_i_fraction"] not in {None, "1"}:
            raise ValueError("mode_i_fraction is dimensionless and must use unit '1'.")
        digest = sha256()
        digest.update(selected_source.encode("utf-8"))
        digest.update(repr(sorted(selected_units.items())).encode("utf-8"))
        for name, values in arrays.items():
            digest.update(name.encode("ascii"))
            digest.update(np.asarray(values, dtype="<f8").tobytes())
        return cls(
            source=selected_source,
            identity_sha256=digest.hexdigest(),
            units=selected_units,
            **{name: value.copy() for name, value in arrays.items()},
        )

    @classmethod
    def read_csv(
        cls,
        path,
        *,
        source: str,
        units: dict[str, str | None] | None = None,
    ) -> "MixedModeBendingCurve":
        """Read the four-column external curve contract without pandas."""

        selected = Path(path)
        records = np.genfromtxt(
            selected,
            delimiter=",",
            names=True,
            dtype=float,
            encoding="utf-8",
        )
        required = {
            "crack_length",
            "load",
            "displacement",
            "mode_i_fraction",
        }
        names = set(records.dtype.names or ())
        if names != required:
            raise ValueError(
                "Mixed-mode CSV columns must be exactly "
                f"{sorted(required)}; received {sorted(names)}."
            )
        return cls.create(
            crack_length=records["crack_length"],
            load=records["load"],
            displacement=records["displacement"],
            mode_i_fraction=records["mode_i_fraction"],
            source=source,
            units=units,
        )


@dataclass(frozen=True)
class MixedModeBendingComparison:
    """Curve-level errors under explicitly declared scientific tolerances."""

    reference_identity_sha256: str
    predicted_identity_sha256: str
    load_relative_l2_error: float
    displacement_relative_l2_error: float
    mode_i_fraction_maximum_error: float
    load_relative_tolerance: float
    displacement_relative_tolerance: float
    mode_i_fraction_absolute_tolerance: float
    units_consistent: bool
    accepted: bool

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.mixed-mode-bending-comparison.v1",
            **self.__dict__,
        }


@dataclass(frozen=True)
class DelaminationBenchmarkSpec:
    """Geometry and evidence contract for DCB, ENF or MMB verification.

    The standard identifiers describe the specimen family.  They do not imply
    certification to a proprietary test standard.  Numerical verification
    still requires a source-identified geometry, material and reference trace.
    """

    kind: str
    width: float
    arm_thickness: float
    elastic_modulus: float
    half_span: float | None = None
    source: str = "declared analytical beam-theory oracle"

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in _DELAMINATION_STANDARDS:
            raise ValueError("Delamination benchmark kind must be dcb, enf, or mmb.")
        values = (
            float(self.width),
            float(self.arm_thickness),
            float(self.elastic_modulus),
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(
                "Width, arm thickness and elastic modulus must be positive."
            )
        span = None if self.half_span is None else float(self.half_span)
        if kind in {"enf", "mmb"} and (span is None or span <= 0.0):
            raise ValueError("ENF and MMB specifications require a positive half_span.")
        if span is not None and (not np.isfinite(span) or span <= 0.0):
            raise ValueError("half_span must be finite and positive when supplied.")
        source = str(self.source).strip()
        if not source:
            raise ValueError("A delamination benchmark must identify its source.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "width", values[0])
        object.__setattr__(self, "arm_thickness", values[1])
        object.__setattr__(self, "elastic_modulus", values[2])
        object.__setattr__(self, "half_span", span)
        object.__setattr__(self, "source", source)

    @property
    def standard_family(self) -> str:
        return _DELAMINATION_STANDARDS[self.kind]

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.delamination-benchmark-spec.v1",
            "kind": self.kind,
            "standard_family": self.standard_family,
            "width": self.width,
            "arm_thickness": self.arm_thickness,
            "elastic_modulus": self.elastic_modulus,
            "half_span": self.half_span,
            "source": self.source,
            "required_evidence": [
                "load_displacement_curve",
                "crack_length_or_process_zone",
                "cohesive_dissipation",
                "mesh_and_increment_convergence",
                "artificial_dissipation_sensitivity",
            ],
        }


@dataclass(frozen=True)
class DelaminationEnergyReleaseCurve:
    """Compliance-derived structural GI/GII evidence versus crack length."""

    crack_length: np.ndarray
    compliance: np.ndarray
    total_energy_release_rate: np.ndarray
    mode_i_energy_release_rate: np.ndarray
    mode_ii_energy_release_rate: np.ndarray
    source: str

    def __post_init__(self) -> None:
        arrays = {
            "crack_length": np.asarray(self.crack_length, dtype=float).reshape(-1),
            "compliance": np.asarray(self.compliance, dtype=float).reshape(-1),
            "total_energy_release_rate": np.asarray(
                self.total_energy_release_rate, dtype=float
            ).reshape(-1),
            "mode_i_energy_release_rate": np.asarray(
                self.mode_i_energy_release_rate, dtype=float
            ).reshape(-1),
            "mode_ii_energy_release_rate": np.asarray(
                self.mode_ii_energy_release_rate, dtype=float
            ).reshape(-1),
        }
        sizes = {values.size for values in arrays.values()}
        if len(sizes) != 1 or next(iter(sizes), 0) < 3:
            raise ValueError(
                "A delamination energy-release curve requires at least three "
                "points in every channel."
            )
        if any(not np.all(np.isfinite(values)) for values in arrays.values()):
            raise ValueError("Delamination energy-release curves must be finite.")
        if np.any(arrays["crack_length"] <= 0.0) or np.any(
            np.diff(arrays["crack_length"]) <= 0.0
        ):
            raise ValueError("crack_length must be positive and strictly increasing.")
        if np.any(arrays["compliance"] <= 0.0):
            raise ValueError("Delamination compliance must be positive.")
        energy_channels = (
            arrays["total_energy_release_rate"],
            arrays["mode_i_energy_release_rate"],
            arrays["mode_ii_energy_release_rate"],
        )
        energy_scale = max(
            (float(np.max(np.abs(values), initial=0.0)) for values in energy_channels),
            default=1.0,
        )
        tolerance = 256.0 * np.finfo(float).eps * max(energy_scale, 1.0)
        if any(np.any(values < -tolerance) for values in energy_channels):
            raise ValueError("Energy-release-rate channels must be nonnegative.")
        if not np.allclose(
            arrays["mode_i_energy_release_rate"]
            + arrays["mode_ii_energy_release_rate"],
            arrays["total_energy_release_rate"],
            rtol=1.0e-10,
            atol=tolerance,
        ):
            raise ValueError("Mode-I and Mode-II channels must sum to total G.")
        source = str(self.source).strip()
        if not source:
            raise ValueError("A delamination curve must identify its source.")
        for name, values in arrays.items():
            selected = values.copy()
            selected.setflags(write=False)
            object.__setattr__(self, name, selected)
        object.__setattr__(self, "source", source)

    @property
    def identity_sha256(self) -> str:
        digest = sha256()
        digest.update(self.source.encode("utf-8"))
        for name in (
            "crack_length",
            "compliance",
            "total_energy_release_rate",
            "mode_i_energy_release_rate",
            "mode_ii_energy_release_rate",
        ):
            digest.update(name.encode("ascii"))
            digest.update(np.asarray(getattr(self, name), dtype="<f8").tobytes())
        return digest.hexdigest()

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.delamination-energy-release-curve.v1",
            "source": self.source,
            "identity_sha256": self.identity_sha256,
            "points": int(self.crack_length.size),
            "crack_length": self.crack_length.tolist(),
            "compliance": self.compliance.tolist(),
            "total_energy_release_rate": self.total_energy_release_rate.tolist(),
            "mode_i_energy_release_rate": self.mode_i_energy_release_rate.tolist(),
            "mode_ii_energy_release_rate": self.mode_ii_energy_release_rate.tolist(),
        }


@dataclass(frozen=True)
class DelaminationBenchmarkAssessment:
    """Acceptance evidence for one structural cohesive benchmark."""

    kind: str
    energy_release_relative_l2_error: float
    energy_release_relative_tolerance: float
    minimum_process_zone_elements: float
    required_process_zone_elements: float
    artificial_dissipation_fraction: float
    maximum_artificial_dissipation_fraction: float
    accepted: bool

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.delamination-benchmark-assessment.v1",
            **self.__dict__,
        }


@dataclass(frozen=True)
class DelaminationConvergenceCertificate:
    """Three-level spatial convergence and structural-reference evidence.

    The certificate is solver independent: a cohesive, VCCT or another
    delamination provider supplies curves recovered from three successively
    refined structural models.  Acceptance requires the fine curve to match
    the source-identified reference, the last refinement change to be small,
    every level to resolve the declared process zone, and artificial
    dissipation to remain bounded.  It therefore cannot be satisfied by a
    single analytical curve relabelled as a finite-element result.
    """

    kind: str
    reference_source: str
    reference_identity_sha256: str
    curve_identity_sha256: tuple[str, ...]
    element_sizes: tuple[float, ...]
    relative_errors_to_reference: tuple[float, ...]
    mode_i_fraction_maximum_errors: tuple[float, ...]
    successive_relative_changes: tuple[float, ...]
    observed_order: float | None
    asymptotic_trend: bool
    minimum_process_zone_elements: float
    maximum_artificial_dissipation_fraction: float
    reference_relative_tolerance: float
    refinement_relative_tolerance: float
    mode_partition_absolute_tolerance: float
    required_process_zone_elements: float
    allowed_artificial_dissipation_fraction: float
    accepted: bool

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.delamination-convergence-certificate.v1",
            **self.__dict__,
        }


def delamination_benchmark_spec(kind, **geometry) -> DelaminationBenchmarkSpec:
    """Create a DCB, ENF or MMB numerical-verification specification."""

    return DelaminationBenchmarkSpec(kind=kind, **geometry)


def dcb_beam_compliance(spec, crack_length):
    """Euler--Bernoulli DCB compliance for two arms of thickness ``h``."""

    if spec.kind != "dcb":
        raise ValueError("dcb_beam_compliance requires a DCB specification.")
    a = np.asarray(crack_length, dtype=float)
    return 8.0 * a**3 / (spec.elastic_modulus * spec.width * spec.arm_thickness**3)


def enf_beam_compliance(spec, crack_length):
    """Classical simple-beam ENF compliance with support half-span ``L``."""

    if spec.kind != "enf":
        raise ValueError("enf_beam_compliance requires an ENF specification.")
    a = np.asarray(crack_length, dtype=float)
    return (2.0 * spec.half_span**3 + 3.0 * a**3) / (
        8.0 * spec.elastic_modulus * spec.width * spec.arm_thickness**3
    )


def compliance_energy_release_curve(
    spec: DelaminationBenchmarkSpec,
    *,
    crack_length,
    load,
    displacement=None,
    compliance=None,
    mode_i_fraction=None,
    source: str | None = None,
) -> DelaminationEnergyReleaseCurve:
    """Recover structural energy release by the compliance derivative.

    ``G = P^2/(2b) dC/da`` is evaluated on the supplied crack coordinates.
    DCB and ENF select pure Mode I and II respectively.  MMB requires the
    mode-I partition from an independently declared analytical or numerical
    source; AgentFEM does not infer it from a scalar load/displacement trace.
    """

    if not isinstance(spec, DelaminationBenchmarkSpec):
        raise TypeError("A DelaminationBenchmarkSpec is required.")
    a = _curve_values(crack_length, name="crack_length")
    if a.size < 3 or np.any(np.diff(a) <= 0.0) or np.any(a <= 0.0):
        raise ValueError(
            "Compliance differentiation needs three increasing crack lengths."
        )
    p = _curve_values(load, name="load")
    if p.size != a.size:
        raise ValueError("load and crack_length must have equal length.")
    if compliance is None:
        if displacement is None:
            raise ValueError("Supply either compliance or displacement.")
        delta = _curve_values(displacement, name="displacement")
        if delta.size != a.size or np.any(np.abs(p) <= np.finfo(float).eps):
            raise ValueError("Displacement needs nonzero matching loads.")
        selected_compliance = delta / p
    else:
        selected_compliance = _curve_values(compliance, name="compliance")
        if selected_compliance.size != a.size:
            raise ValueError("compliance and crack_length must have equal length.")
    derivative = np.gradient(selected_compliance, a, edge_order=2)
    total = p**2 * derivative / (2.0 * spec.width)
    scale = max(float(np.max(np.abs(total), initial=0.0)), 1.0)
    if np.any(total < -1.0e-12 * scale):
        raise ValueError(
            "Compliance decreased enough to produce negative energy release."
        )
    total = np.maximum(total, 0.0)
    if spec.kind == "dcb":
        fraction = np.ones_like(total)
    elif spec.kind == "enf":
        fraction = np.zeros_like(total)
    else:
        if mode_i_fraction is None:
            raise ValueError("MMB recovery requires a declared mode_i_fraction curve.")
        fraction = np.asarray(mode_i_fraction, dtype=float).reshape(-1)
        if fraction.size != a.size or np.any((fraction < 0.0) | (fraction > 1.0)):
            raise ValueError(
                "MMB mode_i_fraction must match the curve and lie in [0, 1]."
            )
    return DelaminationEnergyReleaseCurve(
        crack_length=a.copy(),
        compliance=selected_compliance.copy(),
        total_energy_release_rate=total,
        mode_i_energy_release_rate=total * fraction,
        mode_ii_energy_release_rate=total * (1.0 - fraction),
        source=str(source or spec.source),
    )


def beam_theory_energy_release_curve(spec, *, crack_length, load):
    """Return a DCB/ENF analytical oracle through the same public contract."""

    if spec.kind == "dcb":
        compliance = dcb_beam_compliance(spec, crack_length)
    elif spec.kind == "enf":
        compliance = enf_beam_compliance(spec, crack_length)
    else:
        raise ValueError("MMB needs a source-declared compliance and mode partition.")
    return compliance_energy_release_curve(
        spec,
        crack_length=crack_length,
        load=load,
        compliance=compliance,
        source=f"{spec.source}; classical simple-beam compliance",
    )


def assess_delamination_benchmark(
    spec,
    predicted,
    reference,
    *,
    energy_release_relative_tolerance,
    minimum_process_zone_elements,
    required_process_zone_elements=3.0,
    artificial_dissipation=0.0,
    internal_energy=1.0,
) -> DelaminationBenchmarkAssessment:
    """Apply curve, cohesive-zone resolution and dissipation guardrails."""

    if not isinstance(spec, DelaminationBenchmarkSpec):
        raise TypeError("A DelaminationBenchmarkSpec is required.")
    if not isinstance(predicted, DelaminationEnergyReleaseCurve) or not isinstance(
        reference, DelaminationEnergyReleaseCurve
    ):
        raise TypeError("Predicted and reference values must be energy-release curves.")
    if (
        predicted.crack_length[0] > reference.crack_length[0]
        or predicted.crack_length[-1] < reference.crack_length[-1]
    ):
        raise ValueError("Predicted crack range must cover the reference range.")
    coordinates = reference.crack_length
    observed = np.interp(
        coordinates, predicted.crack_length, predicted.total_energy_release_rate
    )
    error = _relative_l2(observed, reference.total_energy_release_rate)
    tolerance = float(energy_release_relative_tolerance)
    elements = float(minimum_process_zone_elements)
    required = float(required_process_zone_elements)
    numerical = abs(float(artificial_dissipation))
    physical = abs(float(internal_energy))
    if any(
        not np.isfinite(value) or value < 0.0
        for value in (tolerance, elements, required, numerical, physical)
    ):
        raise ValueError(
            "Benchmark tolerances and evidence must be finite and nonnegative."
        )
    fraction = numerical / max(physical, np.finfo(float).eps)
    maximum_fraction = 0.05
    return DelaminationBenchmarkAssessment(
        kind=spec.kind,
        energy_release_relative_l2_error=error,
        energy_release_relative_tolerance=tolerance,
        minimum_process_zone_elements=elements,
        required_process_zone_elements=required,
        artificial_dissipation_fraction=fraction,
        maximum_artificial_dissipation_fraction=maximum_fraction,
        accepted=(
            error <= tolerance
            and elements >= required
            and fraction <= maximum_fraction
        ),
    )


def certify_delamination_convergence(
    spec,
    curves,
    reference,
    *,
    element_sizes,
    process_zone_elements,
    artificial_dissipation_fractions,
    reference_relative_tolerance,
    refinement_relative_tolerance,
    mode_partition_absolute_tolerance=0.02,
    required_process_zone_elements=3.0,
    allowed_artificial_dissipation_fraction=0.05,
) -> DelaminationConvergenceCertificate:
    """Certify a DCB/ENF/MMB curve using three or more structural levels."""

    if not isinstance(spec, DelaminationBenchmarkSpec):
        raise TypeError("A DelaminationBenchmarkSpec is required.")
    selected = tuple(curves)
    if len(selected) < 3 or not all(
        isinstance(item, DelaminationEnergyReleaseCurve) for item in selected
    ):
        raise ValueError("Delamination convergence requires at least three curves.")
    if not isinstance(reference, DelaminationEnergyReleaseCurve):
        raise TypeError("reference must be a DelaminationEnergyReleaseCurve.")
    sizes = tuple(float(value) for value in element_sizes)
    zones = tuple(float(value) for value in process_zone_elements)
    dissipation = tuple(float(value) for value in artificial_dissipation_fractions)
    if not (len(sizes) == len(zones) == len(dissipation) == len(selected)):
        raise ValueError("Every structural level needs size, process-zone and dissipation evidence.")
    if any(not np.isfinite(value) or value <= 0.0 for value in sizes + zones):
        raise ValueError("Element sizes and process-zone resolutions must be positive.")
    if any(not np.isfinite(value) or value < 0.0 for value in dissipation):
        raise ValueError("Artificial-dissipation fractions must be nonnegative.")
    if any(left <= right for left, right in zip(sizes[:-1], sizes[1:])):
        raise ValueError("element_sizes must decrease from coarse to fine.")
    coordinates = reference.crack_length

    def aligned(curve):
        if (
            curve.crack_length[0] > coordinates[0]
            or curve.crack_length[-1] < coordinates[-1]
        ):
            raise ValueError("Every curve must cover the reference crack range.")
        return {
            "total": np.interp(
                coordinates,
                curve.crack_length,
                curve.total_energy_release_rate,
            ),
            "mode_i": np.interp(
                coordinates,
                curve.crack_length,
                curve.mode_i_energy_release_rate,
            ),
        }

    values = tuple(aligned(item) for item in selected)
    errors = tuple(
        _relative_l2(item["total"], reference.total_energy_release_rate)
        for item in values
    )
    changes = tuple(
        _relative_l2(right["total"], left["total"])
        for left, right in zip(values[:-1], values[1:])
    )
    reference_total = np.asarray(reference.total_energy_release_rate, dtype=float)
    reference_mode_i = np.asarray(reference.mode_i_energy_release_rate, dtype=float)
    active = np.abs(reference_total) > np.finfo(float).eps
    if not np.any(active):
        raise ValueError("Reference energy-release curve is identically zero.")
    reference_fraction = reference_mode_i[active] / reference_total[active]
    partition_errors = tuple(
        (
            float("inf")
            if np.any(np.abs(item["total"][active]) <= np.finfo(float).eps)
            else float(
                np.max(
                    np.abs(
                        item["mode_i"][active] / item["total"][active]
                        - reference_fraction
                    ),
                    initial=0.0,
                )
            )
        )
        for item in values
    )
    observed_order = None
    if changes[-1] > np.finfo(float).eps and len(changes) >= 2:
        ratio = sizes[-2] / sizes[-1]
        if ratio > 1.0 and changes[-2] > 0.0:
            observed_order = float(np.log(changes[-2] / changes[-1]) / np.log(ratio))
    asymptotic = bool(
        all(right < left for left, right in zip(errors[:-1], errors[1:]))
        and all(right < left for left, right in zip(changes[:-1], changes[1:]))
    )
    reference_tolerance = float(reference_relative_tolerance)
    refinement_tolerance = float(refinement_relative_tolerance)
    partition_tolerance = float(mode_partition_absolute_tolerance)
    required_zone = float(required_process_zone_elements)
    allowed_dissipation = float(allowed_artificial_dissipation_fraction)
    controls = (
        reference_tolerance,
        refinement_tolerance,
        partition_tolerance,
        required_zone,
        allowed_dissipation,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in controls):
        raise ValueError("Convergence tolerances must be finite and nonnegative.")
    return DelaminationConvergenceCertificate(
        kind=spec.kind,
        reference_source=reference.source,
        reference_identity_sha256=reference.identity_sha256,
        curve_identity_sha256=tuple(item.identity_sha256 for item in selected),
        element_sizes=sizes,
        relative_errors_to_reference=errors,
        mode_i_fraction_maximum_errors=partition_errors,
        successive_relative_changes=changes,
        observed_order=observed_order,
        asymptotic_trend=asymptotic,
        minimum_process_zone_elements=min(zones),
        maximum_artificial_dissipation_fraction=max(dissipation),
        reference_relative_tolerance=reference_tolerance,
        refinement_relative_tolerance=refinement_tolerance,
        mode_partition_absolute_tolerance=partition_tolerance,
        required_process_zone_elements=required_zone,
        allowed_artificial_dissipation_fraction=allowed_dissipation,
        accepted=(
            asymptotic
            and errors[-1] <= reference_tolerance
            and changes[-1] <= refinement_tolerance
            and partition_errors[-1] <= partition_tolerance
            and min(zones) >= required_zone
            and max(dissipation) <= allowed_dissipation
        ),
    )


def _relative_l2(observed: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.linalg.norm(reference))
    if scale <= np.finfo(float).eps:
        return float(np.linalg.norm(observed - reference))
    return float(np.linalg.norm(observed - reference) / scale)


def compare_mixed_mode_bending_curves(
    reference: MixedModeBendingCurve,
    predicted: MixedModeBendingCurve,
    *,
    load_relative_tolerance: float,
    displacement_relative_tolerance: float,
    mode_i_fraction_absolute_tolerance: float,
) -> MixedModeBendingComparison:
    """Compare a computed curve on the reference crack-length coordinates."""

    tolerances = (
        float(load_relative_tolerance),
        float(displacement_relative_tolerance),
        float(mode_i_fraction_absolute_tolerance),
    )
    if any(not np.isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError(
            "Mixed-mode comparison tolerances must be finite and nonnegative."
        )
    if reference.units != predicted.units:
        raise ValueError(
            "Reference and predicted mixed-mode curves must use identical units."
        )
    if (
        predicted.crack_length[0] > reference.crack_length[0]
        or predicted.crack_length[-1] < reference.crack_length[-1]
    ):
        raise ValueError("Predicted crack-length range must cover the reference range.")
    coordinates = reference.crack_length
    load_error = _relative_l2(
        np.interp(coordinates, predicted.crack_length, predicted.load),
        reference.load,
    )
    displacement_error = _relative_l2(
        np.interp(coordinates, predicted.crack_length, predicted.displacement),
        reference.displacement,
    )
    mix_error = float(
        np.max(
            np.abs(
                np.interp(
                    coordinates,
                    predicted.crack_length,
                    predicted.mode_i_fraction,
                )
                - reference.mode_i_fraction
            )
        )
    )
    units_consistent = True
    return MixedModeBendingComparison(
        reference_identity_sha256=reference.identity_sha256,
        predicted_identity_sha256=predicted.identity_sha256,
        load_relative_l2_error=load_error,
        displacement_relative_l2_error=displacement_error,
        mode_i_fraction_maximum_error=mix_error,
        load_relative_tolerance=tolerances[0],
        displacement_relative_tolerance=tolerances[1],
        mode_i_fraction_absolute_tolerance=tolerances[2],
        units_consistent=units_consistent,
        accepted=(
            units_consistent
            and
            load_error <= tolerances[0]
            and displacement_error <= tolerances[1]
            and mix_error <= tolerances[2]
        ),
    )


__all__ = [
    "DelaminationBenchmarkAssessment",
    "DelaminationBenchmarkSpec",
    "DelaminationConvergenceCertificate",
    "DelaminationEnergyReleaseCurve",
    "MixedModeBendingComparison",
    "MixedModeBendingCurve",
    "assess_delamination_benchmark",
    "beam_theory_energy_release_curve",
    "compliance_energy_release_curve",
    "certify_delamination_convergence",
    "compare_mixed_mode_bending_curves",
    "dcb_beam_compliance",
    "delamination_benchmark_spec",
    "enf_beam_compliance",
]
