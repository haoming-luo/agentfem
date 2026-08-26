"""Method-neutral two-dimensional crack geometry and tip evidence.

The records in this module are deliberately independent of FEM, cohesive,
phase-field, XFEM, or neural-field implementations.  They provide one stable
scientific identity for predefined cracks and one result contract for
structure-level stress-intensity extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import cos, hypot, isfinite, pi, sin, sqrt
from typing import Mapping, Protocol, runtime_checkable


Point2D = tuple[float, float]


class UnsupportedCrackGeometryError(ValueError):
    """A fail-closed geometry error with a stable machine-facing code."""

    code = "AFM-FRACTURE-GEOMETRY-001"


@dataclass(frozen=True)
class CrackTip2D:
    """One stable crack-tip identity and its local orientation."""

    crack_id: str
    end: str
    point: Point2D
    crack_tangent: Point2D
    extension_direction: Point2D
    normal: Point2D

    def __post_init__(self) -> None:
        if self.end not in {"start", "end"}:
            raise ValueError("CrackTip2D.end must be 'start' or 'end'.")

    @property
    def tip_id(self) -> str:
        return f"{self.crack_id}:{self.end}"

    def summary(self) -> dict[str, object]:
        return {
            "tip_id": self.tip_id,
            "crack_id": self.crack_id,
            "end": self.end,
            "point": self.point,
            "crack_tangent": self.crack_tangent,
            "extension_direction": self.extension_direction,
            "normal": self.normal,
        }


@dataclass(frozen=True)
class CrackSegment2D:
    """A named, oriented, straight crack segment in the reference domain."""

    crack_id: str
    start: Point2D
    end: Point2D
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier = str(self.crack_id).strip()
        if not identifier:
            raise ValueError("CrackSegment2D.crack_id must not be empty.")
        start = _point(self.start, "CrackSegment2D.start")
        end = _point(self.end, "CrackSegment2D.end")
        if _distance(start, end) <= 1.0e-14:
            raise UnsupportedCrackGeometryError(
                f"{UnsupportedCrackGeometryError.code}: crack {identifier!r} "
                "has zero length."
            )
        object.__setattr__(self, "crack_id", identifier)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def length(self) -> float:
        return _distance(self.start, self.end)

    @property
    def tangent(self) -> Point2D:
        length = self.length
        return (
            (self.end[0] - self.start[0]) / length,
            (self.end[1] - self.start[1]) / length,
        )

    @property
    def normal(self) -> Point2D:
        tangent = self.tangent
        return (-tangent[1], tangent[0])

    @property
    def tips(self) -> tuple[CrackTip2D, CrackTip2D]:
        tangent = self.tangent
        normal = self.normal
        return (
            CrackTip2D(
                self.crack_id,
                "start",
                self.start,
                tangent,
                (-tangent[0], -tangent[1]),
                (-normal[0], -normal[1]),
            ),
            CrackTip2D(
                self.crack_id,
                "end",
                self.end,
                tangent,
                tangent,
                normal,
            ),
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "straight_crack_2d",
            "crack_id": self.crack_id,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "tangent": self.tangent,
            "normal": self.normal,
            "tips": [tip.summary() for tip in self.tips],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CrackSet2D:
    """A validated set of mutually non-intersecting straight cracks."""

    cracks: tuple[CrackSegment2D, ...]
    name: str = "cracks"
    tolerance: float = 1.0e-10
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cracks = tuple(self.cracks)
        if not cracks:
            raise ValueError("CrackSet2D requires at least one crack.")
        if any(not isinstance(item, CrackSegment2D) for item in cracks):
            raise TypeError("CrackSet2D.cracks must contain CrackSegment2D records.")
        identifiers = [item.crack_id for item in cracks]
        if len(set(identifiers)) != len(identifiers):
            raise UnsupportedCrackGeometryError(
                f"{UnsupportedCrackGeometryError.code}: crack IDs must be unique."
            )
        tolerance = float(self.tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("CrackSet2D.tolerance must be finite and positive.")
        name = str(self.name).strip()
        if not name:
            raise ValueError("CrackSet2D.name must not be empty.")
        for index, first in enumerate(cracks):
            for second in cracks[index + 1 :]:
                if _segments_intersect(first, second, tolerance=tolerance):
                    raise UnsupportedCrackGeometryError(
                        f"{UnsupportedCrackGeometryError.code}: cracks "
                        f"{first.crack_id!r} and {second.crack_id!r} intersect, "
                        "touch, or overlap; this contract accepts mutually "
                        "separated straight cracks only."
                    )
        object.__setattr__(self, "cracks", cracks)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def tips(self) -> tuple[CrackTip2D, ...]:
        return tuple(tip for crack in self.cracks for tip in crack.tips)

    def crack(self, crack_id: str) -> CrackSegment2D:
        selected = str(crack_id)
        for crack in self.cracks:
            if crack.crack_id == selected:
                return crack
        raise KeyError(f"Unknown crack ID {selected!r}.")

    def tip(self, tip_id: str) -> CrackTip2D:
        selected = str(tip_id)
        for tip in self.tips:
            if tip.tip_id == selected:
                return tip
        raise KeyError(f"Unknown crack-tip ID {selected!r}.")

    def admissible_tip_radius(
        self,
        tip_id: str,
        *,
        bounds: tuple[float, float, float, float] | None = None,
        safety_factor: float = 0.45,
    ) -> float:
        """Return a conservative ring radius clear of other tips and boundaries.

        The owning crack is intentionally excluded: an interaction-integral
        domain is expected to intersect its two crack faces.  Other crack
        segments, the opposite tip, and optional rectangular domain boundaries
        limit the radius.
        """

        factor = float(safety_factor)
        if not 0.0 < factor < 0.5:
            raise ValueError("safety_factor must satisfy 0 < value < 0.5.")
        tip = self.tip(tip_id)
        owner = self.crack(tip.crack_id)
        distances = [
            _distance(tip.point, other.point)
            for other in owner.tips
            if other.tip_id != tip.tip_id
        ]
        distances.extend(
            _point_segment_distance(tip.point, crack.start, crack.end)
            for crack in self.cracks
            if crack.crack_id != tip.crack_id
        )
        if bounds is not None:
            xmin, xmax, ymin, ymax = (float(item) for item in bounds)
            if not xmin < xmax or not ymin < ymax:
                raise ValueError("bounds must be (xmin, xmax, ymin, ymax).")
            x, y = tip.point
            if x < xmin - self.tolerance or x > xmax + self.tolerance:
                raise UnsupportedCrackGeometryError(
                    f"{UnsupportedCrackGeometryError.code}: tip {tip.tip_id!r} "
                    "lies outside the declared bounds."
                )
            if y < ymin - self.tolerance or y > ymax + self.tolerance:
                raise UnsupportedCrackGeometryError(
                    f"{UnsupportedCrackGeometryError.code}: tip {tip.tip_id!r} "
                    "lies outside the declared bounds."
                )
            distances.extend((x - xmin, xmax - x, y - ymin, ymax - y))
        clearance = min(distances)
        if clearance <= self.tolerance:
            raise UnsupportedCrackGeometryError(
                f"{UnsupportedCrackGeometryError.code}: no positive integration "
                f"radius is available for tip {tip.tip_id!r}."
            )
        return factor * clearance

    def summary(self) -> dict[str, object]:
        return {
            "kind": "crack_set_2d",
            "schema_version": "0.1.0",
            "name": self.name,
            "cracks": [item.summary() for item in self.cracks],
            "tip_ids": [item.tip_id for item in self.tips],
            "tolerance": self.tolerance,
            "metadata": dict(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.summary(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class FractureField2D(Protocol):
    """Provider-neutral field access needed by a future tip integral adapter."""

    def displacement(self, points, *, side: str | None = None): ...

    def displacement_gradient(self, points, *, side: str | None = None): ...

    def stress(self, points, *, side: str | None = None): ...


@dataclass(frozen=True)
class StressIntensityReport:
    """Per-tip mixed-mode result with explicit path-sensitivity evidence."""

    crack_id: str
    tip_id: str
    k_i: float
    k_ii: float
    j_integral: float
    extraction_method: str
    integration_radii: tuple[float, ...]
    k_i_by_radius: tuple[float, ...]
    k_ii_by_radius: tuple[float, ...]
    j_by_radius: tuple[float, ...]
    path_variation: float
    coordinate_system: Mapping[str, object]
    stress_intensity_unit: str = "Pa*sqrt(m)"
    j_unit: str = "J/m^2"
    status: str = "inconclusive"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        radii = tuple(float(item) for item in self.integration_radii)
        channels = (
            tuple(float(item) for item in self.k_i_by_radius),
            tuple(float(item) for item in self.k_ii_by_radius),
            tuple(float(item) for item in self.j_by_radius),
        )
        if len(radii) < 2 or any(len(item) != len(radii) for item in channels):
            raise ValueError("StressIntensityReport requires at least two equal-length rings.")
        if any(not isfinite(item) or item <= 0.0 for item in radii):
            raise ValueError("StressIntensityReport radii must be finite and positive.")
        if any(right <= left for left, right in zip(radii, radii[1:])):
            raise ValueError("StressIntensityReport radii must increase strictly.")
        values = (self.k_i, self.k_ii, self.j_integral, self.path_variation)
        if any(not isfinite(float(item)) for item in values + sum(channels, ())):
            raise ValueError("StressIntensityReport values must be finite.")
        if self.status not in {"accepted", "uncertain", "failed", "inconclusive"}:
            raise ValueError("StressIntensityReport.status is not recognized.")
        object.__setattr__(self, "integration_radii", radii)
        object.__setattr__(self, "k_i_by_radius", channels[0])
        object.__setattr__(self, "k_ii_by_radius", channels[1])
        object.__setattr__(self, "j_by_radius", channels[2])
        object.__setattr__(self, "coordinate_system", dict(self.coordinate_system))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "stress_intensity_report",
            "crack_id": self.crack_id,
            "tip_id": self.tip_id,
            "K_I": self.k_i,
            "K_II": self.k_ii,
            "J": self.j_integral,
            "extraction_method": self.extraction_method,
            "integration_radii": self.integration_radii,
            "K_I_by_radius": self.k_i_by_radius,
            "K_II_by_radius": self.k_ii_by_radius,
            "J_by_radius": self.j_by_radius,
            "path_variation": self.path_variation,
            "coordinate_system": dict(self.coordinate_system),
            "stress_intensity_unit": self.stress_intensity_unit,
            "j_unit": self.j_unit,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LinearElasticFractureMaterial2D:
    """Isotropic elastic constants used by a two-dimensional LEFM result."""

    young_modulus: float
    poisson_ratio: float
    assumption: str = "plane_strain"

    def __post_init__(self) -> None:
        modulus = float(self.young_modulus)
        ratio = float(self.poisson_ratio)
        assumption = str(self.assumption).strip().lower().replace("-", "_")
        if not isfinite(modulus) or modulus <= 0.0:
            raise ValueError("young_modulus must be finite and positive.")
        if not isfinite(ratio) or not -1.0 < ratio < 0.5:
            raise ValueError("poisson_ratio must satisfy -1 < value < 0.5.")
        if assumption not in {"plane_stress", "plane_strain"}:
            raise ValueError("assumption must be 'plane_stress' or 'plane_strain'.")
        object.__setattr__(self, "young_modulus", modulus)
        object.__setattr__(self, "poisson_ratio", ratio)
        object.__setattr__(self, "assumption", assumption)

    @property
    def effective_modulus(self) -> float:
        """Return E' in J=(K_I^2+K_II^2)/E'."""

        if self.assumption == "plane_stress":
            return self.young_modulus
        return self.young_modulus / (1.0 - self.poisson_ratio**2)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "linear_elastic_fracture_material_2d",
            "young_modulus": self.young_modulus,
            "poisson_ratio": self.poisson_ratio,
            "assumption": self.assumption,
            "effective_modulus": self.effective_modulus,
        }


@dataclass(frozen=True)
class RemoteStress2D:
    """A symmetric remote Cauchy-stress tensor in global x-y coordinates."""

    xx: float = 0.0
    yy: float = 0.0
    xy: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(float(item) for item in (self.xx, self.yy, self.xy))
        if any(not isfinite(item) for item in values):
            raise ValueError("RemoteStress2D components must be finite.")
        object.__setattr__(self, "xx", values[0])
        object.__setattr__(self, "yy", values[1])
        object.__setattr__(self, "xy", values[2])

    def resolved(self, tip: CrackTip2D) -> tuple[float, float]:
        """Return normal and shear traction in the tip's right-handed frame."""

        e1x, e1y = tip.extension_direction
        e2x, e2y = tip.normal
        traction_x = self.xx * e2x + self.xy * e2y
        traction_y = self.xy * e2x + self.yy * e2y
        normal = e2x * traction_x + e2y * traction_y
        shear = e1x * traction_x + e1y * traction_y
        return normal, shear

    def summary(self) -> dict[str, object]:
        return {
            "kind": "remote_stress_2d",
            "components": {"xx": self.xx, "yy": self.yy, "xy": self.xy},
            "matrix": ((self.xx, self.xy), (self.xy, self.yy)),
        }


@dataclass(frozen=True)
class StressIntensityReference:
    """An analytical or independently established per-tip LEFM reference."""

    crack_id: str
    tip_id: str
    k_i: float
    k_ii: float
    j_integral: float
    method: str
    coordinate_system: Mapping[str, object]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = tuple(float(item) for item in (self.k_i, self.k_ii, self.j_integral))
        if any(not isfinite(item) for item in values):
            raise ValueError("StressIntensityReference values must be finite.")
        object.__setattr__(self, "k_i", values[0])
        object.__setattr__(self, "k_ii", values[1])
        object.__setattr__(self, "j_integral", values[2])
        object.__setattr__(self, "coordinate_system", dict(self.coordinate_system))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "stress_intensity_reference",
            "crack_id": self.crack_id,
            "tip_id": self.tip_id,
            "K_I": self.k_i,
            "K_II": self.k_ii,
            "J": self.j_integral,
            "method": self.method,
            "coordinate_system": dict(self.coordinate_system),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StressIntensityVerification:
    """Comparison of extracted tip quantities with an independent reference."""

    report: StressIntensityReport
    reference: StressIntensityReference
    relative_k_error: float
    relative_j_error: float
    tolerance: float
    status: str

    def summary(self) -> dict[str, object]:
        return {
            "kind": "stress_intensity_verification",
            "tip_id": self.report.tip_id,
            "relative_K_error": self.relative_k_error,
            "relative_J_error": self.relative_j_error,
            "tolerance": self.tolerance,
            "path_status": self.report.status,
            "status": self.status,
            "reference": self.reference.summary(),
        }


def segment(
    crack_id: str,
    *,
    start: Point2D | None = None,
    end: Point2D | None = None,
    center: Point2D | None = None,
    length: float | None = None,
    angle: float | None = None,
    metadata: Mapping[str, object] | None = None,
) -> CrackSegment2D:
    """Create one straight crack from endpoints or center/length/angle."""

    endpoint_form = start is not None or end is not None
    centered_form = center is not None or length is not None or angle is not None
    if endpoint_form == centered_form:
        raise ValueError(
            "Define a crack by exactly one of (start, end) or "
            "(center, length, angle)."
        )
    if endpoint_form:
        if start is None or end is None:
            raise ValueError("Both start and end are required.")
        return CrackSegment2D(crack_id, start, end, metadata or {})
    if center is None or length is None or angle is None:
        raise ValueError("center, length, and angle are all required.")
    midpoint = _point(center, "center")
    selected_length = float(length)
    selected_angle = float(angle)
    if not isfinite(selected_length) or selected_length <= 0.0:
        raise ValueError("length must be finite and positive.")
    if not isfinite(selected_angle):
        raise ValueError("angle must be finite and expressed in radians.")
    half = 0.5 * selected_length
    offset = (half * cos(selected_angle), half * sin(selected_angle))
    return CrackSegment2D(
        crack_id,
        (midpoint[0] - offset[0], midpoint[1] - offset[1]),
        (midpoint[0] + offset[0], midpoint[1] + offset[1]),
        metadata or {},
    )


def crack_set(
    *cracks: CrackSegment2D,
    name: str = "cracks",
    tolerance: float = 1.0e-10,
    metadata: Mapping[str, object] | None = None,
) -> CrackSet2D:
    return CrackSet2D(tuple(cracks), name, tolerance, metadata or {})


def stress_intensity_report(
    *,
    crack: CrackSet2D,
    tip_id: str,
    integration_radii,
    k_i,
    k_ii,
    j_integral,
    extraction_method: str = "interaction_domain_integral",
    relative_path_tolerance: float = 0.03,
    stress_intensity_unit: str = "Pa*sqrt(m)",
    j_unit: str = "J/m^2",
    metadata: Mapping[str, object] | None = None,
) -> StressIntensityReport:
    """Reduce multiple integration rings to one auditable per-tip report."""

    tip = crack.tip(tip_id)
    radii = tuple(float(item) for item in integration_radii)
    mode_i = tuple(float(item) for item in k_i)
    mode_ii = tuple(float(item) for item in k_ii)
    energy = tuple(float(item) for item in j_integral)
    if not radii or len(mode_i) != len(radii) or len(mode_ii) != len(radii):
        raise ValueError("K_I and K_II must provide one value for every radius.")
    if len(energy) != len(radii):
        raise ValueError("J must provide one value for every radius.")
    tolerance = float(relative_path_tolerance)
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_path_tolerance must be finite and positive.")
    average_i = sum(mode_i) / len(mode_i)
    average_ii = sum(mode_ii) / len(mode_ii)
    average_j = sum(energy) / len(energy)
    scale = max(hypot(average_i, average_ii), 1.0e-30)
    k_variation = max(
        hypot(item_i - average_i, item_ii - average_ii) / scale
        for item_i, item_ii in zip(mode_i, mode_ii)
    )
    j_scale = max(abs(average_j), 1.0e-30)
    j_variation = max(abs(item - average_j) / j_scale for item in energy)
    variation = max(k_variation, j_variation)
    return StressIntensityReport(
        crack_id=tip.crack_id,
        tip_id=tip.tip_id,
        k_i=average_i,
        k_ii=average_ii,
        j_integral=average_j,
        extraction_method=str(extraction_method),
        integration_radii=radii,
        k_i_by_radius=mode_i,
        k_ii_by_radius=mode_ii,
        j_by_radius=energy,
        path_variation=variation,
        coordinate_system={
            "origin": tip.point,
            "crack_tangent": tip.crack_tangent,
            "extension_direction": tip.extension_direction,
            "normal": tip.normal,
            "sign_convention": (
                "Each tip uses a right-handed frame: x1 is the outward crack-"
                "extension direction and x2 its counterclockwise normal. K_I "
                "opens along x2; positive K_II shears along x1."
            ),
        },
        stress_intensity_unit=stress_intensity_unit,
        j_unit=j_unit,
        status="accepted" if variation <= tolerance else "uncertain",
        metadata={
            "relative_path_tolerance": tolerance,
            **dict(metadata or {}),
        },
    )


def linear_elastic_fracture_material(
    *,
    young_modulus: float,
    poisson_ratio: float,
    assumption: str = "plane_strain",
) -> LinearElasticFractureMaterial2D:
    """Create the material part of a two-dimensional LEFM contract."""

    return LinearElasticFractureMaterial2D(
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        assumption=assumption,
    )


def remote_stress(
    *, xx: float = 0.0, yy: float = 0.0, xy: float = 0.0
) -> RemoteStress2D:
    """Create a symmetric remote stress tensor in global coordinates."""

    return RemoteStress2D(xx=xx, yy=yy, xy=xy)


def infinite_plate_stress_intensity(
    *,
    crack: CrackSet2D,
    tip_id: str,
    stress: RemoteStress2D,
    material: LinearElasticFractureMaterial2D,
) -> StressIntensityReference:
    r"""Return the exact infinite-plate SIF for one straight central crack.

    The crack has total length ``2a`` and is subjected to a uniform remote
    stress. Resolving that tensor in each tip's right-handed local frame gives
    ``K_I = sigma_nn sqrt(pi a)`` and ``K_II = tau_1n sqrt(pi a)``. This small
    analytical oracle validates numerical extraction; it is not a finite-domain
    correction formula.
    """

    if not isinstance(stress, RemoteStress2D):
        raise TypeError("stress must be a RemoteStress2D record.")
    if not isinstance(material, LinearElasticFractureMaterial2D):
        raise TypeError("material must be a LinearElasticFractureMaterial2D record.")
    tip = crack.tip(tip_id)
    segment_record = crack.crack(tip.crack_id)
    normal_stress, shear_stress = stress.resolved(tip)
    factor = sqrt(pi * 0.5 * segment_record.length)
    mode_i = normal_stress * factor
    mode_ii = shear_stress * factor
    energy = (mode_i**2 + mode_ii**2) / material.effective_modulus
    return StressIntensityReference(
        crack_id=tip.crack_id,
        tip_id=tip.tip_id,
        k_i=mode_i,
        k_ii=mode_ii,
        j_integral=energy,
        method="analytical_infinite_plate_central_crack",
        coordinate_system={
            "origin": tip.point,
            "extension_direction": tip.extension_direction,
            "normal": tip.normal,
            "handedness": "right",
            "K_I": "positive normal traction opens the crack",
            "K_II": "positive shear traction acts along the extension direction",
        },
        metadata={
            "half_crack_length": 0.5 * segment_record.length,
            "remote_stress": stress.summary(),
            "material": material.summary(),
            "scope": "infinite homogeneous isotropic plate; central straight crack",
        },
    )


def verify_stress_intensity(
    report: StressIntensityReport,
    reference: StressIntensityReference,
    *,
    relative_tolerance: float = 0.03,
) -> StressIntensityVerification:
    """Verify one numerical extraction without hiding its path-sensitivity."""

    if report.tip_id != reference.tip_id or report.crack_id != reference.crack_id:
        raise ValueError("Report and reference must describe the same crack tip.")
    tolerance = float(relative_tolerance)
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive.")
    reference_scale = max(hypot(reference.k_i, reference.k_ii), 1.0e-30)
    k_error = hypot(
        report.k_i - reference.k_i,
        report.k_ii - reference.k_ii,
    ) / reference_scale
    j_scale = max(abs(reference.j_integral), 1.0e-30)
    j_error = abs(report.j_integral - reference.j_integral) / j_scale
    accepted = (
        report.status == "accepted"
        and k_error <= tolerance
        and j_error <= tolerance
    )
    return StressIntensityVerification(
        report=report,
        reference=reference,
        relative_k_error=k_error,
        relative_j_error=j_error,
        tolerance=tolerance,
        status="accepted" if accepted else "failed",
    )


def _point(value, label: str) -> Point2D:
    try:
        x, y = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly two coordinates.") from exc
    point = (float(x), float(y))
    if any(not isfinite(item) for item in point):
        raise ValueError(f"{label} must contain finite coordinates.")
    return point


def _distance(first: Point2D, second: Point2D) -> float:
    return hypot(second[0] - first[0], second[1] - first[1])


def _cross(first: Point2D, second: Point2D, third: Point2D) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _segments_intersect(
    first: CrackSegment2D,
    second: CrackSegment2D,
    *,
    tolerance: float,
) -> bool:
    a, b = first.start, first.end
    c, d = second.start, second.end
    values = (_cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b))
    scale = max(first.length, second.length, 1.0)
    eps = tolerance * scale * scale
    if (
        values[0] * values[1] < -eps * eps
        and values[2] * values[3] < -eps * eps
    ):
        return True
    return any(
        abs(value) <= eps and _on_segment(start, end, point, tolerance=tolerance)
        for value, start, end, point in (
            (values[0], a, b, c),
            (values[1], a, b, d),
            (values[2], c, d, a),
            (values[3], c, d, b),
        )
    )


def _on_segment(start, end, point, *, tolerance):
    return (
        min(start[0], end[0]) - tolerance
        <= point[0]
        <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance
        <= point[1]
        <= max(start[1], end[1]) + tolerance
    )


def _point_segment_distance(point, start, end) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    fraction = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / denominator
    selected = min(1.0, max(0.0, fraction))
    projection = (start[0] + selected * dx, start[1] + selected * dy)
    return _distance(point, projection)


__all__ = [
    "CrackSegment2D",
    "CrackSet2D",
    "CrackTip2D",
    "FractureField2D",
    "LinearElasticFractureMaterial2D",
    "RemoteStress2D",
    "StressIntensityReport",
    "StressIntensityReference",
    "StressIntensityVerification",
    "UnsupportedCrackGeometryError",
    "crack_set",
    "infinite_plate_stress_intensity",
    "linear_elastic_fracture_material",
    "remote_stress",
    "segment",
    "stress_intensity_report",
    "verify_stress_intensity",
]
