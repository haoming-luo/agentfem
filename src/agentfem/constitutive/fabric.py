# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral constitutive foundation for woven reinforcement surfaces.

This module deliberately stops at local surface mechanics.  It supplies the
non-orthogonal yarn kinematics and independently testable tension, trellising,
and bending channels needed by a future finite-rotation fibrous-shell Step;
it does not claim that a shell element, tool contact, or forming process has
been FEM-integrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from agentfem.materials.orientations import FiberFrame


@runtime_checkable
class SurfaceConstitutive(Protocol):
    """Extension contract for a local surface constitutive response."""

    def evaluate(self, deformation_gradient, *, curvature=None): ...

    def as_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class TabulatedResponse:
    """Piecewise-linear scalar constitutive channel with an energy primitive."""

    abscissa: np.ndarray
    ordinate: np.ndarray
    name: str = "response"
    symmetry: str = "none"
    extrapolation: str = "error"

    def __post_init__(self) -> None:
        x = np.asarray(self.abscissa, dtype=float)
        y = np.asarray(self.ordinate, dtype=float)
        symmetry = str(self.symmetry).strip().lower()
        extrapolation = str(self.extrapolation).strip().lower()
        if x.ndim != 1 or y.ndim != 1 or x.size < 2 or x.size != y.size:
            raise ValueError("TabulatedResponse requires equal 1D arrays with at least two points.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(np.diff(x) <= 0.0):
            raise ValueError("TabulatedResponse coordinates must be finite and strictly increasing.")
        if symmetry not in {"none", "odd"}:
            raise ValueError("TabulatedResponse.symmetry must be 'none' or 'odd'.")
        if symmetry == "odd" and (abs(x[0]) > 1.0e-14 or abs(y[0]) > 1.0e-14 or np.any(x < 0.0)):
            raise ValueError("An odd TabulatedResponse must start at (0, 0) and use nonnegative abscissae.")
        if extrapolation not in {"error", "constant", "linear"}:
            raise ValueError("TabulatedResponse.extrapolation must be error, constant, or linear.")
        object.__setattr__(self, "abscissa", x)
        object.__setattr__(self, "ordinate", y)
        object.__setattr__(self, "symmetry", symmetry)
        object.__setattr__(self, "extrapolation", extrapolation)

    def _magnitude_and_sign(self, value: float) -> tuple[float, float]:
        if self.symmetry == "odd":
            return abs(float(value)), -1.0 if value < 0.0 else 1.0
        return float(value), 1.0

    def value(self, coordinate: float) -> float:
        x, sign = self._magnitude_and_sign(float(coordinate))
        if self.extrapolation == "error" and (x < self.abscissa[0] or x > self.abscissa[-1]):
            raise ValueError(
                f"Constitutive channel {self.name!r} covers "
                f"[{self.abscissa[0]:g}, {self.abscissa[-1]:g}]."
            )
        if x < self.abscissa[0]:
            value = self.ordinate[0] if self.extrapolation == "constant" else self._linear(x, 0)
        elif x > self.abscissa[-1]:
            value = self.ordinate[-1] if self.extrapolation == "constant" else self._linear(x, -2)
        else:
            value = float(np.interp(x, self.abscissa, self.ordinate))
        return sign * float(value)

    def tangent(self, coordinate: float) -> float:
        x, _ = self._magnitude_and_sign(float(coordinate))
        index = int(np.searchsorted(self.abscissa, x, side="right") - 1)
        index = min(max(index, 0), self.abscissa.size - 2)
        if self.extrapolation == "constant" and (x < self.abscissa[0] or x > self.abscissa[-1]):
            return 0.0
        if self.extrapolation == "error" and (x < self.abscissa[0] or x > self.abscissa[-1]):
            self.value(coordinate)
        return float(
            (self.ordinate[index + 1] - self.ordinate[index])
            / (self.abscissa[index + 1] - self.abscissa[index])
        )

    def energy(self, coordinate: float) -> float:
        """Return the signed-path integral from zero to ``coordinate``."""

        coordinate = float(coordinate)
        if coordinate == 0.0:
            return 0.0
        lower = min(0.0, coordinate)
        upper = max(0.0, coordinate)
        nodes = [lower]
        nodes.extend(float(item) for item in self.abscissa if lower < item < upper)
        if self.symmetry == "odd":
            nodes.extend(float(-item) for item in self.abscissa if lower < -item < upper)
        nodes.append(upper)
        nodes = sorted(set(nodes))
        integral = 0.0
        for left, right in zip(nodes[:-1], nodes[1:], strict=True):
            integral += 0.5 * (self.value(left) + self.value(right)) * (right - left)
        return float(integral if coordinate > 0.0 else -integral)

    def _linear(self, coordinate: float, index: int) -> float:
        x0, x1 = self.abscissa[index], self.abscissa[index + 1]
        y0, y1 = self.ordinate[index], self.ordinate[index + 1]
        return float(y0 + (y1 - y0) * (coordinate - x0) / (x1 - x0))

    def ufl_value(self, coordinate):
        """Return the piecewise-linear response as a differentiable UFL value.

        Symbolic finite-element fields cannot raise only at quadrature points
        after compilation. Global use therefore requires an explicit constant
        or linear extrapolation policy.
        """

        import ufl

        if self.extrapolation == "error":
            raise ValueError(
                f"Constitutive channel {self.name!r} needs explicit extrapolation "
                "for a global finite-element field."
            )
        selected = abs(coordinate) if self.symmetry == "odd" else coordinate
        value = self._ufl_interpolant(selected)
        if self.symmetry == "odd":
            return ufl.conditional(ufl.lt(coordinate, 0.0), -value, value)
        return value

    def ufl_energy(self, coordinate):
        """Return the UFL primitive consistent with the symbolic response."""

        if self.extrapolation == "error":
            raise ValueError(
                f"Constitutive channel {self.name!r} needs explicit extrapolation "
                "for a global finite-element field."
            )
        selected = abs(coordinate) if self.symmetry == "odd" else coordinate
        primitive = self._ufl_primitive(selected)
        zero = self._primitive_value(0.0)
        return primitive - zero

    def _ufl_interpolant(self, coordinate):
        import ufl

        slopes = np.diff(self.ordinate) / np.diff(self.abscissa)
        if self.extrapolation == "constant":
            below = float(self.ordinate[0])
            above = float(self.ordinate[-1])
        else:
            below = float(self.ordinate[0]) + float(slopes[0]) * (
                coordinate - float(self.abscissa[0])
            )
            above = float(self.ordinate[-1]) + float(slopes[-1]) * (
                coordinate - float(self.abscissa[-1])
            )
        expression = above
        for index in range(self.abscissa.size - 2, -1, -1):
            segment = float(self.ordinate[index]) + float(slopes[index]) * (
                coordinate - float(self.abscissa[index])
            )
            expression = ufl.conditional(
                ufl.le(coordinate, float(self.abscissa[index + 1])),
                segment,
                expression,
            )
        return ufl.conditional(
            ufl.lt(coordinate, float(self.abscissa[0])),
            below,
            expression,
        )

    def _primitive_value(self, coordinate: float) -> float:
        value = float(coordinate)
        cumulative = np.zeros(self.abscissa.size)
        cumulative[1:] = np.cumsum(
            0.5 * (self.ordinate[:-1] + self.ordinate[1:]) * np.diff(self.abscissa)
        )
        slopes = np.diff(self.ordinate) / np.diff(self.abscissa)
        if value < self.abscissa[0]:
            delta = value - float(self.abscissa[0])
            slope = 0.0 if self.extrapolation == "constant" else float(slopes[0])
            return float(self.ordinate[0] * delta + 0.5 * slope * delta**2)
        if value > self.abscissa[-1]:
            delta = value - float(self.abscissa[-1])
            slope = 0.0 if self.extrapolation == "constant" else float(slopes[-1])
            return float(
                cumulative[-1] + self.ordinate[-1] * delta + 0.5 * slope * delta**2
            )
        index = min(
            max(int(np.searchsorted(self.abscissa, value, side="right") - 1), 0),
            self.abscissa.size - 2,
        )
        delta = value - float(self.abscissa[index])
        return float(
            cumulative[index]
            + self.ordinate[index] * delta
            + 0.5 * slopes[index] * delta**2
        )

    def _ufl_primitive(self, coordinate):
        import ufl

        slopes = np.diff(self.ordinate) / np.diff(self.abscissa)
        cumulative = np.zeros(self.abscissa.size)
        cumulative[1:] = np.cumsum(
            0.5 * (self.ordinate[:-1] + self.ordinate[1:]) * np.diff(self.abscissa)
        )

        def segment(index: int):
            delta = coordinate - float(self.abscissa[index])
            return (
                float(cumulative[index])
                + float(self.ordinate[index]) * delta
                + 0.5 * float(slopes[index]) * delta**2
            )

        delta_below = coordinate - float(self.abscissa[0])
        below_slope = 0.0 if self.extrapolation == "constant" else float(slopes[0])
        below = (
            float(self.ordinate[0]) * delta_below
            + 0.5 * below_slope * delta_below**2
        )
        delta_above = coordinate - float(self.abscissa[-1])
        above_slope = 0.0 if self.extrapolation == "constant" else float(slopes[-1])
        expression = (
            float(cumulative[-1])
            + float(self.ordinate[-1]) * delta_above
            + 0.5 * above_slope * delta_above**2
        )
        for index in range(self.abscissa.size - 2, -1, -1):
            expression = ufl.conditional(
                ufl.le(coordinate, float(self.abscissa[index + 1])),
                segment(index),
                expression,
            )
        return ufl.conditional(
            ufl.lt(coordinate, float(self.abscissa[0])),
            below,
            expression,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "tabulated_constitutive_response",
            "name": self.name,
            "abscissa": self.abscissa.tolist(),
            "ordinate": self.ordinate.tolist(),
            "symmetry": self.symmetry,
            "extrapolation": self.extrapolation,
        }


@dataclass(frozen=True)
class FabricKinematics:
    """Non-orthogonal warp/weft surface deformation measures."""

    warp_direction: np.ndarray
    weft_direction: np.ndarray
    warp_strain: float
    weft_strain: float
    shear_angle: float
    current_angle: float

    def as_dict(self) -> dict[str, object]:
        return {
            "warp_direction": self.warp_direction.tolist(),
            "weft_direction": self.weft_direction.tolist(),
            "warp_strain": self.warp_strain,
            "weft_strain": self.weft_strain,
            "shear_angle_radians": self.shear_angle,
            "current_angle_radians": self.current_angle,
        }


@dataclass(frozen=True)
class FabricSurfaceResponse:
    """Local generalized resultants, tangent, energy, and physical measures."""

    kinematics: FabricKinematics
    membrane_resultants: np.ndarray
    bending_moments: np.ndarray
    tangent: np.ndarray
    stored_energy: float

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "fabric_surface_response",
            "kinematics": self.kinematics.as_dict(),
            "membrane_resultants": self.membrane_resultants.tolist(),
            "bending_moments": self.bending_moments.tolist(),
            "generalized_order": ["warp", "weft", "trellising"],
            "tangent": self.tangent.tolist(),
            "stored_energy": self.stored_energy,
        }


@dataclass(frozen=True)
class FabricLayer:
    """One named reinforcement layer in a shared-kinematics stack.

    Layer orientation remains owned by the law's ``FiberFrame``. Repeated
    physical layers are represented by repeated, distinctly named layers;
    AgentFEM does not hide thickness or multiplicity in an ambiguous scalar.
    """

    name: str
    material: "DecoupledFabricSurface"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("FabricLayer.name must be non-empty.")
        if not isinstance(self.material, DecoupledFabricSurface):
            raise TypeError("FabricLayer.material must be DecoupledFabricSurface.")
        object.__setattr__(self, "name", name)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "fabric_layer",
            "name": self.name,
            "material": self.material.as_dict(),
        }


@dataclass(frozen=True)
class FabricLayerResponse:
    """Stable identity and local response of one stack layer."""

    layer_name: str
    response: FabricSurfaceResponse

    @property
    def stored_energy(self) -> float:
        return float(self.response.stored_energy)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "fabric_layer_response",
            "layer_name": self.layer_name,
            "stored_energy": self.stored_energy,
            "response": self.response.as_dict(),
        }


@dataclass(frozen=True)
class FabricStackResponse:
    """Per-layer response without inventing one ambiguous aggregate frame."""

    layers: tuple[FabricLayerResponse, ...]
    stored_energy: float

    def by_name(self, name: str) -> FabricLayerResponse:
        selected = str(name)
        for layer in self.layers:
            if layer.layer_name == selected:
                return layer
        raise KeyError(f"Unknown fabric layer {selected!r}.")

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "fabric_stack_response",
            "stored_energy": self.stored_energy,
            "layers": [layer.as_dict() for layer in self.layers],
            "aggregation": "energy_only; resultants retain layer frames",
        }


@dataclass(frozen=True)
class FabricStack:
    """Named layers sharing one surface deformation.

    This is the provider-neutral constitutive asset needed by multilayer
    forming.  It intentionally retains each layer's non-orthogonal frame and
    response instead of summing unlike ``(warp, weft, trellising)`` vectors.
    A future shell provider may consume ``membrane_energy_ufl`` directly.
    """

    name: str
    layers: tuple[FabricLayer, ...]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        layers = tuple(self.layers)
        if not name:
            raise ValueError("FabricStack.name must be non-empty.")
        if not layers:
            raise ValueError("FabricStack requires at least one layer.")
        if not all(isinstance(layer, FabricLayer) for layer in layers):
            raise TypeError("FabricStack.layers must contain only FabricLayer objects.")
        names = [layer.name for layer in layers]
        if len(names) != len(set(names)):
            raise ValueError("Fabric layer names must be unique within a stack.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "layers", layers)

    @property
    def has_bending(self) -> bool:
        return any(
            np.any(np.abs(layer.material.bending_stiffness) > 1.0e-14)
            for layer in self.layers
        )

    def evaluate(
        self,
        deformation_gradient,
        *,
        curvature_by_layer: Mapping[str, object] | None = None,
    ) -> FabricStackResponse:
        curvatures = {} if curvature_by_layer is None else dict(curvature_by_layer)
        unknown = set(curvatures).difference(layer.name for layer in self.layers)
        if unknown:
            raise KeyError(f"Unknown curvature layer names: {sorted(unknown)!r}.")
        responses = tuple(
            FabricLayerResponse(
                layer_name=layer.name,
                response=layer.material.evaluate(
                    deformation_gradient,
                    curvature=curvatures.get(layer.name),
                ),
            )
            for layer in self.layers
        )
        return FabricStackResponse(
            layers=responses,
            stored_energy=float(sum(item.stored_energy for item in responses)),
        )

    def membrane_energy_ufl(self, displacement):
        energies = [
            layer.material.membrane_energy_ufl(displacement)
            for layer in self.layers
        ]
        total = energies[0]
        for energy in energies[1:]:
            total += energy
        return total

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "fabric_stack",
            "name": self.name,
            "layers": [layer.as_dict() for layer in self.layers],
            "kinematics": "shared_surface_deformation",
            "resultant_policy": "retain_per_layer_frames",
            "maturity": "local_constitutive_foundation",
        }


@dataclass(frozen=True)
class FabricFormingAssessment:
    """Dimensionless utilization report for declared forming limits."""

    utilization: Mapping[str, float]
    governing_mode: str
    maximum_utilization: float
    accepted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "fabric_forming_assessment",
            "utilization": dict(self.utilization),
            "governing_mode": self.governing_mode,
            "maximum_utilization": self.maximum_utilization,
            "accepted": self.accepted,
            "interpretation": "screening_against_user_declared_limits",
        }


@dataclass(frozen=True)
class FabricFormingLimits:
    """User-declared forming limits, separate from constitutive calibration.

    These limits screen a computed state.  They do not by themselves predict
    wrinkling, locking, contact, or manufacturing defects.
    """

    warp_tensile_strain: float | None = None
    weft_tensile_strain: float | None = None
    trellising_angle_radians: float | None = None
    in_plane_curvature: float | None = None
    normal_curvature: float | None = None

    def __post_init__(self) -> None:
        names = (
            "warp_tensile_strain",
            "weft_tensile_strain",
            "trellising_angle_radians",
            "in_plane_curvature",
            "normal_curvature",
        )
        selected = 0
        for name in names:
            value = getattr(self, name)
            if value is None:
                continue
            selected += 1
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite when supplied.")
            object.__setattr__(self, name, float(value))
        if not selected:
            raise ValueError("FabricFormingLimits requires at least one limit.")

    def assess(
        self,
        response: FabricSurfaceResponse,
        *,
        warp_bending=None,
        weft_bending=None,
    ) -> FabricFormingAssessment:
        values: dict[str, float] = {}
        kin = response.kinematics
        if self.warp_tensile_strain is not None:
            values["warp_tension"] = max(0.0, kin.warp_strain) / self.warp_tensile_strain
        if self.weft_tensile_strain is not None:
            values["weft_tension"] = max(0.0, kin.weft_strain) / self.weft_tensile_strain
        if self.trellising_angle_radians is not None:
            values["trellising"] = abs(kin.shear_angle) / self.trellising_angle_radians
        bending = tuple(item for item in (warp_bending, weft_bending) if item is not None)
        if (
            self.in_plane_curvature is not None
            or self.normal_curvature is not None
        ) and not bending:
            raise ValueError(
                "Curvature limits require warp_bending and/or weft_bending kinematics."
            )
        if self.in_plane_curvature is not None and bending:
            values["in_plane_bending"] = max(
                abs(float(item.in_plane_curvature_change)) for item in bending
            ) / self.in_plane_curvature
        if self.normal_curvature is not None and bending:
            values["normal_bending"] = max(
                abs(float(item.normal_curvature_change)) for item in bending
            ) / self.normal_curvature
        governing = max(values, key=values.get)
        maximum = float(values[governing])
        return FabricFormingAssessment(
            utilization=values,
            governing_mode=governing,
            maximum_utilization=maximum,
            accepted=maximum <= 1.0,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "fabric_forming_limits",
            "warp_tensile_strain": self.warp_tensile_strain,
            "weft_tensile_strain": self.weft_tensile_strain,
            "trellising_angle_radians": self.trellising_angle_radians,
            "in_plane_curvature": self.in_plane_curvature,
            "normal_curvature": self.normal_curvature,
            "curvature_unit": "inverse_length",
        }


@dataclass(frozen=True)
class FabricMembraneExpressions:
    """Symbolic observables used by the global woven-membrane provider."""

    warp_direction: object
    weft_direction: object
    warp_strain: object
    weft_strain: object
    shear_angle: object
    membrane_resultants: object
    stored_energy: object


@dataclass(frozen=True)
class DecoupledFabricSurface:
    """Independent yarn tension, trellising shear, and bending channels.

    Resultants are force per reference length; curvatures are supplied in the
    warp, weft, and twist order.  The decoupled bending law prevents membrane
    calibration from silently determining wrinkle resistance.
    """

    name: str
    frame: FiberFrame
    warp_tension: TabulatedResponse
    weft_tension: TabulatedResponse
    shear: TabulatedResponse
    bending_stiffness: np.ndarray
    tension_only: bool = True

    def __post_init__(self) -> None:
        bending = np.asarray(self.bending_stiffness, dtype=float)
        if bending.shape != (3, 3) or not np.all(np.isfinite(bending)):
            raise ValueError("bending_stiffness must be one finite 3x3 matrix.")
        if not np.allclose(bending, bending.T, atol=1.0e-12):
            raise ValueError("bending_stiffness must be symmetric.")
        if np.min(np.linalg.eigvalsh(bending)) < -1.0e-12:
            raise ValueError("bending_stiffness must be positive semidefinite.")
        if self.shear.symmetry != "odd":
            raise ValueError("Fabric trellising shear requires an odd response curve.")
        object.__setattr__(self, "bending_stiffness", bending)

    def evaluate(self, deformation_gradient, *, curvature=None) -> FabricSurfaceResponse:
        F = np.asarray(deformation_gradient, dtype=float)
        warp, weft, stretch_warp, stretch_weft = self.frame.convect(F)
        current_angle = float(np.arccos(np.clip(np.dot(warp, weft), -1.0, 1.0)))
        kinematics = FabricKinematics(
            warp_direction=warp,
            weft_direction=weft,
            warp_strain=stretch_warp - 1.0,
            weft_strain=stretch_weft - 1.0,
            shear_angle=self.frame.reference_angle - current_angle,
            current_angle=current_angle,
        )
        coordinates = np.array(
            [kinematics.warp_strain, kinematics.weft_strain, kinematics.shear_angle]
        )
        curves = (self.warp_tension, self.weft_tension, self.shear)
        resultants = np.empty(3)
        tangent = np.zeros((6, 6))
        stored_energy = 0.0
        for index, (coordinate, curve) in enumerate(zip(coordinates, curves, strict=True)):
            if self.tension_only and index < 2 and coordinate < 0.0:
                resultants[index] = 0.0
                continue
            resultants[index] = curve.value(float(coordinate))
            tangent[index, index] = curve.tangent(float(coordinate))
            stored_energy += curve.energy(float(coordinate))
        selected_curvature = np.zeros(3) if curvature is None else np.asarray(curvature, dtype=float)
        if selected_curvature.shape != (3,) or not np.all(np.isfinite(selected_curvature)):
            raise ValueError("curvature must be one finite 3-component vector.")
        moments = self.bending_stiffness @ selected_curvature
        tangent[3:, 3:] = self.bending_stiffness
        stored_energy += 0.5 * float(selected_curvature @ moments)
        return FabricSurfaceResponse(
            kinematics=kinematics,
            membrane_resultants=resultants,
            bending_moments=moments,
            tangent=tangent,
            stored_energy=stored_energy,
        )

    def membrane_expressions_ufl(self, displacement) -> FabricMembraneExpressions:
        """Return finite-kinematics membrane measures and resultants in UFL."""

        import ufl

        if self.frame.dimension != 2:
            raise ValueError(
                "The in-plane fabric membrane provider requires a 2D FiberFrame."
            )
        if tuple(getattr(displacement, "ufl_shape", ())) != (2,):
            raise ValueError(
                "The in-plane fabric membrane provider requires a 2D displacement."
            )
        F = ufl.Identity(2) + ufl.grad(displacement)
        warp0 = ufl.as_vector(self.frame.warp.tolist())
        weft0 = ufl.as_vector(self.frame.weft.tolist())
        warp = ufl.dot(F, warp0)
        weft = ufl.dot(F, weft0)
        stretch_warp = ufl.sqrt(ufl.inner(warp, warp))
        stretch_weft = ufl.sqrt(ufl.inner(weft, weft))
        cosine = ufl.inner(warp, weft) / (stretch_warp * stretch_weft)
        bounded_cosine = ufl.max_value(-1.0, ufl.min_value(1.0, cosine))
        shear_angle = self.frame.reference_angle - ufl.acos(bounded_cosine)
        warp_strain = stretch_warp - 1.0
        weft_strain = stretch_weft - 1.0

        def yarn_value(curve, strain):
            if not self.tension_only:
                return curve.ufl_value(strain)
            return ufl.conditional(
                ufl.ge(strain, 0.0),
                curve.ufl_value(strain),
                0.0,
            )

        def yarn_energy(curve, strain):
            if not self.tension_only:
                return curve.ufl_energy(strain)
            return ufl.conditional(
                ufl.ge(strain, 0.0),
                curve.ufl_energy(strain),
                0.0,
            )

        energy = (
            yarn_energy(self.warp_tension, warp_strain)
            + yarn_energy(self.weft_tension, weft_strain)
            + self.shear.ufl_energy(shear_angle)
        )
        return FabricMembraneExpressions(
            warp_direction=warp / stretch_warp,
            weft_direction=weft / stretch_weft,
            warp_strain=warp_strain,
            weft_strain=weft_strain,
            shear_angle=shear_angle,
            membrane_resultants=ufl.as_vector(
                (
                    yarn_value(self.warp_tension, warp_strain),
                    yarn_value(self.weft_tension, weft_strain),
                    self.shear.ufl_value(shear_angle),
                )
            ),
            stored_energy=energy,
        )

    def membrane_energy_ufl(self, displacement):
        """Finite-kinematics membrane energy for a two-dimensional UFL field."""

        return self.membrane_expressions_ufl(displacement).stored_energy

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "decoupled_fabric_surface",
            "name": self.name,
            "frame": self.frame.as_dict(),
            "warp_tension": self.warp_tension.as_dict(),
            "weft_tension": self.weft_tension.as_dict(),
            "shear": self.shear.as_dict(),
            "bending_stiffness": self.bending_stiffness.tolist(),
            "tension_only": bool(self.tension_only),
            "maturity": "experimental_fem_integrated",
            "fem_integration": "finite_kinematics_in_plane_membrane",
            "bending_integration": "local_kinematics_only",
        }


def tabulated_response(
    abscissa,
    ordinate,
    *,
    name: str = "response",
    symmetry: str = "none",
    extrapolation: str = "error",
) -> TabulatedResponse:
    return TabulatedResponse(
        abscissa=np.asarray(abscissa, dtype=float),
        ordinate=np.asarray(ordinate, dtype=float),
        name=name,
        symmetry=symmetry,
        extrapolation=extrapolation,
    )


def decoupled_fabric_surface(
    *,
    frame: FiberFrame,
    warp_tension: TabulatedResponse,
    weft_tension: TabulatedResponse,
    shear: TabulatedResponse,
    bending_stiffness,
    tension_only: bool = True,
    name: str = "fabric_surface",
) -> DecoupledFabricSurface:
    return DecoupledFabricSurface(
        name=name,
        frame=frame,
        warp_tension=warp_tension,
        weft_tension=weft_tension,
        shear=shear,
        bending_stiffness=np.asarray(bending_stiffness, dtype=float),
        tension_only=tension_only,
    )


def fabric_layer(
    material: DecoupledFabricSurface,
    *,
    name: str,
) -> FabricLayer:
    """Create one named layer for a shared-kinematics fabric stack."""

    return FabricLayer(
        name=name,
        material=material,
    )


def fabric_stack(
    layers,
    *,
    name: str = "fabric_stack",
) -> FabricStack:
    """Create a checked multilayer fabric asset with stable layer identities."""

    return FabricStack(name=name, layers=tuple(layers))


def fabric_forming_limits(
    *,
    warp_tensile_strain: float | None = None,
    weft_tensile_strain: float | None = None,
    trellising_angle: float | None = None,
    angle_unit: str = "degree",
    in_plane_curvature: float | None = None,
    normal_curvature: float | None = None,
) -> FabricFormingLimits:
    """Create forming-screening limits with an explicit angle unit."""

    selected_unit = str(angle_unit).strip().lower()
    if selected_unit not in {"degree", "degrees", "deg", "radian", "radians", "rad"}:
        raise ValueError("angle_unit must be degree or radian.")
    angle = trellising_angle
    if angle is not None and selected_unit in {"degree", "degrees", "deg"}:
        angle = float(np.deg2rad(angle))
    return FabricFormingLimits(
        warp_tensile_strain=warp_tensile_strain,
        weft_tensile_strain=weft_tensile_strain,
        trellising_angle_radians=angle,
        in_plane_curvature=in_plane_curvature,
        normal_curvature=normal_curvature,
    )


def fabric_membrane_internal_virtual_work(
    displacement,
    test,
    material: DecoupledFabricSurface,
    *,
    measure=None,
):
    """Return the in-plane fabric membrane residual from stored energy."""

    import ufl

    selected_measure = ufl.dx if measure is None else measure
    potential = material.membrane_energy_ufl(displacement) * selected_measure
    return ufl.derivative(potential, displacement, test)


__all__ = [
    "DecoupledFabricSurface",
    "FabricFormingAssessment",
    "FabricFormingLimits",
    "FabricKinematics",
    "FabricLayer",
    "FabricLayerResponse",
    "FabricMembraneExpressions",
    "FabricStack",
    "FabricStackResponse",
    "FabricSurfaceResponse",
    "SurfaceConstitutive",
    "TabulatedResponse",
    "decoupled_fabric_surface",
    "fabric_forming_limits",
    "fabric_layer",
    "fabric_membrane_internal_virtual_work",
    "fabric_stack",
    "tabulated_response",
]
