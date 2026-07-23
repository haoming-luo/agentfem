"""Absorbing/impedance boundary terms for transient finite-element problems."""

from __future__ import annotations

from dataclasses import dataclass

import ufl


def scalar_viscous_boundary_form(velocity, test_function, measure, impedance):
    """Isotropic viscous boundary term, ``Z * v . w``."""

    return impedance * ufl.inner(velocity, test_function) * measure


def normal_tangential_viscous_boundary_form(
    velocity,
    test_function,
    measure,
    *,
    normal,
    normal_impedance,
    tangential_impedance,
):
    """Viscous boundary term with separate normal and tangential impedances."""

    v_normal = ufl.dot(velocity, normal)
    w_normal = ufl.dot(test_function, normal)
    v_tangent = velocity - v_normal * normal
    w_tangent = test_function - w_normal * normal
    return (
        normal_impedance * v_normal * w_normal
        + tangential_impedance * ufl.inner(v_tangent, w_tangent)
    ) * measure


@dataclass(frozen=True)
class ViscousAbsorbingBoundary:
    """Common impedance-style absorbing boundary.

    This represents the same finite-element idea as a boundary damping matrix:
    a weak boundary term proportional to the velocity and virtual displacement.
    """

    measure: object
    normal_impedance: object
    tangential_impedance: object | None = None
    normal: object | None = None
    name: str = "viscous_absorbing_boundary"

    def form(self, velocity, test_function):
        """Return the UFL boundary form for this absorbing boundary."""

        if self.tangential_impedance is None:
            return scalar_viscous_boundary_form(
                velocity,
                test_function,
                self.measure,
                self.normal_impedance,
            )
        if self.normal is None:
            raise ValueError("normal is required when tangential_impedance is provided.")
        return normal_tangential_viscous_boundary_form(
            velocity,
            test_function,
            self.measure,
            normal=self.normal,
            normal_impedance=self.normal_impedance,
            tangential_impedance=self.tangential_impedance,
        )


def lysmer_kuhlemeyer_boundary(
    measure,
    *,
    density,
    pressure_wave_speed,
    shear_wave_speed=None,
    normal=None,
    mode: str = "normal_shear",
):
    """Create a Lysmer-Kuhlemeyer-style viscous absorbing boundary.

    ``scalar`` uses ``rho * cp`` on all components. ``normal_shear`` uses
    ``rho * cp`` in the normal direction and ``rho * cs`` tangentially.
    """

    if mode == "scalar":
        return ViscousAbsorbingBoundary(
            measure=measure,
            normal_impedance=density * pressure_wave_speed,
            name="lysmer_kuhlemeyer_scalar",
        )
    if mode == "normal_shear":
        if normal is None:
            raise ValueError("normal is required for normal_shear absorbing boundary.")
        if shear_wave_speed is None:
            raise ValueError("shear_wave_speed is required for normal_shear mode.")
        return ViscousAbsorbingBoundary(
            measure=measure,
            normal=normal,
            normal_impedance=density * pressure_wave_speed,
            tangential_impedance=density * shear_wave_speed,
            name="lysmer_kuhlemeyer_normal_shear",
        )
    raise ValueError(f"unknown absorbing boundary mode: {mode!r}")
