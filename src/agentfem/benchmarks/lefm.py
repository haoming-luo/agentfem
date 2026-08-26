"""Executable linear-elastic fracture-mechanics reference problems."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import basix
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np
import ufl

from ..constitutive import isotropic_elastic
from ..constitutive import elasticity
from ..fields import displacement
from ..fracture_fem import dolfinx_interaction_integral_report
from ..fracture_geometry import (
    StressIntensityReference,
    StressIntensityReport,
    StressIntensityVerification,
    crack_set,
    infinite_plate_stress_intensity,
    linear_elastic_fracture_material,
    remote_stress,
    segment,
    verify_stress_intensity,
)
from ..mesh import face
from ..models import create
from ..studies import static_solid


@dataclass(frozen=True)
class CenterCrackLEFMBenchmark:
    """One solved center-crack model and its independently extracted evidence."""

    result: object
    stress_intensity: StressIntensityReport
    reference: StressIntensityReference
    verification: StressIntensityVerification
    half_crack_length: float
    remote_stress: float

    @property
    def status(self) -> str:
        if getattr(self.result, "status", None) != "completed":
            return "failed"
        return self.verification.status

    def summary(self) -> dict[str, object]:
        return {
            "kind": "center_crack_lefm_benchmark",
            "status": self.status,
            "half_crack_length": self.half_crack_length,
            "remote_stress": self.remote_stress,
            "stress_intensity": self.stress_intensity.summary(),
            "reference": self.reference.summary(),
            "verification": self.verification.summary(),
        }


def center_crack_lefm_mesh(
    *,
    half_crack_length: float = 1.0,
    half_width: float = 8.0,
    half_height: float = 8.0,
    comm=MPI.COMM_SELF,
):
    """Build the serial, conforming split mesh used by the LEFM benchmark.

    The coincident nodes on the two crack faces are topologically independent;
    the crack tips remain shared. This is a verification fixture, not a general
    crack-meshing API.
    """

    length = float(half_crack_length)
    width = float(half_width)
    height = float(half_height)
    if not all(isfinite(value) and value > 0.0 for value in (length, width, height)):
        raise ValueError("Crack length and plate half-dimensions must be positive.")
    if width < 4.0 * length or height < 4.0 * length:
        raise ValueError("The LEFM fixture requires half-dimensions >= 4a.")
    if int(comm.size) != 1:
        raise ValueError(
            "center_crack_lefm_mesh is a serial verification fixture; "
            "use a partitioned imported mesh for MPI studies."
        )

    a = length
    x = np.unique(
        np.concatenate(
            (
                np.linspace(-width, -2.0 * a, 13),
                np.linspace(-2.0 * a, -1.5 * a, 6),
                np.linspace(-1.5 * a, -0.5 * a, 21),
                np.linspace(-0.5 * a, 0.5 * a, 11),
                np.linspace(0.5 * a, 1.5 * a, 21),
                np.linspace(1.5 * a, 2.0 * a, 6),
                np.linspace(2.0 * a, width, 13),
            )
        )
    )
    y = np.unique(
        np.concatenate(
            (
                np.linspace(-height, -2.0 * a, 13),
                np.linspace(-2.0 * a, -0.5 * a, 16),
                np.linspace(-0.5 * a, 0.5 * a, 21),
                np.linspace(0.5 * a, 2.0 * a, 16),
                np.linspace(2.0 * a, height, 13),
            )
        )
    )
    nx = len(x)
    coordinates = [(x_value, y_value) for y_value in y for x_value in x]

    def base(i, j):
        return j * nx + i

    crack_row = int(np.flatnonzero(np.isclose(y, 0.0))[0])
    duplicates = {}
    for i, value in enumerate(x):
        if -a < value < a:
            duplicates[i] = len(coordinates)
            coordinates.append((value, 0.0))

    def node(i, j, *, lower):
        if lower and j == crack_row and i in duplicates:
            return duplicates[i]
        return base(i, j)

    cells = []
    for j in range(len(y) - 1):
        lower = j < crack_row
        for i in range(nx - 1):
            n00 = node(i, j, lower=lower)
            n10 = node(i + 1, j, lower=lower)
            n01 = node(i, j + 1, lower=lower)
            n11 = node(i + 1, j + 1, lower=lower)
            cells.extend(((n00, n10, n11), (n00, n11, n01)))

    coordinate_element = ufl.Mesh(
        basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    )
    return dolfinx_mesh.create_mesh(
        comm,
        np.asarray(cells, dtype=np.int64),
        coordinate_element,
        np.asarray(coordinates, dtype=float),
    )


def center_crack_mode_i_benchmark(
    *,
    young_modulus: float = 1000.0,
    poisson_ratio: float = 0.25,
    half_crack_length: float = 1.0,
    half_width: float = 8.0,
    half_height: float = 8.0,
    remote_strain: float = 1.0e-3,
    relative_tolerance: float = 0.05,
) -> CenterCrackLEFMBenchmark:
    """Solve and verify a finite-plate Mode-I crack with the public workflow.

    The analytical comparison is the infinite-plate center-crack solution. The
    finite plate is deliberately large relative to the crack, but the geometry
    approximation remains part of the reported verification error.
    """

    modulus = float(young_modulus)
    ratio = float(poisson_ratio)
    strain = float(remote_strain)
    tolerance = float(relative_tolerance)
    if not isfinite(strain) or strain <= 0.0:
        raise ValueError("remote_strain must be finite and tensile (positive).")
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive.")
    lefm_material = linear_elastic_fracture_material(
        young_modulus=modulus,
        poisson_ratio=ratio,
        assumption="plane_stress",
    )
    domain = center_crack_lefm_mesh(
        half_crack_length=half_crack_length,
        half_width=half_width,
        half_height=half_height,
    )
    study = static_solid(dimension=2, assumption="plane_stress")
    model = create(study=study, mesh=domain, name="center_crack_mode_i")
    target = model.field(displacement(domain, degree=2))
    material = isotropic_elastic(
        young=modulus,
        poisson=ratio,
        density=1.0,
        name="LEFM solid",
    )
    model.material(material)
    model.fix(
        target,
        on=face(domain, axis="y", value=half_height, name="top", tag=1),
        component=1,
        value=float(half_height) * strain,
    )
    model.fix(
        target,
        on=face(domain, axis="y", value=-half_height, name="bottom", tag=2),
        component=1,
        value=-float(half_height) * strain,
    )
    model.fix(
        target,
        location=lambda x: np.isclose(x[0], 0.0),
        component=0,
        value=0.0,
    )
    result = model.step(target=target).solve_result()

    a = float(half_crack_length)
    cracks = crack_set(segment("center", start=(-a, 0.0), end=(a, 0.0)))
    stress = elasticity.isotropic_stress(target.value, material, study=study)
    report = dolfinx_interaction_integral_report(
        domain,
        stress,
        ufl.grad(target.value),
        crack=cracks,
        tip_id="center:end",
        material=lefm_material,
        integration_radii=(0.2 * a, 0.3 * a, 0.4 * a),
        quadrature_degree=8,
        relative_path_tolerance=0.03,
        metadata={"benchmark": "center_crack_mode_i"},
    )
    applied_stress = modulus * strain
    reference = infinite_plate_stress_intensity(
        crack=cracks,
        tip_id="center:end",
        stress=remote_stress(yy=applied_stress),
        material=lefm_material,
    )
    verification = verify_stress_intensity(
        report,
        reference,
        relative_tolerance=tolerance,
    )
    return CenterCrackLEFMBenchmark(
        result=result,
        stress_intensity=report,
        reference=reference,
        verification=verification,
        half_crack_length=a,
        remote_stress=applied_stress,
    )


__all__ = [
    "CenterCrackLEFMBenchmark",
    "center_crack_lefm_mesh",
    "center_crack_mode_i_benchmark",
]
