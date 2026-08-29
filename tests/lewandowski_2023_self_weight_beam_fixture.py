"""Fail-closed external fixture for the Lewandowski et al. beam.

The source problem is the finite-strain elastoplastic self-weight beam from
Lewandowski et al. (2023), Section 6.1, and its public MGIS/FEniCS
implementation.  This module deliberately contains no digitised answer.  A
promotion run must provide a curve produced independently from the pinned
upstream implementation; otherwise the assessment remains ``incomplete``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


UPSTREAM_COMMIT = "cb43561d5e36a9ef691ad2c308261448cef44e29"
UPSTREAM_SOLVER_SHA256 = (
    "b2fdebe7edf9e10617f9cd4f350bfab119db9ca9441242959e5bf366a105e0d5"
)
UPSTREAM_BEHAVIOUR_SHA256 = (
    "17456bf1b593eec205f7cc7fb06483fdaf67ec7415713aeb27f33db57bef1393"
)

# These are AgentFEM promotion contracts, not tolerances published by the
# authors.  A caller may tighten them but cannot relax them.
MAXIMUM_NORMALIZED_RMS_ERROR = 0.03
MAXIMUM_NORMALIZED_MAX_ERROR = 0.05


@dataclass(frozen=True)
class SourceArtifact:
    """Immutable identity for one public upstream source artifact."""

    role: str
    url: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class Lewandowski2023BeamDefinition:
    """Scientific inputs shared by the paper and public executable demo."""

    length: float = 1.0
    width: float = 0.04
    height: float = 0.1
    young: float = 210.0e9
    poisson: float = 0.3
    yield_stress: float = 250.0e6
    hardening_modulus: float = 1.0e6
    maximum_body_force: float = 50.0e6
    subdivisions: tuple[int, int, int] = (30, 5, 8)
    increments: int = 30
    displacement_degree: int = 2
    observer: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def summary(self) -> dict[str, object]:
        return {
            "geometry_m": {
                "length": self.length,
                "width": self.width,
                "height": self.height,
            },
            "material_pa": {
                "young": self.young,
                "poisson": self.poisson,
                "yield_stress": self.yield_stress,
                "hardening_modulus": self.hardening_modulus,
            },
            "loading": {
                "body_force_at_unit_factor_n_per_m3": (
                    0.0,
                    0.0,
                    -self.maximum_body_force,
                ),
                "left_face": "fully_fixed",
                "right_face": "ux_zero_symmetry",
            },
            "discretization": {
                "cell": "tetrahedron",
                "box_subdivisions": self.subdivisions,
                "displacement_degree": self.displacement_degree,
                "increments": self.increments,
            },
            "observer": {
                "point_m": self.observer,
                "quantity": "downward_displacement",
                "source": "public_MGIS_FEniCS_script",
            },
        }


DEFINITION = Lewandowski2023BeamDefinition()

UPSTREAM_ARTIFACTS = (
    SourceArtifact(
        role="legacy_fenics_mgis_driver",
        url=(
            "https://gitlab.enpc.fr/navier-fenics/mgis-fenics-demos/-/raw/"
            f"{UPSTREAM_COMMIT}/demos/finite_strain_elastoplasticity/"
            "mgis_fenics_finite_strain_elastoplasticity.py"
        ),
        sha256=UPSTREAM_SOLVER_SHA256,
        size_bytes=7905,
    ),
    SourceArtifact(
        role="mfront_behaviour",
        url=(
            "https://gitlab.enpc.fr/navier-fenics/mgis-fenics-demos/-/raw/"
            f"{UPSTREAM_COMMIT}/demos/finite_strain_elastoplasticity/"
            "LogarithmicStrainPlasticity.mfront"
        ),
        sha256=UPSTREAM_BEHAVIOUR_SHA256,
        size_bytes=577,
    ),
)

REQUIRED_PROMOTION_EVIDENCE = (
    "independent_reference_execution",
    "observer_reconciled",
    "candidate_mesh_converged",
    "candidate_increment_converged",
    "serial_mpi_equivalent",
    "restart_equivalent",
)


def _curve(load_factors, displacements, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    load = np.asarray(load_factors, dtype=float)
    displacement = np.asarray(displacements, dtype=float)
    if load.ndim != 1 or displacement.ndim != 1 or load.shape != displacement.shape:
        raise ValueError(f"{label} load and displacement must be equal 1D arrays.")
    if load.size < 5 or not np.all(np.isfinite(load)) or not np.all(np.isfinite(displacement)):
        raise ValueError(f"{label} curve must contain at least five finite points.")
    if not np.all(np.diff(load) > 0.0):
        raise ValueError(f"{label} load factors must be strictly increasing.")
    if not np.isclose(load[0], 0.0, atol=1.0e-12) or not np.isclose(
        load[-1], 1.0, atol=1.0e-12
    ):
        raise ValueError(f"{label} curve must span load factors zero through one.")
    return load, displacement


def assess_external_curve(
    *,
    candidate_load_factors,
    candidate_displacements,
    reference_load_factors=None,
    reference_displacements=None,
    reference_source_commit: str | None = None,
    reference_solver_sha256: str | None = None,
    reference_behaviour_sha256: str | None = None,
    reference_curve_sha256: str | None = None,
    declared_reference_curve_sha256: str | None = None,
    convergence_evidence: dict[str, bool] | None = None,
    maximum_normalized_rms_error: float = MAXIMUM_NORMALIZED_RMS_ERROR,
    maximum_normalized_max_error: float = MAXIMUM_NORMALIZED_MAX_ERROR,
) -> dict[str, object]:
    """Compare an AgentFEM curve with an independently executed source curve.

    The assessment cannot be promoted using a plot digitised from the paper or
    values emitted by the AgentFEM candidate itself.  The pinned source hashes,
    independent execution, observer reconciliation, convergence, MPI and
    restart evidence are all mandatory.
    """

    rms_limit = float(maximum_normalized_rms_error)
    max_limit = float(maximum_normalized_max_error)
    if not 0.0 < rms_limit <= MAXIMUM_NORMALIZED_RMS_ERROR:
        raise ValueError("The normalized RMS contract may be tightened, not relaxed.")
    if not 0.0 < max_limit <= MAXIMUM_NORMALIZED_MAX_ERROR:
        raise ValueError("The normalized maximum contract may be tightened, not relaxed.")

    candidate_load, candidate_u = _curve(
        candidate_load_factors,
        candidate_displacements,
        label="candidate",
    )
    evidence = {} if convergence_evidence is None else dict(convergence_evidence)
    invalid = tuple(name for name, value in evidence.items() if type(value) is not bool)
    if invalid:
        raise TypeError(
            "convergence_evidence values must be bool; invalid keys: "
            + ", ".join(sorted(invalid))
        )

    missing = [name for name in REQUIRED_PROMOTION_EVIDENCE if not evidence.get(name, False)]
    source_identity_matches = (
        reference_source_commit == UPSTREAM_COMMIT
        and reference_solver_sha256 == UPSTREAM_SOLVER_SHA256
        and reference_behaviour_sha256 == UPSTREAM_BEHAVIOUR_SHA256
    )
    if not source_identity_matches:
        missing.append("pinned_reference_source_identity")
    curve_digest = "" if reference_curve_sha256 is None else str(reference_curve_sha256)
    declared_curve_digest = (
        ""
        if declared_reference_curve_sha256 is None
        else str(declared_reference_curve_sha256)
    )
    curve_identity_matches = (
        len(curve_digest) == 64
        and all(character in "0123456789abcdef" for character in curve_digest.lower())
        and curve_digest == declared_curve_digest
    )
    if not curve_identity_matches:
        missing.append("reference_curve_content_identity")

    if reference_load_factors is None or reference_displacements is None:
        missing.append("independent_reference_curve")
        return {
            "schema": "agentfem.external-curve-assessment.v1",
            "benchmark": "lewandowski_2023_self_weight_beam",
            "comparison_class": "cross_formulation_structure_level",
            "status": "incomplete",
            "accepted": False,
            "missing_evidence": tuple(dict.fromkeys(missing)),
            "candidate_points": int(candidate_load.size),
            "errors": None,
            "reference_curve_sha256": reference_curve_sha256,
        }

    reference_load, reference_u = _curve(
        reference_load_factors,
        reference_displacements,
        label="reference",
    )
    candidate_on_reference = np.interp(reference_load, candidate_load, candidate_u)
    scale = float(np.max(np.abs(reference_u)))
    if scale <= np.finfo(float).eps:
        raise ValueError("The external reference displacement scale must be non-zero.")
    difference = candidate_on_reference - reference_u
    rms_error = float(np.sqrt(np.mean(difference**2)) / scale)
    maximum_error = float(np.max(np.abs(difference)) / scale)
    curve_passed = rms_error <= rms_limit and maximum_error <= max_limit

    if not curve_passed:
        status = "failed"
    elif missing:
        status = "incomplete"
    else:
        status = "accepted"
    return {
        "schema": "agentfem.external-curve-assessment.v1",
        "benchmark": "lewandowski_2023_self_weight_beam",
        "comparison_class": "cross_formulation_structure_level",
        "status": status,
        "accepted": status == "accepted",
        "missing_evidence": tuple(dict.fromkeys(missing)),
        "candidate_points": int(candidate_load.size),
        "reference_points": int(reference_load.size),
        "errors": {
            "normalized_rms": rms_error,
            "normalized_maximum": maximum_error,
        },
        "reference_curve_sha256": reference_curve_sha256,
        "contracts": {
            "maximum_normalized_rms": rms_limit,
            "maximum_normalized_maximum": max_limit,
            "origin": "AgentFEM_project_promotion_contract_not_author_tolerance",
        },
    }
