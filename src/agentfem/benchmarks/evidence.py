"""Join scientific capability claims to executable benchmark evidence.

The audit deliberately answers a narrow question: does the benchmark registry
support the maturity that the capability catalog currently declares?  It does
not promote experimental functionality and it does not turn test count into a
claim of general engineering validity.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..constitutive.catalog import ConstitutiveCapability, capabilities
from .registry import BenchmarkSpec, list_benchmarks


_MATURITY_REQUIREMENTS = {
    "interface_contract": ("interface",),
    "curve_projection_verified": ("curve_projection",),
    "postprocessor": ("postprocess",),
    "material_point_verified": ("material_point",),
    "material_point_and_spectral_verified": ("material_point", "postprocess"),
    "material_point_experimental": ("material_point",),
    "experimental_global_patch": ("material_point", "finite_element"),
    "experimental_global_mpi_restart": (
        "material_point",
        "finite_element",
        "mpi",
        "restart",
    ),
    "fem_integrated_foundation": ("material_point", "finite_element"),
    "fem_integrated": ("finite_element",),
    "experimental_mixed_mode_global_lifecycle": ("finite_element",),
    "experimental_global_facet_consumer": ("finite_element",),
}


@dataclass(frozen=True)
class CapabilityEvidence:
    """Evidence supporting one declared constitutive maturity boundary."""

    capability: str
    maturity: str
    benchmark_ids: tuple[str, ...]
    demonstrated: tuple[str, ...]
    required: tuple[str, ...]
    gaps: tuple[str, ...]

    @property
    def meets_declared_maturity(self) -> bool:
        return not self.gaps

    def as_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "maturity": self.maturity,
            "benchmark_ids": self.benchmark_ids,
            "demonstrated": self.demonstrated,
            "required": self.required,
            "gaps": self.gaps,
            "meets_declared_maturity": self.meets_declared_maturity,
        }


def _dimensions(spec: BenchmarkSpec) -> set[str]:
    dimensions = set(spec.evidence)
    level = spec.level.lower()
    status = spec.status.lower()
    if spec.automated_test.strip():
        dimensions.add("automated")
    if "material_point" in level:
        dimensions.add("material_point")
    if "finite_element" in level or "global_facet" in level:
        dimensions.add("finite_element")
    if "workflow" in level or "global_lifecycle" in level:
        dimensions.add("workflow")
    if "interface" in level:
        dimensions.add("interface")
    if "curve_projection" in level:
        dimensions.add("curve_projection")
    if "postprocess" in level:
        dimensions.add("postprocess")
    joined = " ".join((level, status, spec.reference, spec.automated_test)).lower()
    if "mpi" in joined or "distributed" in joined or "two-rank" in joined:
        dimensions.add("mpi")
    if "external" in joined:
        dimensions.add("external_gate_defined")
        non_demonstrated = (
            "incomplete",
            "pending",
            "not_run",
            "not_promoted",
            "failed",
        )
        if not any(marker in status for marker in non_demonstrated):
            dimensions.add("external")
        else:
            dimensions.discard("external")
    if "restart" in joined or "checkpoint" in joined:
        dimensions.add("restart")
    if status == "release_regression":
        dimensions.add("release_regression")
    return dimensions


def capability_evidence(
    capability: ConstitutiveCapability,
    *,
    benchmarks: tuple[BenchmarkSpec, ...] | None = None,
) -> CapabilityEvidence:
    """Audit one catalog capability against the benchmark registry."""

    selected = (
        list_benchmarks(capability=capability.name)
        if benchmarks is None
        else tuple(item for item in benchmarks if item.capability == capability.name)
    )
    demonstrated: set[str] = set()
    for item in selected:
        demonstrated.update(_dimensions(item))
    required = _MATURITY_REQUIREMENTS.get(capability.maturity, ("automated",))
    gaps = tuple(item for item in required if item not in demonstrated)
    return CapabilityEvidence(
        capability=capability.name,
        maturity=capability.maturity,
        benchmark_ids=tuple(item.identifier for item in selected),
        demonstrated=tuple(sorted(demonstrated)),
        required=tuple(required),
        gaps=gaps,
    )


def audit_capability_evidence() -> tuple[CapabilityEvidence, ...]:
    """Return a stable, machine-readable audit for the whole catalog."""

    return tuple(capability_evidence(item) for item in capabilities())
