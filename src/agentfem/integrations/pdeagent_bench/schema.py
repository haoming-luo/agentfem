"""Strict input contract for the audited PDEAgent-Bench agent view."""

from __future__ import annotations

from collections.abc import Mapping

from agentfem.mesh.specs import SUPPORTED_GEOMETRIES


BENCHMARK_NAME = "PDEAgent-Bench"
BENCHMARK_SCHEMA = "pdeagent-bench.agent-view.v2"
BENCHMARK_COMMIT = "0ba9853f82a78196796fa4eeaf0951eb4c000a00"
SUPPORTED_FAMILIES = frozenset(
    {
        "poisson",
        "heat",
        "linear_elasticity",
        "helmholtz",
        "convection_diffusion",
        "reaction_diffusion",
        "wave",
    }
)


class BenchmarkContractError(ValueError):
    """A stable, machine-readable benchmark contract failure."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


def validate_case_spec(case_spec: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize one public agent-view case specification."""

    if not isinstance(case_spec, Mapping):
        raise BenchmarkContractError("AFM-PDEB-001", "case_spec must be a mapping")
    required = ("pde", "domain", "bc", "output")
    missing = [key for key in required if key not in case_spec]
    if missing:
        raise BenchmarkContractError(
            "AFM-PDEB-001", f"missing required fields: {', '.join(missing)}"
        )
    pde = _mapping(case_spec["pde"], "pde", "AFM-PDEB-001")
    domain = _mapping(case_spec["domain"], "domain", "AFM-PDEB-001")
    output = _mapping(case_spec["output"], "output", "AFM-PDEB-006")
    family = str(pde.get("type") or _equation_type(case_spec)).strip().lower()
    if family not in SUPPORTED_FAMILIES:
        raise BenchmarkContractError(
            "AFM-PDEB-008",
            f"PDE family {family!r} is not supported by this adapter; "
            f"supported families are {tuple(sorted(SUPPORTED_FAMILIES))}",
        )
    domain_type = str(domain.get("type", "")).strip().lower()
    if domain_type not in SUPPORTED_GEOMETRIES:
        raise BenchmarkContractError(
            "AFM-PDEB-004", f"unknown geometry type {domain_type!r}"
        )
    grid = _mapping(output.get("grid"), "output.grid", "AFM-PDEB-006")
    bbox = grid.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) not in {4, 6}:
        raise BenchmarkContractError(
            "AFM-PDEB-006", "output.grid.bbox must contain four or six values"
        )
    dimension = 3 if len(bbox) == 6 else 2
    shape_keys = ("nx", "ny", "nz")[:dimension]
    if any(int(grid.get(key, 0)) < 2 for key in shape_keys):
        raise BenchmarkContractError(
            "AFM-PDEB-006", f"output.grid requires positive {shape_keys}"
        )
    if family in {"heat", "reaction_diffusion", "wave"} or (
        family == "convection_diffusion" and "time" in pde
    ):
        time = pde.get("time")
        if not isinstance(time, Mapping) or "t_end" not in time:
            raise BenchmarkContractError(
                "AFM-PDEB-002", f"{family} requires pde.time.t_end"
            )
        scheme = str(time.get("scheme", "backward_euler")).lower().replace("-", "_")
        accepted = {"backward_euler", "implicit_euler"}
        if family == "reaction_diffusion":
            accepted.add("crank_nicolson")
        if family != "wave" and scheme not in accepted:
            raise BenchmarkContractError(
                "AFM-PDEB-002", f"unsupported {family} time scheme {scheme!r}"
            )
    normalized = dict(case_spec)
    normalized["pde"] = dict(pde)
    normalized["domain"] = dict(domain)
    normalized["bc"] = dict(_mapping(case_spec["bc"], "bc", "AFM-PDEB-005"))
    normalized["output"] = dict(output)
    normalized["_agentfem"] = {
        "schema": BENCHMARK_SCHEMA,
        "benchmark_commit": BENCHMARK_COMMIT,
        "pde_family": family,
        "dimension": dimension,
    }
    return normalized


def _equation_type(case_spec: Mapping[str, object]) -> str:
    classification = case_spec.get("pde_classification", {})
    if isinstance(classification, Mapping):
        return str(classification.get("equation_type", ""))
    return ""


def _mapping(value, name: str, code: str):
    if not isinstance(value, Mapping):
        raise BenchmarkContractError(code, f"{name} must be a mapping")
    return value
