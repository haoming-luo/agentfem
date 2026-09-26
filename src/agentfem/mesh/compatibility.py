# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit source-cell, solver-topology, and quality compatibility.

Reading connectivity is not equivalent to reproducing a source finite-element
formulation. This module describes only the neutral geometry route. Reduced
integration, hybrid variables, incompatible modes, shell directors, beam
sections, cohesive kinematics, and other formulation semantics remain the
responsibility of their dedicated adapters and providers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CellCompatibility:
    """One meshio-style source cell mapped to an AgentFEM solver topology."""

    source_cell_type: str
    topology: str | None
    dimension: int | None
    geometry_degree: int | None
    geometry_basis: str | None
    node_count: int | None
    import_maturity: str
    solver_scope: str
    quality_metric: str | None
    limitations: tuple[str, ...] = ()

    @property
    def solver_ready(self) -> bool:
        return self.import_maturity == "verified"

    def summary(self) -> dict[str, object]:
        return {
            "source_cell_type": self.source_cell_type,
            "topology": self.topology,
            "dimension": self.dimension,
            "geometry_degree": self.geometry_degree,
            "geometry_basis": self.geometry_basis,
            "node_count": self.node_count,
            "import_maturity": self.import_maturity,
            "solver_ready": self.solver_ready,
            "solver_scope": self.solver_scope,
            "quality_metric": self.quality_metric,
            "limitations": self.limitations,
        }


@dataclass(frozen=True)
class TopologyCapability:
    """One narrowly scoped, evidence-bearing runtime topology capability."""

    name: str
    status: str
    scope: str
    evidence: tuple[str, ...] = ()

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "verified": self.verified,
            "scope": self.scope,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class TopologyCompatibility:
    """Runtime solver-topology support independent of source element names.

    A DOLFINx topology says how cells are connected.  It does not select a
    beam, shell, reduced-integration, hybrid, cohesive, or other source
    formulation.  This record therefore answers only whether AgentFEM can
    inspect and quality-audit the runtime topology before a Step provider
    makes the formulation decision.
    """

    topology: str
    dimension: int | None
    runtime_maturity: str
    quality_metric: str | None
    source_cell_types: tuple[str, ...]
    capabilities: tuple[TopologyCapability, ...] = ()
    limitations: tuple[str, ...] = ()

    @property
    def inspectable(self) -> bool:
        return self.runtime_maturity in {"verified", "conditional"}

    @property
    def release_ready(self) -> bool:
        return self.runtime_maturity == "verified"

    def summary(self) -> dict[str, object]:
        return {
            "topology": self.topology,
            "dimension": self.dimension,
            "runtime_maturity": self.runtime_maturity,
            "inspectable": self.inspectable,
            "release_ready": self.release_ready,
            "quality_metric": self.quality_metric,
            "source_cell_types": self.source_cell_types,
            "capabilities": tuple(item.summary() for item in self.capabilities),
            "limitations": self.limitations,
        }


_NEUTRAL_SCOPE = (
    "neutral_geometry_and_named_sets; physical finite-element formulation "
    "must be selected independently"
)
_SOURCE_FORMULATION_LIMIT = (
    "Connectivity does not reproduce source reduced integration, hybrid "
    "variables, hourglass control, incompatible modes, or dedicated "
    "structural/interface kinematics."
)


def _cell(
    source: str,
    topology: str,
    dimension: int,
    degree: int,
    basis: str,
    nodes: int,
    maturity: str,
    metric: str | None,
    *limitations: str,
) -> CellCompatibility:
    return CellCompatibility(
        source_cell_type=source,
        topology=topology,
        dimension=dimension,
        geometry_degree=degree,
        geometry_basis=basis,
        node_count=nodes,
        import_maturity=maturity,
        solver_scope=_NEUTRAL_SCOPE,
        quality_metric=metric,
        limitations=(_SOURCE_FORMULATION_LIMIT, *limitations),
    )


_CELLS = (
    _cell("line", "interval", 1, 1, "complete_lagrange", 2, "conditional", None,
          "Line topology is not by itself a truss, beam, or cable formulation."),
    _cell("line3", "interval", 1, 2, "complete_lagrange", 3, "conditional", None,
          "Quadratic line topology requires an analysis-specific line formulation."),
    _cell("triangle", "triangle", 2, 1, "complete_lagrange", 3, "verified",
          "simplex_mean_ratio"),
    _cell("triangle6", "triangle", 2, 2, "complete_lagrange", 6, "verified",
          "simplex_mean_ratio_with_sampled_scaled_jacobian",
          "Curved geometry still requires positive coordinate-map Jacobians."),
    _cell("quad", "quadrilateral", 2, 1, "tensor_lagrange", 4, "verified",
          "sampled_scaled_jacobian"),
    _cell("quad8", "quadrilateral", 2, 2, "serendipity", 8, "conditional",
          "sampled_scaled_jacobian",
          "The verified DOLFINx XDMF route does not currently accept the eight-node quadrilateral layout."),
    _cell("quad9", "quadrilateral", 2, 2, "tensor_lagrange", 9, "verified",
          "sampled_scaled_jacobian"),
    _cell("tetra", "tetrahedron", 3, 1, "complete_lagrange", 4, "verified",
          "simplex_mean_ratio"),
    _cell("tetra10", "tetrahedron", 3, 2, "complete_lagrange", 10, "verified",
          "simplex_mean_ratio_with_sampled_scaled_jacobian",
          "C3D10H and similar source names additionally require an explicit mixed formulation."),
    _cell("hexahedron", "hexahedron", 3, 1, "tensor_lagrange", 8, "verified",
          "sampled_scaled_jacobian"),
    _cell("hexahedron20", "hexahedron", 3, 2, "serendipity", 20, "verified",
          "sampled_scaled_jacobian"),
    _cell("hexahedron27", "hexahedron", 3, 2, "tensor_lagrange", 27, "verified",
          "sampled_scaled_jacobian"),
    _cell("wedge", "prism", 3, 1, "complete_lagrange", 6, "verified",
          "sampled_scaled_jacobian"),
    _cell("wedge15", "prism", 3, 2, "serendipity", 15, "conditional",
          "sampled_scaled_jacobian",
          "Quadratic prism import and geometry ordering require a format-specific acceptance corpus."),
    _cell("pyramid", "pyramid", 3, 1, "complete_lagrange", 5, "conditional",
          "sampled_scaled_jacobian",
          "The current DOLFINx XDMF reader does not recognise the pyramid topology."),
)
_BY_SOURCE = {item.source_cell_type: item for item in _CELLS}

_TOPOLOGY_DIMENSIONS = {
    "interval": 1,
    "triangle": 2,
    "quadrilateral": 2,
    "tetrahedron": 3,
    "hexahedron": 3,
    "prism": 3,
    "pyramid": 3,
}
_TOPOLOGY_ALIASES = {
    "line": "interval",
    "quad": "quadrilateral",
    "tetra": "tetrahedron",
    "wedge": "prism",
}
_VERIFIED_RUNTIME_TOPOLOGIES = {
    "triangle",
    "quadrilateral",
    "tetrahedron",
    "hexahedron",
    "prism",
}


def _runtime_capabilities(topology: str) -> tuple[TopologyCapability, ...]:
    inspection = TopologyCapability(
        name="topology_inspection",
        status="verified",
        scope="DOLFINx runtime cell identity and dimension inspection",
        evidence=("tests/test_element_contracts.py",),
    )
    quality_status = "conditional" if topology == "interval" else "verified"
    quality = TopologyCapability(
        name="geometry_quality",
        status=quality_status,
        scope=(
            "MPI-global coordinate-map quality using the declared metric"
            if quality_status == "verified"
            else "no released interval quality metric"
        ),
        evidence=("tests/test_mesh_quality.py",),
    )
    conforming_status = "conditional" if topology == "interval" else "verified"
    conforming = TopologyCapability(
        name="conforming_p1_patch",
        status=conforming_status,
        scope="scalar H1 Lagrange P1 affine-gradient reproduction and assembly",
        evidence=("tests/test_mixed_cell_topologies.py",),
    )
    high_order_status = "conditional" if topology == "interval" else "verified"
    high_order = TopologyCapability(
        name="quadratic_geometry_preflight",
        status=high_order_status,
        scope="degree-two coordinate-basis identity and sampled Jacobian quality",
        evidence=("tests/test_mesh_quality.py",),
    )
    return inspection, quality, conforming, high_order


def describe_cell(source_cell_type: str) -> CellCompatibility:
    """Describe a meshio-style cell name without guessing equivalence."""

    selected = str(source_cell_type).strip().lower()
    result = _BY_SOURCE.get(selected)
    if result is not None:
        return result
    return CellCompatibility(
        source_cell_type=selected,
        topology=None,
        dimension=None,
        geometry_degree=None,
        geometry_basis=None,
        node_count=None,
        import_maturity="blocked",
        solver_scope="no declared AgentFEM solver-domain lowering",
        quality_metric=None,
        limitations=(
            "The source cell type is not in the reviewed AgentFEM neutral-geometry matrix.",
        ),
    )


def compatibility_matrix() -> tuple[CellCompatibility, ...]:
    """Return the complete, deterministic neutral-geometry matrix."""

    return _CELLS


def describe_topology(topology: str) -> TopologyCompatibility:
    """Describe one runtime cell topology without inferring formulation.

    ``topology`` accepts DOLFINx/Basix names and the common neutral aliases
    used by meshio.  Prism, pyramid, and interval cells are inspectable but
    remain conditional until analysis-specific providers and release
    benchmarks establish their intended numerical formulation.
    """

    selected = str(topology).strip().lower()
    selected = _TOPOLOGY_ALIASES.get(selected, selected)
    dimension = _TOPOLOGY_DIMENSIONS.get(selected)
    sources = tuple(
        item.source_cell_type for item in _CELLS if item.topology == selected
    )
    if dimension is None:
        return TopologyCompatibility(
            topology=selected,
            dimension=None,
            runtime_maturity="blocked",
            quality_metric=None,
            source_cell_types=(),
            capabilities=(),
            limitations=(
                "No AgentFEM runtime topology and quality contract is declared.",
            ),
        )
    representative = next(
        (item for item in _CELLS if item.topology == selected),
        None,
    )
    metric = None if representative is None else representative.quality_metric
    maturity = (
        "verified" if selected in _VERIFIED_RUNTIME_TOPOLOGIES else "conditional"
    )
    limitations = (
        "Runtime topology does not select or reproduce a source finite-element formulation.",
        _SOURCE_FORMULATION_LIMIT,
    )
    if maturity == "conditional":
        limitations += (
            "Runtime inspection is available, but release-level provider and "
            "benchmark coverage is incomplete for this topology.",
        )
    return TopologyCompatibility(
        topology=selected,
        dimension=dimension,
        runtime_maturity=maturity,
        quality_metric=metric,
        source_cell_types=sources,
        capabilities=_runtime_capabilities(selected),
        limitations=limitations,
    )


__all__ = [
    "CellCompatibility",
    "TopologyCapability",
    "TopologyCompatibility",
    "compatibility_matrix",
    "describe_cell",
    "describe_topology",
]
