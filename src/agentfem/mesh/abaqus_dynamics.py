# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Lossless dynamic keyword inventory, distinct from executable translation."""

from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256


@dataclass(frozen=True)
class DynamicKeyword:
    keyword: str
    options: tuple
    flags: tuple
    data: tuple
    line: int

    def summary(self):
        return {
            "keyword": self.keyword,
            "options": dict(self.options),
            "flags": list(self.flags),
            "data": list(self.data),
            "source_line": self.line,
        }


@dataclass(frozen=True)
class AbaqusDynamicInput:
    source: str
    sha256: str
    unit_system: str
    nodes: object
    elements: object
    semantics: object
    records: tuple
    issues: tuple

    def summary(self):
        types = sorted({e.element_type for e in self.elements.elements})
        return {
            "schema": "agentfem.abaqus-dynamic-input",
            "source": self.source,
            "sha256": self.sha256,
            "unit_system": self.unit_system,
            "node_count": len(self.nodes.labels),
            "element_count": len(self.elements.elements),
            "element_types": types,
            "preserved": [
                "node labels and coordinates",
                "element labels and connectivity",
                "coincident independent nodes",
                "named sets and surfaces",
                "raw dynamic keyword records",
            ],
            "converted": [],
            "issues": list(self.issues),
            "execution": "requires_reviewed_lowering; inventory is not a solver",
            "formulations": [
                {
                    "source": t,
                    "topology_read": True,
                    "abaqus_formula_equivalent": False,
                    "note": "C3D10M auxiliary displacement/hourglass formula is not implemented"
                    if t == "C3D10M"
                    else "numerical parity requires external evidence",
                }
                for t in types
            ],
            "semantics": self.semantics.summary(),
            "records": [r.summary() for r in self.records],
        }

    def node_labels(self, name):
        selected = next(
            (s for s in self.semantics.node_sets if s.name == name.upper()), None
        )
        if selected is None:
            raise KeyError(name)
        return selected.labels

    def nodeset(self, name, *, tolerance=1e-10):
        """Return a physical node selection using preserved source labels."""
        from .._nodal import nodal_selection

        return nodal_selection(
            labels=self.node_labels(name), source=self.nodes, tolerance=tolerance
        )


def read_dynamic_input(path, *, unit_system):
    """Inventory a flattened INP; reject scoped/include decks before label loss.

    BOUNDARY, CLOAD, AMPLITUDE, STEP and output cards retain exact text and
    line numbers. Keeping a card is not a claim that it has been executed.
    Part/assembly/include projects must first use the existing reviewed
    migration route; this reader never merges repeated part-local labels.
    """
    from .abaqus import read_node_table, read_element_table, read_model_semantics

    if not isinstance(unit_system, str) or not unit_system.strip():
        raise ValueError("AFM-ABAQUS-UNITS: declare consistent length/mass/time units.")
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    records = []
    current = None
    for line, raw in enumerate(text.splitlines(), 1):
        row = raw.strip()
        if not row or row.startswith("**"):
            continue
        if row.startswith("*"):
            if current is not None:
                records.append(DynamicKeyword(**current))
            chunks = [x.strip() for x in row.split(",")]
            options = []
            flags = []
            for x in chunks[1:]:
                if "=" in x:
                    k, v = x.split("=", 1)
                    options.append((k.strip().upper(), v.strip()))
                elif x:
                    flags.append(x.upper())
            current = {
                "keyword": chunks[0].upper(),
                "options": tuple(options),
                "flags": tuple(flags),
                "data": (),
                "line": line,
            }
        else:
            if current is None:
                raise ValueError(
                    f"AFM-ABAQUS-DATA: data outside keyword at line {line}."
                )
            current["data"] += (row,)
    if current is not None:
        records.append(DynamicKeyword(**current))
    scoped = {"*PART", "*ASSEMBLY", "*INSTANCE", "*INCLUDE"}
    if any(r.keyword in scoped for r in records):
        raise NotImplementedError(
            "AFM-ABAQUS-SCOPE: use reviewed migration to flatten Part/Instance/INCLUDE before dynamic inventory."
        )
    nodes = read_node_table(path)
    elements = read_element_table(path)
    semantics = read_model_semantics(path)
    known = set(map(int, nodes.labels))
    issues = []
    reported_formulations = set()
    for element in elements.elements:
        if not set(element.connectivity) <= known:
            raise ValueError("AFM-ABAQUS-CONNECTIVITY: missing source node.")
        if (
            element.element_type == "C3D10M"
            and element.element_type not in reported_formulations
        ):
            reported_formulations.add(element.element_type)
            issues.append(
                {
                    "severity": "blocked",
                    "code": "AFM-C3D10M-FORMULA",
                    "element_type": "C3D10M",
                    "message": "Topology is retained; modified interpolation, auxiliary modes and hourglass control are not available. Ordinary P2/HRZ is not an equivalent replacement.",
                }
            )
    # Every execution card is intentionally exposed for review, rather than
    # implying that preservation constitutes a whole-deck implementation.
    for r in records:
        if r.keyword not in {
            "*NODE",
            "*ELEMENT",
            "*NSET",
            "*ELSET",
            "*SURFACE",
            "*HEADING",
        }:
            issues.append(
                {
                    "severity": "review",
                    "code": "AFM-ABAQUS-CARD-PRESERVED",
                    "keyword": r.keyword,
                    "line": r.line,
                    "message": "Source text preserved; explicit case lowering required.",
                }
            )
    return AbaqusDynamicInput(
        str(path),
        sha256(path.read_bytes()).hexdigest(),
        unit_system,
        nodes,
        elements,
        semantics,
        tuple(records),
        tuple(issues),
    )
