"""Focused Abaqus keyword readers used by AgentFEM interoperability.

The mesh topology is delegated to :mod:`meshio`.  This module retains the
information that a generic mesh converter normally discards but scientific
constraints still need: Abaqus node labels and ``*EQUATION`` terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import re
import numpy as np


@dataclass(frozen=True)
class AbaqusNamedSet:
    """A source-labelled Abaqus node or element set."""

    name: str
    labels: tuple[int, ...]
    kind: str

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "count": len(self.labels),
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class AbaqusSurfaceEntry:
    """One source set/element and face identifier in an Abaqus surface."""

    reference: str
    face: str | None = None


@dataclass(frozen=True)
class AbaqusSurface:
    """A node- or element-based Abaqus surface definition."""

    name: str
    surface_type: str
    entries: tuple[AbaqusSurfaceEntry, ...]

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "abaqus_surface",
            "surface_type": self.surface_type,
            "entries": [
                {"reference": item.reference, "face": item.face}
                for item in self.entries
            ],
        }


@dataclass(frozen=True)
class AbaqusModelSemantics:
    """Engineering names retained independently of neutral mesh topology."""

    node_sets: tuple[AbaqusNamedSet, ...] = ()
    element_sets: tuple[AbaqusNamedSet, ...] = ()
    surfaces: tuple[AbaqusSurface, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "node_sets": {item.name: item.summary() for item in self.node_sets},
            "element_sets": {
                item.name: item.summary() for item in self.element_sets
            },
            "surfaces": {item.name: item.summary() for item in self.surfaces},
        }


def read_model_semantics(path: str | Path) -> AbaqusModelSemantics:
    """Read named node/element sets and explicit surface entries."""

    source = Path(path)
    nsets: dict[str, list[int]] = {}
    elsets: dict[str, list[int]] = {}
    surfaces: dict[str, tuple[str, list[AbaqusSurfaceEntry]]] = {}
    keyword, options, flags = "", {}, set()
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            parts = [item.strip() for item in line.split(",")]
            keyword, options, flags = parts[0].upper(), {}, set()
            for item in parts[1:]:
                if "=" in item:
                    key, value = item.split("=", 1)
                    options[key.strip().upper()] = value.strip().upper()
                elif item:
                    flags.add(item.upper())
            if keyword == "*SURFACE":
                name = options.get("NAME")
                if not name:
                    raise ValueError(f"*SURFACE at {source}:{line_number} requires NAME=.")
                surfaces.setdefault(name, (options.get("TYPE", "ELEMENT"), []))
            continue
        values = _csv_values(line)
        if keyword == "*NODE" and options.get("NSET"):
            nsets.setdefault(options["NSET"], []).append(int(values[0]))
        elif keyword == "*ELEMENT" and options.get("ELSET"):
            elsets.setdefault(options["ELSET"], []).append(int(values[0]))
        elif keyword in {"*NSET", "*ELSET"}:
            option = keyword[1:]
            name = options.get(option)
            if not name:
                raise ValueError(f"{keyword} at {source}:{line_number} requires {option}=.")
            selected = (nsets if keyword == "*NSET" else elsets).setdefault(name, [])
            if "GENERATE" in flags:
                if len(values) not in {2, 3}:
                    raise ValueError(f"Invalid {keyword}, GENERATE at {source}:{line_number}.")
                start, end = int(values[0]), int(values[1])
                increment = 1 if len(values) == 2 else int(values[2])
                if increment <= 0 or end < start:
                    raise ValueError("Abaqus set generation requires a positive range.")
                selected.extend(range(start, end + 1, increment))
            else:
                selected.extend(int(value) for value in values)
        elif keyword == "*SURFACE":
            reference = values[0].upper()
            face = values[1].upper() if len(values) > 1 and values[1] else None
            surfaces[options["NAME"]][1].append(AbaqusSurfaceEntry(reference, face))

    def named(values, kind):
        return tuple(AbaqusNamedSet(name, tuple(dict.fromkeys(labels)), kind) for name, labels in sorted(values.items()))

    return AbaqusModelSemantics(
        named(nsets, "abaqus_node_set"), named(elsets, "abaqus_element_set"),
        tuple(AbaqusSurface(name, value[0], tuple(value[1])) for name, value in sorted(surfaces.items())),
    )


@dataclass(frozen=True)
class AbaqusElementDefinition:
    """Scientific identity attached to one Abaqus ``*ELEMENT`` type.

    Neutral mesh formats preserve topology and geometry, but normally discard
    formulation suffixes such as ``H``.  Keeping that distinction beside the
    converted mesh prevents a readable ``tetra10`` grid from being mistaken
    for a verified Abaqus hybrid formulation.
    """

    source_type: str
    topology: str | None = None
    interpolation: str | None = None
    node_count: int | None = None
    family: str = "unknown"
    physics: str = "unknown"
    kinematics: str | None = None
    integration: str | None = None
    formulation: str = "source_defined"
    pressure_interpolation: str | None = None
    additional_pressure_variables: int = 0
    additional_displacement_variables: int = 0
    import_capability: str = "declaration_only"
    neutral_conversion: str = "not_verified"
    solver_capability: str = "not_declared"
    notes: tuple[str, ...] = ()

    @property
    def is_hybrid(self) -> bool:
        return self.pressure_interpolation is not None

    def summary(self) -> dict[str, object]:
        return {
            "source_type": self.source_type,
            "topology": self.topology,
            "interpolation": self.interpolation,
            "node_count": self.node_count,
            "family": self.family,
            "physics": self.physics,
            "kinematics": self.kinematics,
            "integration": self.integration,
            "formulation": self.formulation,
            "pressure_interpolation": self.pressure_interpolation,
            "additional_pressure_variables": self.additional_pressure_variables,
            "additional_displacement_variables": self.additional_displacement_variables,
            "import_capability": self.import_capability,
            "neutral_conversion": self.neutral_conversion,
            "solver_capability": self.solver_capability,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AbaqusElementFormulationDerivation:
    """Evidence for a topology-preserving Abaqus element-keyword rewrite.

    Abaqus element suffixes can change the numerical formulation without
    changing nodes or connectivity.  The derivation record makes that narrow
    source transformation explicit and reproducible instead of asking users
    to duplicate and hand-edit a large mesh file.
    """

    source_path: Path
    derived_path: Path
    source_type: str
    target_type: str
    rewritten_declarations: int
    source_sha256: str
    derived_sha256: str
    manifest_path: Path

    def summary(self) -> dict[str, object]:
        return {
            "kind": "abaqus_element_formulation_derivation",
            "source_path": str(self.source_path),
            "derived_path": str(self.derived_path),
            "source_type": self.source_type,
            "target_type": self.target_type,
            "rewritten_declarations": self.rewritten_declarations,
            "topology_preserved": True,
            "nodes_and_connectivity_preserved": True,
            "source_sha256": self.source_sha256,
            "derived_sha256": self.derived_sha256,
            "manifest_path": str(self.manifest_path),
        }


def derive_element_formulation(
    source: str | Path,
    destination: str | Path,
    *,
    source_type: str,
    target_type: str,
) -> AbaqusElementFormulationDerivation:
    """Derive an Abaqus mesh by changing only matching ``TYPE=`` values.

    This helper is intentionally narrower than a generic text replacement.
    It accepts only element types with the same known topology and node count,
    preserves every non-keyword byte, and writes a provenance sidecar.  A
    ``C3D10`` mesh can therefore become a scientifically explicit ``C3D10H``
    source while retaining exactly the same geometry and connectivity.
    """

    source_path = Path(source)
    derived_path = Path(destination)
    if source_path.resolve() == derived_path.resolve():
        raise ValueError("The derived Abaqus mesh must not overwrite its source.")
    selected_source = str(source_type).strip().upper()
    selected_target = str(target_type).strip().upper()
    source_definition = describe_element_type(selected_source)
    target_definition = describe_element_type(selected_target)
    if (
        source_definition.topology is None
        or target_definition.topology is None
        or source_definition.topology != target_definition.topology
        or _element_node_count(selected_source) != _element_node_count(selected_target)
    ):
        raise ValueError(
            "Element formulation derivation requires known element types with "
            "identical topology and connectivity size."
        )

    original = source_path.read_bytes()
    type_option = re.compile(rb"(?i)(\bTYPE\s*=\s*)([A-Z0-9]+)")
    rewritten = 0
    output_lines: list[bytes] = []
    for raw in original.splitlines(keepends=True):
        stripped = raw.lstrip()
        if not stripped.startswith(b"*") or stripped.startswith(b"**"):
            output_lines.append(raw)
            continue
        keyword = stripped.split(b",", 1)[0].strip().upper()
        if keyword != b"*ELEMENT":
            output_lines.append(raw)
            continue

        def replace(match: re.Match[bytes]) -> bytes:
            nonlocal rewritten
            declared = match.group(2).decode("ascii").upper()
            if declared != selected_source:
                return match.group(0)
            rewritten += 1
            return match.group(1) + selected_target.encode("ascii")

        output_lines.append(type_option.sub(replace, raw, count=1))
    if rewritten == 0:
        raise ValueError(
            f"No *ELEMENT declaration with TYPE={selected_source} was found in "
            f"{source_path}."
        )

    derived = b"".join(output_lines)
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = derived_path.with_name(f".{derived_path.name}.tmp")
    temporary.write_bytes(derived)
    temporary.replace(derived_path)
    manifest_path = derived_path.with_suffix(
        derived_path.suffix + ".formulation.json"
    )
    record = AbaqusElementFormulationDerivation(
        source_path=source_path,
        derived_path=derived_path,
        source_type=selected_source,
        target_type=selected_target,
        rewritten_declarations=rewritten,
        source_sha256=sha256(original).hexdigest(),
        derived_sha256=sha256(derived).hexdigest(),
        manifest_path=manifest_path,
    )
    manifest_path.write_text(
        json.dumps(record.summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def read_element_formulation_derivation(
    source: str | Path,
) -> dict[str, object] | None:
    """Read and validate an adjacent formulation-derivation sidecar."""

    source_path = Path(source)
    manifest_path = source_path.with_suffix(
        source_path.suffix + ".formulation.json"
    )
    if not manifest_path.exists():
        return None
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = sha256(source_path.read_bytes()).hexdigest()
    if record.get("derived_sha256") != actual:
        raise ValueError(
            f"Abaqus formulation derivation manifest is stale for {source_path}."
        )
    return dict(record)


@dataclass(frozen=True)
class AbaqusElement:
    """One source element with its original label and node ordering."""

    label: int
    element_type: str
    connectivity: tuple[int, ...]

    def face_corner_labels(self, face: str) -> tuple[int, ...]:
        """Return source node labels for a supported solid-element face."""

        selected = str(face).upper()
        family = _solid_face_family(self.element_type)
        try:
            local = _SOLID_FACE_CORNERS[family][selected]
        except KeyError as exc:
            raise NotImplementedError(
                f"Abaqus face {selected!r} is not supported for "
                f"element type {self.element_type!r}."
            ) from exc
        return tuple(self.connectivity[index - 1] for index in local)


@dataclass(frozen=True)
class AbaqusElementTable:
    """Source-labelled elements used to reconstruct engineering surfaces."""

    elements: tuple[AbaqusElement, ...]

    def __post_init__(self) -> None:
        labels = [item.label for item in self.elements]
        if len(set(labels)) != len(labels):
            raise ValueError(
                "Repeated Abaqus element labels require part/instance-aware import, "
                "which this surface adapter does not yet infer."
            )

    def element(self, label: int) -> AbaqusElement:
        for item in self.elements:
            if item.label == int(label):
                return item
        raise KeyError(f"Abaqus element label {label} is not present.")


_SOLID_FACE_CORNERS = {
    "tetrahedron": {
        "S1": (1, 2, 3),
        "S2": (1, 4, 2),
        "S3": (2, 4, 3),
        "S4": (3, 4, 1),
    },
    "hexahedron": {
        "S1": (1, 2, 3, 4),
        "S2": (5, 8, 7, 6),
        "S3": (1, 5, 6, 2),
        "S4": (2, 6, 7, 3),
        "S5": (3, 7, 8, 4),
        "S6": (4, 8, 5, 1),
    },
    "wedge": {
        "S1": (1, 2, 3),
        "S2": (4, 6, 5),
        "S3": (1, 4, 5, 2),
        "S4": (2, 5, 6, 3),
        "S5": (3, 6, 4, 1),
    },
}


def _solid_face_family(element_type: str) -> str:
    definition = describe_element_type(element_type)
    if definition.topology in {"tetra", "tetra10"}:
        return "tetrahedron"
    if definition.topology in {"hexahedron", "hexahedron20", "hexahedron27"}:
        return "hexahedron"
    if definition.topology in {"wedge", "wedge15"}:
        return "wedge"
    raise NotImplementedError(
        "Abaqus surface reconstruction currently supports tetrahedral, "
        f"hexahedral, and wedge solid families, not {element_type!r}."
    )


def _element_node_count(element_type: str) -> int:
    definition = describe_element_type(element_type)
    if definition.node_count is None:
        raise NotImplementedError(
            f"Element connectivity parsing is not implemented for {element_type!r}. "
            "The keyword declaration can still be inventoried."
        )
    return int(definition.node_count)


def read_element_table(path: str | Path) -> AbaqusElementTable:
    """Read supported solid connectivity without losing source labels."""

    source = Path(path)
    records: list[AbaqusElement] = []
    element_type: str | None = None
    pending: list[str] = []
    expected = 0

    def consume(line_number: int) -> None:
        nonlocal pending
        while pending:
            if len(pending) < expected + 1:
                return
            values, pending = pending[: expected + 1], pending[expected + 1 :]
            records.append(
                AbaqusElement(
                    int(values[0]),
                    str(element_type),
                    tuple(int(value) for value in values[1:]),
                )
            )

    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            if pending:
                raise ValueError(
                    f"Incomplete Abaqus element at {source}:{line_number - 1}."
                )
            fields = [item.strip() for item in line.split(",")]
            if fields[0].upper() != "*ELEMENT":
                element_type = None
                continue
            options = {}
            for item in fields[1:]:
                if "=" in item:
                    key, value = item.split("=", 1)
                    options[key.strip().upper()] = value.strip().upper()
            element_type = options.get("TYPE")
            if element_type is None:
                raise ValueError(f"*ELEMENT at {source}:{line_number} requires TYPE=.")
            expected = _element_node_count(element_type)
            continue
        if element_type is not None:
            pending.extend(_csv_values(line))
            consume(line_number)
    if pending:
        raise ValueError(f"Incomplete Abaqus element at the end of {source}.")
    if not records:
        raise ValueError(f"No supported *ELEMENT data were found in {source}.")
    return AbaqusElementTable(tuple(records))


_ABAQUS_ELEMENT_LIBRARY: dict[str, dict[str, object]] = {}

# Declarations supported by the pinned meshio 5.3.x Abaqus reader. A neutral
# topology reader does not reproduce reduced integration, hourglass control,
# hybrid variables, shell directors, or cohesive kinematics.
_MESHIO_ABAQUS_TYPES = {
    "B21", "B21H", "B22", "B22H", "B31", "B31H", "B32", "B32H",
    "B33", "B33H", "C3D10", "C3D10H", "C3D10I", "C3D10M", "C3D10MH",
    "C3D15", "C3D20", "C3D20H", "C3D20R", "C3D20RH", "C3D4", "C3D4H",
    "C3D6", "C3D8", "C3D8H", "C3D8I", "C3D8IH", "C3D8R", "C3D8RH",
    "CAX4P", "CPE6", "CPS3", "CPS4", "CPS4R", "R3D3", "S3", "S3R",
    "S3RS", "S4", "S4R", "S4R5", "S4RS", "S4RSW", "S8R", "S8R5",
    "S9R5", "STRI3", "STRI65", "T2D2", "T2D2H", "T2D3", "T2D3H",
    "T3D2", "T3D2H", "T3D3", "T3D3H",
}


def _register_element_types(
    names: tuple[str, ...],
    *,
    topology: str,
    interpolation: str,
    node_count: int,
    family: str,
    physics: str,
    kinematics: str | None = None,
    import_capability: str = "topology_and_semantics",
    solver_capability: str = "topology_only",
    **shared,
) -> None:
    for name in names:
        _ABAQUS_ELEMENT_LIBRARY[name] = {
            "topology": topology,
            "interpolation": interpolation,
            "node_count": node_count,
            "family": family,
            "physics": physics,
            "kinematics": kinematics,
            "import_capability": import_capability,
            "solver_capability": solver_capability,
            **shared,
        }


# Continuum solids.  Native support means that AgentFEM has a corresponding
# public finite-element route; it does not claim byte-for-byte Abaqus element
# equivalence.  Reduced integration, incompatible modes, and hybrid variables
# remain explicit formulation semantics rather than aliases.
_register_element_types(
    ("C3D4",), topology="tetra", interpolation="linear", node_count=4,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="displacement", solver_capability="native_lagrange_analogue",
)
_register_element_types(
    ("C3D4H",), topology="tetra", interpolation="linear", node_count=4,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="hybrid", pressure_interpolation="constant",
    additional_pressure_variables=1,
)
_register_element_types(
    ("C3D5",), topology="pyramid", interpolation="linear", node_count=5,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="displacement",
)
_register_element_types(
    ("C3D5H",), topology="pyramid", interpolation="linear", node_count=5,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="hybrid", pressure_interpolation="source_hybrid",
    notes=("Hybrid pressure variables require an explicit mixed formulation.",),
)
_register_element_types(
    ("C3D6",), topology="wedge", interpolation="linear", node_count=6,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="displacement", solver_capability="native_lagrange_analogue",
)
_register_element_types(
    ("C3D6H",), topology="wedge", interpolation="linear", node_count=6,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="hybrid", pressure_interpolation="source_hybrid",
    notes=("Hybrid pressure variables require an explicit mixed formulation.",),
)
_register_element_types(
    ("C3D15", "C3D15V"), topology="wedge15", interpolation="quadratic", node_count=15,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="displacement",
)
_register_element_types(
    ("C3D15H", "C3D15VH"), topology="wedge15", interpolation="quadratic", node_count=15,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="hybrid", pressure_interpolation="source_hybrid",
    notes=("Hybrid pressure variables require an explicit mixed formulation.",),
)
_register_element_types(
    ("C3D8",), topology="hexahedron", interpolation="linear", node_count=8,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="displacement", integration="full",
    solver_capability="native_lagrange_analogue",
)
_register_element_types(
    ("C3D8R",), topology="hexahedron", interpolation="linear", node_count=8,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="displacement_hourglass_control", integration="reduced",
    notes=("Abaqus hourglass control is not implied by topology conversion.",),
)
_register_element_types(
    ("C3D8H", "C3D8RH"), topology="hexahedron", interpolation="linear", node_count=8,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="hybrid", pressure_interpolation="constant",
    additional_pressure_variables=1,
    notes=("Hybrid pressure variables require an explicit mixed formulation.",),
)
_ABAQUS_ELEMENT_LIBRARY["C3D8RH"]["integration"] = "reduced"
_register_element_types(
    ("C3D8I", "C3D8IH"), topology="hexahedron", interpolation="linear", node_count=8,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="incompatible_modes",
    notes=("Incompatible-mode variables are not reproduced by neutral mesh conversion.",),
)
_register_element_types(
    ("C3D10",), topology="tetra10", interpolation="quadratic", node_count=10,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="displacement", solver_capability="native_lagrange_analogue",
)
_register_element_types(
    ("C3D10H",), topology="tetra10", interpolation="quadratic", node_count=10,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="hybrid", pressure_interpolation="constant",
    additional_pressure_variables=1, solver_capability="native_mixed_analogue",
)
_register_element_types(
    ("C3D10HS",), topology="tetra10", interpolation="quadratic", node_count=10,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="hybrid_improved_surface_stress", pressure_interpolation="linear",
    additional_pressure_variables=4,
)
_register_element_types(
    ("C3D10M",), topology="tetra10", interpolation="modified_quadratic", node_count=10,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="modified_hourglass_control", additional_displacement_variables=3,
)
_register_element_types(
    ("C3D10MH",), topology="tetra10", interpolation="modified_quadratic", node_count=10,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="modified_hybrid_hourglass_control", pressure_interpolation="linear",
    additional_pressure_variables=4, additional_displacement_variables=3,
)
_register_element_types(
    ("C3D20",), topology="hexahedron20", interpolation="quadratic", node_count=20,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="displacement", integration="full",
)
_register_element_types(
    ("C3D20R",), topology="hexahedron20", interpolation="quadratic", node_count=20,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="displacement", integration="reduced",
)
_register_element_types(
    ("C3D20H", "C3D20RH"), topology="hexahedron20", interpolation="quadratic", node_count=20,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="hybrid", pressure_interpolation="linear",
    notes=("Hybrid pressure variables require an explicit mixed formulation.",),
)
_ABAQUS_ELEMENT_LIBRARY["C3D20RH"]["integration"] = "reduced"
_register_element_types(
    ("C3D27", "C3D27R"), topology="hexahedron27", interpolation="quadratic", node_count=27,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="displacement",
)
_ABAQUS_ELEMENT_LIBRARY["C3D27R"]["integration"] = "reduced"
_register_element_types(
    ("C3D27H", "C3D27RH"), topology="hexahedron27", interpolation="quadratic", node_count=27,
    family="continuum_solid", physics="solid_mechanics", kinematics="three_dimensional",
    formulation="hybrid", pressure_interpolation="source_hybrid",
    notes=("Hybrid pressure variables require an explicit mixed formulation.",),
)
_ABAQUS_ELEMENT_LIBRARY["C3D27RH"]["integration"] = "reduced"

# Plane stress, plane strain, generalized plane strain, and axisymmetric solids.
for prefix, kinematics in (
    ("CPS", "plane_stress"),
    ("CPE", "plane_strain"),
    ("CPEG", "generalized_plane_strain"),
    ("CAX", "axisymmetric"),
):
    _register_element_types(
        (f"{prefix}3",), topology="triangle", interpolation="linear", node_count=3,
        family="continuum_solid", physics="solid_mechanics", kinematics=kinematics,
        formulation="displacement", solver_capability="native_lagrange_analogue",
    )
    _register_element_types(
        (f"{prefix}4",), topology="quad", interpolation="linear", node_count=4,
        family="continuum_solid", physics="solid_mechanics", kinematics=kinematics,
        formulation="displacement", integration="full",
        solver_capability="native_lagrange_analogue",
    )
    _register_element_types(
        (f"{prefix}4R",), topology="quad", interpolation="linear", node_count=4,
        family="continuum_solid", physics="solid_mechanics", kinematics=kinematics,
        formulation="displacement_hourglass_control", integration="reduced",
        notes=("Abaqus hourglass control is not implied by topology conversion.",),
    )
    _register_element_types(
        (f"{prefix}6",), topology="triangle6", interpolation="quadratic", node_count=6,
        family="continuum_solid", physics="solid_mechanics", kinematics=kinematics,
        formulation="displacement", solver_capability="native_lagrange_analogue",
    )
    _register_element_types(
        (f"{prefix}8", f"{prefix}8R"), topology="quad8", interpolation="quadratic", node_count=8,
        family="continuum_solid", physics="solid_mechanics", kinematics=kinematics,
        formulation="displacement",
    )
    _ABAQUS_ELEMENT_LIBRARY[f"{prefix}8R"]["integration"] = "reduced"

for name in ("CPE3H", "CPE4H", "CPE4RH", "CPE6H", "CPE8H", "CPE8RH",
             "CAX3H", "CAX4H", "CAX4RH", "CAX6H", "CAX8H", "CAX8RH"):
    base = name.replace("RH", "R").replace("H", "")
    source = dict(_ABAQUS_ELEMENT_LIBRARY[base])
    source.update(
        formulation="hybrid",
        pressure_interpolation="constant" if source["interpolation"] == "linear" else "linear",
        solver_capability="topology_only",
        notes=("Hybrid pressure variables require an explicit mixed formulation.",),
    )
    _ABAQUS_ELEMENT_LIBRARY[name] = source

# Heat-transfer and coupled temperature-displacement topologies.
for name, topology, interpolation, nodes in (
    ("DC2D3", "triangle", "linear", 3), ("DC2D4", "quad", "linear", 4),
    ("DC2D6", "triangle6", "quadratic", 6), ("DC2D8", "quad8", "quadratic", 8),
    ("DC3D4", "tetra", "linear", 4), ("DC3D6", "wedge", "linear", 6),
    ("DC3D8", "hexahedron", "linear", 8), ("DC3D10", "tetra10", "quadratic", 10),
    ("DC3D15", "wedge15", "quadratic", 15), ("DC3D20", "hexahedron20", "quadratic", 20),
):
    _register_element_types(
        (name,), topology=topology, interpolation=interpolation, node_count=nodes,
        family="continuum_heat", physics="heat_transfer",
        kinematics="three_dimensional" if name.startswith("DC3") else "two_dimensional",
        formulation="temperature", solver_capability="native_lagrange_analogue",
    )

# Interface, truss, beam, and shell declarations are parsed and retained now;
# their dedicated kinematics remain separate roadmap items.
for name, topology, nodes, kinematics in (
    ("COH2D4", "quad", 4, "two_dimensional_interface"),
    ("COHAX4", "quad", 4, "axisymmetric_interface"),
    ("COH3D6", "wedge", 6, "three_dimensional_interface"),
    ("COH3D8", "hexahedron", 8, "three_dimensional_interface"),
):
    _register_element_types(
        (name,), topology=topology, interpolation="linear", node_count=nodes,
        family="cohesive", physics="solid_mechanics", kinematics=kinematics,
        formulation="cohesive", notes=("Dedicated interface lowering is required.",),
    )
_register_element_types(
    ("T2D2", "T3D2", "B21", "B31"), topology="line", interpolation="linear", node_count=2,
    family="line_structure", physics="solid_mechanics", formulation="source_defined",
    notes=("Section and line-element kinematics are not inferred from connectivity.",),
)
_register_element_types(
    ("T2D3", "T3D3", "B22", "B32"), topology="line3", interpolation="quadratic", node_count=3,
    family="line_structure", physics="solid_mechanics", formulation="source_defined",
    notes=("Section and line-element kinematics are not inferred from connectivity.",),
)
_register_element_types(
    ("S3", "S3R", "STRI3"), topology="triangle", interpolation="linear", node_count=3,
    family="shell", physics="solid_mechanics", formulation="shell",
    notes=("Shell director, section, and rotational dofs require dedicated lowering.",),
)
_register_element_types(
    ("S4", "S4R", "S4R5"), topology="quad", interpolation="linear", node_count=4,
    family="shell", physics="solid_mechanics", formulation="shell",
    notes=("Shell director, section, and rotational dofs require dedicated lowering.",),
)
_register_element_types(
    ("S8R", "S8R5"), topology="quad8", interpolation="quadratic", node_count=8,
    family="shell", physics="solid_mechanics", formulation="shell",
    notes=("Shell director, section, and rotational dofs require dedicated lowering.",),
)


def describe_element_type(element_type: str) -> AbaqusElementDefinition:
    """Describe formulation information that is lost in neutral mesh I/O."""

    selected = str(element_type).strip().upper()
    details = _ABAQUS_ELEMENT_LIBRARY.get(selected)
    if details is None:
        return AbaqusElementDefinition(source_type=selected)
    resolved = dict(details)
    resolved["neutral_conversion"] = (
        "meshio_reader" if selected in _MESHIO_ABAQUS_TYPES else "not_verified"
    )
    return AbaqusElementDefinition(source_type=selected, **resolved)


def supported_element_types(*, family: str | None = None) -> tuple[str, ...]:
    """Return Abaqus declarations with explicit AgentFEM import semantics."""

    if family is None:
        return tuple(sorted(_ABAQUS_ELEMENT_LIBRARY))
    selected = str(family).strip().lower()
    return tuple(
        name for name in sorted(_ABAQUS_ELEMENT_LIBRARY)
        if str(_ABAQUS_ELEMENT_LIBRARY[name].get("family", "")).lower() == selected
    )


def read_element_definitions(path: str | Path) -> tuple[AbaqusElementDefinition, ...]:
    """Read distinct Abaqus element declarations in source order.

    This intentionally reads keyword headers rather than inferring element
    semantics from the topology produced by meshio.
    """

    path = Path(path)
    selected: list[AbaqusElementDefinition] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("*") or line.startswith("**"):
            continue
        fields = [item.strip() for item in line.split(",")]
        if fields[0].upper() != "*ELEMENT":
            continue
        options = {}
        for item in fields[1:]:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            options[key.strip().upper()] = value.strip()
        element_type = options.get("TYPE")
        if element_type is None:
            raise ValueError(f"Abaqus *ELEMENT declaration in {path} has no TYPE= option.")
        normalized = element_type.upper()
        if normalized not in seen:
            selected.append(describe_element_type(normalized))
            seen.add(normalized)
    return tuple(selected)


_PRESERVED_KEYWORDS = {
    "*HEADING", "*NODE", "*ELEMENT", "*NSET", "*ELSET", "*SURFACE",
    "*EQUATION",
}
_MIGRATION_PLANNED_KEYWORDS = {
    "*AMPLITUDE", "*ASSEMBLY", "*BOUNDARY", "*CLOAD", "*DLOAD",
    "*DYNAMIC", "*ELASTIC", "*END ASSEMBLY", "*END INSTANCE",
    "*END PART", "*END STEP", "*HYPERELASTIC", "*INSTANCE", "*MATERIAL",
    "*PART", "*PLASTIC", "*SOLID SECTION", "*STATIC", "*STEP", "*INCLUDE",
}


@dataclass(frozen=True)
class _AbaqusDeckInventory:
    """Structural inventory that does not require globally unique labels."""

    node_count: int
    element_count: int | None
    keyword_counts: tuple[tuple[str, int], ...]
    part_names: tuple[str, ...]
    instance_names: tuple[str, ...]
    material_names: tuple[str, ...]
    section_count: int
    step_count: int
    include_files: tuple[str, ...]


@dataclass(frozen=True)
class AbaqusSourceFile:
    """One content-addressed file in an Abaqus input-deck source graph."""

    path: Path
    logical_path: str
    source_sha256: str
    include_files: tuple[str, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "logical_path": self.logical_path,
            "source_sha256": self.source_sha256,
            "include_files": list(self.include_files),
        }


@dataclass(frozen=True)
class AbaqusIncludeEdge:
    """One declared ``*INCLUDE`` relation and its resolution status."""

    source: str
    declaration: str
    target: str
    status: str

    def summary(self) -> dict[str, str]:
        return {
            "source": self.source,
            "declaration": self.declaration,
            "target": self.target,
            "status": self.status,
        }


@dataclass(frozen=True)
class AbaqusSourceGraph:
    """Recursive, content-addressed identity of one Abaqus input deck.

    The graph is an inspection asset, not a flattened solver deck. Keeping
    include relations explicit prevents nested files, missing dependencies,
    or cycles from being hidden by eager text concatenation.
    """

    root: Path
    files: tuple[AbaqusSourceFile, ...]
    edges: tuple[AbaqusIncludeEdge, ...]
    fingerprint: str
    issues: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.issues and all(edge.status == "resolved" for edge in self.edges)

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.abaqus-source-graph",
            "schema_version": "0.1.0",
            "root": str(self.root),
            "fingerprint": self.fingerprint,
            "complete": self.complete,
            "files": [item.summary() for item in self.files],
            "edges": [item.summary() for item in self.edges],
            "issues": list(self.issues),
        }


def read_source_graph(path: str | Path) -> AbaqusSourceGraph:
    """Resolve nested Abaqus ``*INCLUDE`` files without mutating the deck.

    Relative include paths are resolved from the file that declares them.
    Missing files and recursive cycles remain inspectable issues instead of
    being ignored or causing unbounded recursion. The graph fingerprint is
    based on logical paths, contents, and edges, so changing any included
    source invalidates downstream conversion evidence.
    """

    root = Path(path).expanduser().resolve()
    if not root.is_file():
        raise FileNotFoundError(f"Abaqus input deck does not exist: {root}")

    files: dict[Path, AbaqusSourceFile] = {}
    edges: list[AbaqusIncludeEdge] = []
    issues: list[str] = []

    def logical(selected: Path) -> str:
        return Path(os.path.relpath(selected, root.parent)).as_posix()

    def include_declarations(selected: Path) -> tuple[str, ...]:
        declared: list[str] = []
        for raw in selected.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line.startswith("*") or line.startswith("**"):
                continue
            keyword, options = _keyword_options(line)
            if keyword != "*INCLUDE":
                continue
            value = options.get("INPUT") or options.get("FILE")
            if value is None or not value.strip():
                declared.append("<unspecified>")
            else:
                declared.append(value.strip().strip('"').strip("'"))
        return tuple(declared)

    def visit(selected: Path, ancestry: tuple[Path, ...]) -> None:
        resolved = selected.expanduser().resolve()
        if resolved in files:
            return
        raw = resolved.read_bytes()
        declarations = include_declarations(resolved)
        files[resolved] = AbaqusSourceFile(
            path=resolved,
            logical_path=logical(resolved),
            source_sha256=sha256(raw).hexdigest(),
            include_files=declarations,
        )
        for declaration in declarations:
            if declaration == "<unspecified>":
                edges.append(
                    AbaqusIncludeEdge(
                        logical(resolved), declaration, "<unspecified>", "missing"
                    )
                )
                issues.append(
                    f"AFM-ABAQUS-INCLUDE-001: {logical(resolved)} declares an "
                    "*INCLUDE without INPUT=."
                )
                continue
            candidate = Path(declaration).expanduser()
            if not candidate.is_absolute():
                candidate = resolved.parent / candidate
            candidate = candidate.resolve()
            target = logical(candidate)
            if candidate in ancestry or candidate == resolved:
                edges.append(
                    AbaqusIncludeEdge(logical(resolved), declaration, target, "cycle")
                )
                issues.append(
                    "AFM-ABAQUS-INCLUDE-002: recursive include cycle: "
                    + " -> ".join(logical(item) for item in (*ancestry, resolved, candidate))
                    + "."
                )
                continue
            if not candidate.is_file():
                edges.append(
                    AbaqusIncludeEdge(logical(resolved), declaration, target, "missing")
                )
                issues.append(
                    f"AFM-ABAQUS-INCLUDE-003: {logical(resolved)} references "
                    f"missing include {declaration!r}."
                )
                continue
            edges.append(
                AbaqusIncludeEdge(logical(resolved), declaration, target, "resolved")
            )
            visit(candidate, (*ancestry, resolved))

    visit(root, ())
    graph_payload = {
        "root": logical(root),
        "files": [
            {
                "logical_path": item.logical_path,
                "source_sha256": item.source_sha256,
                "include_files": list(item.include_files),
            }
            for item in files.values()
        ],
        "edges": [item.summary() for item in edges],
    }
    fingerprint = sha256(
        json.dumps(graph_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return AbaqusSourceGraph(
        root=root,
        files=tuple(files.values()),
        edges=tuple(edges),
        fingerprint=fingerprint,
        issues=tuple(dict.fromkeys(issues)),
    )


def _keyword_options(line: str) -> tuple[str, dict[str, str]]:
    fields = [item.strip() for item in line.split(",")]
    options: dict[str, str] = {}
    for item in fields[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            options[key.strip().upper()] = value.strip()
    return fields[0].upper(), options


def _inspect_deck_structure(source: Path) -> _AbaqusDeckInventory:
    """Count scoped Abaqus assets without pretending that labels are global."""

    counts: dict[str, int] = {}
    node_count = 0
    element_count = 0
    element_count_known = True
    active_keyword = ""
    active_element_type: str | None = None
    pending_connectivity: list[str] = []
    expected_connectivity: int | None = None
    parts: list[str] = []
    instances: list[str] = []
    materials: list[str] = []
    includes: list[str] = []
    section_count = 0
    step_count = 0

    def consume_elements() -> None:
        nonlocal element_count, pending_connectivity
        if expected_connectivity is None:
            return
        record_width = expected_connectivity + 1
        while len(pending_connectivity) >= record_width:
            element_count += 1
            pending_connectivity = pending_connectivity[record_width:]

    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            if active_keyword == "*ELEMENT" and pending_connectivity:
                element_count_known = False
                pending_connectivity = []
            active_keyword, options = _keyword_options(line)
            counts[active_keyword] = counts.get(active_keyword, 0) + 1
            active_element_type = None
            expected_connectivity = None
            if active_keyword == "*ELEMENT":
                active_element_type = options.get("TYPE", "").upper() or None
                if active_element_type is not None:
                    definition = describe_element_type(active_element_type)
                    expected_connectivity = definition.node_count
                if expected_connectivity is None:
                    element_count_known = False
            elif active_keyword == "*PART":
                parts.append(options.get("NAME", "<unnamed>"))
            elif active_keyword == "*INSTANCE":
                instances.append(options.get("NAME", "<unnamed>"))
            elif active_keyword == "*MATERIAL":
                materials.append(options.get("NAME", "<unnamed>"))
            elif active_keyword.endswith(" SECTION"):
                section_count += 1
            elif active_keyword == "*STEP":
                step_count += 1
            elif active_keyword == "*INCLUDE":
                selected = options.get("INPUT") or options.get("FILE")
                includes.append(selected or "<unspecified>")
            continue
        if active_keyword == "*NODE":
            node_count += 1
        elif active_keyword == "*ELEMENT":
            if expected_connectivity is None:
                continue
            pending_connectivity.extend(_csv_values(line))
            consume_elements()
    if active_keyword == "*ELEMENT" and pending_connectivity:
        element_count_known = False
    return _AbaqusDeckInventory(
        node_count=node_count,
        element_count=element_count if element_count_known else None,
        keyword_counts=tuple(sorted(counts.items())),
        part_names=tuple(parts),
        instance_names=tuple(instances),
        material_names=tuple(materials),
        section_count=section_count,
        step_count=step_count,
        include_files=tuple(includes),
    )


@dataclass(frozen=True)
class AbaqusMigrationReport:
    """Side-effect-free inventory for deciding how an input deck can migrate.

    ``preserved`` means AgentFEM retains the source semantics today.  It does
    not mean that every Abaqus numerical formulation has an equivalent native
    solver.  ``planned`` identifies familiar engineering assets that are seen
    and reported but are not silently lowered yet.
    """

    source: Path
    source_sha256: str
    source_graph: AbaqusSourceGraph
    node_count: int
    element_count: int | None
    element_definitions: tuple[AbaqusElementDefinition, ...]
    keyword_inventory: tuple[tuple[str, int, str], ...]
    semantics: AbaqusModelSemantics
    equation_count: int | None
    part_names: tuple[str, ...] = ()
    instance_names: tuple[str, ...] = ()
    material_names: tuple[str, ...] = ()
    section_count: int = 0
    step_count: int = 0
    include_files: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def topology_only_elements(self) -> tuple[str, ...]:
        return tuple(
            item.source_type
            for item in self.element_definitions
            if item.solver_capability in {"topology_only", "not_declared"}
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.abaqus-migration-report",
            "schema_version": "0.2.0",
            "source": str(self.source),
            "source_sha256": self.source_sha256,
            "source_graph": self.source_graph.summary(),
            "node_count": self.node_count,
            "element_count": self.element_count,
            "element_definitions": [item.summary() for item in self.element_definitions],
            "topology_only_elements": list(self.topology_only_elements),
            "keywords": [
                {"keyword": keyword, "count": count, "status": status}
                for keyword, count, status in self.keyword_inventory
            ],
            "equation_count": self.equation_count,
            "parts": list(self.part_names),
            "instances": list(self.instance_names),
            "materials": list(self.material_names),
            "section_count": self.section_count,
            "step_count": self.step_count,
            "include_files": list(self.include_files),
            "model_semantics": self.semantics.summary(),
            "warnings": list(self.warnings),
        }

    def text(self) -> str:
        elements = ", ".join(item.source_type for item in self.element_definitions)
        lines = [
            f"Abaqus source: {self.source}",
            f"Nodes/elements: {self.node_count}/{self.element_count if self.element_count is not None else 'unknown'}",
            f"Element declarations: {elements or '<none>'}",
            f"NSET/ELSET/SURFACE: {len(self.semantics.node_sets)}/{len(self.semantics.element_sets)}/{len(self.semantics.surfaces)}",
            f"Equations: {self.equation_count if self.equation_count is not None else 'requires scoped resolution'}",
            f"Parts/instances/materials: {len(self.part_names)}/{len(self.instance_names)}/{len(self.material_names)}",
            f"Sections/steps/includes: {self.section_count}/{self.step_count}/{len(self.include_files)}",
            f"Source graph: {len(self.source_graph.files)} files, "
            f"{len(self.source_graph.edges)} include edges, "
            f"{'complete' if self.source_graph.complete else 'incomplete'}",
        ]
        if self.topology_only_elements:
            lines.append(
                "Topology-only solver status: " + ", ".join(self.topology_only_elements)
            )
        lines.extend(f"Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def inspect_input(path: str | Path) -> AbaqusMigrationReport:
    """Inspect an Abaqus input deck without converting or solving it."""

    source = Path(path)
    raw = source.read_bytes()
    source_graph = read_source_graph(source)
    deck = _inspect_deck_structure(source)
    definitions = read_element_definitions(source)
    warnings: list[str] = []
    try:
        semantics = read_model_semantics(source)
    except (TypeError, ValueError) as exc:
        semantics = AbaqusModelSemantics()
        warnings.append(
            "Named-set or surface semantics require scoped/nested resolution before "
            f"lowering ({exc})."
        )
    try:
        equation_count: int | None = len(read_equations(source).equations)
    except ValueError as exc:
        if "No equation data" in str(exc):
            equation_count = 0
        else:
            equation_count = None
            warnings.append(
                "Equation terms require scoped set/instance resolution before "
                f"lowering ({exc})."
            )

    inventory = []
    for keyword, count in deck.keyword_counts:
        if keyword in _PRESERVED_KEYWORDS:
            status = "preserved"
        elif keyword in _MIGRATION_PLANNED_KEYWORDS:
            status = "recognized_not_lowered"
        else:
            status = "unclassified"
        inventory.append((keyword, count, status))

    for definition in definitions:
        if definition.solver_capability in {"topology_only", "not_declared"}:
            warnings.append(
                f"{definition.source_type} is retained for topology/source semantics; "
                "no equivalent native element formulation is claimed."
            )
    not_lowered = [
        keyword for keyword, _count, status in inventory
        if status == "recognized_not_lowered"
    ]
    if not_lowered:
        warnings.append(
            "The input deck contains recognized Abaqus assets that require an "
            f"explicit migration decision: {', '.join(not_lowered)}."
        )
    if deck.part_names or deck.instance_names:
        warnings.append(
            "Part and assembly labels are scoped in Abaqus. The inventory is valid, "
            "but direct orphan-mesh lowering requires an explicit instance-aware "
            "flattening step."
        )
    if deck.include_files:
        warnings.append(
            "*INCLUDE dependencies are resolved as a content-addressed source graph; "
            "scoped engineering semantics are not flattened or silently lowered."
        )
    warnings.extend(source_graph.issues)
    return AbaqusMigrationReport(
        source=source,
        source_sha256=sha256(raw).hexdigest(),
        source_graph=source_graph,
        node_count=deck.node_count,
        element_count=deck.element_count,
        element_definitions=definitions,
        keyword_inventory=tuple(inventory),
        semantics=semantics,
        equation_count=equation_count,
        part_names=deck.part_names,
        instance_names=deck.instance_names,
        material_names=deck.material_names,
        section_count=deck.section_count,
        step_count=deck.step_count,
        include_files=deck.include_files,
        warnings=tuple(warnings),
    )


inspect_model = inspect_input


@dataclass(frozen=True)
class AbaqusNodeTable:
    """Abaqus node labels and coordinates in source-file order."""

    labels: np.ndarray
    coordinates: np.ndarray

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        coordinates = np.asarray(self.coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[0] != labels.size:
            raise ValueError("Abaqus node labels and coordinates must have equal length.")
        if coordinates.shape[1] not in {1, 2, 3}:
            raise ValueError("Abaqus node coordinates must have dimension 1, 2, or 3.")
        if np.unique(labels).size != labels.size:
            raise ValueError("Abaqus node labels must be unique.")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Abaqus node coordinates must be finite.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "coordinates", coordinates)

    def index(self, label: int) -> int:
        """Return the source-order index of one node label."""

        matches = np.flatnonzero(self.labels == int(label))
        if matches.size != 1:
            raise KeyError(f"Abaqus node label {label} is not present.")
        return int(matches[0])

    def coordinate(self, label: int) -> np.ndarray:
        """Return a copy of one node coordinate."""

        return self.coordinates[self.index(label)].copy()

    def summary(self) -> dict[str, object]:
        return {
            "kind": "abaqus_node_table",
            "node_count": int(self.labels.size),
            "geometric_dimension": int(self.coordinates.shape[1]),
            "minimum": np.min(self.coordinates, axis=0).tolist(),
            "maximum": np.max(self.coordinates, axis=0).tolist(),
        }


@dataclass(frozen=True)
class EquationTerm:
    """One ``coefficient * nodal_dof`` term in an Abaqus equation."""

    node: int
    dof: int
    coefficient: float

    def __post_init__(self) -> None:
        if int(self.node) <= 0:
            raise ValueError("Abaqus equation node labels must be positive.")
        if int(self.dof) <= 0:
            raise ValueError("Abaqus equation dof numbers must be positive.")
        if not np.isfinite(float(self.coefficient)):
            raise ValueError("Abaqus equation coefficients must be finite.")


@dataclass(frozen=True)
class LinearEquation:
    """One homogeneous Abaqus ``*EQUATION`` constraint."""

    terms: tuple[EquationTerm, ...]

    def __post_init__(self) -> None:
        if len(self.terms) < 2:
            raise ValueError("An Abaqus equation requires at least two terms.")
        if self.terms[0].coefficient == 0.0:
            raise ValueError("The first Abaqus equation coefficient cannot be zero.")

    @property
    def slave(self) -> tuple[int, int]:
        """Return the node/dof eliminated by Abaqus/Standard."""

        return self.terms[0].node, self.terms[0].dof


@dataclass(frozen=True)
class AbaqusEquationSet:
    """A parsed collection of Abaqus linear constraint equations."""

    equations: tuple[LinearEquation, ...]
    source: Path | None = None

    def __post_init__(self) -> None:
        slaves = [equation.slave for equation in self.equations]
        if len(set(slaves)) != len(slaves):
            raise ValueError("An Abaqus dof is eliminated by more than one equation.")

    def summary(self) -> dict[str, object]:
        term_counts: dict[int, int] = {}
        component_counts: dict[int, int] = {}
        for equation in self.equations:
            term_counts[len(equation.terms)] = term_counts.get(len(equation.terms), 0) + 1
            component = equation.terms[0].dof
            component_counts[component] = component_counts.get(component, 0) + 1
        return {
            "kind": "abaqus_equation_set",
            "source": None if self.source is None else str(self.source),
            "equation_count": len(self.equations),
            "term_counts": term_counts,
            "slave_dofs_by_component": component_counts,
        }


@dataclass(frozen=True)
class AbaqusMeshImport:
    """DOLFINx mesh together with Abaqus labels and conversion evidence."""

    fem_mesh: object
    nodes: AbaqusNodeTable
    conversion: object

    @property
    def domain(self):
        return self.fem_mesh.domain

    @property
    def cell_tags(self):
        return self.fem_mesh.cell_tags

    @property
    def facet_tags(self):
        return self.fem_mesh.facet_tags

    @property
    def element_definitions(self) -> tuple[dict[str, object], ...]:
        values = self.conversion.source_metadata.get(
            "abaqus_element_definitions",
            (),
        )
        return tuple(dict(value) for value in values)

    @property
    def model_semantics(self) -> dict[str, object]:
        """Return preserved NSET/ELSET/SURFACE source semantics."""

        return dict(
            self.conversion.source_metadata.get("abaqus_model_semantics", {})
        )

    @property
    def node_sets(self) -> dict[str, object]:
        return dict(self.model_semantics.get("node_sets", {}))

    @property
    def element_sets(self) -> dict[str, object]:
        return dict(self.model_semantics.get("element_sets", {}))

    @property
    def surfaces(self) -> dict[str, object]:
        return dict(self.model_semantics.get("surfaces", {}))

    def surface_faces(self, name: str) -> tuple[tuple[int, str | None], ...]:
        """Expand an element-based surface to source element-label/face pairs."""

        selected = self.surfaces.get(str(name).upper())
        if selected is None:
            raise KeyError(f"Abaqus surface {name!r} is not present.")
        if selected.get("surface_type") != "ELEMENT":
            raise TypeError(f"Abaqus surface {name!r} is not element based.")
        expanded = []
        for entry in selected.get("entries", ()):
            reference = str(entry["reference"]).upper()
            labels = self.element_sets.get(reference, {}).get("labels")
            if labels is None:
                try:
                    labels = (int(reference),)
                except ValueError as exc:
                    raise KeyError(
                        f"Surface {name!r} references unknown ELSET {reference!r}."
                    ) from exc
            expanded.extend((int(label), entry.get("face")) for label in labels)
        return tuple(expanded)

    @property
    def elements(self) -> AbaqusElementTable:
        """Return supported source connectivity used by semantic adapters."""

        return read_element_table(self.conversion.source_path)

    def surface_corner_labels(self, name: str) -> tuple[tuple[int, ...], ...]:
        """Expand an Abaqus surface to ordered corner-node labels."""

        selected = []
        table = self.elements
        for element_label, face in self.surface_faces(name):
            if face is None:
                raise NotImplementedError(
                    "Free-surface generation without an Abaqus face identifier "
                    "is not inferred by this adapter."
                )
            selected.append(table.element(element_label).face_corner_labels(face))
        return tuple(selected)

    def cohesive_interface(
        self,
        *,
        positive_elset: str,
        surface: str,
    ):
        """Lower Abaqus ``ELSET``/``SURFACE`` semantics to a split interface.

        The first source-semantic route consumes linear ``C3D4`` solids and
        an explicit element-based internal surface.  The ELSET declares which
        bulk partition receives duplicate nodes; the SURFACE independently
        proves the exact physical faces.  A mismatch is rejected rather than
        allowing a named surface to silently select another partition edge.
        """

        from agentfem import interfaces

        set_record = self.element_sets.get(str(positive_elset).upper())
        if set_record is None:
            raise KeyError(f"Abaqus ELSET {positive_elset!r} is not present.")
        table = self.elements
        if any(item.element_type.upper() != "C3D4" for item in table.elements):
            raise NotImplementedError(
                "Direct 3D cohesive lowering currently requires one C3D4 block; "
                "quadratic faces need a matching higher-order cohesive kernel."
            )
        label_to_node = {
            int(label): index for index, label in enumerate(self.nodes.labels.tolist())
        }
        element_to_cell = {
            int(item.label): index for index, item in enumerate(table.elements)
        }
        cells = np.asarray(
            [
                [label_to_node[int(label)] for label in item.connectivity]
                for item in table.elements
            ],
            dtype=int,
        )
        try:
            positive_cells = np.asarray(
                [element_to_cell[int(label)] for label in set_record.get("labels", ())],
                dtype=int,
            )
        except KeyError as exc:
            raise KeyError(
                f"Abaqus ELSET {positive_elset!r} references an unknown element."
            ) from exc
        requested = {
            tuple(
                sorted(label_to_node[int(label)] for label in corner_labels)
            )
            for corner_labels in self.surface_corner_labels(surface)
        }
        split = interfaces.split_conforming_cell_interface(
            np.asarray(self.nodes.coordinates, dtype=float),
            cells,
            positive_cells=positive_cells,
        )
        recovered = {
            tuple(sorted(int(value) for value in facet))
            for facet in split.negative_facets
        }
        if requested != recovered:
            missing = sorted(requested - recovered)[:4]
            extra = sorted(recovered - requested)[:4]
            raise ValueError(
                "Abaqus SURFACE does not equal the interface implied by the "
                f"declared ELSET: missing={missing}, extra={extra}."
            )
        return split

    def node_set(self, name: str, *, tolerance: float | None = None):
        """Promote an Abaqus ``NSET`` to a strong-constraint-ready node region."""

        from . import NodeRegion

        selected = self.node_sets.get(str(name).upper())
        if selected is None:
            raise KeyError(f"Abaqus node set {name!r} is not present.")
        source_labels = tuple(int(label) for label in selected.get("labels", ()))
        if not source_labels:
            raise ValueError(f"Abaqus node set {name!r} is empty.")
        source_coordinates = np.asarray(
            [self.nodes.coordinate(label) for label in source_labels], dtype=float
        )
        scale = max(1.0, float(np.max(np.ptp(self.nodes.coordinates, axis=0))))
        selected_tolerance = (
            max(1.0e-12, 1.0e-9 * scale)
            if tolerance is None
            else float(tolerance)
        )
        if selected_tolerance <= 0.0 or not np.isfinite(selected_tolerance):
            raise ValueError("node-set matching tolerance must be positive.")
        for left in range(len(source_labels)):
            for right in range(left + 1, len(source_labels)):
                distance = np.linalg.norm(
                    source_coordinates[left] - source_coordinates[right]
                )
                if distance <= selected_tolerance:
                    raise ValueError(
                        f"Abaqus node set {name!r} contains coincident source "
                        "nodes whose runtime identity would be ambiguous."
                    )
        runtime_coordinates = np.asarray(self.domain.geometry.x, dtype=float)
        found = {
            label
            for label, coordinate in zip(source_labels, source_coordinates)
            if runtime_coordinates.size
            and np.min(np.linalg.norm(runtime_coordinates - coordinate, axis=1))
            <= selected_tolerance
        }
        requested = set(source_labels)
        missing = requested - set().union(*self.domain.comm.allgather(found))
        if missing:
            raise ValueError(
                f"Abaqus node set {name!r} contains {len(missing)} node(s) "
                "outside the selected solver domain; "
                f"examples: {sorted(missing)[:8]}."
            )
        return NodeRegion(
            name=str(name),
            domain=self.domain,
            coordinates=source_coordinates,
            source_labels=source_labels,
            tolerance=selected_tolerance,
        )

    def element_set(self, name: str):
        """Promote a preserved Abaqus ``ELSET`` to a material-ready cell region.

        The converter stores named cell sets as DOLFINx ``MeshTags``.  A tag
        can represent the requested scientific region only when it owns every
        source element in that set.  This count check deliberately rejects
        ambiguous overlapping-set conversions instead of returning a partial
        material region.
        """

        from . import cell_region

        key = str(name).upper()
        selected = self.element_sets.get(key)
        if selected is None:
            raise KeyError(f"Abaqus element set {name!r} is not present.")
        tags_by_name = {
            str(label).upper(): int(tag)
            for label, tag in self.conversion.region_tags.items()
        }
        tag = tags_by_name.get(key)
        if tag is None or self.cell_tags is None:
            raise ValueError(
                f"Abaqus element set {name!r} has no unambiguous converted "
                "cell tag. Reconvert the mesh or resolve overlapping ELSETs."
            )
        owned_cells = self.domain.topology.index_map(self.domain.topology.dim).size_local
        local_count = int(
            np.count_nonzero(
                (np.asarray(self.cell_tags.values) == tag)
                & (np.asarray(self.cell_tags.indices) < owned_cells)
            )
        )
        runtime_count = int(self.domain.comm.allreduce(local_count))
        source_count = len(tuple(selected.get("labels", ())))
        if runtime_count != source_count:
            raise ValueError(
                f"Abaqus element set {name!r} contains {source_count} source "
                f"elements but its converted tag owns {runtime_count}. The set "
                "overlaps another ELSET and cannot safely define a material region."
            )
        return cell_region(
            self.domain,
            self.cell_tags,
            tag=tag,
            name=str(name),
        )

    def boundary(
        self,
        name: str,
        *,
        tag: int = 1,
        tolerance: float | None = None,
    ):
        """Reconstruct an exterior Abaqus ``SURFACE`` as a boundary region.

        Matching uses source node labels recovered from vertex coordinates and
        exact solid-family face conventions.  It does not guess a face from a
        normal or bounding box.  Internal element faces are rejected because
        an exterior ``ds`` region and an interior ``dS`` interface have
        different finite-element meanings.
        """

        from dolfinx import mesh as dxmesh
        from . import mark_facets, tagged_boundary_region

        domain = self.domain
        if domain.topology.dim != 3:
            raise NotImplementedError(
                "Abaqus element-face reconstruction currently supports 3D solids."
            )
        vertex_labels = _runtime_vertex_source_labels(
            domain, self.nodes, tolerance=tolerance
        )

        targets = {
            tuple(sorted(labels)) for labels in self.surface_corner_labels(name)
        }
        fdim = domain.topology.dim - 1
        domain.topology.create_connectivity(fdim, 0)
        domain.topology.create_connectivity(fdim, domain.topology.dim)
        domain.topology.create_connectivity(0, domain.topology.dim)
        connectivity = domain.topology.connectivity(fdim, 0)
        exterior = np.asarray(
            dxmesh.exterior_facet_indices(domain.topology), dtype=np.int32
        )
        selected_facets = []
        for facet in exterior:
            vertices = np.asarray(connectivity.links(int(facet)), dtype=np.int32)
            labels = tuple(sorted(vertex_labels[int(vertex)] for vertex in vertices))
            if labels in targets:
                selected_facets.append(int(facet))

        local_keys = set()
        for facet in selected_facets:
            vertices = np.asarray(connectivity.links(facet), dtype=np.int32)
            local_keys.add(tuple(sorted(vertex_labels[int(vertex)] for vertex in vertices)))
        global_keys = set().union(*domain.comm.allgather(local_keys))
        missing = targets - global_keys
        if missing:
            preview = sorted(missing)[:4]
            raise ValueError(
                f"Abaqus surface {name!r} contains {len(missing)} face(s) that "
                "are not exterior facets of the selected solver domain; "
                f"examples: {preview}."
            )
        facet_tags = mark_facets(domain, selected_facets, int(tag))
        return tagged_boundary_region(
            domain,
            facet_tags,
            tag=int(tag),
            name=str(name),
        )

    def require_formulation(
        self,
        *supported: str,
        operation: str = "selected solution procedure",
    ) -> None:
        """Reject known source formulations not consumed by a solver path."""

        accepted = {str(value) for value in supported}
        incompatible = [
            value
            for value in self.element_definitions
            if value.get("formulation") not in accepted
            and value.get("formulation") != "source_defined"
        ]
        if incompatible:
            labels = ", ".join(
                f"{value.get('source_type')} ({value.get('formulation')})"
                for value in incompatible
            )
            raise NotImplementedError(
                f"{operation} does not consume the imported Abaqus formulation: "
                f"{labels}. Neutral-mesh conversion preserves geometry/topology "
                "but cannot replace the element's pressure or enhanced variables."
            )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "abaqus_mesh_import",
            "source": str(self.conversion.source_path),
            "cell_type": self.conversion.cell_type,
            "nodes": self.nodes.summary(),
            "mesh": self.fem_mesh.summary().as_dict(),
            "region_tags": self.conversion.region_tags,
            "element_definitions": self.element_definitions,
            "node_sets": self.node_sets,
            "element_sets": self.element_sets,
            "surfaces": self.surfaces,
            "source_metadata": self.conversion.source_metadata,
            "warnings": self.conversion.warnings,
        }


def read_node_table(path: str | Path) -> AbaqusNodeTable:
    """Read all ``*NODE`` sections while preserving Abaqus node labels."""

    path = Path(path)
    labels: list[int] = []
    coordinates: list[tuple[float, ...]] = []
    in_nodes = False
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            in_nodes = line.split(",", 1)[0].strip().upper() == "*NODE"
            continue
        if not in_nodes:
            continue
        values = _csv_values(line)
        if len(values) < 2:
            raise ValueError(f"Invalid Abaqus node at {path}:{line_number}.")
        labels.append(int(values[0]))
        coordinates.append(tuple(float(value) for value in values[1:]))
    if not labels:
        raise ValueError(f"No *NODE data were found in {path}.")
    dimensions = {len(value) for value in coordinates}
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent Abaqus node dimensions in {path}.")
    return AbaqusNodeTable(np.asarray(labels), np.asarray(coordinates))


def _runtime_vertex_source_labels(
    domain,
    nodes: AbaqusNodeTable,
    *,
    tolerance: float | None = None,
) -> dict[int, int]:
    """Match local topology vertices to source labels with an explicit tolerance."""

    from dolfinx.cpp.mesh import entities_to_geometry

    source_coordinates = np.asarray(nodes.coordinates, dtype=float)
    if source_coordinates.shape[1] != domain.geometry.dim:
        raise ValueError("Source and runtime geometric dimensions differ.")
    scale = max(1.0, float(np.max(np.ptp(source_coordinates, axis=0))))
    selected_tolerance = (
        max(1.0e-12, 1.0e-9 * scale)
        if tolerance is None
        else float(tolerance)
    )
    if selected_tolerance <= 0.0 or not np.isfinite(selected_tolerance):
        raise ValueError("node matching tolerance must be positive.")

    buckets: dict[tuple[int, ...], list[int]] = {}
    for label, coordinate in zip(nodes.labels, source_coordinates):
        key = tuple(np.rint(coordinate / selected_tolerance).astype(np.int64))
        buckets.setdefault(key, []).append(int(label))

    def source_label(coordinate) -> int:
        key = tuple(np.rint(coordinate / selected_tolerance).astype(np.int64))
        candidates = []
        for shift in product((-1, 0, 1), repeat=domain.geometry.dim):
            candidates.extend(
                buckets.get(
                    tuple(key[index] + shift[index] for index in range(len(key))),
                    (),
                )
            )
        if not candidates:
            raise ValueError("A runtime mesh vertex has no matching Abaqus source node.")
        distances = np.asarray(
            [np.linalg.norm(nodes.coordinate(label) - coordinate) for label in candidates]
        )
        closest = int(np.argmin(distances))
        if distances[closest] > selected_tolerance:
            raise ValueError(
                "A runtime mesh vertex exceeds the Abaqus matching tolerance."
            )
        tied = np.flatnonzero(
            np.isclose(
                distances,
                distances[closest],
                atol=np.finfo(float).eps * scale,
                rtol=0.0,
            )
        )
        if tied.size > 1:
            raise ValueError(
                "Multiple Abaqus source nodes share one runtime vertex coordinate."
            )
        return int(candidates[closest])

    domain.topology.create_connectivity(0, domain.topology.dim)
    vertex_map = domain.topology.index_map(0)
    vertices = np.arange(
        vertex_map.size_local + vertex_map.num_ghosts, dtype=np.int32
    )
    geometry_nodes = np.asarray(
        entities_to_geometry(domain._cpp_object, 0, vertices, False),
        dtype=np.int32,
    ).reshape(-1)
    if geometry_nodes.size != vertices.size:
        raise RuntimeError("A topology vertex did not map to exactly one geometry node.")
    return {
        int(vertex): source_label(domain.geometry.x[int(geometry_node)])
        for vertex, geometry_node in zip(vertices, geometry_nodes)
    }


def read_equations(path: str | Path) -> AbaqusEquationSet:
    """Read Abaqus ``*EQUATION`` data or a keyword-free included equation file.

    Abaqus allows the ``3*N`` term values to continue across multiple lines.
    The include used by the example contains only equation data, which is also
    accepted here.
    """

    path = Path(path)
    records = _data_records(path)
    equations: list[LinearEquation] = []
    index = 0
    while index < len(records):
        line_number, values = records[index]
        index += 1
        if len(values) != 1:
            raise ValueError(
                f"Expected an equation term count at {path}:{line_number}, got {values!r}."
            )
        term_count = int(values[0])
        if term_count < 2:
            raise ValueError(
                f"Abaqus equation at {path}:{line_number} has fewer than two terms."
            )
        flat: list[str] = []
        while len(flat) < 3 * term_count and index < len(records):
            _, continuation = records[index]
            index += 1
            flat.extend(continuation)
        if len(flat) != 3 * term_count:
            raise ValueError(
                f"Abaqus equation at {path}:{line_number} expected "
                f"{3 * term_count} term values, got {len(flat)}."
            )
        terms = tuple(
            EquationTerm(
                node=int(flat[offset]),
                dof=int(flat[offset + 1]),
                coefficient=float(flat[offset + 2]),
            )
            for offset in range(0, len(flat), 3)
        )
        equations.append(LinearEquation(terms))
    if not equations:
        raise ValueError(f"No equation data were found in {path}.")
    return AbaqusEquationSet(tuple(equations), source=path)


def displacement_in_source_order(
    displacement,
    nodes: AbaqusNodeTable,
    *,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Return a vector field ordered by Abaqus source node labels.

    Under MPI each rank contributes its owned dofs and receives the complete
    source-ordered array. This keeps history extraction deterministic without
    treating ghost values as independent data.
    """

    function = getattr(displacement, "value", displacement)
    space = function.function_space
    comm = space.mesh.comm
    block_size = int(space.dofmap.index_map_bs)
    coordinates = np.asarray(space.tabulate_dof_coordinates(), dtype=float)
    values = np.asarray(function.x.array).reshape(-1, block_size)
    dimension = coordinates.shape[1]
    source_coordinates = np.zeros((nodes.labels.size, dimension), dtype=float)
    copied_dimension = min(dimension, nodes.coordinates.shape[1])
    source_coordinates[:, :copied_dimension] = nodes.coordinates[:, :copied_dimension]
    source_buckets: dict[tuple[int, ...], list[int]] = {}
    for source_index, coordinate in enumerate(source_coordinates):
        key = tuple(np.rint(coordinate / tolerance).astype(np.int64))
        source_buckets.setdefault(key, []).append(source_index)
    local: dict[int, np.ndarray] = {}
    owned_count = space.dofmap.index_map.size_local
    for block, coordinate in enumerate(coordinates[:owned_count]):
        key = tuple(np.rint(coordinate / tolerance).astype(np.int64))
        candidates: list[int] = []
        for shift in product((-1, 0, 1), repeat=dimension):
            candidates.extend(
                source_buckets.get(
                    tuple(key[i] + shift[i] for i in range(dimension)),
                    (),
                )
            )
        if not candidates:
            continue
        distances = np.linalg.norm(
            source_coordinates[candidates] - coordinate,
            axis=1,
        )
        closest = int(np.argmin(distances))
        if distances[closest] > tolerance:
            continue
        source_index = int(candidates[closest])
        if source_index in local:
            raise ValueError(
                f"Abaqus node {int(nodes.labels[source_index])} matched "
                "multiple owned field dofs."
            )
        local[source_index] = values[block].copy()

    gathered: dict[int, np.ndarray] = {}
    for rank_values in comm.allgather(local):
        overlap = set(gathered) & set(rank_values)
        if overlap:
            raise ValueError(
                "Abaqus source nodes have duplicate distributed owners: "
                f"{sorted(overlap)[:8]}."
            )
        gathered.update(rank_values)
    missing = set(range(nodes.labels.size)) - set(gathered)
    if missing:
        labels = [int(nodes.labels[index]) for index in sorted(missing)[:8]]
        raise ValueError(
            f"Abaqus nodes have no matching distributed field dofs: {labels}."
        )
    ordered = np.empty((nodes.labels.size, block_size), dtype=float)
    for source_index, value in gathered.items():
        ordered[source_index] = value
    return ordered


def write_deformation_vtu_pair(
    source_path: str | Path,
    nodes: AbaqusNodeTable,
    displacement,
    output_directory: str | Path,
    *,
    deformation_scale: float = 1.0,
    basename: str = "periodic_cell",
) -> tuple[Path, Path]:
    """Write ParaView-ready undeformed and deformed Abaqus meshes."""

    from .formats import require_meshio

    if not np.isfinite(deformation_scale):
        raise ValueError("deformation_scale must be finite.")
    meshio = require_meshio()
    source = meshio.read(Path(source_path), file_format="abaqus")
    values = displacement_in_source_order(displacement, nodes)
    points = np.asarray(source.points, dtype=float)
    if values.shape != points.shape:
        raise ValueError(
            "Abaqus point and displacement shapes differ: "
            f"{points.shape} versus {values.shape}."
        )
    magnitude = np.linalg.norm(values, axis=1)
    point_data = {
        "Displacement": values,
        "DisplacementMagnitude": magnitude,
        "AbaqusNodeLabel": nodes.labels,
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    undeformed_path = output / f"{basename}_undeformed.vtu"
    deformed_path = output / f"{basename}_deformed.vtu"
    common = {
        "cells": source.cells,
        "cell_data": dict(source.cell_data),
        "field_data": dict(source.field_data),
    }
    meshio.write(
        undeformed_path,
        meshio.Mesh(points=points, point_data=point_data, **common),
    )
    meshio.write(
        deformed_path,
        meshio.Mesh(
            points=points + float(deformation_scale) * values,
            point_data=point_data,
            **common,
        ),
    )
    return undeformed_path, deformed_path


def periodic_cell_volume(
    nodes: AbaqusNodeTable,
    *,
    anchor_node: int,
    reference_nodes,
) -> float:
    """Return the reference parallelepiped volume from four control nodes."""

    references = tuple(int(node) for node in reference_nodes)
    if len(references) != 3:
        raise ValueError("periodic_cell_volume requires three reference nodes.")
    origin = nodes.coordinate(int(anchor_node))
    lattice = np.column_stack(
        [nodes.coordinate(node) - origin for node in references]
    )
    volume = abs(float(np.linalg.det(lattice)))
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("Periodic-cell control nodes define a degenerate volume.")
    return volume


def _data_records(path: Path) -> list[tuple[int, list[str]]]:
    records: list[tuple[int, list[str]]] = []
    in_equations = True
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            in_equations = line.split(",", 1)[0].strip().upper() == "*EQUATION"
            continue
        if in_equations:
            records.append((line_number, _csv_values(line)))
    return records


def _csv_values(line: str) -> list[str]:
    return [value.strip() for value in line.split(",") if value.strip()]
