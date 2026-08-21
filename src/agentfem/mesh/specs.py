"""Versioned geometry specifications lowered to AgentFEM meshes.

The contract is intentionally small: a geometry is a mapping with ``type``
and optional ``geometry_params``.  It is useful for configuration files,
parameter campaigns, GUI/agent front ends, and benchmark adapters without
making any of those consumers part of the mesh kernel.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, pi, radians, sin

import numpy as np
from mpi4py import MPI


SUPPORTED_GEOMETRIES = (
    "unit_square",
    "unit_cube",
    "l_shape",
    "square_with_hole",
    "multi_hole",
    "circle",
    "annulus",
    "sector",
    "star",
    "gear",
    "t_junction",
    "eccentric_annulus",
    "dumbbell",
    "periodic_square",
)


def from_geometry_spec(
    specification: Mapping[str, object],
    *,
    resolution: int = 32,
    comm: MPI.Comm = MPI.COMM_WORLD,
):
    """Create an :class:`agentfem.mesh.FEMMesh` from a public geometry spec.

    Structured rectangular domains use DOLFINx directly. General planar
    domains use the optional Gmsh integration and carry one domain and one
    exterior boundary physical group. ``periodic_square`` describes geometry
    only; periodic equality remains an explicit constraint decision.
    """

    from agentfem import mesh as mesh_api

    spec = _mapping(specification, "geometry specification")
    kind = str(spec.get("type", "")).strip().lower()
    if kind not in SUPPORTED_GEOMETRIES:
        raise ValueError(
            f"Unsupported geometry type {kind!r}; expected one of "
            f"{SUPPORTED_GEOMETRIES}."
        )
    selected_resolution = int(resolution)
    if selected_resolution < 2:
        raise ValueError("Geometry resolution must be at least two.")
    if kind == "unit_square":
        domain = mesh_api.rectangle(
            (0.0, 0.0),
            (1.0, 1.0),
            (selected_resolution, selected_resolution),
            comm=comm,
            cell_type="triangle",
        )
        return mesh_api.FEMMesh(domain)
    if kind == "unit_cube":
        domain = mesh_api.cuboid(
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (selected_resolution,) * 3,
            comm=comm,
            cell_type="tetrahedron",
        )
        return mesh_api.FEMMesh(domain)
    if kind == "periodic_square":
        params = _mapping(spec.get("geometry_params", {}), "geometry_params")
        bounds = tuple(
            float(value)
            for value in params.get(
                "bounds",
                params.get("extents", (0.0, 1.0, 0.0, 1.0)),
            )
        )
        if len(bounds) != 4 or bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
            raise ValueError(
                "periodic_square bounds must be (xmin, xmax, ymin, ymax) "
                "with positive spans."
            )
        domain = mesh_api.rectangle(
            (bounds[0], bounds[2]),
            (bounds[1], bounds[3]),
            (selected_resolution, selected_resolution),
            comm=comm,
            cell_type="triangle",
        )
        return mesh_api.FEMMesh(domain)

    gmsh = mesh_api.require_gmsh()
    initialized_here = not gmsh.isInitialized()
    if initialized_here:
        gmsh.initialize()
    try:
        if comm.rank == 0:
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.clear()
            gmsh.model.add(f"agentfem_{kind}")
            surfaces = _build_occ_geometry(gmsh, kind, spec)
            gmsh.model.occ.synchronize()
            surface_tags = [int(tag) for dim, tag in surfaces if int(dim) == 2]
            if not surface_tags:
                raise RuntimeError(f"Geometry {kind!r} produced no planar domain.")
            gmsh.model.addPhysicalGroup(2, surface_tags, 1)
            gmsh.model.setPhysicalName(2, 1, "domain")
            boundary = gmsh.model.getBoundary(
                [(2, tag) for tag in surface_tags],
                combined=True,
                oriented=False,
                recursive=False,
            )
            curve_tags = sorted({int(tag) for dim, tag in boundary if int(dim) == 1})
            if curve_tags:
                gmsh.model.addPhysicalGroup(1, curve_tags, 1)
                gmsh.model.setPhysicalName(1, 1, "boundary")
            characteristic_length = float(
                spec.get("char_length", _geometry_scale(kind, spec) / selected_resolution)
            )
            if characteristic_length <= 0.0:
                raise ValueError("Geometry char_length must be positive.")
            gmsh.option.setNumber("Mesh.MeshSizeMin", characteristic_length)
            gmsh.option.setNumber("Mesh.MeshSizeMax", characteristic_length)
            gmsh.model.mesh.generate(2)
        return mesh_api.import_gmsh_model(gmsh.model, comm, model_rank=0, gdim=2)
    finally:
        if initialized_here:
            gmsh.finalize()


def _build_occ_geometry(gmsh, kind: str, spec: Mapping[str, object]):
    occ = gmsh.model.occ
    params = _mapping(spec.get("geometry_params", {}), "geometry_params")

    if kind == "l_shape":
        vertices = params.get(
            "vertices",
            ((0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)),
        )
        return [(2, _polygon(occ, vertices))]
    if kind == "circle":
        center = _point2(params.get("center", (0.5, 0.5)))
        return [(2, occ.addDisk(*center, 0.0, float(params.get("radius", 0.5)), float(params.get("radius", 0.5))))]
    if kind == "annulus":
        center = _point2(params.get("center", (0.5, 0.5)))
        outer = occ.addDisk(*center, 0.0, float(params.get("outer_r", 0.5)), float(params.get("outer_r", 0.5)))
        inner = occ.addDisk(*center, 0.0, float(params.get("inner_r", 0.25)), float(params.get("inner_r", 0.25)))
        return occ.cut([(2, outer)], [(2, inner)], removeObject=True, removeTool=True)[0]
    if kind in {"square_with_hole", "multi_hole"}:
        outer = tuple(float(v) for v in params.get("outer", (0, 1, 0, 1)))
        base = occ.addRectangle(outer[0], outer[2], 0.0, outer[1] - outer[0], outer[3] - outer[2])
        holes: list[tuple[int, int]] = []
        if kind == "multi_hole":
            for entry in params.get("holes", ()):
                item = _mapping(entry, "hole")
                center = _point2(item.get("c", (0.5, 0.5)))
                radius = float(item.get("r", 0.1))
                holes.append((2, occ.addDisk(*center, 0.0, radius, radius)))
        else:
            hole = _mapping(params.get("inner_hole", {}), "inner_hole")
            hole_kind = str(hole.get("type", "circle"))
            if hole_kind == "circle":
                center = _point2(hole.get("center", (0.5, 0.5)))
                radius = float(hole.get("radius", 0.2))
                holes.append((2, occ.addDisk(*center, 0.0, radius, radius)))
            elif hole_kind == "rect":
                box = tuple(float(v) for v in hole.get("bbox", (0.4, 0.6, 0.4, 0.6)))
                holes.append((2, occ.addRectangle(box[0], box[2], 0.0, box[1] - box[0], box[3] - box[2])))
            elif hole_kind == "polygon":
                holes.append((2, _polygon(occ, hole.get("vertices", ((0.4, 0.4), (0.6, 0.4), (0.5, 0.7))))))
            else:
                raise ValueError(f"Unsupported inner-hole type {hole_kind!r}.")
        return occ.cut([(2, base)], holes, removeObject=True, removeTool=True)[0]
    if kind == "t_junction":
        horizontal = tuple(float(v) for v in params.get("horizontal_rect", (0, 1, 0.8, 1)))
        vertical = tuple(float(v) for v in params.get("vertical_rect", (0.4, 0.6, 0, 0.8)))
        first = occ.addRectangle(horizontal[0], horizontal[2], 0.0, horizontal[1] - horizontal[0], horizontal[3] - horizontal[2])
        second = occ.addRectangle(vertical[0], vertical[2], 0.0, vertical[1] - vertical[0], vertical[3] - vertical[2])
        return occ.fuse([(2, first)], [(2, second)], removeObject=True, removeTool=True)[0]
    if kind == "eccentric_annulus":
        outer_cfg = _mapping(params.get("outer_circle", {"c": (0.5, 0.5), "r": 0.5}), "outer_circle")
        inner_cfg = _mapping(params.get("inner_circle", {"c": (0.65, 0.5), "r": 0.2}), "inner_circle")
        outer_c, inner_c = _point2(outer_cfg["c"]), _point2(inner_cfg["c"])
        outer_r, inner_r = float(outer_cfg["r"]), float(inner_cfg["r"])
        outer = occ.addDisk(*outer_c, 0.0, outer_r, outer_r)
        inner = occ.addDisk(*inner_c, 0.0, inner_r, inner_r)
        return occ.cut([(2, outer)], [(2, inner)], removeObject=True, removeTool=True)[0]
    if kind == "dumbbell":
        radius = float(params.get("radius", 0.25))
        left_cfg = _mapping(
            params.get(
                "left_circle",
                {"c": params.get("left_center", (0.25, 0.5)), "r": radius},
            ),
            "left_circle",
        )
        right_cfg = _mapping(
            params.get(
                "right_circle",
                {"c": params.get("right_center", (0.75, 0.5)), "r": radius},
            ),
            "right_circle",
        )
        left_center = _point2(left_cfg["c"])
        right_center = _point2(right_cfg["c"])
        bar_width = float(params.get("bar_width", 0.2))
        bar_center = 0.5 * (left_center[1] + right_center[1])
        bridge = _mapping(
            params.get(
                "bridge",
                {
                    "x_min": left_center[0],
                    "x_max": right_center[0],
                    "y_min": bar_center - 0.5 * bar_width,
                    "y_max": bar_center + 0.5 * bar_width,
                },
            ),
            "bridge",
        )
        lc, rc = _point2(left_cfg["c"]), _point2(right_cfg["c"])
        left = occ.addDisk(*lc, 0.0, float(left_cfg["r"]), float(left_cfg["r"]))
        right = occ.addDisk(*rc, 0.0, float(right_cfg["r"]), float(right_cfg["r"]))
        bar = occ.addRectangle(float(bridge["x_min"]), float(bridge["y_min"]), 0.0, float(bridge["x_max"]) - float(bridge["x_min"]), float(bridge["y_max"]) - float(bridge["y_min"]))
        return occ.fuse([(2, left)], [(2, right), (2, bar)], removeObject=True, removeTool=True)[0]
    if kind == "sector":
        center = _point2(params.get("center", (0, 0)))
        radius = float(params.get("radius", 1.0))
        angle = radians(float(params.get("angle", 90.0)))
        vertices = [center]
        vertices.extend((center[0] + radius * cos(a), center[1] + radius * sin(a)) for a in np.linspace(0.0, angle, 25))
        return [(2, _polygon(occ, vertices))]
    if kind in {"star", "gear"}:
        center = _point2(params.get("center", (0, 0)))
        count = int(params.get("points" if kind == "star" else "teeth", 5 if kind == "star" else 8))
        inner = float(params.get("inner_r", params.get("base_r", 0.5)))
        outer = float(params.get("outer_r", inner + float(params.get("tooth_h", 0.2))))
        offset = -0.5 * pi if kind == "star" else 0.0
        vertices = []
        for index in range(2 * count):
            radius = outer if index % 2 == 0 else inner
            angle = offset + index * pi / count
            vertices.append((center[0] + radius * cos(angle), center[1] + radius * sin(angle)))
        return [(2, _polygon(occ, vertices))]
    raise ValueError(f"No OCC builder for geometry {kind!r}.")


def _polygon(occ, vertices) -> int:
    points = [_point2(value) for value in vertices]
    if len(points) < 3:
        raise ValueError("A polygon needs at least three vertices.")
    tags = [occ.addPoint(x, y, 0.0) for x, y in points]
    lines = [occ.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
    return occ.addPlaneSurface([occ.addCurveLoop(lines)])


def _geometry_scale(kind: str, spec: Mapping[str, object]) -> float:
    params = _mapping(spec.get("geometry_params", {}), "geometry_params")
    if kind == "sector":
        return float(params.get("radius", 1.0))
    if kind in {"annulus", "circle"}:
        return 2.0 * float(params.get("outer_r", params.get("radius", 0.5)))
    return 1.0


def _point2(value) -> tuple[float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 2:
        raise ValueError("A planar point must contain two coordinates.")
    return values


def _mapping(value, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value
