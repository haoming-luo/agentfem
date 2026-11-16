# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Consistent load integration for composite (not quadratic) shape functions."""

import numpy as np
from .. import loads
from ..provenance import collective_call
from .composite_tet import _EDGES


def _constant(value, size):
    raw = getattr(value, "value", value)
    try:
        array = np.asarray(raw, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise NotImplementedError(
            "Composite distributed loads require spatially constant coefficients."
        ) from exc
    if array.size != size or not np.isfinite(array).all():
        raise ValueError(f"Composite load requires {size} finite components.")
    return array


class CompositeDistributedLoad:
    def __init__(self, internal, asset):
        import basix

        self.internal = internal
        self.asset = asset
        self.base = asset.load if isinstance(asset, loads.AmplitudeLoad) else asset
        V = internal.displacement.function_space
        self.domain = V.mesh
        self.weights = []

        def prepare():
            base = self.base
            if type(base) is loads.BodyLoad:
                self.kind = "body"
                _constant(base.value, 3)
                measure = base.measure
                if (
                    measure.ufl_domain() is not None
                    and measure.ufl_domain() != self.domain.ufl_domain()
                ):
                    raise ValueError("Body load measure belongs to another mesh.")
                if measure.integral_type() != "cell":
                    raise ValueError("Body force requires a cell measure.")
                sid = measure.subdomain_id()
                tags = measure.subdomain_data()
                if sid == "everywhere":
                    active = set(range(len(internal.elements)))
                elif isinstance(sid, (int, np.integer)) and tags is not None:
                    active = set(map(int, tags.find(sid)))
                else:
                    raise NotImplementedError(
                        "Composite body force supports full-domain or tagged cell measures."
                    )
                for c, e in enumerate(internal.elements):
                    if c in active:
                        self.weights.append(
                            (internal.cell_nodes[c], e.scalar_mass.sum(axis=1), None)
                        )
            elif type(base) in (loads.BoundaryLoad, loads.PressureLoad):
                self.kind = (
                    "pressure" if type(base) is loads.PressureLoad else "traction"
                )
                _constant(
                    base.pressure if self.kind == "pressure" else base.value,
                    1 if self.kind == "pressure" else 3,
                )
                if self.kind == "pressure" and base.configuration != "reference":
                    raise NotImplementedError(
                        "Composite pressure requires reference configuration."
                    )
                region = base.location
                if region is None or region.domain is not self.domain:
                    raise ValueError(
                        "Composite traction needs a named boundary on this mesh."
                    )
                if self.kind == "pressure":
                    import ufl

                    if any(
                        ufl.algorithms.expand_indices(base.traction[i])
                        != ufl.algorithms.expand_indices(
                            (-base.pressure * ufl.FacetNormal(self.domain))[i]
                        )
                        for i in range(3)
                    ):
                        raise NotImplementedError(
                            "Composite pressure requires geometric outward normals; custom normals are not supported."
                        )
                if region.facet_tags is None:
                    from dolfinx import mesh

                    facets = mesh.locate_entities_boundary(
                        self.domain, 2, region.marker
                    )
                else:
                    facets = region.facet_tags.find(region.tag)
                selected = set(map(int, facets))
                self.domain.topology.create_connectivity(3, 2)
                faces = basix.topology(basix.CellType.tetrahedron)[2]
                xyz = V.tabulate_dof_coordinates()
                for cell, ids in enumerate(internal.cell_nodes):
                    cf = self.domain.topology.connectivity(3, 2).links(cell)
                    for local, facet in enumerate(cf):
                        if int(facet) not in selected:
                            continue
                        face = faces[local]
                        x = xyz[ids[face]]
                        normal = np.cross(x[1] - x[0], x[2] - x[0])
                        area = np.linalg.norm(normal) / 2
                        normal /= 2 * area
                        if np.dot(normal, x.mean(0) - xyz[ids[:4]].mean(0)) < 0:
                            normal *= -1
                        edges = [
                            4 + i
                            for i, pair in enumerate(_EDGES)
                            if set(pair) <= set(face)
                        ]
                        local_nodes = np.array([*face, *edges])
                        weights = area * np.r_[np.full(3, 1 / 12), np.full(3, 1 / 4)]
                        self.weights.append((ids[local_nodes], weights, normal))
            else:
                raise NotImplementedError(
                    f"Unsupported composite load {type(base).__name__}; use nodal, constant body/traction/reference pressure loads."
                )

        collective_call(prepare, comm=internal.comm, label="Composite load preparation")

    def assemble_vector(self):
        from petsc4py import PETSc

        internal = self.internal
        vector = internal.displacement.x.petsc_vec.duplicate()

        def local():
            scale = (
                float(_constant(self.asset.scale, 1)[0])
                if isinstance(self.asset, loads.AmplitudeLoad)
                else 1.0
            )
            value = _constant(
                self.base.pressure if self.kind == "pressure" else self.base.value,
                1 if self.kind == "pressure" else 3,
            )
            with vector.localForm() as form:
                form.set(0)
                for ids, weight, normal in self.weights:
                    force = -value[0] * normal if self.kind == "pressure" else value
                    np.add.at(
                        form.array.reshape(-1, 3), ids, scale * weight[:, None] * force
                    )

        try:
            collective_call(local, comm=internal.comm, label="Composite load assembly")
            vector.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
            return vector
        except Exception:
            vector.destroy()
            raise
