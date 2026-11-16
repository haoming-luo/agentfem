# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Compact accepted U/V/A observations and explicit CSV/JSON delivery."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
from mpi4py import MPI


@dataclass(frozen=True)
class DynamicHistoryRequest:
    name: str
    variable: str
    component: object
    nodes: object = None
    at: object = None
    unit: str | None = None
    time_unit: str | None = None
    _sampling: dict = field(default_factory=dict, compare=False, repr=False)
    _bindings: dict = field(default_factory=dict, compare=False, repr=False)

    def summary(self):
        return {
            "kind": "dynamic_history",
            "name": self.name,
            "variable": self.variable,
            "component": self.component,
            "unit": self.unit,
            "time_unit": self.time_unit,
            "selection": self.nodes.summary()
            if self.nodes is not None
            else {
                "coordinates": list(self.at),
                "method": "finite_element_interpolation_not_nearest_node",
            },
        }

    def evaluate_transient(self, step, time_value):
        from .. import fields

        value = fields.unwrap(
            getattr(step.state, {"U": "u", "V": "v", "A": "a"}[self.variable])
        )
        if self.nodes is None:
            key = ("spatial", id(value.function_space))
            if key not in self._bindings:
                self._bindings[key] = _bind_spatial_point(value.function_space, self.at)
            dofs, weights, evidence = self._bindings[key]
            self._sampling.update(evidence)
            bs = value.function_space.dofmap.index_map_bs
            local = (
                np.zeros(bs)
                if dofs is None
                else weights @ value.x.array.reshape(-1, bs)[dofs]
            )
            vector = value.function_space.mesh.comm.allreduce(local, op=MPI.SUM)
        else:
            key = id(value.function_space)
            if key not in self._bindings:
                self._bindings[key] = self.nodes.bind(value)
            V, indices = self._bindings[key]
            if len(indices) != 1:
                raise ValueError("AFM-HISTORY-NODE: select one node per named station.")
            n = indices[0]
            bs = V.dofmap.index_map_bs
            local = (
                np.zeros(bs) if n < 0 else value.x.array[n * bs : (n + 1) * bs].copy()
            )
            vector = V.mesh.comm.allreduce(local, op=MPI.SUM)
            self._sampling.update(self.nodes.summary())
        if self.component in ("magnitude", "cumulative_max"):
            scalar = float(np.linalg.norm(vector))
            if self.component == "cumulative_max" and step.history_records:
                scalar = max(
                    scalar, float(step.history_records[-1].get(self.name, scalar))
                )
        else:
            scalar = float(vector[int(self.component)])
        if not np.isfinite(scalar):
            raise ValueError("AFM-HISTORY-NONFINITE: invalid accepted response.")
        return scalar


def dynamic_history(
    name,
    *,
    nodes=None,
    at=None,
    variables=("U", "V", "A"),
    dimension=3,
    length_unit=None,
    time_unit=None,
):
    """Return scalar requests consumed by Step.solve_result(history=...).

    Records components, magnitude and the cumulative maximum at recorded
    accepted stations. No temporal interpolation or nearest-node substitution.
    Supply units explicitly; unspecified units remain unspecified.
    """
    if (nodes is None) == (at is None):
        raise ValueError("AFM-HISTORY: choose nodes or a spatial point.")
    if dimension not in (2, 3):
        raise ValueError("AFM-HISTORY: dimension must be 2 or 3.")
    if (
        not name
        or not variables
        or len(set(variables)) != len(variables)
        or any(v not in ("U", "V", "A") for v in variables)
    ):
        raise ValueError("AFM-HISTORY: name and distinct U/V/A variables required.")
    if at is not None:
        at = tuple(float(x) for x in at)
        if len(at) != dimension or not np.all(np.isfinite(at)):
            raise ValueError("AFM-HISTORY: invalid point.")
    if nodes is not None:
        from .._nodal import NodalSelection

        if not isinstance(nodes, NodalSelection):
            raise TypeError(
                "AFM-HISTORY: nodes must be a mesh.nodal_selection or imported nodeset."
            )
        identities = nodes.labels or nodes.points
        if len(identities) > 1:
            expanded = []
            for index, identity in enumerate(identities):
                selected = (
                    NodalSelection(
                        labels=(identity,),
                        source=nodes.source,
                        tolerance=nodes.tolerance,
                    )
                    if nodes.labels
                    else NodalSelection(points=(identity,), tolerance=nodes.tolerance)
                )
                suffix = f"N{identity}" if nodes.labels else f"P{index}"
                expanded.extend(
                    dynamic_history(
                        f"{name}[{suffix}]",
                        nodes=selected,
                        variables=variables,
                        dimension=dimension,
                        length_unit=length_unit,
                        time_unit=time_unit,
                    )
                )
            return tuple(expanded)
    units = {"U": length_unit, "V": None, "A": None}
    if length_unit and time_unit:
        units.update(V=f"{length_unit}/{time_unit}", A=f"{length_unit}/{time_unit}^2")
    result = []
    for v in variables:
        for c in (*range(dimension), "magnitude", "cumulative_max"):
            suffix = (
                str(c + 1)
                if isinstance(c, int)
                else ("MAG" if c == "magnitude" else "MAX")
            )
            result.append(
                DynamicHistoryRequest(
                    f"{name}.{v}{suffix}", v, c, nodes, at, units[v], time_unit
                )
            )
    return tuple(result)


def write_dynamic_history(result, path, *, metadata=None, names=None):
    """Write a CSV and JSON sidecar from identically sampled result histories."""
    if names is None:
        names = [
            x["name"]
            for x in result.metadata.get("transient", {}).get("history_requests", [])
        ]
        names += [
            x
            for x in (
                "kinetic_energy",
                "strain_energy",
                "external_work",
                "artificial_energy",
                "initial_mechanical_energy",
                "total_mechanical_energy",
                "external_force_1",
                "external_force_2",
                "external_force_3",
                "energy_balance_error",
            )
            if x in result.histories
        ]
    histories = tuple(result.histories[name] for name in names)
    if not histories:
        raise ValueError("AFM-HISTORY-EXPORT: result has no histories.")
    times = np.asarray(histories[0].abscissa)
    for h in histories:
        if not np.array_equal(times, h.abscissa) or np.asarray(h.values).ndim != 1:
            raise ValueError(
                "AFM-HISTORY-EXPORT: scalar histories with identical time axes required."
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack([times, *[h.values for h in histories]]),
        delimiter=",",
        header=",".join(["time", *[h.name for h in histories]]),
        comments="",
    )
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema": "agentfem.dynamic-history",
                "time_unit": histories[0].abscissa_unit,
                "columns": [{"name": h.name, "unit": h.unit} for h in histories],
                "samples": len(times),
                "processing": "raw accepted samples; no smoothing",
                "sampling": result.metadata.get("transient", {}).get(
                    "sampling_evidence", {}
                ),
                "requests": result.metadata.get("transient", {}).get(
                    "history_requests", []
                ),
                "metadata": metadata or {},
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return path, sidecar


def _bind_spatial_point(V, at):
    """Freeze a deterministic scalar-Lagrange interpolation stencil."""
    from dolfinx import geometry
    from ..provenance import collective_call, collective_canonical_record

    comm = V.mesh.comm
    point = np.pad(np.asarray(at, dtype=float), (0, 3 - len(at)))[None, :]
    collective_canonical_record(point.tolist(), comm=comm, label="History station")

    def locate():
        if (
            V.element.basix_element.value_size != 1
            or V.dofmap.bs != V.dofmap.index_map_bs
        ):
            raise NotImplementedError(
                "AFM-HISTORY-SPACE: blocked nodal Lagrange interpolation required."
            )
        n = V.mesh.topology.index_map(V.mesh.topology.dim).size_local
        if not n:
            return -1
        tree = geometry.bb_tree(
            V.mesh,
            V.mesh.topology.dim,
            entities=np.arange(n, dtype=np.int32),
            padding=1e-10,
        )
        candidates = geometry.compute_collisions_points(tree, point)
        cells = geometry.compute_colliding_cells(V.mesh, candidates, point).links(0)
        return int(min(cells)) if len(cells) else -1

    cell = collective_call(locate, comm=comm, label="History point location")
    owner = comm.allreduce(comm.rank if cell >= 0 else comm.size, op=MPI.MIN)
    if owner == comm.size:
        raise ValueError("AFM-HISTORY-POINT: station lies outside the mesh.")

    def stencil():
        if comm.rank != owner:
            return None, None, None
        geometry_nodes = V.mesh.geometry.dofmaps[0][cell]
        reference = V.mesh.geometry.cmaps[0].pull_back(
            point, V.mesh.geometry.x[geometry_nodes]
        )
        weights = V.element.basix_element.tabulate(0, reference)[0, 0, :, 0].copy()
        dofs = V.dofmap.cell_dofs(cell).copy()
        evidence = {
            "method": "finite_element_interpolation",
            "coordinates": point[0].tolist(),
            "owner_rank": int(owner),
            "input_cell": int(V.mesh.topology.original_cell_index[cell]),
            "global_dof_blocks": V.dofmap.index_map.local_to_global(dofs).tolist(),
            "reference_coordinates": reference[0].tolist(),
            "weights": weights.tolist(),
            "tie_break": "lowest rank, lowest owned cell; use node identity at a seam",
        }
        return dofs, weights, evidence

    dofs, weights, evidence = collective_call(
        stencil, comm=comm, label="History interpolation"
    )
    evidence = comm.bcast(evidence, root=owner)
    return dofs, weights, evidence
