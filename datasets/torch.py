"""Thin, optional bridges from scientific FEM data to PyTorch tensors.

AgentFEM keeps units, field roles, case provenance, and mesh policy. PyTorch
keeps tensor execution, autodiff, optimization, and model architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from .core import ScientificDataset


@dataclass(frozen=True)
class TorchDatasetBundle:
    """PyTorch dataset plus the schema needed to interpret its columns."""

    dataset: object
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    normalized_inputs: bool
    scientific_summary: dict[str, object]

    def loader(self, *, batch_size: int = 64, shuffle: bool = True, seed: int = 0):
        torch = _torch()
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        generator = torch.Generator().manual_seed(int(seed))
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=min(int(batch_size), len(self.dataset)),
            shuffle=bool(shuffle),
            generator=generator,
        )


@dataclass(frozen=True)
class FEMFieldSample:
    """One FEM field representation with coordinates and scientific encoding."""

    coordinates: np.ndarray
    values: np.ndarray
    encoding: dict[str, object]
    metadata: dict[str, object]
    mask: np.ndarray | None = None
    comm: object | None = field(default=None, repr=False, compare=False)

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        comm = self.comm
        is_writer = comm is None or comm.rank == 0
        if is_writer:
            output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "coordinates": self.coordinates,
            "values": self.values,
            "encoding_json": json.dumps(self.encoding, sort_keys=True),
            "metadata_json": json.dumps(self.metadata, sort_keys=True),
        }
        if self.mask is not None:
            arrays["mask"] = np.asarray(self.mask, dtype=bool)
        if is_writer:
            np.savez_compressed(output, **arrays)
        if comm is not None:
            comm.barrier()
        return output

    def torch(self, *, dtype: str = "float32", device: str = "cpu"):
        torch = _torch()
        selected_dtype = getattr(torch, dtype)
        tensors = {
            "coordinates": torch.as_tensor(
                self.coordinates,
                dtype=selected_dtype,
                device=device,
            ),
            "values": torch.as_tensor(
                self.values,
                dtype=selected_dtype,
                device=device,
            ),
        }
        if self.mask is not None:
            tensors["mask"] = torch.as_tensor(
                self.mask,
                dtype=torch.bool,
                device=device,
            )
        return tensors


def to_torch(
    dataset: ScientificDataset,
    *,
    normalized_inputs: bool = True,
    dtype: str = "float32",
    device: str = "cpu",
) -> TorchDatasetBundle:
    """Expose a validated campaign dataset as a PyTorch ``TensorDataset``."""

    torch = _torch()
    if not hasattr(torch, dtype):
        raise ValueError(f"Unknown torch dtype {dtype!r}.")
    selected_dtype = getattr(torch, dtype)
    features = torch.as_tensor(
        dataset.x_matrix(normalized=normalized_inputs),
        dtype=selected_dtype,
        device=device,
    )
    targets = torch.as_tensor(
        dataset.y_matrix(),
        dtype=selected_dtype,
        device=device,
    )
    return TorchDatasetBundle(
        dataset=torch.utils.data.TensorDataset(features, targets),
        input_names=dataset.parameter_space.feature_names,
        output_names=dataset.output_names,
        normalized_inputs=bool(normalized_inputs),
        scientific_summary=dataset.summary(),
    )


def fem_field_sample(function, encoding) -> FEMFieldSample:
    """Export owned nodal coefficients for external neural/PINN tooling.

    The first bridge is intentionally serial and ``mesh_dofs`` only. A
    distributed field needs global dof identities and a graph/mesh partition
    manifest before concatenation can be scientifically safe.
    """

    selected = getattr(function, "value", function)
    space = selected.function_space
    comm = space.mesh.comm
    summary = encoding.summary() if hasattr(encoding, "summary") else dict(encoding)
    if comm.size != 1:
        raise NotImplementedError(
            "Distributed field export requires global dof identities and a "
            "partition manifest; use a serial export or an external distributed adapter."
        )
    if summary.get("representation") != "mesh_dofs":
        raise ValueError(
            "fem_field_sample currently requires FieldEncoding(representation='mesh_dofs')."
        )
    coordinates = np.asarray(space.tabulate_dof_coordinates(), dtype=float)
    block_size = int(space.dofmap.index_map_bs)
    owned = int(space.dofmap.index_map.size_local)
    values = np.asarray(selected.x.array[: owned * block_size], dtype=float)
    values = values.reshape((owned, block_size))
    if len(coordinates) != owned:
        coordinates = coordinates[:owned]
    return FEMFieldSample(
        coordinates=coordinates,
        values=values,
        encoding=summary,
        metadata={
            "field_name": selected.name,
            "dofs": owned,
            "components": block_size,
            "mesh_policy": summary.get("mesh_policy"),
            "source": "dolfinx_owned_coefficients",
        },
        comm=comm,
    )


def fem_observation_sample(
    function,
    grid,
    *,
    name: str | None = None,
    unit: str | None = None,
    role: str = "output",
    components=(),
    outside: str = "raise",
    fill_value: float = 0.0,
) -> FEMFieldSample:
    """Sample a FEM field on a reusable structured observation grid.

    The same function works in serial and MPI because AgentFEM's point sampler
    resolves distributed ownership collectively. ``outside='mask'`` is useful
    when a Cartesian tensor grid covers voids or geometry outside the mesh;
    masked values are replaced explicitly and the mask is exported alongside
    the field for a neural-operator condition channel.
    """

    from .. import results
    from ..surrogates import FieldEncoding

    selected = getattr(function, "value", function)
    selected_outside = str(outside).lower().replace("-", "_")
    if selected_outside not in {"raise", "mask"}:
        raise ValueError("fem_observation_sample outside must be 'raise' or 'mask'.")
    coordinates = np.asarray(grid.points(), dtype=float)
    values = np.asarray(
        results.sample_points(
            selected,
            coordinates,
            missing="raise" if selected_outside == "raise" else "nan",
        )
    )
    value_shape = tuple(getattr(selected, "ufl_shape", ()))
    flat_values = values.reshape((grid.point_count, -1))
    mask = np.all(np.isfinite(flat_values), axis=1).reshape(grid.shape, order=grid.order)
    if selected_outside == "mask":
        values = np.where(np.isfinite(values), values, float(fill_value))
    shaped_values = values.reshape((*grid.shape, *value_shape), order=grid.order)
    field_name = str(name or getattr(selected, "name", "field"))
    encoding = FieldEncoding(
        name=field_name,
        role=role,
        unit=unit,
        components=tuple(components),
        representation="structured_grid",
        shape=shaped_values.shape,
        geometry_encoding="explicit_cartesian_axes",
        mesh_policy="mesh_independent_coordinates",
        metadata={
            "observation_grid": grid.summary(),
            "layout": "grid_axes_then_value_components",
            "outside": selected_outside,
            "fill_value": float(fill_value) if selected_outside == "mask" else None,
            "mask_semantics": "true_inside_mesh",
        },
    )
    return FEMFieldSample(
        coordinates=coordinates,
        values=shaped_values,
        encoding=encoding.summary(),
        metadata={
            "field_name": field_name,
            "source": "agentfem_mpi_point_sampling",
            "grid_shape": grid.shape,
            "value_shape": value_shape,
            "point_count": grid.point_count,
            "inside_count": int(np.count_nonzero(mask)),
        },
        mask=mask if selected_outside == "mask" else None,
        comm=selected.function_space.mesh.comm,
    )


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch export requires optional dependency 'torch'. Install "
            "AgentFEM with the 'ml' extra after installing the FEniCSx stack."
        ) from exc
    return torch


__all__ = [
    "FEMFieldSample",
    "TorchDatasetBundle",
    "fem_field_sample",
    "fem_observation_sample",
    "to_torch",
]
