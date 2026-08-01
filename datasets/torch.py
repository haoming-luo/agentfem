"""Thin, optional bridges from scientific FEM data to PyTorch tensors.

AgentFEM keeps units, field roles, case provenance, and mesh policy. PyTorch
keeps tensor execution, autodiff, optimization, and model architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """One serial nodal FEM field with coordinates and scientific encoding."""

    coordinates: np.ndarray
    values: np.ndarray
    encoding: dict[str, object]
    metadata: dict[str, object]

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            coordinates=self.coordinates,
            values=self.values,
            encoding_json=json.dumps(self.encoding, sort_keys=True),
            metadata_json=json.dumps(self.metadata, sort_keys=True),
        )
        return output

    def torch(self, *, dtype: str = "float32", device: str = "cpu"):
        torch = _torch()
        selected_dtype = getattr(torch, dtype)
        return {
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


__all__ = ["FEMFieldSample", "TorchDatasetBundle", "fem_field_sample", "to_torch"]
