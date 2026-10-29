# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""MPI-safe output helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dolfinx import fem
from dolfinx import io
from mpi4py import MPI

from . import fields as field_api


def ensure_output_dir(path: Path, comm: MPI.Comm) -> None:
    """Create an output directory once, then synchronize all ranks."""

    if comm.rank == 0:
        path.mkdir(parents=True, exist_ok=True)
    comm.barrier()


@dataclass
class CSVLogger:
    """Rank-zero CSV writer for time histories and scalar diagnostics."""

    path: Path
    header: tuple[str, ...]
    comm: MPI.Comm

    def initialize(self) -> None:
        """Create the parent directory and write the CSV header."""

        ensure_output_dir(self.path.parent, self.comm)
        if self.comm.rank == 0:
            self.path.write_text(",".join(self.header) + "\n", encoding="utf-8")
        self.comm.barrier()

    def append(self, *values) -> None:
        """Append one row on rank zero."""

        if self.comm.rank == 0:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(",".join(_format_csv_value(value) for value in values) + "\n")


class XDMFTimeSeries:
    """Small context manager for writing a mesh and time-dependent fields."""

    def __init__(self, path: Path, domain, mode: str = "w") -> None:
        self.path = path
        self.domain = domain
        self.mode = mode
        self._file = None

    def __enter__(self):
        ensure_output_dir(self.path.parent, self.domain.comm)
        self._file = io.XDMFFile(self.domain.comm, str(self.path), self.mode)
        self._file.__enter__()
        self._file.write_mesh(self.domain)
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._file.__exit__(exc_type, exc, tb)

    def write_fields(self, time: float, *fields) -> None:
        """Write one or more finite-element functions at a time value."""

        for field in fields:
            self._file.write_function(field_api.unwrap(field), time)


class ParaViewTimeSeries:
    """Collective VTK/PVD series with one geometry carrying all fields.

    DOLFINx writes one parallel unstructured-grid dataset per saved time. Point
    fields and DG0 cell fields remain distinct attributes on that dataset, so
    ParaView's Warp filter no longer receives duplicate geometry blocks.
    This is a presentation product; checkpoint and integration-point state
    retain their separate scientific storage contracts.
    """

    def __init__(self, path: Path, domain, mode: str = "w") -> None:
        self.path = Path(path)
        self.domain = domain
        self.mode = mode
        self._file = None

    def __enter__(self):
        ensure_output_dir(self.path.parent, self.domain.comm)
        self._file = io.VTKFile(self.domain.comm, str(self.path), self.mode)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()
        return False

    def write_fields(self, time: float, *fields) -> None:
        selected = [field_api.unwrap(field) for field in fields]
        if not selected:
            raise ValueError("ParaViewTimeSeries requires at least one field.")
        self._file.write_function(selected, float(time))


class ResultWriter:
    """Named visualization writer for one mesh and a stable field list.

    New serial XDMF results carry all fields on a single reference geometry;
    use Warp By Vector once. Choose ``.pvd`` for collective MPI visualization.
    Legacy append mode retains the low-level DOLFINx archive layout.
    """

    def __init__(self, path: Path, domain, fields=(), mode: str = "w", *, storage="single_file") -> None:
        self.path = Path(path)
        if self.path.suffix.lower() not in {".xdmf", ".xmf", ".pvd"}:
            raise ValueError("ResultWriter requires .xdmf/.xmf or .pvd, not a hand-written raw topology format.")
        if mode not in {"w", "a"}:
            raise ValueError("ResultWriter mode must be 'w' or 'a'.")
        self.domain = domain
        self.fields = tuple(fields)
        self.mode = mode
        if storage not in {"single_file", "frame_files"}:
            raise ValueError("ResultWriter storage must be 'single_file' or 'frame_files'.")
        if storage == "frame_files" and (mode != "w" or domain.comm.size != 1
                                         or self.path.suffix.lower() == ".pvd"):
            raise ValueError("Frame-file output requires a fresh serial XDMF series.")
        self._unified = False
        if self.path.suffix.lower() == ".pvd":
            self._series = ParaViewTimeSeries(self.path, self.domain, self.mode)
        elif mode == "w" and domain.comm.size == 1:
            from .results.output import UnifiedXDMFTimeSeries
            if storage == "frame_files":
                from .results._live_xdmf import LiveXDMFTimeSeries
                self._series = LiveXDMFTimeSeries(self.path, deformation_scale=0.0)
            else:
                self._series = UnifiedXDMFTimeSeries(self.path, deformation_scale=0.0)
            self._unified = True
        else:
            self._series = XDMFTimeSeries(self.path, self.domain, self.mode)

    def __enter__(self):
        self._series.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._series.__exit__(exc_type, exc, tb)

    def write(self, time: float, *fields) -> None:
        """Write either explicit fields or the writer's default field list."""

        selected = fields if fields else self.fields
        if not selected:
            raise ValueError("ResultWriter.write requires fields or default fields.")
        if any(field_api.unwrap(field).function_space.mesh is not self.domain
               for field in selected):
            raise ValueError("ResultWriter fields must belong to its declared mesh.")
        if self._unified and self._series.primary_name is None:
            self._series.primary_name = field_api.unwrap(selected[0]).name
        self._series.write_fields(time, *(field_api.unwrap(field) for field in selected))

    def artifacts(self) -> dict[str, Path]:
        """Inventory an emitted serial unified XDMF view and its heavy data.

        This is byte-delivery metadata, not scientific verification. During a
        run the published partial/closed-frame index is returned; only files
        actually referenced by that index are included, never a directory glob.
        PVD and legacy distributed archives retain their existing publication
        routes and are not silently given an incomplete artifact inventory.
        """
        import xml.etree.ElementTree as ET

        if not self._unified:
            raise NotImplementedError("ResultWriter.artifacts currently inventories unified serial XDMF only.")
        if not self._series._frame_count:
            return {}
        # A previous final index may still exist while a replacement writer
        # owns a new partial index. Never inventory that stale final view.
        index = self._series.partial_path if self._series._partial_owned else self.path
        if not index.is_file():
            return {}
        files = set()
        for item in ET.parse(index).iter("DataItem"):
            if item.get("Format") == "HDF":
                filename, separator, dataset = (item.text or "").strip().partition(":")
                if not separator or not dataset.startswith("/"):
                    raise ValueError("Malformed emitted HDF reference.")
                path = (index.parent / filename).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                files.add(path)
        return {"field_output": index, **{f"field_data_{i}": path for i, path in enumerate(sorted(files))}}


def interpolate_for_xdmf(field, *, degree: int = 1, name: str | None = None):
    """Interpolate a field to an XDMF-friendly Lagrange output space.

    DOLFINx XDMF output requires the function degree to match the mesh geometry
    degree. For higher-order analysis fields on a linear mesh, write an
    interpolated visualization copy while keeping the solve itself high-order.
    """

    field = field_api.unwrap(field)
    domain = field.function_space.mesh
    shape = getattr(field, "ufl_shape", ())
    element = ("Lagrange", degree) if len(shape) == 0 else ("Lagrange", degree, shape)
    V_out = fem.functionspace(domain, element)
    output = fem.Function(V_out, name=name or getattr(field, "name", "Output"))
    output.interpolate(field)
    return output


def _format_csv_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.16e}"
    return str(value)
