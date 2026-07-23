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


class ResultWriter:
    """Named result writer for one mesh and a stable field list."""

    def __init__(self, path: Path, domain, fields=(), mode: str = "w") -> None:
        self.path = Path(path)
        self.domain = domain
        self.fields = tuple(fields)
        self.mode = mode
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
        self._series.write_fields(time, *selected)


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
