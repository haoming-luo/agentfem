# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Accepted-frame output coordination; numerical state remains solver-owned."""

from pathlib import Path
import hashlib
import warnings
import xml.etree.ElementTree as ET

import h5py
import numpy as np


class OutputError(ValueError):
    """Addressable output failure for CLI clients and agents."""

    stage = "result_output"

    def __init__(self, message, *, code="AFM-OUTPUT-INCOMPLETE"):
        super().__init__(message)
        self.code = code


def inspect_output(path):
    """Check a local XDMF/PVD dependency graph without claiming physical validity."""
    root_path = Path(path).resolve()
    errors, dependencies, visited = [], [], set()
    coordinates = []

    def visit(p):
        p = p.resolve()
        if p in visited:
            return
        visited.add(p)
        if not p.is_file():
            errors.append(f"Missing output dependency: {p}")
            return
        dependencies.append(str(p))
        if p.suffix.lower() not in {".xdmf", ".xmf", ".pvd", ".pvtu"}:
            return
        try:
            tree = ET.parse(p)
            if p == root_path:
                nodes = (
                    tree.findall(".//Time")
                    if p.suffix.lower() in {".xdmf", ".xmf"}
                    else tree.findall(".//DataSet")
                )
                coordinates.extend(
                    float(n.get("Value", n.get("timestep", "nan"))) for n in nodes
                )
                if not nodes:
                    errors.append(f"No output frames: {p}")
                if any(not np.isfinite(t) for t in coordinates):
                    errors.append(f"Invalid frame coordinate: {p}")
            if p.suffix.lower() in {".xdmf", ".xmf"}:
                for item in tree.findall(".//DataItem"):
                    if item.get("Format", "").upper() != "HDF":
                        continue
                    filename, dataset = (item.text or "").strip().rsplit(":", 1)
                    heavy = (p.parent / filename).resolve()
                    visit(heavy)
                    if heavy.is_file():
                        with h5py.File(heavy, "r") as h5:
                            data = h5[dataset]
                            expected = tuple(
                                int(v) for v in item.get("Dimensions", "").split()
                            )
                            if expected and data.shape != expected:
                                errors.append(
                                    f"Shape mismatch: {heavy}:{dataset}: {data.shape} != {expected}"
                                )
            else:
                for item in tree.iter():
                    ref = item.get("file") or item.get("Source")
                    if ref:
                        visit(p.parent / ref)
        except (OSError, ValueError, KeyError, ET.ParseError) as exc:
            errors.append(f"Invalid output {p}: {exc}")

    visit(root_path)
    return {
        "valid": not errors,
        "errors": errors,
        "dependencies": dependencies,
        "coordinates": sorted(set(coordinates)),
    }


class OutputSeries:
    """Stream accepted frames to existing backends and register their artifacts.

    Call collectively under MPI. A custom solver must call write_accepted only
    after committing an increment. This class does not decide convergence.
    Fixed topology, geometry and field spaces are required within one segment.
    """

    def __init__(
        self,
        result,
        path,
        *,
        paraview=False,
        coordinate_name="time",
        primary_name=None,
        primary_semantic_name=None,
        strict=True,
        overwrite=False,
        deformation_scale=0.0,
        compression=4,
        store_reference_geometry=True,
    ):
        self.result, self.path = result, Path(path)
        if self.path.suffix.lower() != ".xdmf":
            raise ValueError("Output series path must end in .xdmf.")
        if not str(coordinate_name).strip():
            raise ValueError("coordinate_name must be nonempty.")
        self.paraview, self.coordinate_name = bool(paraview), str(coordinate_name)
        self.primary_name, self.semantic = primary_name, primary_semantic_name
        self.strict, self.overwrite = bool(strict), bool(overwrite)
        self.scale, self.compression = float(deformation_scale), compression
        self.store_reference_geometry = store_reference_geometry
        self.writer = self.viewer = self.comm = None
        self.last = None
        self.display_error = None
        self.count = 0
        self.signature = None
        self.closed = False
        self.marker = self.path.with_suffix(".incomplete")
        self.record = dict(
            status="pending",
            requested_path=str(self.path),
            coordinate_name=self.coordinate_name,
            frames=0,
            strict=self.strict,
            segment=str(self.path),
        )

    def __enter__(self):
        self.result.metadata.setdefault("output_series", []).append(self.record)
        return self

    def _collective_check(self, operation):
        if self.comm.size == 1:
            return operation()
        error = None
        try:
            value = operation()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            value = None
        errors = self.comm.allgather(error)
        if any(errors):
            raise ValueError(
                "AFM-OUTPUT-CONTRACT: " + "; ".join(e for e in errors if e)
            )
        return value

    def write_fields(self, coordinate, *fields):
        """Internal adapter for existing accepted-state Procedure callbacks."""
        if not fields:
            raise ValueError("An accepted frame needs a primary field.")
        if self.writer is None:
            vector = len(fields[0].ufl_shape) == 1
            self.primary_name = self.primary_name or ("U" if vector else fields[0].name)
            self.semantic = self.semantic or (
                "Displacement" if vector else fields[0].name
            )
        return self.write_accepted(
            coordinate=coordinate, primary=fields[0], fields=fields[1:]
        )

    def write_accepted(self, *, coordinate, primary, fields=()):
        if self.closed:
            raise RuntimeError("Output series is closed.")
        primary = getattr(primary, "value", primary)
        auxiliary = tuple(getattr(f, "value", f) for f in fields)
        domain = primary.function_space.mesh
        if self.comm is None:
            self.comm = domain.comm

        def preflight():
            time = float(coordinate)
            if not np.isfinite(time) or (self.last is not None and time <= self.last):
                raise ValueError(
                    "Coordinates must be finite and strictly increasing; restart in a new segment."
                )
            items = (primary, *auxiliary)
            if any(f.function_space.mesh is not domain for f in items):
                raise ValueError(
                    "Fields must share one mesh; no implicit cross-mesh projection."
                )
            names = [self.primary_name or primary.name, *(f.name for f in auxiliary)]
            magnitude = "UMAG" if names[0] == "U" else names[0] + "_MAG"
            if magnitude in names[1:]:
                raise ValueError(
                    "Auxiliary name collides with generated vector magnitude."
                )
            if len(set(names)) != len(names):
                raise ValueError("Output field names must be unique.")
            shape = tuple(primary.ufl_shape)
            semantic = self.semantic or primary.name
            if self.scale and semantic not in {"Displacement", "Mode shape"}:
                raise ValueError(
                    "Deformed geometry requires explicit displacement or mode-shape semantics."
                )
            signature = (
                id(domain),
                hashlib.sha256(domain.geometry.x.tobytes()).hexdigest(),
                tuple(
                    (
                        f.name,
                        str(f.function_space.ufl_element()),
                        tuple(f.ufl_shape),
                        f.function_space.dofmap.index_map.size_local,
                    )
                    for f in items
                ),
            )
            if self.signature is not None and signature != self.signature:
                raise ValueError(
                    "Mesh geometry or field contract changed; start a new segment."
                )
            if any(not np.all(np.isfinite(f.x.array)) for f in items):
                raise ValueError(
                    "Nonfinite field values cannot enter an accepted output frame."
                )
            return time, signature, semantic, shape

        time, signature, semantic, shape = self._collective_check(preflight)
        contract = (
            str(self.path.resolve()),
            self.paraview,
            self.scale,
            self.primary_name,
            self.semantic,
            self.coordinate_name,
            self.strict,
            tuple((entry[0], entry[1], entry[2]) for entry in signature[2]),
        )
        if any(item != contract for item in self.comm.allgather(contract)):
            raise ValueError(
                "AFM-OUTPUT-CONTRACT: ranks supplied different output schemas."
            )
        coordinates = self.comm.allgather(time)
        if any(t != time for t in coordinates):
            raise ValueError(
                "AFM-OUTPUT-COORDINATE: ranks supplied different coordinates."
            )
        if self.writer is None:
            from .output import UnifiedXDMFTimeSeries
            from .. import io

            def prepare():
                if self.comm.rank == 0:
                    if not self.overwrite and any(
                        p.exists()
                        for p in (self.path, self.path.with_suffix(".h5"), self.marker)
                    ):
                        raise FileExistsError(
                            "Output already exists; use a new segment/path or explicit overwrite."
                        )
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self.marker.write_text("Output is not finalized.\n")

            self._collective_check(prepare)
            self.signature = signature
            self.record.update(
                status="writing",
                primary_semantic_name=semantic,
                geometry="deformed" if self.scale else "reference",
                warp_field=(self.primary_name or primary.name)
                if semantic in {"Displacement", "Mode shape"} and not self.scale
                else None,
                layout="single_uniform_grid"
                if self.comm.size == 1
                else "collective_grid_per_field",
                start=time,
                field_names=[
                    self.primary_name or primary.name,
                    *(f.name for f in auxiliary),
                ],
            )
            if self.comm.size == 1:
                self.writer = UnifiedXDMFTimeSeries(
                    self.path,
                    deformation_scale=self.scale,
                    primary_name=self.primary_name or primary.name,
                    primary_semantic_name=semantic,
                    compression=self.compression,
                    store_reference_geometry=self.store_reference_geometry,
                )
            else:
                if self.scale:
                    raise NotImplementedError(
                        "MPI series preserves reference geometry; use Warp for presentation."
                    )
                self.writer = io.XDMFTimeSeries(self.path, domain)
            self.writer.__enter__()
            if self.paraview:
                self.viewer = io.ParaViewTimeSeries(
                    self.path.with_suffix(".pvd"), domain
                )
                try:
                    self._collective_check(self.viewer.__enter__)
                except Exception as exc:
                    self.display_error = str(exc)
                    self.viewer = None
        if self.comm.size == 1:
            output_fields = (primary, *auxiliary)
        else:
            from .. import io

            maps = getattr(domain.geometry, "cmaps", ())
            degree = int(getattr(maps[0], "degree", 1)) if maps else 1
            from .output import _is_discontinuous

            output_fields = []
            conversions = []
            for index, function in enumerate((primary, *auxiliary)):
                if _is_discontinuous(function):
                    if function.function_space.element.basix_element.degree != 0:
                        raise NotImplementedError(
                            "MPI output requires explicit DG0 recovery for discontinuous auxiliary fields."
                        )
                    converted = function
                else:
                    converted = io.interpolate_for_xdmf(
                        function,
                        degree=degree,
                        name=(self.primary_name or primary.name)
                        if index == 0
                        else function.name,
                    )
                    conversions.append(
                        dict(
                            name=converted.name,
                            method="interpolate_to_geometry_degree",
                            degree=degree,
                        )
                    )
                output_fields.append(converted)
            output_fields = tuple(output_fields)
            self.record["field_conversions"] = conversions
        for f in output_fields:
            f.x.scatter_forward()
        self._collective_check(lambda: self.writer.write_fields(time, *output_fields))
        if self.viewer is not None and self.display_error is None:
            display_fields = output_fields
            if self.primary_name and output_fields[0].name != self.primary_name:
                from dolfinx import fem

                renamed = fem.Function(
                    output_fields[0].function_space, name=self.primary_name
                )
                renamed.x.array[:] = output_fields[0].x.array
                display_fields = (renamed, *output_fields[1:])
            try:
                self._collective_check(
                    lambda: self.viewer.write_fields(time, *display_fields)
                )
            except Exception as exc:
                self.display_error = str(exc)
                warnings.warn(
                    "Optional ParaView output failed: " + str(exc), RuntimeWarning
                )

        self.last, self.count = time, self.count + 1
        self.record.update(frames=self.count, end=time)
        self.record["fields"] = [
            dict(
                name=self.primary_name or primary.name if i == 0 else f.name,
                element=str(f.function_space.ufl_element()),
                physical_shape=list(f.ufl_shape),
                unit=next(
                    (
                        v.unit
                        for v in self.result.fields.values()
                        if getattr(v.field, "value", v.field) is f
                    ),
                    None,
                ),
            )
            for i, f in enumerate((primary, *auxiliary))
        ]

    def __exit__(self, exc_type, exc, tb):
        error = exc
        if self.viewer is not None:
            try:
                self.viewer.__exit__(exc_type, exc, tb)
            except Exception as close_error:
                self.display_error = self.display_error or str(close_error)
        if self.writer is not None:
            try:
                self.writer.__exit__(exc_type, exc, tb)
            except Exception as close_error:
                error = error or close_error
        self.closed = True
        if error is None and not self.count:
            error = ValueError("Output series wrote no accepted frames.")
        if self.comm is not None:
            errors = self.comm.allgather(None if error is None else str(error))
            if any(errors) and error is None:
                error = RuntimeError("; ".join(e for e in errors if e))
        if error is None:
            check = inspect_output(self.path) if self.comm.rank == 0 else None
            check = self.comm.bcast(check, root=0)
            if len(check.get("coordinates", ())) != self.count:
                check["errors"].append(
                    "Written frame count differs from accepted frame count."
                )
                check["valid"] = False
            if not check["valid"]:
                error = ValueError("; ".join(check["errors"]))
        if error is not None:
            self.record.update(status="failed", error=str(error))
            self.result.status = "completed_with_output_errors"
            if exc is None:
                raise error
            return False
        if self.paraview and self.display_error is None:
            check = (
                inspect_output(self.path.with_suffix(".pvd"))
                if self.comm.rank == 0
                else None
            )
            check = self.comm.bcast(check, root=0)
            if not check["valid"]:
                self.display_error = "; ".join(check["errors"])
        self.record.update(
            status="completed",
            scientific_artifact=str(self.path),
            recommended_visualization_artifact=str(
                self.path.with_suffix(".pvd") if self.paraview else self.path
            ),
        )
        prefix = "fields"
        existing = self.result.artifacts.get("fields_xdmf")
        if existing is not None and Path(existing) != self.path:
            prefix = f"fields_segment_{len(self.result.metadata['output_series'])}"
        self.result.add_artifact(prefix + "_xdmf", self.path)
        self.result.add_artifact(prefix + "_hdf5", self.path.with_suffix(".h5"))
        if self.paraview:
            self.result.add_artifact(
                prefix + "_paraview", self.path.with_suffix(".pvd")
            )
        if self.comm.rank == 0:
            self.marker.unlink(missing_ok=True)
        if self.display_error:
            self.record.update(
                status="completed_with_output_errors",
                presentation_error=self.display_error,
                recommended_visualization_artifact=str(self.path),
            )
            self.result.status = "completed_with_output_errors"
            if self.strict:
                raise ValueError(self.display_error)
        return False


def output_series(
    result,
    path,
    *,
    paraview=False,
    coordinate_name="time",
    primary_name=None,
    primary_semantic_name=None,
    strict=True,
    overwrite=False,
    deformation_scale=0.0,
    compression=4,
    store_reference_geometry=True,
):
    """Stream solver-accepted frames with optional ParaView output and registration."""
    return OutputSeries(
        result,
        path,
        paraview=paraview,
        coordinate_name=coordinate_name,
        primary_name=primary_name,
        primary_semantic_name=primary_semantic_name,
        strict=strict,
        overwrite=overwrite,
        deformation_scale=deformation_scale,
        compression=compression,
        store_reference_geometry=store_reference_geometry,
    )


def validate_published_output(result):
    """Validate declared output only; scalar-only projects require no field files."""
    records = list(result.metadata.get("output_series", ()))
    legacy = result.metadata.get("field_output")
    if legacy:
        records.append(legacy)
    if (
        not records
        and any(Path(p).suffix.lower() == ".pvd" for p in result.artifacts.values())
        and not any(
            Path(p).suffix.lower() == ".xdmf" for p in result.artifacts.values()
        )
    ):
        result.metadata["output_advice"] = (
            "Only presentation files are registered. Use output_series with an .xdmf path "
            "to retain the default field dataset, or declare presentation-only output explicitly."
        )
    errors = []
    strict = bool(result.metadata.get("output_policy", {}).get("strict", False))
    for record in records:
        strict = strict or bool(record.get("strict", False))
        if record.get("status") in {"pending", "writing", "failed"}:
            errors.append(
                "Output has not completed: "
                + str(record.get("requested_path", "fields"))
            )
        if record.get("presentation_error"):
            errors.append("Presentation output failed: " + record["presentation_error"])
        path = record.get("scientific_artifact")
        if path:
            check = inspect_output(path)
            errors.extend(check["errors"])
            if Path(path).with_suffix(".incomplete").exists():
                errors.append(f"Output is incomplete: {path}")
        display = record.get("recommended_visualization_artifact")
        if display and display != path:
            errors.extend(inspect_output(display)["errors"])
    if errors:
        result.status = "completed_with_output_errors"
        result.metadata["output_validation"] = {"status": "failed", "errors": errors}
        message = "AFM-OUTPUT-INCOMPLETE: " + "; ".join(errors)
        if strict:
            raise OutputError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    elif records:
        result.metadata["output_validation"] = {"status": "completed"}
