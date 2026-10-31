# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""A checked visualization view of immutable, reference-mesh result segments."""

from copy import deepcopy
from decimal import Decimal
import hashlib
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


def join_xdmf_segments(sources, destination, *, expected_times=None) -> Path:
    """Publish one temporal index without copying or modifying heavy data.

    Accepts single-mesh uniform-frame temporal XDMF with direct HDF references.
    Reference geometry, topology and field schema must agree. Adjacent segments
    may share one boundary time only if every saved array is identical there.
    Disjoint stations are retained without interpolation. Supply expected_times
    to detect missing requested stations; sparse output alone cannot establish
    temporal completeness or checkpoint/scientific continuity.

    The result is a snapshot of the published indexes, not a live subscription.
    Keep all source HDF5 files. Deformed geometry, remeshing, external XML
    includes and implicit mesh references require a separate transfer contract.
    """
    import h5py
    import numpy as np

    paths = tuple(Path(source).resolve() for source in sources)
    destination = Path(destination).resolve()
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Supply distinct, ordered XDMF segment paths.")
    if destination in paths or destination.exists():
        raise FileExistsError("Choose a fresh view path; source output is preserved.")
    signatures = {}

    def dataset(item, source, *, digest=False):
        if item.get("Format") != "HDF" or list(item):
            raise ValueError("Only direct HDF DataItems are supported.")
        filename, sep, name = (item.text or "").strip().partition(":")
        if not filename or not sep or not name.startswith("/"):
            raise ValueError("Invalid HDF reference.")
        heavy = (source.parent / filename).resolve()
        key = (heavy, name)
        with h5py.File(heavy, "r") as store:
            array = store[name]
            if not isinstance(array, h5py.Dataset) or array.dtype.kind not in "biuf":
                raise ValueError("Expected a real numeric HDF dataset.")
            shape = tuple(int(n) for n in item.get("Dimensions", "").split())
            if not shape or shape != array.shape:
                raise ValueError("XDMF dimensions do not match the HDF dataset.")
            identity = (shape, array.dtype.str)
            if digest and key not in signatures:
                checksum = hashlib.sha256()
                # Bound memory even for large meshes and boundary-state arrays.
                rows = max(1, 1024 * 1024 // max(1, array.dtype.itemsize *
                            int(np.prod(shape[1:]))))
                for start in range(0, shape[0], rows):
                    checksum.update(np.ascontiguousarray(array[start:start + rows]).tobytes())
                signatures[key] = (*identity, checksum.hexdigest())
        item.text = os.path.relpath(heavy, destination.parent) + ":" + name
        return signatures[key] if digest else identity

    def time_value(value):
        try:
            result = float(value)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("Invalid frame time.") from error
        if not math.isfinite(result):
            raise ValueError("Frame times must be finite.")
        return Decimal.from_float(result)

    output = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(output, "Domain")
    ET.SubElement(domain, "Information", Name="agentfem_view", Value="visualization_only")
    target = ET.SubElement(domain, "Grid", Name="Results", GridType="Collection",
                           CollectionType="Temporal")
    mesh_identity = schema = previous = None
    retained = {}
    for source in paths:
        root = ET.parse(source).getroot()
        source_domain = root.find("Domain")
        if source_domain is None:
            raise ValueError("Missing XDMF Domain.")
        collections = source_domain.findall("Grid")
        if (len(collections) != 1 or collections[0].get("CollectionType") != "Temporal"
                or not collections[0].findall("Grid")):
            raise ValueError("Expected one nonempty temporal collection.")
        ET.SubElement(domain, "Information", Name="source_segment",
                      Value=os.path.relpath(source, destination.parent))
        if any(item.get("Value") == "incomplete" for item in source_domain.findall("Information")):
            ET.SubElement(domain, "Information", Name="source_status", Value="incomplete")
        local_previous = None
        for index, original in enumerate(collections[0].findall("Grid")):
            grid = deepcopy(original)
            if (grid.get("GridType") != "Uniform" or
                    any(item.tag not in {"Time", "Topology", "Geometry", "Attribute", "Information"}
                        for item in grid)):
                raise ValueError("Only explicit uniform frames are supported.")
            times = grid.findall("Time")
            if len(times) != 1 or times[0].get("Value") is None:
                raise ValueError("Every frame needs one explicit time.")
            station = time_value(times[0].get("Value"))
            if local_previous is not None and station <= local_previous:
                raise ValueError("Segment times must increase strictly.")
            local_previous = station
            mesh = []
            for tag in ("Topology", "Geometry"):
                elements = grid.findall(tag)
                if len(elements) != 1 or len(elements[0]) != 1 or elements[0][0].tag != "DataItem":
                    raise ValueError("Expected explicit topology and geometry arrays.")
                mesh.append((tuple(sorted(elements[0].attrib.items())),
                             dataset(elements[0][0], source, digest=True)))
            if mesh_identity is not None and mesh != mesh_identity:
                raise ValueError("Segments require the same reference mesh and ordering.")
            mesh_identity = mesh
            fields = {}
            boundary = previous is not None and station == previous
            for attribute in grid.findall("Attribute"):
                name = attribute.get("Name")
                center = attribute.get("Center")
                identity = (center, name)
                if not name or center not in {"Node", "Cell"} or identity in fields:
                    raise ValueError("Missing or duplicate field identity.")
                items = attribute.findall("DataItem")
                if len(items) != 1:
                    raise ValueError("Expected one direct DataItem per field.")
                fields[identity] = (tuple(sorted(attribute.attrib.items())),
                                    dataset(items[0], source, digest=boundary)[:2])
            if not fields or (schema is not None and fields != schema):
                raise ValueError("Field schema changes between frames.")
            schema = fields
            if previous is not None and station <= previous:
                if not boundary or index != 0:
                    raise ValueError("Segments overlap or are not in time order.")
                old_grid, old_source = retained[station]
                # Resolve against the source, not against already rebased view refs.
                old_attributes = {(a.get("Center"), a.get("Name")): a
                                  for a in old_grid.findall("Attribute")}
                for attribute in original.findall("Attribute"):
                    identity = (attribute.get("Center"), attribute.get("Name"))
                    old = deepcopy(old_attributes[identity].find("DataItem"))
                    new = deepcopy(attribute.find("DataItem"))
                    if dataset(old, old_source, digest=True) != dataset(new, source, digest=True):
                        raise ValueError("Shared boundary frame has conflicting field values.")
                continue
            target.append(grid)
            retained[station] = (original, source)
            previous = station
    if expected_times is not None:
        expected = {time_value(value) for value in expected_times}
        if not expected.issubset(retained):
            raise ValueError("Requested output stations are missing from the segments.")
    ET.SubElement(domain, "Information", Name="temporal_coverage",
                  Value="requested_stations_present" if expected_times is not None else "not_certified")
    ET.indent(output, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            ET.ElementTree(output).write(handle, encoding="utf-8", xml_declaration=True)
            handle.flush()
            os.fsync(handle.fileno())
        # Publish only a complete descriptor, without replacing an existing file.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
