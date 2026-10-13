# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Presentation-only conversion of single-mesh DOLFINx XDMF archives."""

from copy import deepcopy
from decimal import Decimal
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def consolidate_xdmf_fields(source, destination) -> Path:
    """Group synchronized fields on one grid per time, without copying HDF5.

    The original archive is preserved. This deliberately accepts only the
    single reference mesh / per-field temporal layout emitted by DOLFINx.
    Heterogeneous meshes, asynchronous fields and external XML includes are
    rejected rather than guessed. The returned descriptor depends on the
    original heavy-data files; keep them when sharing the result.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError("Choose a new XDMF descriptor; original output is preserved.")
    root = ET.parse(source).getroot()
    domain = root.find("Domain")
    if domain is None:
        raise ValueError("Missing XDMF Domain.")
    meshes = domain.findall("Grid[@GridType='Uniform']")
    if len(meshes) != 1:
        raise ValueError("Expected exactly one reference mesh.")
    mesh = meshes[0]
    topology, geometry = mesh.find("Topology"), mesh.find("Geometry")
    if topology is None or geometry is None or mesh.find("Attribute") is not None:
        raise ValueError("Expected a reference mesh without result attributes.")
    collections = domain.findall("Grid[@CollectionType='Temporal']")
    if not collections or len(domain.findall("Grid")) != len(collections) + 1:
        raise ValueError("Expected per-field temporal collections on one mesh.")
    frames = {}
    expected_times = None
    allowed_pointer = "xpointer(/Xdmf/Domain/Grid[@GridType='Uniform'][1]/*[self::Topology or self::Geometry])"
    for collection in collections:
        times = set()
        for grid in collection.findall("Grid"):
            children = list(grid)
            includes = [item for item in children if item.tag.endswith("}include")]
            if (grid.get("GridType") != "Uniform" or len(includes) != 1
                    or includes[0].get("href")
                    or includes[0].get("xpointer") != allowed_pointer
                    or any(item.tag not in ("Time", "Attribute", includes[0].tag)
                           for item in children)):
                raise ValueError("Unsupported mesh reference: refusing to merge distinct geometry.")
            time = grid.find("Time")
            if time is None or time.get("Value") is None:
                raise ValueError("Every result frame needs an explicit time.")
            key = Decimal(time.get("Value"))
            if not key.is_finite() or key in times:
                raise ValueError("Non-finite or duplicate frame time.")
            times.add(key)
            attributes = grid.findall("Attribute")
            if not attributes:
                raise ValueError("Result frame has no attributes.")
            frame = frames.setdefault(key, {})
            for attribute in attributes:
                name = attribute.get("Name")
                identity = (attribute.get("Center"), name)
                if not name or identity in frame:
                    raise ValueError("Missing or duplicate field identity at one time.")
                frame[identity] = deepcopy(attribute)
        if not times or (expected_times is not None and times != expected_times):
            raise ValueError("Fields have asynchronous time grids; no interpolation is implied.")
        expected_times = times
    contracts = [set(frame) for frame in frames.values()]
    if any(contract != contracts[0] for contract in contracts):
        raise ValueError("Fields change between frames.")
    output = ET.Element("Xdmf", Version="3.0")
    target = ET.SubElement(ET.SubElement(output, "Domain"), "Grid",
                           Name="Fields", GridType="Collection", CollectionType="Temporal")
    for time, attributes in sorted(frames.items()):
        grid = ET.SubElement(target, "Grid", Name="Fields", GridType="Uniform")
        grid.extend((deepcopy(topology), deepcopy(geometry)))
        ET.SubElement(grid, "Time", Value=str(time))
        grid.extend(attributes.values())
    # Rebase only file references, preserving all arrays, units and connectivity.
    for item in output.iter("DataItem"):
        if item.get("Format") == "HDF":
            filename, separator, dataset = (item.text or "").strip().partition(":")
            if not separator or not dataset.startswith("/"):
                raise ValueError("Unsupported HDF5 reference.")
            heavy = (source.parent / filename).resolve()
            if not heavy.is_file():
                raise FileNotFoundError(heavy)
            item.text = os.path.relpath(heavy, destination.parent) + ":" + dataset
    ET.indent(output, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        ET.ElementTree(output).write(handle, encoding="utf-8", xml_declaration=True)
    return destination
