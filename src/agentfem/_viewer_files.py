# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Local result selection restricted to an explicitly selected directory."""

from pathlib import Path
import xml.etree.ElementTree as ET

SUFFIXES = frozenset({".xdmf", ".xmf", ".vtu", ".vtk", ".pvd"})


class ResultDirectory:
    """A local browser scope, not a network file-serving endpoint.

    Read trusted simulation output only. This checks XML companion paths;
    it is not a sandbox for malicious VTK/HDF5 files or native readers.
    """

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Select a result directory.")

    def resolve(self, name):
        path = (self.root / name).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("Select a file inside the opened result directory.")
        return path

    def entries(self, directory="."):
        current = self.resolve(directory)
        if not current.is_dir():
            raise ValueError("Select a result directory.")
        entries = []
        for child in sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold())):
            if child.name.startswith("."):
                continue
            try:
                resolved = self.resolve(child)
            except (OSError, ValueError):
                continue
            if resolved.is_dir() or resolved.suffix.lower() in SUFFIXES:
                entries.append({"title": ("Folder · " if resolved.is_dir() else "") + child.name,
                                "value": str(child.relative_to(self.root)),
                                "directory": resolved.is_dir()})
            if len(entries) >= 1000:
                raise ValueError("This folder has too many entries; open a more specific result directory.")
        return entries

    def result(self, name):
        path = self.resolve(name)
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            raise ValueError("Select XDMF, VTU, VTK or PVD output.")
        pending, seen = [path], set()
        while pending:
            source = pending.pop()
            if source in seen:
                continue
            seen.add(source)
            if len(seen) > 10000:
                raise ValueError("Result index has too many referenced files.")
            if source.suffix.lower() not in {".xdmf", ".xmf", ".pvd"}:
                continue
            document = ET.parse(source)
            for node in document.iter():
                tag = node.tag.rsplit("}", 1)[-1]
                reference = None
                if tag == "DataItem" and node.get("Format", "XML").upper() in {"HDF", "BINARY"}:
                    reference = (node.text or "").strip()
                    if node.get("Format", "XML").upper() == "HDF":
                        reference = reference.rsplit(":", 1)[0]
                elif tag == "DataSet" and source.suffix.lower() == ".pvd":
                    reference = node.get("file")
                elif tag == "include":
                    raise ValueError("Expand external XML includes before opening this result.")
                if reference is not None:
                    companion = self.resolve(source.parent / reference)
                    if not companion.is_file():
                        raise ValueError("Result companion is not a file.")
                    if source.suffix.lower() == ".pvd" and companion.suffix.lower() != ".vtu":
                        raise ValueError("Local PVD viewing currently requires VTU frame files.")
                    pending.append(companion)
        return path
