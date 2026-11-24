# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Local result selection restricted to an explicitly selected directory."""

from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

SUFFIXES = frozenset({".xdmf", ".xmf", ".vtu", ".vtk", ".pvd"})


def result_browser_root(requested):
    """A file opens its containing project, or just its folder if standalone."""
    requested = Path(requested).expanduser().resolve(strict=True)
    if requested.is_dir():
        return requested
    for parent in requested.parents:
        if (parent / "agentfem.toml").is_file():
            return parent
    return requested.parent


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

    def projects(self):
        """Discover only the selected root and its immediate children."""
        from agentfem.project import ProjectConfig

        items = []
        candidates = [self.root] + [self.resolve(item["value"])
                                  for item in self.entries() if item["directory"]]
        for candidate in candidates:
            config = candidate / "agentfem.toml"
            if not config.is_file():
                continue
            try:
                project = ProjectConfig.load(self.resolve(config))
                if not project.output_directory.is_relative_to(candidate):
                    continue
            except (ValueError, OSError, TypeError):
                continue
            items.append({"title": project.name,
                          "value": str(candidate.relative_to(self.root))})
        counts = Counter(item["title"] for item in items)
        for item in items:
            if counts[item["title"]] > 1:
                item["title"] += " · " + item["value"]
        return items

    def runs(self, project):
        """Use the project's configured output directory, without loading code."""
        from agentfem.project import ProjectConfig

        root = self.resolve(project)
        config = ProjectConfig.load(self.resolve(root / "agentfem.toml"))
        if not config.output_directory.is_relative_to(root):
            raise ValueError("Project output directory must remain inside its project.")
        if not config.output_directory.exists():
            return []
        return [{"title": Path(item["value"]).name, "value": item["value"]}
                for item in self.entries(config.output_directory) if item["directory"]]

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
