"""Compatibility wrappers for external mesh-format conversion.

New code should use ``agentfem.mesh.formats`` or the convenience functions in
``agentfem.mesh``.
"""

from __future__ import annotations

from .mesh.formats import (
    MeshConversionResult,
    convert_abaqus_inp_to_xdmf,
    convert_comsol_export_to_xdmf,
    convert_nastran_to_xdmf,
    convert_to_xdmf,
    describe_supported_external_formats,
    detect_mesh_format,
    require_meshio,
)

__all__ = [
    "MeshConversionResult",
    "convert_abaqus_inp_to_xdmf",
    "convert_comsol_export_to_xdmf",
    "convert_nastran_to_xdmf",
    "convert_to_xdmf",
    "describe_supported_external_formats",
    "detect_mesh_format",
    "require_meshio",
]
