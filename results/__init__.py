"""Simulation results, quantities of interest, and dataset bridges."""

from .core import (
    FieldResult,
    HistoryResult,
    ResultQuantity,
    SimulationResult,
    dof_statistics,
    from_solution,
)
from .quantities import average, integral, l2_norm, quadrature_extrema
from .finite_strain import (
    HomogenizedFrame,
    finite_strain_cell_fields,
    homogenize_periodic_cell,
    write_homogenized_csv,
    write_homogenized_history,
)
from .field_catalog import (
    FieldVariable,
    field_variable,
    preselected_fields,
    resolve_field_variables,
)
from .output import (
    FieldOutput,
    FieldOutputArtifacts,
    field_output,
    read_unified_xdmf_series,
    write_deformed_vtk_series,
    write_unified_xdmf_series,
)
from .visualization import (
    render_deformation_animation,
    render_deformation_comparison,
    render_unified_xdmf_animation,
    render_unified_xdmf_comparison,
    render_vtk_series_animation,
)

__all__ = [
    "FieldResult",
    "FieldOutput",
    "FieldOutputArtifacts",
    "FieldVariable",
    "HistoryResult",
    "HomogenizedFrame",
    "ResultQuantity",
    "SimulationResult",
    "dof_statistics",
    "from_solution",
    "average",
    "integral",
    "finite_strain_cell_fields",
    "field_output",
    "field_variable",
    "homogenize_periodic_cell",
    "l2_norm",
    "quadrature_extrema",
    "preselected_fields",
    "read_unified_xdmf_series",
    "render_deformation_comparison",
    "render_deformation_animation",
    "render_unified_xdmf_animation",
    "render_unified_xdmf_comparison",
    "resolve_field_variables",
    "render_vtk_series_animation",
    "write_homogenized_csv",
    "write_homogenized_history",
    "write_deformed_vtk_series",
    "write_unified_xdmf_series",
]
