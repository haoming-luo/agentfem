"""Scientific datasets linking learned samples to simulation evidence."""

from .core import DatasetSplit, ScientificDataset
from .external import (
    ExternalDatasetAudit,
    ExternalDatasetManifest,
    ExternalFile,
    SpreadsheetSheet,
    SpreadsheetWorkbook,
    read_xlsx_workbook,
    science_supershear_dryad_manifest,
    science_supershear_v5_research_task,
)
from .schema import Quantity, Sample, decode_quantities
from .torch import (
    FEMFieldSample,
    TorchDatasetBundle,
    fem_field_sample,
    fem_observation_sample,
    to_torch,
)

__all__ = [
    "DatasetSplit",
    "ExternalDatasetAudit",
    "ExternalDatasetManifest",
    "ExternalFile",
    "FEMFieldSample",
    "Quantity",
    "Sample",
    "ScientificDataset",
    "SpreadsheetSheet",
    "SpreadsheetWorkbook",
    "TorchDatasetBundle",
    "decode_quantities",
    "fem_field_sample",
    "fem_observation_sample",
    "read_xlsx_workbook",
    "science_supershear_dryad_manifest",
    "science_supershear_v5_research_task",
    "to_torch",
]
