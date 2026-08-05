"""Scientific datasets linking learned samples to simulation evidence."""

from .core import DatasetSplit, ScientificDataset
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
    "FEMFieldSample",
    "Quantity",
    "Sample",
    "ScientificDataset",
    "TorchDatasetBundle",
    "decode_quantities",
    "fem_field_sample",
    "fem_observation_sample",
    "to_torch",
]
