"""Surrogate, reduced-order, and physics-learning contracts.

AgentFEM owns scientific inputs, outputs, provenance, validation, and
applicability. Numerical learning remains delegated to focused libraries.
"""

from .base import (
    Prediction,
    QuantityMetrics,
    SurrogateValidationReport,
    validate_predictions,
)
from .domain import (
    BoxApplicabilityDomain,
    GuardedSurrogate,
    OutOfDomainError,
)
from .linear import (
    PODRidgeSurrogate,
    RidgeSurrogate,
    TrainedPODRidge,
    TrainedRidge,
)
from .physics import (
    FieldEncoding,
    NeuralOperatorSpec,
    ObservationGrid,
    PhysicsCondition,
    PhysicsResidual,
    PINNSpec,
    regular_grid,
)
from .torch_adapter import TorchMLPSurrogate, TrainedTorchMLP
from .pinn_torch import PINNTrainingRecord, TorchPINNAdapter
from .training import SurrogateTrainingRun, train

__all__ = [
    "BoxApplicabilityDomain",
    "FieldEncoding",
    "GuardedSurrogate",
    "NeuralOperatorSpec",
    "ObservationGrid",
    "OutOfDomainError",
    "PINNSpec",
    "PINNTrainingRecord",
    "PODRidgeSurrogate",
    "PhysicsCondition",
    "PhysicsResidual",
    "Prediction",
    "QuantityMetrics",
    "RidgeSurrogate",
    "SurrogateValidationReport",
    "SurrogateTrainingRun",
    "TorchMLPSurrogate",
    "TorchPINNAdapter",
    "TrainedPODRidge",
    "TrainedRidge",
    "TrainedTorchMLP",
    "validate_predictions",
    "regular_grid",
    "train",
]
