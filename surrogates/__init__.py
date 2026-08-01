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
    PhysicsCondition,
    PhysicsResidual,
    PINNSpec,
)
from .torch_adapter import TorchMLPSurrogate, TrainedTorchMLP
from .pinn_torch import PINNTrainingRecord, TorchPINNAdapter

__all__ = [
    "BoxApplicabilityDomain",
    "FieldEncoding",
    "GuardedSurrogate",
    "NeuralOperatorSpec",
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
    "TorchMLPSurrogate",
    "TorchPINNAdapter",
    "TrainedPODRidge",
    "TrainedRidge",
    "TrainedTorchMLP",
    "validate_predictions",
]
