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

__all__ = [
    "BoxApplicabilityDomain",
    "FieldEncoding",
    "GuardedSurrogate",
    "NeuralOperatorSpec",
    "OutOfDomainError",
    "PINNSpec",
    "PODRidgeSurrogate",
    "PhysicsCondition",
    "PhysicsResidual",
    "Prediction",
    "QuantityMetrics",
    "RidgeSurrogate",
    "SurrogateValidationReport",
    "TorchMLPSurrogate",
    "TrainedPODRidge",
    "TrainedRidge",
    "TrainedTorchMLP",
    "validate_predictions",
]
