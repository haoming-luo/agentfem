"""Scientific-learning contracts and compatibility entry points.

``learning`` is the public umbrella for simulation-to-learning workflows.
Exact concepts remain explicit: surrogates approximate declared mappings,
neural operators learn function-to-function maps, and neural-field solvers
optimize one physical field problem.  The established ``surrogates`` module
remains public throughout the 0.2.x series.
"""

from ..surrogates import (
    AffineCoordinateMap,
    BoxApplicabilityDomain,
    FieldEncoding,
    GuardedSurrogate,
    NeuralOperatorSpec,
    ObservationGrid,
    OutOfDomainError,
    PINNSpec,
    PINNTrainingRecord,
    PODRidgeSurrogate,
    Prediction,
    PhysicsCondition,
    PhysicsResidual,
    QuantityMetrics,
    RidgeSurrogate,
    SurrogateValidationReport,
    SurrogateTrainingRun,
    TorchMLPSurrogate,
    TorchPINNAdapter,
    TrainedPODRidge,
    TrainedRidge,
    TrainedTorchMLP,
    regular_grid,
    train,
    validate_predictions,
)
from .core import (
    ConditionSpec,
    IntegrationEvidence,
    IntegrationPlan,
    IntegrationRule,
    NeuralFieldSpec,
    NeuralRepresentation,
    ObjectiveTerm,
    SamplingPlan,
    TrainableParameter,
    integration_consistency_check,
)
from .execution import NeuralFieldExecutionRequest


__all__ = [
    "AffineCoordinateMap",
    "BoxApplicabilityDomain",
    "ConditionSpec",
    "FieldEncoding",
    "GuardedSurrogate",
    "IntegrationEvidence",
    "IntegrationPlan",
    "IntegrationRule",
    "NeuralFieldSpec",
    "NeuralFieldExecutionRequest",
    "NeuralRepresentation",
    "NeuralOperatorSpec",
    "ObjectiveTerm",
    "ObservationGrid",
    "OutOfDomainError",
    "PINNSpec",
    "PINNTrainingRecord",
    "PODRidgeSurrogate",
    "Prediction",
    "PhysicsCondition",
    "PhysicsResidual",
    "QuantityMetrics",
    "RidgeSurrogate",
    "SamplingPlan",
    "SurrogateTrainingRun",
    "SurrogateValidationReport",
    "TorchMLPSurrogate",
    "TorchPINNAdapter",
    "TrainableParameter",
    "TrainedPODRidge",
    "TrainedRidge",
    "TrainedTorchMLP",
    "regular_grid",
    "train",
    "validate_predictions",
    "integration_consistency_check",
]
