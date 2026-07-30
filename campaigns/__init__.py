"""Parameter studies and traceable batch execution for AgentFEM.

The campaign layer owns scientific case identity and execution evidence. It
does not prescribe how one finite-element case is built or solved.
"""

from .core import (
    Campaign,
    CampaignCase,
    CampaignPlan,
    CampaignReport,
    CaseOutcome,
    CaseRunRecord,
    ExecutionPolicy,
    case_id,
    create,
)
from .parameters import (
    ChoiceParameter,
    IntegerParameter,
    ParameterSpace,
    RealParameter,
    SamplingPlan,
    explicit,
    full_factorial,
    latin_hypercube,
    random,
)

__all__ = [
    "Campaign",
    "CampaignCase",
    "CampaignPlan",
    "CampaignReport",
    "CaseOutcome",
    "CaseRunRecord",
    "ChoiceParameter",
    "ExecutionPolicy",
    "IntegerParameter",
    "ParameterSpace",
    "RealParameter",
    "SamplingPlan",
    "case_id",
    "create",
    "explicit",
    "full_factorial",
    "latin_hypercube",
    "random",
]
