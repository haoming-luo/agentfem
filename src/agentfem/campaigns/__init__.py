"""Parameter studies and traceable batch execution for AgentFEM.

The campaign layer owns scientific case identity and execution evidence. It
does not prescribe how one finite-element case is built or solved.
"""

from importlib import import_module as _import_module

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

_LAZY_EXPORTS = {
    "Campaign": "core",
    "CampaignCase": "core",
    "CampaignPlan": "core",
    "CampaignReport": "core",
    "CaseOutcome": "core",
    "CaseRunRecord": "core",
    "ExecutionPolicy": "core",
    "case_id": "core",
    "create": "core",
    "local_processes": "core",
    "CampaignSpecification": "config",
    "load_specification": "config",
    "specification_from_dict": "config",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value

__all__ = [
    "Campaign",
    "CampaignCase",
    "CampaignPlan",
    "CampaignReport",
    "CampaignSpecification",
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
    "load_specification",
    "local_processes",
    "random",
    "specification_from_dict",
]
