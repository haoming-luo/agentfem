"""Declarative JSON specifications for parameter campaigns."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping

from ..datasets import Quantity

from .core import Campaign, ExecutionPolicy
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


@dataclass(frozen=True)
class CampaignSpecification:
    """Validated declarative part of a campaign.

    Python still supplies the trusted ``build`` and/or ``evaluate`` function.
    The file controls parameters, sampling, outputs, and execution policy; it
    never imports or evaluates arbitrary code.
    """

    name: str
    parameter_space: ParameterSpace
    sampling: SamplingPlan
    outputs: tuple[Quantity, ...]
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    description: str = ""

    def create_campaign(self, *, build=None, evaluate=None) -> Campaign:
        return Campaign(
            name=self.name,
            parameter_space=self.parameter_space,
            outputs=self.outputs,
            build=build,
            evaluate=evaluate,
            execution=self.execution,
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "campaign_specification",
            "name": self.name,
            "description": self.description,
            "parameter_space": self.parameter_space.summary(),
            "sampling": self.sampling.summary(),
            "outputs": tuple(item.summary() for item in self.outputs),
            "execution": self.execution.summary(),
        }


def load_specification(path: str | Path) -> CampaignSpecification:
    """Load a safe JSON campaign specification."""

    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError(
            "Campaign specifications currently use JSON. YAML may be added "
            "later as an optional syntax without changing the semantic model."
        )
    record = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Campaign specification must contain a JSON object.")
    return specification_from_dict(record)


def specification_from_dict(record: Mapping[str, object]) -> CampaignSpecification:
    """Validate a dictionary and construct a campaign specification."""

    name = str(record.get("name", "campaign")).strip()
    if not name:
        raise ValueError("Campaign specification name must not be empty.")
    parameter_records = record.get("parameters")
    if not isinstance(parameter_records, list) or not parameter_records:
        raise ValueError("Campaign specification requires a nonempty parameters list.")
    parameters = tuple(_parameter(item) for item in parameter_records)
    space = ParameterSpace.create(
        *parameters,
        name=str(record.get("parameter_space_name", f"{name}_parameters")),
    )
    sampling_record = record.get("sampling")
    if not isinstance(sampling_record, Mapping):
        raise ValueError("Campaign specification requires a sampling object.")
    sampling = _sampling(space, sampling_record)
    output_records = record.get("outputs")
    if not isinstance(output_records, list) or not output_records:
        raise ValueError("Campaign specification requires a nonempty outputs list.")
    outputs = tuple(_quantity(item) for item in output_records)
    execution_record = record.get("execution", {})
    if not isinstance(execution_record, Mapping):
        raise ValueError("Campaign execution must be an object.")
    execution = ExecutionPolicy(
        mode=str(execution_record.get("mode", "serial")),
        fail_fast=bool(execution_record.get("fail_fast", False)),
        resume=bool(execution_record.get("resume", True)),
    )
    return CampaignSpecification(
        name=name,
        parameter_space=space,
        sampling=sampling,
        outputs=outputs,
        execution=execution,
        description=str(record.get("description", "")),
    )


def _parameter(record) -> object:
    if not isinstance(record, Mapping):
        raise ValueError("Each campaign parameter must be an object.")
    kind = str(record.get("kind", "")).lower().replace("-", "_")
    common = {
        "name": record.get("name", ""),
        "description": str(record.get("description", "")),
        "nominal": record.get("nominal"),
    }
    if kind == "real":
        return RealParameter(
            **common,
            lower=record["lower"],
            upper=record["upper"],
            unit=record.get("unit"),
            scale=str(record.get("scale", "linear")),
        )
    if kind == "integer":
        return IntegerParameter(
            **common,
            lower=record["lower"],
            upper=record["upper"],
            unit=record.get("unit"),
        )
    if kind == "choice":
        return ChoiceParameter(
            **common,
            choices=tuple(record["choices"]),
        )
    raise ValueError(
        f"Unknown parameter kind {kind!r}; expected real, integer, or choice."
    )


def _sampling(space: ParameterSpace, record: Mapping[str, object]) -> SamplingPlan:
    method = str(record.get("method", "")).lower().replace("-", "_")
    if method in {"latin_hypercube", "lhs"}:
        return latin_hypercube(
            space,
            int(record["count"]),
            seed=int(record.get("seed", 0)),
        )
    if method in {"random", "random_uniform"}:
        return random(
            space,
            int(record["count"]),
            seed=int(record.get("seed", 0)),
        )
    if method == "full_factorial":
        levels = record.get("levels", 3)
        if isinstance(levels, Mapping):
            levels = {str(key): int(value) for key, value in levels.items()}
        else:
            levels = int(levels)
        return full_factorial(space, levels)
    if method == "explicit":
        samples = record.get("samples")
        if not isinstance(samples, list):
            raise ValueError("Explicit sampling requires a samples list.")
        return explicit(space, samples)
    raise ValueError(
        f"Unknown sampling method {method!r}; expected latin_hypercube, "
        "random, full_factorial, or explicit."
    )


def _quantity(record) -> Quantity:
    if not isinstance(record, Mapping):
        raise ValueError("Each campaign output must be an object.")
    return Quantity(
        name=record.get("name", ""),
        shape=tuple(record.get("shape", ())),
        unit=record.get("unit"),
        kind=str(record.get("kind", "quantity_of_interest")),
        description=str(record.get("description", "")),
        field_encoding=record.get("field_encoding"),
    )
