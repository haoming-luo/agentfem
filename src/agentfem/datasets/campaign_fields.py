"""Quality-gated assembly of field datasets from campaign evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from ..ir.schema import to_json_safe
from .fields import ScientificFieldDataset


@dataclass(frozen=True)
class FieldCaseData:
    """Physical fields extracted from one successful campaign case."""

    fields: Mapping[str, object]
    coordinates: Mapping[str, object] = field(default_factory=dict)
    masks: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", dict(self.fields))
        object.__setattr__(self, "coordinates", dict(self.coordinates))
        object.__setattr__(self, "masks", dict(self.masks))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.fields:
            raise ValueError("FieldCaseData requires at least one physical field.")


@dataclass(frozen=True)
class FieldDatasetAssembler:
    """Declarative bridge from accepted Campaign records to complete fields.

    AgentFEM owns case identity, quality gates, provenance, parameters and the
    field-dataset contract.  The caller-owned ``extract`` function owns the
    scientific decision of which fields and coordinates represent one case.
    It receives ``(CampaignCase, CaseOutcome)`` and returns ``FieldCaseData``.
    """

    encodings: tuple[object, ...]
    extract: Callable[[object, object], FieldCaseData | Mapping[str, object]]
    parameter_names: tuple[str, ...] | None = None
    name: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        encodings = tuple(self.encodings)
        if not encodings:
            raise ValueError("FieldDatasetAssembler requires field encodings.")
        if not callable(self.extract):
            raise TypeError("FieldDatasetAssembler.extract must be callable.")
        names = tuple(_encoding_name(item) for item in encodings)
        if len(set(names)) != len(names):
            raise ValueError("FieldDatasetAssembler encoding names must be unique.")
        parameter_names = (
            None
            if self.parameter_names is None
            else tuple(str(item).strip() for item in self.parameter_names)
        )
        if parameter_names is not None:
            if any(not item for item in parameter_names):
                raise ValueError("FieldDatasetAssembler parameter names must not be empty.")
            if len(set(parameter_names)) != len(parameter_names):
                raise ValueError("FieldDatasetAssembler parameter names must be unique.")
        object.__setattr__(self, "encodings", encodings)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def assemble(
        self,
        report,
        *,
        allow_partial: bool = False,
        minimum_cases: int = 1,
        minimum_trust_level: str | None = None,
        quality: str | None = None,
    ) -> ScientificFieldDataset:
        """Assemble fields only from cases accepted by Campaign's data gate."""

        require_dataset = getattr(report, "require_dataset", None)
        if not callable(require_dataset) or not hasattr(report, "records"):
            raise TypeError("FieldDatasetAssembler requires a CampaignReport.")
        accepted = require_dataset(
            allow_partial=allow_partial,
            minimum_samples=minimum_cases,
            minimum_trust_level=minimum_trust_level,
            quality=quality,
        )
        accepted_ids = {sample.case_id for sample in accepted.samples}
        records = tuple(
            record
            for record in report.records
            if record.successful and record.case.case_id in accepted_ids
        )
        if len(records) != len(accepted_ids):
            raise RuntimeError("Accepted Campaign samples and successful records disagree.")

        case_data = tuple(
            _field_case_data(self.extract(record.case, record.outcome))
            for record in records
        )
        field_names = tuple(_encoding_name(item) for item in self.encodings)
        extracted_field_names = _common_group_names(case_data, "fields")
        if set(extracted_field_names) != set(field_names):
            raise ValueError(
                "Extracted fields must match declared encodings exactly; "
                f"missing={tuple(sorted(set(field_names) - set(extracted_field_names)))!r}, "
                f"extra={tuple(sorted(set(extracted_field_names) - set(field_names)))!r}."
            )
        fields = _stack_case_group(case_data, "fields", field_names)
        coordinate_names = _common_group_names(case_data, "coordinates")
        mask_names = _common_group_names(case_data, "masks")
        coordinates = _stack_case_group(case_data, "coordinates", coordinate_names)
        masks = _stack_case_group(case_data, "masks", mask_names)

        parameter_names = (
            tuple(report.plan.parameter_space.names)
            if self.parameter_names is None
            else self.parameter_names
        )
        missing_parameters = set(parameter_names).difference(report.plan.parameter_space.names)
        if missing_parameters:
            raise ValueError(
                "FieldDatasetAssembler parameters are absent from the Campaign space: "
                f"{tuple(sorted(missing_parameters))!r}."
            )
        parameters = {
            name: _numeric_parameters(name, records)
            for name in parameter_names
        }
        case_metadata = tuple(
            {
                **to_json_safe(data.metadata),
                "campaign_case_id": record.case.case_id,
                "campaign_parameters": to_json_safe(record.case.parameters),
                "campaign_duration_seconds": float(record.duration_seconds),
                "campaign_reused": bool(record.reused),
                "campaign_provenance": to_json_safe(record.outcome.provenance),
                "campaign_artifacts": to_json_safe(record.outcome.artifacts),
            }
            for record, data in zip(records, case_data, strict=True)
        )
        return ScientificFieldDataset(
            case_ids=tuple(record.case.case_id for record in records),
            encodings=self.encodings,
            fields=fields,
            coordinates=coordinates,
            parameters=parameters,
            masks=masks,
            case_metadata=case_metadata,
            name=self.name or f"{report.name}_fields",
            metadata={
                **dict(self.metadata),
                "source": "agentfem_campaign_report",
                "campaign": report.name,
                "campaign_valid": bool(report.valid),
                "campaign_sampling": report.plan.sampling.summary(),
                "campaign_runtime": to_json_safe(report.runtime),
                "campaign_scientific_inputs": to_json_safe(report.scientific_inputs),
                "campaign_dataset_metadata": to_json_safe(accepted.metadata),
                "quality_gate": {
                    "allow_partial": bool(allow_partial),
                    "minimum_cases": int(minimum_cases),
                    "minimum_trust_level": minimum_trust_level,
                    "quality": quality,
                },
            },
        )


def _encoding_name(encoding: object) -> str:
    if isinstance(encoding, Mapping):
        record = dict(encoding)
    else:
        summary = getattr(encoding, "summary", None)
        if not callable(summary):
            raise TypeError(
                "Field encodings must be mappings or expose a summary() method."
            )
        record = dict(summary())
    name = str(record.get("name", "")).strip()
    if not name:
        raise ValueError("Every field encoding requires a non-empty name.")
    return name


def _field_case_data(value) -> FieldCaseData:
    if isinstance(value, FieldCaseData):
        return value
    if isinstance(value, Mapping):
        return FieldCaseData(**dict(value))
    raise TypeError("Field extraction must return FieldCaseData or a compatible mapping.")


def _common_group_names(cases: tuple[FieldCaseData, ...], attribute: str) -> tuple[str, ...]:
    first = tuple(getattr(cases[0], attribute))
    expected = set(first)
    for index, case in enumerate(cases[1:], start=1):
        actual = set(getattr(case, attribute))
        if actual != expected:
            raise ValueError(
                f"Field case {index} {attribute} names differ; "
                f"expected={tuple(sorted(expected))!r}, actual={tuple(sorted(actual))!r}."
            )
    return first


def _stack_case_group(
    cases: tuple[FieldCaseData, ...],
    attribute: str,
    names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in names:
        values = [np.asarray(getattr(case, attribute)[name]) for case in cases]
        shapes = tuple(value.shape for value in values)
        if len(set(shapes)) != 1:
            raise ValueError(
                f"Field case {attribute} {name!r} shapes must match; got {shapes!r}."
            )
        try:
            arrays[name] = np.stack(values, axis=0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Field case {attribute} {name!r} values cannot form one numeric array."
            ) from exc
    return arrays


def _numeric_parameters(name: str, records) -> np.ndarray:
    values = np.asarray([record.case.parameters[name] for record in records])
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(
            f"Field-dataset parameter {name!r} must be numeric; categorical parameters "
            "belong in case metadata or a declared numeric encoding."
        )
    numeric = np.asarray(values, dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Field-dataset parameter {name!r} contains non-finite values.")
    return numeric


__all__ = ["FieldCaseData", "FieldDatasetAssembler"]
