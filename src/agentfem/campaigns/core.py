"""Traceable, resumable serial, local-process, and sharded campaigns."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import socket
from time import perf_counter
from typing import Callable, Mapping

from ..datasets import Quantity, Sample, ScientificDataset
from ..ir.schema import to_json_safe
from ..provenance import (
    ORIGIN,
    content_fingerprint,
    runtime_manifest,
    scientific_input_manifest,
)
from ..results import SimulationResult
from .execution import run_local_processes
from .parameters import ParameterSpace, SamplingPlan


CAMPAIGN_SCHEMA = "agentfem.campaign"
CAMPAIGN_SCHEMA_VERSION = "0.1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def case_id(
    campaign_name: str,
    parameters: Mapping[str, object],
    *,
    schema_version: str = CAMPAIGN_SCHEMA_VERSION,
) -> str:
    """Return a deterministic scientific case identity."""

    payload = {
        "campaign": str(campaign_name),
        "parameters": to_json_safe(parameters),
        "schema_version": schema_version,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()[:20]


@dataclass(frozen=True)
class ExecutionPolicy:
    """Declared execution behavior for the current campaign runner."""

    mode: str = "serial"
    fail_fast: bool = False
    resume: bool = True
    workers: int | None = None

    def __post_init__(self) -> None:
        mode = self.mode.lower().replace("-", "_")
        if mode not in {"serial", "local_process"}:
            raise ValueError("Campaign execution mode must be serial or local_process.")
        workers = None if self.workers is None else int(self.workers)
        if workers is not None and workers < 1:
            raise ValueError("Campaign workers must be at least one.")
        if mode == "serial" and workers not in {None, 1}:
            raise ValueError("Serial campaign execution accepts at most one worker.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "workers", workers)

    def summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "fail_fast": self.fail_fast,
            "resume": self.resume,
            "workers": self.workers,
            "process_start_method": "spawn" if self.mode == "local_process" else None,
        }


@dataclass(frozen=True)
class CampaignCase:
    """One immutable case in a campaign plan."""

    case_id: str
    index: int
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", dict(self.parameters))

    def summary(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "index": self.index,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class CampaignPlan:
    """Immutable cases and their design-of-experiment evidence."""

    name: str
    parameter_space: ParameterSpace
    sampling: SamplingPlan
    cases: tuple[CampaignCase, ...]

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        if not cases:
            raise ValueError("CampaignPlan requires at least one case.")
        ids = tuple(case.case_id for case in cases)
        if len(set(ids)) != len(ids):
            raise ValueError(
                "CampaignPlan contains duplicate parameter samples/case identities."
            )
        object.__setattr__(self, "cases", cases)

    def shard(self, index: int, count: int) -> "CampaignPlan":
        """Return deterministic round-robin work for one external worker."""

        selected_index = int(index)
        selected_count = int(count)
        if selected_count <= 0 or not 0 <= selected_index < selected_count:
            raise ValueError("Shard requires count > 0 and 0 <= index < count.")
        selected = tuple(
            case for position, case in enumerate(self.cases)
            if position % selected_count == selected_index
        )
        if not selected:
            raise ValueError(
                f"Shard {selected_index}/{selected_count} contains no campaign cases."
            )
        sampling = SamplingPlan(
            space=self.parameter_space,
            samples=tuple(case.parameters for case in selected),
            method=f"{self.sampling.method}_shard",
            seed=self.sampling.seed,
            metadata={
                **dict(self.sampling.metadata or {}),
                "parent_case_count": len(self.cases),
                "shard_index": selected_index,
                "shard_count": selected_count,
            },
        )
        return CampaignPlan(
            name=self.name,
            parameter_space=self.parameter_space,
            sampling=sampling,
            cases=selected,
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_SCHEMA,
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "name": self.name,
            "case_count": len(self.cases),
            "sampling": self.sampling.summary(),
            "cases": [case.summary() for case in self.cases],
        }


@dataclass(frozen=True)
class CaseOutcome:
    """Successful case outputs plus links to scientific evidence."""

    outputs: Mapping[str, object]
    provenance: Mapping[str, object] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", dict(self.outputs))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "artifacts", dict(self.artifacts))


@dataclass(frozen=True)
class CaseRunRecord:
    """Execution evidence for one attempted case."""

    case: CampaignCase
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    outcome: CaseOutcome | None = None
    error_type: str | None = None
    error_message: str | None = None
    reused: bool = False
    execution: Mapping[str, object] = field(default_factory=dict)

    @property
    def successful(self) -> bool:
        return self.status == "completed" and self.outcome is not None

    def summary(self) -> dict[str, object]:
        result = {
            "case": self.case.summary(),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "reused": self.reused,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "execution": dict(self.execution),
        }
        if self.outcome is not None:
            result["outcome"] = {
                "outputs": self.outcome.outputs,
                "provenance": self.outcome.provenance,
                "artifacts": self.outcome.artifacts,
            }
        return result


@dataclass(frozen=True)
class CampaignReport:
    """Case-level evidence and the successful scientific dataset."""

    name: str
    plan: CampaignPlan
    records: tuple[CaseRunRecord, ...]
    dataset: ScientificDataset | None
    output_directory: Path | None = None
    runtime: Mapping[str, object] = field(default_factory=dict)
    execution: Mapping[str, object] = field(default_factory=dict)
    scientific_inputs: Mapping[str, object] = field(default_factory=dict)

    @property
    def completed(self) -> int:
        return sum(record.successful for record in self.records)

    @property
    def failed(self) -> int:
        return len(self.records) - self.completed

    @property
    def valid(self) -> bool:
        return self.failed == 0

    def summary(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_SCHEMA,
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "name": self.name,
            "valid": self.valid,
            "case_count": len(self.records),
            "completed": self.completed,
            "failed": self.failed,
            "dataset": None if self.dataset is None else self.dataset.summary(),
            "output_directory": (
                None if self.output_directory is None else str(self.output_directory)
            ),
            "runtime": dict(self.runtime),
            "execution": dict(self.execution),
            "scientific_inputs": dict(self.scientific_inputs),
            "records": [record.summary() for record in self.records],
        }

    def require_dataset(
        self,
        *,
        allow_partial: bool = False,
        minimum_samples: int = 1,
        minimum_trust_level: str | None = None,
        quality: str | None = None,
    ) -> ScientificDataset:
        """Return training data only when campaign evidence is acceptable.

        Failed simulations are valuable evidence, but silently training on the
        remaining cases can bias a learned model. The default therefore
        rejects partial campaigns. An explicit ``allow_partial=True`` marks
        the returned dataset with the failed case identities so that this
        review decision survives dataset serialization. When
        ``minimum_trust_level`` is supplied, successful simulations whose
        verification evidence is below that level are rejected as well. A
        named ``quality`` preset is stricter: every sample must have passed an
        assessment at the requested policy level, and the acceptance decision
        is copied into the returned dataset metadata.
        """

        if minimum_samples < 1:
            raise ValueError("minimum_samples must be at least one.")
        if quality is not None and minimum_trust_level is not None:
            raise ValueError(
                "Choose either a quality preset or minimum_trust_level, not both."
            )
        if self.dataset is None:
            raise RuntimeError(
                f"Campaign {self.name!r} produced no successful dataset; "
                f"{self.failed} case(s) failed."
            )
        failed_ids = tuple(
            record.case.case_id for record in self.records if not record.successful
        )
        if failed_ids and not allow_partial:
            raise RuntimeError(
                f"Campaign {self.name!r} has {self.failed} failed case(s) "
                f"{failed_ids}; review them or pass allow_partial=True explicitly."
            )
        if len(self.dataset.samples) < minimum_samples:
            raise RuntimeError(
                f"Campaign {self.name!r} produced {len(self.dataset.samples)} "
                f"successful sample(s), fewer than required {minimum_samples}."
            )
        quality_decision = None
        if quality is not None:
            from ..verification import quality_policy, trust_rank

            selected_policy = quality_policy(quality)
            minimum_trust_level = selected_policy.minimum_trust_level
            unassessed = []
            for sample in self.dataset.samples:
                summary = sample.provenance.get("simulation_result", {})
                evidence = (
                    summary.get("verification")
                    if isinstance(summary, Mapping)
                    else None
                )
                assessed_policy = (
                    evidence.get("quality_policy")
                    if isinstance(evidence, Mapping)
                    else None
                )
                acceptable = bool(
                    isinstance(evidence, Mapping)
                    and evidence.get("acceptable")
                )
                if assessed_policy is None:
                    unassessed.append((sample.case_id, None))
                    continue
                assessed = quality_policy(assessed_policy)
                if (
                    trust_rank(assessed.minimum_trust_level)
                    < trust_rank(selected_policy.minimum_trust_level)
                    or not acceptable
                ):
                    unassessed.append((sample.case_id, assessed_policy))
            if unassessed:
                raise RuntimeError(
                    f"Campaign {self.name!r} contains samples that did not "
                    f"pass quality policy {selected_policy.name!r}: "
                    f"{tuple(unassessed)}."
                )
            quality_decision = {
                "accepted": True,
                "policy": selected_policy.name,
                "minimum_trust_level": selected_policy.minimum_trust_level,
                "sample_count": len(self.dataset.samples),
            }
        if minimum_trust_level is not None:
            from ..verification import trust_rank

            required_rank = trust_rank(minimum_trust_level)
            below = []
            for sample in self.dataset.samples:
                summary = sample.provenance.get("simulation_result", {})
                level = (
                    summary.get("trust_level", "not_computed")
                    if isinstance(summary, Mapping)
                    else "not_computed"
                )
                if trust_rank(level) < required_rank:
                    below.append((sample.case_id, level))
            if below:
                raise RuntimeError(
                    f"Campaign {self.name!r} contains samples below required "
                    f"trust level {minimum_trust_level!r}: {tuple(below)}."
                )
        if not failed_ids and quality_decision is None:
            return self.dataset
        return ScientificDataset(
            parameter_space=self.dataset.parameter_space,
            quantities=self.dataset.quantities,
            samples=self.dataset.samples,
            name=self.dataset.name,
            metadata={
                **dict(self.dataset.metadata or {}),
                **(
                    {}
                    if not failed_ids
                    else {
                        "partial_campaign_acceptance": {
                            "accepted": True,
                            "failed_case_ids": failed_ids,
                            "failed_case_count": len(failed_ids),
                            "campaign": self.name,
                        }
                    }
                ),
                **(
                    {}
                    if quality_decision is None
                    else {"quality_acceptance": quality_decision}
                ),
            },
        )


class Campaign:
    """Build and evaluate a collection of immutable scientific cases.

    ``build(parameters)`` may construct an AgentFEM model, a model/step bundle,
    or any user-defined case object. ``evaluate(case)`` must return either a
    mapping of named outputs, :class:`CaseOutcome`, or
    :class:`agentfem.results.SimulationResult`.
    """

    def __init__(
        self,
        *,
        name: str,
        parameter_space: ParameterSpace,
        outputs: tuple[Quantity, ...],
        evaluate: Callable[
            [object],
            Mapping[str, object] | CaseOutcome | SimulationResult,
        ],
        build: Callable[[Mapping[str, object]], object] | None = None,
        metadata: Mapping[str, object] | None = None,
        scientific_inputs: Mapping[str, object] | None = None,
        execution: ExecutionPolicy | None = None,
    ) -> None:
        selected_name = str(name).strip()
        if not selected_name:
            raise ValueError("Campaign.name must not be empty.")
        quantities = tuple(outputs)
        if not quantities:
            raise ValueError("Campaign requires at least one output Quantity.")
        if len({quantity.name for quantity in quantities}) != len(quantities):
            raise ValueError("Campaign output names must be unique.")
        if not callable(evaluate):
            raise TypeError("Campaign.evaluate must be callable.")
        if build is not None and not callable(build):
            raise TypeError("Campaign.build must be callable or None.")
        self.name = selected_name
        self.parameter_space = parameter_space
        self.outputs = quantities
        self.evaluate = evaluate
        self.build = build
        self.metadata = dict(metadata or {})
        self.scientific_inputs = dict(scientific_inputs or {})
        self.execution = execution or ExecutionPolicy()

    def plan(self, sampling: SamplingPlan) -> CampaignPlan:
        """Bind one compatible sampling plan to deterministic case IDs."""

        if sampling.space.summary() != self.parameter_space.summary():
            raise ValueError("Sampling plan parameter space differs from the campaign.")
        cases = tuple(
            CampaignCase(
                case_id=case_id(self.name, sample),
                index=index,
                parameters=self.parameter_space.validate(sample),
            )
            for index, sample in enumerate(sampling.samples)
        )
        return CampaignPlan(
            name=self.name,
            parameter_space=self.parameter_space,
            sampling=sampling,
            cases=cases,
        )

    def run(
        self,
        plan: CampaignPlan | SamplingPlan,
        *,
        output_directory: str | Path | None = None,
        policy: ExecutionPolicy | None = None,
        comm=None,
    ) -> CampaignReport:
        """Execute or resume a campaign and materialize successful samples.

        When ``comm`` is supplied, every rank participates in each collective
        FEM case while rank zero alone persists manifests and artifacts. This
        is within-case MPI. Case-level distribution should use deterministic
        plan shards and separate jobs.
        """

        selected_plan = self.plan(plan) if isinstance(plan, SamplingPlan) else plan
        if selected_plan.name != self.name:
            raise ValueError("CampaignPlan belongs to a different campaign.")
        if (
            selected_plan.parameter_space.summary()
            != self.parameter_space.summary()
        ):
            raise ValueError("CampaignPlan parameter space differs from the campaign.")
        selected_policy = policy or self.execution
        output = None if output_directory is None else Path(output_directory)
        rank = getattr(comm, "rank", 0)
        if output is not None and rank == 0:
            (output / "cases").mkdir(parents=True, exist_ok=True)
            _write_json(output / "plan.json", selected_plan.summary())
        if comm is not None and hasattr(comm, "barrier"):
            comm.barrier()

        records_by_id: dict[str, CaseRunRecord] = {}
        pending_cases = []
        runtime = runtime_manifest() if rank == 0 else None
        campaign_inputs = (
            scientific_input_manifest(
                {
                    "declared": self.scientific_inputs,
                    "build": self.build,
                    "evaluate": self.evaluate,
                    "parameter_space": self.parameter_space.summary(),
                    "outputs": [quantity.summary() for quantity in self.outputs],
                },
                label=f"campaign:{self.name}",
            )
            if rank == 0
            else None
        )
        if comm is not None and hasattr(comm, "bcast"):
            runtime = comm.bcast(runtime, root=0)
            campaign_inputs = comm.bcast(campaign_inputs, root=0)
        assert runtime is not None
        assert campaign_inputs is not None
        for case in selected_plan.cases:
            record_path = None if output is None else output / "cases" / f"{case.case_id}.json"
            reused = (
                _load_completed_record(record_path, case)
                if selected_policy.resume and record_path is not None and rank == 0
                else None
            )
            if comm is not None and hasattr(comm, "bcast"):
                reused = comm.bcast(reused, root=0)
            if reused is not None:
                records_by_id[case.case_id] = reused
                continue
            pending_cases.append(case)

        if selected_policy.mode == "local_process":
            if comm is not None:
                raise ValueError(
                    "local_process executes independent cases and cannot be nested "
                    "inside one within-case MPI communicator. Use plan shards for "
                    "separate MPI jobs."
                )
            workers = selected_policy.workers or max(
                1,
                min(max(len(pending_cases), 1), os.cpu_count() or 1),
            )
            batch = run_local_processes(
                self._run_case,
                pending_cases,
                workers=workers,
                fail_fast=selected_policy.fail_fast,
            )
            execution_evidence = batch.evidence
            for record in batch.records:
                records_by_id[record.case.case_id] = record
        else:
            execution_evidence = {
                "provider": "within_case_mpi" if comm is not None else "serial",
                "start_method": None,
                "requested_workers": 1,
                "effective_workers": 1,
                "case_count": 0,
                "stopped_early": False,
                "fail_fast_overshoot_limit": 0,
            }
            for case in pending_cases:
                record = _synchronize_mpi_record(
                    self._run_case(
                        case,
                        "within_case_mpi" if comm is not None else "serial",
                    ),
                    comm=comm,
                    rank=rank,
                )
                records_by_id[case.case_id] = record
                execution_evidence["case_count"] += 1
                if not record.successful and selected_policy.fail_fast:
                    execution_evidence["stopped_early"] = True
                    break

        records = tuple(
            records_by_id[case.case_id]
            for case in selected_plan.cases
            if case.case_id in records_by_id
        )
        if output is not None and rank == 0:
            for record in records:
                if record.reused:
                    continue
                _write_json(
                    output / "cases" / f"{record.case.case_id}.json",
                    record.summary(),
                )

        samples = tuple(
            Sample(
                case_id=record.case.case_id,
                inputs=record.case.parameters,
                outputs=record.outcome.outputs,
                provenance=record.outcome.provenance,
                artifacts=record.outcome.artifacts,
            )
            for record in records
            if record.successful and record.outcome is not None
        )
        dataset = (
            None
            if not samples
            else ScientificDataset(
                parameter_space=self.parameter_space,
                quantities=self.outputs,
                samples=samples,
                name=f"{self.name}_dataset",
                metadata={
                    **self.metadata,
                    "campaign": self.name,
                    "sampling": selected_plan.sampling.summary(),
                    "build": _callable_identity(self.build),
                    "evaluate": _callable_identity(self.evaluate),
                    "runtime_fingerprint": runtime["fingerprint"],
                    "scientific_input_fingerprint": campaign_inputs["fingerprint"],
                    "scientific_input_coverage": {
                        "complete": campaign_inputs["complete"],
                        "missing": campaign_inputs["missing"],
                    },
                    "execution": execution_evidence,
                },
            )
        )
        report = CampaignReport(
            name=self.name,
            plan=selected_plan,
            records=tuple(records),
            dataset=dataset,
            output_directory=output,
            runtime=runtime,
            execution=execution_evidence,
            scientific_inputs=campaign_inputs,
        )
        if output is not None and rank == 0:
            if dataset is not None:
                dataset.write(output / "dataset")
            _write_json(output / "report.json", report.summary())
        if comm is not None and hasattr(comm, "barrier"):
            comm.barrier()
        return report

    def _run_case(
        self,
        case: CampaignCase,
        execution_mode: str = "serial",
    ) -> CaseRunRecord:
        started_at = _utc_now()
        started = perf_counter()
        try:
            built = (
                dict(case.parameters)
                if self.build is None
                else self.build(dict(case.parameters))
            )
            raw = self.evaluate(built)
            outcome = _as_case_outcome(
                raw,
                expected_names=tuple(quantity.name for quantity in self.outputs),
            )
            expected = {quantity.name for quantity in self.outputs}
            actual = set(outcome.outputs)
            if actual != expected:
                raise ValueError(
                    "Case output keys differ from the campaign contract; "
                    f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
                )
            validated_outputs = {
                quantity.name: _validated_quantity_value(
                    quantity,
                    outcome.outputs[quantity.name],
                )
                for quantity in self.outputs
            }
            provenance = {
                **dict(outcome.provenance),
                **_case_provenance(
                    built,
                    parameters=case.parameters,
                    declared=self.scientific_inputs,
                ),
            }
            selected_outcome = CaseOutcome(
                outputs=validated_outputs,
                provenance=provenance,
                artifacts=outcome.artifacts,
            )
            return CaseRunRecord(
                case=case,
                status="completed",
                started_at=started_at,
                finished_at=_utc_now(),
                duration_seconds=perf_counter() - started,
                outcome=selected_outcome,
                execution=_worker_identity(execution_mode),
            )
        except Exception as exc:  # case failures are campaign data, not process failure
            return CaseRunRecord(
                case=case,
                status="failed",
                started_at=started_at,
                finished_at=_utc_now(),
                duration_seconds=perf_counter() - started,
                error_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                error_message=str(exc),
                execution=_worker_identity(execution_mode),
            )

    def summary(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_SCHEMA,
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "name": self.name,
            "parameter_space": self.parameter_space.summary(),
            "outputs": [quantity.summary() for quantity in self.outputs],
            "build": _callable_identity(self.build),
            "evaluate": _callable_identity(self.evaluate),
            "metadata": self.metadata,
            "scientific_inputs": scientific_input_manifest(
                self.scientific_inputs,
                label=f"campaign:{self.name}:declared",
            ),
            "execution": self.execution.summary(),
        }


def create(**kwargs) -> Campaign:
    """Create a :class:`Campaign` using the public functional spelling."""

    return Campaign(**kwargs)


def local_processes(
    *,
    workers: int | None = None,
    fail_fast: bool = False,
    resume: bool = True,
) -> ExecutionPolicy:
    """Use spawned local processes for independent campaign cases."""

    return ExecutionPolicy(
        mode="local_process",
        workers=workers,
        fail_fast=fail_fast,
        resume=resume,
    )


def _as_case_outcome(
    raw: Mapping[str, object] | CaseOutcome | SimulationResult,
    *,
    expected_names: tuple[str, ...],
) -> CaseOutcome:
    """Normalize a campaign evaluator result without serializing live fields."""

    if isinstance(raw, CaseOutcome):
        return raw
    if isinstance(raw, SimulationResult):
        return CaseOutcome(
            outputs=raw.outputs(expected_names),
            provenance={
                "software_origin": dict(ORIGIN),
                "simulation_result": raw.summary(),
            },
            artifacts={
                name: str(path) for name, path in raw.artifacts.items()
            },
        )
    if isinstance(raw, Mapping):
        return CaseOutcome(outputs=raw)
    raise TypeError(
        "Campaign.evaluate must return a mapping, CaseOutcome, or "
        "SimulationResult."
    )


def _case_provenance(
    built,
    *,
    parameters: Mapping[str, object],
    declared: Mapping[str, object],
) -> dict[str, object]:
    input_manifest = scientific_input_manifest(
        {
            "parameters": parameters,
            "declared": declared,
            "built": built,
        },
        label="campaign_case",
    )
    provenance: dict[str, object] = {
        "scientific_inputs": input_manifest,
        "scientific_input_fingerprint": input_manifest["fingerprint"],
    }
    candidate = built
    if isinstance(built, Mapping):
        candidate = built.get("model", built)
    model = getattr(candidate, "model", candidate)
    to_ir = getattr(model, "to_ir", None)
    if callable(to_ir):
        try:
            model_ir = to_ir(
                metadata={"purpose": "campaign_case_provenance"}
            )
            provenance["model_ir"] = model_ir
            provenance["model_fingerprint"] = content_fingerprint(model_ir)
        except (TypeError, ValueError):
            provenance["model_ir_available"] = False
    return provenance


def _synchronize_mpi_record(
    local_record: CaseRunRecord,
    *,
    comm,
    rank: int,
) -> CaseRunRecord:
    if comm is None or not hasattr(comm, "bcast"):
        return local_record
    if hasattr(comm, "allgather"):
        local_status = {
            "rank": rank,
            "successful": local_record.successful,
            "error_type": local_record.error_type,
            "error_message": local_record.error_message,
        }
        statuses = comm.allgather(local_status)
        failures = [status for status in statuses if not status["successful"]]
    else:
        failures = []
    root_record = local_record if rank == 0 else None
    if failures and rank == 0:
        details = "; ".join(
            f"rank {item['rank']}: {item['error_type']}: {item['error_message']}"
            for item in failures
        )
        root_record = CaseRunRecord(
            case=local_record.case,
            status="failed",
            started_at=local_record.started_at,
            finished_at=_utc_now(),
            duration_seconds=local_record.duration_seconds,
            error_type="agentfem.campaigns.MPIRankFailure",
            error_message=f"One or more MPI ranks failed: {details}",
        )
    return comm.bcast(root_record, root=0)


def _validated_quantity_value(quantity: Quantity, value):
    selected = quantity.validate(value)
    return selected.item() if quantity.shape == () else selected


def _callable_identity(function) -> dict[str, object] | None:
    if function is None:
        return None
    result = {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", getattr(function, "__name__", None)),
    }
    try:
        source = inspect.getsource(function).encode("utf-8")
    except (OSError, TypeError):
        source = None
    if source is not None:
        result["source_sha256"] = sha256(source).hexdigest()
    return result


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            to_json_safe(record),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_completed_record(path: Path, case: CampaignCase) -> CaseRunRecord | None:
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        case_record = record["case"]
        if (
            record["status"] != "completed"
            or case_record["case_id"] != case.case_id
            or case_record["parameters"] != to_json_safe(case.parameters)
        ):
            return None
        outcome = record["outcome"]
        return CaseRunRecord(
            case=case,
            status="completed",
            started_at=record["started_at"],
            finished_at=record["finished_at"],
            duration_seconds=float(record["duration_seconds"]),
            outcome=CaseOutcome(
                outputs=outcome["outputs"],
                provenance=outcome.get("provenance", {}),
                artifacts=outcome.get("artifacts", {}),
            ),
            reused=True,
            execution=record.get("execution", {}),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _worker_identity(mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "hostname": socket.gethostname(),
        "process_id": os.getpid(),
        "process_name": multiprocessing.current_process().name,
        "start_method": multiprocessing.get_start_method(allow_none=True),
    }


__all__ = [
    "CAMPAIGN_SCHEMA",
    "CAMPAIGN_SCHEMA_VERSION",
    "Campaign",
    "CampaignCase",
    "CampaignPlan",
    "CampaignReport",
    "CaseOutcome",
    "CaseRunRecord",
    "ExecutionPolicy",
    "case_id",
    "create",
    "local_processes",
]
