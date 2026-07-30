"""Traceable, resumable campaign orchestration.

The first implementation deliberately executes cases serially.  A campaign
plan can be deterministically sharded for MPI jobs, schedulers, or services,
but AgentFEM does not pretend that Python threads are a safe parallel FEM
executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping

from ..datasets import Quantity, Sample, ScientificDataset
from ..ir.schema import to_json_safe
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

    def __post_init__(self) -> None:
        mode = self.mode.lower().replace("-", "_")
        if mode != "serial":
            raise ValueError(
                "AgentFEM 0.1 campaigns currently execute mode='serial'. "
                "Use CampaignPlan.shard(...) to distribute independent plans "
                "through an external MPI launcher, scheduler, or service."
            )
        object.__setattr__(self, "mode", mode)

    def summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "fail_fast": self.fail_fast,
            "resume": self.resume,
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
            "records": [record.summary() for record in self.records],
        }


class Campaign:
    """Build and evaluate a collection of immutable scientific cases.

    ``build(parameters)`` may construct an AgentFEM model, a model/step bundle,
    or any user-defined case object. ``evaluate(case)`` must return either a
    mapping of named outputs or :class:`CaseOutcome`.
    """

    def __init__(
        self,
        *,
        name: str,
        parameter_space: ParameterSpace,
        outputs: tuple[Quantity, ...],
        evaluate: Callable[[object], Mapping[str, object] | CaseOutcome],
        build: Callable[[Mapping[str, object]], object] | None = None,
        metadata: Mapping[str, object] | None = None,
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
        selected_policy = policy or ExecutionPolicy()
        output = None if output_directory is None else Path(output_directory)
        rank = getattr(comm, "rank", 0)
        if output is not None and rank == 0:
            (output / "cases").mkdir(parents=True, exist_ok=True)
            _write_json(output / "plan.json", selected_plan.summary())
        if comm is not None and hasattr(comm, "barrier"):
            comm.barrier()

        records = []
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
                records.append(reused)
                continue
            record = _synchronize_mpi_record(
                self._run_case(case),
                comm=comm,
                rank=rank,
            )
            records.append(record)
            if record_path is not None and rank == 0:
                _write_json(record_path, record.summary())
            if not record.successful and selected_policy.fail_fast:
                break

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
                },
            )
        )
        report = CampaignReport(
            name=self.name,
            plan=selected_plan,
            records=tuple(records),
            dataset=dataset,
            output_directory=output,
        )
        if output is not None and rank == 0:
            if dataset is not None:
                dataset.write(output / "dataset")
            _write_json(output / "report.json", report.summary())
        if comm is not None and hasattr(comm, "barrier"):
            comm.barrier()
        return report

    def _run_case(self, case: CampaignCase) -> CaseRunRecord:
        started_at = _utc_now()
        started = perf_counter()
        try:
            built = (
                dict(case.parameters)
                if self.build is None
                else self.build(dict(case.parameters))
            )
            raw = self.evaluate(built)
            outcome = raw if isinstance(raw, CaseOutcome) else CaseOutcome(outputs=raw)
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
                **_case_provenance(built),
                **dict(outcome.provenance),
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
        }


def create(**kwargs) -> Campaign:
    """Create a :class:`Campaign` using the public functional spelling."""

    return Campaign(**kwargs)


def _case_provenance(built) -> dict[str, object]:
    provenance: dict[str, object] = {}
    candidate = built
    if isinstance(built, Mapping):
        candidate = built.get("model", built)
    model = getattr(candidate, "model", candidate)
    to_ir = getattr(model, "to_ir", None)
    if callable(to_ir):
        try:
            provenance["model_ir"] = to_ir(
                metadata={"purpose": "campaign_case_provenance"}
            )
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
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


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
]
