"""Reusable field, history, diagnostic, and presentation output plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .finite_strain import (
    PeriodicCellHistoryRecorder,
    _aligned_increment_evidence,
    cauchy_stress_invariants,
    finite_strain_diagnostics,
    hill_mandel_periodic_path,
    homogenize_periodic_path,
    write_homogenized_csv,
    write_homogenized_history,
)
from .output import FieldOutput, FieldOutputArtifacts, field_output


@dataclass(frozen=True)
class OutputContext:
    """Completed analysis state supplied to output requests."""

    directory: Path
    basename: str
    model: object
    step: object
    result: object
    target: object
    material: object
    field_artifacts: FieldOutputArtifacts

    @property
    def comm(self):
        function = getattr(self.target, "value", self.target)
        return function.function_space.mesh.comm


@dataclass(frozen=True)
class SolverHistoryRequest:
    """Record accepted-increment convergence history."""

    name: str = "solver_history"

    def apply(self, context: OutputContext) -> None:
        info = getattr(context.step, "last_solve_info", None)
        increments = tuple(getattr(info, "increments", ()))
        if not increments:
            return
        factors = [item.load_factor for item in increments]
        context.result.add_histories(
            factors,
            {
                "newton_residual": [
                    item.residual_norm for item in increments
                ],
                "newton_iterations": [
                    item.iterations for item in increments
                ],
                "increment_size": [
                    item.load_factor - item.start_load_factor
                    for item in increments
                ],
            },
            abscissa_name="load_factor",
            abscissa_unit=None,
            descriptions={
                "newton_residual": "Reduced equilibrium residual at convergence.",
                "newton_iterations": "Newton iterations in the accepted increment.",
                "increment_size": "Accepted normalized load increment.",
            },
        )

    def summary(self) -> dict[str, object]:
        return {"kind": self.name}


@dataclass(frozen=True)
class HistoryRequest:
    """Evaluate one scientific quantity on every accepted output frame.

    ``evaluate`` receives ``(snapshot, output_context)``.  The request is
    intentionally agnostic to whether the value is a probe, integral,
    resultant, energy, or application-defined quantity.
    """

    name: str
    evaluate: object
    coordinate: object | None = None
    unit: str | None = None
    abscissa_name: str | None = None
    abscissa_unit: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("HistoryRequest.name must not be empty.")
        if not callable(self.evaluate):
            raise TypeError("HistoryRequest.evaluate must be callable.")
        if self.coordinate is not None and not callable(self.coordinate):
            raise TypeError("HistoryRequest.coordinate must be callable or None.")

    def apply(self, context: OutputContext) -> None:
        snapshots = tuple(getattr(context.step, "snapshots", ()))
        if not snapshots:
            raise ValueError(
                f"History request {self.name!r} requires accepted snapshots."
            )
        if self.coordinate is None:
            coordinate_name, coordinate_unit, abscissa = _snapshot_abscissa(
                snapshots
            )
        else:
            abscissa = [self.coordinate(snapshot, context) for snapshot in snapshots]
            coordinate_name = self.abscissa_name or "coordinate"
            coordinate_unit = self.abscissa_unit
        context.result.add_history(
            self.name,
            abscissa,
            [self.evaluate(snapshot, context) for snapshot in snapshots],
            unit=self.unit,
            abscissa_name=self.abscissa_name or coordinate_name,
            abscissa_unit=(
                self.abscissa_unit
                if self.abscissa_unit is not None
                else coordinate_unit
            ),
            description=self.description,
        )

    def evaluate_transient(self, step, time_value: float) -> float:
        """Evaluate this request after one accepted transient increment."""

        value = np.asarray(self.evaluate(step, float(time_value)))
        if value.size != 1:
            raise ValueError(
                f"Transient history {self.name!r} must evaluate to one scalar; "
                "request one component or define separate named histories."
            )
        selected = float(value.reshape(-1)[0])
        if not np.isfinite(selected):
            raise ValueError(f"Transient history {self.name!r} is not finite.")
        return selected

    def summary(self) -> dict[str, object]:
        return {
            "kind": "history_request",
            "name": self.name,
            "unit": self.unit,
            "abscissa_name": self.abscissa_name or "automatic",
            "description": self.description,
        }


@dataclass(frozen=True)
class ProbeHistoryRequest:
    """Record a field value at one physical point on every accepted frame."""

    name: str
    at: tuple[float, ...]
    field: object | None = None
    component: int | None = None
    unit: str | None = None
    description: str = ""

    def apply(self, context: OutputContext) -> None:
        from .quantities import probe

        def evaluate(snapshot, selected_context):
            if self.field is None:
                selected = getattr(snapshot, "solution")
            elif callable(self.field):
                selected = self.field(snapshot, selected_context)
            else:
                raise TypeError(
                    "A post-solve probe history requires field=callback; live "
                    "field objects are supported by transient online histories."
                )
            value = probe(selected, at=self.at)
            if self.component is not None:
                value = np.asarray(value)[int(self.component)]
            return value

        HistoryRequest(
            name=self.name,
            evaluate=evaluate,
            unit=self.unit,
            description=self.description,
        ).apply(context)

    def evaluate_transient(self, step, time_value: float) -> float:
        """Sample one live field after an accepted transient increment."""

        from .quantities import probe

        if self.field is None:
            selected = step.current if hasattr(step, "current") else step.state.u
        elif callable(self.field):
            selected = self.field(step, float(time_value))
        else:
            selected = self.field
        value = np.asarray(probe(selected, at=self.at))
        if self.component is not None:
            value = np.asarray(value[int(self.component)])
        if value.size != 1:
            raise ValueError(
                f"Transient probe history {self.name!r} is vector-valued; "
                "pass component=... or define separate named histories."
            )
        selected_value = float(value.reshape(-1)[0])
        if not np.isfinite(selected_value):
            raise ValueError(
                f"Transient probe history {self.name!r} is not finite."
            )
        return selected_value

    def summary(self) -> dict[str, object]:
        return {
            "kind": "probe_history",
            "name": self.name,
            "at": self.at,
            "component": self.component,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class PeriodicCellHistoryRequest:
    """Record complete tensor histories for a finite-strain periodic cell."""

    constraint: object
    basename: str = "homogenized_history"

    def bind(self, step, material, *, has_external_power: bool = False) -> None:
        """Attach an accepted-increment recorder before the solve begins."""

        if has_external_power:
            raise NotImplementedError(
                "periodic_cell_history Hill-Mandel evidence currently requires "
                "a quasistatic affine cell without body-force or natural-load "
                "power terms."
            )
        if not hasattr(step, "accepted_observers"):
            raise TypeError(
                "periodic_cell_history requires an affine nonlinear step; "
                "it cannot be attached to an ordinary boundary-condition path."
            )
        recorders = getattr(step, "accepted_history_recorders", None)
        if recorders is None:
            recorders = {}
            step.accepted_history_recorders = recorders
        if self.basename in recorders:
            raise ValueError(
                f"Periodic history basename {self.basename!r} is already bound."
            )
        recorder = PeriodicCellHistoryRecorder(material, self.constraint)
        recorders[self.basename] = recorder
        step.accepted_observers = (*tuple(step.accepted_observers), recorder)

    def apply(self, context: OutputContext) -> None:
        recorder = getattr(context.step, "accepted_history_recorders", {}).get(
            self.basename
        )
        if recorder is None:
            snapshots = tuple(context.step.snapshots)
            frames = homogenize_periodic_path(
                snapshots,
                context.material,
                constraint=self.constraint,
            )
            hill_mandel = hill_mandel_periodic_path(
                snapshots,
                context.material,
                constraint=self.constraint,
                frames=frames,
            )
            increment_info = tuple(
                getattr(snapshot, "solve_info", None) for snapshot in snapshots
            )
            history_source = "saved_spatial_frames"
        else:
            frames = tuple(recorder.frames)
            hill_mandel = tuple(recorder.hill_mandel)
            increment_info = tuple(recorder.increment_info)
            history_source = "every_accepted_increment"
        if not frames:
            raise RuntimeError("Periodic-cell accepted history is empty after solve.")
        stress_states = tuple(
            cauchy_stress_invariants(frame.cauchy_stress) for frame in frames
        )
        aligned_hill = (
            ((0.0, 0.0, 0.0, 0.0),)
            if len(frames) == 1
            else (
                (0.0, 0.0, 0.0, 0.0),
                *tuple(
                    (
                        item.microscopic_work_density,
                        item.macroscopic_work_density,
                        item.residual,
                        item.relative_error,
                    )
                    for item in hill_mandel
                ),
            )
        )
        aligned_convergence = _aligned_increment_evidence(
            frames,
            increment_info,
        )
        factors = [frame.load_factor for frame in frames]
        energy_components: dict[str, list[float]] = {}
        component_availability = tuple(
            (
                frame.elastic_energy_density is not None,
                frame.hardening_energy_density is not None,
            )
            for frame in frames
        )
        dissipation_availability = tuple(
            frame.plastic_dissipation_density is not None for frame in frames
        )
        if any(any(item) for item in component_availability):
            if not all(all(item) for item in component_availability):
                raise RuntimeError(
                    "Homogenized stored-energy components must be available "
                    "together on every accepted frame."
                )
            energy_components = {
                "homogenized_elastic_energy_density": [
                    float(frame.elastic_energy_density) for frame in frames
                ],
                "homogenized_hardening_energy_density": [
                    float(frame.hardening_energy_density) for frame in frames
                ],
            }
        if any(dissipation_availability):
            if not all(dissipation_availability):
                raise RuntimeError(
                    "Homogenized plastic dissipation must be available on every "
                    "accepted frame when the provider declares it."
                )
            dissipation_history = [
                float(frame.plastic_dissipation_density) for frame in frames
            ]
            scale = max((abs(value) for value in dissipation_history), default=1.0)
            tolerance = 1.0e-12 * max(scale, 1.0)
            if any(
                right < left - tolerance
                for left, right in zip(
                    dissipation_history[:-1], dissipation_history[1:]
                )
            ):
                raise RuntimeError(
                    "Homogenized cumulative plastic dissipation decreased "
                    "between accepted frames."
                )
            energy_components["homogenized_plastic_dissipation_density"] = (
                dissipation_history
            )
        context.result.add_histories(
            factors,
            {
                "homogenized_deformation_gradient": [
                    frame.deformation_gradient for frame in frames
                ],
                "homogenized_green_lagrange_strain": [
                    frame.green_lagrange_strain for frame in frames
                ],
                "homogenized_logarithmic_strain": [
                    frame.logarithmic_strain for frame in frames
                ],
                "homogenized_first_piola_stress": [
                    frame.first_piola_stress for frame in frames
                ],
                "homogenized_cauchy_stress": [
                    frame.cauchy_stress for frame in frames
                ],
                "homogenized_J": [
                    frame.deformation_jacobian for frame in frames
                ],
                "homogenized_strain_energy_density": [
                    frame.strain_energy_density for frame in frames
                ],
                **energy_components,
                "homogenized_mean_cauchy_stress": [
                    state.mean_stress for state in stress_states
                ],
                "homogenized_von_mises_cauchy_stress": [
                    state.von_mises_stress for state in stress_states
                ],
                "homogenized_stress_triaxiality": [
                    0.0 if state.triaxiality is None else state.triaxiality
                    for state in stress_states
                ],
                "homogenized_normalized_lode_parameter": [
                    (
                        0.0
                        if state.normalized_lode_parameter is None
                        else state.normalized_lode_parameter
                    )
                    for state in stress_states
                ],
                "homogenized_stress_state_defined": [
                    float(state.deviatoric_state_defined)
                    for state in stress_states
                ],
                "hill_mandel_microscopic_work_density": [
                    item[0] for item in aligned_hill
                ],
                "hill_mandel_macroscopic_work_density": [
                    item[1] for item in aligned_hill
                ],
                "hill_mandel_residual": [item[2] for item in aligned_hill],
                "hill_mandel_relative_error": [
                    item[3] for item in aligned_hill
                ],
                "accepted_increment_defined": [
                    float(item[0]) for item in aligned_convergence
                ],
                "accepted_increment_size": [
                    0.0 if not item[0] else item[1]
                    for item in aligned_convergence
                ],
                "accepted_newton_iterations": [
                    0.0 if not item[0] else item[2]
                    for item in aligned_convergence
                ],
                "accepted_residual_norm": [
                    0.0 if not item[0] else item[3]
                    for item in aligned_convergence
                ],
                "accepted_periodic_equation_mismatch": [
                    0.0 if not item[0] else item[4]
                    for item in aligned_convergence
                ],
                "accepted_attempt": [
                    0.0 if not item[0] else item[5]
                    for item in aligned_convergence
                ],
            },
            abscissa_name="load_factor",
            abscissa_unit=None,
            descriptions={
                "homogenized_stress_triaxiality": (
                    "Mean Cauchy stress divided by macroscopic von Mises stress; "
                    "consult homogenized_stress_state_defined."
                ),
                "homogenized_normalized_lode_parameter": (
                    "Normalized Lode parameter in [-1, 1]; consult "
                    "homogenized_stress_state_defined."
                ),
                "homogenized_stress_state_defined": (
                    "One when deviatoric macro stress defines triaxiality and "
                    "Lode state, zero otherwise."
                ),
                "hill_mandel_microscopic_work_density": (
                    "Trapezoidal microscopic first-Piola work over the accepted "
                    "increment, normalized by complete reference-cell volume."
                ),
                "hill_mandel_macroscopic_work_density": (
                    "Trapezoidal macroscopic first-Piola work over the accepted "
                    "increment."
                ),
                "hill_mandel_residual": (
                    "Microscopic minus macroscopic accepted-increment work density."
                ),
                "hill_mandel_relative_error": (
                    "Absolute Hill-Mandel residual normalized by the larger work "
                    "magnitude."
                ),
                "accepted_increment_defined": (
                    "One for an accepted nonlinear increment and zero for the "
                    "initial state or unavailable saved-frame evidence."
                ),
                "accepted_increment_size": (
                    "Accepted macroscopic load-factor increment; consult "
                    "accepted_increment_defined."
                ),
                "accepted_newton_iterations": (
                    "Newton iterations used by the accepted increment."
                ),
                "accepted_residual_norm": (
                    "Final nonlinear residual norm of the accepted increment."
                ),
                "accepted_periodic_equation_mismatch": (
                    "Maximum periodic affine equation mismatch after acceptance."
                ),
                "accepted_attempt": (
                    "Attempt number on which the increment was accepted; values "
                    "above one expose cutback or retry."
                ),
                "homogenized_elastic_energy_density": (
                    "Volume-averaged recoverable elastic stored-energy density."
                ),
                "homogenized_hardening_energy_density": (
                    "Volume-averaged isotropic-hardening stored-energy density."
                ),
                "homogenized_plastic_dissipation_density": (
                    "Volume-averaged cumulative irrecoverable plastic "
                    "dissipation density."
                ),
            },
        )
        final = frames[-1]
        context.result.add_quantities(
            {
                "homogenized_first_piola_stress": final.first_piola_stress,
                "homogenized_cauchy_stress": final.cauchy_stress,
                "homogenized_strain_energy_density": (
                    final.strain_energy_density
                ),
                "homogenized_stress_consistency_error": (
                    final.stress_consistency_error
                ),
                "solid_reference_fraction": final.solid_reference_fraction,
                "solid_current_fraction": final.solid_current_fraction,
                "homogenized_history_frame_count": len(frames),
                "maximum_hill_mandel_relative_error": max(
                    (item.relative_error for item in hill_mandel),
                    default=0.0,
                ),
                **(
                    {
                        "homogenized_elastic_energy_density": (
                            final.elastic_energy_density
                        ),
                        "homogenized_hardening_energy_density": (
                            final.hardening_energy_density
                        ),
                        **(
                            {
                                "homogenized_plastic_dissipation_density": (
                                    final.plastic_dissipation_density
                                )
                            }
                            if final.plastic_dissipation_density is not None
                            else {}
                        ),
                    }
                    if energy_components
                    else {}
                ),
            }
        )
        context.result.metadata["homogenized_history"] = {
            "source": history_source,
            "frame_count": len(frames),
            "spatial_output_frame_count": len(context.step.snapshots),
            "undefined_stress_state_encoding": (
                "zero placeholder with homogenized_stress_state_defined=0"
            ),
            "hill_mandel_scope": (
                "quasistatic finite strain; no body-force or inertia power"
            ),
            "accepted_increment_evidence": (
                "increment size, Newton iterations, residual, periodic mismatch, "
                "and accepted attempt aligned with every macro frame"
            ),
        }
        final_state = stress_states[-1]
        if final_state.deviatoric_state_defined:
            context.result.add_quantities(
                {
                    "homogenized_stress_triaxiality": final_state.triaxiality,
                    "homogenized_normalized_lode_parameter": (
                        final_state.normalized_lode_parameter
                    ),
                }
            )
        npz = context.directory / f"{self.basename}.npz"
        csv = context.directory / f"{self.basename}.csv"
        if context.comm.rank == 0:
            write_homogenized_history(
                npz,
                frames,
                hill_mandel=hill_mandel,
                increment_info=increment_info,
            )
            write_homogenized_csv(
                csv,
                frames,
                hill_mandel=hill_mandel,
                increment_info=increment_info,
            )
        context.comm.barrier()
        context.result.add_artifact("homogenized_history_npz", npz)
        context.result.add_artifact("homogenized_history_csv", csv)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "periodic_cell_history",
            "constraint": getattr(self.constraint, "name", type(self.constraint).__name__),
            "basename": self.basename,
        }


@dataclass(frozen=True)
class SourceNodeHistoryRequest:
    """Record U and current coordinates using source-mesh node labels."""

    nodes: object
    points: tuple[tuple[str, int], ...]

    def apply(self, context: OutputContext) -> None:
        from ..mesh.abaqus import displacement_in_source_order

        snapshots = tuple(context.step.snapshots)
        factors = [snapshot.load_factor for snapshot in snapshots]
        displacements = [
            displacement_in_source_order(snapshot.solution, self.nodes)
            for snapshot in snapshots
        ]
        for point_name, node_label in self.points:
            source_index = self.nodes.index(node_label)
            values = np.asarray(
                [frame[source_index] for frame in displacements]
            )
            coordinate = self.nodes.coordinate(node_label)
            key = point_name.lower()
            context.result.add_history(
                f"{key}_displacement",
                factors,
                values,
                abscissa_name="load_factor",
                abscissa_unit=None,
                description=f"Source node {node_label} displacement history.",
            )
            context.result.add_history(
                f"{key}_coordinate",
                factors,
                coordinate + values,
                abscissa_name="load_factor",
                abscissa_unit=None,
                description=f"Source node {node_label} current-coordinate history.",
            )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "source_node_history",
            "points": dict(self.points),
        }


@dataclass(frozen=True)
class FiniteStrainDiagnosticRequest:
    """Record physical admissibility and constraint checks."""

    constraint: object | None = None
    quadrature_degree: int = 4

    def __post_init__(self) -> None:
        if int(self.quadrature_degree) <= 0:
            raise ValueError("quadrature_degree must be positive.")

    def apply(self, context: OutputContext) -> None:
        context.result.add_quantities(
            finite_strain_diagnostics(
                context.target,
                constraint=self.constraint,
                quadrature_degree=int(self.quadrature_degree),
            ),
            kind="diagnostic",
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "finite_strain_diagnostics",
            "quadrature_degree": int(self.quadrature_degree),
            "periodic_constraint": self.constraint is not None,
        }


@dataclass(frozen=True)
class PresentationOutput:
    """Optional serial rendering from the scientific XDMF/HDF5 series."""

    comparison: bool = True
    animation: str | None = "gif"
    scalar: str = "UMAG"
    fps: int = 2

    def __post_init__(self) -> None:
        animation = (
            None
            if self.animation is None
            else str(self.animation).lower().lstrip(".")
        )
        if animation not in {None, "gif", "mp4"}:
            raise ValueError(
                "Presentation animation must be 'gif', 'mp4', or None."
            )
        if int(self.fps) <= 0:
            raise ValueError("Presentation fps must be positive.")
        object.__setattr__(self, "animation", animation)

    def apply(self, context: OutputContext) -> None:
        xdmf = context.field_artifacts.unified_xdmf
        if context.comm.size > 1 or xdmf is None:
            context.result.metadata["presentation"] = {
                "status": "deferred",
                "reason": (
                    "Render from the parallel scientific XDMF after the MPI run."
                ),
            }
            return
        from .visualization import (
            render_unified_xdmf_animation,
            render_unified_xdmf_comparison,
        )

        if self.comparison:
            path = render_unified_xdmf_comparison(
                xdmf,
                context.directory / f"{context.basename}_comparison.png",
                scalar=self.scalar,
            )
            context.result.add_artifact("deformation_comparison", path)
        if self.animation is not None:
            path = render_unified_xdmf_animation(
                xdmf,
                context.directory
                / f"{context.basename}_deformation.{self.animation}",
                scalar=self.scalar,
                fps=int(self.fps),
            )
            context.result.add_artifact("deformation_animation", path)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "presentation_output",
            "comparison": self.comparison,
            "animation": self.animation,
            "scalar": self.scalar,
            "fps": int(self.fps),
        }


@dataclass(frozen=True)
class OutputPlan:
    """One declarative output contract for a completed finite-strain step."""

    directory: Path
    field: FieldOutput
    requests: tuple[object, ...] = ()
    presentation: PresentationOutput | None = None
    basename: str = "results"
    write_model_ir: bool = True
    write_manifest: bool = True

    def __post_init__(self) -> None:
        directory = Path(self.directory)
        basename = str(self.basename).strip()
        if not basename:
            raise ValueError("OutputPlan.basename must not be empty.")
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "requests", tuple(self.requests))
        object.__setattr__(self, "basename", basename)

    @property
    def every(self):
        """Expose field cadence to the incremental step."""

        return self.field.every

    def required_factors(self) -> tuple[float, ...]:
        return self.field.required_factors()

    def bind(self, step, material, *, has_external_power: bool = False) -> None:
        """Bind requests that need accepted states before the solve."""

        for request in self.requests:
            binder = getattr(request, "bind", None)
            if binder is not None:
                binder(
                    step,
                    material,
                    has_external_power=bool(has_external_power),
                )

    def finalize(
        self,
        *,
        model,
        step,
        result,
        target,
        material,
        metadata: Mapping[str, object] | None = None,
    ):
        """Write requested products and enrich ``result`` in one operation."""

        identity = self.identity()
        existing = result.metadata.get("output_plan")
        if (
            isinstance(existing, Mapping)
            and existing.get("status") == "completed"
            and existing.get("identity") == identity
        ):
            return result

        function = getattr(target, "value", target)
        domain = function.function_space.mesh
        comm = domain.comm
        if comm.rank == 0:
            self.directory.mkdir(parents=True, exist_ok=True)
        comm.barrier()
        artifacts = self.field.write_finite_strain(
            self.directory,
            domain=domain,
            snapshots=step.snapshots,
            material=material,
            basename=self.basename,
        )
        for field in artifacts.final_fields:
            selected_name = field.name
            existing_field = result.fields.get(selected_name)
            if (
                existing_field is not None
                and existing_field.location == "quadrature_points"
            ):
                recovered_name = f"{selected_name}_CELL"
                if recovered_name in result.fields:
                    # A stateful constitutive transaction has already attached
                    # both the raw quadrature field and its declared cell
                    # recovery. Do not let a visualization artifact silently
                    # replace the raw scientific field's location semantics.
                    continue
                selected_name = recovered_name
            result.add_field(
                selected_name,
                field,
                location="cells",
                description="P0 finite-strain visualization field.",
            )
        _register_field_artifacts(result, artifacts)
        context = OutputContext(
            directory=self.directory,
            basename=self.basename,
            model=model,
            step=step,
            result=result,
            target=target,
            material=material,
            field_artifacts=artifacts,
        )
        for request in self.requests:
            request.apply(context)
        if self.presentation is not None:
            self.presentation.apply(context)
        result.metadata["output_plan"] = {
            **self.summary(step=step, artifacts=artifacts),
            "status": "completed",
            "identity": identity,
        }
        if metadata:
            result.metadata.update(dict(metadata))

        ir_path = self.directory / f"{self.basename}.afir.json"
        manifest_path = self.directory / f"{self.basename}.result.json"
        if self.write_model_ir:
            result.add_artifact("model_ir", ir_path)
            model.write_ir(ir_path)
        if self.write_manifest:
            result.add_artifact("result_manifest", manifest_path)
            if comm.rank == 0:
                result.write_manifest(manifest_path, include_histories=True)
        comm.barrier()
        return result

    def identity(self) -> str:
        """Return a deterministic identity used for idempotent finalization."""

        return f"{self.directory.resolve()}::{self.basename}"

    def summary(
        self,
        *,
        step=None,
        artifacts: FieldOutputArtifacts | None = None,
    ) -> dict[str, object]:
        return {
            "kind": "output_plan",
            "basename": self.basename,
            "field": self.field.summary(),
            "requests": [
                request.summary() for request in self.requests
            ],
            "presentation": (
                None if self.presentation is None else self.presentation.summary()
            ),
            "frame_count": (
                None if step is None else len(getattr(step, "snapshots", ()))
            ),
            "saved_load_factors": (
                None
                if step is None
                else [
                    snapshot.load_factor
                    for snapshot in getattr(step, "snapshots", ())
                ]
            ),
            "parallel_scientific_output": (
                artifacts is not None and artifacts.reference_xdmf is not None
            ),
        }


def output_plan(
    directory,
    *,
    field: FieldOutput | None = None,
    requests=(),
    presentation: PresentationOutput | None = None,
    basename: str = "results",
) -> OutputPlan:
    """Create a complete finite-strain output plan."""

    return OutputPlan(
        directory=Path(directory),
        field=field_output() if field is None else field,
        requests=tuple(requests),
        presentation=presentation,
        basename=basename,
    )


def solver_history() -> SolverHistoryRequest:
    return SolverHistoryRequest()


def history(
    name: str,
    evaluate,
    *,
    coordinate=None,
    unit: str | None = None,
    abscissa_name: str | None = None,
    abscissa_unit: str | None = None,
    description: str = "",
) -> HistoryRequest:
    """Create a scalar history evaluated on accepted analysis states.

    Finite-strain output plans call ``evaluate(snapshot, context)`` after the
    solve. Transient steps call ``evaluate(step, physical_time)`` immediately
    after every accepted increment.
    """

    return HistoryRequest(
        name=name,
        evaluate=evaluate,
        coordinate=coordinate,
        unit=unit,
        abscissa_name=abscissa_name,
        abscissa_unit=abscissa_unit,
        description=description,
    )


def probe_history(
    name: str,
    *,
    at,
    field=None,
    component: int | None = None,
    unit: str | None = None,
    description: str = "",
) -> ProbeHistoryRequest:
    """Create a point-probe history for accepted static or transient states."""

    point = tuple(float(value) for value in np.asarray(at).reshape(-1))
    if not point or not np.all(np.isfinite(point)):
        raise ValueError("probe_history at= must contain finite coordinates.")
    return ProbeHistoryRequest(
        name=str(name),
        at=point,
        field=field,
        component=component,
        unit=unit,
        description=description,
    )


def _snapshot_abscissa(snapshots):
    """Infer a physical time or normalized load coordinate without guessing."""

    if all(hasattr(snapshot, "time") for snapshot in snapshots):
        return "time", "s", [float(snapshot.time) for snapshot in snapshots]
    if all(hasattr(snapshot, "load_factor") for snapshot in snapshots):
        return (
            "load_factor",
            None,
            [float(snapshot.load_factor) for snapshot in snapshots],
        )
    raise ValueError(
        "Snapshots expose neither a common time nor load_factor; pass coordinate=."
    )


def periodic_cell_history(
    constraint,
    *,
    basename: str = "homogenized_history",
) -> PeriodicCellHistoryRequest:
    return PeriodicCellHistoryRequest(constraint, basename)


def source_node_history(nodes, **points: int) -> SourceNodeHistoryRequest:
    if not points:
        raise ValueError("source_node_history requires at least one named node.")
    return SourceNodeHistoryRequest(
        nodes=nodes,
        points=tuple((str(name), int(label)) for name, label in points.items()),
    )


def finite_strain_checks(
    *,
    constraint=None,
    quadrature_degree: int = 4,
) -> FiniteStrainDiagnosticRequest:
    return FiniteStrainDiagnosticRequest(constraint, quadrature_degree)


def presentation(
    *,
    comparison: bool = True,
    animation: str | None = "gif",
    scalar: str = "UMAG",
    fps: int = 2,
) -> PresentationOutput:
    return PresentationOutput(comparison, animation, scalar, fps)


def _register_field_artifacts(result, artifacts: FieldOutputArtifacts) -> None:
    if artifacts.reference_xdmf is not None:
        result.add_artifact("scientific_field_history", artifacts.reference_xdmf)
        if artifacts.deformed_pvd is None:
            result.add_artifact("field_history", artifacts.reference_xdmf)
    if artifacts.unified_xdmf is not None:
        result.add_artifact("field_history", artifacts.unified_xdmf)
    if artifacts.deformed_pvd is not None:
        result.add_artifact("field_history", artifacts.deformed_pvd)
        result.add_artifact("fields_paraview", artifacts.deformed_pvd)
        result.add_artifact("deformed_field_history", artifacts.deformed_pvd)
