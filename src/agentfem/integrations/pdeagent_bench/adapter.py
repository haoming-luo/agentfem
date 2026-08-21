"""Lower PDEAgent-Bench's public case view to AgentFEM operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from time import perf_counter

import numpy as np
import ufl
from dolfinx import fem, mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import expressions, fields, mesh, results, solvers

from .schema import BENCHMARK_COMMIT, BENCHMARK_SCHEMA, BenchmarkContractError, validate_case_spec


@dataclass(frozen=True)
class BenchmarkPolicy:
    """One public numerical policy, independent of benchmark case identity."""

    planar_resolution: int = 32
    spatial_resolution: int = 14
    planar_degree: int = 3
    spatial_degree: int = 2
    relative_tolerance: float = 1.0e-10
    absolute_tolerance: float = 1.0e-12
    ksp_type: str = "cg"
    pc_type: str = "hypre"

    def degree(self, dimension: int) -> int:
        return int(self.spatial_degree if dimension == 3 else self.planar_degree)

    def resolution(
        self,
        dimension: int,
        domain_spec: Mapping[str, object],
        pde_spec: Mapping[str, object] | None = None,
    ) -> int:
        """Choose resolution from exposed geometry and expression bandwidth."""

        frequency = _expression_frequency(pde_spec or {})
        if dimension == 3:
            return max(int(self.spatial_resolution), 4 * frequency)
        char_length = domain_spec.get("char_length")
        if char_length is not None:
            geometric = max(2, int(round(1.0 / float(char_length))))
        else:
            geometric = int(self.planar_resolution)
        return max(geometric, min(32, 6 * frequency))

    def linear_options(self, *, indefinite: bool = False) -> solvers.LinearSolverOptions:
        return solvers.LinearSolverOptions(
            ksp_type="preonly" if indefinite else self.ksp_type,
            pc_type="lu" if indefinite else self.pc_type,
            rtol=None if indefinite else self.relative_tolerance,
            atol=None if indefinite else self.absolute_tolerance,
            error_if_not_converged=True,
        )


@dataclass(frozen=True)
class BenchmarkSolveResult:
    """Validated solution array plus explicit execution evidence."""

    values: np.ndarray
    solver_info: dict[str, object]
    initial_values: np.ndarray | None = None

    def as_benchmark_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "u": self.values,
            "solver_info": self.solver_info,
        }
        if self.initial_values is not None:
            payload["u_initial"] = self.initial_values
        return payload


def solve_case(
    case_spec: Mapping[str, object],
    policy: BenchmarkPolicy | None = None,
) -> dict[str, object]:
    """Solve one audited PDEAgent-Bench agent-view case.

    The adapter consumes only fields exposed to a participating agent.  It
    never reads oracle mesh, finite-element order, solver settings, or a
    manufactured solution.
    """

    selected = policy or BenchmarkPolicy()
    case = validate_case_spec(case_spec)
    started = perf_counter()
    dimension = int(case["_agentfem"]["dimension"])
    family = str(case["_agentfem"]["pde_family"])
    resolution = selected.resolution(dimension, case["domain"], case["pde"])
    degree = selected.degree(dimension)
    imported = mesh.from_spec(case["domain"], resolution=resolution)
    domain = imported.domain

    try:
        if family == "poisson":
            solved, info = _solve_poisson(case, domain, selected, degree)
            initial = None
        elif family == "heat":
            solved, info, initial = _solve_heat(case, domain, selected, degree)
        elif family == "linear_elasticity":
            solved, info = _solve_elasticity(case, domain, selected, degree)
            initial = None
        elif family == "helmholtz":
            solved, info = _solve_helmholtz(case, domain, selected, degree)
            initial = None
        else:  # validation owns the public failure, this protects refactors
            raise BenchmarkContractError("AFM-PDEB-008", f"unsupported PDE {family}")
        grid = _sample(case, solved, vector=(family == "linear_elasticity"))
        initial_grid = None if initial is None else _sample(case, initial, vector=False)
        _require_output_coverage(case, grid)
    except BenchmarkContractError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"AFM-PDEB-009: {family} solve failed for "
            f"{case.get('id', '<unnamed>')}: {type(exc).__name__}: {exc}"
        ) from exc

    solver_info = {
        "mesh_resolution": resolution,
        "element_degree": degree,
        "ksp_type": info.get("ksp_type", selected.ksp_type),
        "pc_type": info.get("pc_type", selected.pc_type),
        "rtol": selected.relative_tolerance,
        "agentfem_adapter_schema": BENCHMARK_SCHEMA,
        "benchmark_commit": BENCHMARK_COMMIT,
        "pde_family": family,
        "dimension": dimension,
        "num_dofs": int(solved.function_space.dofmap.index_map.size_global),
        "coverage": float(np.mean(np.isfinite(grid))),
        "solve_wall_time_sec": float(perf_counter() - started),
        **info,
    }
    return BenchmarkSolveResult(grid, solver_info, initial_grid).as_benchmark_dict()


solve = solve_case


def _solve_poisson(case, domain, policy, degree):
    unknown = fields.scalar_unknown(domain, name="u", degree=degree)
    pde = case["pde"]
    kappa = _coefficient(pde, "kappa", unknown.space, default=1.0)
    source = _known_field(pde.get("source_term", 0.0), unknown.space)
    bcs = _dirichlet_bcs(case, unknown.value)
    a = ufl.inner(kappa * ufl.grad(unknown.trial), ufl.grad(unknown.test)) * ufl.dx
    L = source * unknown.test * ufl.dx
    _, info = solvers.solve_linear_problem(
        fem.form(a), fem.form(L), unknown.value, bcs=bcs, options=policy.linear_options(), return_info=True
    )
    return unknown.value, _linear_info(info, policy)


def _solve_helmholtz(case, domain, policy, degree):
    unknown = fields.scalar_unknown(domain, name="u", degree=degree)
    pde = case["pde"]
    params = pde.get("pde_params", {})
    wave_number = float(params.get("k", params.get("wave_number", 10.0)))
    source = _known_field(pde.get("source_term", 0.0), unknown.space)
    bcs = _dirichlet_bcs(case, unknown.value)
    a = (
        ufl.inner(ufl.grad(unknown.trial), ufl.grad(unknown.test))
        - wave_number**2 * unknown.trial * unknown.test
    ) * ufl.dx
    L = source * unknown.test * ufl.dx
    _, info = solvers.solve_linear_problem(
        fem.form(a),
        fem.form(L),
        unknown.value,
        bcs=bcs,
        options=policy.linear_options(indefinite=True),
        return_info=True,
    )
    payload = _linear_info(info, policy, indefinite=True)
    payload["wave_number"] = wave_number
    return unknown.value, payload


def _solve_elasticity(case, domain, policy, degree):
    dimension = int(domain.geometry.dim)
    unknown = fields.vector_unknown(
        domain, name="u", degree=degree, dim=dimension
    )
    pde = case["pde"]
    params = pde.get("pde_params", {})
    if "lambda" in params and "mu" in params:
        lame_lambda, shear_modulus = float(params["lambda"]), float(params["mu"])
    else:
        young = float(params.get("E", 1.0))
        poisson = float(params.get("nu", 0.3))
        shear_modulus = young / (2.0 * (1.0 + poisson))
        lame_lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    source_spec = pde.get("source_term", [0.0] * dimension)
    if not isinstance(source_spec, Sequence) or isinstance(source_spec, (str, bytes)):
        source_spec = [source_spec] * dimension
    source = _known_field(source_spec, unknown.space)
    bcs = _dirichlet_bcs(case, unknown.value)

    def strain(value):
        return ufl.sym(ufl.grad(value))

    def stress(value):
        eps = strain(value)
        return (
            2.0 * shear_modulus * eps
            + lame_lambda * ufl.tr(eps) * ufl.Identity(dimension)
        )

    a = ufl.inner(stress(unknown.trial), strain(unknown.test)) * ufl.dx
    L = ufl.inner(source, unknown.test) * ufl.dx
    _, info = solvers.solve_linear_problem(
        fem.form(a), fem.form(L), unknown.value, bcs=bcs, options=policy.linear_options(), return_info=True
    )
    payload = _linear_info(info, policy)
    payload.update({"lame_lambda": lame_lambda, "shear_modulus": shear_modulus})
    return unknown.value, payload


def _solve_heat(case, domain, policy, degree):
    unknown = fields.scalar_unknown(domain, name="u", degree=degree)
    previous = fem.Function(unknown.space, name="u_previous")
    pde = case["pde"]
    time_cfg = pde["time"]
    t0 = float(time_cfg.get("t0", 0.0))
    t_end = float(time_cfg["t_end"])
    nominal_dt = float(time_cfg.get("dt", 0.01))
    steps = int(np.ceil((t_end - t0) / nominal_dt - 1.0e-14))
    if steps < 1:
        raise BenchmarkContractError("AFM-PDEB-002", "heat analysis needs at least one step")
    dt = (t_end - t0) / steps
    time = fem.Constant(domain, t0)
    initial = pde.get("initial_condition", 0.0)
    expressions.interpolate(previous, initial, parameters={"t": time})
    initial_field = fem.Function(unknown.space, name="u_initial")
    initial_field.x.array[:] = previous.x.array
    initial_field.x.scatter_forward()
    kappa = _coefficient(
        pde, "kappa", unknown.space, default=1.0, parameters={"t": time}
    )
    source_spec = pde.get("source_term", 0.0)
    source = _known_field(source_spec, unknown.space, parameters={"t": time})
    bcs, boundary_values = _dirichlet_bcs(
        case, unknown.value, parameters={"t": time}, track_values=True
    )
    a = (
        unknown.trial * unknown.test
        + dt * kappa * ufl.inner(ufl.grad(unknown.trial), ufl.grad(unknown.test))
    ) * ufl.dx
    L = (previous * unknown.test + dt * source * unknown.test) * ufl.dx
    with solvers.prepare_linear_problem(
        a, L, unknown.value, bcs=bcs, options=policy.linear_options()
    ) as problem:
        for step in range(1, steps + 1):
            time.value = t0 + step * dt
            expressions.interpolate(source, source_spec, parameters={"t": time})
            _refresh_dirichlet_values(boundary_values, parameters={"t": time})
            problem.solve()
            previous.x.array[:] = unknown.value.x.array
            previous.x.scatter_forward()
        last_info = problem.last_solve_info
    if last_info is None:
        raise RuntimeError("Heat problem did not execute a time step.")
    payload = _linear_info(last_info, policy)
    payload.update({"num_timesteps": steps, "dt": dt, "matrix_reused": True})
    return unknown.value, payload, initial_field


def _coefficient(pde, name, space, *, default, parameters=None):
    domain = space.mesh
    spec = pde.get("coefficients", {}).get(name, {"type": "constant", "value": default})
    if isinstance(spec, (int, float, str)):
        try:
            return fem.Constant(domain, float(spec))
        except (TypeError, ValueError):
            return _known_field(spec, space, parameters=parameters)
    kind = str(spec.get("type", "constant")).lower()
    if kind == "constant":
        return fem.Constant(domain, float(spec.get("value", default)))
    if kind == "expr":
        return _known_field(spec.get("expr", default), space, parameters=parameters)
    raise BenchmarkContractError("AFM-PDEB-003", f"unknown coefficient type {kind!r}")


def _known_field(source, space, *, parameters=None):
    """Interpolate known scientific data without specializing the weak form.

    Formula changes update field values while the compiled operator structure
    remains reusable across cases and parameter campaigns.
    """

    value = fem.Function(space)
    try:
        expressions.interpolate(value, source, parameters=parameters)
    except Exception as exc:
        raise BenchmarkContractError(
            "AFM-PDEB-003", f"could not interpolate scientific expression {source!r}: {exc}"
        ) from exc
    return value


def _dirichlet_bcs(case, function, *, parameters=None, track_values: bool = False):
    raw = case.get("bc", {}).get("dirichlet")
    if raw is None:
        raise BenchmarkContractError("AFM-PDEB-005", "Dirichlet boundary data are required")
    entries = raw if isinstance(raw, list) else [raw]
    bcs = []
    tracked = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise BenchmarkContractError("AFM-PDEB-005", "Dirichlet entry must be a mapping")
        dofs = _boundary_dofs(function.function_space, str(entry.get("on", "all")))
        value = fem.Function(function.function_space)
        expressions.interpolate(value, entry.get("value", 0.0), parameters=parameters)
        bcs.append(fem.dirichletbc(value, dofs))
        tracked.append((value, entry.get("value", 0.0)))
    return (bcs, tracked) if track_values else bcs


def _refresh_dirichlet_values(tracked, *, parameters=None):
    for value, source in tracked:
        expressions.interpolate(value, source, parameters=parameters)


def _boundary_dofs(space, selector: str):
    domain = space.mesh
    fdim = domain.topology.dim - 1
    key = selector.strip().lower()
    if key in {"all", "*"}:
        facets = dolfinx_mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        return fem.locate_dofs_topological(space, fdim, facets)
    axis_and_value = {
        "x0": (0, 0.0), "xmin": (0, 0.0), "x1": (0, 1.0), "xmax": (0, 1.0),
        "y0": (1, 0.0), "ymin": (1, 0.0), "y1": (1, 1.0), "ymax": (1, 1.0),
        "z0": (2, 0.0), "zmin": (2, 0.0), "z1": (2, 1.0), "zmax": (2, 1.0),
    }
    if key not in axis_and_value or axis_and_value[key][0] >= domain.geometry.dim:
        raise BenchmarkContractError("AFM-PDEB-005", f"unknown boundary selector {selector!r}")
    axis, value = axis_and_value[key]
    return fem.locate_dofs_geometrical(space, lambda x: np.isclose(x[axis], value))


def _sample(case, function, *, vector: bool):
    grid = case["output"]["grid"]
    shape = [int(grid["nx"]), int(grid["ny"])]
    if "nz" in grid:
        shape.append(int(grid["nz"]))
    sample = results.sample_rectilinear_grid(
        function,
        bbox=grid["bbox"],
        shape=shape,
        reduction="magnitude" if vector else None,
        padding=1.0e-9,
    )
    return np.asarray(sample.values, dtype=float)


def _require_output_coverage(case, values):
    finite = np.isfinite(values)
    if not np.any(finite):
        raise BenchmarkContractError("AFM-PDEB-007", "output has no finite in-domain values")
    domain_type = str(case["domain"].get("type", ""))
    if domain_type in {"unit_square", "unit_cube", "periodic_square"} and not np.all(finite):
        missing = int(finite.size - np.count_nonzero(finite))
        raise BenchmarkContractError(
            "AFM-PDEB-007", f"structured-domain output contains {missing} missing samples"
        )


def _linear_info(info, policy, *, indefinite=False):
    return {
        "ksp_type": "preonly" if indefinite else policy.ksp_type,
        "pc_type": "lu" if indefinite else policy.pc_type,
        "converged": bool(info.converged),
        "converged_reason": int(info.converged_reason),
        "iterations": int(info.iterations),
        "residual_norm": float(info.residual_norm),
    }


def _expression_frequency(pde_spec: Mapping[str, object]) -> int:
    """Estimate trigonometric bandwidth from public expression strings.

    This is a uniform numerical heuristic, not a case-id table.  Expressions
    such as ``sin(4*pi*x)`` request proportionally more cells per wavelength.
    """

    sources: list[str] = []

    def collect(value):
        if isinstance(value, str):
            sources.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                collect(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                collect(item)

    collect(pde_spec.get("source_term"))
    collect(pde_spec.get("initial_condition"))
    collect(pde_spec.get("coefficients"))
    frequency = 1.0
    phase_pattern = re.compile(
        r"(?:sin|cos|tan)\s*\(\s*(?:(\d+(?:\.\d+)?)\s*\*\s*)?pi\s*\*"
    )
    for source in sources:
        for match in phase_pattern.finditer(source):
            frequency = max(frequency, float(match.group(1) or 1.0))
    return max(1, int(np.ceil(frequency)))
