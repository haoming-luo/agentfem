"""Lower PDEAgent-Bench's public case view to AgentFEM operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from time import perf_counter

import numpy as np
import ufl
from dolfinx import fem, mesh as dolfinx_mesh
from dolfinx.fem.petsc import LinearProblem, create_vector
from mpi4py import MPI
from petsc4py import PETSc

from agentfem import expressions, fields, mesh, operators, results, solvers, spaces

from .schema import (
    BENCHMARK_COMMIT,
    BENCHMARK_SCHEMA,
    BenchmarkContractError,
    validate_case_spec,
)


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

    def flow_resolution(
        self,
        dimension: int,
        domain_spec: Mapping[str, object],
        pde_spec: Mapping[str, object],
        boundary_spec: Mapping[str, object],
        output_spec: Mapping[str, object],
    ) -> int:
        """Choose a public geometry- and bandwidth-aware flow resolution."""

        frequency = _expression_frequency(pde_spec)
        if dimension == 3:
            # Three-dimensional Taylor--Hood fields need enough cells to
            # resolve both velocity curvature and the pressure constraint.
            # Keep this policy independent of benchmark case identity.
            return max(4, 2 * frequency)
        base = self.resolution(dimension, domain_spec, pde_spec)
        domain_type = str(domain_spec.get("type", ""))
        grid = output_spec.get("grid", {})
        samples = max(int(grid.get("nx", 0)), int(grid.get("ny", 0)))
        requested = (
            int(np.ceil(0.64 * samples))
            if domain_type in {"unit_square", "periodic_square"}
            else int(np.ceil((2.0 / 3.0) * samples))
        )
        cap = 72 if domain_type in {"unit_square", "periodic_square"} else 112
        return max(base, min(cap, requested))

    def linear_options(
        self, *, indefinite: bool = False
    ) -> solvers.LinearSolverOptions:
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
    resolution = (
        selected.flow_resolution(
            dimension,
            case["domain"],
            case["pde"],
            case["bc"],
            case["output"],
        )
        if family in {"stokes", "navier_stokes"}
        else selected.resolution(dimension, case["domain"], case["pde"])
    )
    degree = selected.degree(dimension)
    if (
        family in {"stokes", "navier_stokes"}
        and dimension == 3
        and str(case["domain"].get("type", "")) == "unit_cube"
    ):
        imported = mesh.FEMMesh(
            mesh.cuboid(
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                (resolution,) * 3,
                cell_type="hexahedron",
            )
        )
    else:
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
        elif family == "stokes":
            solved, info = _solve_stokes(case, domain, selected, degree)
            initial = None
        elif family == "navier_stokes":
            solved, info = _solve_navier_stokes(case, domain, selected, degree)
            initial = None
        elif family == "helmholtz":
            solved, info = _solve_helmholtz(case, domain, selected, degree)
            initial = None
        elif family == "convection_diffusion":
            solved, info, initial = _solve_convection_diffusion(
                case, domain, selected, degree
            )
        elif family == "reaction_diffusion":
            solved, info, initial = _solve_reaction_diffusion(
                case, domain, selected, degree
            )
        elif family == "burgers":
            solved, info, initial = _solve_burgers(case, domain, selected, degree)
        elif family == "biharmonic":
            solved, info = _solve_biharmonic(case, domain, selected, degree)
            initial = None
        elif family == "wave":
            solved, info, initial = _solve_wave(case, domain, selected, degree)
        else:  # validation owns the public failure, this protects refactors
            raise BenchmarkContractError("AFM-PDEB-008", f"unsupported PDE {family}")
        grid = _sample(
            case,
            solved,
            vector=family in {"linear_elasticity", "stokes", "navier_stokes"},
        )
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
        fem.form(a),
        fem.form(L),
        unknown.value,
        bcs=bcs,
        options=policy.linear_options(),
        return_info=True,
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
    unknown = fields.vector_unknown(domain, name="u", degree=degree, dim=dimension)
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
        return 2.0 * shear_modulus * eps + lame_lambda * ufl.tr(eps) * ufl.Identity(
            dimension
        )

    a = ufl.inner(stress(unknown.trial), strain(unknown.test)) * ufl.dx
    L = ufl.inner(source, unknown.test) * ufl.dx
    _, info = solvers.solve_linear_problem(
        fem.form(a),
        fem.form(L),
        unknown.value,
        bcs=bcs,
        options=policy.linear_options(),
        return_info=True,
    )
    payload = _linear_info(info, policy)
    payload.update({"lame_lambda": lame_lambda, "shear_modulus": shear_modulus})
    return unknown.value, payload


def _solve_stokes(case, domain, policy, degree):
    """Solve steady incompressible Stokes flow with Taylor--Hood elements."""

    dimension = int(domain.geometry.dim)
    if dimension == 3:
        return _solve_stokes_block(case, domain, policy, velocity_degree=3)
    structured_planar = dimension == 2 and str(case["domain"].get("type", "")) in {
        "unit_square",
        "periodic_square",
    }
    velocity_degree = max(3, int(degree)) if structured_planar else 2
    pressure_degree = velocity_degree - 1
    unknown = fields.velocity_pressure(
        domain,
        name="VelocityPressure",
        velocity_degree=velocity_degree,
        pressure_degree=pressure_degree,
    )
    velocity, pressure = unknown.trial
    test_velocity, test_pressure = unknown.test
    pde = case["pde"]
    viscosity = float(pde.get("pde_params", {}).get("nu", 1.0))
    source_spec = pde.get("source_term", [0.0] * dimension)
    if not isinstance(source_spec, Sequence) or isinstance(source_spec, (str, bytes)):
        source_spec = [source_spec] * dimension
    velocity_space, _ = unknown.space.sub(0).collapse()
    source = _known_field(source_spec, velocity_space)
    bcs = _mixed_velocity_bcs(case, unknown.space)
    pressure_reference = _velocity_boundary_is_complete(
        case.get("bc", {}).get("dirichlet"), dimension
    )
    if pressure_reference:
        bcs.append(_reference_pressure_bc(unknown.space))

    viscous = operators.viscous_flow_operator(
        velocity, test_velocity, viscosity
    ).expression
    pressure_term = operators.pressure_coupling_operator(
        pressure, test_velocity
    ).expression
    continuity = operators.incompressibility_operator(
        velocity, test_pressure
    ).expression
    load = ufl.inner(source, test_velocity) * ufl.dx
    _, info = solvers.solve_linear_problem(
        fem.form(viscous + pressure_term + continuity),
        fem.form(load),
        unknown.value,
        bcs=bcs,
        options=solvers.direct_solver(package="mumps"),
        return_info=True,
    )
    velocity_field = unknown.collapsed_velocity(name="Velocity")
    pressure_field = unknown.collapsed_pressure(name="Pressure")
    payload = _linear_info(info, policy, indefinite=True)
    payload.update(
        {
            "formulation": "Taylor-Hood",
            "element_degree": velocity_degree,
            "velocity_degree": velocity_degree,
            "pressure_degree": pressure_degree,
            "viscosity": viscosity,
            "pressure_reference": "interior_point" if pressure_reference else "natural",
            "mixed_num_dofs": int(
                unknown.space.dofmap.index_map.size_global
                * unknown.space.dofmap.index_map_bs
            ),
            "pressure_l2_norm": float(
                np.sqrt(
                    domain.comm.allreduce(
                        fem.assemble_scalar(
                            fem.form(pressure_field * pressure_field * ufl.dx)
                        ),
                        op=MPI.SUM,
                    )
                )
            ),
        }
    )
    return velocity_field, payload


def _solve_stokes_block(case, domain, policy, *, velocity_degree: int):
    """Solve three-dimensional Stokes flow with an explicit block contract."""

    dimension = int(domain.geometry.dim)
    pressure_degree = int(velocity_degree) - 1
    velocity_space = spaces.vector_space(domain, degree=velocity_degree)
    pressure_space = spaces.scalar_space(domain, degree=pressure_degree)
    velocity = ufl.TrialFunction(velocity_space)
    pressure = ufl.TrialFunction(pressure_space)
    test_velocity = ufl.TestFunction(velocity_space)
    test_pressure = ufl.TestFunction(pressure_space)
    pde = case["pde"]
    viscosity = float(pde.get("pde_params", {}).get("nu", 1.0))
    source_spec = pde.get("source_term", [0.0] * dimension)
    if not isinstance(source_spec, Sequence) or isinstance(source_spec, (str, bytes)):
        source_spec = [source_spec] * dimension
    source = _known_field(source_spec, velocity_space)
    velocity_field = fem.Function(velocity_space, name="Velocity")
    pressure_field = fem.Function(pressure_space, name="Pressure")
    bcs = _dirichlet_bcs(case, velocity_field)

    a00 = viscosity * ufl.inner(
        ufl.grad(velocity), ufl.grad(test_velocity)
    ) * ufl.dx
    a01 = -pressure * ufl.div(test_velocity) * ufl.dx
    a10 = -test_pressure * ufl.div(velocity) * ufl.dx
    pressure_mass = (1.0 / viscosity) * pressure * test_pressure * ufl.dx
    block_operator = [[a00, a01], [a10, None]]
    block_load = [
        ufl.inner(source, test_velocity) * ufl.dx,
        ufl.ZeroBaseForm((test_pressure,)),
    ]
    preconditioner = [[a00, None], [None, pressure_mass]]
    problem = LinearProblem(
        block_operator,
        block_load,
        u=[velocity_field, pressure_field],
        P=preconditioner,
        kind="nest",
        bcs=bcs,
        petsc_options_prefix="agentfem_pdebench_stokes3d_",
        petsc_options={
            "ksp_type": "minres",
            "ksp_rtol": max(policy.relative_tolerance, 1.0e-6),
            "ksp_atol": policy.absolute_tolerance,
            "ksp_max_it": 1000,
            "ksp_error_if_not_converged": True,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "additive",
        },
    )

    null_vector = create_vector(fem.extract_function_spaces(problem.L), "nest")
    velocity_null, pressure_null = null_vector.getNestSubVecs()
    velocity_null.set(0.0)
    pressure_null.set(1.0)
    null_vector.normalize()
    nullspace = PETSc.NullSpace().create(vectors=[null_vector])
    problem.A.setNullSpace(nullspace)
    velocity_block = problem.A.getNestSubMatrix(0, 0)
    velocity_block.setOption(PETSc.Mat.Option.SPD, True)
    if problem.P_mat is not None:
        preconditioner_velocity = problem.P_mat.getNestSubMatrix(0, 0)
        preconditioner_pressure = problem.P_mat.getNestSubMatrix(1, 1)
        preconditioner_velocity.setOption(PETSc.Mat.Option.SPD, True)
        preconditioner_pressure.setOption(PETSc.Mat.Option.SPD, True)
    block_pc = problem.solver.getPC()
    block_pc.setUp()
    velocity_solver, pressure_solver = block_pc.getFieldSplitSubKSP()
    velocity_solver.setType("preonly")
    velocity_solver.getPC().setType("gamg")
    pressure_solver.setType("preonly")
    pressure_solver.getPC().setType("jacobi")

    solved_velocity, solved_pressure = problem.solve()
    solved_velocity.x.scatter_forward()
    solved_pressure.x.scatter_forward()
    solver = problem.solver
    payload = {
        "ksp_type": "minres",
        "pc_type": "fieldsplit",
        "rtol": max(policy.relative_tolerance, 1.0e-6),
        "converged": int(solver.getConvergedReason()) > 0,
        "converged_reason": int(solver.getConvergedReason()),
        "iterations": int(solver.getIterationNumber()),
        "residual_norm": float(solver.getResidualNorm()),
        "formulation": "block_taylor_hood",
        "element_degree": int(velocity_degree),
        "velocity_degree": int(velocity_degree),
        "pressure_degree": int(pressure_degree),
        "viscosity": viscosity,
        "pressure_reference": "constant_nullspace",
        "mixed_num_dofs": int(
            velocity_space.dofmap.index_map.size_global
            * velocity_space.dofmap.index_map_bs
            + pressure_space.dofmap.index_map.size_global
            * pressure_space.dofmap.index_map_bs
        ),
        "pressure_l2_norm": float(
            np.sqrt(
                domain.comm.allreduce(
                    fem.assemble_scalar(
                        fem.form(solved_pressure * solved_pressure * ufl.dx)
                    ),
                    op=MPI.SUM,
                )
            )
        ),
    }
    nullspace.destroy()
    null_vector.destroy()
    return solved_velocity, payload


def _solve_navier_stokes(case, domain, policy, degree):
    """Solve steady incompressible Navier--Stokes with a Stokes predictor."""

    dimension = int(domain.geometry.dim)
    velocity_degree = max(3, int(degree)) if dimension == 2 else 2
    pressure_degree = velocity_degree - 1
    unknown = fields.velocity_pressure(
        domain,
        name="VelocityPressure",
        velocity_degree=velocity_degree,
        pressure_degree=pressure_degree,
    )
    trial_velocity, trial_pressure = unknown.trial
    test_velocity, test_pressure = unknown.test
    pde = case["pde"]
    viscosity = float(pde.get("pde_params", {}).get("nu", 0.1))
    source_spec = pde.get("source_term", [0.0] * dimension)
    if not isinstance(source_spec, Sequence) or isinstance(source_spec, (str, bytes)):
        source_spec = [source_spec] * dimension
    velocity_space, _ = unknown.space.sub(0).collapse()
    source = _known_field(source_spec, velocity_space)
    bcs = _mixed_velocity_bcs(case, unknown.space)
    pressure_reference = _velocity_boundary_is_complete(
        case.get("bc", {}).get("dirichlet"), dimension
    )
    if pressure_reference:
        bcs.append(_reference_pressure_bc(unknown.space))

    # A Stokes predictor gives Newton a physically scaled, divergence-free
    # initial state without embedding a manufactured or case-specific guess.
    predictor = (
        operators.viscous_flow_operator(
            trial_velocity, test_velocity, viscosity
        ).expression
        + operators.pressure_coupling_operator(trial_pressure, test_velocity).expression
        + operators.incompressibility_operator(trial_velocity, test_pressure).expression
    )
    load = ufl.inner(source, test_velocity) * ufl.dx
    solvers.solve_linear_problem(
        fem.form(predictor),
        fem.form(load),
        unknown.value,
        bcs=bcs,
        options=solvers.direct_solver(package="mumps"),
    )

    velocity, pressure = ufl.split(unknown.value)
    residual = (
        viscosity * ufl.inner(ufl.grad(velocity), ufl.grad(test_velocity))
        + ufl.inner(ufl.dot(ufl.grad(velocity), velocity), test_velocity)
        - pressure * ufl.div(test_velocity)
        - test_pressure * ufl.div(velocity)
        - ufl.inner(source, test_velocity)
    ) * ufl.dx
    jacobian = ufl.derivative(residual, unknown.value, ufl.TrialFunction(unknown.space))
    options = solvers.NonlinearSolverOptions(
        rtol=1.0e-9,
        atol=1.0e-11,
        max_it=40,
        line_search_type="bt",
        ksp_type="preonly",
        pc_type="lu",
        factor_solver_type="mumps",
    )
    _, info = solvers.solve_nonlinear_problem(
        fem.form(residual),
        unknown.value,
        bcs=bcs,
        jacobian_form=fem.form(jacobian),
        options=options,
        petsc_options_prefix="agentfem_pdebench_ns_",
    )
    velocity_field = unknown.collapsed_velocity(name="Velocity")
    payload = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "converged": bool(info.converged),
        "converged_reason": int(info.converged_reason),
        "iterations": int(info.iterations),
        "residual_norm": float(info.function_norm),
        "formulation": "steady_newton_taylor_hood",
        "velocity_degree": velocity_degree,
        "pressure_degree": pressure_degree,
        "viscosity": viscosity,
        "initialization": "stokes_predictor",
        "pressure_reference": "interior_point" if pressure_reference else "natural",
    }
    return velocity_field, payload


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
        raise BenchmarkContractError(
            "AFM-PDEB-002", "heat analysis needs at least one step"
        )
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
    payload.update(
        {
            "num_timesteps": steps,
            "n_steps": steps,
            "time_scheme": "backward_euler",
            "dt": dt,
            "matrix_reused": True,
        }
    )
    return unknown.value, payload, initial_field


def _solve_convection_diffusion(case, domain, policy, degree):
    """Solve steady or backward-Euler advection--diffusion with optional SUPG."""

    unknown = fields.scalar_unknown(domain, name="u", degree=degree)
    pde = case["pde"]
    params = pde.get("pde_params", {})
    epsilon = float(params.get("epsilon", 1.0))
    beta_values = tuple(float(value) for value in params.get("beta", (1.0, 0.0)))
    dimension = int(domain.geometry.dim)
    if len(beta_values) != dimension:
        raise BenchmarkContractError(
            "AFM-PDEB-003",
            f"convection vector has {len(beta_values)} components for a {dimension}D domain",
        )
    beta = operators.as_velocity(beta_values)
    stabilization = str(params.get("stabilization", "none")).lower() == "supg"
    source_spec = pde.get("source_term", 0.0)
    time_cfg = pde.get("time")
    time = fem.Constant(domain, float(time_cfg.get("t0", 0.0)) if time_cfg else 0.0)
    source = _known_field(source_spec, unknown.space, parameters={"t": time})
    bcs, boundary_values = _dirichlet_bcs(
        case, unknown.value, parameters={"t": time}, track_values=True
    )

    advected = ufl.dot(beta, ufl.grad(unknown.trial))
    a_spatial = (
        epsilon * ufl.inner(ufl.grad(unknown.trial), ufl.grad(unknown.test))
        + advected * unknown.test
    ) * ufl.dx
    L_spatial = source * unknown.test * ufl.dx
    tau = None
    if stabilization and np.linalg.norm(beta_values) > 0.0:
        tau = operators.intrinsic_time_scale(domain, beta_values)
        streamline_test = ufl.dot(beta, ufl.grad(unknown.test))
        strong_trial = advected - epsilon * ufl.div(ufl.grad(unknown.trial))
        a_spatial += tau * strong_trial * streamline_test * ufl.dx
        L_spatial += tau * source * streamline_test * ufl.dx

    initial_field = None
    if time_cfg is None:
        _, info = solvers.solve_linear_problem(
            fem.form(a_spatial),
            fem.form(L_spatial),
            unknown.value,
            bcs=bcs,
            options=policy.linear_options(indefinite=True),
            return_info=True,
        )
        payload = _linear_info(info, policy, indefinite=True)
        payload.update(
            {
                "stabilization": "supg" if stabilization else "none",
                "epsilon": epsilon,
                "beta": beta_values,
                "steady": True,
            }
        )
        return unknown.value, payload, initial_field

    previous = fem.Function(unknown.space, name="u_previous")
    expressions.interpolate(
        previous, pde.get("initial_condition", 0.0), parameters={"t": time}
    )
    initial_field = fem.Function(unknown.space, name="u_initial")
    initial_field.x.array[:] = previous.x.array
    initial_field.x.scatter_forward()
    t0, t_end, dt, steps = _time_grid(time_cfg)
    a = (unknown.trial * unknown.test) * ufl.dx + dt * a_spatial
    L = (previous * unknown.test) * ufl.dx + dt * L_spatial
    if tau is not None:
        streamline_test = ufl.dot(beta, ufl.grad(unknown.test))
        a += tau * unknown.trial * streamline_test * ufl.dx
        L += tau * previous * streamline_test * ufl.dx
    with solvers.prepare_linear_problem(
        a, L, unknown.value, bcs=bcs, options=policy.linear_options(indefinite=True)
    ) as problem:
        for step in range(1, steps + 1):
            time.value = t0 + step * dt
            expressions.interpolate(source, source_spec, parameters={"t": time})
            _refresh_dirichlet_values(boundary_values, parameters={"t": time})
            problem.solve()
            previous.x.array[:] = unknown.value.x.array
            previous.x.scatter_forward()
        info = problem.last_solve_info
    if info is None:
        raise RuntimeError("Transient convection--diffusion did not execute a step.")
    payload = _linear_info(info, policy, indefinite=True)
    payload.update(
        {
            "stabilization": "supg" if stabilization else "none",
            "epsilon": epsilon,
            "beta": beta_values,
            "steady": False,
            "num_timesteps": steps,
            "n_steps": steps,
            "time_scheme": "backward_euler",
            "dt": dt,
            "matrix_reused": True,
        }
    )
    return unknown.value, payload, initial_field


def _solve_reaction_diffusion(case, domain, policy, degree):
    """Solve transient linear or nonlinear reaction--diffusion equations."""

    unknown = fields.scalar_unknown(domain, name="u", degree=degree)
    pde = case["pde"]
    params = pde.get("pde_params", {})
    epsilon = float(params.get("epsilon", 1.0))
    reaction = params.get("reaction", {"type": "linear", "alpha": 0.0})
    reaction_type = str(reaction.get("type", "linear")).lower()
    time_cfg = pde["time"]
    t0, _, dt, steps = _time_grid(time_cfg)
    time = fem.Constant(domain, t0)
    previous = fem.Function(unknown.space, name="u_previous")
    initial_spec = pde.get("initial_condition", 0.0)
    source_spec = pde.get("source_term", 0.0)
    expressions.interpolate(previous, initial_spec, parameters={"t": time})
    source = _known_field(source_spec, unknown.space, parameters={"t": time})
    initial_field = fem.Function(unknown.space, name="u_initial")
    initial_field.x.array[:] = previous.x.array
    initial_field.x.scatter_forward()
    bcs, boundary_values = _dirichlet_bcs(
        case, unknown.value, parameters={"t": time}, track_values=True
    )
    scheme = str(time_cfg.get("scheme", "backward_euler")).lower().replace("-", "_")

    if reaction_type == "linear":
        alpha = float(reaction.get("alpha", 0.0))
        theta = 0.5 if scheme == "crank_nicolson" else 1.0
        previous_source = fem.Function(unknown.space, name="source_previous")
        previous_source.x.array[:] = source.x.array
        previous_source.x.scatter_forward()
        a = (
            unknown.trial * unknown.test
            + dt
            * theta
            * (
                epsilon * ufl.inner(ufl.grad(unknown.trial), ufl.grad(unknown.test))
                + alpha * unknown.trial * unknown.test
            )
        ) * ufl.dx
        L = (
            previous * unknown.test
            - dt
            * (1.0 - theta)
            * (
                epsilon * ufl.inner(ufl.grad(previous), ufl.grad(unknown.test))
                + alpha * previous * unknown.test
            )
            + dt * (theta * source + (1.0 - theta) * previous_source) * unknown.test
        ) * ufl.dx
        with solvers.prepare_linear_problem(
            a, L, unknown.value, bcs=bcs, options=policy.linear_options(indefinite=True)
        ) as problem:
            for step in range(1, steps + 1):
                time.value = t0 + step * dt
                expressions.interpolate(source, source_spec, parameters={"t": time})
                _refresh_dirichlet_values(boundary_values, parameters={"t": time})
                problem.solve()
                previous.x.array[:] = unknown.value.x.array
                previous.x.scatter_forward()
                previous_source.x.array[:] = source.x.array
                previous_source.x.scatter_forward()
            info = problem.last_solve_info
        if info is None:
            raise RuntimeError("Linear reaction--diffusion did not execute a step.")
        payload = _linear_info(info, policy, indefinite=True)
        nonlinear_iterations: list[int] = []
    else:
        value = unknown.value
        test = unknown.test
        reaction_value = operators.reaction_expression(value, reaction)
        residual = (
            (value - previous) * test
            + dt * epsilon * ufl.inner(ufl.grad(value), ufl.grad(test))
            + dt * reaction_value * test
            - dt * source * test
        ) * ufl.dx
        jacobian = ufl.derivative(residual, value, unknown.trial)
        options = solvers.NonlinearSolverOptions(
            rtol=policy.relative_tolerance,
            atol=policy.absolute_tolerance,
            max_it=35,
            line_search_type="bt",
            ksp_type="preonly",
            pc_type="lu",
        )
        nonlinear_iterations = []
        info = None
        for step in range(1, steps + 1):
            time.value = t0 + step * dt
            expressions.interpolate(source, source_spec, parameters={"t": time})
            _refresh_dirichlet_values(boundary_values, parameters={"t": time})
            value.x.array[:] = previous.x.array
            value.x.scatter_forward()
            _, info = solvers.solve_nonlinear_problem(
                fem.form(residual),
                value,
                bcs=bcs,
                jacobian_form=fem.form(jacobian),
                options=options,
                petsc_options_prefix="agentfem_pdebench_rd_",
            )
            if not info.converged:
                raise RuntimeError(
                    f"reaction--diffusion Newton failed at step {step}: {info.as_dict()}"
                )
            nonlinear_iterations.append(int(info.iterations))
            previous.x.array[:] = value.x.array
            previous.x.scatter_forward()
        if info is None:
            raise RuntimeError("Nonlinear reaction--diffusion did not execute a step.")
        payload = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "converged": bool(info.converged),
            "converged_reason": int(info.converged_reason),
            "iterations": int(info.iterations),
            "residual_norm": float(info.function_norm),
        }

    payload.update(
        {
            "reaction_type": reaction_type,
            "epsilon": epsilon,
            "scheme": scheme,
            "time_scheme": scheme,
            "num_timesteps": steps,
            "n_steps": steps,
            "dt": dt,
            "nonlinear_iterations": nonlinear_iterations,
            "nonlinear_iterations_total": sum(nonlinear_iterations),
        }
    )
    return unknown.value, payload, initial_field


def _solve_burgers(case, domain, policy, degree):
    """Solve multidimensional scalar Burgers transport semi-implicitly."""

    unknown = fields.scalar_unknown(domain, name="u", degree=degree)
    pde = case["pde"]
    viscosity = float(pde.get("pde_params", {}).get("nu", 0.01))
    t0 = float(pde.get("t0", 0.0))
    t_end = float(pde.get("t_final", 0.1))
    nominal_dt = float(pde.get("dt", 0.01))
    steps = max(1, int(np.ceil((t_end - t0) / nominal_dt - 1.0e-14)))
    dt = (t_end - t0) / steps
    time = fem.Constant(domain, t0)
    previous = fem.Function(unknown.space, name="u_previous")
    expressions.interpolate(
        previous, pde.get("initial_condition", 0.0), parameters={"t": time}
    )
    initial_field = fem.Function(unknown.space, name="u_initial")
    initial_field.x.array[:] = previous.x.array
    initial_field.x.scatter_forward()
    source_spec = pde.get("source_term", 0.0)
    source = _known_field(source_spec, unknown.space, parameters={"t": time})
    periodic = "periodic" in case.get("bc", {})
    if periodic:
        bcs, boundary_values = [], []
    else:
        bcs, boundary_values = _dirichlet_bcs(
            case, unknown.value, parameters={"t": time}, track_values=True
        )

    convection = operators.burgers_convection_operator(
        previous, unknown.trial, unknown.test
    ).expression
    a = (
        unknown.trial * unknown.test
        + dt * viscosity * ufl.inner(ufl.grad(unknown.trial), ufl.grad(unknown.test))
    ) * ufl.dx + dt * convection
    L = (previous + dt * source) * unknown.test * ufl.dx
    info = None
    solved = unknown.value
    if periodic:
        try:
            from ...constraints.mpc import rectangular_periodic_mpc

            periodicity = rectangular_periodic_mpc(unknown.value.function_space)
            problem = solvers.prepare_mpc_linear_problem(
                a,
                L,
                unknown.value,
                periodicity,
                options=policy.linear_options(indefinite=True),
                petsc_options_prefix="agentfem_pdebench_burgers_periodic_",
            )
        except ImportError as exc:
            raise BenchmarkContractError(
                "AFM-PDEB-009",
                "periodic PDE cases require the optional dolfinx_mpc backend",
            ) from exc
        for step in range(1, steps + 1):
            time.value = t0 + step * dt
            expressions.interpolate(source, source_spec, parameters={"t": time})
            problem.solve()
            previous.x.array[:] = unknown.value.x.array
            previous.x.scatter_forward()
        solved = unknown.value
        info = problem.last_solve_info
    else:
        with solvers.prepare_linear_problem(
            a,
            L,
            unknown.value,
            bcs=bcs,
            options=policy.linear_options(indefinite=True),
        ) as problem:
            for step in range(1, steps + 1):
                time.value = t0 + step * dt
                expressions.interpolate(source, source_spec, parameters={"t": time})
                _refresh_dirichlet_values(boundary_values, parameters={"t": time})
                problem.solve()
                previous.x.array[:] = unknown.value.x.array
                previous.x.scatter_forward()
            info = problem.last_solve_info
    if info is None:
        raise RuntimeError("Burgers analysis did not execute a step.")
    payload = _linear_info(info, policy, indefinite=True)
    payload.update(
        {
            "formulation": "semi_implicit_scalar_burgers",
            "viscosity": viscosity,
            "time_scheme": "backward_euler_semi_implicit",
            "num_timesteps": steps,
            "n_steps": steps,
            "dt": dt,
            "stabilization": "none",
            "boundary_method": "periodic_mpc" if periodic else "dirichlet",
        }
    )
    return solved, payload, initial_field


def _solve_biharmonic(case, domain, policy, degree):
    """Solve ``Delta^2 u=f`` through ``w=-Delta u`` and two Poisson blocks."""

    primary = fields.scalar_unknown(domain, name="u", degree=degree)
    auxiliary = fields.scalar_unknown(domain, name="w", degree=degree)
    source = _known_field(case["pde"].get("source_term", 0.0), primary.space)
    raw = case.get("bc", {}).get("dirichlet")
    entries = raw if isinstance(raw, list) else [raw]
    if len(entries) != 1 or not isinstance(entries[0], Mapping):
        raise BenchmarkContractError(
            "AFM-PDEB-005", "biharmonic split requires one public boundary expression"
        )
    entry = entries[0]
    selector = str(entry.get("on", "all"))
    boundary_source = entry.get("value", 0.0)
    primary_bc = _dirichlet_bcs(case, primary.value)

    auxiliary_boundary = fem.Function(auxiliary.space, name="w_boundary")
    auxiliary_boundary_representation = _interpolate_auxiliary_boundary(
        auxiliary_boundary,
        boundary_source,
        domain,
    )
    auxiliary_dofs = _boundary_dofs(auxiliary.space, selector)
    auxiliary_bc = fem.dirichletbc(auxiliary_boundary, auxiliary_dofs)

    auxiliary_form = operators.split_laplacian_operator(
        auxiliary.trial, auxiliary.test, name="K_auxiliary"
    ).expression
    _, auxiliary_info = solvers.solve_linear_problem(
        fem.form(auxiliary_form),
        fem.form(source * auxiliary.test * ufl.dx),
        auxiliary.value,
        bcs=[auxiliary_bc],
        options=policy.linear_options(),
        return_info=True,
    )
    primary_form = operators.split_laplacian_operator(
        primary.trial, primary.test, name="K_primary"
    ).expression
    _, primary_info = solvers.solve_linear_problem(
        fem.form(primary_form),
        fem.form(auxiliary.value * primary.test * ufl.dx),
        primary.value,
        bcs=primary_bc,
        options=policy.linear_options(),
        return_info=True,
    )
    payload = _linear_info(primary_info, policy)
    payload.update(
        {
            "formulation": "mixed_two_poisson",
            "auxiliary_field": "w=-laplacian(u)",
            "auxiliary_boundary": "negative_laplacian_of_public_boundary_expression",
            "auxiliary_boundary_representation": auxiliary_boundary_representation,
            "auxiliary_iterations": int(auxiliary_info.iterations),
        }
    )
    return primary.value, payload


def _interpolate_auxiliary_boundary(target, boundary_source, domain) -> str:
    """Interpolate ``-Delta(g)`` without assuming a structured mesh.

    DOLFINx can directly tabulate the symbolic derivative on meshes created by
    its native constructors.  Some imported Gmsh meshes reject that expression
    during interpolation even though its UFL domain has the same mesh cargo.
    The fallback first represents ``g`` in the target finite-element space,
    keeping every coefficient on the actual imported mesh, and then evaluates
    the same operator.  Polynomial boundary data within the selected element
    order remain exact; general data retain the normal interpolation accuracy.
    """

    boundary_ufl = expressions.as_ufl(boundary_source, domain)
    if not ufl.domain.extract_domains(boundary_ufl):
        auxiliary_ufl = fem.Constant(domain, 0.0)
        representation = "exact_constant"
    else:
        auxiliary_ufl = operators.auxiliary_laplacian_boundary(boundary_ufl)
        representation = "symbolic"
    interpolation_points = target.function_space.element.interpolation_points
    try:
        target.interpolate(fem.Expression(auxiliary_ufl, interpolation_points))
    except RuntimeError as exc:
        if "different mesh" not in str(exc).lower():
            raise
        primary_boundary = fem.Function(
            target.function_space,
            name="u_boundary_on_mesh",
        )
        expressions.interpolate(primary_boundary, boundary_source)
        auxiliary_ufl = operators.auxiliary_laplacian_boundary(primary_boundary)
        target.interpolate(fem.Expression(auxiliary_ufl, interpolation_points))
        representation = "same_mesh_finite_element"
    target.x.scatter_forward()
    return representation


def _solve_wave(case, domain, policy, degree):
    """Solve the scalar wave equation with average-acceleration Newmark."""

    unknown = fields.scalar_unknown(domain, name="u", degree=degree)
    pde = case["pde"]
    time_cfg = pde["time"]
    t0, _, dt, steps = _time_grid(time_cfg)
    wave_speed = float(pde.get("pde_params", {}).get("c", 1.0))
    time = fem.Constant(domain, t0)
    displacement = unknown.value
    velocity = fem.Function(unknown.space, name="velocity")
    acceleration = fem.Function(unknown.space, name="acceleration")
    expressions.interpolate(
        displacement, pde.get("initial_condition", 0.0), parameters={"t": time}
    )
    expressions.interpolate(
        velocity, pde.get("initial_velocity", 0.0), parameters={"t": time}
    )
    initial_field = fem.Function(unknown.space, name="u_initial")
    initial_field.x.array[:] = displacement.x.array
    initial_field.x.scatter_forward()
    source_spec = pde.get("source_term", 0.0)
    source = _known_field(source_spec, unknown.space, parameters={"t": time})
    bcs, boundary_values = _dirichlet_bcs(
        case, displacement, parameters={"t": time}, track_values=True
    )

    # Consistent initial acceleration: M a0 = f0 - K u0.  Homogeneous
    # acceleration values on constrained displacement dofs avoid injecting a
    # spurious boundary impulse; Newmark then enforces the prescribed motion.
    zero_bcs = []
    for bc in bcs:
        zero = fem.Function(unknown.space)
        zero_bcs.append(fem.dirichletbc(zero, bc.dof_indices()[0]))
    mass = unknown.trial * unknown.test * ufl.dx
    initial_rhs = (
        source * unknown.test
        - wave_speed**2 * ufl.inner(ufl.grad(displacement), ufl.grad(unknown.test))
    ) * ufl.dx
    solvers.solve_linear_problem(
        fem.form(mass),
        fem.form(initial_rhs),
        acceleration,
        bcs=zero_bcs,
        options=policy.linear_options(),
    )

    beta_newmark = 0.25
    gamma_newmark = 0.5
    predicted_u = fem.Function(unknown.space, name="predicted_u")
    predicted_v = fem.Function(unknown.space, name="predicted_v")
    effective = (
        unknown.trial * unknown.test
        + beta_newmark
        * dt**2
        * wave_speed**2
        * ufl.inner(ufl.grad(unknown.trial), ufl.grad(unknown.test))
    ) * ufl.dx
    rhs = (
        predicted_u * unknown.test + beta_newmark * dt**2 * source * unknown.test
    ) * ufl.dx
    with solvers.prepare_linear_problem(
        effective,
        rhs,
        displacement,
        bcs=bcs,
        options=policy.linear_options(),
    ) as problem:
        for step in range(1, steps + 1):
            predicted_u.x.array[:] = (
                displacement.x.array
                + dt * velocity.x.array
                + dt**2 * (0.5 - beta_newmark) * acceleration.x.array
            )
            predicted_v.x.array[:] = (
                velocity.x.array + dt * (1.0 - gamma_newmark) * acceleration.x.array
            )
            predicted_u.x.scatter_forward()
            predicted_v.x.scatter_forward()
            time.value = t0 + step * dt
            expressions.interpolate(source, source_spec, parameters={"t": time})
            _refresh_dirichlet_values(boundary_values, parameters={"t": time})
            problem.solve()
            new_acceleration = (displacement.x.array - predicted_u.x.array) / (
                beta_newmark * dt**2
            )
            velocity.x.array[:] = (
                predicted_v.x.array + gamma_newmark * dt * new_acceleration
            )
            acceleration.x.array[:] = new_acceleration
            velocity.x.scatter_forward()
            acceleration.x.scatter_forward()
        info = problem.last_solve_info
    if info is None:
        raise RuntimeError("Wave analysis did not execute a time step.")
    payload = _linear_info(info, policy)
    payload.update(
        {
            "time_integrator": "newmark_average_acceleration",
            "wave_speed": wave_speed,
            "num_timesteps": steps,
            "n_steps": steps,
            "time_scheme": "newmark_average_acceleration",
            "dt": dt,
            "matrix_reused": True,
        }
    )
    return displacement, payload, initial_field


def _time_grid(time_cfg):
    t0 = float(time_cfg.get("t0", 0.0))
    t_end = float(time_cfg["t_end"])
    nominal_dt = float(time_cfg.get("dt", 0.01))
    steps = int(np.ceil((t_end - t0) / nominal_dt - 1.0e-14))
    if steps < 1:
        raise BenchmarkContractError(
            "AFM-PDEB-002", "time interval needs at least one step"
        )
    return t0, t_end, (t_end - t0) / steps, steps


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
            "AFM-PDEB-003",
            f"could not interpolate scientific expression {source!r}: {exc}",
        ) from exc
    return value


def _dirichlet_bcs(case, function, *, parameters=None, track_values: bool = False):
    raw = case.get("bc", {}).get("dirichlet")
    if raw is None:
        raise BenchmarkContractError(
            "AFM-PDEB-005", "Dirichlet boundary data are required"
        )
    entries = raw if isinstance(raw, list) else [raw]
    bcs = []
    tracked = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise BenchmarkContractError(
                "AFM-PDEB-005", "Dirichlet entry must be a mapping"
            )
        dofs = _boundary_dofs(function.function_space, str(entry.get("on", "all")))
        value = fem.Function(function.function_space)
        expressions.interpolate(value, entry.get("value", 0.0), parameters=parameters)
        bcs.append(fem.dirichletbc(value, dofs))
        tracked.append((value, entry.get("value", 0.0)))
    return (bcs, tracked) if track_values else bcs


def _mixed_velocity_bcs(case, mixed_space, *, parameters=None):
    """Build velocity data on the first subspace of a mixed flow space."""

    raw = case.get("bc", {}).get("dirichlet")
    if raw is None:
        raise BenchmarkContractError(
            "AFM-PDEB-005", "incompressible flow requires velocity Dirichlet data"
        )
    entries = raw if isinstance(raw, list) else [raw]
    velocity_space, _ = mixed_space.sub(0).collapse()
    bcs = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise BenchmarkContractError(
                "AFM-PDEB-005", "Dirichlet entry must be a mapping"
            )
        selector = str(entry.get("on", "all"))
        dofs = _mixed_boundary_dofs(mixed_space.sub(0), velocity_space, selector)
        value = fem.Function(velocity_space)
        expressions.interpolate(
            value,
            entry.get("value", [0.0] * velocity_space.mesh.geometry.dim),
            parameters=parameters,
        )
        bcs.append(fem.dirichletbc(value, dofs, mixed_space.sub(0)))
    return bcs


def _reference_pressure_bc(mixed_space):
    """Fix pressure near the domain centre, away from corner singularities."""

    pressure_space, _ = mixed_space.sub(1).collapse()
    domain = pressure_space.mesh
    coordinates = pressure_space.tabulate_dof_coordinates()
    owned = pressure_space.dofmap.index_map.size_local
    local_min = np.min(domain.geometry.x, axis=0)
    local_max = np.max(domain.geometry.x, axis=0)
    lower = np.array(
        [domain.comm.allreduce(float(value), op=MPI.MIN) for value in local_min]
    )
    upper = np.array(
        [domain.comm.allreduce(float(value), op=MPI.MAX) for value in local_max]
    )
    centre = 0.5 * (lower + upper)
    local = None
    if owned:
        candidates = coordinates[:owned, : domain.geometry.dim]
        distances = np.sum((candidates - centre[: domain.geometry.dim]) ** 2, axis=1)
        index = int(np.argmin(distances))
        local = (
            float(distances[index]),
            tuple(float(value) for value in candidates[index]),
        )
    candidates = [candidate for candidate in domain.comm.allgather(local) if candidate]
    if not candidates:
        raise BenchmarkContractError(
            "AFM-PDEB-005", "pressure space has no dof for a reference value"
        )
    _, point = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))

    def marker(x):
        selected = np.ones(x.shape[1], dtype=bool)
        for axis, value in enumerate(point[: domain.geometry.dim]):
            selected &= np.isclose(x[axis], value)
        return selected

    dofs = fem.locate_dofs_geometrical((mixed_space.sub(1), pressure_space), marker)
    value = fem.Function(pressure_space)
    return fem.dirichletbc(value, dofs, mixed_space.sub(1))


def _velocity_boundary_is_complete(raw, dimension: int) -> bool:
    """Return whether public velocity data cover the structured outer boundary."""

    entries = raw if isinstance(raw, list) else [raw]
    selectors = {
        str(entry.get("on", "all")).strip().lower()
        for entry in entries
        if isinstance(entry, Mapping)
    }
    if selectors & {"all", "*"}:
        return True
    aliases = (
        ({"x0", "xmin"}, {"x1", "xmax"}),
        ({"y0", "ymin"}, {"y1", "ymax"}),
        ({"z0", "zmin"}, {"z1", "zmax"}),
    )
    return all(
        bool(selectors & lower) and bool(selectors & upper)
        for lower, upper in aliases[:dimension]
    )


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
    marker = _boundary_marker(domain, selector)
    return fem.locate_dofs_geometrical(space, marker)


def _mixed_boundary_dofs(subspace, collapsed_space, selector: str):
    domain = collapsed_space.mesh
    fdim = domain.topology.dim - 1
    key = selector.strip().lower()
    if key in {"all", "*"}:
        facets = dolfinx_mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        return fem.locate_dofs_topological((subspace, collapsed_space), fdim, facets)
    return fem.locate_dofs_geometrical(
        (subspace, collapsed_space), _boundary_marker(domain, selector)
    )


def _boundary_marker(domain, selector: str):
    key = selector.strip().lower()
    axis_and_side = {
        "x0": (0, "min"),
        "xmin": (0, "min"),
        "x1": (0, "max"),
        "xmax": (0, "max"),
        "y0": (1, "min"),
        "ymin": (1, "min"),
        "y1": (1, "max"),
        "ymax": (1, "max"),
        "z0": (2, "min"),
        "zmin": (2, "min"),
        "z1": (2, "max"),
        "zmax": (2, "max"),
    }
    if key not in axis_and_side or axis_and_side[key][0] >= domain.geometry.dim:
        raise BenchmarkContractError(
            "AFM-PDEB-005", f"unknown boundary selector {selector!r}"
        )
    axis, side = axis_and_side[key]
    local = (
        float(np.min(domain.geometry.x[:, axis]))
        if side == "min"
        else float(np.max(domain.geometry.x[:, axis]))
    )
    operation = MPI.MIN if side == "min" else MPI.MAX
    value = domain.comm.allreduce(local, op=operation)
    return lambda x: np.isclose(x[axis], value)


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
        raise BenchmarkContractError(
            "AFM-PDEB-007", "output has no finite in-domain values"
        )
    domain_type = str(case["domain"].get("type", ""))
    if domain_type in {"unit_square", "unit_cube", "periodic_square"} and not np.all(
        finite
    ):
        missing = int(finite.size - np.count_nonzero(finite))
        raise BenchmarkContractError(
            "AFM-PDEB-007",
            f"structured-domain output contains {missing} missing samples",
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
