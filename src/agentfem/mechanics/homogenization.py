# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Small-strain periodic elasticity and directional effective properties."""

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from .. import constraints, operators
from ..constitutive import elasticity
from ..results import SimulationResult


def elastic_engineering_properties(stiffness):
    """Directional E_i and nu_ij=-S_ji/S_ii; no isotropy assumption.

    Voigt order is xx,yy,xy in 2D and xx,yy,zz,yz,xz,xy in 3D.
    Strains use engineering shear; stresses use tensor shear.
    For plane strain, these are constrained in-plane properties.
    """
    C = np.asarray(stiffness, dtype=float)
    if C.shape not in ((3, 3), (6, 6)) or not np.all(np.isfinite(C)):
        raise ValueError("AFM-RVE-001: expected a finite 3x3 or 6x6 stiffness.")
    scale = np.linalg.norm(C)
    if scale == 0 or np.linalg.norm(C - C.T) / scale > 1.0e-7:
        raise ValueError("AFM-RVE-002: effective stiffness must be symmetric.")
    if np.min(np.linalg.eigvalsh((C + C.T) / 2)) <= 1.0e-12 * scale:
        raise ValueError(
            "AFM-RVE-003: cell is singular, unstable or too ill-conditioned for compliance recovery."
        )
    S = np.linalg.inv(C)
    dim = 2 if C.shape == (3, 3) else 3
    E = 1.0 / np.diag(S)[:dim]
    nu = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i != j:
                nu[i, j] = -S[j, i] / S[i, i]
    return {
        "compliance": S,
        "young_moduli": E,
        "poisson_ratios": nu,
        "shear_moduli": 1.0 / np.diag(S)[dim:],
    }


def periodic_elasticity(
    model,
    target,
    *,
    anchor,
    amplitude=1.0e-3,
    periodic_tolerance=1.0e-9,
    name="periodic_elasticity",
):
    """Solve all 3/6 macrostrain cases on a rectangular matching periodic cell.

    The mesh contains the solid phase; the complete cell bounding-box volume
    normalizes stress and energy. ``target`` holds periodic fluctuations, not
    total displacement. No external loads or pre-existing constraints are
    admitted. ``anchor`` must be one solid node on the minimum-side boundary
    (not a maximum-face slave). Uses the existing exact rectangular MPC and
    ordinary linear-static Model.step for every strain case.
    """
    from ..solvers import LinearSolverOptions

    study, w = model.study, target.value
    domain = w.function_space.mesh
    dim = domain.geometry.dim
    if (
        study.analysis != "linear_static"
        or study.physics != "solid_mechanics"
        or dim not in (2, 3)
        or study.assumption not in (None, "plane_stress", "plane_strain")
    ):
        raise ValueError(
            "AFM-RVE-004: requires 2D plane stress/strain or 3D linear-static solid."
        )
    if model.loads or model.constraints or model.boundary_models or model.eigenstrains:
        raise ValueError(
            "AFM-RVE-005: cell study must have no external loads, constraints or eigenstrains."
        )
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("AFM-RVE-006: amplitude must be finite and positive.")
    coords = domain.geometry.x[:, :dim]
    lower = np.array(
        [
            domain.comm.allreduce(
                float(np.min(coords[:, i], initial=np.inf)), op=MPI.MIN
            )
            for i in range(dim)
        ]
    )
    upper = np.array(
        [
            domain.comm.allreduce(
                float(np.max(coords[:, i], initial=-np.inf)), op=MPI.MAX
            )
            for i in range(dim)
        ]
    )
    anchor = np.asarray(anchor, dtype=float)
    if (
        anchor.shape != (dim,)
        or not np.all(np.isfinite(anchor))
        or np.any(np.isclose(anchor, upper))
    ):
        raise ValueError(
            "AFM-RVE-007: anchor must be a solid node and not a maximum-face periodic slave."
        )
    pin = constraints.pin(target, at=anchor)
    bcs = tuple(item.bc for item in constraints.dirichlet_constraints((pin,)))
    owned = sum(bc.dof_indices()[1] for bc in bcs)
    if domain.comm.allreduce(owned, op=MPI.SUM) != dim:
        raise ValueError("AFM-RVE-008: anchor must constrain exactly one solid node.")
    mpc = constraints.rectangular_periodic_mpc(
        target, bcs=bcs, tolerance=periodic_tolerance
    )
    volume = float(np.prod(upper - lower))

    def integral(expr):
        return float(
            domain.comm.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM)
        )

    solid_volume = integral(1.0 * ufl.dx(domain=domain))
    K = model.stiffness(target)
    pairs = [(i, i) for i in range(dim)] + (
        [(0, 1)] if dim == 2 else [(1, 2), (0, 2), (0, 1)]
    )
    C = np.zeros((len(pairs), len(pairs)))
    energy_errors, fluctuation_fields, equilibrium_errors, periodic_errors = (
        [],
        [],
        [],
        [],
    )
    test = ufl.TestFunction(w.function_space)
    for column, (i, j) in enumerate(pairs):
        strain = np.zeros((dim, dim))
        strain[i, j] = amplitude if i == j else amplitude / 2.0
        strain[j, i] = strain[i, j]
        macro = ufl.as_matrix(strain.tolist())
        terms, stresses, measures = [], [], []
        for assignment in model.materials:
            measure = (
                assignment.region.measure
                if assignment.region is not None
                else ufl.dx(domain=domain)
            )
            stress_macro = elasticity.stress_from_strain(
                macro, assignment.item, study=study
            )
            terms.append(-ufl.inner(stress_macro, ufl.sym(ufl.grad(test))) * measure)
            stresses.append(
                elasticity.stress_from_strain(
                    macro + ufl.sym(ufl.grad(w)), assignment.item, study=study
                )
            )
            measures.append(measure)
        F = operators.from_ufl(sum(terms), name=f"macrostrain_{column}")
        step = model.step(
            target=target,
            K=K,
            F=F,
            constraints=(pin, mpc),
            name=f"cell_strain_{column}",
            solver_options=LinearSolverOptions(mpc_assembly="algebraic"),
        )
        step.solve()
        periodic_errors.append(
            step.problem.last_lifecycle_summary["periodic_relative_error"]
        )
        equilibrium_errors.append(
            step.problem.last_lifecycle_summary["relative_residual"]
        )
        for row, (a, b) in enumerate(pairs):
            C[row, column] = (
                sum(
                    integral(stress[a, b] * dx)
                    for stress, dx in zip(stresses, measures)
                )
                / volume
                / amplitude
            )
        micro = (
            sum(
                integral(ufl.inner(stress, macro + ufl.sym(ufl.grad(w))) * dx)
                for stress, dx in zip(stresses, measures)
            )
            / volume
        )
        macro_work = amplitude**2 * C[column, column]
        energy_errors.append(
            abs(micro - macro_work)
            / max(abs(micro), abs(macro_work), np.finfo(float).tiny)
        )
        saved = w.copy()
        saved.name = f"Fluctuation_{column + 1}"
        fluctuation_fields.append(saved)
    if (
        max(energy_errors) > 1.0e-7
        or max(equilibrium_errors) > 1.0e-7
        or max(periodic_errors) > 1.0e-8
    ):
        raise RuntimeError(
            "AFM-RVE-010: periodic equilibrium or Hill-Mandel consistency failed; effective properties not accepted."
        )
    properties = elastic_engineering_properties(C)
    result = SimulationResult(
        name=name,
        metadata={
            "scope": "small_strain_matching_rectangular_periodic_cell",
            "voigt_order": [f"{i + 1}{j + 1}" for i, j in pairs],
            "shear_convention": "engineering_strain_tensor_stress",
            "normalization": "complete_cell_volume",
            "assumption": study.assumption,
            "periodic_constraint": mpc.diagnostics(),
            "amplitude": amplitude,
            "poisson_definition": "nu_ij=-S_ji/S_ii; loading i, transverse j",
        },
    )
    result.add_quantities(
        {
            "effective_stiffness": C,
            **properties,
            "cell_volume": volume,
            "solid_fraction": solid_volume / volume,
            "hill_mandel_relative_errors": energy_errors,
            "equilibrium_relative_errors": equilibrium_errors,
            "periodic_relative_errors": periodic_errors,
            "stiffness_symmetry_error": np.linalg.norm(C - C.T) / np.linalg.norm(C),
        },
        kind="homogenization",
    )
    for index, saved in enumerate(fluctuation_fields, 1):
        result.add_field(
            f"Fluctuation_{index}",
            saved,
            description="Periodic displacement fluctuation for engineering macrostrain basis.",
        )
    return result


def apparent_poisson_ratio(
    target,
    *,
    axial_axis,
    transverse_axis,
    axial_faces,
    transverse_faces,
    axial_length,
    transverse_length,
    strain_tolerance=1.0e-12,
):
    """Measure finite-specimen strains from area-averaged gauge displacements.

    Face pairs are ordered minus/plus named boundary regions. The caller
    declares gauge lengths; this is an apparent specimen property, not an
    infinite-cell homogenized material constant.
    """
    u = getattr(target, "value", target)
    comm = u.function_space.mesh.comm
    dim = u.ufl_shape[0]
    if axial_axis == transverse_axis or any(
        i not in range(dim) for i in (axial_axis, transverse_axis)
    ):
        raise ValueError("Gauge axes must be distinct displacement components.")
    if any(
        not np.isfinite(x) or x <= 0
        for x in (axial_length, transverse_length, strain_tolerance)
    ):
        raise ValueError(
            "Gauge lengths and strain_tolerance must be positive and finite."
        )

    def mean(region, axis):
        ds = region.measure
        area = comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds)), op=MPI.SUM)
        if area <= 0:
            raise ValueError("Gauge boundary has zero measure.")
        return (
            comm.allreduce(fem.assemble_scalar(fem.form(u[axis] * ds)), op=MPI.SUM)
            / area
        )

    axial = (
        mean(axial_faces[1], axial_axis) - mean(axial_faces[0], axial_axis)
    ) / axial_length
    transverse = (
        mean(transverse_faces[1], transverse_axis)
        - mean(transverse_faces[0], transverse_axis)
    ) / transverse_length
    if abs(axial) <= strain_tolerance:
        raise ValueError(
            "Axial gauge strain is too small for a meaningful Poisson ratio."
        )
    return {
        "apparent_poisson_ratio": -transverse / axial,
        "axial_gauge_strain": axial,
        "transverse_gauge_strain": transverse,
        "gauge_lengths": (axial_length, transverse_length),
        "gauge_regions": [r.name for r in (*axial_faces, *transverse_faces)],
    }
