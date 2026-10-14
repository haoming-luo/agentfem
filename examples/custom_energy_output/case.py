"""Custom elastic potential: shared public solve, projection and output APIs."""

import numpy as np
import ufl
from dolfinx import fem
from agentfem import mesh, fields, constraints, problems, results, project


def main():
    run = project.current_run()
    domain = mesh.rectangle((0.0, 0.0), (1.0, 0.2), (12, 3))
    u = fields.displacement(domain, degree=2).value
    u.name = "Displacement"
    v = ufl.TestFunction(u.function_space)
    strain = ufl.sym(ufl.grad(u))
    # Dimensionless elastic energy plus a convex quartic stiffening term.
    norm2 = ufl.inner(strain, strain)
    energy = 50 * norm2 + 10 * norm2**2
    load = fem.Constant(domain, np.array([0.0, 0.0], dtype=float))
    potential = (energy - ufl.dot(load, u)) * ufl.dx
    residual = ufl.derivative(potential, u, v)
    support = constraints.fixed(
        u, on=mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0))
    )
    problem = problems.nonlinear(residual, u, constraints=support)
    stress = 100 * strain + 40 * norm2 * strain
    result = results.SimulationResult(name="custom_energy")
    result.add_field(
        "Displacement", u, processing={"method": "primary_finite_element_solution"}
    )
    with results.output_series(
        result,
        run.artifact("fields.xdmf"),
        paraview=True,
        coordinate_name="load_factor",
        primary_name="U",
        primary_semantic_name="Displacement",
    ) as series:
        for factor in np.linspace(0.0, 1.0, 5):
            load.value[:] = (0.0, -0.02 * factor)
            problem.solve()
            sigma = results.project(stress, domain=domain, name="Stress")
            density = results.project(energy, domain=domain, name="Energy")
            series.write_accepted(coordinate=factor, primary=u, fields=(sigma, density))
    if domain.comm.rank == 0:
        run.publish(result)
    return result


if __name__ == "__main__":
    main()
