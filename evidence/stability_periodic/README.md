# Stability and periodic-elasticity handoff

Branch: `feat/stability-periodic-properties`; base: `98cf85a8`.

## Delivered scope

- Public `Study(analysis="linear_buckling", ...)` / `Model.step` route for
  initial-stress buckling of linear-elastic 2D and 3D solids. Separate named
  fixed-base and scalable-reference states; common constrained-DOF elimination;
  GNHEP eigenproblem, residuals and normalized mode fields.
- `mechanics.periodic_elasticity` recovers full 2D/3D effective stiffness on
  rectangular matching periodic meshes, including material partitions, full-cell
  volume averaging, engineering shear conventions and directional properties.
- Parameterized re-entrant honeycomb mesh and connected finite-specimen option;
  independent gauge-based apparent Poisson ratio.
- Optional algebraic MPC projection, `T.T @ A @ T`, used by the new homogenization
  workflow. Existing native MPC default remains unchanged.
- Two runnable projects, ParaView outputs, public API guide and scientific cards.

The applications contain geometry, physical inputs and output requests. Buckling
assembly, eigensolution, periodic load cases and property extraction reside in
library modules.

## Validation on 2026-10-10

Local runtime: DOLFINx 0.11.0 with PETSc/SLEPc and dolfinx_mpc.

- 111 focused serial tests passed (new numerical tests plus architecture,
  solvers, common/engineering workflows and attribution).
- All 11 new tests passed on two MPI ranks.
- Both example projects executed successfully.
- Repository Ruff and whitespace checks passed; generated documentation and all
  53 scientific cards/imports checked successfully.
- Cantilever solid at L/depth=20: refined critical stress 0.513145 versus Euler
  0.514042 (0.175% difference). Continuum/end effects mean refinement need not
  converge exactly to slender-beam theory.
- Periodic honeycomb nu_xy about -1.433; connected 3x3 specimen apparent nu_xy
  about -1.303. Different boundaries and measurements are deliberately reported
  separately. Constituent Poisson ratio is +0.3.
- Refinement data are in `2026-10-10.json`. The shear stiffness remains more
  mesh-sensitive than nu_xy; this is not a blanket all-output convergence claim.

Reproduce from repository root with the FEM environment:

```sh
PYTHONPATH=src python -m pytest -q tests/test_stability_periodic.py tests/test_architecture_contract.py tests/test_solvers.py tests/test_common_workflows.py tests/test_engineering_workflows.py tests/test_attribution.py
PYTHONPATH=src mpiexec -n 2 python -m pytest -q tests/test_stability_periodic.py
PYTHONPATH=src python tests/stability_periodic_evidence_driver.py
PYTHONPATH=src python tests/stability_periodic_plot_driver.py
PYTHONPATH=src python build_knowledge.py --check --check-imports
python build_docs.py --check
```

## Review points and limits

The optional algebraic path was necessary because this local native MPC path
produced a periodic displacement field with non-negligible Hill–Mandel error on
the re-entrant mesh. Explicit projection removes that discrepancy. This is a
bounded local finding, not a claim that all upstream MPC implementations fail.
The graph/root mapping is gathered across ranks; this path is not yet an
extreme-scale distributed implementation.

Linear buckling assumes equilibrated supplied linear reference states and
conservative loading. It does not certify the physical validity of a supplied
state, nor predict imperfect-structure ultimate capacity. No follower loads,
contact stability, elastoplastic buckling or general postbuckling claim.
Periodic homogenization is small-strain linear elasticity on matching rectangular
cells; no nonmatching interpolation or large-deformation auxetic claim.

Next independent increments: mode-based geometric imperfections with mesh-quality
checks; nonlinear displacement-control continuation; general equilibrium-driven
arc length with shallow-arch validation. Those are not included in this branch.
