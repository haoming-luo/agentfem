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

## Additional increment: mode imperfections and nonlinear loading

The same branch now includes a reversible, quality-checked mode imperfection
utility and a displacement-controlled response extractor. The nonlinear solver,
cutback and rollback are reused from the existing hyperelastic workflow.

- Final combined regression: 158 tests passed, including constitutive,
  architecture, solvers, workflows, buckling and imperfection coverage; five
  new tests passed on two MPI ranks. Ruff, REUSE, generated documentation and
  all 54 scientific cards/imports also passed.
- Triangle, quadrilateral, tetrahedron and hexahedron geometry amplitude,
  restoration and inversion rejection are tested.
- The guided-end column uses matching linear and nonlinear end constraints.
  At end shortening 0.05, amplitude 0.02 gives compression force 1.98445;
  amplitude 0.10 gives 1.76232. Increasing nx from 24 to 40 and reducing maximum
  increment from 0.10 to 0.05 gives 1.98425 for amplitude 0.02.
- `imperfect_column.json` stores full curves. Reproduce with
  `PYTHONPATH=src python tests/imperfect_column_evidence_driver.py`.
- `examples/imperfect_column` exports a CSV curve and final ParaView field.

The perturbed reference configuration is stress-free, and amplitudes have length
units. Existing boundary coordinates move: preserve boundary facet tags with
`tagged_boundary_region` rather than reusing old geometric selectors.

The displacement-control increment does not itself supply arc length. The
subsequent bounded continuation increment is documented below. Plasticity and
contact stability remain future work; internal refinement is not external validation.


## Additional increment: reusable spherical continuation

`solvers.ArcLengthPath` is independent of materials and finite-element assembly.
It consumes pure internal-force/tangent callbacks, a reference force and an
optional fixed force. A bordered sparse Newton solve enforces equilibrium and
the scaled arc constraint; orientation follows the previous accepted increment.
Failed correctors cut back without changing the accepted state.

`procedures.arc_length()` routes ordinary `model.step` to the first serial
hyperelastic adapter. No MPI gathering, contact, plasticity, history-dependent
material support or automatic bifurcation switching is implied. The older
cohesive-specific continuation is unchanged.

- Analytical two-bar arch: maximum force discrepancy about 1.02e-12, with load
  reversal and both turning points traversed.
- 2D/3D uniform hyperelastic patches agree with analytical nominal stress.
- Clamped continuum arch: upper and lower load limit points traversed. Comparing
  nx=32/arc=0.04 with nx=64/arc=0.02 at common crown displacements gives a
  normalized maximum curve difference of about 1.20% over [0.02,0.32]. This is
  an internal mesh/step comparison, not a commercial-solver comparison.
- Failed trial rollback, fixed-load equilibrium, force-unit rescaling, unsupported
  boundary/solver options and explicit distributed-adapter rejection are tested.
- Run `PYTHONPATH=src python tests/arc_length_evidence_driver.py` to reproduce
  `arc_length.json` and the figure. Run the standard project in
  `examples/shallow_arch_arc_length` for CSV and ParaView output.

Validation after the continuation increment: 165 tests passed, with the
multi-rank-only rejection test skipped in serial. Under two ranks the arc-length
test file reports 7 passed and 1 intentional skip (the serial continuum arch).
Ruff, REUSE, generated documentation and all 55 scientific cards/imports pass.
These are local checks; hosted PR CI has not been represented as completed.
