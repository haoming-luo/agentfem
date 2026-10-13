# Geometric elasticity: implementation and numerical verification

Branch: `feat/geometric-elasticity`. Base: `44360903`. Date: 2026-10-11.

## Delivered capability

An explicit Saint-Venant–Kirchhoff material integrates with the standard
`model.step(...)` finite-strain workflow in 3D, plane strain and exactly condensed
plane stress. Energy differentiation supplies the consistent tangent. Existing
linear-elastic and Neo-Hookean material choices retain their meanings.

The work includes admissibility checks, automatic cutback and rollback,
checkpoint restart, full three-dimensional stress/strain outputs for reduced
models, conservative load-controlled response extraction, and integration with
the existing serial arc-length continuation. The material is intended for
finite rotations with small material strains, not unrestricted large strain.

Point constraints now reject a coordinate with no globally owned matching
DOFs. Metadata records requested and actual support coordinates. MPI-local
absence is not an error when a matching DOF exists on another rank. The
re-entrant honeycomb example now explicitly anchors a real left-boundary node.

## Measured structural results

All values below come from computed fields and saved histories, not imposed
beam formulas inside the finite-element solver. JSON summaries and CSV curves
are in `results/`; the scripts regenerate XDMF/HDF5 fields locally.

### Euler load from a solved elastic preload

The cantilever has L=200 mm, h=b=10 mm, E=210000 MPa and nu=0.3. Its slender-beam
Euler reference is 10794.879814 N. Solid discretizations have finite shear and
end effects; therefore refinement need not converge exactly to Euler theory.

| Model | Mesh | Critical load / N | Difference from Euler |
|---|---|---:|---:|
| Plane stress | 40 x 2 | 10781.724514 | 0.1219% |
| Plane stress | 80 x 4 | 10779.601091 | 0.1415% |
| Plane stress | 120 x 6 | 10779.112245 | 0.1461% |
| 3D solid | 40 x 3 x 3 | 10805.002256 | 0.0938% |

### Imperfect column under 10.7 kN dead load

A stress-free Euler-mode perturbation gives initial tip offsets of 0.1 or
0.2 mm. Traction is specified on the reference end face, which remains free
to rotate. The reported displacement is the *additional* transverse tip
motion relative to the imperfect reference geometry.

| Initial offset / mm | 40 x 2 tip / mm | 80 x 4 tip / mm | 120 x 6 tip / mm | Last refinement change |
|---:|---:|---:|---:|---:|
| 0.1 | 11.728988 | 11.952521 | 12.004779 | 0.4372% |
| 0.2 | 19.249377 | 19.490089 | 19.545816 | 0.2859% |

Halving the load increment from 0.05 to 0.025 at the finest mesh changed final
displacement by less than 1e-6% for both offsets. Reconstructed support forces
balance the prescribed 10700 N to within 1e-4 N (the recorded errors are much
smaller). These are nonlinear equilibria below the perfect-column Euler load;
solver failure is not used as evidence of instability.

For this smooth elastic benchmark, explicit full Newton with automatic
cutback reduced the 0.1 mm coarse-mesh run from hundreds of backtracking
increments to 40, preserving the final displacement. This is a benchmark
observation, not a universal speedup or a changed library default.

### Large rotation against an independent elastica solution

A slender cantilever with L/h=100 and nu=0 is loaded by a fixed-direction tip
traction. The independent inextensible beam boundary-value problem is
`theta'' + alpha*cos(theta)=0`, with clamped rotation and zero free-end moment;
`alpha=P*L^2/(E*I)`. SciPy solves this BVP independently of AgentFEM.

| alpha | Reference tip y/L | 80-cell solid relative difference |
|---:|---:|---:|
| 1 | 0.301720774 | 0.00644% |
| 2 | 0.493457480 | 0.00782% |
| 4 | 0.669964181 | 0.00989% |

The alpha=4 reference end rotation is about 64.2 degrees. The solid includes
extension and shear, unlike the inextensible reference; agreement is assessed
within the slender-beam approximation, not as equality of identical models.

### Continuation through a limit point

The existing shallow-arch project was rerun with StVK. A 32-cell/0.04-step
run and a 64-cell/0.02-step run both pass the load maximum. Their peak factors
are 0.195409468 and 0.195483648, a difference of approximately 0.038%.
Maximum equilibrium residuals are below 7e-9. This checks the new material
with the existing generic continuation; it does not introduce contact,
plasticity, follower loads or a new MPI arc-length solver.

### MPI and corrected honeycomb

Two MPI ranks passed homogeneous 2D plane-stress and 3D traction checks. The
computed tip displacement 0.000998503987 agrees with the independent cubic
StVK uniaxial relation. A globally valid pin is accepted despite one rank
having no local support DOFs. This is correctness evidence, not scaling data.

The corrected honeycomb gives periodic nu_xy=-1.432522057 and finite-array
apparent nu_xy=-1.302608043. These are different boundary-value problems and
are deliberately not forced to agree. The installed-package run reproduces
the corrected support and writes results. The original nonexistent-origin
support fails with `AFM-CONSTRAINT-PIN-EMPTY`.

## Verification layers and reproducibility

- `tests/test_geometric_elasticity.py`: material derivatives, objectivity,
  dimensional reductions, admissibility/rollback, restart, standard provider,
  output conventions, unsupported combinations and shared tangent dispatch.
- `tests/test_geometric_elasticity_structures.py`: freshly solved column,
  independent elastica comparison and limit-point continuation.
- `mpi_patch.py`: two-rank point-constraint and homogeneous-solution checks.
- `run_euler.py`, `run_convergence.py`, `run_elastica.py`, `run_arc_length.py`:
  regenerate the numerical tables and detailed curves.
- `plot_results.py`: renders the accompanying validation figure from saved
  numerical data. Displacement shapes labelled x1 are actual-size deformations.

Run from the repository root in the supported FEniCSx environment:

```sh
PYTHONPATH=src python -m pytest tests/test_geometric_elasticity.py tests/test_geometric_elasticity_structures.py -q
PYTHONPATH=src mpiexec -n 2 python validation/geometric_elasticity/mpi_patch.py
PYTHONPATH=src python validation/geometric_elasticity/run_euler.py
PYTHONPATH=src python validation/geometric_elasticity/run_convergence.py
PYTHONPATH=src python validation/geometric_elasticity/run_elastica.py
PYTHONPATH=src python validation/geometric_elasticity/run_arc_length.py
PYTHONPATH=src python validation/geometric_elasticity/plot_results.py
agentfem run --project examples/geometric_elastic_column --output /tmp/column --json
agentfem run --project examples/geometric_elastic_cantilever --output /tmp/cantilever --json
```

## Interpretation boundaries

The convenience StVK workflow currently accepts one material assignment and
reference conservative loading. Automatic dynamics, pressure/follower loads,
initial stress, thermal eigenstrain, contact, plasticity and near-incompressible
mixed treatment are not part of this delivery. Positive Jacobian and thickness
checks are quadrature-sampled, not a mathematical proof everywhere. A valid
pin does not alone guarantee removal of all rigid-body modes.

Independent references: COMSOL's discussion of geometrically nonlinear
linear-elastic materials; Bisshopp and Drucker (1945), DOI
[10.1090/qam/13360](https://doi.org/10.1090/qam/13360).

## Repository regression status

The complete serial suite completed with **2455 passed, 157 skipped and 2
failed**. The skipped cases largely require MPI ranks or explicitly enabled
external/long-running fixtures. The two failures also reproduce without this
branch's changes:

1. `test_multi_void_fixed_stack_matches_versioned_golden`: the recorded solver
   identity differs from the current solver serialization. Reproduced from a
   clean archive of base commit `44360903`; numerical invariant gates pass.
2. `test_unreleased_checkout_does_not_reuse_a_published_tag_version`: base HEAD
   is beyond release tag v0.4.1 but still declares version 0.4.1. Reproduced on
   the original checkout. Version/release-contract changes are not silently
   included in this mechanics feature branch.

The numerical reference values and release tests were not weakened to hide
these pre-existing failures. This branch should be reviewed together with the
maintainer's release-version and Golden-identity maintenance before merging.

A wheel was built and installed in an isolated environment; both column and
large-rotation cantilever projects completed through the public CLI. The
corrected honeycomb also completed from an installed wheel. The final local
wheel has SHA-256
`668a8ce7c3ca3c2a19e697856bbd80ca1188e89e83153edb0953225dbb296d1c`.
This is a local validation build, not a newly published release.

Final targeted regression on the final source: **90 passed, 1 skipped**,
covering the new material/structures, existing constitutive models, mode
imperfections, continuation, buckling and periodic homogenization.
