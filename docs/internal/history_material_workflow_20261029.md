# History-material workflow: bounded integration sprint

The hydrogel user's feedback is a workflow regression report, not evidence
that a uniaxially fitted law defines a three-dimensional material. No private
experimental data or personal information is retained here.

## Scope and ownership

1. Make upgrade guidance aware of explicitly declared Step lowerers. Preserve
   scientific warnings; never exempt an entire extension directory.
2. Exercise the public writer with 2D displacement and mixed point/cell fields
   through a real external reader and Warp, including intermediate frames.
3. Integrate the existing finite-strain `UserMaterial` and quadrature driver
   into an experimental implicit displacement Step. Reuse the Newton attempt
   runtime and PETSc, not the J2 constitutive implementation.

The first route is serial, one homogeneous material, reference dead loads,
strong end-displacement constraints, 3D or explicitly declared plane strain.
It requires positive physical duration; normalized loading coordinate and
physical material time are distinct. The material returns symmetric Cauchy
stress and declared reference `dP/dF`; the existing driver converts stress to P.
No arbitrary UMAT conversion, contact, MPC, mixed pressure, plane stress,
temperature history or MPI admission is implied.

Every trial evaluates from the same committed F and material state. Rejected
trials restore displacement, all response fields, material state and load
values without calling a time-dependent material with a fabricated dt.
Initialization and restart must not integrate the material again.

## Gates

- Two independent history laws through one ordinary `model.step` route.
- Fixed-old-state tangent checks; material-point/structural patch agreement.
- Physical-time sensitivity and exact subdivision on cutback.
- Forced failure leaves accepted state and response unchanged.
- Continuous versus restarted history; corrupt restart is atomic.
- One-grid 2D/3D output and real reader/warp checks.
- Unsupported combinations fail before numerical work.

The material author remains responsible for the physics, parameter identity,
algorithmic tangent and applicability. An optional energy is not manufactured
when absent. Local substepping needs its own full-update tangent, so it is not
automatically applied to arbitrary providers in this first route.

## References

- [DOLFINx external operators: nonlinear constitutive integration](https://a-latyshev.github.io/dolfinx-external-operator/demo/05_demo_plasticity_mohr_coulomb.html)
  separates external constitutive evaluation from residual/Jacobian assembly.
- [DOLFINx output API](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.io.html)
  documents field-space restrictions; compatible visualization is a framework
  responsibility, not permission to copy unrelated coefficient arrays.
- [FEniCS/MFront integration](https://thelfer.github.io/mgis/web/FEniCSBindings.html)
  separates global gradient/force/stiffness assembly from local stress/history/
  tangent evaluation. This informs the ownership boundary, not a claim of
  compatibility with that implementation or its historical limitations.

## Status

Implemented on `feat/history-material-workflow`, based on `80d97762`.
PR #133 remains a separate pending integration and is not modified by this
branch. No released version/tag is overwritten and no private user data is used.

## Acceptance evidence (2026-10-29)

One consolidated targeted run in the real `fenicsx-env`, with source
`PYTHONPATH=src` and BLAS/OpenMP threads limited to one, passed **412 tests**;
**3 tests were skipped because they require two MPI ranks**. This is not a
full repository run and does not promote this serial Step to MPI support.

The exact selection was:

```sh
python -m pytest -q \
  tests/test_finite_strain_material_step.py \
  tests/test_extension_upgrade_context.py tests/test_common_workflows.py \
  tests/test_small_strain_material_step.py tests/test_xdmf_layout.py \
  tests/test_live_xdmf.py tests/test_quadrature_physical_recovery.py \
  tests/test_finite_strain_j2_standard.py tests/test_user_material.py \
  tests/test_user_material_inspection.py tests/test_quadrature_snapshot_ownership.py \
  tests/test_incremental_runtime.py tests/test_project_cli.py \
  tests/test_documentation.py tests/test_architecture_contract.py \
  tests/test_workflow_usability.py tests/test_engineering_workflows.py
```

Evidence covers:

- Both history laws, scalar/batch parity, degree-one/two 2D/3D homogeneous patch agreement;
  fixed-old-state material tangents and superposed rotation covariance.
- Nonuniform 2D/3D dead-load equilibrium, loading/unloading branch tangent
  directions and reaction balance. The damage tangent check avoids the
  nonsmooth history-switch point rather than loosening a tolerance there.
- An independent exact relaxation convolution with first-order time refinement,
  physical-duration sensitivity and clean material-requested cutback.
- Continuous/restarted paths; corrupted ledger/gradient/boundary-state rejection;
  no material evaluation during construction or restore; reporting and disk
  failures do not contaminate accepted numerical state.
- Ordinary project CLI checks and runs; no project-owned Newton or state loop.
  Real VTK time selection and Warp preserve cells, stress and physical AGE.
- Existing small-strain, finite-J2/contact, project, architecture and output
  contracts. Dispatch uses the declared kinematic convention, not structural
  Python protocol matching alone.
- Shared admission checks reject DG displacement, wrong component counts and
  mixed-field subspaces before assembly; no missing interface/pressure equation
  is hidden by a successful scalar material evaluation.
- Completed-result/reaction publication refuses externally modified accepted
  displacement or material identity. Recursive result publication is refused
  before opening an output writer, so it cannot replace the active view.

A separate disposable installed-use check built a pure-Python wheel with
`AGENTFEM_BUILD_NATIVE=0`, installed it in a new system-site-packages venv sharing
the existing numerical runtime, and copied only the four example files.
Imports resolved to that venv's site-packages with `PYTHONPATH` unset.
Both ordinary CLI runs completed 16 accepted increments on 64 quadrilateral
cells. All 16 frames per run were separately read and warped through VTK,
including checking that AGE matches each physical time. Shared mesh plus
visualization data occupied 603,059 bytes (viscoelastic) and 572,200 bytes
(damage). These are small workflow fixtures, **not performance evidence**.
This is not a fresh dependency installation, Windows acceptance, native
accelerator acceptance or a release-package delivery.

Generated API/knowledge records and correctness lint are checked alongside
the tests. Failed exploratory checks were corrected rather than relabeled:
an early small/finite protocol overlap duplicated rejection diagnostics;
an initial external-reader script incorrectly assumed triangular rather than
the example's quadrilateral mesh; source test subprocesses required a coherent
source import path rather than the older installed package.

## Remaining gates

1. Review/integrate this bounded slice after its predecessor; keep it experimental.
2. Design plane-stress reduction around a locally solved transverse equilibrium,
   effective 3D kinematics and condensed full-update tangent, not matrix truncation.
3. Admit temperature histories, multiple regions and MPI separately, each with
   lifecycle and independent scientific references. Contact composition remains
   a distinct numerical gate, not an automatic benefit of a material interface.
4. Validate real constitutive laws/calibrations with their owners. These synthetic
   examples do not establish hydrogel validity, fracture regularization, a
   creep-fatigue life envelope or general energy closure.

## Next reduction gate: reuse, do not truncate

This was the next gate at the initial workflow submission. Its subsequently
implemented bounded slice is documented in
[the plane-stress record](history_plane_stress_20261029.md); the broader
anisotropic/director scope remains unimplemented.
AgentFEM already has specialized plane-stress StVK and hyperelastic routes in
`constitutive/geometric_elasticity.py` and `constitutive/hyperelasticity.py`.
The missing piece is reduction of a replaceable **history-dependent** material
while preserving its effective 3D kinematics and accepted lifecycle. Existing
specialized routes should remain independent regression references.

For a first bounded planar, reflection-symmetric material class, embed the
in-plane gradient and solve for a positive thickness stretch with `P33=0` at
each quadrature point. Check the remaining transverse traction components;
zero transverse shear cannot be inferred for arbitrary anisotropy. A general
membrane director/three-component transverse solve is a different admission
gate, not an automatic extension of the scalar solve.

At fixed committed state and physical `dt`, differentiate the **whole local
update** and eliminate the transverse unknown through the Schur complement
`A_aa - A_ab solve(A_bb, A_ba)`. Here `a` denotes in-plane gradient components,
and `b` the declared transverse unknowns. Do not slice a 9-by-9 tangent or use
an elastic predictor tangent. A singular transverse block, nonpositive volume
or failed local equilibrium must fail explicitly and permit a global cutback;
it must not be concealed by clipping the stretch or silently relaxing tolerance.

The reduction must participate in the existing material transaction:

- Every thickness/local Newton candidate starts from the same accepted
  material state, effective `F_n` and transverse state; local iterations do
  not accumulate fictitious material history.
- Commit the converged effective 3D gradient and transverse state only with
  the accepted global increment. Restore them together after rejected trials
  and from checkpoints. Do not overwrite user state-variable names.
- Cauchy/Piola conversion, Jacobian and thickness diagnostics use that same
  effective gradient. Passing the plane-strain `F33=1` embedding to the existing
  driver and replacing only its stress would produce inconsistent measures.
- Record reference thickness and volume-versus-area conventions. Require an
  explicit constitutive reduction declaration; do not infer multiaxial response
  from a uniaxial experimental fit.

Acceptance should progress from local traction closure and finite differences
of the reduced response, through specialized StVK/neo-Hookean comparisons and
the small-strain limit, to two history laws on the same structural workflow.
Then repeat hold/reversal, local and global rejection, restart and output tests.
Plane-stress history support remains rejected until those gates pass. Nearly
incompressible mixed formulations and softening regularization are separate
requirements, not benefits obtained automatically through condensation.

Design references:

- [Ahmadi et al., plane-stress hyperelastic finite elements, section 3.1](https://arxiv.org/html/2410.22562v1):
  quadrature-level transverse equilibrium and its consistent condensed tangent.
  Its hyperelastic derivation is a reference for reduction, not validation of
  an arbitrary history-material algorithm.
- [Sandia LAMÉ, plane-stress rate plasticity](https://www.sandia.gov/files/sierra/SM_LAME_5_24/main/models/plane_stress_rate_plasticity.html):
  a specialized plasticity treatment with explicit transverse stress conditions,
  not a generic adapter obtained by truncating a 3D material matrix.
