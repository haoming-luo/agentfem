# Bounded history-material plane stress

## Decision and ownership

Extend the material-only workflow, not the material law or a project solver.
`Study` declares plane stress, `PlaneStressReduction` declares reflection
symmetry and local controls, the neutral provider returns a full 3D response,
and the existing Procedure/State owns global acceptance and restart.

The scalar unknown is positive thickness stretch, represented by `log(F33)`.
The local residual is P33 and its logarithmic derivative is A3333 times F33.
All local candidates reuse the same accepted F/state and physical time step.
The accepted full material response remains unchanged. A separate 4×4 Schur
complement supplies in-plane assembly; it is not relabeled as a full dP/dF.
Cauchy-to-Piola conversion uses the effective 3D F and determinant.

Accepted F33 is stored alongside F and response arrays in the existing
transaction and checkpoint. Loading a checkpoint does not call the material.
The effective gradient is also a raw quadrature result and a recovered
`F_CELL` visualization field. No project-managed thickness cache is required.

## Admission and exclusions

- Serial homogeneous 2D, strong constraints and reference dead loads, as in
  the parent workflow; no new contact/MPC/MPI/thermal-history admission.
- Explicit reflection-symmetric 3D material and compatible history, planar F,
  checked transverse shear/coupling and positive transverse stiffness.
- Unit reference thickness; no shell/director, general anisotropic reduction,
  mixed incompressibility, automatic multiaxial inference or localization fix.
- The first reduction uses scalar local updates inside the bounded quadrature
  driver. It is not a new array-batch optimization or a performance promise.
- Local relative tolerance scales with physical stress/stiffness, without a
  unit-sized floor. Absolute tolerance has stress units. Both enter identity.
- Existing specialized StVK/Neo-Hookean reductions remain independent routes.

## Independent checks

`tests/test_history_plane_stress.py` includes:

1. Closed Green-strain transverse solutions through loading, hold and reversal
   for two educational history laws; original request history remains intact.
2. Condensed algorithmic tangent versus independently perturbed reduced
   responses, on both smooth damage/history branches.
3. Separate specialized Neo-Hookean thickness/stress solution; actual 3D
   Cauchy/Piola conversion and stress-unit rescaling.
4. Positive-thickness local safeguarding; incompatible shear/coupling,
   singular stiffness and local nonconvergence rejected explicitly.
5. Degree-one/two structural patches and nonuniform assembled tangent checks.
6. Traction-controlled structure versus independently solved scalar equilibrium
   and analytical history recurrences, not the reduction's own output as oracle.
7. Nonuniform damage manufacture: u=(a*x*x,0), analytically differentiated
   reference body force, quadratic transverse Green-strain elimination;
   linear-element spatial refinement and quadratic-element exactness.
8. Viscous time refinement against independently integrated continuous-time
   coupled transverse/history ODEs, rather than a finer call to the same update.
9. Failed trials, material-requested cutback, continuous/restarted equality,
   identity changes and corrupt effective-gradient atomic rejection.
10. Real VTK time selection and Warp: one grid retains stress, state and
    thickness; the ordinary four-file project tests both material choices and
    both plane assumptions through `agentfem check/run`.
11. Interrupted field output is readable before restarting. A zero-copy joined
    continuation view matches every saved field at every accepted time of an
    uninterrupted run for both educational laws.
12. Built-in finite-strain J2 local reduction agrees with an independent
    uniaxial logarithmic return through loading, hold, unloading and reverse
    plastic flow, including Fp and accumulated plastic strain. This local
    check does not change its specialized global Step admission.

The public `check_plane_stress_tangent` reports the separate reduced 4×4
derivative, component order and reduction policy. Wrong derivatives are
detected; old history is not mutated. It supplements, not replaces, the full
3D `check_material_tangent` and independent material validation.

Extreme stress-unit tests exposed an avoidable numerical issue in the first
Schur implementation: forming the product of two stiffnesses before dividing
could overflow or silently underflow. The implementation now forms the
dimensionless ratio first and evaluates dimensional norms after normalization.
Equivalent scales from 1e-200 through 1e200 preserve the reduced response.
This is a representability regression test, not a useful physical unit range
or a change in constitutive equations or numerical tolerances.

These laws are synthetic educational fixtures, not hydrogel calibration or
general fracture/thermodynamic verification. The damage tangent must be tested
away from its nonsmooth maximum-history switch; a centered difference across
that switch is not either smooth branch's derivative. The test explicitly
asserts which branch it occupies instead of weakening tolerances.

## Integration regression caught by full CI

Parent PR #134's initial Linux full test run completed with 5413 passed,
428 skipped and one failure. The generic provider could catch a built-in J2
request after its specialized provider rejected an unsupported linear-solver
policy, producing a wrong option error. Commit `295bfa80` excludes that built-in
material from generic fallback; 93 targeted tests passed and one MPI-only
test was skipped. The fix was pushed to #134 independently of this reduction.
The failed full run is not relabeled as a passing release gate.

The subsequent parent run `40844625630` at `295bfa80` completed successfully,
including FEniCSx and macOS installed-wheel acceptance. That result applies to
the parent workflow, not automatically to this stacked plane-stress change.

## Accepted-state ownership audit

After the initial reduction commit, eight new failing regressions exposed two
ownership gaps in the shared history Step: modified live increment/constraint
targets could retain an obsolete checkpoint identity, and changes to accepted
material fields or a final observer's displacement could escape continuation
checks. The fix compares the existing input identity and uses a compact hash of
the contiguous accepted material/response buffers at transaction boundaries.
It does not call the material, retain a second field snapshot, or add checks
inside the Newton/material-point loop. Time and live loading values are checked
alongside the fields. Rejected external mutations cannot be published or saved
as accepted data; restoring into a fresh Step remains available.

Thirteen targeted ownership cases cover policy/target changes, history,
stress/tangent, time/loading/boundary edits, final observers, reporting before
publication, and failed-restore laundering. The guard detects accidental live
edits, not malicious archive modification; no new cryptographic archive promise
or performance benefit is claimed. Hashing remains proportional to accepted
field bytes and occurs only at these lifecycle boundaries.

## Installed-use and output evidence

The initial targeted source selection passed **418 tests**; after the ownership
audit the same selection passed **431 tests**, with **4 skipped**
because their tests require two MPI ranks. It ran in the real `fenicsx-env`
with `PYTHONPATH=src` and OpenMP/OpenBLAS threads limited to one:

```sh
python -m pytest -q \
  tests/test_history_plane_stress.py tests/test_finite_strain_material_step.py \
  tests/test_affine_adaptive_preconditioner.py tests/test_user_material.py \
  tests/test_material_batch_evidence.py tests/test_finite_strain_j2_material_map.py \
  tests/test_documentation.py tests/test_architecture_contract.py \
  tests/test_constitutive_models.py tests/test_extension_upgrade_context.py \
  tests/test_live_xdmf.py tests/test_xdmf_layout.py tests/test_common_workflows.py \
  tests/test_quadrature_physical_recovery.py tests/test_quadrature_snapshot_ownership.py
```

Correctness lint, generated API/knowledge checks and strict documentation
build also passed. Existing unreduced checkpoint identity/summary shape is
preserved; only the new reduction has its additional identity and schema.
This is targeted evidence, not a full-repository or MPI release certificate.

A disposable pure-Python wheel was installed into a separate system-site-
packages venv sharing the already available numerical runtime. With no source
PYTHONPATH, imports resolved to that venv. The ordinary four-file project
completed 16 accepted increments on 64 quadrilateral cells for each plane-
stress material choice. All 32 frames were independently read and warped
through VTK, checking Cauchy S33, positive thickness, physical AGE and retained
cell fields. Visualization occupied 696,475 bytes for the viscous law and
686,479 bytes for damage. These small fixtures are workflow checks, not
performance measurements or Windows/fresh-dependency installation evidence.

The project requires both `mechanics.FiniteStrainMaterialStep` and
`constitutive.PlaneStressReduction` by API identity. A matching development
version string alone is not enough. No release installer or public binary
distribution is produced by this check.

## References

- [Ahmadi et al., plane stress in finite deformation hyperelasticity](https://arxiv.org/html/2410.22562v1): local transverse equilibrium and consistent condensation.
- [Sierra/SM plane stress rate-dependent plasticity](https://www.sandia.gov/files/sierra/SM_LAME_5_24/main/models/plane_stress_rate_plasticity.html): specialized constitutive reduction with transverse conditions; not a generic history-provider validation.
- Existing AgentFEM neutral `UserMaterial`, quadrature transactions and shared
  Newton attempt remain the implementation contracts; no new solver runtime.

## Release boundary

This is an experimental source change stacked on the material-only workflow,
not a new released version. Wider scientific validation and separate
composition gates remain on the product roadmap. This work makes no claim
about composite full-run speed, Windows native acceptance or arbitrary UMATs.
