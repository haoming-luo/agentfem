# Dynamic Cohesive Fracture Architecture

This document fixes the first AgentFEM fracture scope before implementation is
allowed to grow sideways.  The target is a finite-strain, dynamically loaded
solid separated along a declared weak path.  It is not a generic fracture
framework and it is not a phase-field implementation.

## Scientific target and evidence

The first computational target is the fixed weak-interface problem of Wang,
Fineberg, and Needleman (JMPS 203, 106213, 2025).  The later experimental
target is Wang, Shi, and Fineberg, *Science* 381, 415--419 (2023), DOI
`10.1126/science.adg7693`.  Its Dryad dataset
`10.5061/dryad.7wm37pvz6` contains crack-speed, SED/KED, Mach-cone, wave-speed,
and material-response data.  Experimental conditions used for calibration
must be separated from retained prediction conditions.

The JMPS model description establishes the essential numerical problem:

- an isotropic hyperelastic sheet;
- remote impact tension;
- an edge pre-crack;
- a zero-thickness cohesive surface directly ahead of the pre-crack;
- separation traction determined by displacement jump;
- a transition controlled strongly by the cohesive length scale and the
  stored energy in the body.

A mesh alone cannot define this problem.  Reproduction additionally requires
the bulk constitutive equation, cohesive equation and parameters, thickness or
two-dimensional reduction, complete loading history, damping/mass-scaling
choices, stable-step policy, and the crack-tracking algorithm.

## Architectural decision

The feature is split into four independently verifiable consumers:

```text
Finite-strain bulk residual        Paired-facet cohesive residual
          |                                  |
          +---------- force assembly --------+
                             |
          lumped mass + central difference lifecycle
                             |
       accepted state / energy / output / checkpoint
```

The existing `ExplicitDynamicsStep` remains the lifecycle owner.  A new
finite-strain provider supplies a residual recomputed from the current
deformation; a cohesive interface supplies an additional explicit internal
force.  Existing linear explicit dynamics keeps its present provider and
Golden result.  Fracture does not create a second private time loop.

`Study` continues to declare second-order solid dynamics and the dimensional
assumption.  `SolutionProcedure` declares explicit central difference and,
later, its stability policy.  Finite strain is a kinematic/formulation choice;
the cohesive surface is an interface model.  Neither becomes a new overloaded
Study name.

## State ownership

The bulk compressible Neo-Hookean model is stateless.  Every cohesive
integration point owns committed and trial values of at least:

- maximum effective opening;
- irreversible damage;
- dissipated interface energy;
- optional regularization state, only when explicitly requested.

An attempted update is replaceable.  Only an accepted time increment commits
it.  A failed step, rejected stability check, or restart boundary must not
partly advance interface damage.  Checkpoints bind state arrays to a stable
interface-pair identity, quadrature rule, law schema, mesh fingerprint, and
physical time.

`interfaces.BilinearCohesiveLaw` and `interfaces.CohesiveTransaction` are the
first material-point implementation of this contract.  They are marked
experimental until a paired-facet global consumer passes the verification
ladder.

## Interface topology

Ordinary continuous displacement degrees of freedom cannot represent a jump
on a shared facet.  The MVP therefore uses two geometrically coincident sides
with independent displacement degrees of freedom.  The mesh adapter must:

1. split the declared weak path and duplicate its nodes before constructing
   the DOLFINx mesh;
2. retain source-side and source-facet identities;
3. pair negative and positive facets by explicit identity, using geometry only
   as a checked reconstruction aid;
4. establish a deterministic normal and node/quadrature permutation;
5. reject missing, duplicate, ambiguous, or noncoincident partners;
6. work in serial first and then assign each pair one deterministic MPI owner.

Because the split sides are topological exterior facets, a normal DOLFINx
interior-facet `dS` form is not by itself the global cohesive element.  The
first 2D consumer will assemble paired-facet nodal forces explicitly and add
them to the same residual vector as the bulk force.  This also makes equal and
opposite interface force and energy tests direct.  A later mixed-dimensional
or custom-kernel implementation may replace the assembler without changing
the public `CohesiveSurface` contract.

## Bulk dynamics

The first route is a Total-Lagrangian compressible Neo-Hookean body:

\[
 \rho_0 \ddot{u}_i = P_{iJ,J} + \rho_0 b_i,
 \qquad
 P = \frac{\partial\psi}{\partial F}.
\]

At every time increment, bulk internal force is assembled from the active
displacement.  The recoverable bulk energy is the integral of the same
potential used by that residual.  The current quadratic `u.T K u / 2`
diagnostic is valid only for the linear provider and must not be reused for
finite-strain fracture.

Initial acceleration is computed from equilibrium of the transferred state,
not silently set to zero.  Prescribed displacement, velocity, and acceleration
histories must remain kinematically compatible.  The stable increment records
separate estimates from body elements and cohesive stiffness, their safety
factor, and the controlling entity.

## Near incompressibility and two-dimensional reduction

The existing mixed `u-p` Neo-Hookean static formulation is not automatically
an explicit formulation: pressure has no ordinary inertial equation and the
current mixed solve is monolithic.  It must not be attached to central
difference merely because it avoids locking statically.

The V4 mechanism route now uses a local finite-strain plane-stress reduction
at `nu=0.49`. At every material point a positive thickness stretch is solved
such that `P33=0`; the in-plane stress, energy, and Schur-condensed acoustic
tangent all derive from that same condition. This is the correct membrane
reduction for the named two-dimensional route, but it is not by itself a
general proof against in-plane volumetric locking. A trusted broader route
still requires an F-bar/selective-volumetric treatment or a deliberately
designed stabilized explicit `u-p` scheme. Artificially lowering bulk modulus
to obtain the desired crack speed remains forbidden.

The current 2D Neo-Hookean implementation is plane strain.  A thin-sheet claim
requires either:

- a local thickness-stretch solution enforcing the declared finite-strain
  plane-stress condition, with its energy and consistent in-plane response;
  or
- a thin three-dimensional solid model.

The first option is now implemented and locally verified. Thin 3D remains the
required geometry-assumption cross-check.

## Preload and step transition

Preload and dynamics are ordinary ordered analysis steps connected by a typed
state-transfer object, not a special `preload_then_dynamic` solver.  The
transfer records:

- source and destination step identities;
- displacement and mesh/configuration convention;
- transferred constitutive/interface state;
- initialized velocity and acceleration;
- force imbalance and energy before/after transfer.

The first acceptance test is a held prestrained body: after transfer with no
release or impact, it must remain in equilibrium without an artificial wave.
A fully dynamic smooth preload is retained as an independent cross-check.

## Wave speeds

Reference unstretched isotropic speeds are useful V1 checks but are not a
complete supershear classifier.  In a prestrained finite-deformation state,
incremental bulk speeds are direction-dependent and must be obtained from the
instantaneous material tangent/acoustic tensor using a declared reference or
current-configuration convention.

The Rayleigh speed of a prestrained, incrementally anisotropic half-space is
not obtained reliably by inserting one effective modulus into the classical
isotropic formula.  AgentFEM will first validate directional incremental bulk
waves.  Prestrained surface-wave speed requires a separately verified secular
or Stroh solution before it is used as a regime boundary.

## Energy ledger

Every saved history distinguishes:

- bulk recoverable strain energy;
- kinetic energy;
- cohesive recoverable energy;
- cohesive irreversible dissipation;
- external work, including prescribed-motion work;
- declared numerical damping or stabilization dissipation;
- residual balance and normalized balance error.

The first declared damping perturbation is mass proportional,
``f_d = alpha M v_mid``. It is disabled by default. Its accepted nonnegative
work is reported separately as ``numerical_damping_dissipation`` and cannot
disappear inside a generic balance residual.

No unlabelled `total_energy` is sufficient.  Interface dissipation is
integrated from the same traction--separation law that supplies forces.  At
complete bilinear separation its value must equal the declared fracture
energy times reference interface area.

## Crack observations

Crack position is an observation derived from the ordered interface state,
not a new solver unknown.  The first definition uses a declared damage/opening
threshold and spatial interpolation between cohesive points.  Speed is fitted
over a physical time/length window; a single newly failed element is never a
reported instantaneous velocity.  Every history records threshold, window,
mesh spacing, and uncertainty/smoothing policy.

`sub-Rayleigh`, `supershear`, and `spall-like` are classifications with
evidence.  Spall-like separation additionally considers the fraction and time
span of near-simultaneous interface failure; it is not merely a very large
finite-difference crack speed.

## Verification gates

### V0 -- local mathematics

- rigid rotation objectivity of bulk stress/energy;
- finite-strain patch tests;
- uniaxial/biaxial Neo-Hookean paths;
- exact cohesive envelope area;
- loading/unloading/reloading and compression closure;
- cohesive commit/rollback/restart identity;
- pre-crack and intact-interface initialization.

### V1 -- waves

- unstretched longitudinal and shear waves;
- directional incremental waves after homogeneous prestrain;
- mesh, element, and time-step convergence;
- numerical versus acoustic-tensor speed, with an initial target near 2%.

### V2 -- energy

- no-fracture conservation;
- one-interface separation balance;
- convergence of the complete energy ledger.

### V3 -- classical fracture guardrail

- sub-Rayleigh Mode-I propagation;
- no numerical crossing caused by an unstable increment;
- stability across mesh, time-step, and declared damping sweeps.

### V4/V5 -- publications

- JMPS parameter trends and separation-mode transitions first;
- Science Dryad field and history comparisons second;
- calibration cases and retained prediction cases remain distinct.

Until V0--V3 pass, the feature is `experimental`.  `validated` is reserved for
the named benchmark and parameter range whose evidence is published.

## First 30-hour foundation

The weekend foundation is intentionally narrower than a supershear claim:

1. freeze this architecture and a machine-readable capability scope;
2. implement and test one irreversible Mode-I cohesive law and transaction;
3. implement deterministic 2D paired-interface topology and its validation;
4. assemble equal-and-opposite interface nodal forces for one segment and a
   strip of segments;
5. connect current Total-Lagrangian Neo-Hookean internal force to the existing
   explicit lifecycle without changing linear Explicit;
6. replace quadratic finite-strain energy with the constitutive integral and
   establish the typed fracture energy ledger;
7. add a held-prestrain transfer test and a one-interface dynamic opening test;
8. add checkpoint identity for cohesive state;
9. define V1/V3 benchmark specifications and record missing author inputs;
10. only then run the first sub-Rayleigh-to-supershear parameter exploration.

### Implemented in the first foundation slice

- an irreversible bilinear Mode-I law with exact envelope area, unloading,
  reloading, compression closure, precrack initialization, and material-point
  commit/rollback/restart;
- an array-level conforming-interface splitter that duplicates only the
  declared positive side and retains the original-to-duplicate identity;
- deterministic coincident line-facet pairing with explicit orientation and
  rejection of shared, missing, or ambiguous node identities;
- a two-point 2D interface kernel with equal-and-opposite force and integrated
  recoverable/dissipated energy;
- a serial DOLFINx dof adapter and a composite bulk-plus-interface residual;
- an executable split-mesh adapter for triangular and quadrilateral cells,
  with automatic recovery of independent coincident displacement dofs from
  retained DOLFINx input-node identity;
- a Neo-Hookean Total-Lagrangian Explicit provider that recomputes internal
  force from the active deformation and retains the existing linear Explicit
  provider unchanged;
- a visible body/interface stable-increment screening estimate;
- typed bulk/kinetic/cohesive energy channels plus trapezoidal natural-load
  work, strong prescribed-motion work, and absolute/relative balance error;
- amplitude-driven Explicit constraints that impose displacement, midpoint
  velocity, whole-step velocity, and acceleration from one declared history;
- cohesive auxiliary state in the shared transient checkpoint envelope;
- quasi-static displacement to Explicit `u/v/a` transfer with an equilibrium
  guard or an explicitly declared release mode;
- threshold-interpolated crack position, window-fitted crack speed, Mach-angle
  helper, and a separation classifier that requires independent spall
  evidence.
- the analytical Neo-Hookean material tangent and homogeneous small-on-large
  acoustic tensor, including explicit pull-back/push-forward of reference and
  current propagation directions. This supplies the V1 analytical oracle used
  by the numerical arrival-time benchmark below; it does not substitute for a
  prestrained Rayleigh secular solution.

This slice is still `experimental`. The dof adapter is serial-only; interface
mesh splitting is not yet wired to Abaqus/Gmsh import; MPC/contact/weak-constraint
work requires dedicated dual variables; general near-incompressibility and the
prestrained surface-wave secular problem remain implementation gates.

### Automated V1--V3 guardrails

The first named verification ladder is now executable:

- **V1:** a longitudinal wave packet is propagated by finite-strain Explicit
  at zero and ten-percent held prestrain. The measured reference-coordinate
  speed converges toward the material acoustic-tensor prediction and the
  80-cell errors are below two percent;
- **V2:** no-fracture wave energy improves under spatial refinement. A smooth
  prescribed separation then drives one interface to complete failure; its
  dissipation equals the declared ``Gamma`` and the final balance error
  converges below ``1e-5``;
- **V3:** a precracked long interface advances multiple facets. Its
  threshold-interpolated, seven-frame fitted speed remains below ``0.8 c_R``
  under 40-to-60-facet refinement, explicit time-step refinement, and a small
  declared mass-proportional damping perturbation. Damping work is a typed
  energy channel and final balance errors remain below ``5e-4``.

This remains V3 for the named compressible plane-strain cohesive strip, not a
general dynamic-fracture validation. Supershear claims are evaluated by the
separate V4 mechanism gate below.

### Executable V4 mechanism gate

`benchmarks.jmps_weak_interface_transition_v4()` now runs three cases through
one public finite-strain Explicit lifecycle:

1. a homogeneous 12% plane-stress preload followed by precrack release gives
   a contiguous crack-like front at approximately `0.96 c_R`;
2. the same body and interface with a zero-slope smooth remote impact gives a
   resolved front at approximately `1.10 c_s` and `0.56 c_d`, while no more
   than 4% of the ligament fails in one increment;
3. a weaker, larger-cohesive-length interface under the same impact fails
   across the ligament within one thickness shear-wave time and is classified
   `spall_like`, rather than assigning physical meaning to its super-dilatational
   apparent front speed.

Every case transfers preload without an energy jump and closes the complete
energy ledger within 0.5% at the declared 30-facet mechanism resolution. The
crack-speed fit window spans at least three interface cells at the prestrained
shear-wave speed. The classifier also rejects an apparent speed above `c_d`
as `unresolved_discrete_failure` unless independent distributed-spall evidence
is present.

This is an **experimental V4 numerical mechanism benchmark**. It establishes
the crack-like/supershear/spall distinction in the intended architecture; it
does not claim curve-level reproduction of Wang, Fineberg, and Needleman's
unpublished input deck. Promotion to publication reproduction still requires
the authors' dimensions, exact cohesive law and parameters, impact history,
mesh sequence, and post-processing convention, followed by mesh/time-step
convergence and a thin-3D check. The immutable evidence and tolerances live in
`knowledge/benchmarks/jmps_weak_interface_transition_v4.json`.

## Inputs to request from the authors

Priority order:

1. complete computational input/source and postprocessing scripts;
2. exact bulk and cohesive equations with every parameter;
3. dimensions, thickness, pre-crack, and weak-interface construction;
4. preload/impact histories and the applied boundary convention;
5. plane-stress, plane-strain, shell, or 3D assumption;
6. element family, mesh sequence, and smallest interface spacing;
7. crack position/speed and field-processing definitions;
8. damping, mass scaling, stable-time-step factor, and boundary treatment.
