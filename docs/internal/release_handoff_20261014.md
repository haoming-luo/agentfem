# Release handoff — numerical policies, not frozen case machinery

This records the 2026-10-14 user acceptance requirement. It is not a release
announcement. A recipient must execute the reviewed project with ordinary
`agentfem run`, without an AI agent or access to the developer's temporary
workspace. Runtime speed may depend on hardware; scientific reproducibility is
assessed with declared tolerances, not universal bitwise equality.

## Storage closure

The private workspace cleanup retained one complete terminal visualization,
its HDF5 fields, accepted material arrays, the final checkpoint, compact history
and three mid/late body replay stations. Obsolete large trial artifacts and
ZIPs were explicitly inventoried and removed. Historical result manifests were
not rewritten to disguise missing old checkpoint links. No private data or
absolute private path belongs in the public repository.

Use the existing `checkpointing.every(..., keep_last=2)` for the next candidate
unless a particular validation run explicitly needs more stations. Keep a
single canonical complete result. Do not repeatedly copy a frozen environment
or ZIP a candidate after every small edit.

## First implementation: numerical mass policy

`model.step(mass_scaling=...)` now accepts `time.FixedMassScaling` and
`time.TargetIncrementMassScaling` on the bounded serial response-Hex provider.
The latter computes nodal factors in software from named, unscaled stiffness
and damping majorants supplied by a deterministic provider. Missing or extra
operator roles fail closed; active body, hourglass, bonds, contact pairs and
bulk viscosity are inventoried by the Step. An explicit added-mass budget is
mandatory. Physical material density is never overwritten.

Fixed planning and periodic accepted-state updates reuse existing numerical
inertia State, energy/momentum accounting, macro-step rollback and restart.
Cadence skips bound evaluation and nodal copies on intervening increments.
Reference geometry/mass buffers are immutable and reused. A declaration is
not a generic nonlinear stable-step estimator. A callback returning plausible
positive arrays is not proof of its bound, and a successful scaled run is not
evidence of quasi-static accuracy.

The existing retained private project is deliberately still the frozen
experiment. It has NOT been silently migrated from its reviewed nodal factors
to a new mass distribution. Its complete-run timing is not a timing of the new
policy. A reproducible software policy removes the need for a precomputed NPZ,
but the current private law/contact combination still needs its appropriate
reviewed bound provider and sensitivity acceptance before that migration.

## Required before promoting the new recipient project

1. Supply reusable Operator/material bound providers for the *admitted*
   source-law/contact combination. Derive mass from each new mesh, density,
   declared law bounds and numerical controls. Verify coverage, domains and
   negative-curvature limitations; do not rename the old NPZ loader into a
   new public function and call that automation. Retain the old experiment
   only as a regression reference.
2. Separate cohesive, rigid-contact and deformable-contact enforcement
   parameters. Record overlap and initial-adjustment policy. Unsupported
   ALL EXTERIOR/self-contact remains a documented limit, not an implicit claim
   or an unnecessary rewrite of the entire contact engine for this release.
3. Use ordinary Step completion and one-grid output. The response provider now
   accepts an XDMF output path and ordinary accepted-history declarations;
   material state columns still require explicit semantic selection. Dyadic
   substeps now admit pure configured observations at accepted macro stations,
   while retaining per-subincrement physical work accounting.
4. Keep a compiled user law separate from general software execution. Existing
   ExplicitUserMaterial is the neutral batch/state mechanism. A restricted
   VUMAT adapter is a separate compatibility task: shear order, material frame,
   state/parameter sizes, energy units, source/ABI identity, errors and compiler
   portability must be tested. Arbitrary UMAT/VUMAT drop-in support is not a
   release claim of the current interface.
5. Freeze one source-built candidate. Check fresh `agentfem run`, missing input
   errors, atomic failure, restart, bounded checkpoint retention and independent
   ParaView volume/Warp reading. Measure the complete run only after short,
   mid/late and sensitivity gates pass. Do not make machine-specific timing
   thresholds a compatibility requirement. Windows collaborator acceptance is
   separate from the local macOS evidence.

## Local candidate acceptance, 2026-10-14

- Source-tree focused checks: 233 passed, including policy validation,
  operator inventory, failure rollback, cadence-aware checkpoint/restart,
  numerical energy accounting, history, retention and presentation guards.
- Source-built native macOS arm64 wheel: 78 focused checks passed from its
  independent installation, with no source tree on the import path. The
  installed mass-policy module was byte-compared to the candidate source.
- A small standalone elastic/tool project ran twice using ordinary
  `agentfem run`. All 26 HDF5 datasets agreed exactly on this machine. Both
  runs completed 100 increments and kept only the last two checkpoints.
  This demonstrates repeatable software execution, not universal bitwise
  reproducibility or validation of the private composite problem.
- Independent ParaView 6.1.1 read the final saved frame: one grid, eight
  volumetric Hex8 cells, complete point displacement, nonzero deformation,
  zero coordinate error against reference coordinates plus displacement.
  The reader checker now admits `--frame last`; checking only a zero initial
  frame is insufficient evidence of deformed-output usability.
- The retained full composite result was also independently read: one grid,
  255,041 volumetric Hex8 cells and 296,074 points. Its existing full-run
  timing is 14,646.99 seconds (about 4 h 4 min), from the earlier frozen
  experiment, **not** this candidate. No new full-run acceleration or
  physical equivalence to Abaqus is claimed here.

These checks do not close the original-case provider, sensitivity and
complete installed-run gates above. This remains an unreleased candidate.

## Scientific reference

### Recipient source package and accepted observations

The recipient candidate is a source-built, independently installed project:
one readable physical case, operational TOML, private user law and one-time
local preparation. No developer checkout or saved inertia array is required.
Shared trace construction now uses tagged boundaries and input-node identities;
the private source labels and initial-adjustment hypothesis remain in the case.
Material export consumes the named state schema and the ordinary one-grid writer.
Accepted interface observations report current-coordinate force/moment balance,
reference-frame opening/slip and the pairing's integration measure. No overlap
or nonlinear stability claim is inferred from those diagnostic quantities.

The latest observation boundary passed 130 focused real-FEniCSx tests, including
triangular/Q1 pairing, no-bond behavior, immutable observation, trial rejection,
checkpoint recovery, contact composition, mass preparation and substeps.
The earlier installed candidate passed 151 focused tests, plus 38 CLI/history
checks. A two-rank warped-Q1 boundary-extraction check passed; it does not certify
MPI response-material execution. A new source freeze still needs its own
installed original-mesh entrypoint/export/recovery check before handoff.

The original-mesh time-step comparison is explicitly a partial loaded-path
screen. It must retain failed channels and cannot certify either full-path
convergence or quasi-static accuracy. The old completed reference remains
identified separately from this candidate and its new initial mass distribution.

### Frozen recipient candidate: installed entrypoint and recovery

The source-built `016e665d` candidate passed 144 focused installed response,
interface, mass and state checks, 103 installed native/output checks and 78
installed CLI/project checks. These suites overlap; their counts are not a
claim of that many distinct new scientific benchmarks.

The original mesh ran through ordinary `agentfem run` from outside the project
directory. A 600-increment pause followed by recovery to 1,200 increments
matches a fresh 1,200-increment run exactly across all 385 compared accepted
state/metadata leaves. The actual ParaView reader sees one three-dimensional
Hex8 grid and a complete nonzero point displacement; Warp agrees with reference
coordinates plus displacement. The confidential source law is built locally,
and its named case parameters are checked against the supplied material data.
Independent virgin and retained-damaged material-point replay passes the
declared numerical tolerances; it is not a global vendor comparison.

The recipient package keeps numerical mass preparation in AgentFEM and contains
no required inertia-array input. It includes the pinned core source archive,
one-time local build instructions, case and operational TOML, private material,
compact acceptance and a clearly labelled earlier full reference. A different
platform rebuilds its native artifacts; a Mac binary is not a Windows artifact.

The loaded time-grid screen and a complete run of this new source freeze remain
separate gates. At the 0.02-second prefix of the initial-mass experiment,
numerical kinetic energy is about 55% of accumulated external work. Its small
discrete energy-ledger residual does not establish quasi-static accuracy.
Do not promote this candidate to a validated reproduction or transfer the
retained full run's performance claim to it.

### Initial preparation implemented, original-case acceptance still pending

The bounded response-Hex provider now also admits `time.InitialMassScaling`.
It reuses the current element/reference geometry, elastic bond assembler,
filtered rigid traces, deformable candidate neighborhoods and linear viscosity.
The material supplies an explicit reference modulus, not an invented tangent.
No nodal-factor file is needed. Shared candidate-patch vertices are deduplicated;
the frozen-Q1 diagonal estimate is tested against independently assembled
stencils. Mass-budget rejection, deterministic factor execution, checkpoint
identity and operator inventory have targeted numerical regressions.

This deliberately does not promote initial elastic data to nonlinear-path
certification. Moving projection derivatives, future contact neighborhoods,
damage and curved negative stiffness remain outside its estimate. A new
recipient project must record and compare the generated mass, then test
the response; it cannot inherit the old full run's timing or physical claims.

The source implementation passed 109 targeted checks and the independent
source-built native installation passed 76 related checks. The original
255,041-cell project now prepares through ordinary `agentfem run` without an
inertia-file input. Preparation took 21.4 seconds on the development host;
this is not a transferable performance promise. The new reference estimate
changes the mass distribution (total numerical mass about 2.2% below the
retained experiment), so old results are not relabelled as new acceptance.

Fresh continuous and interrupted/resumed runs at 1,200 increments agreed
exactly on all 385 checked accepted-state/metadata leaves, including material
and contact state. This is same-installation recovery evidence, not a promise
of bitwise agreement across compilers or hardware.

At the initial physical station 0.0006 s, halving the time increment while
holding generated mass fixed gave relative L2 changes of about 0.020% in
displacement, 0.58% in stress and 0.0061% in the moving-tool force histories.
Acceleration changed about 20.4%; the predeclared all-channel 2% screening
therefore **failed**. Do not weaken that recorded check or describe the path
as time-converged. This prefix does not exercise the full damage evolution.

A separate same-time-grid experiment reduced the generated added-mass ratio
from 750.74 to 187.78, without changing density, material law or the prescribed
motion. At this same initial station the stress and moving-tool force histories
changed by about 56% and 46% respectively (maximum absolute stress change
0.00562 and resultant change 0.128 in the declared model units). This also
fails the screening, and is not evidence that mass scaling is negligible.
Early forces are small and damage is not yet exercised: retain both absolute
and relative changes, then assess representative loaded stations before any
quasi-static or whole-path claim. Full installed-run acceptance remains open.
The core implementation is a candidate, not an announcement of completed
reproduction.

[Abaqus mass-scaling documentation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-massscaling.htm)
distinguishes fixed and variable scaling, and requires checking inertial effects.
AgentFEM keeps the same separation between a declared numerical strategy and
material density, without claiming equivalent vendor algorithms. Contact
stiffness is part of the combined demand; a body-only estimate is not a
whole-model stable-increment certificate.
