# Cyclic Cohesive Fatigue Architecture

## Scope

AgentFEM's first fatigue-crack-growth route is a declared zero-thickness
cohesive surface subjected to cyclic loading. It is not the existing S--N
fatigue postprocessor under a new name, and it is not yet a free-path crack
method. The initial research target is one or two surface cracks in a
three-dimensional cylindrical specimen under axial force cycles.

The product capability is deliberately more general than that specimen:

```text
named cyclic load
      |
peak/valley equilibrium solves
      |
cyclic cohesive material transaction
      |
adaptive cycle block (begin / commit / rollback)
      |
named-interface state, energy, checkpoint and 3D observations
```

Geometry generation, CT registration and a particular cylinder mesh are
research assets built on this contract. They do not define the solver API.

Two deliberately separate reference laws implement this lifecycle. The
historical Mode-I law evolves from positive normal-opening extrema. The
experimental mixed-mode law consumes either the complete local vectors at an
accepted proportional valley/peak pair or an explicitly ordered closed path,
separates local cohesive \(G_I\) and \(G_{II}\) channels, and applies a declared
BK or power-law interaction. No driver is silently selected from observed
kinematics: mixed-mode cycling requires a `MixedModeBilinearCohesiveLaw` and
either `MixedModeEnergyRangeDriver` or
`OrderedMixedModeEnergyPathDriver`.

Multi-interface projects should run two preflights before cycle execution:

1. `audit_split_interface_rigid_modes(...)` on the split topology and declared
   strong constraints;
2. `cohesive.audit_mode_i(...)` after the initial elastic peak/valley pair.

The first catches disconnected-body mechanisms. The second catches a model
that was declared Mode-I but develops excessive tangential jump. Neither
check repairs a physically incomplete model by adding a hidden point support.

## Independent coordinates

Physical time, cycle count and output frame are different quantities:

- physical time resolves inertia, frequency and waveform when those effects
  matter;
- cycle count advances fatigue state and is the independent coordinate of a
  cycle-jump calculation;
- output frames are requested observations and never determine state
  evolution.

`procedures.cyclic_fatigue()` therefore declares `control="cycle_increments"`
on a Standard, stateful, quasi-static peak/valley procedure. It does not reuse
Explicit time merely to count cycles. `fatigue_fracture.ForceCycle` can still
produce a physical-time amplitude when the waveform must be resolved. Sine,
triangle and experiment-supplied tabular cycles are supported; sine/triangle
cycles may declare peak and valley hold fractions, while tabular holds are
represented explicitly by repeated values at distinct phases.

For a physically resolved cycle, the existing engineering load path remains
the consumer:

```python
cycle = fatigue_fracture.force_cycle(
    fmin=226.0,
    fmax=2262.0,
    frequency=frequency,
)

model.distributing_coupling(
    force=(0.0, 0.0, cycle.maximum),
    on=regions.loaded_end,
    reference_point=loading_point,
    amplitude=cycle.normalized_amplitude(),
)
```

The quasi-static global cycle controller consumes the same declared minimum
and maximum directly, without resolving millions of waveform periods.

```python
step = fatigue_fracture.global_cyclic_fatigue_step(
    cycle=cycle,
    stop_cycle=1_000_000,
    interfaces=cohesive_interfaces,
    state=fatigue_fracture.field_state(displacement=displacement),
    solve_equilibrium=solve_peak_or_valley,
    jump=fatigue_fracture.CycleJumpPolicy(
        maximum_damage_increment=0.01,
    ),
    landing_cycles=(1, 10, 100, 1_000, 10_000),
)
step.run()
```

The callback remains an explicit and replaceable integration boundary. For the
native finite-strain route, `fracture.FiniteStrainCohesiveEquilibrium` now
assembles the UFL bulk residual/tangent and the paired-facet cohesive
force/algorithmic tangent into one PETSc Newton system. It accepts a physical
scalar load setter and may return strong-constraint reaction, control
displacement and bulk-energy evidence through reusable callbacks. Custom
equilibrium providers remain valid for other constitutive or constraint
formulations; the cycle controller never replaces or approximates the global
solve.

```python
equilibrium = fracture.FiniteStrainCohesiveEquilibrium(
    residual=bulk_plus_interfaces,
    tangent=bulk_tangent,
    displacement=displacement,
    load_parameter=axial_force_parameter,
    bcs=strong_constraints,
    solver_options=solvers.newton(),
    reaction=axial_reaction,
    control_displacement=loaded_end_displacement,
)
```

The interface tangent is integrated from the constitutive `dt/ddelta` at each
quadrature point. Strong constraints are eliminated only after the bulk and
cohesive matrices have been combined, including couplings between coincident
but independent trace nodes. The same kernel is covered by 2D/3D directional-
derivative tests, serial PETSc assembly and a sparse-owner two-rank MPI test.

## Constitutive protocol

`CyclicCohesiveLaw` composes, rather than replaces, a monotonic
`BilinearCohesiveLaw`. Without a cycle-state advance it returns exactly the
wrapped monotonic traction, tangent, damage and dissipation. The reference
fatigue evolution uses the positive local opening extrema:

\[
R_\delta = \frac{\delta_{\min}^{+}}{\delta_{\max}^{+}},\qquad
\Delta \bar\delta =
\frac{\delta_{\max}^{+}-\delta_{\min}^{+}}{\delta_f},
\]

\[
\frac{\mathrm d D_f}{\mathrm d N}
= C
\left\langle
\frac{\Delta\bar\delta-\Delta\bar\delta_{\mathrm{th}}}
{1-\Delta\bar\delta_{\mathrm{th}}}
\right\rangle^{m}
\left(\frac{\delta_{\max}^{+}}{\delta_f}\right)^q
(1-D_f)^p.
\]

The residual term is part of the rate, not a separate additive source. Local
opening extrema, rather than one globally imposed load ratio, allow shielding
and closure to alter the driving cycle at each interface point.

This power-law range model is an experimental reference implementation. It is
not asserted to be universal for metals, elastomers, adhesives or composites.
Alternative fatigue laws should retain the same public transaction:

- evaluate monotonic equilibrium trials;
- begin one cycle block from accepted peak/valley states;
- commit or rollback all history atomically;
- export every state array for restart;
- report threshold, load-ratio convention, evolution variable, monotonic
  limit and maturity.

The state separates maximum monotonic opening, fatigue damage, last positive
opening extrema, accumulated cycles and fatigue dissipation. Compression uses
the monotonic closure penalty and neither heals fatigue damage nor creates
fatigue dissipation.

### Mixed-mode energy-range driver

The mixed-mode transaction stores the two *actual* local peak/valley jump
vectors, not independent componentwise minima and maxima. With local normal
component \(\delta_n\) and tangential vector \(\boldsymbol\delta_t\), its first
reference driver uses

\[
G_I^{\mathrm{coh}}=\frac12 K_n\langle\delta_n\rangle^2,
\qquad
G_{II}^{\mathrm{coh}}=\frac12 K_t\|\boldsymbol\delta_t\|^2,
\]

\[
\Delta G_I^{\mathrm{coh}}=G_{I,\max}^{\mathrm{coh}}
-G_{I,\min}^{\mathrm{coh}},\qquad
\Delta G_{II}^{\mathrm{coh}}=G_{II,\max}^{\mathrm{coh}}
-G_{II,\min}^{\mathrm{coh}}.
\]

These are work-conjugate **local cohesive-law drivers**. The `COH` qualifier
is retained in output names because they are not structural energy-release
rates obtained by a J-integral or VCCT. The range mode fraction is

\[
\beta=\frac{\Delta G_{II}^{\mathrm{coh}}}
{\Delta G_I^{\mathrm{coh}}+\Delta G_{II}^{\mathrm{coh}}},
\]

and, for the BK option,

\[
G_c(\beta)=G_{Ic}+(G_{IIc}-G_{Ic})\beta^\eta.
\]

Pure-mode threshold fractions are mixed by the same declared interaction.
The normalized range above that threshold drives the same exact residual-
power cycle integration used by the Mode-I transaction. The public result
contract exposes `JUMP_MIN_LOCAL`, `JUMP_MAX_LOCAL`, `GI_COH_RANGE`,
`GII_COH_RANGE`, `G_COH_CRITICAL`, `G_COH_THRESHOLD`,
`G_COH_RANGE_NORM`, local load ratio, cycle count and separated monotonic and
fatigue dissipation.

A peak/valley pair is sufficient only for proportional or near-proportional
cycles. The extrema driver therefore checks mode-mix change and tangential-
direction reversal and rejects an apparent non-proportional path. For a
resolved non-proportional waveform, `OrderedJumpCyclePath` carries strictly
increasing phase stations from zero to one, complete local jump vectors at
every station and an exactly closed final state. The ordered driver retains
the positive variation of each cohesive energy channel on every segment,
evaluates the active variation with its own mode mix and sums the resulting
fatigue measure. Thus a Mode-I-to-Mode-II transfer at constant total elastic
energy is not silently erased. It records path length, reversal count and
station count. Decreasing portions of a channel do not generate a second
fictitious damage increment.

This is intentionally not rainflow reconstruction. The solver must provide
the within-cycle stations that matter; AgentFEM does not invent a path from
two extrema. A fixed supplied path retains exact one-cycle/cycle-jump
equivalence under the reference rate integration, and the complete path state
participates in rollback and restart.

## Generalized work and cycle-block energy

`CyclicWorkEnergyLedger` closes the cycle transaction over named generalized
force--motion pairs. A channel can represent a natural load, reference-point
force/moment, prescribed motion, MPC reaction, weak constraint or contact
constraint. `reference_point_work_sample(...)` combines translation/force and
rotation/moment in one work-conjugate vector. Force signs must follow the
declared convention of work done on the model.

For resolved stations, each channel uses trapezoidal work

\[
\Delta W_q = \tfrac12(\mathbf Q_i+\mathbf Q_{i+1})\cdot
(\mathbf q_{i+1}-\mathbf q_i).
\]

The block separately receives total material-energy channels at its accepted
start and trial end states, such as recoverable
bulk/interface energy, kinetic energy and monotonic/fatigue/cohesive/numerical
dissipation. Their trial final-minus-initial change is compared with external
work. Post-damage verification points are not multiplied as though they were
additional physical cycles. For a cycle jump, work of the solved
representative closed cycle is multiplied by the accepted integer block and
explicitly labelled an estimate; all-cycle material dissipation may still be
supplied by the exact trial state change. The ledger is committed only when fields, interfaces,
post-damage equilibrium and the cycle ledger are all accepted. A failed
energy check rolls everything back together.

The ledger is a complete public accounting contract, not a claim that every
solver provider already extracts every reaction automatically. Native and
custom providers expose their actual reference-point, prescribed-motion,
MPC, weak/contact reaction channels through `CyclicEquilibriumPoint`; missing
channels cannot be replaced by a guessed force.

## Cycle jump

`CycleJumpPolicy` bounds a proposed integer block by declared maximum damage
and crack-front increments and never steps over a required output/checkpoint
cycle. Its decision records start/end cycle, reason, controlling rates,
predicted increments and exact landing target. `CycleJumpLedger` preserves
every accepted or rejected proposal, its error estimate and cutback message,
and is itself restartable.

The policy is a proposal, not a global error estimator.
`GlobalCyclicFatigueStep` now supplies the structural lifecycle:

1. solve the accepted extrema or every declared ordered-path station;
2. predict a cycle block from current material and optional front rates;
3. begin the complete block transaction;
4. re-solve the degraded maximum and closing states;
5. compare the pre- and post-degradation peak opening and accept only if
   damage, structural-feedback and optional energy errors lie within tolerance;
6. otherwise rollback and cut back the cycle block.

At constant extrema the reference material-point rate is integrated
analytically, so one 100-cycle material update equals 100 one-cycle updates.
That property is necessary but does not prove structural cycle-jump accuracy:
the extrema change as a crack grows.

## Multiple interfaces and restart

`fracture.CohesiveForceCollection` composes independently named cohesive
forces and can be consumed by the existing finite-strain cohesive residual and
energy monitor. Each name retains its own topology, normal, material,
precrack, response and restart record. Collection begin/commit/rollback is
atomic.

`interfaces.split_conforming_named_interfaces(...)` performs one audited split
for several disjoint two- or three-dimensional manifolds. All named surfaces
share one solver mesh but receive independent duplicated nodes, physical facet
identity, material and state. Ambiguous interfaces that share source nodes are
rejected until an explicit cohesive-junction topology is available.

Portable cohesive checkpoint schema v2 stores every declared state field by
ordered physical facet key and quadrature point. It remains independent of MPI
rank and DOF numbering. The reader retains compatibility with the monotonic
schema v1. This includes every component of mixed-mode peak/valley jumps and
all fatigue-driver evidence. An automated acceptance writes overlapping
physical facets on two MPI ranks and restores them in a different order on one
rank.

The global controller additionally checkpoints the accepted cycle ledger,
named interfaces and bulk field shards. Bulk fields currently require the same
MPI partition and rank count; physical-facet interface state remains portable
across partitioning. This difference is recorded rather than hidden.

## Three-dimensional observations

`observe_surface_crack(...)` works on a triangular cohesive surface. It
reports failed area, connected components, maximum and area-weighted mean COD,
and the embedded crack front formed by edges shared by one failed and one
intact facet. Surface boundary edges are excluded by default so the crack mouth
is not counted as the propagating front.

`surface_crack_interaction(...)` reports the minimum distance between two
named fronts and explicit growth-rate ratios relative to a single-crack
baseline. A ratio below one is shielding and a ratio above one is
amplification. This makes the scientific rule visible: calibrate on the single
crack, then predict the double crack without refitting.

`SurfaceCrackTracker` consumes stable physical facet keys and gives every
connected component a persistent identity. One-to-one growth retains that
identity; birth, death, merge and split are explicit topology events. A merge
creates a new identity with all parent IDs instead of arbitrarily selecting
one old crack as the survivor. The tracker is restartable, and one tracking
frame computes all same-surface pair ligaments without asking a user to split
the observation manually. MPI use requires globally stable facet keys rather
than rank-local indices.

`paris_evidence(...)` is deliberately a postprocessor, not a crack-growth
law. It differentiates an accepted crack-size history, records the declared
fit mask or cycle interval and fits

\[
\frac{\mathrm da}{\mathrm dN}=C\,\mathcal D^m,
\]

where the caller declares whether \(\mathcal D\) is \(\Delta K\), \(\Delta G\)
or another documented driver. The fitted relation tests an emergent simulation
response; it never feeds the fitted Paris parameters back into the cohesive
solver. CT-to-mesh registration should reuse the existing explicit
`surrogates.AffineCoordinateMap` (`x_model = x_CT @ A.T + b`) and store its
coordinate systems and units with comparison evidence; registration does not
belong in a plotting script.

## Verification ladder

The current executable foundation covers:

1. exact monotonic recovery when no fatigue cycles are advanced;
2. no damage under a static hold or below the fatigue threshold;
3. irreversible damage and no false closure dissipation;
4. exact constant-extrema one-cycle/cycle-jump equivalence;
5. cycle-block rollback and local restart equivalence;
6. all-field physical-facet checkpoint round trip;
7. real consumption by the existing three-dimensional surface assembler;
8. named-interface aggregation and three-dimensional front geometry;
9. global peak/valley transaction, post-damage equilibrium feedback and
   automatic cycle cutback;
10. exact cycle landings plus interrupted/durable restart equivalence;
11. atomic two-interface splitting on one three-dimensional solver mesh.
12. analytic 2D/3D cohesive tangents against force directional derivatives;
13. native force-controlled Neo-Hookean/cohesive Newton equilibrium with
    strong-constraint reaction evidence;
14. real FEM peak/valley, fatigue update, degraded re-equilibration, closing
    and interrupted/restarted equivalence;
15. two-rank sparse-owner cohesive tangent and distributed Newton assembly;
16. stable same-surface component identity, restart and explicit merge ancestry;
17. deterministic Paris postprocessing on a synthetic known power relation;
18. pure Mode-I, pure Mode-II and BK mixed-mode energy-range decomposition;
19. tangential-basis rotation invariance and explicit rejection of
    non-proportional peak/valley paths;
20. mixed-mode cycle-jump/exact-cycle equivalence, rollback and restart;
21. real 2D and two-rank MPI consumers with standard cyclic interface fields;
22. physical-facet-keyed mixed cyclic state written on two ranks and restored
    under a reordered one-rank partition.
23. ordered non-proportional path resolution, exact-cycle/jump equivalence,
    rollback and restart with path evidence;
24. named generalized-work integration for reference points, prescribed
    motion and MPC channels, with atomic energy rejection and restart;
25. DCB/ENF analytical structural energy-release oracles plus an MMB contract
    that requires independently declared mode partition, process-zone
    resolution and numerical-dissipation evidence.

Promotion to a cylinder-validated fatigue-crack-growth capability still requires:

- automatic reaction extraction for every native reference-point, MPC,
  weak/contact and prescribed-motion provider (the public ledger contract is
  now available);
- force-controlled cylinder examples and mesh/cycle-jump convergence;
- calibrated mixed-mode fatigue parameters and executed DCB/ENF/MMB finite-
  element convergence plus source-identified external structural curves;
- cross-partition portability for the bulk field part of the global checkpoint;
- symmetric two-crack and large-spacing limiting cases;
- an external numerical benchmark and retained experimental prediction cases.

Until those gates pass, the feature is described as an experimental global
cycle-lifecycle consumer and cohesive-facet foundation, not as a validated
cylinder fatigue solver.

## Primary references

- Roe and Siegmund, *Engineering Fracture Mechanics* 70 (2003) 209--232,
  DOI `10.1016/S0013-7944(02)00034-6`.
- Bak, Turon, Lindgaard and Lund, *International Journal for Numerical Methods
  in Engineering* 106 (2016) 163--191, DOI `10.1002/nme.5117`.
- Dávila, NASA/TP--2018-219838, *From S--N to the Paris Law with a New
  Mixed-Mode Cohesive Fatigue Model*.
- Leone et al., NASA NTRS 20205010748, *Scalability of Cohesive Fatigue
  Analyses Using Explicit Solvers*.
- Carreras et al., *A simulation method for fatigue-driven delamination in
  layered structures involving non-negligible fracture process zones and
  arbitrarily shaped crack fronts*, arXiv `1905.05000`.
- NASA/TM-2020, *Improved Benchmarking of Cohesive Elements in Abaqus Standard
  for Predicting Delamination Growth*.
- ASTM D5528/D5528M, D7905/D7905M and D6671/D6671M specimen families for DCB,
  ENF and MMB, respectively.
