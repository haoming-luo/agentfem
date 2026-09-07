# Dynamics and waves

AgentFEM separates the physical study from the solution procedure. Structural
dynamics can use an implicit Standard-like route or an explicit
central-difference route without changing the meaning of materials, regions,
loads, fields, and results.

## Current routes

| Procedure | Character | Typical use |
| --- | --- | --- |
| Modal analysis | Generalized Hermitian eigensolve | Natural frequencies and mass-normalized mode shapes |
| Newmark | Implicit | Structural response with controllable numerical parameters |
| Generalized-alpha | Implicit | Dynamics with high-frequency numerical dissipation |
| Central difference | Explicit | Wave propagation and short transient events |

## Modal analysis

Modal analysis uses the same material, region, field, and constraint language
as a static solid. Strongly constrained degrees of freedom are removed from
the assembled matrices before SLEPc solves

\[
\mathbf K\boldsymbol\phi_j
=\omega_j^2\mathbf M\boldsymbol\phi_j.
\]

```python
study = studies.modal_solid(dimension=2, assumption="plane_stress")
model = models.create(study=study, mesh=domain)
u = model.field(fields.displacement(domain, degree=2))
model.material(steel)
model.clamp(u, on=fixed_end)

result = model.step(target=u, modes=6).solve_result()
frequencies = result.quantity("frequencies")
mode_1 = result.field("Mode_1")
```

The result retains eigenvalues, angular frequencies, frequencies, relative
eigenpair residuals, mass-orthogonality and stiffness-diagonalization errors,
eigenvalue clusters, and each mass-normalized live mode field. For an isolated
mode, AgentFEM resolves the arbitrary sign by making the largest global
component positive. Vectors spanning a repeated or numerically clustered
eigenvalue are not individually unique: solvers may return any rotated basis
of the same eigenspace. AgentFEM therefore compares that cluster through
canonical correlations, principal angles, and projection distance of the
whole invariant subspace. It also records when a requested mode count cuts a
cluster, so downstream comparison does not assign physical identity to an
incomplete basis. A mass-normalized mode shape has no physical displacement
amplitude until it is multiplied by a modal coordinate. `slepc4py` is an
optional execution dependency because a dense array eigensolver is not a
scalable replacement for distributed finite-element modal analysis.
The finite-element provider checks the assembled free-DOF stiffness and mass
operators for symmetry before declaring the generalized Hermitian problem to
SLEPc. This also applies to a complete user-supplied `K/M` pair. An
unsymmetric operator is rejected rather than silently interpreted as
Hermitian.
When `target_frequency` is supplied, AgentFEM returns the requested modes
nearest that frequency and orders the selected set by increasing frequency.
The compact result writer preserves the field name (for example `Mode_1`) and
labels it as a mode shape rather than silently advertising it as displacement
`U`.

## Frequency and decay post-processing

`agentfem.dynamics` provides solver-independent processing for uniformly
sampled histories:

```python
spectrum = dynamics.spectrum(time, displacement_history)
frf = dynamics.frequency_response(time, applied_force, displacement_history)
damping = dynamics.damping_from_free_decay(displacement_history)
```

The FFT reports a one-sided, amplitude-corrected spectrum. FRF bins without a
meaningful input signal are marked invalid instead of generating hidden large
ratios. Free-decay processing reports logarithmic decrement, damping ratio,
and quality factor. These operations consume arrays and return structured
objects that can be converted to `SimulationResult`; they do not require a
particular beam or excitation.

## Linear viscoelastic spectra

The first viscoelastic foundation is a generalized-Maxwell spectrum with a
standard-linear-solid convenience factory:

```python
material = constitutive.GeneralizedMaxwell.from_prony(
    instantaneous_modulus=3.0e9,
    ratios=(0.20, 0.15, 0.10),
    relaxation_times=(1.0e-3, 1.0, 1.0e3),
    shift=constitutive.WLFShift(
        reference_temperature=293.15,
        c1=17.44,
        c2=51.6,
    ),
)

E_relax = material.relaxation_modulus(time)
E_storage = material.storage_modulus(2.0 * np.pi * frequency)
E_loss = material.loss_modulus(2.0 * np.pi * frequency)
tan_delta = material.loss_factor(2.0 * np.pi * frequency)
```

The same object owns an exact generalized-Maxwell branch update for a linear
strain increment and a commit/restore state. A complete material history can
use the common Procedure/State/Result lifecycle:

```python
step = material.history(time, strain, temperature=temperature)
result = step.solve_result()

stress = result.histories["stress"]
energy_error = result.histories["energy_balance_error"]
restart_state = step.last_response.final_state
```

For a stress-relaxation test, initialize the physical history explicitly:

```python
initial = material.initial_state(strain[0], condition="instantaneous")
result = material.history(time, strain, initial_state=initial).solve_result()
```

`condition="equilibrated"` instead means the declared initial strain has been
held until every Maxwell branch has relaxed. This prevents a nonzero initial
strain from silently acquiring an unspecified past.

The result includes accepted stress and branch-overstress histories, the exact
algorithmic modulus, recoverable energy, independently integrated mechanical
work, nonnegative viscous dissipation, and their balance error. A restarted
history begins from the copied accepted `MaxwellState`; an incompatible first
strain or branch layout fails before advancement.

The scalar object above remains a material-point procedure. For a global 3D
solid, use the isotropic tensor material through the same public workflow:

```python
material = model.material(
    constitutive.IsotropicGeneralizedMaxwell.from_prony(
        instantaneous_young_modulus=1.0e6,
        instantaneous_poisson_ratio=0.30,
        shear_relaxation_ratios=(0.25, 0.15),
        bulk_relaxation_ratios=(0.25, 0.15),
        relaxation_times=(0.2, 2.0),
    )
)

step = model.step(
    target=u,
    material=material,
    duration=5.0,
    incrementation=steps.automatic(
        initial=0.1,
        minimum=1.0e-4,
        maximum=0.25,
    ),
    time_error_tolerance=1.0e-3,
    amplitude=amplitudes.tabular(
        (0.0, 0.5, 5.0),
        (0.0, 1.0, 1.0),
    ),
)
result = step.solve_result(output="outputs/viscoelastic/fields.xdmf")
```

This route compares each proposed full time increment with two half increments.
The larger of the relative endpoint displacement and stress differences is the
recorded `time_error_estimate`. An excessive estimate or failed equilibrium
causes one atomic rollback of displacement and every Maxwell state variable,
followed by the declared cutback. Only the two-half-step solution is committed;
accepted and rejected attempts remain in the result evidence.

Use `steps=...` for a prescribed uniform time grid. Loading events and widely
separated relaxation times can instead be resolved explicitly without thousands
of empty increments:

```python
step = model.step(
    target=u,
    material=material,
    duration=50.0,
    time_points=(0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 50.0),
    amplitude=load_then_hold,
)
```

Adaptive `incrementation=steps.automatic(...)` is mutually exclusive with a
prescribed `steps`/`time_points` path. The prescribed routes are useful when
output must land on an experiment clock; the automatic route resolves the path
from a tolerance and records its decisions. A checkpoint preserves the accepted
adaptive path and next proposed increment so a resumed run does not silently
choose a different continuation. With `portable=True`, nodal displacement and
every committed Maxwell branch are stored by physical mesh identity and can be
resumed with a different MPI partition or rank count.

This first global provider solves small-strain quasi-static equilibrium on a
prescribed or automatically resolved physical-time path. Each accepted increment
commits the exact generalized-Maxwell quadrature state. Results expose `S`, `E`, `SENER`,
`VDENER`, `MISES`, their cell-recovered visualization fields, `RF`, and the
work--stored-energy--dissipation ledger. A WLF or Arrhenius material consumes
an explicitly supplied temperature scalar or field. Portable checkpoints reject
a changed material, amplitude, temperature, time grid, physical quadrature
identity or nodal field identity before restart. Response fields are rebuilt
from the accepted branch state without advancing relaxation time.

This is a bounded global FEM foundation, not a claim of nonlinear finite-strain
viscoelasticity, physical aging or direct harmonic assembly.
The independent three-dimensional
[Abaqus viscoelastic-rod benchmark](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-viscorod.htm)
checks prescribed traction, near-incompressible lateral contraction and the
published 0.001 s and 50 s creep response.
WLF and Arrhenius shifts reject singular, non-finite, or non-positive shift
factors instead of allowing an invalid temperature range into a state update.

For reviewed relaxation data and user-declared relaxation times,
`constitutive.fit_relaxation_prony(...)` provides a deterministic positive
reference fit. Automatic spectrum selection, multi-experiment uncertainty,
and constitutive-model recommendation belong to the future identification
layer rather than to the finite-element material itself.

The solution procedure and the constraint enforcement are checked together
before assembly. Projection periodicity is a serial, non-strict nodal
projection supported by central difference. Newmark and generalized-alpha do
not silently reinterpret it as a Dirichlet condition: model validation emits
`AFM-CONSTRAINT-PROCEDURE-001` and directs the user to an exact affine/MPC
backend. A serial-only constraint requested under MPI similarly emits
`AFM-CONSTRAINT-PARALLEL-001` before the solver starts.

`constraint.summary()` exposes the same capability contract to humans, agents,
and future GUIs. `PeriodicProjectionConstraint.diagnostics(field)` adds the
pair count, coordinate pairing error, unmatched count, and live field
mismatch. Exact rectangular matching-face construction is shared through
`constraints.rectangular_periodic_mpc`. Linear consumers use
`solvers.prepare_mpc_linear_problem`, which shares solver policy, convergence
evidence, repeated-solve state transfer, and MPI behavior across procedures.
Strong boundary conditions must be supplied when the MPC is constructed; a
late Dirichlet condition that overlaps an owned slave is rejected before
assembly. Nonlinear consumers still select an MPC-aware provider explicitly
because ordinary DOLFINx and `dolfinx_mpc` assembly are not interchangeable.

## Engineering questions to make explicit

- mass representation and density;
- damping model and whether it is physical or numerical;
- stable time increment for explicit analysis;
- source amplitude and time support;
- absorbing, periodic, or reflective boundary behavior;
- field sampling cadence versus integration cadence;
- kinetic, strain, external-work, and balance histories.
- mode residuals, modal truncation, frequency resolution, windowing, and FRF
  input observability;
- instantaneous versus equilibrium modulus and the temperature-shift
  convention for viscoelastic spectra.

## Go deeper

- [Scientific operator contracts](../operator_contracts.md)
- [Stable steps and output](../step_and_output_architecture.md)
- [Wave packet with an inclusion](../examples/wave_packet_inclusion.md)
- [Linear viscoelastic material contract](../reference/scientific_function_reference.md#linear-viscoelastic-relaxation-spectrum)
