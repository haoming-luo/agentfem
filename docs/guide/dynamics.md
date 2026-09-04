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
eigenpair residuals, and each mass-normalized live mode field. `slepc4py` is
an optional execution dependency because a dense array eigensolver is not a
scalable replacement for distributed finite-element modal analysis.

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
strain increment and a commit/restore state. This release therefore supports
material-point relaxation paths and time/frequency spectra. A global FEM
transient provider consuming those internal variables is a separate promotion
gate and is not implied by the public material name.

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
`constraints.rectangular_periodic_mpc`; consumers still select an MPC-aware
linear or nonlinear solver explicitly because ordinary DOLFINx and
`dolfinx_mpc` assembly are not interchangeable.

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
