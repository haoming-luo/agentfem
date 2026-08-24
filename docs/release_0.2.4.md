# AgentFEM 0.2.4

AgentFEM 0.2.4 deepens the high-temperature and cyclic inelastic workflow
without creating a second user language. The public path remains:

```text
Study -> Model -> scientific assets -> model.step(...) -> SimulationResult
```

## Cyclic plasticity

`constitutive.chaboche(...)` adds exponential isotropic saturation and one or
more Armstrong--Frederick backstresses to the existing global small-strain J2
route. It uses committed/trial quadrature state, global Newton equilibrium,
non-monotone tabular amplitudes, rollback/cutback, and restart. Results expose
`S`, `PE`, `PEEQ`, total backstress `ALPHA`, `MISES`, and their presentation
fields.

Material-point shifted-yield, tangent and reversal tests are combined with a
global cyclic and checkpoint-equivalence test. The implementation follows the
public Abaqus combined-hardening definition and published-style parameters;
it remains experimental until a structure-level stabilized-hysteresis curve
and complete dynamic-recovery energy closure are accepted.

## Sequential heat, creep and evidence

Accepted `FieldHistory` archives now retain the source Study, procedure and
one-way transfer role in addition to time, interpolation, range policy and
content identity. `assessments.sequential_energy_ledger(...)` links this
handoff while keeping thermal and mechanical residuals in separate layers.
It does not construct a fictitious monolithic conservation error.

`DwellInterval` and `creep_fatigue_from_result(...)` form the engineering V1
result consumer. They extract governing stress and temperature from named
histories, call a project-owned rupture-time relation with an explicit source,
and combine the resulting creep time fraction with the existing fatigue
assessment and declared interaction diagram. No design-code or material data
is embedded in the open core.

## Compatibility and release boundary

Existing linear-isotropic J2 checkpoints retain the released v4 schema.
Chaboche checkpoints use an extensible v6 state-variable list. The release
does not claim plane-stress or finite-strain cyclic plasticity, a coupled
creep--fatigue damage law, monolithic thermo-mechanics, or qualification to a
particular high-temperature design code.

The release gate requires the full serial suite, MPI suites, scientific
knowledge generation, strict documentation build, clean wheel/sdist,
installed-wheel identity, project templates, and representative workflow
smokes.
