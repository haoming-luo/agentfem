# Heat transfer

AgentFEM supports steady conduction and implicit transient heat transfer through
the same study/model/step/result lifecycle used by solid mechanics.

## Current routes

- steady heat transfer with conductivity and thermal boundary conditions;
- implicit Euler transient heat transfer with structured increments;
- temperature field output and history recording;
- thermal-field handoff to thermoelastic stress workflows;
- common result, progress, checkpoint, and verification concepts.

## Typical workflow

```text
thermal study → mesh/regions → temperature field → conductivity/capacity
              → temperature/flux/convection conditions → transient or steady step
              → temperature/flux histories → verification → optional stress handoff
```

The most important next depth is energy accounting, richer surface exchange,
robust transient checkpointing, and coupled thermo-mechanical state transfer.

## Go deeper

- [Thermal stress and creep procedures](../solution_procedures_and_thermal_creep.md)
- [Results and post-processing](results.md)
- [Transient-heat example](../examples/index.md#transient-heat-transfer)
