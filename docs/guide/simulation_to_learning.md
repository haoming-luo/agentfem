# Simulation to learning

AgentFEM does not replace PyTorch or a user's neural-network stack. It makes the
path from a parameterized finite-element model to accepted, traceable training
data much shorter and more systematic.

## Workflow

```text
parameter space → reproducible FEM campaign → result policy
                → accepted ScientificDataset → NumPy/PyTorch adapter
                → built-in or user model → validation/applicability guard
                → learned prediction or FEM fallback
```

## Available building blocks

- parameter campaigns with persistent case identity and parallel execution;
- dataset acceptance rules rather than silent collection of failed cases;
- NumPy and optional PyTorch adapters;
- ridge and POD-ridge baselines plus an optional PyTorch MLP;
- user-owned PyTorch estimators without inheritance from a proprietary model
  class;
- training/validation separation, applicability checks, and high-fidelity FEM
  fallback.

This observation and dataset contract is the basis for future neural operators
and digital-twin state updating. Those directions require stable field
encodings, sensor identity, time alignment, and uncertainty—not just another
neural-network class.

## Go deeper

- [Results and campaigns](../results_and_campaigns.md)
- [AI-native campaigns and learning](../ai_native_learning.md)
- [Digital-twin direction](../digital_twin_direction.md)
- [Surrogate-campaign example](../examples/index.md#elasticity-surrogate-campaign)
