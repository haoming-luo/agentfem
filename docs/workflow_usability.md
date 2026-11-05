---
title: "Check, run and compare AgentFEM simulation results"
description: "Use the AgentFEM CLI to inspect projects, run cases, locate output files and compare declared quantities or histories with explicit tolerances."
---

# Check, run, inspect, compare

One project remains one readable `case.py`, with operational metadata in
`agentfem.toml`. These commands do not require an AI assistant.

```bash
agentfem check --runtime
agentfem run --name baseline
agentfem show latest
agentfem show latest --files
```

`check` checks project structure and Python syntax without executing the case.
`--runtime` also checks numerical runtime readiness. Neither promises that a
particular mesh, material, boundary condition or solver combination is supported;
the model's own preflight remains authoritative.

`show --files` lists recorded artifacts and reports missing files with exit code
2. Keep an XDMF file together with its referenced HDF5 files. This command checks
recorded paths, not the internal contents of visualization files. Normal `show`
keeps the file list short; JSON and `--files` retain the complete list.

## Compare selected results

```bash
agentfem compare outputs/001-baseline outputs/002-candidate \
  --quantity displacement_max --atol 1e-8 --rtol 1e-5
```

Choose quantities explicitly, with an absolute tolerance in their recorded
units. Repeat `--quantity` or use `--history reaction`. One invocation applies
the same tolerances to all selected observables; use separate invocations when
different quantities need different tolerances.

The comparison requires completed result manifests, identical declared units
and shapes, and, for histories, stored samples with identical increasing
coordinates. It never silently converts units, interpolates histories or maps
fields between meshes. Missing units are not assumed dimensionless: record `1`
explicitly. Missing or incompatible data fails closed. Relative error against
a zero reference norm is unavailable; absolute error is still reported.

Acceptance is componentwise `abs(candidate-reference) <= atol + rtol*abs(reference)`.
Exit code 0 means the selected observables meet these tolerances; code 2 means
failure or incompatibility. Add `--json` for machine-readable output. This is
not independent scientific validation and does not promote either result's trust.

## A small real finite-element campaign

```bash
agentfem init wall-study --template heat-campaign
cd wall-study
agentfem check --runtime
agentfem run --name first -- --samples 3
agentfem run --name resumed -- --samples 3
agentfem show latest --files
```

The installed template solves steady heat conduction, varies the hot boundary
temperature deterministically, and exports a scalar-response dataset and audit.
The project-local `campaign-data` ledger lets the unchanged plan reuse completed
cases. Edit the physical problem in `case.py`; consult the template README before
changing an existing campaign. This is a bounded workflow example, not a trained
surrogate or a claim that arbitrary thermal models are validated.
