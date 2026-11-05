# Accepted small-strain user-material histories

## Scope and ownership

Reuse the existing material-point recorder and physical-time small-strain Step.
The ordinary `model.step(material_points=..., duration=...)` records raw `S`,
total infinitesimal `E`, and named local state at selected original cell/point
identities. The legacy no-duration and MPI checkpoint paths are unchanged.
No extra material calls, projected fields, temperature inference or life model
are introduced. Units follow the model or state schema; unknown units stay unknown.

## Acceptance boundary

Prepare the detached point frame before committing state. Commit failure,
including failure after partial commit, restores displacement, response and
named history. Only then append a frame. Selection and units enter scientific
identity; geometry and quadrature are checked before advance, output and save.
V2 checkpoint manifests retain the existing hashed history payload and validate
its complete accepted-time ledger before restoring any live fields.

## Evidence

The targeted suite passed 133 tests with one dedicated two-rank admission test
skipped in serial. Tests cover raw tensor association, result NPZ roundtrip,
continuous versus restarted history, rejected increments, corrupt/missing
payloads, selection changes, and failures before/after constitutive commit.
The ordinary relaxation example has an opt-in `--material-point` flag.

The ordinary CLI was also exercised in an isolated copy of that example using
the candidate source package and real FEniCSx: `agentfem run --name full --
--material-point`, followed by a second run loading the midpoint checkpoint.
Both completed; all 16 times and all values in `sample.S`, `sample.E` and
`sample.EV` were exactly equal across continuous and resumed execution.
This is source-candidate CLI evidence, not a clean-machine installation claim.

After integrating the independently tested P92 tensor reference, the selected
history path also passes a real reference-material ramp/hold/restart check.
Raw `S` agrees with the material's Mandel-stress state after conversion;
all selected frames match continuous execution exactly and end at 8 seconds.
The combined structural/history/CI-routing suite passes 88 tests. Accumulated
inelastic `P` remains a state variable, not a signed strain-life input.

This is a data/lifecycle capability, not P92 calibration. Further physical
validation must retain the recorded strain measure and original material point.
