# Material-verification audit — 2026-10-29

## Reproduced defects and bounded repair

A frozen material-point dataclass still contains writable arrays. The original
full tangent checker passed its baseline request directly to the provider; a
provider which modified old F could finish the check and alter the caller's
request. The reduced checker protected the caller through local copies but did
not detect mutation inside a candidate. Both could mislabel evidence as
fixed-old-state differentiation. A failing regression reproduces this before
the repair.

Every verification evaluation now gets its own request, and observable input
array changes are rejected, including when the provider subsequently raises.
The same guard covers small-strain verification. The plane-stress line search
propagates this contract error rather than retrying it as a local numerical
failure. Production integration-point updates have not acquired this audit.
This is not a sandbox against malicious providers or proof of hidden-state
purity; material history must still live in the declared state contract.

The small-strain tangent checker also reused unscaled squared norms. It now
shares the unit-robust finite-strain error calculation and refuses
unrepresentable perturbations. Tensor/engineering shear conventions remain
distinct. Scalar and batch stress symmetry use point-local normalized tensors;
large neighbors do not conceal smaller nonsymmetric responses. Small-strain
symmetrization avoids overflowing the intermediate sum of large equal entries.

## Evidence

- Targeted affected regression: **732 passed**, 75.86 s in the real FEniCSx
  environment, one numerical thread. Includes automatic persistence, forced
  interruption and output continuity, full/reduced and small-strain materials,
  array transport, ordinary project execution and architecture/documentation.
- Baseline and later-call input mutation, provider-failure combinations,
  repeated fixed history, both shear conventions, intentionally wrong tangents
  across stress scales 1e-200…1e200, stress symmetry 1e-300…1e308, zero/subnormal
  responses, and heterogeneous batch scales have explicit regression coverage.
- Correctness lint, generated API and knowledge import checks pass. The initial
  sandboxed knowledge import was blocked by MPI network initialization; the
  permitted-environment check succeeded. That infrastructure error was not a
  failed numerical test.

No new material physics, dimensional reduction scope, MPI admission, arbitrary
energy-law claim or performance gain is implied. Rotation testing must declare
history and external-field frames; shapes alone cannot establish those rules.
