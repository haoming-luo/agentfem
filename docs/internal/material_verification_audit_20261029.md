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

## Follow-up: explicit-frame rotation evidence and real contracts

`check_material_rotation` now compares one discrete update under independently
declared old/new proper rotations. Every named history variable must specify
invariant, spatial-vector or spatial-tensor transformation; external field
arrays without frame declarations are refused. Reference/corotated parameters
are unchanged. Cauchy stress, first Piola stress, `dP/dF` and each history
variable have separate error channels. Optional characteristic scales are
explicit and recorded; neither a large clock nor a different unit system can
hide a small history variable's error. This is covariance evidence, not a
derivative, thermodynamic or general material-objectivity proof.

Negative controls deliberately break stress, tangent or history transformation.
Another control scales an otherwise covariant tangent: rotation passes while
the independent derivative check fails. Existing logarithmic J2 analytic and
numerical tangent paths are checked through loading and reversal, without
broadening their structural admission. The runnable README snippet is tested
for both educational local laws.

The audit also reproduced silent complex-to-real conversion in real material
requests. Scalar, object-batch and columnar boundaries now reject unsupported
complex values before casting; physical time and optional scalar channels must
be actual finite real scalars. Finite F must also have a finite positive
determinant. Invalid driver inputs are rejected before the provider and restore
trial storage; complex checkpoint displacement/history is refused atomically.
This does not remove or redefine separate frequency-domain complex physics.

The existing two-rank ordered/columnar late-chunk failure tests pass (two tests
per rank). They establish collective rollback for the material driver only;
the new global history Step remains serial.

Final affected regression: **888 passed**, 79.93 s, in the real FEniCSx
environment with one numerical thread. This includes the documented local
verification snippet, both history laws, plane-stress and full 3D checks,
existing J2, checkpoint failure/recovery, real VTK output, CLI and
extension-aware guidance. Early failures in the new scalar-channel test were
fixture errors (undeclared 9x9 tangent and the small-strain energy field name);
the corrected test and the final affected selection pass.

An isolated installed-package check, without `PYTHONPATH`, also passes the
README's tangent/rotation snippet for both laws and six ordinary CLI runs:
pause, automatic-checkpoint resume and uninterrupted reference per law.
Each joined view has 16 unique frames and 64 cells, zero measured field
difference, identical final checkpoint arrays and actual VTK/Warp field
preservation. Each run retains two scheduled archives. The candidate wheel
SHA256 is `fe23d7d6c03f993c5b679919d4a47c2359adad4b70449b8109a8482be033ab2d`;
native accelerators were disabled for this workflow acceptance, and the user's
installation was not replaced. This is not a performance or cross-platform
bitwise-reproducibility claim.

### Primary references and interpretation

- [Sierra/SM frame indifference](https://www.sandia.gov/files/sierra/SM_Theory_5_20/main/frame_indifference.html)
  motivates left-rotated F and covariant spatial stress. The implemented
  two-endpoint discrete check does not certify every time discretization or
  objective stress rate.
- [TFEL generic behaviour interface](https://thelfer.github.io/tfel/web/generic-behaviours-interface.html)
  distinguishes stress/tangent conventions and material-axis transformations.
  Observer covariance here is not a replacement for an anisotropic material
  orientation adapter.
