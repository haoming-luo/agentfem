# Validation Reference

Minimum validation after code changes:

1. Compile/import touched modules.
2. Check mesh summaries and required tags when a workflow depends on labels.
3. Run `model.validate()` and report issue codes and paths when public modeling
   assets changed.
4. Check AF-IR JSON safety and determinism when serialization changed.
5. Run a small serial case when the change affects execution.
6. Run a small MPI case when the change affects parallel behavior.
7. Report untested assumptions and unsupported backend capabilities.
8. Keep computed/converged/verified/validated claims distinct. Use named
   quality presets for routine checks; use `verification.VerificationClaim`
   and record the reference validity domain for scientific promotion.
9. For mesh/time accuracy, require an ordered coarse-to-fine
   `verification.ConvergenceStudy`; one converged solve is not convergence.

For modeling changes, also check units, boundary-condition type, and output
availability.

For constitutive changes, record one of these maturity levels:

- `material_point_verified`: formula, state, and local load-path evidence;
- `fem_integrated`: quadrature/global state and nonlinear/time solve evidence;
- `postprocessor`: consumes results but does not alter the FEM equilibrium.

Every advancement requires a benchmark-registry entry and automated test.
Run `benchmarks.audit_capability_evidence()` before changing a catalog
maturity label. The audit must satisfy the evidence required by the declared
maturity; a passing audit does not promote an experimental capability or erase
its stated limitations.
For external meshes, verify both volume and boundary named-set preservation.
For external scientific benchmarks, pin the upstream commit, retain the exact
public input view, classify execution/accuracy/time failures separately, and
stratify incompatible or inconsistent dataset subsets rather than adding
case-specific exceptions. Never use withheld reference fields to select a
mesh, formulation, or answer.
For Abaqus equation-driven workflows, additionally verify source node-label
mapping, unique/cycle-free slave equations, exact post-solve equation
mismatch, positive sampled `det(F)`, and scale-one deformed output. A serial
equation backend must reject MPI execution explicitly.
For periodic-cell output, distinguish one analysis step from its load
increments, verify every requested XDMF field at every saved factor, normalize
effective stresses by the complete cell volume, and compare direct
first-Piola integration with the transformed Cauchy-stress integral.
For finite-strain output, resolve the conventional `E` request to `LE` and
require `GREEN` explicitly. Verify the unified XDMF/HDF5 series: frame count,
time values, shared topology, retained reference coordinates,
`x + scale*u`, point/cell field presence, and physical scale metadata.
Treat UMAT/UHYPER support as an interface specification until quadrature state,
trial/commit/rollback, tensor conventions, compiler ABI, and consistent-tangent
comparisons are executable. A callable shared library alone is not FEM
compatibility.
For campaigns, verify `SimulationResult -> declared QoIs -> ScientificDataset`
without serializing live fields. Use `quality="engineering"` for ordinary
simulation-to-learning admission and `quality="release"` only when every
sample carries release-grade scientific evidence.
Review the campaign scientific-input manifest: files must be content-hashed,
public objects must expose an IR/summary contract, and opaque paths must remain
an explicit coverage gap. For convergence, require explicit fixed coordinates
for all non-refined parameters and retain failed or missing cases as
inconclusive evidence.
For response experiments, verify every baseline/perturbation case, parameter
bound, output shape, step convention, and missing-case decision. Check
finite-difference step sensitivity before interpreting a derivative. Preserve
event brackets and censoring; interpolation does not replace local time-step
refinement for discontinuous events.
