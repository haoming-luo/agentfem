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
8. Keep computed/converged/verified/validated claims distinct. Use
   `verification.VerificationClaim` and record the reference validity domain.
9. For mesh/time accuracy, require an ordered coarse-to-fine
   `verification.ConvergenceStudy`; one converged solve is not convergence.

For modeling changes, also check units, boundary-condition type, and output
availability.

For constitutive changes, record one of these maturity levels:

- `material_point_verified`: formula, state, and local load-path evidence;
- `fem_integrated`: quadrature/global state and nonlinear/time solve evidence;
- `postprocessor`: consumes results but does not alter the FEM equilibrium.

Every advancement requires a benchmark-registry entry and automated test.
For external meshes, verify both volume and boundary named-set preservation.
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
without serializing live fields. Use `minimum_trust_level="verified"` when
release or learning data must exclude merely successful simulations.
