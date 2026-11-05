---
title: "Experimental strain-history material runtime in AgentFEM"
description: "Understand the bounded bridge from a reviewed 3D VUMAT to AgentFEM response materials, including declarations, state identity and explicit trust requirements."
---

# Declared strain-history materials (experimental)

`constitutive.StrainHistoryRuntime` lowers a reviewed legacy 3D VUMAT to the
existing response-only material and native Hex execution interfaces. It does
not create a solver, own accepted state, or claim arbitrary Abaqus compatibility.

```python
with constitutive.StrainHistoryRuntime(
    "material/material.toml", parameters=material_parameters,
    density=physical_density, cache_directory=".agentfem/material-build",
    trusted=True, workers=1, cell_workers=1,
) as runtime:
    model.material(runtime.material)
    step = model.step(target=u, execution_kernel=runtime.kernel, **options)
    result = step.solve_result()
```

The material directory needs the original source and a declaration. An optional
`optimized/` source remains user-owned, never rewritten automatically. The
original law is retained for reference updates and native per-point repair.
AgentFEM generates the transport/fusion code, reuses the standard VUMAT bridge,
and builds content-addressed libraries without installing packages or downloading
anything. Source review and explicit trust are required: native code is executable.

## Declaration

This is an illustrative two-parameter law, not a recommended material or limit:

```toml
schema = "agentfem.strain-history.v1"
source = "law.f90"
# accelerated_source = "optimized/law.f90"  # explicitly reviewed equivalent law
property_names = ["lambda", "mu"]
parameter_bounds = [{name="mu", minimum=0.0, exclusive_minimum=true}]
state_count = 3
unused_channels = ["coordinates", "characteristic_length", "relative_spin",
  "temperature", "stretch", "deformation_gradient", "fields", "absolute_time",
  "material_name", "old_energy_for_response"]
kinematics = "linear_right_stretch_path"
energy = "incremental_internal_work"
thread_safe = true
maximum_stretch_increment = 0.001
increment_ceiling = 1.0e-5
state_bounds = [{index=0, minimum=0.0, maximum=1.0, tolerance=1.0e-12}]
```

Parameter order follows `property_names`; the caller must provide exactly those
names with finite values in a consistent unit system. Optional parameter bounds
declare inclusive or exclusive endpoints and are checked before compilation.
Parameter/state counts
are not tied to a particular damage law. Bounds use zero-based source STATEV
indices and do not clip values. The runtime stores zero-initialized source
history followed by six old corotational stresses in the `history` state field.
Its schema and full source/policy identity participate in ordinary restart.
Non-virgin initialization is not provided by this convenience runtime.
New native-body checkpoints separate worker/tile scheduling from that scientific
identity; see [restart compatibility](native_explicit_execution.md#restart-compatibility-versus-execution-provenance).
Legacy checkpoints still need the original execution configuration before they
can be restored and saved in the new format. This does not admit a different
material binary, numerical increment policy or state encoding.

`fuse_viscosity=True` optionally joins core bulk-viscosity arithmetic to the
native material/body cell pass. It does not change the source material or its
state schema, and requires a dispatcher supporting that capability. The default
is unchanged; choose this experimental execution option only after a relevant
whole-step comparison. See [joined body execution](native_explicit_execution.md#optional-joined-body-and-bulk-viscosity-execution)
for validation, fallback and restart boundaries.

`unused_channels` is a substantive source-review assertion, not a convenience
toggle. The source may use properties, density, strain increment, previous
stress/history and time increment. It must not use the listed context channels
to compute response. Placeholders are supplied only under that declaration.
All physical memory must live in the supplied state/stress arrays; hidden SAVE
or COMMON history is not admissible. Native fusion requires an explicit
reentrancy declaration even with one worker, since independent runtimes may
share a loaded library. Non-reentrant sources retain `compile_vumat` and a
separately reviewed serialized driver. A declaration does not prove thread safety.
The fused batch protocol also requires pointwise response: a point must not
depend on the other points in its batch, the batch size or its partitioning.
Thread safety alone does not establish that property. Cross-point/nonlocal
material models need a different, explicitly declared execution contract.

The selected finite-increment path integrates along linear right stretch and
pushes corotational stress to spatial Cauchy stress. Tensor shear uses
11,22,33,12,23,31. This is an explicit numerical choice, not a promise to reproduce
every host's objective integration. The reference density is divided by the
mean old/new Jacobian. Old specific energies start at zero for each call: the
source must supply additive internal work per current mass, independent of the
old energy in its stress/history update. The returned increment is multiplied
by reference density. It is **not stored energy or irreversible dissipation**.
Sources requiring temperature, characteristic length regularization, absolute
time, fields, energy-driven evolution, or another kinematic convention must use
the complete `compile_vumat` / `VumatBatchInput` contract and a reviewed driver.

The increment ceiling and maximum relative principal-stretch change are local
material integration controls, not body/contact stability certificates. A cheap
sufficient screen admits small increments; unresolved points use a singular-value
check. Violations request cutback through the existing material contract.

## Execution and evidence

The generated native function fuses kinematics, user response and stress mapping,
then uses the existing native material/body dispatcher. A rejected kinematic
point is repaired from the original source before publication. Invalid state or
nonfinite output rejects the batch; no old or zero stress is substituted.
The owning Step remains responsible for trial/commit/rollback and checkpoint.

The supplied optimized source is an executable alternative, not something the
manifest certifies. Compare original and optimized responses, including damage,
work and accepted-state histories. Changing the runtime identity requires a new
run; do not remove checks to continue an old checkpoint. Worker budgets are
operational choices, not transferable performance guarantees. This path currently
requires the POSIX native dispatcher and GNU Fortran (macOS/Linux/WSL2).

Public tests exercise multiple state widths, worker counts, original/fused
response parity, rejected increments, bounded state, missing work and polar
repair. Private project replay and full-run timing are separate evidence gates.
