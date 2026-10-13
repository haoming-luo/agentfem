# Built-in explicit execution

AgentFEM owns the reusable Hex8 and Q1 geometry arithmetic. A user material
supplies its constitutive law; it does not need to implement Piola conversion,
cell force, hourglass stabilization, contact search or state acceptance.

## Ordinary modeling

```python
kernel = elements.explicit_hex_kernel(backend="auto")
with boundary_models.q1_contact_runtime(backend="auto") as contact:
    # Pass contact.execution to the ordinary deformable_contact_pair(...).
    # Materials, mesh, boundary definitions and physical inputs are unchanged.
    step = model.step(target=u, execution_kernel=kernel, **step_options)
    result = step.solve_result()
```

These are execution choices for the existing bounded serial response-only
Hex8 route, not a new Model or solver. Its existing unsupported-combination
checks, stability limits, accepted-state transaction, restart and result trust
levels remain in force. No MPI capability is added by selecting native code.

`explicit_hex_kernel` accepts the ordinary array-batch material protocol and
arbitrary state schemas. It contracts Cauchy stress into work-conjugate first
Piola stress, material force and hourglass force/energy without a global
stiffness matrix. Installed native builds additionally fuse mean-gradient
construction with the existing sufficient geometry screen and provide bounded
bulk-viscosity arithmetic. Ambiguous geometry retains the Bernstein checks;
ill-conditioned bulk conversion retains the reference implementation.

`cell_workers=1` is the reproducible default for native gradient, validity
screen, bulk-viscosity and nodal-to-cell gathering. A larger explicit budget
(1–16) partitions only disjoint cells, preserves arithmetic order inside each
cell, and joins all workers before returning. Small batches stay serial.
There is no retained scratch or worker pool to close. Material `workers` and
`cell_workers` belong to different sequential phases; neither changes the time
grid, constitutive law or inertia. Contact overlap can still compete for CPU,
so select budgets from whole-increment measurements on the intended machine.

Native auxiliary outputs are allocated directly as immutable core-owned byte
buffers. The execution boundary still checks finite values, geometry and
material admission, but it no longer needs to copy full cell arrays merely to
detach callback storage. Gathering freezes the smaller nodal inputs, checks
every connectivity index, and emits the solver's existing Basix cell order;
this is not a new mesh ordering or an output-format conversion. Previously
published arrays remain valid across later calls, errors and rollback.

`q1_contact_runtime` owns only geometry workers and caches. Its optional native
backend includes Q1 projection, exact-rational convexity recovery, batch BVH
search, closed-surface classification, conservative distance bounds and swept
interval separation. Q1 faces are not replaced by triangles. Unresolved
projections and paths retain the reference algorithm; ambiguity is not resolved
by arbitrarily choosing the first face. Use the context manager to join workers.
Defaults are conservative: no process workers and one batch worker. Select and
measure larger worker budgets per machine; do not infer them from another case.

## Compiled user materials

Provider authors may use `elements.native_explicit_hex_kernel` to fuse a
reentrant C material update and the same core force arithmetic in one batch.
The C ABI, state width, material parameters, density, implementation identity,
exact fallback and optional validation callback must be declared. No fixed
material name, parameter count or damage-state layout appears in the dispatcher.
Rejected points are repaired only after all workers join, before immutable
outputs are published. A repair failure cannot commit partial state.
The revision-2 dispatcher also restricts body-force repair to rejected points
and avoids starting repair workers for unaffected slices. A one-worker call
executes directly without creating a temporary thread. This changes execution
work, not the material law, fallback, time increment or accepted-state owner.
Execution revision remains part of checkpoint identity; an ongoing frozen run
must not be silently switched to another dispatcher revision.
Revision 3 adds the explicitly recorded `cell_workers` execution budget and
the immutable native auxiliary path. It does not authorize migration from an
older checkpoint identity. Retain the frozen runtime for that continuation or
start a new run; compare archived stations without rewriting their identities.
Revision 4 also reuses the existing strict affine-remainder screen inside the
gradient workers. The standalone and fused routes share one C implementation;
no threshold is relaxed. Inconclusive cells still take the same SVD/Bernstein
fallback. This avoids repacking and serial dispatch of already certifiable
rotated/deformed cells rather than skipping geometry admission.

This ABI is trusted executable code, not a sandbox: do not load function
addresses from a model file. ABI v1 supports stress, internal state and internal
work. It deliberately rejects temperature/field inputs and optional repair
energy channels it cannot represent. It neither invents a tangent nor labels
internal work as stored energy. Material increment limits remain owned by the
existing material contract, not by this performance adapter.

## Installation and reproducibility

Source builds attempt optional native compilation by default; set
`AGENTFEM_BUILD_NATIVE=0` for a pure-Python build. Missing compilers retain the
reference implementation. The fused pthread material dispatcher is currently
POSIX-only. Native Q1 geometry also requires GMP headers and library in the
build Python prefix, or an explicit `AGENTFEM_GMP_PREFIX`; builds do not download
GMP. Native wheels must be built/repaired and tested for their destination
platform, including their shared-library dependencies. A locally built macOS
wheel is not a portable Linux/Windows release artifact.

`auto` explicitly records the selected implementation in kernel identity;
`native` fails if its required extension is unavailable. Keep execution choices
unchanged across a restart. An implementation change requires a separately
reviewed migration or a fresh run, not removal of checkpoint identity checks.

`agentfem doctor --json` reports `execution.accelerators` for the installed Hex
arithmetic, material/body dispatcher and Q1 geometry. `available` means the
extension can load; `not_built` means absent; `load_failed` records a loader or
interface error (for example a missing GMP library). This probe starts no worker
pool and does not select a backend. The Step's execution identity remains the
authority for actual use; availability is neither a speed claim nor a validation
badge. Optional absence does not disable the reference solver.

## Verification and performance

`examples/elastic_bar_wave.py` is a public free-bar standing-wave example with
an independent small-amplitude analytical reference, XDMF output and elapsed
time. It uses no private geometry or constitutive binary. Its analytical check
is the linear limit of the chosen finite-strain elastic law, not a general
large-strain or contact validation.

The migration tests include arbitrary state widths, repaired batches,
unpublished failures, reference/compiled comparisons, warped and concave Q1
surfaces, seam ambiguity, translated geometry, crossing paths and restart.
Full-model elapsed time and late-stage numerical equivalence are separate
acceptance gates; passing these local tests establishes neither a whole-run
speedup nor independent vendor agreement.

Native builds also perform Cauchy-stress symmetry admission in one scalar
scan per point, without a full-mesh absolute-stress temporary. The relative
tolerance and tiny-value floor are identical to the NumPy reference; tests
cover threshold neighbors, extreme scales, unaligned buffers and nonfinite
data. The ordinary Python fallback remains supported. This reduces temporary
storage and repeated passes; it is not a measured whole-run speedup.

Response-only body rollback now shares immutable accepted gradient, force and
displacement bytes, with a fresh array descriptor for every snapshot. Live
trial data is frozen at publication; old snapshots remain valid across commit,
rejection and restart. This is ownership, not a cache of possibly stale fields.
It avoids copying these three arrays again for every rollback capture. It does
not remove validation, callback isolation, or copies of mutable solver fields.
Both field-backed and owned-array material modes retain the same numerical
results in the targeted regression tests; full-run throughput is a separate gate.

The design follows the separation of operator action from global matrix
storage described in [MFEM's assembly levels](https://mfem.org/howto/assembly_levels/)
and keeps broad-phase spatial search distinct from narrow-phase geometry,
as in [DOLFINx geometry](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.geometry.html).
These references motivate the architecture, not the numerical certification
of this implementation or a borrowed speedup claim.
