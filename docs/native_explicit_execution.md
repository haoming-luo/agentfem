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

A general second-order dynamics Study reports `linearity="unspecified"`
unless the user explicitly declares it through `Study`. Time order and the
explicit/implicit preference do not establish linear physics. The actual
material, kinematics and boundary operators determine the response; a
declaration alone does not enable an unsupported capability.
`SolutionProcedure.nonlinear` describes the global solution algorithm, not
whether a central-difference residual can contain nonlinear constitutive or
contact contributions. In particular, explicit dynamics can advance nonlinear
responses without a Newton loop or global tangent, as described in the
[Abaqus explicit formulation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-expdynamic.htm).
This metadata correction changes neither numerical advancement nor accepted
state; frozen reports retain the declaration emitted by their original runtime.

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
The standalone `explicit_hex_kernel` retains no worker pool. The compiled
material binding below reuses one instance-owned executor. Material `workers`
and `cell_workers` belong to different sequential phases; neither changes the time
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

Reference-projection process workers are isolated numeric services: they do
not reimport `case.py` or initialize another FEM/MPI runtime. They remain asleep
between batches, limit their own nested arithmetic threads, and preserve the
reference projection algorithm. A crashed or timed-out service fails the
request and is cleaned up; it never substitutes a guessed projection. The
parent solver's environment and signal handlers are not changed. This avoids
hidden communication-thread contention without changing the model or time step.
`geometry_timeout=300.0` bounds a worker batch in elapsed seconds. Increase it
for unusually expensive batches or set it to `None` to disable the deadline.
It is an operational failure guard, not a physical increment or a performance
qualification threshold, and is not part of scientific checkpoint identity.

For the bounded closed-solid deformable-contact route,
`q1_contact_runtime(..., separation_reuse="anchored")` optionally retains an
immutable separation certificate. It measures relative movement from that
certified configuration rather than repeatedly subtracting accumulated travel.
Small oscillations can therefore avoid needless reclassification. The bound
covers both endpoints and the complete linear increment using convex Q1
weights; it never freezes an active normal, contact force or slip state.
Nonpositive bounds still invoke current-geometry search and the ordinary swept
guard remains active. Rollback/restart discards this derived cache. Open patches
retain the existing route; changing topology, self-intersection and curved
temporal trajectories are not admitted by this certificate.

The default `separation_reuse="incremental"` is unchanged. The optional policy
is recorded in execution identity and is an experiment until whole-increment
measurements show a benefit. Additional bound evaluation can itself cost time.
The reusable idea resembles [displacement-bounded neighbor reuse in LAMMPS](https://docs.lammps.org/Developer_par_neigh.html),
but a particle cutoff is not substituted for the deformable Q1 distance problem.

### Current-geometry closed Q1 signs

`q1_contact_runtime(..., sign_policy="interior_normal")` is an opt-in execution
policy for the **reviewed embedded closed-solid** route. At an unambiguous
global closest point strictly inside a smooth Q1 patch, the current outward
normal determines the signed distance. The normal is reconstructed from that
patch's bilinear geometry; no triangle surrogate or cached force is used.
Current bilinear flux volume checks orientation for a connected shell.
Disconnected/nested components, edges, vertices, near-zero offsets and unresolved
alignment retain the existing independently checked rays. Arbitrary surface
self-intersection detection is not supplied by this optimization.
The smooth-interior sign argument is discussed by
[Bærentzen and Aanæs](https://doi.org/10.1109/TVCG.2005.49). Their triangular
edge/vertex pseudonormal construction is not substituted for the Q1 fallback.

The default remains `sign_policy="two_rays"`. Both policies use the same
closest-point search, ambiguity decisions, interpolation, forces and state
transactions. Only this core-owned arithmetic choice is excluded from the
accepted-coordinate checkpoint identity; it remains visible in execution
provenance. Restore recomputes current geometry and discards derived caches.
Callback identities and other search settings are not silently made compatible.
Compare complete accepted steps before choosing a policy: reduced ray counts
do not by themselves prove a faster solve when body and contact work overlap.

## Compiled user materials

### Restart compatibility versus execution provenance

New checkpoints from the built-in native-material binding record a versioned
restart compatibility digest separately from the actual execution digest.
Material and cell worker counts, and static versus dynamic tile scheduling,
may change on resume because this ABI requires pointwise, batch-partition-
invariant material updates. Material code and parameters, state encoding,
dispatcher revision, numerical increment policy, mesh, mass, viscosity and
hourglass definitions remain checked. This is not MPI or cross-binary state
portability, nor permission to alter the numerical method.
The standalone built-in `explicit_hex_kernel` also permits changes to its
independent cell-worker budget, but still rejects changing native/NumPy
arithmetic or admitted auxiliary capabilities on resume.

Older v4 body records still require their original exact execution identity.
To move one to the new format, restore it with that compatible configuration
and save a **new** checkpoint before adjusting workers. Never edit the old
archive or its identity. Unknown schemas reject. The original execution
configuration remains visible in summaries and the saved record.

Custom `ExplicitHexBodyKernel` implementations keep exact-identity recovery
unless the author explicitly supplies a nonempty `restart_identity` mapping.
That declaration is a scientific compatibility responsibility, not automatic
verification: include everything that changes discrete updates or state
meaning. A kernel name is not enough. The descriptor is frozen independently
of caller-owned mappings. Core still validates restored arrays and performs
atomic rejection; this change does not bypass checkpoint integrity checks.

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

Revision 5 reuses sleeping native workers for the material and auxiliary
phases, with the same fixed contiguous cell slices and per-cell arithmetic.
The executor owns neither material state nor retained numerical buffers. Each
batch completes before repair, publication, rollback or error propagation.
It refuses concurrent/reentrant use, close while busy and reuse after fork.
Use the kernel as a context manager or call `kernel.close()` in the provider's
cleanup; close is idempotent and previously returned immutable arrays remain
valid. Object destruction is a cleanup backstop, not a substitute for explicit
ownership. Single-worker bindings start no background worker. No process-global
pool, scheduling-interval change or automatic CPU-budget selection is made.

Revision 6 fuses finite output, Cauchy symmetry and nonnegative hourglass-energy
checks into the body workers before publication. Repaired points pass the same
checks after exact repair. The checked built-in dispatcher emits immutable byte
storage; its internal transport no longer scans those same arrays again through
ordinary public constructors. External callbacks still use full validation.
No constitutive arithmetic, acceptance threshold or physical time grid changes.
This revision is recorded in execution identity; retain the previous runtime
for an old checkpoint rather than rewriting that identity. Local equivalence
tests alone are not a whole-run timing claim.

`tile_size=N` is an opt-in material scheduling experiment. Instead of assigning
one equal slice to every worker, joined workers claim disjoint contiguous tiles
from an instance-local queue. This can reduce slow-worker tails on heterogeneous
CPUs or uneven material workloads; small tiles can instead increase dispatch
overhead. The default `None` preserves static slices and existing execution
identity. No automatic hardware tuning or speed guarantee is implied.

The trusted provider must produce the same pointwise response independently of
batch partitioning. Stress, state, work and body arithmetic retain original cell
order; this is not a parallel reduction. Exact repair runs only after all tiles
join, preserves global rejected-point ordering, and repairs only those points.
Queue scheduling and tile size remain recorded in execution provenance. New
versioned body records allow these operational changes under the partition-
invariance contract above; legacy records still require exact identity.
Test representative damaged states as well as startup before choosing a tile size.
The approach uses the existing pthread executor, without adding OpenMP or oneTBB
dependencies; their [dynamic scheduling](https://openmp.org/spec-html/5.0/openmpsu41.html)
and [chunking tradeoffs](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/Automatic_Chunking.html)
provide general design context, not evidence of an AgentFEM speedup.

### Optional joined body and bulk-viscosity execution

`native_explicit_hex_kernel(..., fuse_viscosity=True)` (also available through
`StrainHistoryRuntime`) evaluates the existing bulk-viscosity arithmetic in
the material/body cell pass. The default remains `False`. Each worker gathers
its cell velocity into local scratch rather than creating another full-mesh
cell-velocity array and launching a separate auxiliary pass. This option
requires the installed `run_damped_checked` dispatcher; an older native build
rejects the explicit request instead of silently ignoring it.

The material law, internal work, viscosity coefficients, dissipative power,
hourglass response and damping stability bound are unchanged. Body and
viscosity remain separate physical contributions, despite sharing execution.
All workers and exact material repairs finish before publication; invalid
body or damping output rejects the entire trial. Ill-conditioned viscosity
conversion retains the reference fallback without repeating material updates.
No extra state owner, time integrator or global worker pool is introduced.

Execution provenance records the selected policy. The versioned restart
contract permits switching this policy because the per-cell arithmetic and
scientific inputs remain unchanged; old exact-identity checkpoints must first
be restored with their compatible configuration and saved in the new format.
Cross-policy recovery, failed-trial rollback and static/tiled execution have
targeted regression coverage. This is an opt-in execution experiment, not a
whole-model speed guarantee: overlapping contact work can hide a body gain.
Select it using a representative whole-step comparison on the target machine.

```python
with elements.native_explicit_hex_kernel(
    material, c_function, properties=parameters, density=density,
    identity=implementation_identity, fallback=reference_update,
    workers=material_workers, cell_workers=cell_workers,
) as kernel:
    step = model.step(target=u, execution_kernel=kernel, **step_options)
    result = step.solve_result()
```

Rigid-contact residuals in serial execution use indexed boundary-node storage,
rather than allocate a full-volume zero field for every contact pair. The
backend-neutral trace retains its dense default; `assemble(..., compact=True)`
returns explicitly numbered rows and the full logical node count.
`dense_nodal_residual()` materializes the original view if needed. Only exact
structural zeros are omitted: quadrature, contact response, contribution order,
force/moment and energy checks remain unchanged. MPI assembly retains the
existing dense local-plus-ghost path.

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
`native` fails if its required extension is unavailable. Only the worker and
scheduling choices explicitly covered by the versioned restart contract may
change across a restart. An arithmetic implementation change requires a
separately reviewed migration or a fresh run, not removal of identity checks.

Native Q1 geometry declares independent, ordered query partitioning. New
deformable-contact checkpoints (v2) therefore allow changing projection and
exact-geometry worker counts while retaining the arithmetic revision, exact
arithmetic provider, geometry, law, candidate coverage, search limits and reuse
policy. Runtime summaries still record the actual worker counts. Derived search
data is rebuilt from accepted coordinates; no old projection or force is reused.
Legacy v1 checkpoints remain exact-configuration records: load with the original
settings and save a new checkpoint before tuning workers. External
`Q1ContactExecution` providers remain strict unless they explicitly supply a
`restart_identity` compatibility declaration. Such a declaration is a trusted
provider contract, not an automatic equivalence certificate.

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
