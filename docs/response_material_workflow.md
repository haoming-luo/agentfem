# Response-only materials in an ordinary Step (experimental)

`constitutive.explicit_user_material` adapts a trusted user callback without
inventing a tangent or fixing the number of internal variables. The callback
receives a `MaterialPointArrayBatchInput` and returns an
`ExplicitMaterialBatchOutput`. Core validates the response, publishes trial
fields, and commits only after the complete increment succeeds. The callback
must not mutate accepted history or commit global state.

```python
material = constitutive.explicit_user_material(
    state_schema=schema, update=user_update,
    identity={"law": "my_law", "revision": "pinned-source-hash"},
    density=physical_density,
)
model.material(material)
step = model.step(
    target=u,
    element_policy=elements.uniform_strain_hex8(
        kinematics="finite_strain", hourglass_modulus=modulus,
    ),
    dt=dt, steps=steps, time_increment_ceiling=reviewed_ceiling,
    stress_free_initial_state=True,
)
result = step.solve_result()
```

The current admission is serial, three-dimensional, one full-domain material,
and finite uniform-strain Hex8. It starts from a stress-free, virgin state.
Multiple `RigidContactPair` model assets may use frictionless planes or cylinders
with prescribed translation. Cylinder projection retains the existing backend's
geometric limitations; this does not add end-cap, self-contact, or general
ALL EXTERIOR enforcement. Rotation, external loads, constraints, eigenstrains,
regional material dispatch and MPI are refused by this provider.

One `boundary_models.deformable_contact_pair` is admitted with explicit
bilinear Q1 traces, independent node maps and candidate selection. It reuses
the existing one-sided reference-area penalty operator, including conservative
path rejection. It is not symmetric ALL EXTERIOR, self-contact or hard contact.

`interfaces.reference_trace(..., geometry="bilinear")` preserves warped Q1
faces. `interfaces.initial_nodal_bond` binds an explicitly eligible, coincident
node set once; `fracture.nonmatching_cohesive_force` passes that pairing to the
Step as `cohesive_force`. Only isotropic elastic bonds are admitted here. The
default `bond_contact_policy="disjoint"` rejects shared trace nodes;
`"independent"` explicitly adds the declared contributions without switching,
rebonding or silently disabling either law. One-sided integration still needs
refinement and side-swap assessment.

Optional `boundary_models.RigidContactTracePolicy` selects Gauss or nodal
integration. Only uniform frictionless nodal laws may coalesce coincident
quadrature nodes by summing their reference areas. A cylinder transverse-strip
bound requires an explicit unit direction, displacement budget, and compatible
nonrotating motion. The budget is checked over every displacement node before
each evaluation; exceeding it rejects the trial rather than omitting contact.

`bulk_viscosity=elements.BulkViscosityPolicy(...)` selects the existing
volume-equivalent length rule. Its reference density must equal the physical
material density. Artificial viscosity work is not constitutive dissipation.

Optional `mass_scaling_factor` is a finite scalar or a nodal array of factors
at least one, in the current serial displacement-node order. It changes only
numerical inertia, not physical material density. Restart verifies inertia
identity; changing nodal order or scaling is not a portable continuation.
The summary records added mass. A successful scaled run does not establish
quasi-static accuracy or mass sensitivity.

The caller's increment ceiling is **not a stability certificate**, including
when contact and damping are present. This provider does not select an automatic
stable step. Internal work is reported as work, not relabelled stored energy.
`completed` does not imply independent physical verification.

`time.ExplicitSubstepPolicy` enables bounded dyadic retries for typed contact
path failures and optionally material cutback requests. Intermediate work is
recorded along accepted subincrements; a terminal failure rolls back the whole
macro increment. This serial path requires `history_every=1`. Ordinary unknown
exceptions remain failures, not reasons to keep trying. Retry counts describe
the current process and are not a cumulative restart history.

`elements.ExplicitHexBodyKernel` is an optional trusted execution extension.
It receives immutable arrays and fixed reference geometry, not finite-element
Fields or commit operations. Core retains increment checks, output validation,
trial publication and acceptance. The kernel identity participates in restart
admission. This removes the need to replace global methods for fused body
arithmetic; it does not itself prove a native kernel mathematically equivalent
or faster. Private constitutive constants and native binaries remain external.
An optional `prepare_kinematics` callback fuses the mean gradient and the
reference-perturbation sufficient screen. It must not certify cell validity
from the mean determinant alone. Uncertified cells retain core affine and
Bernstein checks, and all mean gradients retain positive-determinant admission.

The tests exercise ordinary Step selection, response contracts, moving-tool
work, damping/scaling combinations and exact checkpoint continuation on tiny
meshes. They do not establish full industrial-case equivalence or speed.

`boundary_models.Q1ContactExecution` supplies instance-local trusted closest
point, closed-solid classification, conservative distance-bound and swept-path
callbacks. Core still selects candidates,
resolves ambiguities, constructs forces and owns transactions. Callback results
are structurally checked; this is not a proof that a plugin found the global
minimum or classified a solid correctly. Identity is included in contact
restart admission. Reference fallbacks and refitted geometry keep the same
instance configuration; unrelated models are unchanged.
Distance and swept certificates are trusted numerical algorithms, not arbitrary
heuristics: uncertain geometry must use the reference proof or remain
unresolved. A false separation certificate can skip a physical contact and is
not made safe merely by returning a well-shaped boolean array.

`interfaces.project_reference_points` and `propose_reference_adjustment`
support explicit pre-model preparation. Adjustment returns a proposal, never
mutates a live bulk mesh, and does not infer bond eligibility. Apply through
original node identities, validate affected bulk cells and rebuild pairing.

`step.accepted_material_fields()` returns detached read-only cell records for
export: state schema, internal variables, Cauchy stress, deformation gradient
and cell identity. It rejects an active trial. Private state-variable names and
their interpretation remain the external material provider's responsibility.

`material_storage="owned_arrays"` is an explicit serial execution option for
this bounded Procedure. Immutable accepted arrays are the single material-state
owner; quadrature Functions are internal publication buffers. User material
callbacks must consume the supplied request only, not capture/read internal
quadrature Functions. Use `accepted_material_fields()` for observation and the
ordinary checkpoint API for persistence. Those boundaries publish accepted
data; rollback restores the accepted owner. Arbitrary internal-field observers
are not supported in this mode. The default `"fields"` keeps ordinary field
ownership. Both modes retain material validation and finite checks; this
choice does not change the law, time step or inertia. Multiple named state
variables are supported, not a fixed material-specific history length.

## Remaining promotion gates

- Full-case validation of the newly composed interfaces and migrated kernels.
- Remaining batch-search and bulk damping acceleration without process-wide
  patches; validate the newly wired geometry/path callbacks on the real case.
- Full-case timing and observation audit of the opt-in accepted-array owner.
- A separately frozen installed project using these interfaces, followed by a
  complete timed run and comparison at identical accepted stations.

The earlier frozen private-backend run is a baseline, not an end-to-end test of
this new workflow. Confidential source laws and meshes remain external.
