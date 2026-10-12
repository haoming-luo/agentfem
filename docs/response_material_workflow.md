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

The tests exercise ordinary Step selection, response contracts, moving-tool
work, damping/scaling combinations and exact checkpoint continuation on tiny
meshes. They do not establish full industrial-case equivalence or speed.

## Remaining promotion gates

- Initial bonding and contact-overlap semantics through existing interface laws.
- Instance-local optimized body execution, without process-wide monkey patches.
- Accepted-state ownership with explicit field publication for all observers.
- A separately frozen installed project using these interfaces, followed by a
  complete timed run and comparison at identical accepted stations.

The earlier frozen private-backend run is a baseline, not an end-to-end test of
this new workflow. Confidential source laws and meshes remain external.
