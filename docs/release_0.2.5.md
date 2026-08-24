# AgentFEM 0.2.5

AgentFEM 0.2.5 closes two user-facing gaps without changing the physical
meaning of supported models: two-dimensional results are directly usable for
deformed-shape visualization, and incompatible constraint/procedure
combinations are rejected before numerical assembly.

## Direct 2D deformation visualization

A two-dimensional displacement remains a two-component finite-element
unknown. When the XDMF geometry is stored as XYZ, AgentFEM now writes the
visualization representation as `(Ux, Uy, 0)`. The HDF5/XDMF field therefore
has three components and can be selected directly by ParaView or PyVista Warp
By Vector. Three-dimensional fields are unchanged.

`U` remains the standard finite-element symbol stored in the field file;
`Displacement` remains its public scientific name. The result manifest records
that alias together with physical/model dimension, storage components and
`warp_compatible=true`, avoiding a duplicated full-size field.

## Constraint preflight

Constraints now publish one inspectable capability record covering supported
analyses, solution procedures, strictness and MPI scope. `model.step(...)`
checks that record before lowering. A projection-only periodic constraint
cannot accidentally enter an implicit Newmark system and fail later through a
Dirichlet-only backend call.

The public rectangular `dolfinx_mpc` constructor is shared with the
PDEAgent-Bench integration. It exposes a strict distributed MPC object without
claiming that every Step provider already owns MPC assembly; provider-specific
consumption remains explicit.

## Runtime and promotion evidence

`agentfem doctor` distinguishes the code being executed from stale installed
distribution metadata and names the MPI launcher matching the active
`mpi4py`. Platform acceptance now retains Linux/macOS two-rank installed-wheel
records. Real WSL2 evidence remains fail-closed.

The deterministic release smoke and a fresh AI-agent behavioral trial are now
separate evidence classes. AgentFEM can prove that its machine entrypoints work
without pretending that an unfamiliar agent has independently understood and
explained a model.

## Compatibility

The public grammar remains:

```text
Study -> Model -> scientific assets -> model.step(...) -> SimulationResult
```

Existing 2D result files are immutable and still require a Calculator-created
three-component vector or regeneration with 0.2.5. No constitutive capability
is promoted by this maintenance release.
