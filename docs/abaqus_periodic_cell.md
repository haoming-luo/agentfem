# Abaqus C3D10 Periodic Cell

## What This Capability Proves

The `examples/abaqus_c3d10_periodic_cell/` workflow exercises several layers
together:

1. a custom-extension `.dat` file is read explicitly as Abaqus syntax;
2. `C3D10` is mapped to meshio `tetra10` and read by DOLFINx as a quadratic
   tetrahedral geometry;
3. Abaqus node labels are retained independently of DOLFINx ordering;
4. all linear `*EQUATION` terms are parsed, validated, and matched to vector
   displacement degrees of freedom;
5. chained edge and corner equations become an exact affine reduction;
6. a 3D compressible Neo-Hookean equilibrium is solved incrementally;
7. XDMF/HDF5 time-series fields, VTU, PNG, GIF/MP4, exact macro histories,
   AF-IR, and convergence evidence are written from one top-level model.

The constraint has the algebraic form

```text
u = T q + u_bar(F_macro)
```

and the nonlinear equations are reduced variationally:

```text
r(q) = T^T R(u)
K_reduced(q) = T^T K(u) T
```

This is the same elimination principle used by Abaqus/Standard linear
equations, expressed as an inspectable AgentFEM object.

## Scientific Boundary

The original Abaqus deck is not fully reproducible from the files supplied.
Its `*USER MATERIAL` and `*MPC,USER` implementations are absent. AgentFEM
therefore preserves the mesh and linear periodic equations, but deliberately
substitutes:

- a documented compressible Neo-Hookean energy;
- a complete prescribed macroscopic deformation gradient.

This is a meaningful interoperability and finite-deformation validation case.
It is not a material-response comparison with the original user subroutines.

## Public Workflow

```python
cell = mesh.read_abaqus_mesh(
    "R1f10n30vc.dat",
    "output/mesh.xdmf",
    cell_type="tetra10",
)
equations = mesh.abaqus.read_equations("R1f10n30vc.mpc")
u = model.field(fields.displacement(cell.domain, degree=2))

periodicity = model.constraint(
    constraints.abaqus_periodic_cell(
        u,
        nodes=cell.nodes,
        equations=equations,
        deformation_gradient=F_macro,
        anchor_node=1,
        reference_nodes=(7, 9, 4),
    )
)
output = results.field_output(
    "U", "S", "E", "EVOL",
    every="increment",
    configuration="deformed",
    backend="xdmf",
)
incrementation = steps.automatic(
    initial=0.25,
    minimum=1.0e-4,
    maximum=0.5,
    max_increments=10,
)
step = model.step(
    target=u,
    constraints=periodicity,
    incrementation=incrementation,
    output=output,
    solver_options=AffineNewtonOptions(max_it=25),
    status_file="output/periodic_cell.sta",
)
result = step.solve_result()
output.write_finite_strain(
    "output",
    domain=cell.domain,
    snapshots=step.snapshots,
    material=material,
)
```

## One step, several increments, several frames

This case defines one nonlinear static Step. Its automatic controller chooses
the accepted Increment count from convergence behavior. `max_increments=10` is
a safety limit, not a request to perform ten increments. A failed attempt is
rolled back and retried with a smaller increment.

`every="increment"` saves the initial state and every accepted Increment. This
matches the meaning of Abaqus `*OUTPUT, FIELD, FREQUENCY=1`: output frequency
is counted in increments, not analysis steps. The final state is always
retained. If six evenly spaced result states are required independently of
automatic solver decisions, use `intervals=6`; these are output marks, and the
saved states become Frames only in the result dataset.

The console reports preprocessing, Step, Increment, Attempt, Newton Iteration,
postprocessing, and output stages. `periodic_cell.sta` is flushed after each
accepted increment or cutback so a long run can be monitored without waiting
for completion.

## Spatial fields and macro histories

| Abaqus request | AgentFEM result | Storage |
| --- | --- | --- |
| `U` | displacement | unified XDMF/HDF5 |
| `S` | Cauchy stress | unified XDMF/HDF5 + macro NPZ/CSV |
| `E` in finite strain | spatial logarithmic strain `LE` | unified XDMF/HDF5 |
| `GREEN` | explicit Green--Lagrange strain | when requested |
| `EVOL` | current element volume | unified XDMF/HDF5 |
| `P` | first Piola stress | unified XDMF/HDF5 + macro NPZ/CSV |
| `F`, `J` | deformation gradient and determinant | unified XDMF/HDF5 |
| `MISES`, `SENER` | equivalent stress and strain energy | unified XDMF/HDF5 |
| `SDV` | not applicable to the current stateless Neo-Hookean law | — |

The result manifest additionally stores vector `U` and current `COORD`
histories at Abaqus control nodes `RIGHT`, `TOP`, and `FRONT`. `RF` and `TF`
are deliberately not approximated from homogenized stress: exact affine-MPC
reaction recovery requires verified constraint multipliers and is listed as a
remaining result capability.

The unified XDMF/HDF5 series is both the direct ParaView product and the
scientific field record: topology is shared, reference coordinates are
retained, every frame stores coordinates `x+u`, and all fields share that
grid. The authoritative homogenized
response is computed from the UFL forms before visualization projection:

```text
P_bar = (1 / V_cell) integral_over_solid(P dV0)
sigma_bar = (1 / (J_bar V_cell)) integral_over_solid(J sigma dV0)
P_bar_check = J_bar sigma_bar F_bar^-T
```

The complete cell volume appears in the denominator; the void carries zero
stress. Both first-Piola routes are evaluated and their maximum difference is
recorded. Exact histories go to NPZ and CSV, so plots and training data do not
depend on re-averaging visualization samples.

## Import pipeline

```text
Abaqus .dat ──meshio(file_format="abaqus")──> XDMF/HDF5 ──> DOLFINx
     └────────AgentFEM node-label parser───────────────┘
Abaqus .mpc ──AgentFEM *EQUATION parser───────────────> affine reduction
```

The neutral conversion makes topology selection and any set loss visible in a
manifest. Labels and equations are retained separately because generic mesh
conversion cannot preserve every periodic-constraint requirement.

## Evidence and Failure Checks

A credible run should report:

- 14,942 imported nodes and 8,781 `tetra10` cells;
- 4,212 equation constraints and 40,602 independent displacement dofs;
- Newton convergence for every load factor;
- a near-machine-zero equation mismatch;
- positive sampled `det(F)` values;
- finite averaged first-Piola and Cauchy stresses;
- consistency between two homogenized stress transformations;
- an undeformed/deformed VTU pair, a scale-one comparison image, and an
  incremental animation.

The code rejects duplicate equation slaves, cyclic equation graphs, missing
node-to-dof matches, non-positive target `det(F)`, non-finite residuals, and
parallel execution of the current serial elimination backend.

## Current Engineering Limit

The exact equation backend is serial. The example rejects `mpiexec -n 2`
instead of silently running two duplicate serial jobs. Sparse direct MUMPS is the trustworthy
reference policy for the present 40k-dof example. Production distributed
support needs:

- global node-label/dof ownership;
- distributed construction of `T` and `u_bar`;
- ghost-consistent chained-equation resolution;
- reduced near-nullspace transfer for AMG;
- MPI validation of reactions, homogenized stress, and periodic mismatch.

The public constraint semantics need not change when that backend is added.
`dolfinx_mpc` is the leading backend candidate, but it is a compiled optional
dependency and the current DOLFINx 0.11 environment does not contain it.
AgentFEM will advertise parallel Abaqus-equation support only after the
arbitrary chained equations and finite-strain solve pass two-rank parity tests.

## User-material migration is a separate capability

Mesh/equation import does not imply that arbitrary `UMAT` or `UHYPER` source
already runs. `docs/abaqus_user_material_bridge.md` defines the required
quadrature state, tensor conversions, compiler boundary, and validation
ladder. UHYPER is the narrower first target; stateful UMAT integration follows
only after a global constitutive driver exists.
