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
5. chained edge and corner equations become an exact serial or distributed
   affine reduction;
6. a C3D10H-equivalent P2/DG0 mixed formulation solves a quasi-incompressible
   Neo-Hookean equilibrium incrementally;
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

## Scientific definition

The matrix follows the quasi-incompressible Neo-Hookean energy used in the
porous-composite study:

```text
W = mu/2 * (J^(-2/3) I1 - 3) + kappa/2 * (J - 1)^2
kappa = 10^4 mu
```

The mixed potential introduces one constant pressure unknown per cell. At
stationarity, eliminating pressure recovers the energy above. The
RIGHT/TOP/FRONT control nodes declare three-dimensional uniaxial stress:
RIGHT-U1 is `0.2`, TOP-U2 and FRONT-U3 are solved, and all macroscopic shear
components are zero. Thus both transverse resultants vanish independently.
The actual macroscopic deformation gradient is reconstructed from the
converged control-node motion at every frame. A finite random RVE may produce
slightly different transverse stretches; equality emerges with effective
transverse isotropy and is not imposed as an extra kinematic constraint.
The versioned `u=0.2` reference run gives
`F=diag(1.2, 0.9178618773, 0.9180572899)`, while both `P22` and `P33` are
below `2e-11` in magnitude.

## Public Workflow

```python
cell = mesh.read_abaqus_mesh(
    "R1f10n30vc.dat",
    "output/mesh.xdmf",
    cell_type="tetra10",
)
equations = mesh.abaqus.read_equations("R1f10n30vc.mpc")
unknown = model.field(fields.displacement_pressure(cell.domain))
material = model.material(
    constitutive.mixed_neo_hookean(
        shear_modulus=1.0,
        bulk_modulus=1.0e4,
    )
)

periodicity = model.constraint(
    constraints.abaqus_periodic_cell(
        unknown,
        nodes=cell.nodes,
        equations=equations,
        control_displacements=(
            (0.2, 0.0, 0.0),
            (0.0, None, 0.0),
            (0.0, 0.0, None),
        ),
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
    target=unknown,
    material=material,
    constraints=periodicity,
    incrementation=incrementation,
    output=output,
    solver_options=AffineNewtonOptions(max_it=25),
    status_file="output/periodic_cell.sta",
)
result = step.solve_result()
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
histories at control nodes `RIGHT`, `TOP`, and `FRONT`.

In serial, the unified XDMF/HDF5 series is both the direct ParaView product and
the scientific field record: topology is shared, reference coordinates are
retained, every frame stores coordinates `x+u`, and all fields share that
grid. Under MPI, DOLFINx writes a collective reference-configuration XDMF/HDF5
history with `U` and all requested cell fields; direct deformed rendering is a
separate serial postprocess. The authoritative homogenized
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

The checked-in source geometry declares `C3D10`. AgentFEM records the source
identity, derives the C3D10H formulation declaration without regenerating
nodes or connectivity, and selects its monolithic P2/DG0 provider. The DG0
field contributes one independent pressure unknown per cell. `tetra10`
describes geometry; the hybrid formulation remains an explicit scientific
choice.

## Evidence and Failure Checks

A credible run should report:

- 14,942 imported nodes and 8,781 `tetra10` cells;
- 4,212 equation constraints while both transverse macro dofs remain
  independent;
- Newton convergence for every load factor;
- a near-machine-zero equation mismatch;
- positive sampled `det(F)` values;
- finite averaged first-Piola and Cauchy stresses;
- consistency between two homogenized stress transformations;
- serial presentation products (comparison image and incremental animation),
  or a collective MPI XDMF field history.

The code rejects duplicate equation slaves, cyclic equation graphs, missing
node-to-dof matches, non-positive target `det(F)`, and non-finite residuals.

## Constraint backends

This C3D10H workflow uses an explicit sparse affine transformation so the
mixed pressure dofs and the free TOP-U2 and FRONT-U3 macro components remain
independent unknowns. The same public constraint object also lowers fully
prescribed displacement formulations to `dolfinx_mpc` for distributed
execution.
