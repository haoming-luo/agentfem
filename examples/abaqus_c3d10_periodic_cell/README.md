# Abaqus C3D10 periodic cell → AgentFEM finite deformation

This example is a migration and validation workflow, not a claim that two
different constitutive models are identical.

What is preserved and tested:

- the 14,942 Abaqus node labels and the 8,781 `C3D10` quadratic tetrahedra;
- the `MATRIX` element set;
- all 4,212 linear `*EQUATION` constraints, including chained edge/corner
  relations;
- three-dimensional finite-deformation kinematics;
- incremental nonlinear equilibrium and inspectable convergence evidence.

What is deliberately changed:

- `geom.inp` uses `*USER MATERIAL`; its implementation is not in this folder;
- `*MPC,USER` likewise depends on an unavailable user subroutine;
- AgentFEM therefore uses its explicit compressible Neo-Hookean model and
  prescribes the complete macroscopic deformation gradient directly.

The top-level model is
[`agentfem_periodic_hyperelastic.py`](agentfem_periodic_hyperelastic.py).
Run a quick verification with:

```bash
python agentfem_periodic_hyperelastic.py --stretch 1.05
```

Run the visible finite-deformation case with:

```bash
python agentfem_periodic_hyperelastic.py --stretch 1.20
```

Run one distributed simulation across two MPI ranks with:

```bash
mpiexec -n 2 python agentfem_periodic_hyperelastic.py --stretch 1.20
```

This is within-case parallelism: DOLFINx partitions the mesh, every rank
assembles its cells, `dolfinx_mpc` distributes the chained equation graph, and
PETSc/MUMPS solves one distributed system. It is not two independent copies of
the same serial job. `dolfinx_mpc` must match the installed DOLFINx minor
version.

The main Python file contains the visible, Abaqus-style analysis controls:
automatic initial/minimum/maximum increment sizes, a maximum of ten accepted
increments, cutback limits, and the maximum Newton iterations per attempt.
Ten is an upper bound, not a requested count. AgentFEM may reach the target in
fewer increments when convergence is fast.

With `every="increment"`, AgentFEM saves the initial configuration and every
accepted increment. The final converged increment is always retained. To ask
for six evenly spaced result intervals independently of automatic solve
increments, change the output request to `intervals=6`.

## Result products

The complete output contract is declared once:

```python
output = results.output_plan(
    output_directory,
    field=results.field_output(
        "U", "S", "E", "EVOL", "F", "P", "MISES", "J", "SENER",
        every="increment",
        configuration="deformed",
        backend="xdmf",
    ),
    requests=(
        results.solver_history(),
        results.periodic_cell_history(periodicity),
        results.source_node_history(nodes, RIGHT=7, TOP=9, FRONT=4),
        results.finite_strain_checks(constraint=periodicity),
    ),
    presentation=results.presentation(animation="gif"),
)
```

For finite strain, the conventional request `E` resolves to spatial
logarithmic strain `LE`. Green--Lagrange strain is available explicitly as
`GREEN`; it is not mislabeled as Abaqus `E`.

In serial, the default plan writes one logical ParaView result dataset:

- `periodic_cell.xdmf`: the small temporal index;
- `periodic_cell.h5`: compressed topology, retained reference coordinates,
  deformed geometry at every frame, `U`, `UMAG`, `S`, `LE`, `EVOL`, `F`, `P`,
  `MISES`, `J`, and `SENER`.

Open the XDMF in ParaView to play the actual deformation and switch fields
without a Warp filter or multi-block selection. No directory of per-frame VTU
files is required. `SDV` is absent because the substituted Neo-Hookean model
has no state variables.

Under MPI, collective DOLFINx I/O writes
`periodic_cell_parallel.xdmf` and its HDF5 heavy data. This scientific record
contains the reference mesh, displacement, and all requested cell fields for
every accepted increment. Use ParaView's Warp By Vector for presentation.
Directly deformed compact geometry, PNG, and GIF/MP4 rendering remain serial
postprocessing products; they are intentionally not produced concurrently by
multiple ranks.

The visualization and scientific-history products have different jobs:

- `periodic_cell_deformation.gif` shows the true-scale incremental shape;
- `periodic_cell_comparison.png` compares undeformed and final states;
- `homogenized_history.npz` stores integrated tensor histories for Python/ML;
- `homogenized_history.csv` provides the macro response in a transparent table;
- one unified XDMF/HDF5 series serves both scientific traceability and direct
  deformed visualization.

The result manifest also records vector `U` and current `COORD` histories at
the original Abaqus `RIGHT`, `TOP`, and `FRONT` control nodes. `RF` and `TF`
are not fabricated: under exact affine MPC elimination they require recovery
and verification of constraint multipliers, which remains a separate result
capability.

Run `postprocess_homogenized_response.py` to create a case-specific macro plot.
It reads NPZ rather than re-averaging XDMF cell-center samples: the NPZ history
retains quadrature-integrated stresses, complete-RVE normalization, and the
consistency check between direct first-Piola integration and the Cauchy-stress
transform. The plot is intentionally not generated by the generic AgentFEM
result layer. The supplied `avestress.py` remains an Abaqus/ODB reference
implementation.

## What happens to the `.dat` mesh

AgentFEM directly parses Abaqus node labels and the separate `.mpc` equations.
For topology it explicitly asks meshio to parse the custom-extension `.dat`
file as Abaqus syntax, writes a documented XDMF/HDF5 conversion, and lets
DOLFINx read that converted mesh. The workflow is therefore an explicit source
→ neutral mesh → DOLFINx pipeline, not a hidden native Abaqus reader.

Serial execution uses AgentFEM's explicit affine transformation. MPI execution
flattens the same chained equation graph to independent masters, maps Abaqus
labels to global DOLFINx dofs, supplies every owned and ghost slave relation,
and uses `dolfinx_mpc` for distributed assembly and back-substitution. The two
paths are required to match in reduced residual, Newton convergence, and
periodic mismatch. AMG near-nullspace transfer and affine-MPC reaction recovery
remain future parallel capabilities.
