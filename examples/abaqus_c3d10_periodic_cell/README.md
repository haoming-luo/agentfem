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

The output request is declared once:

```python
field_output = results.field_output(
    "U", "S", "E", "EVOL", "F", "P", "MISES", "J", "SENER",
    every="increment",
    configuration="deformed",
    backend="xdmf",
)
```

For finite strain, the conventional request `E` resolves to spatial
logarithmic strain `LE`. Green--Lagrange strain is available explicitly as
`GREEN`; it is not mislabeled as Abaqus `E`.

The default plan writes one logical ParaView result dataset:

- `periodic_cell.xdmf`: the small temporal index;
- `periodic_cell.h5`: compressed topology, retained reference coordinates,
  deformed geometry at every frame, `U`, `UMAG`, `S`, `LE`, `EVOL`, `F`, `P`,
  `MISES`, `J`, and `SENER`.

Open the XDMF in ParaView to play the actual deformation and switch fields
without a Warp filter or multi-block selection. No directory of per-frame VTU
files is required. `SDV` is absent because the substituted Neo-Hookean model
has no state variables.

The visualization and scientific-history products have different jobs:

- `periodic_cell_deformation.gif` shows the true-scale incremental shape;
- `periodic_cell_comparison.png` compares undeformed and final states;
- `homogenized_history.npz` stores integrated tensor histories for Python/ML;
- `homogenized_history.csv` provides the macro response in a transparent table;
- `homogenized_response.png` plots effective stress against macro strain;
- one unified XDMF/HDF5 series serves both scientific traceability and direct
  deformed visualization.

The result manifest also records vector `U` and current `COORD` histories at
the original Abaqus `RIGHT`, `TOP`, and `FRONT` control nodes. `RF` and `TF`
are not fabricated: under exact affine MPC elimination they require recovery
and verification of constraint multipliers, which remains a separate result
capability.

Run `postprocess_homogenized_response.py` to reproduce the macro plot. It reads
NPZ rather than re-averaging XDMF cell-center samples: the NPZ history retains
quadrature-integrated stresses, complete-RVE normalization, and the consistency
check between direct first-Piola integration and the Cauchy-stress transform.
The supplied `avestress.py` remains an Abaqus/ODB reference implementation.

## What happens to the `.dat` mesh

AgentFEM directly parses Abaqus node labels and the separate `.mpc` equations.
For topology it explicitly asks meshio to parse the custom-extension `.dat`
file as Abaqus syntax, writes a documented XDMF/HDF5 conversion, and lets
DOLFINx read that converted mesh. The workflow is therefore an explicit source
→ neutral mesh → DOLFINx pipeline, not a hidden native Abaqus reader.

Current limitation: exact Abaqus equation elimination is serial. The
scientific constraint object and reduction are reusable, but distributed
ownership/ghost handling needs a dedicated parallel MPC backend. The example
rejects a multi-rank launch rather than presenting duplicated serial jobs as
parallel computation. Other DOLFINx AgentFEM examples remain true MPI runs.
