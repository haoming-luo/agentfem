# Abaqus C3D10H periodic cell

This example demonstrates a quasi-incompressible, finite-deformation periodic
cell assembled from reusable AgentFEM capabilities:

- the 14,942 source node labels and 8,781 quadratic tetrahedra;
- the `MATRIX` element set;
- all 4,212 linear `*EQUATION` constraints, including chained edge/corner
  relations;
- direct Abaqus `C3D10H` import with preserved hybrid/constant-pressure
  identity;
- a P2-displacement/DG0-pressure mixed solution formulation;
- quasi-incompressible Neo-Hookean finite-deformation kinematics;
- three-dimensional uniaxial-stress control through RIGHT/TOP/FRONT reference
  points;
- incremental nonlinear equilibrium and inspectable convergence evidence.

The material is declared by its physical moduli. With default
`bulk_to_shear_ratio=1e4`, its pressure-condensed stored energy is

```text
W = mu/2 * (J^(-2/3) I1 - 3) + kappa/2 * (J - 1)^2
```

which maps to Abaqus polynomial parameters `C10=mu/2` and `D1=2/kappa`.
AgentFEM solves pressure as one independent constant value per cell rather
than forcing a nearly incompressible penalty through displacement alone.
RIGHT-U1 prescribes the axial extension, while TOP-U2 and FRONT-U3 are solved
as independent transverse macro degrees of freedom. All macro shear
components are zero. This produces zero transverse resultants without
pre-imposing an isochoric lateral stretch.

The versioned `u=0.2` reference run recovers
`F=diag(1.2, 0.9178618773, 0.9180572899)`, with both `P22` and `P33` below
`2e-11` in magnitude. The two independently solved lateral stretches need
not be exactly equal for a finite random cell; their difference is below
`2e-4` in this realization.

The case has a deliberately small public layout:

```text
case.py                              top-level AgentFEM model
input/periodic_cell_c3d10h.dat       direct Abaqus C3D10H mesh
input/periodic_cell_equations.mpc    periodic equations
postprocess.py                       AgentFEM history plot
abaqus_reference/                    original Abaqus comparison files
```

The top-level model is [`case.py`](case.py).
Run the default C3D10H case with an axial displacement of `0.2`:

```bash
python case.py
```

For a smaller loading smoke test:

```bash
python case.py --displacement 0.01
```

Material scale and compressibility remain explicit controls:

```bash
python case.py \
  --shear-modulus 1.0 --bulk-to-shear-ratio 10000
```

The C3D10H mixed periodic route is currently serial. Distributed
displacement-only periodic equations remain a separate tested capability.

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
        "U", "S", "E", "EVOL", "F", "P", "PRESSURE", "MISES", "J", "SENER",
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
For the mixed route, `PRESSURE` is the independent positive-in-compression
DG0 unknown; `P` remains the first-Piola stress tensor. They are intentionally
different names.

In serial, the default plan writes one logical ParaView result dataset:

- `periodic_cell.xdmf`: the small temporal index;
- `periodic_cell.h5`: compressed topology, retained reference coordinates,
  deformed geometry at every frame, `U`, `UMAG`, `S`, `LE`, `EVOL`, `F`, `P`,
  `PRESSURE` when applicable, `MISES`, `J`, and `SENER`.

Open the XDMF in ParaView to play the actual deformation and switch fields
without a Warp filter or multi-block selection. No directory of per-frame VTU
files is required. This hyperelastic material is stateless, so no `SDV` field
is generated.

The visualization and scientific-history products have different jobs:

- `periodic_cell_deformation.gif` shows the true-scale incremental shape;
- `periodic_cell_comparison.png` compares undeformed and final states;
- `homogenized_history.npz` stores integrated tensor histories for Python/ML;
- `homogenized_history.csv` provides the macro response in a transparent table;
- one unified XDMF/HDF5 series serves both scientific traceability and direct
  deformed visualization.

The result manifest also records vector `U` and current `COORD` histories at
the `RIGHT`, `TOP`, and `FRONT` control nodes.

Run `postprocess.py` to create a case-specific macro plot.
It reads NPZ rather than re-averaging XDMF cell-center samples: the NPZ history
retains quadrature-integrated stresses, complete-RVE normalization, and the
consistency check between direct first-Piola integration and the Cauchy-stress
transform. The plot is intentionally not generated by the generic AgentFEM
result layer. The files under `abaqus_reference/` are comparison provenance,
not AgentFEM runtime inputs.

## What happens to the `.dat` mesh

AgentFEM reads `input/periodic_cell_c3d10h.dat` directly. It parses the Abaqus
node labels and `C3D10H` declaration, and parses the separate `.mpc` equations.
For geometry it asks meshio to interpret the custom-extension `.dat` file as
Abaqus syntax, writes a documented XDMF/HDF5 conversion, and lets DOLFINx read
the resulting `tetra10` mesh. The conversion manifest retains the original
`C3D10H` hybrid/constant-pressure identity even though neutral XDMF records
only geometric topology.

The case then selects AgentFEM's verified P2/DG0 mixed provider. A direct
`C3D10H` source cannot be silently consumed by the ordinary displacement-only
hyperelastic route: formulation validation rejects that mismatch. No derived
Abaqus mesh is created by this example.

Serial execution uses AgentFEM's explicit sparse affine transformation. The
same constraint module also provides a `dolfinx_mpc` backend for distributed,
fully prescribed displacement formulations.
