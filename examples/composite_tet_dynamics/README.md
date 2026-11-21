# Composite Tet10 public workflow

This minimal project demonstrates the public Study → mesh → field formulation →
material → constraints/load → Step → result workflow. It is not a blade case or
an Abaqus comparison. SI-consistent parameters are deliberately simple.

Run with the standard `agentfem run` command from this project, or under MPI via
`mpiexec -n 2 python case.py` in the installed numerical environment.

The explicit field declaration selects the composite provider automatically:

```python
u = model.field(fields.displacement(domain,
    element_policy=elements.composite_tet10_policy(mean_dilatation=True)))
step = model.step(target=u, dt="auto", steps=5)
```

Outputs: XDMF/HDF5 with exact twelve-subtet displacement interpolation; native
coefficients; U/V/A; cell-average S/E/MISES/SENER; CSV histories; standard result
manifest. ParaView can Warp By Vector using U directly. No Extract Block or
P2 reinterpretation is necessary. MPI output gathers onto rank zero.

Supported: affine straight-sided tetrahedra, constant linear elastic materials
and material regions, nodal forces, spatially constant body force/traction and
reference pressure, scalar amplitudes, strong fixed/moving Dirichlet constraints,
same-partition restart. Stress is reported as a cell average, not a peak estimate.

Unsupported requests are rejected: curved geometry, plasticity/contact, follower
pressure, custom pressure normals, general spatial load expressions, periodic
constraints, cross-partition restart. This formulation is **not C3D10M**.
See `docs/theory/composite_tetrahedron.md` for derivation and verification.
