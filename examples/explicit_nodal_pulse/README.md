# Explicit nodal half-sine pulse

Run `agentfem run` inside this folder for the standard project/result manifest.
Alternatively, in the tested AgentFEM/FEniCSx environment:

```sh
python examples/explicit_nodal_pulse/case.py --output outputs/pulse --element tet15_nodal
```

Use `--element tet10_hrz` for ordinary P2 with HRZ inertia. Both names are
independent methods; neither means Abaqus C3D10M. The case is a clamped 3D steel
block in mm–N–tonne–s, with a true corner nodal force and a 25 μs half-sine pulse.
It is an integration demonstration, not the blade or a convergence claim.

Outputs: `fields.xdmf`, `fields.h5`, `sensor.csv`, `sensor.json`, `run.json`.
For enriched Tet15, all native nodal values and source cell DOFs remain in HDF5;
the XDMF display uses explicitly labeled linear subdivisions. The complete
field-output example currently runs serially. MPI nodal histories/restart have
a separate verification driver in `validation/dynamic_parity`.
