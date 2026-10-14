# Custom nonlinear potential with standard results

Run `agentfem run --project examples/custom_energy_output` from the repository,
or run `agentfem run` inside a copied project with the installed package.

The dimensionless example uses a convex quadratic-plus-quartic strain energy,
P2 displacement and a body load on a clamped rectangle. Each converged load
level produces DG0 stress and energy through the public projection API.

Open `fields.pvd` or `fields.xdmf` in ParaView. Both contain five load levels;
`U` is the Warp vector. Geometry is undeformed, so Warp scale 1 displays actual
computed displacement. The output record identifies the load-factor axis.

No custom writer or private solver internals are used. This example demonstrates
output integration; it is not a calibrated engineering material model.
