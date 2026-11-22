# {{PROJECT_NAME}}

3D SI linear cantilever: U/S fields, interpolated tip_U2, fixed support-on-body reaction. Reaction y is positive under negative-y traction. Use the manifest-recommended visualization file; Warp uses reference geometry exactly once.

agentfem check
agentfem run
agentfem show --json

Small composition/teaching fixture, not a mesh-convergence certificate.

Before solving, run `python case.py --review` in the installed environment. This builds the actual model and prints a JSON review card; boundary quadrature may compile, but no Step is solved or run result published. Ordinary `agentfem check` remains a structure/runtime check.
