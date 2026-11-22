# {{PROJECT_NAME}}

Serial SI homogeneous hyperelastic displacement loading without natural loads/contact/MPC. The curve uses retained accepted snapshots only; mean boundary displacement is not a point probe. Reaction is at the loaded end, global +x. No interpolation of missing stations.

agentfem check
agentfem run
agentfem show --json

Small composition/teaching fixture, not a mesh-convergence certificate.

Before solving, run `python case.py --review` in the installed environment. This builds the actual model and prints a JSON review card; boundary quadrature may compile, but no Step is solved or run result published. Ordinary `agentfem check` remains a structure/runtime check.
