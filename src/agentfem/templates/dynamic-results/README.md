# {{PROJECT_NAME}}

SI linear Newmark beam: six accepted increments, tip_U2 at every accepted station, and sparse field output every three increments. History time is in seconds; frame cadence does not change the solver clock.

agentfem check
agentfem run
agentfem show --json

Small composition/teaching fixture, not a mesh-convergence certificate.

Before solving, run `python case.py --review` in the installed environment. This builds the actual model and prints a JSON review card; boundary quadrature may compile, but no Step is solved or run result published. Ordinary `agentfem check` remains a structure/runtime check.
