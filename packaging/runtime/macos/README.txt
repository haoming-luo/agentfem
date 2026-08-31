AgentFEM Runtime for macOS

This package installs AgentFEM together with its compatible Python,
FEniCSx/DOLFINx, PETSc, MPI, HDF5, and form-compilation runtime. It does not
replace the system Python and does not modify the user's shell profile.

The recommended Complete profile also installs the separately licensed Gmsh
mesh generator and Python API, so CAD/STEP-to-mesh workflows work without a
second package installation. The lean Core profile excludes Gmsh. The exact
profile, package lock, component licenses, and source references are recorded
with every installer. AgentFEM itself remains Apache-2.0; bundled third-party
components retain their own licenses.

After installation, open "AgentFEM Terminal.command" from the Applications
folder. The launcher activates the private runtime, runs a health check on its
first launch, and opens a terminal ready for AgentFEM projects.

User projects are stored outside the runtime and are never removed when the
runtime is upgraded or uninstalled.
