# Portable Projects and Execution Profiles

AgentFEM keeps one scientific project across laptops, workstations, and
clusters. Materials, meshes, loads, constraints, steps, outputs, and
verification rules remain in `case.py` and its project assets. The execution
profile only selects a runtime and a default MPI scale.

## One model, two execution routes

An installed template declares operational profiles in `agentfem.toml`:

```toml
[run]
output_directory = "outputs"
default_profile = "local"

[execution.local]
runtime = "auto"
ranks = 1

[execution.cluster]
runtime = "fenicsx-petsc"
ranks = 2
required_capabilities = ["mpi_distributed_mesh"]
```

The tables may contain only execution concerns. They do not redefine the
study, constitutive model, mesh, boundary conditions, solver tolerances, or
verification contract.

Run the project locally:

```bash
agentfem check --runtime
agentfem run
```

Run the same project through PETSc/MPI on a server:

```bash
agentfem check --profile cluster --runtime
agentfem run --profile cluster --mpi 64
```

The selected profile and resolved runtime are written to the execution and
result evidence. If the current machine lacks a requested capability,
AgentFEM fails before numerical lowering. It never silently substitutes a
different scientific procedure.

## Transport an entire project

Create a deterministic, integrity-checked project bundle:

```bash
agentfem pack --output cantilever.afm
agentfem inspect cantilever.afm
```

An `.afm` file contains `agentfem.toml`, the Python entrypoint, meshes, tables,
and other project-local scientific assets. Every member has a SHA-256 digest.
Outputs, version-control metadata, caches, `.env` files, private keys, and
credential files are excluded. Symbolic links are rejected so a bundle cannot
silently capture files outside the project.

The packer also rejects common literal machine-absolute file inputs in project
Python modules. Dynamic paths cannot be proven portable statically, so project
code should resolve assets from `Path(__file__).resolve().parent`.

It can be unpacked for review:

```bash
agentfem unpack cantilever.afm ./cantilever
```

or executed directly:

```bash
agentfem run --project cantilever.afm --profile cluster --mpi 64
```

Direct execution validates all paths, sizes, and hashes before materializing a
temporary project workspace. Results are written next to the bundle under
`outputs/`, not inside the temporary directory. The bundle identity remains in
the result manifest.

## Serial--parallel equivalence

Floating-point results from different sparse solvers and MPI partitions need
not be bit-identical. They must satisfy a declared numerical tolerance:

```bash
agentfem compare-runs \
  outputs/model/serial/result.json \
  outputs/model/mpi/result.json \
  --rtol 1e-8 --atol 1e-10 \
  --quantity displacement_max_abs \
  --quantity reaction_force_resultant \
  --quantity strain_energy \
  --write runtime-equivalence.json
```

The comparison checks common numerical quantities, units, shapes, completion
status, absolute error, and normalized error. A requested quantity that is
missing or scientifically incompatible rejects the comparison. This produces
a machine-readable acceptance record rather than relying on visual agreement.
Per-quantity overrides avoid applying one dimensional absolute tolerance to
displacement, force, and energy simultaneously.

## Checkpoint boundary

Portable checkpoints use global physical identities rather than execution-
local degree-of-freedom numbering. Procedures that implement that contract can
restart across supported MPI partition counts. A legacy or explicitly
partition-bound checkpoint is rejected when the partition changes; AgentFEM
does not relabel it as portable.

Native Windows is currently the serial linear development and validation
route. The full PETSc runtime remains the nonlinear, exact-MPC, and distributed
route. The public project language and result contract are shared by both.
