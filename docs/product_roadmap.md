# AgentFEM Product Roadmap

## Product Position

AgentFEM should first become an unusually clear, dependable, and extensible
finite-element tool on FEniCSx for a bounded set of real engineering problems.
Human readability and agent operability are product qualities of the same
public Python language; they are not reasons to postpone solver depth.

The near-term target is not feature-count parity with Abaqus or ANSYS. It is to
be better within selected workflows:

- less ceremony from model definition to a trustworthy result;
- engineering vocabulary and visible `K/M/C/F` or residual structure;
- direct escape hatches to UFL, DOLFINx, and PETSc;
- batch simulation and learning-data export as ordinary operations;
- explicit capability limits instead of silent approximations;
- benchmarks attached to every serious material/solver claim.

AF-IR remains an experimental record and research track. It must earn a larger
role through executable consumers and real cases. The Python product, numerical
core, results, verification, and documentation have priority.

## Current Capability Boundary

| Capability | Maturity | What is usable now | Important limit |
| --- | --- | --- | --- |
| Linear elasticity | FEM-integrated | 2D isotropic/selected anisotropic, static and dynamics building blocks | broader 3D verification and reactions remain |
| Compressible Neo-Hookean | FEM-integrated | nonlinear static solve; 3D and 2D plane strain forms | one-material convenience step; no 2D plane stress local solve |
| J2 plasticity | material-point verified | 3D radial return and exact uniaxial path updates | no quadrature-state FEM driver or consistent global tangent |
| Power-law creep | material-point verified | constant-stress, relaxation, tensor increments, exact piecewise-constant stress paths | no global adaptive creep step |
| Stress-life fatigue | postprocessor | Basquin/tabulated S-N, rainflow, Goodman, Miner assessment from named result histories | no multiaxial critical-plane method |
| External CAE mesh | integrated Abaqus path + conversion interface | generic meshio conversion; Abaqus node labels; verified C3D10 import; linear equation parsing | full solver-deck sections/material cards are not imported |
| Abaqus periodic equations | serial + two-rank FEM-integrated | exact chained affine elimination, distributed `dolfinx_mpc`, and 3D Neo-Hookean load path | AMG near-nullspace transfer, reactions, and scaling studies remain |
| Abaqus user-material bridge | interface contract | solver-neutral material-point input/output and migration specification | no compiled adapter or quadrature-state global driver |
| Result/data flow | integrated foundation | compact deformed XDMF/HDF5 time series, fields, histories, QoIs, integrals/averages/norms, campaign/dataset bridge | reactions, point probes, restart remain |

The same table is queryable in code through
`constitutive.capabilities()` and `benchmarks.list_benchmarks()`.

## Release Gates

A public solver/material capability advances through these levels:

1. **formula implemented** — typed parameters and declared assumptions;
2. **material point verified** — analytical/invariant checks and load paths;
3. **finite-element integrated** — state storage, tangent, nonlinear/time step,
   convergence evidence, and field output;
4. **benchmark verified** — mesh/time convergence and an external reference;
5. **workflow ready** — readable example, failure cases, MPI/output behavior,
   and user documentation.

A name in `constitutive/` does not imply level 3. The maturity catalog prevents
an agent, user, or README from confusing these levels.

## Delivery Sequence

### P0: harden the usable core

- one `SimulationResult` contract for linear, nonlinear, and transient steps;
- standard QoIs: integrals, averages, norms, extrema, reactions, energies, and
  histories;
- compact unified XDMF/HDF5 visualization and output manifests;
- JSON-configured and Python-configured campaigns producing the same dataset;
- serial, MPI, docs, package, and example release gates;
- clear solver convergence/failure evidence.

### P1: nonlinear solid mechanics

- finish Neo-Hookean load-controlled and multi-region verification;
- harden the implemented stateless periodic-cell automatic incrementation with
  forced-cutback regression cases, then add homogenized tangent checks and a
  distributed affine/MPC backend;
- introduce quadrature-state storage and a constitutive update protocol;
- integrate J2 isotropic hardening with a consistent algorithmic tangent;
- extend the current deformation-controlled automatic path to general
  displacement/load control;
- then add tabulated hardening, kinematic hardening, and finite-strain
  plasticity only when driven by real applications.

For Abaqus material migration, implement UHYPER energy adaptation before the
more general UMAT path. UMAT requires quadrature state, trial/commit/rollback,
tensor and rotation conventions, compiler/ABI handling, and a consistent
tangent. Advance compatibility one restricted subroutine class at a time,
gated by material-point, one-element, and load-path comparisons. The public
AgentFEM model language should consume a neutral material-point protocol rather
than depend on Abaqus interfaces directly.

### P2: time-dependent materials and life

- global implicit creep step with adaptive increments and restartable state;
- creep/relaxation single-element verification followed by NAFEMS cases;
- named result histories feed an auditable fatigue assessment now; add
  automatic stress extraction at named regions/points and fatigue fields;
- multiaxial fatigue only after a chosen engineering criterion and reference
  dataset are explicit.

### P3: mesh and model interoperability

- verify Abaqus `.inp`, Nastran bulk-data, Gmsh, Exodus, and MED meshes;
- map volume sets and boundary sets to named AgentFEM regions;
- preserve source identities, checksums, conversion choices, and warnings;
- extend the now-preserved Abaqus node labels and linear equations to common
  node/element-set semantics where meshio loses information;
- treat ANSYS CDB and full solver decks as separate adapters, not generic mesh
  conversion.

### P4: AI-native operation

- maintain one public API and one validation path for humans and agents;
- make errors addressable and capabilities queryable;
- pair every public function family with compact reference examples;
- add tool/service endpoints around the same campaign/result contracts;
- evolve AF-IR only when a loader, validator, migration, or independent
  consumer requires a stable semantic record.

## Definition of “Better”

Within a supported problem class, AgentFEM is competitive when an experienced
engineer can:

1. read the model without reconstructing generated backend code;
2. modify a material, region, load, or step locally;
3. inspect the governing operator/residual and solver evidence;
4. run one case or thousands with the same case builder;
5. obtain visualization, scalar histories, and training data without a second
   extraction project;
6. reproduce the result from a benchmarked open workflow.

This is a narrower and more defensible route to excellence than imitating the
entire feature surface of mature commercial CAE suites.
