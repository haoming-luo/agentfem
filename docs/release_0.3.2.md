# AgentFEM 0.3.2

AgentFEM 0.3.2 strengthens the scientific middle layer: dynamics,
viscoelasticity, nonlinear state, exact constraints and result evidence now
share clearer ownership and more complete verification paths.

## Structural dynamics and viscoelasticity

- Structural modal analysis now uses the common `Study -> Model -> Step ->
  SimulationResult` workflow, with residual, orthogonality, cluster and MPI
  evidence.
- Generalized-Maxwell materials support exact material-point histories,
  restartable three-dimensional transient analysis and direct harmonic
  response with independent bulk and shear spectra.
- Harmonic solutions publish real, imaginary, amplitude and phase fields,
  together with independently assembled residual and cycle-energy evidence.

## Mechanics and trustworthy execution

- The public NAFEMS LE10 thick-plate workflow adds curved quadratic geometry,
  an external stress target, mesh convergence and serial/two-rank evidence for
  three-dimensional linear elasticity.
- Exact periodic MPCs enter ordinary static and thermal workflows through a
  shared prepared-solve lifecycle, with construction diagnostics and
  provider-owned reaction/work evidence.
- Finite-strain J2 state now separates recoverable elastic and hardening energy
  from plastic dissipation and carries those quantities through output,
  rollback and restart.
- DCB and ENF benchmark families add assembled compliance, convergence and
  scoped cohesive-propagation evidence without turning benchmark assumptions
  into general solver claims.

## Data, installation and ownership

- `ScientificFieldDataset` provides a dependency-light, lossless field-data
  contract for neural operators and companion learning providers.
- Human-readable run names, concise terminal summaries and latest-run lookup
  improve ordinary use without weakening structured identities for agents.
- Windows project custody is separated from the replaceable WSL2 runtime, and
  runtime removal exports a recovery snapshot before unregistering the
  distribution.
- Complete Runtime replacement is transactional; density-aware gravity and
  Abaqus gravity migration enter the same load, reaction and energy contracts.
- Global and China reliability routes share one strictly whitelisted event
  schema and remain independent of models, meshes, parameters and results.
- Finite-element contribution lowering moves from the public Model facade into
  the Operator layer while preserving the readable public workflow.

## Verification boundary

The exact release wheel is exercised through installed project templates,
serial and MPI tests, representative scientific workflows, strict
documentation and package-integrity gates. Generalized-Maxwell harmonic
analysis remains experimental pending an independent structural
frequency-response benchmark and near-resonance solver evidence. The Windows
Complete Runtime remains a WSL2 Preview rather than a native Windows solver.
