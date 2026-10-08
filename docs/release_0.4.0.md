# AgentFEM 0.4.0

AgentFEM 0.4 consolidates the foundation for dependable finite-element
workflows. It keeps one public route from engineering definition to numerical
procedure, accepted state, and inspectable results. This is an architectural
milestone, not a blanket promotion of every experimental analysis.

## What this release establishes

- **Model and Operator:** shared mesh, element, function-space, and quality
  contracts reject unsupported combinations before assembly. Typed time inputs
  distinguish load changes from changes that invalidate an operator.
- **Procedure and State:** numerical strategies own advancement and reuse;
  accepted/trial state owns commit, rollback, and declared portable restart.
- **Result and Verification:** constraint force, accepted-path work, energy,
  field identity, performance evidence, and failure status remain explicit.
- **Extensions:** an independently installed provider exercises the ordinary
  workflow without modifying the core or adding a framework dependency.

The six foundation gates cover compatibility, quality, time inputs, bounded
constraint closure, MPI result/state/checkpoint lifecycle, and external
extension. Release evidence binds the source commit and wheel checksum:
complete serial regression, representative MPI tests, independent extension
acceptance, and Linux/macOS installation of the same wheel. Publication waits
for the foundation audit as well as the existing scientific and optional
dependency checks.

## Reliability improvements since 0.3.8

Nonlinear and mixed finite-strain paths retain accepted-history identity across
restart. RVE diagnostics distinguish spatial refinement, load-path refinement,
and consistent-tangent checks; a passing lifecycle check is not substituted for
an external accuracy comparison. Manifest sealing excludes only its own
self-reference, preserves external artifact integrity, and reports write
failures rather than silently accepting incomplete output.

## Scientific scope

Existing capability maturity is preserved in the versioned release contract.
Finite-strain J2/RVE remains **experimental**. The Zhang--Feng--Khandelwal
comparison is **not promoted**: spatial and other external validation gates
remain incomplete. Passing serial/MPI restart and a bounded tangent check does
not close those gates. See the [RVE reference](reference/rve_homogenization_and_statistics.md)
and [roadmap](product_roadmap.md) for the exact limits.

Contact evidence applies to the documented bounded routes; it does not imply
general industrial forming, arbitrary deformable contact, or unrestricted
rigid-body dynamics. Backend numerical infrastructure remains owned by
FEniCSx/Basix/PETSc/MPI.

## Installation and compatibility

Install or upgrade through the [getting-started guide](getting_started.md).
The Python release and complete offline runtimes are separate artifacts:
an older runtime is not relabelled as 0.4. A new offline runtime is offered only
after its own build and acceptance. Real Windows/WSL2 installation and unsigned
macOS runtime limitations remain separate from hosted Linux/macOS wheel tests.

Existing public examples and compatibility imports are included in the release
ladder. Cite the software version actually used; the 0.4 citation points to the
immutable release until a matching version archive DOI is available.

## Next: bounded engineering progress

The next cycle prioritizes reproducible user workflows, supported mesh and
formulation combinations, and performance measured with fixed inputs. One
bounded scientific validation track may proceed alongside maintenance; an
individual paper reproduction will not hold unrelated product delivery open
indefinitely. See the [product roadmap](product_roadmap.md).
