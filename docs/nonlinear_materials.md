# Nonlinear Materials and Maturity

## Design Boundary

AgentFEM separates three things that commercial input decks often present
together:

- material parameters;
- the local constitutive update;
- the global finite-element solution procedure.

This matters for path-dependent materials. A correct radial return at one
integration point is necessary, but it is not a working elastoplastic FEM
solver until quadrature state, consistent tangents, increment control,
convergence evidence, and restart are present.

Use `constitutive.capabilities()` before selecting a nonlinear law.

## Compressible Neo-Hookean Hyperelasticity

`constitutive.NeoHookeanProperties` and
`model.step(...)` is the canonical public entry point. It dispatches a
`nonlinear_static` Study through an inspectable step-provider registry; the older
`model.hyperelastic_step(...)` remains a lower-level compatibility route, not
a pattern to duplicate for every material model. New constitutive families
register a provider that declares which analysis/material protocol they can
lower. The public model language does not acquire one method per material.
The formulation uses

```text
F = I + grad(u)
psi = mu/2 (tr(F^T F) - d) - mu ln(J) + lambda/2 ln(J)^2
```

and solves the first variation of total potential. Ordinary Dirichlet problems
use PETSc SNES. Abaqus periodic equations use exact affine reduction and an
incremental reduced Newton path. In both cases the tangent is differentiated
from the same residual. This follows the same
compressible Neo-Hookean energy and automatic differentiation structure as the
[official DOLFINx hyperelasticity demo](https://docs.fenicsproject.org/dolfinx/main/cpp/demos/demo_hyperelasticity.html).

`constitutive.kinematics(u)` exposes the standard `F`, `C`, `J`, and
Green--Lagrange measures to reusable internals and expert workflows. Ordinary
model scripts do not need to import UFL for these quantities.
`solvers.newton(...)` supplies one public nonlinear policy for both SNES and
affine-reduction paths; constraint implementation no longer changes the
top-level solver language.

Two-dimensional use is plane strain. Plane stress needs a local
out-of-plane-stretch solve and is deliberately rejected by the convenience
step rather than approximated silently.

The 3D Abaqus periodic-cell example additionally verifies quadratic tetrahedral
geometry, macro-deformation control, equation mismatch, averaged stress, and
quadrature-point `det(F)` bounds. It substitutes Neo-Hookean behavior for the
unavailable Abaqus user material and does not claim constitutive equivalence.

## J2 Plasticity

`constitutive.J2LinearIsotropicHardening` implements a small-strain,
rate-independent Mises material-point update:

```text
trial elastic predictor
f_trial = q_trial - (sigma_y0 + H p_old)
Delta gamma = f_trial / (3G + H)
radial correction of deviatoric stress
```

The implementation verifies that the corrected stress lies on the hardened
yield surface, includes an exact uniaxial update, and returns the analytical
algorithmic consistent tangent. The algorithmic
family is the standard closest-point radial return described in the
[MOOSE radial-return documentation](https://mooseframework.inl.gov/moose/source/materials/RadialReturnStressUpdate.html)
and [Abaqus isotropic elastoplasticity theory](https://docs.software.vt.edu/abaqusv2024/English/SIMACAETHERefMap/simathe-c-isoelastoplast.htm).

For a three-dimensional `nonlinear_static` study, `model.step(...)` now lowers
this material to a global DOLFINx path. `PE` and `PEEQ` are committed at Basix
quadrature points; `S` and `DDSDDE` are trial fields updated during Newton.
Failed attempts restore displacement and committed material state before
automatic cutback. A serial checkpoint contains displacement, accepted load
factor, plastic strain, equivalent plastic strain, and a schema version.

The current boundary is explicit: 3D small strain, one material region,
natural-load incrementation, time-invariant strong supports, and serial
execution/restart. Plane stress, distributed execution/restart, finite-strain
plasticity, and a general UMAT path remain future work.

## Power-Law Creep

`constitutive.PowerLawCreep` provides a normalized Mises time-hardening law,
exact constant-stress integration, a relaxation solution, and associative
tensor increments. The constant-stress and relaxation formulas are checked
against the equations used in the
[Abaqus creep-integration verification](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-creep.htm).

`constitutive.integrate_stress_history(...)` is the next reusable layer: it
integrates piecewise-constant scalar or tensor stress intervals with the exact
time-hardening increment and returns a named `CreepHistory`. It is useful for
material tests and prescribed stress paths, but is explicitly not a global FE
creep solver.

`constitutive.ArrheniusPowerLawCreep` adds a normalized temperature factor
whose coefficient is calibrated at a declared reference temperature. This is
the appropriate local basis for high-temperature component workflows, but it
does not yet promote creep to a global coupled solver.

The next creep milestone is to reuse the implemented J2 quadrature transaction
and restart schema, add an implicit time-local update and error estimate, then
pass relaxation, one-element, restart, and external high-temperature
benchmarks. A new creep-law name alone is not that milestone.

## Stress-Life Fatigue

The fatigue module is a result postprocessor:

- Basquin and tabulated log-log S-N curves;
- turning-point extraction and rainflow cycle counting;
- optional linear Goodman mean-stress correction;
- Palmgren-Miner cumulative damage.

NASA fatigue guidance describes rainflow counting as the bridge from a stress
response history to cumulative S-N damage; see the
[NASA spectral fatigue report](https://ntrs.nasa.gov/api/citations/20160012240/downloads/20160012240.pdf).
AgentFEM currently accepts a scalar/equivalent stress history. Multiaxial
critical-plane fatigue is a separate future capability.

`constitutive.assess_history(...)` returns counted cycles, Miner damage, and
the life in repeated copies of that history. `assess_result_history(...)`
accepts a named `SimulationResult` history and preserves its source in the
assessment. This connects analysis results to fatigue without making fatigue a
solver step or hiding which scalar history was used.

## Verification Inventory

`benchmarks.list_benchmarks()` returns the current test-linked obligations.
Examples teach use; benchmarks carry a criterion and automated evidence.
