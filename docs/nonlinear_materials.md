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
yield surface and also includes an exact uniaxial update. The algorithmic
family is the standard closest-point radial return described in the
[MOOSE radial-return documentation](https://mooseframework.inl.gov/moose/source/materials/RadialReturnStressUpdate.html)
and [Abaqus isotropic elastoplasticity theory](https://docs.software.vt.edu/abaqusv2024/English/SIMACAETHERefMap/simathe-c-isoelastoplast.htm).

It is not yet exposed as `model.plastic_step(...)`. That name would imply a
quadrature-field driver and consistent global tangent that do not yet exist.

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

The next solver milestone is not another creep-law name. It is restartable
quadrature state plus adaptive global time integration, followed by NAFEMS
creep benchmarks.

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
