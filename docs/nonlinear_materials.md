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

`neo_hookean(...)` in two dimensions is plane strain.
`neo_hookean_plane_stress(...)` is a separate finite-strain membrane material:
it solves a positive out-of-plane stretch satisfying `P33=0` and condenses the
same energy and tangent. The Study assumption must match the material; the
provider rejects a mismatch before assembly.

The 3D Abaqus periodic-cell example additionally verifies quadratic tetrahedral
geometry, macro-deformation control, equation mismatch, averaged stress, and
quadrature-point `det(F)` bounds. It substitutes Neo-Hookean behavior for the
unavailable Abaqus user material and does not claim constitutive equivalence.

## Mooney--Rivlin Hyperelasticity

`constitutive.mooney_rivlin(...)` provides the three-dimensional compressible
energy

```text
psi = C10 (I1_bar - 3) + C01 (I2_bar - 3) + K/2 (J - 1)^2
```

`constitutive.mooney_rivlin_plane_stress(...)` provides the exact
incompressible thin-sheet reduction reported as Eq. (17) by Wang, Fineberg,
and Needleman. Both materials are consumed through the same `model.step(...)`
language as Neo-Hookean solids. The finite-element residual, recoverable
energy, first-Piola/Cauchy stress, material tangent, Explicit stable-increment
estimate, and small-on-large wave analysis all use the declared energy.

The JMPS-inspired weak-interface benchmark accepts this material directly via
`bulk_material=...`; changing the constitutive law does not copy its mesh,
cohesive state, preload transfer, time integration, energy ledger, or crack
observer. The three-dimensional penalty form and the exact two-dimensional
incompressible reduction remain distinct capabilities; neither is presented
as a general locking-free three-dimensional Explicit formulation.

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

The same material can run an inspectable material-point history without a
global finite-element solve. The loading path is a tensor-valued scientific
input: its reversal and hold knots are exact, refinements subdivide rather than
move those knots, and its content fingerprint travels with the result.

```python
import numpy as np
from agentfem import constitutive

steel = constitutive.J2LinearIsotropicHardening(
    young=210_000.0,
    poisson=0.3,
    yield_stress=250.0,
    hardening_modulus=1_000.0,
)
strain = np.zeros((4, 3, 3))
strain[:, 0, 0] = (0.0, 0.004, -0.002, 0.003)
path = constitutive.material_strain_path(
    (0.0, 1.0, 2.0, 3.0),
    strain,
    coordinate_name="load_coordinate",
)
result = steel.history(path).solve_result()
```

Material histories use the ordinary Campaign and ScientificDataset contracts;
there is no constitutive-model-specific batch runner. Declare fixed-length
trajectory outputs with `datasets.Quantity(..., kind="history")`. Campaign
then extracts the matching `SimulationResult` histories, preserves their shared
coordinate and fingerprint, and refuses to combine cases whose history axes
differ. This makes parameter sweeps, resume, failure isolation, provenance and
dataset packaging identical for J2, Chaboche, creep, viscoelasticity and future
history-producing procedures.

The same path contract supports explicit mixed control.  A symmetric Boolean
mask selects strain-controlled tensor components; every remaining component
uses the supplied stress target.  AgentFEM solves the unconstrained strains
with the constitutive algorithmic tangent.  This represents ordinary uniaxial
stress, stress control, and tension--torsion tests without silently fixing the
transverse strains:

```python
strain_control = np.zeros((3, 3), dtype=bool)
strain_control[0, 0] = True
path = constitutive.material_mixed_path(
    time,
    strain=axial_strain_targets,
    stress=np.zeros_like(axial_strain_targets),
    strain_control=strain_control,
)
result = steel.history(path).solve_result()
```

`TabulatedIsotropicHardening` is the corresponding piecewise-linear scientific
asset for cyclic-hardening tables.  It records its interpolation and
extrapolation rule and supplies the exact hardening-storage integral consumed
by the energy ledger.

Material-data generation defaults to `linearization="none"`: it returns the
same accepted stress and state without constructing a tangent that no global
Newton iteration will consume. `linearization="consistent"` adds the tangent
history when it is actually required.

Energy names are deliberately narrow. `plastic_work` is signed work and is not
renamed as dissipation. For linear-isotropic J2, the initial-yield component is
a complete rate-independent plastic-dissipation channel. For Chaboche it is
split into reference-yield and Armstrong--Frederick dynamic-recovery
dissipation. The accepted backward-Euler update also reports its nonnegative
time-discretization contribution separately, so numerical dissipation is not
presented as material heat. Each accepted plastic increment satisfies

$$
\Delta W^p
=\Delta\Psi_{\mathrm{iso}}+\Delta\Psi_{\mathrm{kin}}
+\Delta D_{y}+\Delta D_{\mathrm{rec}}+\Delta D_{\mathrm{BE}}.
$$

All cumulative channels participate in quadrature commit/rollback and
checkpoint/restart. This closes the implemented discrete constitutive ledger;
it does not by itself validate a particular parameter calibration.

The public SIMULIA OFHC-copper comparison now supplies that independent
material-point check.  Using the published table, calibrated $C=33.55$ GPa and
$\gamma=701.3$, and stress-free transverse components, AgentFEM recovers the
published symmetric-cycle final PEEQ of 23.67% and the nonproportional
tension--torsion saturated normal stress of 143.1 MPa within a separately
declared 1% AgentFEM gate. The published SIMULIA 316-steel asymmetric stress
path is also automated for the stated one- versus two-backstress ratcheting
trend and a stress-control residual below $10^{-7}$ MPa. Its reference response is
published only as a graph, so AgentFEM labels this as path/control/trend
evidence rather than inventing a numerical Golden. A structure-level
ratcheting comparison now reconstructs the public shouldered specimen and
keeps every peak and reversal as a mandatory target. Between those targets,
automatic cutback limits the maximum accepted PEEQ increment. Its five-cycle
certificate passes independent path-integration and spatial-mesh changes of
0.290% and 0.192%, respectively. A 245-cell full 100-cycle run also reaches
all 302 physical knots and passes the predeclared raster-curve error gate with
a final equilibrium residual below $10^{-9}$. Because the reference is a
digitized plot and the error lies close to the declared limit, this remains
external comparison evidence rather than an exact Golden or automatic
maturity promotion. The global axisymmetric chain is also regression-tested
on a uniformly loaded annular tube: its cycle-peak gauge strain must match the
independent material point while global Newton equilibrium closes.

For a three-dimensional `nonlinear_static` study, `model.step(...)` now lowers
this material to a global DOLFINx path. `PE` and `PEEQ` are committed at Basix
quadrature points; `S` and `DDSDDE` are trial fields updated during Newton.
`DDSDDE` is the analytical consistent linearization of the accepted discrete
backward-Euler return map, including isotropic hardening and dynamic recovery;
mixed stress control globalizes that tangent with residual-reducing line
search at reversals.
Failed attempts restore displacement and committed material state before
automatic cutback. Complete named `CellRegion` assignments may dispatch
different J2 parameter sets without changing the Step API. An otherwise
converged attempt is also rejected when its
equivalent plastic-strain increment exceeds
`maximum_inelastic_increment`. A portable checkpoint contains displacement,
accepted step coordinate, the adaptive next-increment proposal, plastic
state, energy/work history, amplitude identity, and schema version.

The current boundary is explicit: 3D small strain, complete nonoverlapping
material regions, natural or strong-displacement loading, structurally
benchmarked MPI global equilibrium, and full-Step restart across MPI rank
counts. Committed quadrature state can additionally be
saved collectively and restored across MPI rank counts using physical-cell,
quadrature-rule, mesh, material, and state-schema identity. A named
tabular amplitude may load, unload, and reverse while the internal step
coordinate remains monotone. Strong prescribed-displacement paths record
generalized reaction, external work, internal energy, and balance histories.
The result retains `S/PE/PEEQ/MISES` at constitutive integration points and
adds separately named `*_CELL` weighted DG0 recovery fields. The recovered
fields preserve element and material boundaries and are never labeled as raw
integration-point values or smoothed nodal contours.
Plane stress, finite-strain kinematic hardening, external distributed
finite-strain structural validation, and a general UMAT path remain future
work. A separate experimental logarithmic finite-strain J2 route already
provides ordinary strong-boundary and affine/MPC global equilibrium. Its mixed
lowerings are intended to mitigate volumetric locking, but locking-convergence
evidence and external promotion evidence are not yet complete.

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

The global 3D route reuses the J2 quadrature transaction, implements a
backward-Euler local update with analytical consistent tangent, and passes
relaxation, forced-cutback, restart, and weighted field-recovery contracts. It
also reproduces the official Abaqus `creep_usr_creep.inp` held-stress case:
the published time-hardening constants, 20,000 psi stress, and 100,000 s
duration give the closed-form equivalent creep strain 0.1. The versioned
100-increment contract declares its expected backward-Euler error instead of
freezing a solver-specific number as physical truth.

The next milestone is nonuniform multi-element and high-temperature component
evidence, followed by accepted temperature-field coupling and portable
quadrature state. A new creep-law name alone is not that milestone.

## Creep Damage, Sinh Flow, and Modified Theta

`KachanovRabotnovCreep` couples effective-stress creep and scalar damage:

```text
epsilon_dot = A (q / sigma_ref)^n / (1 - omega)^n
omega_dot   = B (q / sigma_ref)^m / (1 - omega)^phi
```

For a piecewise-constant stress interval, AgentFEM integrates both equations
analytically. One interval and any subdivision therefore recover the same
material-point state up to floating-point tolerance. `SinhCreep` provides the
separate hyperbolic-sine Mises rate family used when a power law is too rigid
over a wide stress range.

`ModifiedThetaProjection` represents

```text
epsilon = epsilon_0 + A1 (1 - exp(-alpha t))
                    + B1 (exp(alpha t) - 1)
```

and supplies deterministic nonnegative fitting, strain/rate projection, and a
time-to-strain criterion without adding SciPy to the core. It is classified as
a curve/life assessment, not a global FE stress update.

The release demo `examples/creep_hot_wall_assessment.py` connects an implicit
heat-transfer solve, sequential thermoelastic stress, the governing sampled
equivalent stress, K-R damage history, and modified-theta projection through
one `SimulationResult`. Its material constants are explicitly illustrative.
It demonstrates the power-component workflow and data contract while the
global quadrature creep step remains a visible next gate.

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
