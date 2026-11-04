# P92 sinh screen and prescribed-boundary Newton correction

## Decision

The tested hyperbolic-sine candidates do not close the long low-stress P92
relaxation discrepancy. Keep the existing Norton negative-transfer evidence;
do not publish a calibrated P92 card or infer a life model from this screen.
The experiment also exposed a reusable creep Step robustness defect, fixed
without changing the constitutive law or weakening the convergence tolerances.

## Protocol and sources

Use the checksum-pinned RF and SLR first cycles from
[BAM/Zenodo 7198219 v1](https://doi.org/10.5281/zenodo.7198219),
Sonntag et al. (2022), CC BY 4.0. Temperature is treated as isothermal at
893.15 K; recorded drift is not fitted. Experimental nominal stress is
approximated by small-strain axial Cauchy stress, as in the frozen baseline.

The comparator integrates

`sigma_dot = E * strain_dot - r200 * sign(sigma) *
 [sinh(abs(sigma)/s0) / sinh(200/s0)]^n`.

The predeclared exponent grid is 1, 2, 4, 8. For each exponent, fit only r200
and s0 on the same RF first-hold window as the baseline, using trapezoidal
time weights and three initial guesses. Bounds are r200=1e-5..1e3 MPa/s and
s0=5..2000 MPa. Select the lowest RF objective, **not** the lowest SLR error.
This finite grid is a mechanism screen, not an exhaustive exclusion of the
whole sinh family or parameter-identifiability proof.

The modulus remains the predeclared RF initial-ramp estimate, 146259.633 MPa.
Only the conditional hold calibration starts from a measured stress. Full
cycles start at zero stress and retain history through every input knot,
reversal and hold. SLR already informed prior diagnosis and is not blind
validation, even though it does not enter these fits.

The standalone Radau comparator does not call an AgentFEM material update.
The FEM path uses existing `isotropic_sinh`, `creep_solid`, `model.step` and
co-located material histories. Numerical checks include the exact n=1 sinh
hold solution, scalar/core rate mapping, rate derivatives, structural reaction
and lateral equilibrium, temporal refinement and checkpoint continuation.

## Executed findings

| Exponent | RF conditional-fit RMSE, MPa | Conditional low-hold endpoint, MPa |
| --- | ---: | ---: |
| 1 | 0.544965 | 79.7960 |
| 2 | 0.544835 | 79.8823 |
| 4 | 0.537875 | 80.9747 |
| 8 | 0.506468 | 83.2872 |

The separate low-hold discriminator starts at the measured 84.41 MPa;
the observed endpoint is 45.30 MPa. It is explicitly **not** a stress reset
inside the full-path calculation. This screen provides little evidence that
changing only the autonomous stress-rate shape will explain that hold.

RF selects n=8, r200=1.613675462 MPa/s and s0=218.198858 MPa.
Full measured-path time-weighted RMSE is RF 3.88740 MPa and SLR 65.91926 MPa,
versus the frozen Norton's 3.73952 and 67.06971 MPa. Long SLR hold RMSE is
72.30141 MPa. The improvement is insufficient to claim material validation.

After the Newton correction below, FEM/reference maximum stress errors are:

| Path | Increments | Maximum error, MPa |
| --- | --- | --- |
| RF | 474 / 948 / 1896 | 2.20061 / 1.12781 / 0.571336 |
| SLR | 925 / 1850 / 3700 | 2.05178 / 1.05862 / 0.555810 |

Both retain decreasing errors, finest error below 1 MPa, lateral stress below
1e-6 MPa and prescribed-strain error below 1e-10. These are numerical gates,
not experimental acceptance limits. The full evidence is
`evidence/creep/2026-11-05-p92-sinh-screen.json`.

## A general robustness fix discovered by real-data verification

On pre-fix main a5164b83, the RF refinement-4 path failed at increment 129,
2.517 to 2.537 s. Its initial lifted residual was 6.23796e-6; full correction
gave 1.02256e-5 and half correction 9.86494e-6. Backtracking rejected them.
The defect was not failure of the local creep corrector: it was measuring
line-search descent from a displacement that still violated the new boundary.

Impose the current Dirichlet values on the trial displacement **before** the
first Newton material evaluation, then search only over free corrections.
The caller retains the previous accepted displacement/state and restores both
on rejection. This matches the existing finite-strain UserMaterial Step's
approach, reusing DOLFINx's
[set_bc contract](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.petsc.html)
and the existing Newton lifecycle. It is not a new solver, relaxed tolerance,
or additional material parameter. The same original full-path run now passes
all three temporal levels for both experimental paths.

Local validation (fenicsx-env, single process, threaded numerical libraries
limited to one thread): 77 focused creep/P92/coupled-material tests passed,
followed by 54 manufactured-field, rotating-history, fatigue-reference and
damage-history tests. CI scope checks passed 49 tests; Ruff and `git diff
--check` passed. These are not new multi-rank or clean-install results.

## Next scientific decision

Review whether the existing joint creep/plasticity law can change the relevant
hold mechanism before adding recovery. Its local simultaneous update already
owns joint state and a consistent tangent, following the same broad approach
as [MOOSE's coupled creep/plasticity update](https://mooseframework.inl.gov/moose/source/materials/ComputeCreepPlasticityStress.html).
It does not include static recovery. A candidate recovery extension should
explicitly distinguish plastic-flow-driven dynamic recovery from physical-time
recovery, and map conventions against
[NEML's recovery documentation](https://neml.readthedocs.io/en/dev/hardening/non/special_chaboche.html).
These are design references, not completed independent cross-code comparisons.

The next discriminator must separate two mechanisms rather than assume that
adding a plastic branch fixes relaxation. Under a fixed uniaxial total strain,
ordinary rate-independent plasticity without recovery can remain inactive as
stress relaxes inside its yield surface. In that branch it cannot change the
autonomous creep law. This is a bounded observation, not a conclusion for all
multiaxial histories or nonzero backstresses. Test the actual accepted state,
yield margin and plastic activity before selecting an extension.

For a recovery candidate, first reproduce a prescribed-backstress decay with
zero plastic flow, then a coupled hold. Specify normalization and signs rather
than copying parameter values: NEML documents two different placements of the
sqrt(3/2) factor. ANSYS also distinguishes
[static recovery in Chaboche hardening](https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/ans_mat/amp8sq21dldm.html)
from plastic-flow-driven evolution. Neither reference establishes that recovery
alone explains this P92 dataset. Prefer a project material provider for this
mechanism experiment; promote a reusable law only after its local and global
verification, without introducing a new structure solver.

Before any new law is admitted, require fixed-old-state tangent checks,
stationary and rotating-history references, physical-time refinement,
energy conventions and atomic restart. Keep a new complete test condition
for validation; do not relabel SLR as unseen after model selection.
