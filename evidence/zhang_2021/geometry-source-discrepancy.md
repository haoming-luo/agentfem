# Explicit geometry provenance for the Zhang Table 5 diagnostic

## Source discrepancy

The locally archived 75-page author manuscript of Zhang, Feng and Khandelwal,
*A computational framework for homogenization and multiscale stability analyses
of nonlinear periodic materials*, was reviewed visually on 2026-10-07.
Reference: [journal DOI](https://doi.org/10.1002/nme.6802),
[author preprint](https://arxiv.org/abs/2010.02371).
The inspected PDF SHA-256 is
`1ce6d0539c364f18b8f19be7a46dec6f6489b9024752a85e6f70d516888afd18`.
This note describes that inspected manuscript; it does not establish whether
the final publisher PDF contains the same discrepancy.

Pages 26–27, section 3.2.1, specify inclusions at `(-0.2, 0.2)` and
`(-0.2, -0.2)`, with a void at `(0.2, 0)`. Figure 10(a), the deformed
configuration in Figure 11, and the thumbnails in Table 6 instead show a
white void at lower left and blue inclusions at upper left and right.
Table 5 explicitly refers to the cell in Figure 10(a).

The existing fixture implemented the prose literally. It now exposes two
explicit `geometry_source` values:

| Source | Inclusion centres | Void centre |
| --- | --- | --- |
| `section-3.2.1-text` (existing default) | `(-0.2, 0.2)`, `(-0.2, -0.2)` | `(0.2, 0)` |
| `figure-10a` | `(-0.2, 0.2)`, `(0.2, 0)` | `(-0.2, -0.2)` |

Both retain the stated radius 0.15, material parameters, plane-strain
kinematics and macroscopic shear. This is an explicit source hypothesis,
not calibration of geometry to the target numbers. Region-centroid tests
verify the assignment. Convergence, MPI and supercell audits include the
geometry source in their controls and reject mixed-source comparisons.

## Controlled first comparison

The same checkout, mesh target 0.20, Q9/DPC1 interpolation, degree-4
quadrature and 40 fixed increments give:

| Source | Cells | First-Piola relative L2 error | Tangent relative Frobenius error | Primal elastic energy error |
| --- | ---: | ---: | ---: | ---: |
| Prose | 315 | 4.845% | 14.150% | 20.539% |
| Figure 10(a) | 315 | 1.076% | 0.459% | 29.868% |
| Figure 10(a), mesh target 0.10 | 804 | 0.132% | 0.357% | 9.945% |
| Figure 10(a), mesh target 0.07 | 1602 | 0.176% | 0.597% | 5.330% |
| Figure 10(a), mesh target 0.05 | 2859 | 0.099% | 0.507% | 0.014% |

At 804 cells all stress components also pass the existing componentwise
contract. Primal energy is 0.0026639731, condensed energy is 0.0023614218,
and the pressure-constraint defect energy is 0.0003025512. The original
Table 5 energy comparison remains against primal Hencky elastic energy;
the condensed channel is not silently substituted. All remaining convergence
and external promotion requirements continue to apply.

At 2859 cells the stress component checks and all three published-observable
comparisons pass the existing 3% project contract. This is **numerical agreement,
not full benchmark promotion**. Between the last two meshes, stress changes by
0.0777%, tangent by 0.1160%, but primal energy by 5.3442%, exceeding the existing
0.5% energy convergence contract. The energy changes are not monotonically
decreasing. Agreement with one published discrete mesh must not be relabelled
as continuum convergence. The fine solve accepted all 40 increments in 583.6 s;
the maximum accepted Hill--Mandel relative residual is 2.283e-10.

These candidates were produced from a dirty development checkout with stable
start/end runtime and benchmark identities. The audit correctly reports
`content_bound: false` and does not authorize promotion. Previously archived
MPI, restart and supercell checks concern the prose geometry; they do not
automatically qualify the figure geometry. Next gates are a clean-source
repeat, controlled energy/mesh diagnostics, and the figure-specific path,
quadrature and lifecycle checks. Neither material parameters nor tolerances
were changed in this investigation.

Reproduce each candidate with the FEniCSx environment, `PYTHONPATH=src:tests`,
and `tests/zhang_2021_plane_strain_driver.py --geometry-source figure-10a
--mesh-size H --increments 40 --output OUTPUT`. Use H = 0.20, 0.10, 0.07,
0.05. Pass their assessment JSON files as repeated `--mesh-run` arguments to
`tests/zhang_2021_plane_strain_promotion.py`.

The local diagnostic archive `rve-geometry-20261007/diagnostic-evidence.tar.gz`
contains all five runs and the mesh audit. Archive SHA-256:
`641e4cfb3d0048b930741a50f8f29a604c8043a12eb78bbfbc9beedbb3f85d36`.
It is retained with local project outputs, not claimed as a publicly downloadable
or promoted Golden archive.

The evidence strongly motivates checking figure-based geometry before
attributing the former stress/tangent discrepancies to the constitutive
implementation. It does not yet establish full benchmark reproduction.
