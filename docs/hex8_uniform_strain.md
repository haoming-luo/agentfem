# Uniform-gradient Hex8 development boundary

The private `elements._uniform_hex.UniformHex8` is a local verification
candidate, not a public element selection or a completed Explicit provider.
It is not advertised as an Abaqus C3D8R reproduction.

## Formulation

Basix supplies Hex8 node order, trilinear interpolation and geometric
quadrature. Reference preparation integrates volume and shape gradients once.
The material subsequently sees one volume-average engineering strain vector
in the order `(xx, yy, zz, 2xy, 2yz, 2xz)`. A symmetric positive-definite
six-by-six elasticity tensor maps that vector to physical stress components.
The preparation quadrature does not imply multiple constitutive updates.

Four reference hourglass vectors are normalized by `sqrt(8)`. The corrected
vectors are `gamma = Gamma - grad_mean @ (X_centered.T @ Gamma)`, annihilating
affine displacement. The candidate's explicitly chosen stiffness energy is
`scale * modulus * V^(1/3) * sum((gamma.T @ u)^2) / 2`.
This normalization and coefficient are declared choices, not commercial code
defaults; bending and wave evidence must constrain their appropriate use.
Physical and artificial energies remain distinct. Consistent row-sum masses
are integrated rather than assuming distorted cells have equal nodal mass.

Jacobian checks sample preparation quadrature and corners. They reject detected
inversion and degeneracy but are not a proof of positivity everywhere in a
general trilinear map. Mesh-quality and supported-distortion limits remain
part of any future global admission policy.

## Local evidence

Targeted tests verify regular/distorted affine response,
infinitesimal rigid translation/rotation, six physical stiffness modes,
twelve controlled hourglass modes, six remaining rigid null modes,
energy-gradient/tangent consistency, positive mass, geometric scaling, and
independent higher-order integration of mean gradients and mass. General
positive-definite orthotropic elasticity is exercised. The mass-scaled
combined stiffness row-sum is a conservative cell spectral bound, not yet a
whole-model time-step policy.

No finite-rotation objectivity, material damage, deletion, viscous/relaxation
control, MPI or restart capability follows from these checks.

## Compact execution and serial global evidence

`UniformHexBatch` retains mean gradients, four corrected hourglass vectors,
positive nodal masses and coefficients. Shared material stiffness is stored
once; heterogeneous tensors are supported. Geometry preparation and response
evaluation are chunked. No per-cell 24-by-24 matrices are retained. Response
iteration can consume chunks without storing all output fields.

A 2,048-cell, single-thread local microbenchmark on 2026-10-09 compared five
response repetitions against the scalar reference operator. Median response
times were 0.00192 s (batch) and 0.01785 s (scalar), approximately 9.3x. Compact
stored arrays occupied 1,081,632 bytes versus 18,874,368 bytes for two dense
cell matrices alone. This is not a complete solver speedup or peak-memory
measurement; global DOFs, connectivity and temporary gathers add storage.

The private `UniformHexResidual` uses DOLFINx blocked Q1 DOF maps and the
existing central-difference integrator. Serial affine internal forces agree
with independent fully integrated UFL assembly. Separate resultant and moment
checks pass. An analytic longitudinal standing wave converges under mesh
refinement, with a bounded energy error. The conservative time-step bound uses
positive cell mass and stiffness Rayleigh bounds; additional interface/contact
stiffness is not included and must be composed separately.

A slender cantilever (length 10, unit square section, E=100, nu=0, uniformly
distributed end shear) was checked against Euler-Bernoulli deflection with
the rectangular-section Timoshenko shear correction. This is a beam-theory
reference, not an exact three-dimensional solution or a commercial-code run.

| Hourglass scale | 20x2x4 relative tip error | 40x4x8 relative tip error | Fine-mesh artificial energy / total |
| --- | ---: | ---: | ---: |
| 0.05 | 4.883% | 1.177% | 0.375% |
| 0.10 | 3.275% | 0.799% | 0.747% |
| 0.20 | 0.205% | 0.051% | 1.484% |

All three choices improve under refinement. This does not select a universal
default coefficient or establish accuracy for a single element through the
thickness. Tests are in `test_uniform_hex.py` and `test_uniform_hex_global.py`.

## Next implementation gate

Compact preparation and serial affine/bending/wave checks are implemented.
DOLFINx retains mesh/DOF ownership; Procedure retains time integration and
stability composition. The next gate is an explicit element policy, ordinary
Step admission and separate physical/artificial energy in Result metadata,
with unsupported nonlinear or MPI paths rejected before execution.

## Sources

- Flanagan and Belytschko, 1981, *A uniform strain hexahedron and quadrilateral
  with orthogonal hourglass control*, https://doi.org/10.1002/nme.1620170504.
- Sandia, *Sierra/SM Theory Manual*, section 15.1, uniform-gradient Hex8 and
  orthogonal hourglass control:
  https://www.sandia.gov/files/sierra/SM_Theory_5_20/main/element_formulations.html.
  The candidate uses the affine-removal construction but explicitly documents
  its own stiffness normalization rather than claiming full Sierra equivalence.
- Abaqus verification documentation, *Performance of continuum and shell
  elements for linear analysis of bending problems*:
  https://docs.software.vt.edu/abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-linbending.htm.
  Motivation for thickness refinement, hourglass-energy inspection and not
  equating patch-test success with reliable bending performance.
