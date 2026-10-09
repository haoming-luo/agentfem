# Uniform-gradient Hex8 development boundary

The experimental `elements.uniform_strain_hex8(...)` policy lowers ordinary
`model.step()` to a serial small-strain elastic explicit provider. The private
`elements._uniform_hex.UniformHex8` remains its local verification oracle.
This is not advertised as an Abaqus C3D8R reproduction.

## Ordinary workflow

```python
policy = elements.uniform_strain_hex8(
    hourglass_modulus=material_shear_scale,
    hourglass_scale=0.1,
)
step = model.step(target=u, element_policy=policy, dt="auto", steps=100)
result = step.solve_result(output="fields.xdmf")
```

The model uses a 3D `studies.dynamic_solid()` Study, continuous Q1 hexahedra,
registered constant isotropic or anisotropic elasticity, positive density and
ordinary strong constraints. Fixed material orientation and complete disjoint
material regions are supported. The engineering-shear order is shared with
the existing elasticity module. A prescribed time increment above the
conservative bulk-plus-hourglass bound is rejected. `dt="auto"` uses that
bound with safety 0.8; it does not tune an adaptive nonlinear increment.

The result distinguishes `strain_energy`, `kinetic_energy`, `hourglass_energy`,
physical `total_mechanical_energy` and `total_discrete_energy`. These components
are not a verified external-work balance. Serial interrupted/continuous runs
agree; changed operator identity rejects a checkpoint atomically. Public
portable restart, MPI, finite deformation, evolving material orientation,
damage/deletion, eigenstrain, additional contact/interface operators and
operator-changing time inputs are not admitted by this provider.

See `examples/uniform_hex_wave.py` for a runnable ordinary-workflow example.
On 2026-10-09 a candidate wheel was built without isolation downloads, installed
into a separate environment and used outside the source directory to execute
this 100-step example. Field output and the result manifest were written;
the original installed package was not replaced. This was a local candidate
acceptance, not a public version release.

## Formulation

Basix supplies Hex8 node order, trilinear interpolation and geometric
quadrature. Reference preparation integrates volume and shape gradients once.
The material subsequently sees one volume-average engineering strain vector
in the existing elasticity order `(xx, yy, zz, 2yz, 2xz, 2xy)`. A symmetric positive-definite
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
control or MPI capability follows from these local checks. Serial restart is
tested separately through the ordinary Step lifecycle.

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

## Remaining implementation gates

Compact preparation and serial affine/bending/wave checks are implemented.
DOLFINx retains mesh/DOF ownership; Procedure retains time integration and
stability composition. The experimental policy, ordinary Step, serial restart
and separate artificial energy are implemented. Remaining work includes
reviewed external-work closure, bounded-distortion admission, richer loading
evidence and eventual interface/MPI composition. No implicit numerical
equivalence to imported commercial reduced-integration elements is assumed.

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
