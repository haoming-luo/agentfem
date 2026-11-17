# Composite tetrahedral operators and the C3D10M boundary

## Scope and public entry point

`elements.composite_tet10(coordinates)` constructs reusable **local** small-strain
operators for a straight-sided, affine ten-node tetrahedron. It returns consistent
and positive lumped mass, projected gradients, strain matrices, stiffness, and a
central-difference local time-step bound. Coordinates follow Abaqus Tet10 ordering:
vertices 0–3; edges 01, 12, 20, 03, 13, 23. Vector DOFs are node-major xyz.

The local API now backs the registered `composite_tet10_explicit` whole-mesh
Step provider. Use `fields.displacement(domain, element_policy=
elements.composite_tet10_policy(mean_dilatation=True))`, then ordinary
`model.step(target=u, dt="auto", steps=...)`. DOLFINx owns the mesh, DOF mapping
and state fields; the provider supplies composite force, mass and recovery.
P2 is used **only as the ten-node coefficient storage layout**. Generic P2
model stiffness/mass calls reject a declared composite field.

This does not register `C3D10M` as executable, change INP interpretation, or
replace the existing P2/HRZ and Tet15 routes. Curved/moved mid-edge geometry is
rejected. The implementation is independently written from mathematical definitions.

## What public sources establish

Abaqus documents piecewise-linear modified-tetrahedron interpolation, three
additional displacement variables, lumped mass, and internal-mode hourglass
control. These statements do not specify enough equations to reconstruct the
Abaqus implementation uniquely. In particular, three independent internal
variables cannot be replaced silently by a kinematically tied centre.

The published composite formulation by Ostien et al. supplies a related open
mathematical starting point: twelve linear subtetrahedra and a parent linear
gradient projection. This module implements that affine projection construction;
its optional mean-dilatation stabilization below is an explicitly identified
AgentFEM derivation, not a claim of proprietary or published-code equivalence.

## From interpolation to projected gradient

Let X_a, a=0,...,9 denote physical nodal coordinates. The auxiliary centre is
X_c = (X_4+...+X_9)/6, and its displacement is constrained by the **same** average.
Let T be the 11 by 10 interpolation matrix: its first ten rows are the identity
and its final row is (0,0,0,0,1/6,...,1/6). On subtetrahedron s, linear barycentric
functions n_i^s give the ten external-node functions N_a = sum_i n_i^s T_ia.
They reproduce constants and affine displacement exactly.

Let lambda_alpha, alpha=0,...,3 be the parent barycentric functions and V its
volume. The L2 projection is defined, rather than fitted, by

    G_ab = integral_V lambda_a lambda_b dV
    H_aij = integral_V lambda_a (dN_i/dX_j) dV
    Bbar_ij(X) = sum_ab lambda_a(X) inverse(G)_ab H_bij.

The code integrates each subtetrahedron using a positive four-point degree-two
rule. Thus G, H and the consistent mass are integrated exactly for the declared
affine geometry. Gradients are transformed using the inverse geometry Jacobian.
Reference-coordinate integration avoids spurious conditioning from global
coordinate translation.

## Mass: consistent versus lumped

For density rho, the scalar consistent mass is

    M = rho sum_s V_s/20 T_s^T (I + 11^T) T_s.

The vector mass is M tensor I_3. Row sums give positive scalar nodal masses:

    vertex mass = rho V / 32; edge-node mass = 7 rho V / 48.

The four vertex and six edge contributions sum to rho V. Consistent mass
integrates rigid-rotation inertia exactly. Lumped mass preserves total mass and
centroid but is not claimed to integrate the second moment exactly. The centre
elimination here is kinematic, not a dynamic condensation approximation.

## Stiffness and a discovered zero-mode issue

Use engineering strain order xx, yy, zz, xy, yz, xz and corresponding physical
stress order. Let Bbar_epsilon be the strain matrix formed from Bbar and C a
constant symmetric positive-definite elasticity matrix. The projected stiffness
is Kp = integral Bbar_epsilon^T C Bbar_epsilon dV.

The tested affine element has six rigid-body null modes. If its volumetric strain
is replaced by the volume mean (B-bar), **three additional null modes appear**.
This was detected by the new eigenvalue regression, not assumed from a successful
linear solve. The uncorrected mean-dilatation branch is research-only.

For mean dilatation, the default correction uses unresolved deviatoric strain:

    R = dev(B_piecewise - Bbar_epsilon)
    Ks = integral R^T C R dV
    K = Kp_mean_dilatation + Ks.

`dev` subtracts one third of the normal-strain trace and preserves engineering
shears. Ks is positive semidefinite, vanishes on affine displacement and adds no
bulk-modulus penalty. No adjustable hourglass coefficient is introduced. For
constant isotropic C, projection orthogonality makes this precisely the sum of
fully integrated piecewise deviatoric energy and mean volumetric energy. It is
therefore a defined variational discretization, not an empirical stiffness fix.
For anisotropic C the same positive energy is defined, but no assertion of the
isotropic energy identity or general locking-free behavior is made.

The plain projected branch defaults to **no** correction. `stabilize=False`
explicitly exposes the uncorrected operator for formulation research. The
small-strain tangent equals the returned K; energy is u^T K u/2 and force is K u.

## Explicit time-step support

The positive lumped mass gives

    dt_critical = 2 / sqrt(lambda_max(M_lump^(-1/2) K M_lump^(-1/2))).

The local bound scales with length, sqrt(density), and inverse sqrt(stiffness).
The local API returns the mathematical bound without a safety factor. The
whole-mesh provider selects 0.8 times the minimum owned-cell bound over all MPI
ranks. Since global stiffness and positive lumped mass are assembled sums, the
cell Rayleigh bound is conservative for the assembled linear system. Fixed
Dirichlet elimination cannot increase the admissible Rayleigh maximum. Prescribed
motion is handled by the existing time integrator. Contact and additional custom
stiffness are not admitted. A supplied dt above this bound is rejected.

## Verification and integration

`tests/test_composite_tet.py` checks partition gradients, affine completeness,
constant engineering strain, positive mass, exact consistent-mass inertia,
rigid motions, stiffness rank, energy directional derivatives, all 24 vertex
permutations, length/density/stiffness scaling, and rejection of unsupported input.
The stabilization test also retains the failing nine-null-mode formulation as
an explicit regression; it must never be mistaken for a usable default.

The whole-mesh implementation is tested in `tests/test_composite_tet_workflow.py`:
public field/provider selection, MPI owned-cell force and mass assembly,
consistent loads, patch equilibrium, distinct anisotropic shear components,
material partitioning, moving supports, work accounting, same-partition restart,
exact spatial interpolation and XDMF/HDF5 outputs. Tests use synthetic unit meshes;
no blade or application simulation was performed.

## Whole-mesh execution and contracts

- Elastic material regions must cover owned cells exactly once. Core engineering
  shear order xx,yy,zz,yz,xz,xy is explicitly permuted to the kernel order.
- Nodal forces use source-node binding. Constant body loads integrate the
  composite shape functions: vertex weights V/32 and edge weights 7V/48.
- Constant face traction integrates the four linear face subtriangles: each
  vertex receives A/12, each edge node A/4. Reference pressure uses geometric
  outward normals with the declared inward sign. Scalar amplitudes reuse the
  platform's accepted time-input lifecycle. General spatial expressions,
  custom pressure normals and follower pressure are rejected.
- Owned cells contribute once; PETSc reverse ghost accumulation assembles the
  nodal arrays. No replicated global solve or separate mesh owner is introduced.
- Explicit U/V/A updates, fixed/moving strong Dirichlet constraints, the dynamic
  work ledger and checkpoint lifecycle reuse the existing platform mechanisms.
  Checkpoint identity includes geometry, DOFs, stiffness, mass and initial inputs.
  Same-partition restart is supported; repartitioned restart is not claimed.
- `hourglass_energy` records the projection-residual stabilization contribution
  separately. It is not material dissipation. The ordinary projected branch
  records zero stabilization energy. A rigid translation is removed before
  local force/energy arithmetic; this exact null-mode operation reduces loss of
  significance. Roundoff-sized negative quadratic energies are tested against
  an absolute-operation error scale, not a tolerance relative to a near-zero
  final energy; larger negative energies still raise an error.
- Per-cell quadrature scratch is discarded after preparation. Only compact
  recovery coefficients, the fixed stiffness, mass and material data remain.

## Output and physical interpolation

`step.solve_result(output="fields.xdmf", field_variables=("S","E","MISES","SENER"))`
writes U/V/A and optional cell fields. Both serial and MPI executions use the
same collective gather-to-root XDMF/HDF5 writer. This avoids requiring parallel
HDF5 but requires rank-zero memory for the complete output frame; it is not a
large-scale distributed I/O claim.

The output contains twelve linear tetrahedra per source cell and an auxiliary
centre with the exact mean of the six edge-node values. It therefore represents
the piecewise displacement field exactly, rather than visually replacing it
with P2. HDF5 retains native ten-node coefficients, source cell DOF connectivity,
original cell IDs and the display-to-parent map. Cell orientation is normalized
for visualization. S/E are volume averages of the declared projected response;
MISES is the mean of integration-point equivalent stress; SENER includes the
separately recorded stabilization energy. These are not smoothed peak stresses.

`results.probe`, `sample_points`, and dynamic-history stations use the same true
piecewise interpolation. `results.write_result_fields` preserves this layout.
The ordinary low-level unified P2 writer rejects composite fields instead of
silently drawing the wrong interpolation. XDMF is directly usable in ParaView
with Warp By Vector U; no secondary PVD is required. Existing output files and
partial indices are not overwritten by this writer.

Remaining scope: curved geometry, finite-strain materials,
contact, general spatial loads, cross-partition restart, and distributed output
without root gathering. C3D10M equivalence additionally requires its actual
independent internal-mode, integration and stabilization definitions. None of
these limitations blocks use of the explicitly named, tested linear composite
Step implemented here. See `examples/composite_tet_dynamics` for the standard
project entry point.

## Sources

- Ostien et al., *A 10-node composite tetrahedral finite element for solid mechanics*,
  IJNME 107 (2016), 1145–1170: https://doi.org/10.1002/nme.5218
- Sierra/SM Theory, section 15.3:
  https://www.sandia.gov/files/sierra/SM_Theory_5_26/main/element_formulations.html
- Abaqus solid element library (internal displacement variables):
  https://docs.software.vt.edu/abaqusv2025/English/SIMACAEELMRefMap/simaelm-r-3delem.htm
- Abaqus solid element verification (piecewise-linear modified interpolation):
  https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-3delem.htm
- Abaqus solid element selection (mass and internal-mode control):
  https://docs.software.vt.edu/abaqusv2026/English/SIMACAEELMRefMap/simaelm-c-solidcont.htm


## Whole-mesh convergence and execution improvements (2026-11-17)

The independent manufactured bending field is

\[
u_x=-xy,\quad u_y=\tfrac12[x^2+\nu(y^2-z^2)],\quad u_z=\nu yz,
\]

on a centred unit cube. Its only nonzero stress is $\sigma_{xx}=-Ey$,
$\nabla\cdot\sigma=0$, and total elastic energy is $E/24$. Exact displacement
is prescribed on the exterior; interior DOFs are solved using assembled production
cell stiffness and an independent sparse direct solver. The norm in the report is
**mass-weighted nodal error**, not a claim of exact continuous L2 error. Uniform
and perturbed meshes at 2/3/4 subdivisions test convergence. Ordinary P1/P2
reference operators use independent Basix polynomial gradients and quadrature.

At Poisson ratio 0.4999, plain projected integration still locks badly in energy.
Mean dilatation substantially reduces this error, but does not make coarse meshes
accurate. All error values, including unfavorable results, remain in
`validation/dynamic_parity/composite_accuracy`.

A longitudinal eigenmode with zero transverse displacement, Poisson ratio zero,
and fixed axial ends has exact $\omega_1=\pi$ for unit length/modulus/density.
Refinement 4/8/16 gives the spatial frequency check. The central-difference
frequency relation $\omega_h=2\arcsin(\omega\Delta t/2)/\Delta t$ separately
checks temporal dispersion. This is not a completed transient wave-pulse test.

Affine reference integration is cached once. Physical gradients are transformed
before the deviatoric split. Stiffness contractions use an explicit efficient
contraction path; energy contracts gradients with displacements before forming
strain, avoiding repeated B matrices. The mathematical operators are unchanged.
The writer checks a local geometry/ownership fingerprint each frame and sends
mesh metadata only once. Root gathering of field values remains a limitation;
this improvement does not claim distributed parallel HDF5.

## Independent interior variables: formulation research

`agentfem.elements.composite_tet.composite_tet11` is an experimental 33-DOF
kernel with ten boundary nodes and one independent centre. Its whole-mesh elastic
route is selected by `composite_tet10_policy(independent_center=True)`. With $u=[u_n,u_i]$, tying the centre to the six edge
nodes is the linear map $u=T u_n$, giving $K_t=T^TKT$ and $M_t=T^TMT$.
Tests recover the existing tied kernel exactly (with the same stabilization).
Releasing the centre gives three extra zero modes under projection alone;
positive deviatoric residual stabilization restores six rigid-body modes.

Static internal equilibrium gives

\[
u_i=-K_{ii}^{-1}K_{in}u_n,\qquad
K_{\mathrm{stat}}=K_{nn}-K_{ni}K_{ii}^{-1}K_{in}.
\]

This lowers stationary strain energy relative to imposing the tie. It is NOT an
exact dynamic replacement: with $D(\omega)=K-\omega^2M$, elimination instead gives
$D_{nn}-D_{ni}D_{ii}^{-1}D_{in}$. Interior positive mass must be retained or a
separately verified reduction used. These results advance the internal-variable
question without claiming to reproduce proprietary C3D10M hourglass equations.

## Generic small-strain material updates

The production **tied-centre Tet10** Step accepts `SmallStrainExplicitMaterial`,
a wrapper around the existing `SmallStrainUserMaterial` protocol with density
and a symmetric positive elastic tangent envelope. The envelope is engineering
strain in xx/yy/zz/xy/yz/xz order and supplies the stability estimate. Every
trial tangent is checked for positivity and against this bound. Rate dependence,
multiple material regions and mean-dilatation stabilization with nonlinear
materials are currently rejected. Supporting an arbitrary material protocol is
not a guarantee that an arbitrary constitutive model satisfies these restrictions.

A native J2 adapter reuses the existing radial-return and consistent tangent:

```python
law = constitutive.J2LinearIsotropicHardening(
    young=210e9, poisson=0.3, yield_stress=250e6, hardening_modulus=1e9)
model.material(constitutive.explicit_small_strain_j2(law, density=7800))
u = model.field(fields.displacement(
    domain, element_policy=elements.composite_tet10_policy()))
step = model.step(target=u, dt="auto", steps=100)
result = step.solve_result(output="plastic.xdmf", field_variables=("S", "SENER"))
# Native accepted (owned cell, four quadrature points) values:
peeq = result.fields["STATE_PEEQ"].field
```

Strain is evaluated at the four projected integration points. Each trial reads
only accepted state; stress and internal variables are committed after the
successful global increment, and restored on failure or restart. Stored energy
and cumulative dissipation are distinct. Stress output uses accepted material
response, never elastic reconstruction. Raw named state fields are copied into
`SimulationResult`; checkpoint storage preserves the same-partition state.


## Independent-centre whole-mesh acceptance (2026-11-17)

`composite_tet10_policy(independent_center=True, mean_dilatation=True)` now keeps
all three centre displacement, velocity and acceleration DOFs in the ordinary
Step state. A P2+bubble Basix element supplies DOF ownership **only**; physical
interpolation and integration remain the composite formulation. Internal-node
positive mass participates in every explicit step and the time-step estimate.
No static condensation is used for dynamics.

Verification compares all DOFs against independent global M/K central difference,
checks positive constrained spectra, restart equality, serial/MPI agreement and
exact native-centre output/probes. The displayed centre is the computed internal
node, not the mean of edge displacements. Native and display node counts coincide
for this branch. Unsupported storage-policy combinations reject early.

This closes the independent internal-variable **implementation** gate for the
explicitly named experimental formulation. It does not close the distinct gate
of numerical identity with proprietary Abaqus C3D10M.

The material gate includes global cyclic tension/compression with every accepted
stress/state compared to the established J2 law, and monotonic cumulative plastic
dissipation. The generic nonlinear route remains on the tied-centre, non-B-bar
formulation; extending nonlinear stabilization to the independent-centre branch
is outside this accepted combination and is rejected explicitly.
