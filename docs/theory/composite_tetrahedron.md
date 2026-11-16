# Composite tetrahedral operators and the C3D10M boundary

## Scope and public entry point

`elements.composite_tet10(coordinates)` constructs reusable **local** small-strain
operators for a straight-sided, affine ten-node tetrahedron. It returns consistent
and positive lumped mass, projected gradients, strain matrices, stiffness, and a
central-difference local time-step bound. Coordinates follow Abaqus Tet10 ordering:
vertices 0–3; edges 01, 12, 20, 03, 13, 23. Vector DOFs are node-major xyz.

This is a formulation-development API, not a whole-mesh Step provider. It does
not register `C3D10M` as executable, change INP interpretation, or replace the
existing P2/HRZ and Tet15 routes. Curved/moved mid-edge geometry is rejected.
The implementation is independently written from mathematical definitions.

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
A global driver must still apply its safety factor and include contact, constraints
and other stiffness contributions. This API does not select a production run dt.

## Verification and next integration gate

`tests/test_composite_tet.py` checks partition gradients, affine completeness,
constant engineering strain, positive mass, exact consistent-mass inertia,
rigid motions, stiffness rank, energy directional derivatives, all 24 vertex
permutations, length/density/stiffness scaling, and rejection of unsupported input.
The stabilization test also retains the failing nine-null-mode formulation as
an explicit regression; it must never be mistaken for a usable default.

Remaining implementation work is independent of external blade files: a registered
execution provider, distributed assembly/ownership, internal-state semantics,
and physical-field output for the piecewise interpolation. An ordinary Basix P2
space cannot represent this displacement basis. These must be implemented before
whole-mesh use. C3D10M equivalence additionally requires its actual internal-mode,
integration and stabilization definitions, not just the name or connectivity.
No application simulation was performed as part of this development increment.

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
