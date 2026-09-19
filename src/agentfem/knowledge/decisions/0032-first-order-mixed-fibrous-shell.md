# Select the first fibrous-shell discretization by stability evidence

## Context

Textile reinforcement forming needs membrane tension/trellising, transverse
shear, in-plane fibre bending, and normal fibre bending without deriving all
four channels from one classical thickness law. In-plane fibre curvature is a
directional derivative of a current unit-fibre direction. If that direction is
computed only from a conventional displacement field, a conforming weak form
requires second derivatives or a verified neighbouring-element
reconstruction. Standard DOLFINx Lagrange spaces are C0 and do not provide that
continuity by themselves.

Rotation-free patch elements and C1 isogeometric shells are valid published
routes, but embedding one of them directly in a material object would make the
constitutive law own element technology and would not reuse AgentFEM's normal
operator/procedure/result lifecycle.

## Decision

AgentFEM will not promote a nominal mixed element merely because its block
forms assemble. A flat-patch linearized compatibility probe was run with H1
primal norms and L2 multiplier norms on 1x1, 2x2, and 4x4 triangular meshes.
The naive P1/DG0 and P2/DG1 pairs became rank deficient under refinement.
P2/CG1 stayed full rank but beta decreased from 0.188 to 0.0964 to 0.0501;
P2/DG0 decreased from 0.240 to 0.0617 to 0.0149. These values do not reject
all mixed methods or dual norms, but they reject these unmodified pairs as the
first public provider.

The first implementation track will therefore be the published rotation-free
neighbouring-element route on ordinary C0 surface meshes. It starts with one
partition-aware cell/facet neighbourhood contract shared with DG and estimator
algorithms. Curvature reconstruction, facet consistency and boundary moments
remain operator responsibilities and must pass the gates below before a Step
exists. A stabilized or locally condensed mixed route remains a later
independent provider and must use `DiscreteInfSupStudy` before promotion.

The constitutive boundary remains unchanged. A mixed candidate, if resumed,
will separate:

- midsurface motion;
- the material normal/director needed for transverse slip and shear;
- independent current warp and weft directions needed for first-gradient
  fibre-curvature terms.

Compatibility, unit-length, and orientation constraints belong to the
Operator/provider boundary, not to `Model` or the constitutive law. Any future
mixed prototype must compare multiplier, augmented-Lagrangian, and condensed
routes; it must not select a large penalty merely because it converges on one
mesh.
The existing `DecoupledFibrousShell` remains element-neutral and receives only
objective point kinematics.

For the initial no-slip layer, the exact constraint set is minimal: one scalar
unit-director equation and two three-component equations equating the
independent warp/weft fields to their convected reference directions. Fibre
unit length and tangency are consequences and remain diagnostics, not extra
multiplier equations. A future slip-enabled layer must declare a different
constraint contract rather than weakening this one implicitly.

A C1/isogeometric or stable mixed provider may be added later behind the same
constitutive and result contracts. It is an independent provider, not a second
public modeling language.

## Promotion gates

The provider stays unavailable to ordinary projects until all of these pass:

1. rigid translation and finite rigid rotation produce zero internal energy;
2. constant membrane, transverse-shear, in-plane-bending, and normal-bending
   patches isolate the intended channel;
3. reconstructed curvature and interior-facet consistency converge under
   refinement; any mixed alternative must also track constraint residuals,
   field norms and `DiscreteInfSupStudy` with benchmark-declared bounds;
4. membrane/shear locking is measured in thin limits and controlled without a
   mesh-dependent user constant;
5. Newton uses a consistent full block Jacobian and reports every block;
6. boundary forces and moments close external work and stored energy;
7. serial/MPI results preserve layer, fibre-family, and field identity;
8. at least one public cantilever-bending and one public in-plane-bending
   benchmark converge independently of the implementation tests.

Tool contact, friction, inter-ply slip, quasi-static explicit dynamics, and
wrinkle validation are later Procedure/Verification gates. Passing shell patch
tests does not imply a validated forming process.

## Consequences

- The public workflow remains `Study -> Model -> assets -> model.step ->
  SimulationResult`.
- `Model` does not acquire shell-specific numerical state.
- Constitutive calibration remains reusable across mixed, rotation-free, and
  future isogeometric providers.
- The neighbouring-element route reuses C0 meshes but adds a topology and
  reconstruction stencil that must be MPI- and orientation-safe.
- Failure to satisfy mixed compatibility or locking gates is reported as an
  unavailable capability rather than hidden behind a nominal shell result.

## Primary references

- Liang, Colmars, and Boisse (2017),
  <https://doi.org/10.1016/j.compositesa.2017.04.024>.
- Bai et al. (2020), <https://doi.org/10.1016/j.compositesa.2020.106135>.
- Steer et al. (2021), <https://doi.org/10.1016/j.ijsolstr.2021.03.001>.
- Duong, Itskov, and Sauer (2022), <https://doi.org/10.1002/nme.6937>.
