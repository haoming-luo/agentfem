# Abaqus UMAT and UHYPER Migration Boundary

## What “compatible” should mean

AgentFEM should not advertise arbitrary Abaqus user subroutines as directly
portable merely because their Fortran entry-point names are known. A credible
bridge must preserve the material-point contract:

- old and new deformation gradients, time, temperature, field variables,
  properties, and state variables enter one update;
- the update returns Cauchy stress, updated state, energy, and an algorithmic
  tangent in a declared tensor convention;
- the global nonlinear driver stores state at quadrature points, commits it
  only after a converged increment, and uses the returned tangent consistently;
- one-point, one-element, and full load-path comparisons establish equivalence.

`constitutive.user_material` now records this solver-neutral boundary. It is an
interface contract, not an executable Abaqus runtime.

## The neutral contract is explicit

A finite-strain tangent cannot be identified by its array shape alone.
AgentFEM therefore separates three public assets:

- `MaterialStateVariable` names one scalar or tensor internal variable and
  declares its shape, scalar-broadcast or explicit tensor initial value, unit,
  description and optional output field name; this permits, for example, an
  identity initial plastic deformation gradient rather than silently filling
  every tensor state with zero;
- `MaterialStateSchema` gives the complete state vector a stable versioned
  identity and owns initialization, validation and unpacking;
- `MaterialTangentConvention` declares the stress measure, conjugate
  kinematic perturbation, reference/current configuration, storage layout,
  component order, shear convention and objective-rate convention.

For example, a native total-Lagrangian provider may declare

```python
tangent = constitutive.MaterialTangentConvention.first_piola_deformation_gradient()
```

which represents

\[
\mathbb A = \frac{\partial\mathbf P}{\partial\mathbf F}.
\]

The restricted Abaqus UMAT route instead declares the Abaqus spatial Jacobian
and engineering-shear ordering:

```python
tangent = constitutive.MaterialTangentConvention.abaqus_umat()
```

These declarations are not interchangeable conversions. A provider or adapter
must implement and verify the transformation required by the global residual
it serves.

An update becomes eligible for a future global Newton consumer only through

```python
response = constitutive.validated_material_update(material, point)
```

This call checks the input state against the material schema, runs the update,
requires a complete tangent and state declaration, and rejects any schema or
convention drift in the returned response. Legacy `MaterialPointOutput`
objects without those declarations remain inspectable but fail closed when
`require_global_newton_contract()` is called.

## Inspect before adapting

An existing Fortran asset can now enter a deterministic inspection gate:

```bash
agentfem inspect-user-material legacy.for \
  --write legacy.inspection.json --json
```

The command retains a SHA-256 source identity, detects a single UMAT or UHYPER
entry point, inventories include files, distinguishes known Abaqus utility
calls from project subroutines, and recommends one of three routes:

- restricted UHYPER energy adapter;
- restricted UMAT material-point adapter;
- manual interface identification/adaptation.

This is useful migration automation, not source translation. A clean report
means only that the source is a candidate for the next adapter stage. A call to
an Abaqus utility is an explicit blocker until its semantics are replaced or a
compatible support library is provided.

## Why UHYPER is the first practical bridge

UHYPER is narrower than UMAT. It describes isotropic hyperelastic energy and
its derivatives with respect to invariants. That maps naturally to
AgentFEM/UFL energy formulations and has no general inelastic stress-history
algorithm to reproduce. A first adapter can therefore:

1. compile a restricted UHYPER routine into a shared library;
2. call it at selected invariants;
3. map the returned energy derivatives into an AgentFEM constitutive law;
4. compare stress and tangent over prescribed deformation paths;
5. validate a single finite element before a periodic cell.

## Why UMAT requires a constitutive driver

UMAT is a stateful integration algorithm. Abaqus calls it at every material
point and expects updated Cauchy stress, `STATEV`, and `DDSDDE`. A useful bridge
therefore requires more than a Python wrapper:

- quadrature-point storage for stress and state variables;
- trial/commit/rollback semantics across Newton iterations;
- Abaqus tensor ordering and engineering-shear conversion;
- finite-rotation and objective-rate conventions;
- the exact meaning of the old/new deformation gradients;
- handling of time-step suggestions and possibly nonsymmetric tangents;
- explicit policy for Abaqus utility calls, includes, orientations, thermal
  coupling, and element-dependent modified deformation gradients.

A routine limited to the standard arguments, `PROPS`, and `STATEV` is a
reasonable first migration target. Routines that call Abaqus utilities or
depend on solver internals need source-level adaptation and cannot be promised
as drop-in compatible.

Abaqus defines `DDSDDE` for finite-strain UMAT as a Jacobian associated with
the increment of Kirchhoff stress and the strain increment, with direct
components followed by engineering shear components. It also notes that local
orientations rotate the stored basis and that some first-order elements pass a
modified deformation gradient. AgentFEM records these facts as adapter
metadata; it does not relabel `DDSDDE` as
\(\partial\mathbf P/\partial\mathbf F\).

## Progressive implementation route

| Stage | Deliverable | Evidence gate |
| --- | --- | --- |
| 0 | Solver-neutral material-point input/output contract | validation tests |
| 0.5 | Source inspection and route selection | fingerprinted machine-readable report |
| 1 | UHYPER energy adapter | deformation-path stress/tangent comparison |
| 2 | Quadrature state, trial/commit/rollback | one-element inelastic test |
| 3 | Restricted UMAT shared-library adapter | identical material-point paths |
| 4 | Global nonlinear integration | one-element and benchmark agreement |
| 5 | Wider Abaqus conventions and utilities | capability matrix per routine |

The important architectural decision is that AgentFEM's global solver consumes
the neutral material-point protocol. Native Python/C++ materials and Abaqus
adapters become alternative providers behind the same boundary. This avoids
making the public model language depend on Abaqus, while preserving a realistic
route for valuable user-material libraries.

## References

- [Abaqus 2025 UMAT reference](https://docs.software.vt.edu/abaqusv2025/English/SIMACAESUBRefMap/simasub-c-umat.htm),
  including `DDSDDE`, component storage, deformation-gradient and orientation
  conventions.
- C. Miehe, “Numerical computation of algorithmic (consistent) tangent moduli
  in large-strain computational inelasticity,” *Computer Methods in Applied
  Mechanics and Engineering* 134 (1996), 223--240.
  [doi:10.1016/0045-7825(96)01019-5](https://doi.org/10.1016/0045-7825(96)01019-5).
