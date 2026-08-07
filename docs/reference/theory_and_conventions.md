# Theory and conventions

This page is the mathematical entry point for the public AgentFEM workflow. It
states the conventions that affect model meaning and points to the detailed
material, procedure, output, and API references. Equations describe implemented
or explicitly identified formulations; they are not a catalogue of every
finite-element method.

## Notation and configuration

Unless a guide states otherwise, \(\Omega\) is the reference domain,
\(\Gamma\) its boundary, \(\mathbf{u}\) displacement, \(T\) temperature,
\(\boldsymbol{\sigma}\) Cauchy stress, and \(\mathbf{v}\) an admissible test
function. Material parameters and loads must use one consistent unit system;
AgentFEM records unit roles but does not rescale arbitrary numerical input
silently.

Small-strain analyses use the infinitesimal strain tensor

\[
\boldsymbol{\varepsilon}(\mathbf{u})
= \operatorname{sym}(\nabla \mathbf{u})
= \frac{1}{2}\left(\nabla\mathbf{u}+\nabla\mathbf{u}^{T}\right).
\]

Finite-strain analyses use total-Lagrangian kinematics

\[
\mathbf{F}=\mathbf{I}+\nabla\mathbf{u},\qquad
J=\det\mathbf{F},\qquad
\mathbf{C}=\mathbf{F}^{T}\mathbf{F},\qquad
\mathbf{E}=\frac{1}{2}(\mathbf{C}-\mathbf{I}).
\]

The local condition \(J>0\) is required for an orientation-preserving
deformation. It is checked at quadrature points in supported finite-strain
workflows; a positive global volume does not replace this local condition.

## Static equilibrium

For a small-strain solid, the strong form is

\[
-\nabla\!\cdot\boldsymbol{\sigma}=\mathbf{b}
\quad\text{in }\Omega,
\qquad
\boldsymbol{\sigma}\mathbf{n}=\bar{\mathbf{t}}
\quad\text{on }\Gamma_t.
\]

The corresponding weak equilibrium statement is

\[
\int_{\Omega}
\boldsymbol{\sigma}(\mathbf{u}):\boldsymbol{\varepsilon}(\mathbf{v})\,d\Omega
=
\int_{\Omega}\mathbf{b}\cdot\mathbf{v}\,d\Omega
+\int_{\Gamma_t}\bar{\mathbf{t}}\cdot\mathbf{v}\,d\Gamma.
\]

Linearization and discretization give \(\mathbf{K}\mathbf{u}=\mathbf{F}\).
Nonlinear statics instead solves an incremental residual equation
\(\mathbf{R}(\mathbf{u})=\mathbf{0}\) using a consistent or explicitly
identified tangent \(\mathbf{K}_t=\partial\mathbf{R}/\partial\mathbf{u}\).

## Linear isotropic elasticity

The implemented small-strain isotropic relation is

\[
\boldsymbol{\sigma}
=\lambda\operatorname{tr}(\boldsymbol{\varepsilon})\mathbf{I}
+2\mu\boldsymbol{\varepsilon},
\qquad
\mu=\frac{E}{2(1+\nu)},
\qquad
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}.
\]

For a two-dimensional solid, the Study must state one of the following:

| Assumption | Meaning in the present formulation |
| --- | --- |
| Plane strain | Out-of-plane strain is constrained to zero; the three-dimensional Lamé constants are retained. |
| Plane stress | Out-of-plane stress is zero; the in-plane volumetric coefficient is \(\lambda_{ps}=E\nu/(1-\nu^2)\). |

Axisymmetric elasticity requires radial kinematics and weighted integration and
is not currently implied by either two-dimensional assumption.

## Thermal balance and thermoelastic strain

The transient heat equation used by the thermal workflow is

\[
\rho c_p\dot{T}-\nabla\!\cdot(k\nabla T)=Q.
\]

Steady heat transfer omits the capacity term. The current implicit transient
route uses backward Euler unless the procedure documentation states otherwise.
For sequential thermal stress, isotropic free thermal strain is

\[
\boldsymbol{\varepsilon}_{\mathrm{th}}
=\alpha(T-T_{\mathrm{ref}})\mathbf{I},
\]

and the mechanical stress is evaluated from
\(\boldsymbol{\varepsilon}-\boldsymbol{\varepsilon}_{\mathrm{th}}\) with the
selected plane-stress, plane-strain, or three-dimensional constitutive
relation.

## Structural dynamics

The semi-discrete second-order system is

\[
\mathbf{M}\ddot{\mathbf{u}}
+\mathbf{C}\dot{\mathbf{u}}
+\mathbf{K}\mathbf{u}
=\mathbf{F}(t).
\]

AgentFEM keeps the physical Study separate from the SolutionProcedure:

| Procedure family | Current route | Numerical character |
| --- | --- | --- |
| Standard | Newmark or generalized-\(\alpha\) | Global implicit solve at each accepted time increment |
| Explicit | Central difference with lumped mass | No global stiffness solve at each time increment; stability restricts \(\Delta t\) |

Step, increment, nonlinear iteration, failed attempt, and output frame are
distinct concepts throughout progress, checkpoint, and result records.

## Compressible Neo-Hookean finite strain

For the implemented compressible Neo-Hookean material, the strain-energy
density is

\[
\psi(\mathbf{F})
=\frac{\mu}{2}\left(\operatorname{tr}\mathbf{C}-d\right)
-\mu\ln J
+\frac{\lambda}{2}(\ln J)^2,
\]

where \(d\) is the spatial dimension. The two-dimensional form is a
plane-strain restriction with unit out-of-plane stretch; it is not a finite-
strain plane-stress model. Nearly incompressible behavior should use the
documented displacement-pressure formulation rather than treating an imported
hybrid element name as a complete numerical formulation.

## J2 plasticity and creep state

The supported global J2 route is three-dimensional, small-strain, associative
Mises plasticity with linear isotropic hardening. Its trial yield function and
closed-form plastic multiplier are

\[
f_{\mathrm{trial}}
=q_{\mathrm{trial}}-(\sigma_{y0}+H p_n),
\qquad
\Delta\gamma=\frac{f_{\mathrm{trial}}}{3G+H}
\quad\text{when }f_{\mathrm{trial}}>0.
\]

The supported global creep route is three-dimensional, small-strain,
isothermal Mises power-law creep. Backward Euler solves

\[
\Delta\gamma
=\Delta t\,A(t_{n+1})
\left(\frac{q_{n+1}}{\sigma_{\mathrm{ref}}}\right)^n,
\qquad
q_{n+1}=q_{\mathrm{trial}}-3G\Delta\gamma.
\]

For both routes, quadrature state remains trial state during global Newton
iterations. Stress, plastic/creep strain, equivalent inelastic strain, and the
next-increment state are committed atomically only after the full increment is
accepted. Rejected attempts roll back the transaction before cutback.

Detailed assumptions, parameters, consistent tangents, output fields, tests,
and benchmarks are listed in the
[scientific function reference](scientific_function_reference.md).

## Result locations and recovery

Result meaning includes both the variable and its location:

| Location | Typical quantities | Interpretation |
| --- | --- | --- |
| Nodal / point data | `U`, `T`, selected recovered presentation fields | Values associated with finite-element interpolation nodes |
| Cell data | `S`, `E`, `MISES`, `SENER` for standard projected output | One explicitly projected or averaged value per cell |
| Integration points | `S`, `PE`, `PEEQ`, `CE`, `CEEQ`, constitutive state | Primary nonlinear material evidence before nodal smoothing |
| History | Reactions, energies, probes, resultants, iteration and time records | Scalar or small-vector evolution with an explicit abscissa |

Weighted cell recovery of an integration-point field is

\[
\bar{q}_e=\frac{\sum_p w_p q_{ep}}{\sum_p w_p}.
\]

This operation does not extrapolate to nodes, smooth across neighboring
elements, or average across material boundaries. Presentation fields must not
silently overwrite the constitutive evidence from which they were derived.
See [output variables and field semantics](../result_field_semantics.md) for
the complete naming and processing contract.

## Where detailed theory belongs

Every mature analysis procedure or material page should answer the same
questions:

1. What governing equation or constitutive relation is implemented?
2. Which kinematic, dimensional, and material assumptions apply?
3. Which integration and global-solution algorithm consumes it?
4. Which state and output variables are available, and at what locations?
5. Which combinations are supported or rejected?
6. Which tests, benchmarks, and references support the implementation?

This structure keeps beginner examples concise while making the mathematical
definition available when an engineer needs to inspect or cite it.
