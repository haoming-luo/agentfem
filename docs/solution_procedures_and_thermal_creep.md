# Solution Procedures, Thermal Stress, and the Creep Route

## Problem type is not a solver algorithm

`Study` answers what is being modeled: solid mechanics or heat transfer,
static or transient, dimension, and kinematic assumptions.
`SolutionProcedure` answers how it is solved:

| Physical equation | Standard route | Explicit route |
| --- | --- | --- |
| \(K u=F\) | linear solve | not applicable |
| \(R(u)=0\) | incremental Newton | not yet available |
| \(C\dot T+KT=Q\) | implicit Euler | not yet available |
| \(R(u,CE,t)=0\) | backward-Euler creep + Newton | not applicable |
| \(M\ddot u+C\dot u+Ku=F\) | Newmark or generalized-\(\alpha\) | central difference |

This keeps `second_order_dynamics` from meaning “central difference” merely
because that was the first implemented algorithm. It also keeps
`incrementation`, Newton settings, time integration, and output cadence as
separate decisions.

The implicit structural route currently assumes linear operators and
time-invariant prescribed displacement supports. Newmark uses average
acceleration by default. Generalized-\(\alpha\) derives
\(\alpha_m,\alpha_f,\beta,\gamma\) from the spectral radius at infinite
frequency and provides controllable high-frequency damping. Nonlinear
implicit dynamics and moving supports require additional residual,
linearization, and prescribed-kinematics work.

## The useful first thermal-mechanical route

Many component analyses do not need a monolithic temperature-displacement
solve. If temperature affects stress but deformation and dissipation do not
meaningfully affect heat transfer, the transparent route is:

1. solve \(\rho c_p \dot T-\nabla\cdot(k\nabla T)=Q\);
2. retain the temperature history;
3. evaluate
   \(\epsilon_{\mathrm{th}}=\alpha(T-T_{\mathrm{ref}})I\);
4. solve mechanical equilibrium with the named equivalent operator
   \(F_{\mathrm{thermal}}\).

AgentFEM stores \(E,\nu,\rho,\alpha,k,c_p,T_{\mathrm{ref}}\) in one
thermoelastic property object. Constants may be replaced by inspectable
temperature tables. Tabulated conductivity or specific heat automatically
selects a nonlinear implicit-Euler residual using the conservative enthalpy
difference \([h(T_{n+1})-h(T_n)]/\Delta t\), with \(dh/dT=\rho c_p(T)\).
The thermal step can capture its
accepted fields in a `FieldHistory`; the receiving creep step samples that
history at its own physical increment endpoints. Interpolation, range policy,
units, time coordinates, and the content hash remain part of the transfer
contract. Saved nodal histories are keyed by physical DOF coordinates and can
be restored across changed MPI partitions and rank counts. Plane strain keeps
the constrained out-of-plane thermal strain; plane stress uses the reduced
constitutive relation.

This follows the same modeling distinction documented by Abaqus: fully
coupled analysis is needed when stress and temperature mutually influence
each other; sequential analysis is appropriate when the coupling is
effectively one way. The current route does not claim monolithic coupling.

## The first global creep state consumer

The local `ArrheniusPowerLawCreep` relation is normalized at a declared
reference temperature:

\[
A(T)=A_{\mathrm{ref}}
\exp\left[-\frac{Q}{R}
\left(\frac{1}{T}-\frac{1}{T_{\mathrm{ref}}}\right)\right].
\]

The implementation uses the equivalent exponent form in code; all
temperatures are absolute. The global three-dimensional Mises power-law route
now accepts either an isothermal material or a scalar/finite-element
temperature input for the normalized Arrhenius law. It provides:

- committed/trial `CE` and `CEEQ` at every quadrature point;
- a safeguarded backward-Euler local scalar solve over \(\Delta t\);
- an analytical algorithmic consistent tangent consumed by global Newton;
- atomic commit after acceptance and rollback after Newton, local, or
  maximum-creep-increment failure;
- complete regional material assignment, fixed or automatic physical-time
  increments, dissipation history, and portable full-Step checkpoint/restart;
- MPI-safe regional constitutive dispatch and a portable quadrature-state
  archive keyed by original physical cell, quadrature point, rule, material
  contract, and mesh fingerprint;
- integration-point temperature consumption, `TEMP` output, increment-wise
  temperature evidence, and temperature identity in checkpoints;
- scalar, finite-element, or physical-time `FieldHistory` temperature input;
  attempted increments move the live temperature to their endpoint, while
  rollback and restart restore it to the accepted physical time;
- a three-dimensional homogeneous stress-relaxation Golden contract.

This is implemented by reusing `QuadratureTransaction`; creep does not own a
second private state store. Checkpoint recovery reconstructs accepted stress
from displacement and committed `CE` without advancing a fictitious time
increment. The route follows the same consistent-linearization principle that
underpins robust Newton convergence in inelastic finite elements.

Arrhenius power-law creep is therefore a global consumer, while Sinh and
Kachanov--Rabotnov remain material-point capabilities. The accepted transient
temperature history is now an explicit transfer object; it does not silently
promote damage, mesh regularization, or structural rupture prediction.

## Current boundary and next gates

Implemented now:

- explicit central difference and implicit Newmark/generalized-\(\alpha\);
- implicit-Euler heat transfer as a model-owned Step;
- conservative state-dependent conductivity/capacity using the same Step,
  Result, progress, heat-ledger, rollback, checkpoint, and MPI contracts;
- accepted-time field capture and temperature-history transfer to creep;
- coordinate-keyed field-history persistence verified in both two-rank-to-one
  and one-rank-to-two-rank directions;
- bounded temperature-property tables for sequential thermoelastic stiffness
  and thermal expansion;
- sequential isotropic thermal expansion as a visible vector operator;
- normalized Arrhenius temperature dependence at material-point and global
  integration-point level;
- global 3D J2 quadrature state with regional materials, analytical tangent,
  and portable full-Step restart;
- cumulative J2 restart history, analytical uniaxial Golden verification,
  quadrature S/PE/PEEQ/MISES and nodal RF result fields, cyclic amplitude, physical-increment cutback,
  prescribed work, and internal-energy decomposition;
- exact material-point Kachanov-Rabotnov creep-damage updates, Sinh creep,
  and modified-theta curve projection;
- a sequential hot-wall FEM-to-creep assessment example with explicit
  calibration and maturity boundaries.
- global 3D power-law creep with backward Euler, analytical tangent, shared
  transaction, automatic cutback, CE/CEEQ/S/MISES/RF/TEMP, prescribed
  Arrhenius temperature fields, dissipation history, regional materials,
  portable full-Step restart, MPI-portable quadrature state, and a
  relaxation Golden contract;
- a 3D component contract in which accepted transient heat states drive the
  global Arrhenius creep step on the same physical clock.
- the published NAFEMS R0027 Test 7 thick-cylinder stress oracle and an
  executable elastic-preload-to-creep structural route. The fast analytical
  contract is automated. The scheduled structural gate uses a one-layer Q2
  hexahedral quarter sector, direct volume-weighted quadrature-point stress
  errors, and a creep endpoint-rate integration estimator. The 4 by 8 mesh
  crosses the declared 8% radial/hoop/axial error gate; the 3 by 6 mesh does
  not, so the refinement evidence remains visible rather than being replaced
  by a smoothed contour comparison. Halving the endpoint-rate tolerance from
  `5e-4` to `2.5e-4` changes the final maximum CEEQ by approximately `1.5e-7`
  relatively and leaves all three reported terminal stress errors unchanged
  to their reported precision.

For creep, nonlinear equilibrium convergence and time-integration accuracy
are separate decisions. `creep_strain_error_tolerance` limits

\[
\max_q\left|\dot{\bar\varepsilon}^{cr}_{q,n+1}
-\dot{\bar\varepsilon}^{cr}_{q,n}\right|\Delta t,
\]

at owned integration points. An otherwise converged increment that exceeds
the tolerance is rolled back and cut back atomically. This follows the public
NAFEMS/Abaqus `CETOL` meaning; `maximum_inelastic_increment` remains the
separate bound on the magnitude of the accepted CEEQ increment.

The J2 and creep Steps compile their live displacement-to-quadrature strain
expression once. Newton iterations, line-search trials, rollback recovery and
result reconstruction reevaluate its coefficients without repeatedly asking
FFCx to rediscover the same compiled expression. This is an implementation
detail below the public workflow, but it is important for long adaptive paths.

Homogeneous isothermal creep regions also use a vectorized quadrature update.
This is an execution optimization, not a second constitutive model: the batch
follows the same safeguarded backward-Euler equation and returns the same
algorithmic tangent as the scalar material-point update. Temperature-dependent
and multi-material regions retain pointwise material dispatch so scientific
semantics take precedence over batching.

Next gates:

1. improve the 3D global Newton/increment controller toward the public
   axisymmetric deck's 40-increment execution efficiency without weakening
   the scientific gates, and add intermediate-time observables when a
   transient rather than terminal benchmark contract is available;
2. exercise tabulated heat properties and accepted history transfer on an
   external 3D power-component benchmark with time-step convergence;
3. finish field/energy/checkpoint products on top of the common complete
   execution-event trace, accepted-increment histories, status files, and
   result manifest across heat, implicit dynamics, and explicit dynamics;
4. exercise the portable full-Step archive on larger partition changes and
   scheduled HPC restart campaigns;
5. add multi-element and external power-component benchmarks, then promote
   the experimental global Newton MPI path;
6. introduce K-R/Liu--Murakami damage only with near-failure time control and
   a declared mesh-regularization policy;
7. add a distributed archive backend only when field histories outgrow the
   current compact root-gathered format;
8. monolithic coupling only after a real case demonstrates two-way feedback.

References:

- [Abaqus fully coupled thermal-stress analysis](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEANLRefMap/simaanl-c-couptempdisp.htm)
- [Abaqus heat-transfer procedures](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEANLRefMap/simaanl-c-heatproc.htm)
- [Abaqus CREEP user subroutine](https://docs.software.vt.edu/abaqusv2024/English/SIMACAESUBRefMap/simasub-c-creep.htm)
- [Chung and Hulbert generalized-\(\alpha\) paper](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/50422/1640100803_ftp.pdf?isAllowed=y&sequence=1)
- [Simo and Taylor, consistent tangent operators](https://escholarship.org/uc/item/9cp19009)
- [Duxbury, Crook, and Lyons, consistent plasticity/creep integration](https://doi.org/10.1002/nme.1620370803)
- [NAFEMS R0027 Test 7 as reproduced in the Abaqus verification guide](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-nafemscreep.htm)
