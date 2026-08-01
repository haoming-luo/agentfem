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

AgentFEM now stores \(E,\nu,\rho,\alpha,k,c_p,T_{\mathrm{ref}}\) in one
thermoelastic property object, while heat and mechanics remain separate
models sharing the mesh, material, and temperature field. Plane strain keeps
the constrained out-of-plane thermal strain; plane stress uses the reduced
constitutive relation.

This follows the same modeling distinction documented by Abaqus: fully
coupled analysis is needed when stress and temperature mutually influence
each other; sequential analysis is appropriate when the coupling is
effectively one way. The current route does not claim monolithic coupling.

## Why high-temperature creep is the next state consumer

The local `ArrheniusPowerLawCreep` relation is normalized at a declared
reference temperature:

\[
A(T)=A_{\mathrm{ref}}
\exp\left[-\frac{Q}{R}
\left(\frac{1}{T}-\frac{1}{T_{\mathrm{ref}}}\right)\right].
\]

The implementation uses the equivalent exponent form in code; all
temperatures are absolute. This local law is useful for calibration and
material-point checks, but power-component credibility requires more:

- creep strain and equivalent creep strain at every integration point;
- an implicit local update and local error estimate over \(\Delta t\);
- global equilibrium at the end of every accepted time increment;
- commit only after convergence and rollback after cutback;
- temperature interpolation at the same integration points;
- restart equivalence, relaxation and one-element paths, then an external
  benchmark.

The implemented J2 quadrature transaction already supplies the essential
committed/trial distinction, analytical tangent channel, rollback, and serial
checkpoint pattern. Global creep should generalize and consume that contract;
it should not introduce a second case-specific state store.

## Current boundary and next gates

Implemented now:

- explicit central difference and implicit Newmark/generalized-\(\alpha\);
- implicit-Euler heat transfer as a model-owned Step;
- sequential isotropic thermal expansion as a visible vector operator;
- normalized Arrhenius temperature dependence at material-point level;
- global 3D J2 quadrature state with analytical tangent and serial restart.
- exact material-point Kachanov-Rabotnov creep-damage updates, Sinh creep,
  and modified-theta curve projection;
- a sequential hot-wall FEM-to-creep assessment example with explicit
  calibration and maturity boundaries.

Next gates:

1. finish field/energy/checkpoint products on top of the now-common
   structured progress events, accepted-increment histories, and result
   manifest across heat, implicit dynamics, and explicit dynamics;
2. portable MPI state identity based on global cells/material regions rather
   than a rank-local array layout;
3. stateful global creep using the common transaction and automatic time
   increments;
4. temperature-dependent property tables and interpolation policies;
5. monolithic coupling only after a real case demonstrates two-way feedback.

References:

- [Abaqus fully coupled thermal-stress analysis](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEANLRefMap/simaanl-c-couptempdisp.htm)
- [Abaqus heat-transfer procedures](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEANLRefMap/simaanl-c-heatproc.htm)
- [Abaqus CREEP user subroutine](https://docs.software.vt.edu/abaqusv2024/English/SIMACAESUBRefMap/simasub-c-creep.htm)
- [Chung and Hulbert generalized-\(\alpha\) paper](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/50422/1640100803_ftp.pdf?isAllowed=y&sequence=1)
