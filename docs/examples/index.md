# Examples

Examples are executable parts of the software, not screenshots of possible
features. Each example identifies the physical problem, numerical route,
important output, and present maturity. Release examples carry numerical
contracts; engineering examples exercise broader workflows that still require
problem-specific qualification.

## Example index

| Example | Physics and procedure | Maturity |
| --- | --- | --- |
| [Wave packet with an inclusion](wave_packet_inclusion.md) | Heterogeneous solid wave, explicit central difference | Release |
| [2D static elasticity](#2d-static-elasticity) | Plane-strain linear solid, direct linear solve | Release |
| [Axisymmetric thick cylinder](#axisymmetric-thick-cylinder) | Full-revolution linear solid, direct linear solve | Release |
| [Transient heat transfer](#transient-heat-transfer) | Heat equation, backward Euler | Release |
| [Abaqus periodic hyperelastic cell](#abaqus-periodic-hyperelastic-cell) | Imported 3D quadratic mesh, equations, finite strain | Engineering |
| [Implicit creep relaxation](#implicit-creep-relaxation) | 3D power-law creep, global/local Newton and cutback | Engineering |
| [Elasticity surrogate campaign](#elasticity-surrogate-campaign) | Repeated FEM, accepted dataset, surrogate and fallback | Release |
| [Science supershear data protocol](#science-supershear-data-protocol) | Public-data identity, XLSX inspection, research handoff | Experimental |

## Wave packet with an inclusion

<span class="af-status af-status--release">Release</span>

A Gaussian-modulated elastic wave crosses a stiff circular inclusion. The
workflow brings together heterogeneous material regions, explicit central
difference, a time-dependent source, periodic projection, an absorbing
boundary, probes, progress events, and time-series fields.

[Read the complete flagship example](wave_packet_inclusion.md)
· [Source code](https://github.com/haoming-luo/agentfem/blob/main/examples/wave_packet_inclusion_2d.py)

## 2D static elasticity

<span class="af-status af-status--release">Release</span>

A compact cantilever demonstrates Study, mesh, named boundaries, displacement,
isotropic elasticity, strong constraints, traction, a linear step, standard
`U/S/E/MISES` output, a Golden observable, and release-quality verification.

```bash
python examples/static_elasticity_2d.py
```

[Source code](https://github.com/haoming-luo/agentfem/blob/main/examples/static_elasticity_2d.py)
· [Linear-solid guide](../guide/solid_mechanics.md)
· [Golden benchmark record](https://github.com/haoming-luo/agentfem/blob/main/src/agentfem/knowledge/benchmarks/linear_static_cantilever.json)

## Axisymmetric thick cylinder

<span class="af-status af-status--release">Release</span>

A Q2 meridian model of a long pressurized cylinder demonstrates native
axisymmetric strain, full `(r, theta, z)` stress, the `2*pi*r` physical measure,
pressure loading, standard output, and comparison with the analytical Lamé
solution. The same formulation is consumed by the stateful J2 and implicit
creep providers.

```bash
python examples/axisymmetric_thick_cylinder.py
```

[Source code](https://github.com/haoming-luo/agentfem/blob/main/examples/axisymmetric_thick_cylinder.py)
· [Solid-mechanics guide](../guide/solid_mechanics.md#axisymmetric-solids)
· [Benchmark record](https://github.com/haoming-luo/agentfem/blob/main/src/agentfem/knowledge/benchmarks/axisymmetric_lame_cylinder.json)

## Transient heat transfer

<span class="af-status af-status--release">Release</span>

This case exercises capacity and conduction operators, backward-Euler time
integration, accepted-increment progress, temperature histories, unified field
output, and a release Golden observable.

```bash
python examples/transient_heat_2d.py
```

[Source code](https://github.com/haoming-luo/agentfem/blob/main/examples/transient_heat_2d.py)
· [Heat-transfer guide](../guide/heat_transfer.md)

## Abaqus periodic hyperelastic cell

<span class="af-status af-status--engineering">Engineering</span>

This workflow directly imports an Abaqus C3D10H mesh and equation constraints,
preserves quadratic-node and hybrid/constant-pressure identity, solves a
P2/DG0 finite-strain Neo-Hookean problem with exact periodic constraints, and
writes homogenized and visualization results.

**Research background.** This benchmark is informed by Haoming Luo's work with
Julie Diani and Kostas Danas on the finite-strain response of nonlinear elastic
composites containing particles or voids: [*Numerical estimation via remeshing
and analytical modeling of nonlinear elastic composites*](https://www.sciencedirect.com/science/article/pii/S0997753823001687).

Read the [complete periodic-cell workflow](../abaqus_c3d10h_periodic_cell.md) before
running the case; it explains the source files, element mapping, periodic
semantics, nonlinear controls, and output.

[Source directory](https://github.com/haoming-luo/agentfem/tree/main/examples/abaqus_c3d10h_periodic_cell)
· [Finite-strain theory](../reference/theory_and_conventions.md#compressible-neo-hookean-finite-strain)

## Implicit creep relaxation

<span class="af-status af-status--engineering">Engineering</span>

A three-dimensional power-law creep problem exercises quadrature state,
backward-Euler constitutive integration, analytical consistent tangent,
physical-time automatic incrementation, rollback/cutback, creep output, energy
evidence, and restart. The same global step can consume a prescribed
temperature field through the Arrhenius material factory.

```bash
python examples/implicit_creep_relaxation_3d.py
```

[Source code](https://github.com/haoming-luo/agentfem/blob/main/examples/implicit_creep_relaxation_3d.py)
· [Creep and inelasticity](../guide/creep_and_inelasticity.md)
· [Constitutive equations](../reference/theory_and_conventions.md#j2-plasticity-and-creep-state)

## Hot-wall creep assessment

<span class="af-status af-status--engineering">Engineering</span>

The hot-wall case performs a thermal/thermoelastic finite-element workflow and
then an explicitly identified local creep assessment. It demonstrates how the
software distinguishes a global field solution from a material-point or
assessment-level model.

```bash
python examples/creep_hot_wall_assessment.py
```

[Source code](https://github.com/haoming-luo/agentfem/blob/main/examples/creep_hot_wall_assessment.py)

## Elasticity surrogate campaign

<span class="af-status af-status--release">Release</span>

The campaign varies declared parameters, runs the same finite-element model,
records failures and evidence, creates an accepted scientific dataset, trains
a surrogate, checks its applicability domain, and retains an FEM fallback.

```bash
python examples/static_elasticity_surrogate_campaign.py
```

[Source code](https://github.com/haoming-luo/agentfem/blob/main/examples/static_elasticity_surrogate_campaign.py)
· [Simulation to learning](../guide/simulation_to_learning.md)

## Science supershear data protocol

`examples/science_supershear_v5_protocol.py` verifies the pinned Dryad file
identities and inventories the public Science 2023 crack-speed, wave-speed,
Mach-cone, material-response, and SED/KED workbooks. It is the data boundary
for the [Science Supershear V5 protocol](../research/science_supershear_v5.md),
not a parameter fit disguised as an example.

## Reading an example

Do not copy only the final solver call. Read the Study, dimensional assumption,
mesh and regions, material data, loads and constraints, procedure, output
request, result policy, and benchmark evidence as one scientific workflow.
When adapting an example, any change to physics, discretization, material law,
or loading may require new verification evidence.
