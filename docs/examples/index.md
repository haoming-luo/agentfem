# Example gallery

Examples are organized by the engineering question they answer. Release
workflows are executable assets with numerical contracts; exploratory examples
are marked separately rather than presented as equally mature.

## Linear solid mechanics

<div class="grid cards" markdown>

-   <span class="af-status af-status--release">Release</span>
    **2D static elasticity**

    A compact cantilever model demonstrating study, mesh, regions, displacement,
    material, constraints, traction, solve, standard fields, and verification.

    [:octicons-mark-github-16: Source](https://github.com/haoming-luo/agentfem/blob/main/examples/static_elasticity_2d.py)

</div>

## Heat transfer

<div class="grid cards" markdown>

-   <span class="af-status af-status--release">Release</span>
    **Transient heat transfer**

    Implicit time integration, structured progress, temperature histories, and
    unified field output.

    [:octicons-mark-github-16: Source](https://github.com/haoming-luo/agentfem/blob/main/examples/transient_heat_2d.py)

-   <span class="af-status af-status--engineering">Engineering</span>
    **Hot-wall creep assessment**

    A thermal/thermoelastic workflow followed by an explicitly identified local
    creep assessment.

    [:octicons-mark-github-16: Source](https://github.com/haoming-luo/agentfem/blob/main/examples/creep_hot_wall_assessment.py)

</div>

## Waves and dynamics

<div class="grid cards" markdown>

-   <span class="af-status af-status--release">Release</span>
    **Wave packet and material inclusion**

    Explicit wave propagation through a heterogeneous domain with source,
    boundary models, progress, histories, and time-series fields.

    [:octicons-mark-github-16: Source](https://github.com/haoming-luo/agentfem/blob/main/examples/wave_packet_inclusion_2d.py)

</div>

## Nonlinear solids

<div class="grid cards" markdown>

-   <span class="af-status af-status--engineering">Engineering</span>
    **Abaqus C3D10 periodic hyperelastic cell**

    External quadratic tetrahedra, equation constraints, distributed
    periodicity, finite strain, homogenized response, and visualization output.

    [:octicons-book-24: Workflow](../abaqus_periodic_cell.md)
    [:octicons-mark-github-16: Source](https://github.com/haoming-luo/agentfem/tree/main/examples/abaqus_c3d10_periodic_cell)

</div>

## Creep and time-dependent solids

<div class="grid cards" markdown>

-   <span class="af-status af-status--engineering">Engineering</span>
    **3D implicit creep relaxation**

    Global isothermal power-law creep with quadrature state, physical-time
    cutback, standard creep output, energy evidence, and serial restart.

    [:octicons-mark-github-16: Source](https://github.com/haoming-luo/agentfem/blob/main/examples/implicit_creep_relaxation_3d.py)

</div>

## Simulation to learning

<div class="grid cards" markdown>

-   <span class="af-status af-status--release">Release</span>
    **Elasticity surrogate campaign**

    Parameter sampling, repeated FEM runs, dataset acceptance, surrogate
    validation, applicability guard, and FEM fallback.

    [:octicons-mark-github-16: Source](https://github.com/haoming-luo/agentfem/blob/main/examples/static_elasticity_surrogate_campaign.py)

</div>

## Reading an example well

Do not copy only the solver call. Read the study, regions, material assumptions,
loads and constraints, output request, result policy, and benchmark evidence as
one scientific workflow.
