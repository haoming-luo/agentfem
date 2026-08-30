# Wave packet with an inclusion

<span class="af-status af-status--release">Release</span>

This example sends a Gaussian-modulated elastic wave packet through a stiff
circular inclusion in a two-dimensional solid. It is the first flagship
dynamic example because the physical model, time integration, boundary
treatment, observations, and result fields remain visible in one executable
workflow.

## Problem

The displacement field satisfies linear elastodynamics,

\[
\rho \ddot{\mathbf u}=\nabla\!\cdot\boldsymbol\sigma,
\qquad
\boldsymbol\sigma
=\lambda\,\operatorname{tr}(\boldsymbol\varepsilon)\mathbf I
+2\mu\boldsymbol\varepsilon,
\]

under a plane-strain assumption. The circular inclusion has the same density
and Poisson ratio as the matrix and twice its Young modulus. A prescribed
wave packet enters at the left boundary, the top and bottom boundaries are
periodic, and a viscous absorbing boundary reduces reflection at the right
edge.

| Model choice | Example value or route |
| --- | --- |
| Domain | \(1.2\,\mu\mathrm{m}\times0.24\,\mu\mathrm{m}\) plate |
| Discretization | structured quadrilateral mesh, first-order displacement |
| Matrix | \(E=227.5\,\mathrm{GPa}\), \(\nu=0.27\), \(\rho=2900\,\mathrm{kg/m^3}\) |
| Inclusion | \(E_\mathrm{i}=2E_\mathrm{m}\), equal \(\nu\) and \(\rho\) |
| Source | Gaussian-modulated sinusoidal displacement at \(40.78\,\mathrm{GHz}\) |
| Transverse boundary | top--bottom periodic projection |
| Outgoing boundary | Lysmer--Kuhlemeyer viscous absorber |
| Procedure | lumped-mass explicit central difference |

## AgentFEM workflow

The public script follows the same order as the engineering problem. Material
regions, source history, boundary behavior, and the explicit solution step are
declared rather than hidden inside a backend loop.

```python
study = studies.dynamic_solid(
    dimension=2,
    assumption="plane_strain",
    method="explicit",
)
model = models.create(study=study, mesh=domain)
u = model.field(fields.displacement(domain, degree=1))

regions = mesh.partition_cells(
    domain,
    matrix=~inclusion,
    stiff_inclusion=inclusion,
)
model.material(matrix_material, region=regions.matrix)
model.material(inclusion_material, region=regions.stiff_inclusion)

periodic = model.periodic(u, master=bottom, slave=top, match_axis="x")
source = model.fix(u, on=left, components=0, value=source_pulse)
absorbing = model.absorbing_boundary(
    on=right,
    density=matrix_material.density,
    pressure_wave_speed=matrix_cp,
    shear_wave_speed=matrix_cs,
)

dynamic_state = state.second_order_state(u)
residual = model.force_balance(
    internal=model.internal_force(dynamic_state.u),
    absorbing=model.boundary_force(absorbing, dynamic_state.v_mid),
)
step = model.step(
    target=u,
    state=dynamic_state,
    residual=residual,
    prescribed=[source],
    constraints=[periodic],
    dt=dt,
    steps=steps,
)
result = step.solve_result(
    output="wave_packet_inclusion_2d.xdmf",
    fields=(dynamic_state.u, dynamic_state.v, material_id),
)
```

The complete script also estimates material wave speeds, derives the stable
increment from the fastest region, records inclusion probes and periodic
mismatch, reports progress, and controls the field-output cadence.

## Run the example

From the repository root in an AgentFEM environment:

```bash
python examples/wave_packet_inclusion_2d.py
```

The full example writes
`examples_output/wave_packet_inclusion_2d.xdmf` with displacement, velocity,
and material-region fields over time. Open the XDMF file in ParaView to play
the transient response, inspect the interaction with the inclusion, and plot
receiver histories.

[View the complete source](https://github.com/haoming-luo/agentfem/blob/main/examples/wave_packet_inclusion_2d.py)
· [Dynamics and waves](../guide/dynamics.md)

## Verification contract

The release does not rely on visual inspection alone. A reduced counterpart
of this workflow is pinned by the machine-readable
[`agentfem.benchmark.wave_release`](https://github.com/haoming-luo/agentfem/blob/main/src/agentfem/knowledge/benchmarks/wave_release.json)
contract. The automated test checks:

- matrix and inclusion wave speeds;
- the declared Courant number;
- peak global and receiver displacement;
- receiver threshold-arrival time;
- top--bottom periodic mismatch.

Run the contract directly with:

```bash
python -m pytest -q \
  tests/test_release_goldens.py::test_wave_release_patch_matches_versioned_golden
```

This Golden contract protects the public software path against regression. It
is distinct from a mesh- and time-converged external validation of attenuation
or absorbing-boundary accuracy.

## Research background

The example is informed by Haoming Luo's work with Anne Tanguy, Anthony
Gravouil, and Valentina Giordano on acoustic wave-packet propagation and
attenuation in biphasic solids:
[*Thermal Transport in a 2D Nanophononic Solid: Role of Bi-Phasic Materials
Properties on Acoustic Attenuation and Thermal Diffusivity*](https://doi.org/10.3390/nano9101471).
