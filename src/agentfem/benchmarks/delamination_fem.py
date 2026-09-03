"""Executable finite-element delamination benchmark providers.

These providers are deliberately separate from the analytical oracles in
``mixed_mode``.  A result produced here comes from an assembled bulk and
zero-thickness interface problem; it is never relabelled beam theory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mixed_mode import (
    DelaminationBenchmarkSpec,
    DelaminationEnergyReleaseCurve,
    compliance_energy_release_curve,
)


@dataclass(frozen=True)
class DCBFiniteElementPoint:
    """One load--opening solution for a discretized DCB specimen."""

    crack_length: float
    effective_crack_length: float
    load: float
    opening: float
    compliance: float
    element_size: float
    elements_per_arm: int
    newton_iterations: int
    residual_norm: float

    def summary(self) -> dict[str, object]:
        return {
            "kind": "dcb_finite_element_point",
            **self.__dict__,
        }


@dataclass(frozen=True)
class DCBFiniteElementCurve:
    """Structure-level DCB compliance and energy-release evidence."""

    specification: DelaminationBenchmarkSpec
    points: tuple[DCBFiniteElementPoint, ...]
    energy_release: DelaminationEnergyReleaseCurve
    source: str

    @property
    def element_size(self) -> float:
        return max(point.element_size for point in self.points)

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.dcb-finite-element-curve.v1",
            "specification": self.specification.summary(),
            "source": self.source,
            "points": [point.summary() for point in self.points],
            "energy_release": self.energy_release.summary(),
            "discretization": {
                "cell": "Q1 quadrilateral",
                "interface": "paired zero-thickness line facets",
                "kinematics": "small-strain plane stress or plane strain",
                "precrack": "fully failed interface facets",
            },
        }


def dcb_finite_element_curve(
    spec: DelaminationBenchmarkSpec,
    *,
    crack_length,
    load: float,
    specimen_length: float,
    elements_along: int,
    elements_per_arm: int,
    poisson: float = 0.3,
    assumption: str = "plane_stress",
    interface_stiffness: float | None = None,
    solver_options=None,
) -> DCBFiniteElementCurve:
    """Solve a linear-elastic DCB family with one fixed-path interface.

    The interface is geometrically split over the complete specimen. Facets
    behind each requested crack tip are initialized as a precrack; facets
    ahead of it remain an intact, high-stiffness cohesive tie.  The function
    returns independently assembled compliance points and derives ``G_I``
    from their finite-element compliance curve.

    This provider is an executable verification rung, not an ASTM material
    test reduction.  Crack tips must align with the uniform axial mesh so the
    requested and represented geometries cannot silently differ.
    """

    if not isinstance(spec, DelaminationBenchmarkSpec) or spec.kind != "dcb":
        raise ValueError("dcb_finite_element_curve requires a DCB specification.")
    lengths = np.asarray(crack_length, dtype=float).reshape(-1)
    if lengths.size < 3 or np.any(np.diff(lengths) <= 0.0):
        raise ValueError("DCB finite-element evidence needs three increasing cracks.")
    selected_load = float(load)
    length = float(specimen_length)
    nx = int(elements_along)
    ny = int(elements_per_arm)
    if (
        not np.isfinite(selected_load)
        or selected_load <= 0.0
        or not np.isfinite(length)
        or length <= float(lengths[-1])
        or nx < 4
        or ny < 1
    ):
        raise ValueError("DCB load, specimen length and mesh counts are invalid.")
    dx = length / nx
    mesh_indices = np.rint(lengths / dx).astype(int)
    effective = mesh_indices * dx
    tolerance = 128.0 * np.finfo(float).eps * max(length, 1.0)
    if np.any(np.abs(effective - lengths) > tolerance):
        raise ValueError(
            "Every DCB crack length must align with the uniform axial mesh."
        )
    selected_assumption = str(assumption).strip().lower().replace("-", "_")
    if selected_assumption not in {"plane_stress", "plane_strain"}:
        raise ValueError("DCB assumption must be plane_stress or plane_strain.")
    stiffness = (
        1.0e6 * spec.elastic_modulus / spec.arm_thickness
        if interface_stiffness is None
        else float(interface_stiffness)
    )
    if not np.isfinite(stiffness) or stiffness <= 0.0:
        raise ValueError("interface_stiffness must be finite and positive.")

    points = tuple(
        _dcb_point(
            spec,
            crack_index=int(index),
            load=selected_load,
            specimen_length=length,
            elements_along=nx,
            elements_per_arm=ny,
            poisson=float(poisson),
            assumption=selected_assumption,
            interface_stiffness=stiffness,
            solver_options=solver_options,
        )
        for index in mesh_indices
    )
    source = f"AgentFEM assembled DCB Q1/cohesive curve; {nx}x{2 * ny} bulk cells"
    energy = compliance_energy_release_curve(
        spec,
        crack_length=[point.effective_crack_length for point in points],
        load=[point.load for point in points],
        compliance=[point.compliance for point in points],
        source=source,
    )
    return DCBFiniteElementCurve(
        specification=spec,
        points=points,
        energy_release=energy,
        source=source,
    )


def _dcb_point(
    spec,
    *,
    crack_index,
    load,
    specimen_length,
    elements_along,
    elements_per_arm,
    poisson,
    assumption,
    interface_stiffness,
    solver_options,
) -> DCBFiniteElementPoint:
    # Heavy numerical imports remain local so benchmark metadata and analytical
    # oracles stay usable in lightweight documentation environments.
    import ufl
    from dolfinx import fem
    from dolfinx import mesh as dolfinx_mesh
    from mpi4py import MPI

    from agentfem import (
        constitutive,
        constraints,
        fields,
        fracture,
        interfaces,
        operators,
        results,
        solvers,
        studies,
    )

    if MPI.COMM_WORLD.size != 1:
        raise NotImplementedError(
            "The initial DCB finite-element verification provider is serial; "
            "distributed cohesive assembly is verified through its separate gate."
        )
    h = float(spec.arm_thickness)
    nx = int(elements_along)
    ny_arm = int(elements_per_arm)
    x_values = np.linspace(0.0, float(specimen_length), nx + 1)
    y_values = np.linspace(-h, h, 2 * ny_arm + 1)
    points = np.asarray(
        [(x, y) for y in y_values for x in x_values],
        dtype=float,
    )

    def node(i, j):
        return j * (nx + 1) + i

    cells = np.asarray(
        [
            [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
            for j in range(2 * ny_arm)
            for i in range(nx)
        ],
        dtype=np.int64,
    )
    mid = ny_arm
    interface_facets = np.asarray(
        [[node(i, mid), node(i + 1, mid)] for i in range(nx)],
        dtype=np.int64,
    )
    upper_cells = np.arange(ny_arm * nx, 2 * ny_arm * nx, dtype=np.int64)
    split = interfaces.split_conforming_line_interface(
        points,
        cells,
        interface_facets,
        positive_cells=upper_cells,
    )
    domain = interfaces.create_dolfinx_split_mesh(
        split,
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    displacement = fields.displacement(domain)
    study = studies.static_solid(dimension=2, assumption=assumption)
    material = constitutive.isotropic_elastic(
        young=spec.elastic_modulus,
        poisson=poisson,
        density=1.0,
        name="DCB isotropic elastic",
    )
    internal = operators.internal_force_vector(
        displacement.value,
        displacement.test,
        material,
        study=study,
        measure=spec.width * ufl.dx,
    )

    facet_dimension = domain.topology.dim - 1
    upper_left = dolfinx_mesh.locate_entities_boundary(
        domain,
        facet_dimension,
        lambda x: np.isclose(x[0], 0.0) & (x[1] >= -1.0e-12),
    )
    lower_left = dolfinx_mesh.locate_entities_boundary(
        domain,
        facet_dimension,
        lambda x: np.isclose(x[0], 0.0) & (x[1] <= 1.0e-12),
    )
    tagged_facets = np.concatenate((upper_left, lower_left))
    tagged_values = np.concatenate(
        (
            np.full(upper_left.size, 1, dtype=np.int32),
            np.full(lower_left.size, 2, dtype=np.int32),
        )
    )
    if np.unique(tagged_facets).size != tagged_facets.size:
        raise RuntimeError("DCB crack-mouth facet sets must not overlap.")
    order = np.argsort(tagged_facets)
    tags = dolfinx_mesh.meshtags(
        domain,
        facet_dimension,
        tagged_facets[order],
        tagged_values[order],
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
    load_parameter = fem.Constant(domain, 0.0)
    traction = load_parameter / h
    external = operators.OperatorForm(
        name="DCB crack-mouth load",
        expression=(
            traction * displacement.test[1] * ds(1)
            - traction * displacement.test[1] * ds(2)
        ),
        kind="surface_force",
        role="vector",
        family="dcb_mode_i",
    )
    bulk = operators.residual_operator(
        (internal - external).expression,
        name="R_DCB_bulk",
        family="small_strain_linear_elasticity",
    )
    tangent = operators.linearize(bulk, displacement)
    law = interfaces.bilinear_cohesive(
        strength=interface_stiffness,
        fracture_energy=interface_stiffness,
        initial_stiffness=interface_stiffness,
    )
    cohesive = fracture.cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
        tangential="tie",
        tangential_stiffness=interface_stiffness,
        thickness=spec.width,
    )
    topology = cohesive.assembler.topology
    facet_midpoints = np.mean(
        split.coordinates[topology.negative_nodes],
        axis=1,
    )
    represented_crack = float(crack_index * specimen_length / nx)
    precrack = facet_midpoints[:, 0] < represented_crack
    if int(np.count_nonzero(precrack)) != int(crack_index):
        raise RuntimeError("DCB precrack facets do not match the requested mesh tip.")
    cohesive.initialize_precrack(precrack)
    residual = fracture.FiniteStrainCohesiveResidual(bulk, cohesive)
    right_bottom = lambda x: np.isclose(x[0], specimen_length) & np.isclose(x[1], -h)
    right_top = lambda x: np.isclose(x[0], specimen_length) & np.isclose(x[1], h)
    bcs = (
        constraints.component_dirichlet(
            displacement,
            0,
            on=right_bottom,
            value=0.0,
            name="remove rigid x translation",
        ).bc,
        constraints.component_dirichlet(
            displacement,
            1,
            on=right_bottom,
            value=0.0,
            name="remove rigid y motion",
        ).bc,
        constraints.component_dirichlet(
            displacement,
            0,
            on=right_top,
            value=0.0,
            name="remove rigid rotation",
        ).bc,
    )
    dof_coordinates = displacement.space.tabulate_dof_coordinates()

    def opening(function):
        values = function.x.array.reshape((-1, 2))
        left = np.isclose(dof_coordinates[:, 0], 0.0)
        upper = left & (dof_coordinates[:, 1] > 1.0e-12)
        lower = left & (dof_coordinates[:, 1] < -1.0e-12)
        return float(np.mean(values[upper, 1]) - np.mean(values[lower, 1]))

    equilibrium = fracture.FiniteStrainCohesiveEquilibrium(
        residual,
        tangent,
        displacement,
        bcs=bcs,
        load_parameter=load_parameter,
        solver_options=solver_options
        or solvers.newton(
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            maximum_iterations=12,
            linear_solver=solvers.direct_solver(),
        ),
        control_displacement=opening,
        reaction=lambda _function: results.reaction_resultant(
            residual,
            on=lambda x: np.isclose(x[0], 0.0) & (x[1] > 1.0e-12),
            component=1,
        ),
    )
    solved = equilibrium(load=load, branch="monotonic", cycle=0)
    if not solved["converged"]:
        raise RuntimeError("DCB finite-element equilibrium did not converge.")
    opening_value = float(solved["control_displacement"])
    if not np.isfinite(opening_value) or opening_value <= 0.0:
        raise RuntimeError("DCB finite-element opening is not positive and finite.")
    return DCBFiniteElementPoint(
        crack_length=represented_crack,
        effective_crack_length=represented_crack,
        load=float(load),
        opening=opening_value,
        compliance=opening_value / float(load),
        element_size=max(specimen_length / nx, h / ny_arm),
        elements_per_arm=ny_arm,
        newton_iterations=int(solved["iterations"]),
        residual_norm=float(equilibrium.last_info.residual_norm),
    )


__all__ = [
    "DCBFiniteElementCurve",
    "DCBFiniteElementPoint",
    "dcb_finite_element_curve",
]
