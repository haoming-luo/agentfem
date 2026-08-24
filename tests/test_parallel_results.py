from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from dolfinx import fem
from mpi4py import MPI
import pytest
import ufl

from agentfem import (
    constitutive,
    datasets,
    fields,
    mesh,
    models,
    results,
    studies,
    surrogates,
)


def test_point_and_path_sampling_are_partition_independent():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed point sampling requires at least two MPI ranks")

    domain = mesh.rectangle(
        (0.0, 0.0),
        (2.0, 1.0),
        (8, 4),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    field = fem.Function(V, name="response")
    field.interpolate(lambda x: np.vstack((x[0] + x[1], x[0] - x[1])))
    field.x.scatter_forward()

    points = np.array(
        ((0.0, 0.0), (0.5, 0.25), (1.0, 0.5), (1.5, 0.75), (2.0, 1.0))
    )
    expected = np.column_stack(
        (points[:, 0] + points[:, 1], points[:, 0] - points[:, 1])
    )
    np.testing.assert_allclose(
        results.sample_points(field, points),
        expected,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        results.probe(field, at=(1.0, 0.5)),
        [1.5, 0.5],
    )
    path = results.sample_path(
        field,
        start=(0.0, 0.0),
        end=(2.0, 1.0),
        count=5,
    )
    np.testing.assert_allclose(path.values, expected, atol=1.0e-14)

    grid = results.sample_rectilinear_grid(
        field,
        bbox=(0.0, 2.0, 0.0, 1.0),
        shape=(5, 3),
        reduction="magnitude",
    )
    xx, yy = np.meshgrid(grid.axes[0], grid.axes[1], indexing="xy")
    expected_magnitude = np.sqrt((xx + yy) ** 2 + (xx - yy) ** 2)
    np.testing.assert_allclose(grid.values, expected_magnitude, atol=1.0e-14)
    assert grid.inside.all()


def test_point_sampling_rejects_rank_inconsistent_requests():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed point sampling requires at least two MPI ranks")

    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 2),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    field = fem.Function(V, name="rank_safe")
    field.interpolate(lambda x: x[0])
    point = ((0.25 + 0.1 * MPI.COMM_WORLD.rank, 0.5),)

    with pytest.raises(ValueError, match="identical coordinates"):
        results.sample_points(field, point)


def test_observation_grid_is_identical_across_mesh_partitions(tmp_path):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed observation sampling requires at least two MPI ranks")

    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (6, 6),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    field = fem.Function(V, name="temperature")
    field.interpolate(lambda x: 300.0 + x[0] + 2.0 * x[1])
    field.x.scatter_forward()
    grid = surrogates.regular_grid(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(5, 4),
    )

    sample = datasets.fem_observation_sample(field, grid, unit="K")

    expected = 300.0 + grid.axes[0][:, None] + 2.0 * grid.axes[1][None, :]
    np.testing.assert_allclose(sample.values, expected, atol=1.0e-13)
    assert sample.mask is None
    digests = MPI.COMM_WORLD.allgather(sample.values.tobytes())
    assert all(item == digests[0] for item in digests)
    root = str(tmp_path) if MPI.COMM_WORLD.rank == 0 else None
    root = MPI.COMM_WORLD.bcast(root, root=0)
    output = sample.write(Path(root) / "observation.npz")
    if MPI.COMM_WORLD.rank == 0:
        assert output.is_file()


def test_small_strain_projection_is_distributed_and_partition_safe():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed projection requires at least two MPI ranks")

    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.25),
        (6, 2),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    displacement = fields.displacement(domain).value
    displacement.interpolate(lambda x: np.vstack((0.01 * x[0], -0.002 * x[1])))
    material = constitutive.isotropic_elastic(
        young=200.0e9,
        poisson=0.3,
        density=7800.0,
    )
    study = studies.linear_static(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_stress",
    )

    stress, strain = results.small_strain_cell_fields(
        displacement,
        material,
        study=study,
        variables=("S", "E"),
    )

    np.testing.assert_allclose(
        results.average(strain, measure=ufl.dx(domain=domain)),
        np.diag([0.01, -0.002]),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    assert stress.name == "S"


def test_parallel_paraview_series_has_one_dataset_with_point_and_cell_fields(tmp_path):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("collective ParaView output requires at least two MPI ranks")
    root = str(tmp_path) if MPI.COMM_WORLD.rank == 0 else None
    root = MPI.COMM_WORLD.bcast(root, root=0)
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.25),
        (4, 2),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    displacement = fields.displacement(domain).value
    displacement.interpolate(lambda x: np.vstack((0.01 * x[0], 0.0 * x[1])))
    stress = fem.Function(fem.functionspace(domain, ("DG", 0)), name="MISES")
    stress.x.array[:] = 10.0
    output = results.write_parallel_vtk_series(
        Path(root) / "mixed_parallel.pvd",
        (SimpleNamespace(solution=displacement, load_factor=1.0),),
        ((stress,),),
    )
    MPI.COMM_WORLD.barrier()

    if MPI.COMM_WORLD.rank == 0:
        datasets_xml = ET.parse(output).findall(".//DataSet")
        assert len(datasets_xml) == 1
        parallel_grid = output.parent / datasets_xml[0].attrib["file"]
        tree = ET.parse(parallel_grid)
        displacement_array = tree.find(
            ".//PPointData/PDataArray[@Name='Displacement']"
        )
        assert displacement_array is not None
        assert displacement_array.attrib["NumberOfComponents"] == "3"
        assert tree.find(".//PCellData/PDataArray[@Name='MISES']") is not None


def test_named_boundary_reaction_and_static_result_are_distributed(tmp_path):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed reaction extraction requires at least two MPI ranks")

    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (8, 2),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.linear_static(
            physics="solid_mechanics",
            dimension=2,
            assumption="plane_stress",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        constitutive.isotropic_elastic(
            young=1.0e3,
            poisson=0.0,
            density=1.0,
        )
    )
    left_geometry = mesh.face(domain, axis="x", value=0.0, name="left", tag=1)
    bottom_geometry = mesh.face(domain, axis="y", value=0.0, name="bottom", tag=2)
    right_geometry = mesh.face(domain, axis="x", value=1.0, name="right", tag=3)
    left = mesh.tagged_boundary_region(
        domain, left_geometry.facet_tags, tag=1, name="left"
    )
    bottom = mesh.tagged_boundary_region(
        domain, bottom_geometry.facet_tags, tag=2, name="bottom"
    )
    right = mesh.tagged_boundary_region(
        domain, right_geometry.facet_tags, tag=3, name="right"
    )
    model.fix(displacement, on=left, component=0, value=0.0)
    model.fix(displacement, on=bottom, component=1, value=0.0)
    model.traction((1.0, 0.0), on=right)

    boundary_evidence = model.audit_boundaries(strict=True)
    assert boundary_evidence["left"]["measure"] == pytest.approx(0.2)
    assert boundary_evidence["right"]["measure"] == pytest.approx(0.2)
    assert boundary_evidence["bottom"]["measure"] == pytest.approx(1.0)

    step = model.step(target=displacement)
    root = str(tmp_path) if MPI.COMM_WORLD.rank == 0 else None
    root = MPI.COMM_WORLD.bcast(root, root=0)
    simulation = step.solve_result(output=Path(root) / "static_result.xdmf")

    assert results.reaction_resultant(
        step.problem,
        on=left,
        component=0,
    ) == pytest.approx(-0.2)
    np.testing.assert_allclose(
        simulation.quantities["external_force_resultant"].value,
        [0.2, 0.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        simulation.quantities["reaction_force_resultant"].value,
        [-0.2, 0.0],
        atol=1.0e-11,
    )
    assert simulation.quantities["relative_force_balance_error"].value < 1.0e-10
    assert {"S", "E", "MISES"} <= set(simulation.fields)
    assert "SENER" not in simulation.fields
    assert simulation.artifacts["fields_paraview"].is_file()
    assert simulation.metadata["field_output"]["layout"] == (
        "scientific_xdmf_plus_single_paraview_dataset"
    )
    if MPI.COMM_WORLD.rank == 0:
        datasets_xml = ET.parse(
            simulation.artifacts["fields_paraview"]
        ).findall(".//DataSet")
        assert len(datasets_xml) == 1
