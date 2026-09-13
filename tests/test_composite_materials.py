from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from agentfem import constitutive, fields, materials, mesh, models, results, studies
from agentfem.mesh import abaqus_migration


def _lamina():
    return constitutive.orthotropic_plane_stress_2d(
        ex=135.0e9,
        ey=10.0e9,
        nuxy=0.3,
        gxy=5.0e9,
        density=1600.0,
        name="carbon_epoxy_lamina",
    )


def test_material_frames_are_right_handed_and_separate_from_material():
    frame = materials.MaterialFrame.from_angle(90.0, name="transverse")
    np.testing.assert_allclose(frame.basis, [[0.0, -1.0], [1.0, 0.0]], atol=1.0e-14)
    material = _lamina()
    assignment = materials.oriented(material, frame)

    assert assignment.material is material
    assert assignment.orientation is frame
    assert assignment.density == material.density
    assert assignment.as_dict()["orientation"]["right_handed"] is True

    with pytest.raises(ValueError, match="orthonormal"):
        materials.MaterialFrame("bad", [[1.0, 0.2], [0.0, 1.0]])


def test_convected_material_frame_uses_rotation_while_fibers_can_trellis():
    frame = materials.MaterialFrame.from_angle(0.0, evolution="convected")
    deformation = np.array([[1.0, 0.4], [0.0, 1.0]])
    current = frame.current_basis(deformation)
    np.testing.assert_allclose(current.T @ current, np.eye(2), atol=1.0e-12)
    assert np.linalg.det(current) == pytest.approx(1.0)

    fibers = materials.fiber_frame([1.0, 0.0], [0.0, 1.0])
    warp, weft, _, _ = fibers.convect(deformation)
    assert np.dot(warp, weft) != pytest.approx(0.0)

    constructed = materials.material_frame([1.0, 0.0, 0.0], [0.2, 1.0, 0.0])
    np.testing.assert_allclose(constructed.basis.T @ constructed.basis, np.eye(3))


def test_orthotropic_3d_recovers_isotropic_limit():
    material = constitutive.orthotropic_elastic_3d(
        ex=100.0,
        ey=100.0,
        ez=100.0,
        nuxy=0.25,
        nuxz=0.25,
        nuyz=0.25,
        gxy=40.0,
        gxz=40.0,
        gyz=40.0,
        density=1.0,
    )
    expected = np.array(
        [
            [120.0, 40.0, 40.0, 0.0, 0.0, 0.0],
            [40.0, 120.0, 40.0, 0.0, 0.0, 0.0],
            [40.0, 40.0, 120.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 40.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 40.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 40.0],
        ]
    )
    np.testing.assert_allclose(material.stiffness_voigt, expected)
    assert material.as_dict()["voigt_order"] == ["11", "22", "33", "23", "13", "12"]


def test_oriented_plane_stress_enters_standard_model_and_ufl_operator():
    domain = mesh.rectangle(
        (0.0, 0.0), (1.0, 1.0), (1, 1), comm=MPI.COMM_SELF, cell_type="triangle"
    )
    study = studies.static_solid(dimension=2, assumption="plane_stress")
    model = models.create(study=study, mesh=domain, name="oriented_patch")
    displacement = model.field(fields.displacement(domain, degree=1))
    frame = materials.MaterialFrame.from_angle(90.0)
    assigned = model.material(_lamina(), orientation=frame)
    displacement.value.interpolate(lambda x: np.vstack((x[0], np.zeros_like(x[0]))))

    sigma = constitutive.stress(displacement.value, assigned, study=study)
    sigma_xx = fem.assemble_scalar(fem.form(sigma[0, 0] * ufl.dx))
    expected = _lamina().stiffness_voigt[1, 1]
    assert sigma_xx == pytest.approx(expected, rel=1.0e-12)
    operator = model.stiffness(displacement)
    assert operator.kind == "elastic_stiffness_operator"
    material_fields = results.small_strain_cell_fields(
        displacement,
        assigned,
        study=study,
        variables=("S_MATERIAL", "E_MATERIAL"),
    )
    assert tuple(item.name for item in material_fields) == ("S_MATERIAL", "E_MATERIAL")
    local_stress = material_fields[0].x.array.reshape((-1, 4))[0].reshape((2, 2))
    assert local_stress[1, 1] == pytest.approx(expected, rel=1.0e-11)


def test_oriented_3d_orthotropy_enters_standard_ufl_stress():
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="tetrahedron",
    )
    study = studies.static_solid(dimension=3)
    displacement = fields.displacement(domain, degree=1)
    displacement.value.interpolate(
        lambda x: np.vstack((x[0], np.zeros_like(x[0]), np.zeros_like(x[0])))
    )
    material = constitutive.orthotropic_elastic_3d(
        ex=150.0,
        ey=12.0,
        ez=8.0,
        nuxy=0.2,
        nuxz=0.15,
        nuyz=0.25,
        gxy=5.0,
        gxz=4.0,
        gyz=3.0,
        density=1.0,
    )
    frame = materials.material_frame([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    assigned = materials.oriented(material, frame)
    sigma = constitutive.stress(displacement.value, assigned, study=study)
    sigma_xx = fem.assemble_scalar(fem.form(sigma[0, 0] * ufl.dx))

    assert sigma_xx == pytest.approx(material.stiffness_voigt[1, 1], rel=1.0e-12)


def test_symmetric_cross_ply_has_zero_membrane_bending_coupling():
    lamina = _lamina()
    section = materials.laminate(
        [
            materials.ply(lamina, 0.125e-3, angle=0.0, name="bottom_0"),
            materials.ply(lamina, 0.125e-3, angle=90.0, name="lower_90"),
            materials.ply(lamina, 0.125e-3, angle=90.0, name="upper_90"),
            materials.ply(lamina, 0.125e-3, angle=0.0, name="top_0"),
        ],
        name="cross_ply",
    )
    A, B, D = section.abd()

    np.testing.assert_allclose(B, 0.0, atol=1.0e-10)
    assert np.min(np.linalg.eigvalsh(A)) > 0.0
    assert np.min(np.linalg.eigvalsh(D)) > 0.0
    assert len(section.section_points()) == 12
    assert len({point.id for point in section.section_points()}) == 12

    response = section.evaluate([1.0e-3, 0.0, 0.0], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(response.membrane_force, A @ np.array([1.0e-3, 0.0, 0.0]))
    np.testing.assert_allclose(response.bending_moment, D @ np.array([0.0, 0.0, 2.0]))
    assert response.as_dict()["section_points"][0]["id"].startswith("bottom_0:")


def test_reviewed_abaqus_composite_inventory_lowers_without_guessing(tmp_path):
    source = tmp_path / "layup.inp"
    source.write_text(
        "\n".join(
            (
                "*Element, type=C3D8, elset=LAYUP",
                "1,1,2,3,4,5,6,7,8",
                "*Solid Section, elset=LAYUP, composite",
                "0.000125, MAT_A, 0.0",
                "0.000125, MAT_B, 90.0",
            )
        ),
        encoding="utf-8",
    )
    inventory = abaqus_migration.plan(source).sections[0]
    section = materials.laminate_from_abaqus_section(
        inventory,
        {"MAT_A": _lamina(), "MAT_B": _lamina()},
        reviewed_by="composite-reviewer",
    )

    assert section.source == "reviewed_abaqus_composite_section"
    assert section.metadata["reviewed_by"] == "composite-reviewer"
    assert [ply.angle_degrees for ply in section.plies] == [0.0, 90.0]
    with pytest.raises(ValueError, match="reviewed_by"):
        materials.laminate_from_abaqus_section(
            inventory, {"MAT_A": _lamina(), "MAT_B": _lamina()}, reviewed_by=""
        )


def test_angle_ply_transform_preserves_elastic_energy():
    stiffness = _lamina().stiffness_voigt
    transformed = materials.transformed_reduced_stiffness(stiffness, 37.0)
    strain = np.array([0.002, -0.0003, 0.001])
    angle = np.deg2rad(37.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    global_tensor = np.array([[strain[0], 0.5 * strain[2]], [0.5 * strain[2], strain[1]]])
    local_tensor = rotation.T @ global_tensor @ rotation
    local_strain = np.array([local_tensor[0, 0], local_tensor[1, 1], 2.0 * local_tensor[0, 1]])

    global_energy = 0.5 * strain @ transformed @ strain
    local_energy = 0.5 * local_strain @ stiffness @ local_strain
    assert global_energy == pytest.approx(local_energy, rel=1.0e-12)


def test_fabric_surface_keeps_tension_shear_and_bending_independent():
    frame = materials.fiber_frame([1.0, 0.0], [0.0, 1.0], name="plain_weave")
    tension = constitutive.tabulated_response(
        [0.0, 0.05, 0.10], [0.0, 50.0, 120.0], extrapolation="linear"
    )
    shear = constitutive.tabulated_response(
        [0.0, 0.2, 0.5], [0.0, 10.0, 80.0], symmetry="odd", extrapolation="linear"
    )
    surface = constitutive.decoupled_fabric_surface(
        frame=frame,
        warp_tension=tension,
        weft_tension=tension,
        shear=shear,
        bending_stiffness=np.diag([2.0, 3.0, 1.0]),
    )

    identity = surface.evaluate(np.eye(2))
    np.testing.assert_allclose(identity.membrane_resultants, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(identity.bending_moments, 0.0)
    assert identity.stored_energy == pytest.approx(0.0)

    response = surface.evaluate([[1.05, 0.2], [0.0, 1.0]], curvature=[0.1, 0.0, -0.2])
    assert response.kinematics.warp_strain == pytest.approx(0.05)
    assert response.kinematics.shear_angle > 0.0
    assert response.membrane_resultants[0] == pytest.approx(50.0)
    np.testing.assert_allclose(response.bending_moments, [0.2, 0.0, -0.2])
    assert response.tangent.shape == (6, 6)
    assert response.stored_energy > 0.0
    assert surface.as_dict()["fem_integration"] == "not_yet_available"


def test_fabric_tension_only_does_not_create_compressive_yarn_force():
    frame = materials.fiber_frame([1.0, 0.0], [0.0, 1.0])
    tension = constitutive.tabulated_response(
        [0.0, 0.1], [0.0, 100.0], extrapolation="linear"
    )
    shear = constitutive.tabulated_response(
        [0.0, 0.2], [0.0, 10.0], symmetry="odd", extrapolation="linear"
    )
    surface = constitutive.decoupled_fabric_surface(
        frame=frame,
        warp_tension=tension,
        weft_tension=tension,
        shear=shear,
        bending_stiffness=np.zeros((3, 3)),
    )
    response = surface.evaluate([[0.9, 0.0], [0.0, 1.0]])
    assert response.membrane_resultants[0] == 0.0
    assert response.stored_energy == pytest.approx(0.0)


def test_tabulated_energy_handles_negative_domains_and_odd_channels():
    signed = constitutive.tabulated_response(
        [-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], extrapolation="linear"
    )
    odd = constitutive.tabulated_response(
        [0.0, 1.0], [0.0, 2.0], symmetry="odd", extrapolation="linear"
    )

    assert signed.energy(-0.5) == pytest.approx(0.25)
    assert signed.energy(0.5) == pytest.approx(0.25)
    assert odd.energy(-0.5) == pytest.approx(0.25)
    assert odd.energy(0.5) == pytest.approx(0.25)
