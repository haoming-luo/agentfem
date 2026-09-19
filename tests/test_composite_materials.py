from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from agentfem import (
    constitutive,
    fields,
    materials,
    mechanics,
    mesh,
    models,
    results,
    studies,
)
from agentfem.step_providers import step_capability
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
    np.testing.assert_allclose(
        response.membrane_force, A @ np.array([1.0e-3, 0.0, 0.0])
    )
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
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    global_tensor = np.array(
        [[strain[0], 0.5 * strain[2]], [0.5 * strain[2], strain[1]]]
    )
    local_tensor = rotation.T @ global_tensor @ rotation
    local_strain = np.array(
        [local_tensor[0, 0], local_tensor[1, 1], 2.0 * local_tensor[0, 1]]
    )

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
    assert surface.as_dict()["fem_integration"] == "finite_kinematics_in_plane_membrane"


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


def test_director_shell_kinematics_are_objective_and_detect_curvature():
    reference = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    angle = np.deg2rad(41.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rigid = mechanics.director_shell_kinematics(
        reference,
        rotation @ reference,
        rotation @ np.array([0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(rigid.membrane_strain, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(rigid.transverse_shear, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(rigid.curvature_change, 0.0, atol=1.0e-14)
    assert rigid.area_ratio == pytest.approx(1.0)

    bent = mechanics.director_shell_kinematics(
        reference,
        reference,
        [0.0, 0.0, 1.0],
        director_gradient=np.array([[-2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    )
    assert bent.curvature_change[0, 0] == pytest.approx(2.0)
    np.testing.assert_allclose(bent.membrane_strain, 0.0)

    current = np.array([[1.2, 0.1], [0.2, 0.9], [0.1, -0.2]])
    director = np.array([-0.1, 0.2, 0.97])
    director /= np.linalg.norm(director)
    gradient = np.array([[0.1, -0.2], [0.05, 0.08], [0.0, 0.03]])
    before = mechanics.director_shell_kinematics(
        reference,
        current,
        director,
        director_gradient=gradient,
    )
    after = mechanics.director_shell_kinematics(
        reference,
        rotation @ current,
        rotation @ director,
        director_gradient=rotation @ gradient,
    )
    np.testing.assert_allclose(after.membrane_strain, before.membrane_strain)
    np.testing.assert_allclose(after.transverse_shear, before.transverse_shear)
    np.testing.assert_allclose(after.curvature_change, before.curvature_change)
    np.testing.assert_allclose(after.current_normal, rotation @ before.current_normal)


def test_fiber_curve_kinematics_separate_in_plane_and_normal_bending():
    tangents = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    reference_direction = np.array([1.0, 0.0, 0.0])
    in_plane = mechanics.fiber_curve_kinematics(
        tangents,
        tangents,
        reference_direction,
        reference_direction,
        current_direction_gradient=np.array(
            [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]
        ),
    )
    assert in_plane.in_plane_curvature_change == pytest.approx(2.0)
    assert in_plane.normal_curvature_change == pytest.approx(0.0)

    normal = mechanics.fiber_curve_kinematics(
        tangents,
        tangents,
        reference_direction,
        reference_direction,
        current_direction_gradient=np.array(
            [[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]]
        ),
    )
    assert normal.in_plane_curvature_change == pytest.approx(0.0)
    assert normal.normal_curvature_change == pytest.approx(3.0)

    angle = np.deg2rad(31.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = mechanics.fiber_curve_kinematics(
        rotation @ tangents,
        rotation @ tangents,
        rotation @ reference_direction,
        rotation @ reference_direction,
        current_direction_gradient=rotation
        @ np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]),
    )
    assert rotated.in_plane_curvature_change == pytest.approx(2.0)
    assert rotated.normal_curvature_change == pytest.approx(0.0)


def test_surface_deformation_gradient_maps_tangent_plane_and_is_objective():
    reference = np.array([[1.0, 0.2], [0.0, 1.1], [0.1, -0.1]])
    current = np.array([[1.2, 0.1], [0.3, 0.9], [0.2, -0.3]])
    deformation = mechanics.surface_deformation_gradient(reference, current)
    reference_normal = np.cross(reference[:, 0], reference[:, 1])
    reference_normal /= np.linalg.norm(reference_normal)
    current_normal = np.cross(current[:, 0], current[:, 1])
    current_normal /= np.linalg.norm(current_normal)

    np.testing.assert_allclose(deformation @ reference, current)
    np.testing.assert_allclose(deformation @ reference_normal, current_normal)

    angle = np.deg2rad(28.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = mechanics.surface_deformation_gradient(reference, rotation @ current)
    np.testing.assert_allclose(rotated, rotation @ deformation)


def test_multilayer_fabric_stack_retains_varying_layer_frames_and_energy():
    tension = constitutive.tabulated_response(
        [0.0, 0.1], [0.0, 100.0], extrapolation="linear"
    )
    shear = constitutive.tabulated_response(
        [0.0, 0.5], [0.0, 20.0], symmetry="odd", extrapolation="linear"
    )

    def surface(angle, name):
        radians = np.deg2rad(angle)
        c, s = np.cos(radians), np.sin(radians)
        return constitutive.decoupled_fabric_surface(
            name=name,
            frame=materials.fiber_frame(
                [c, s], [-s, c], name=f"{name}_frame"
            ),
            warp_tension=tension,
            weft_tension=tension,
            shear=shear,
            bending_stiffness=np.zeros((3, 3)),
        )

    stack = constitutive.fabric_stack(
        [
            constitutive.fabric_layer(
                surface(0.0, "zero"),
                name="family_0",
                physical_layer_ids=("ply_0_bottom", "ply_0_top"),
            ),
            constitutive.fabric_layer(surface(45.0, "bias"), name="ply_45"),
        ],
        name="forming_stack",
    )
    deformation = np.array([[1.1, 0.2], [0.0, 1.0]])
    response = stack.evaluate(deformation)

    assert tuple(item.layer_name for item in response.layers) == ("family_0", "ply_45")
    assert response.by_name("family_0").multiplicity == 2
    assert response.by_name("family_0").response.kinematics.warp_strain != pytest.approx(
        response.by_name("ply_45").response.kinematics.warp_strain
    )
    assert response.stored_energy == pytest.approx(
        response.by_name("family_0").stored_energy
        + response.by_name("ply_45").stored_energy
    )
    np.testing.assert_allclose(
        response.by_name("family_0").membrane_resultants,
        2.0 * response.by_name("family_0").response.membrane_resultants,
    )
    assert stack.as_dict()["physical_layer_count"] == 3
    assert stack.as_dict()["resultant_policy"] == "retain_per_layer_frames"

    with pytest.raises(ValueError, match="Physical layer IDs must be unique"):
        constitutive.fabric_stack(
            [
                constitutive.fabric_layer(
                    surface(0.0, "first"),
                    name="first",
                    physical_layer_ids=("duplicate",),
                ),
                constitutive.fabric_layer(
                    surface(45.0, "second"),
                    name="second",
                    physical_layer_ids=("duplicate",),
                ),
            ]
        )


def test_multilayer_fabric_stack_builds_one_additive_symbolic_energy():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    displacement = fields.displacement(domain, degree=1)
    displacement.value.interpolate(
        lambda x: np.vstack((0.05 * x[0] + 0.02 * x[1], np.zeros_like(x[0])))
    )
    first = _fabric_membrane()
    angle = np.deg2rad(45.0)
    second = constitutive.decoupled_fabric_surface(
        frame=materials.fiber_frame(
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)],
        ),
        warp_tension=first.warp_tension,
        weft_tension=first.weft_tension,
        shear=first.shear,
        bending_stiffness=np.zeros((3, 3)),
    )
    stack = constitutive.fabric_stack(
        [
            constitutive.fabric_layer(
                first,
                name="family_0",
                physical_layer_ids=("ply_0_bottom", "ply_0_top"),
            ),
            constitutive.fabric_layer(second, name="ply_45"),
        ]
    )

    stacked = fem.assemble_scalar(
        fem.form(stack.membrane_energy_ufl(displacement.value) * ufl.dx)
    )
    separate = fem.assemble_scalar(
        fem.form(
            (
                2.0 * first.membrane_energy_ufl(displacement.value)
                + second.membrane_energy_ufl(displacement.value)
            )
            * ufl.dx
        )
    )
    assert stacked == pytest.approx(separate, rel=1.0e-13)


def test_forming_limits_are_explicit_screening_not_a_wrinkle_claim():
    surface = _fabric_membrane()
    response = surface.evaluate([[1.05, 0.2], [0.0, 1.0]])
    tangents = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    warp_bending = mechanics.fiber_curve_kinematics(
        tangents,
        tangents,
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        current_direction_gradient=np.array(
            [[0.0, 0.0], [1.5, 0.0], [0.0, 0.0]]
        ),
    )
    limits = constitutive.fabric_forming_limits(
        warp_tensile_strain=0.1,
        weft_tensile_strain=0.1,
        trellising_angle=20.0,
        in_plane_curvature=1.0,
    )
    assessment = limits.assess(response, warp_bending=warp_bending)

    assert assessment.governing_mode == "in_plane_bending"
    assert assessment.maximum_utilization == pytest.approx(1.5)
    assert assessment.accepted is False
    assert assessment.as_dict()["interpretation"] == (
        "screening_against_user_declared_limits"
    )

    with pytest.raises(ValueError, match="Curvature limits require"):
        limits.assess(response)


def test_fibrous_shell_local_law_keeps_four_energy_channels_independent():
    tangents = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    director = np.array([0.1, -0.2, 1.0])
    director /= np.linalg.norm(director)
    shell_kinematics = mechanics.director_shell_kinematics(
        tangents,
        tangents,
        director,
    )
    warp_bending = mechanics.fiber_curve_kinematics(
        tangents,
        tangents,
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        current_direction_gradient=np.array(
            [[0.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        ),
    )
    weft_bending = mechanics.fiber_curve_kinematics(
        tangents,
        tangents,
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        current_direction_gradient=np.array(
            [[0.0, -4.0], [0.0, 0.0], [0.0, 5.0]]
        ),
    )
    membrane_2d = _fabric_membrane()
    membrane_3d = constitutive.decoupled_fabric_surface(
        frame=materials.fiber_frame(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ),
        warp_tension=membrane_2d.warp_tension,
        weft_tension=membrane_2d.weft_tension,
        shear=membrane_2d.shear,
        bending_stiffness=np.zeros((3, 3)),
    )
    law = constitutive.decoupled_fibrous_shell(
        membrane_3d,
        transverse_shear_stiffness=np.diag([10.0, 20.0]),
        in_plane_bending_stiffness=np.diag([2.0, 3.0]),
        normal_bending_stiffness=np.diag([4.0, 5.0]),
    )

    response = law.evaluate(
        mechanics.surface_deformation_gradient(tangents, tangents),
        shell_kinematics=shell_kinematics,
        warp_bending=warp_bending,
        weft_bending=weft_bending,
    )

    np.testing.assert_allclose(response.membrane.membrane_resultants, 0.0)
    np.testing.assert_allclose(response.in_plane_curvature, [2.0, 4.0])
    np.testing.assert_allclose(response.normal_curvature, [3.0, 5.0])
    np.testing.assert_allclose(response.in_plane_bending_moments, [4.0, 12.0])
    np.testing.assert_allclose(response.normal_bending_moments, [12.0, 25.0])
    np.testing.assert_allclose(
        response.transverse_shear_resultants,
        np.diag([10.0, 20.0]) @ shell_kinematics.transverse_shear,
    )
    assert response.tangent.shape == (9, 9)
    np.testing.assert_allclose(response.tangent, response.tangent.T)
    assert set(response.energy_channels) == {
        "membrane",
        "transverse_shear",
        "in_plane_bending",
        "normal_bending",
    }
    assert response.energy_channels["membrane"] == pytest.approx(0.0)
    assert response.stored_energy == pytest.approx(
        sum(response.energy_channels.values())
    )


def test_fibrous_shell_refuses_overlapping_legacy_bending_ownership():
    with pytest.raises(ValueError, match="zero bending stiffness"):
        constitutive.decoupled_fibrous_shell(
            _fabric_membrane(bending=1.0),
            transverse_shear_stiffness=np.eye(2),
            in_plane_bending_stiffness=np.eye(2),
            normal_bending_stiffness=np.eye(2),
        )


def _fabric_membrane(*, bending=0.0):
    tension = constitutive.tabulated_response(
        [0.0, 0.05, 0.10],
        [0.0, 50.0, 120.0],
        extrapolation="linear",
    )
    shear = constitutive.tabulated_response(
        [0.0, 0.2, 0.5],
        [0.0, 10.0, 80.0],
        symmetry="odd",
        extrapolation="linear",
    )
    return constitutive.decoupled_fabric_surface(
        frame=materials.fiber_frame([1.0, 0.0], [0.0, 1.0]),
        warp_tension=tension,
        weft_tension=tension,
        shear=shear,
        bending_stiffness=np.eye(3) * float(bending),
    )


def test_fabric_membrane_enters_standard_step_and_solves_a_loaded_patch(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(study=studies.static_membrane(), mesh=domain)
    displacement = model.field(fields.displacement(domain, degree=1))
    material = model.material(_fabric_membrane())
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 1.0),
        name="right",
        tag=2,
    )
    model.fix(displacement, on=left)
    model.traction((1.0, 0.0), on=right)

    capability = step_capability(
        model,
        target=displacement,
        options={"material": material},
    )
    assert capability["provider"]["name"] == "decoupled_fabric_membrane_static"
    output = results.output_plan(
        tmp_path / "fabric_membrane",
        field=results.field_output(
            "U",
            "FABRIC_STRAIN",
            "FABRIC_N",
            "FABRIC_WARP",
            "FABRIC_WEFT",
            "SENER",
            configuration="reference",
        ),
    )
    step = model.step(
        target=displacement,
        material=material,
        increments=2,
        output=output,
        progress=False,
    )
    simulation = step.solve_result()

    assert step.last_solve_info.converged
    assert step.summary()["result_field_recovery"] == "provider"
    assert np.max(displacement.value.x.array[::2]) > 0.0
    checks = step.last_solve_info.increments[-1].checks
    assert checks["minimum_quadrature_J"] > 0.0
    assert checks["membrane_energy"] > 0.0
    assert simulation.status == "completed"
    assert {
        "Displacement",
        "FABRIC_GENERALIZED_STRAIN",
        "FABRIC_GENERALIZED_RESULTANT",
        "FABRIC_WARP_DIRECTION",
        "FABRIC_WEFT_DIRECTION",
        "SENER",
    }.issubset(simulation.fields)
    assert simulation.fields["FABRIC_GENERALIZED_STRAIN"].processing["method"] == (
        "global_l2_projection"
    )
    assert simulation.artifacts["field_history"].is_file()
    assert simulation.artifacts["result_manifest"].is_file()


def test_multilayer_fabric_membrane_preserves_per_layer_result_identity(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(study=studies.static_membrane(), mesh=domain)
    displacement = model.field(fields.displacement(domain, degree=1))
    first = _fabric_membrane()
    angle = np.deg2rad(45.0)
    second = constitutive.decoupled_fabric_surface(
        name="bias_surface",
        frame=materials.fiber_frame(
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)],
        ),
        warp_tension=first.warp_tension,
        weft_tension=first.weft_tension,
        shear=first.shear,
        bending_stiffness=np.zeros((3, 3)),
    )
    stack = constitutive.fabric_stack(
        [
            constitutive.fabric_layer(
                first,
                name="ply 0",
                physical_layer_ids=("ply 0 bottom", "ply 0 top"),
            ),
            constitutive.fabric_layer(second, name="ply +45"),
        ],
        name="forming_stack",
    )
    material = model.material(stack)
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 1.0),
        name="right",
        tag=2,
    )
    model.fix(displacement, on=left)
    model.traction((1.0, 0.0), on=right)
    output = results.output_plan(
        tmp_path / "fabric_stack",
        field=results.field_output(
            "U",
            "FABRIC_STRAIN",
            "FABRIC_N",
            "FABRIC_WARP",
            "FABRIC_WEFT",
            "SENER",
            configuration="reference",
        ),
    )

    step = model.step(
        target=displacement,
        material=material,
        increments=2,
        output=output,
        progress=False,
    )
    simulation = step.solve_result()

    manifest = step.summary()["result_field_manifest"]
    assert tuple(item["layer_name"] for item in manifest) == ("ply 0", "ply +45")
    assert manifest[0]["physical_layer_ids"] == ["ply 0 bottom", "ply 0 top"]
    assert manifest[0]["multiplicity"] == 2
    assert tuple(item["field_prefix"] for item in manifest) == (
        "FABRIC_LAYER_000_PLY_0",
        "FABRIC_LAYER_001_PLY_45",
    )
    expected = {
        "SENER",
        "FABRIC_LAYER_000_PLY_0__FABRIC_GENERALIZED_STRAIN",
        "FABRIC_LAYER_000_PLY_0__FABRIC_GENERALIZED_RESULTANT",
        "FABRIC_LAYER_001_PLY_45__FABRIC_GENERALIZED_STRAIN",
        "FABRIC_LAYER_001_PLY_45__FABRIC_GENERALIZED_RESULTANT",
        "FABRIC_LAYER_000_PLY_0__SENER",
        "FABRIC_LAYER_001_PLY_45__SENER",
    }
    assert expected.issubset(simulation.fields)
    assert simulation.metadata["problem"]["result_field_manifest"] == manifest
    assert simulation.artifacts["field_history"].is_file()


def test_membrane_provider_refuses_to_hide_shell_bending():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(study=studies.static_membrane(), mesh=domain)
    displacement = model.field(fields.displacement(domain))
    material = model.material(_fabric_membrane(bending=1.0))

    with pytest.raises(NotImplementedError, match="cannot consume bending stiffness"):
        model.step(target=displacement, material=material, progress=False)

    stacked_model = models.create(study=studies.static_membrane(), mesh=domain)
    stacked_displacement = stacked_model.field(fields.displacement(domain))
    stack = constitutive.fabric_stack(
        [
            constitutive.fabric_layer(
                _fabric_membrane(),
                name="membrane_layer",
            ),
            constitutive.fabric_layer(
                _fabric_membrane(bending=1.0),
                name="bending_layer",
            ),
        ]
    )
    stacked_material = stacked_model.material(stack)
    with pytest.raises(NotImplementedError, match="cannot consume bending stiffness"):
        stacked_model.step(
            target=stacked_displacement,
            material=stacked_material,
            progress=False,
        )


def test_fabric_capability_declares_membrane_without_claiming_a_shell():
    capability = constitutive.capability("fabric_surface")

    assert capability.maturity == "experimental_fem_integrated"
    assert "in-plane membrane" in capability.available_scope
    assert "multilayer stack" in capability.available_scope
    assert "four-channel" in capability.available_scope
    assert any("not yet a shell" in item for item in capability.limitations)
