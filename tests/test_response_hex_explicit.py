from copy import deepcopy

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import problems, state, time
from agentfem.constitutive._explicit_material_driver import ExplicitQuadratureResponse
from agentfem.diagnostics import MechanicalEnergyMonitor
from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual
from agentfem.mechanics._response_hex_explicit import (
    ResponseHexExplicitResidual,
    ResponseHexWorkMonitor,
)
from test_explicit_material_quadrature import ResponseOnly


def test_unaccepted_response_cannot_be_restored():
    step = make_step()
    record = step.residual.snapshot()
    record["fields"]["P"][0] = 1.0
    before = step.residual.snapshot()
    with pytest.raises(ValueError, match="unaccepted"):
        step.residual.restore(record)
    assert step.residual.snapshot() == before


def make_step(dt=1e-4, *, contact=False, bond=False):
    if bond:
        from test_nonmatching_global import _blocks

        domain = _blocks(1, 2, cell_type="hexahedron")
    else:
        domain = mesh.create_unit_cube(
            MPI.COMM_SELF, 2, 1, 1, cell_type=mesh.CellType.hexahedron
        )
    u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
    history = state.second_order_state(u)
    law = ResponseOnly()
    response = ExplicitQuadratureResponse.create(domain, law.state_schema)
    internal = FiniteUniformHexResidual(
        history.u.value,
        response,
        density=1.0,
        hourglass_modulus=1.0,
        hourglass_scale=0.1,
    )
    cohesive = None
    if bond:
        from test_quadrilateral_global import _trace
        from agentfem import fracture, interfaces
        from agentfem._interface_quadrilateral import BilinearReferenceTrace
        from agentfem._initial_nodal_bond import initial_nodal_q1_pairing

        a, na = _trace(u.function_space, False)
        b, nb = _trace(u.function_space, True)
        a = BilinearReferenceTrace(a.vertices, a.quadrilaterals, tolerance=1e-10)
        b = BilinearReferenceTrace(b.vertices, b.quadrilaterals, tolerance=1e-10)
        pair = initial_nodal_q1_pairing(
            a, b, eligible_nodes=np.ones(len(a.vertices), dtype=bool), tolerance=1e-10
        )
        cohesive = fracture.nonmatching_cohesive_force(
            pair,
            u,
            interfaces.elastic_cohesive(normal_stiffness=100, tangential_stiffness=100),
            negative_dofs=na,
            positive_dofs=nb,
        )
    residual = ResponseHexExplicitResidual(
        internal,
        law,
        time_increment_ceiling=1e-3,
        stress_free_initial_state=True,
        material_identity={"test": "linear-response", "modulus": 20.0},
        cohesive=cohesive,
    )
    xyz = u.function_space.tabulate_dof_coordinates()
    history.v.value.x.array[:] = (0.001 * xyz).ravel()
    if bond:
        for nodes in internal.cell_nodes:
            history.v.value.x.array.reshape(-1, 3)[nodes, 2] += 0.01 * np.sign(
                xyz[nodes, 2].mean()
            )
    monitor = MechanicalEnergyMonitor(mass=internal.mass_diagonal)
    if contact:
        from agentfem import boundary_models, fracture, mesh as af_mesh

        body_monitor = ResponseHexWorkMonitor(residual)

        for side in (0, 1):
            facets = mesh.locate_entities_boundary(
                domain, 2, lambda x, side=side: np.isclose(x[0], side)
            )
            tags = mesh.meshtags(
                domain, 2, facets, np.full(len(facets), 1, dtype=np.int32)
            )
            region = af_mesh.tagged_boundary_region(
                domain, tags, tag=1, name=f"side_{side}"
            )
            adapter = boundary_models.dolfinx_boundary_region_contact_trace(
                region, u.function_space
            )
            direction = 1.0 if side == 0 else -1.0
            schedule = boundary_models.prescribed_rigid_motion_schedule(
                boundary_models.prescribed_rigid_motion(
                    translation=(direction * 1e-4, 0, 0), name=f"motion_{side}"
                ),
                end_time=0.001,
                name=f"schedule_{side}",
            )
            projector = boundary_models.rigid_plane(
                point=(side, 0, 0), normal=(direction, 0, 0), name=f"plane_{side}"
            )
            if contact == "cylinder":
                projector = boundary_models.rigid_cylinder(
                    axis_point=(side - direction * 0.8, 0, 0.5),
                    axis_direction=(0, 1, 0),
                    radius=1,
                    axial_interval=(-1, 2),
                    name=f"roller_{side}",
                )
            residual = boundary_models.dolfinx_explicit_contact_residual(
                residual,
                adapter=adapter,
                displacement=u,
                projector=projector,
                penalty=100,
                maximum_stable_time_increment=1e-3,
                motion_schedule=schedule,
                name=f"contact_{side}",
            )
        monitor = fracture.DynamicEnergyLedger(
            energy=body_monitor, state=history, mass=internal, residual=residual
        )
    step = problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=internal),
        residual=residual,
        history_monitor=monitor,
        dt=dt,
        steps=10,
        progress=False,
    )
    return step


@pytest.mark.parametrize(
    "contact,bond", [(True, False), ("cylinder", False), ("cylinder", True)]
)
def test_response_only_nested_moving_contact_and_restart(tmp_path, contact, bond):
    reference = make_step(contact=contact, bond=bond)
    reference.run()
    partial = make_step(contact=contact, bond=bond)
    partial.run(until_step=4)
    path = partial.save_checkpoint(tmp_path / "contact_response")
    resumed = make_step(contact=contact, bond=bond)
    resumed.load_checkpoint(path)
    resumed.run()
    for name in ("u", "v", "a"):
        np.testing.assert_array_equal(
            getattr(resumed.state, name).value.x.array,
            getattr(reference.state, name).value.x.array,
        )
    assert resumed.residual.snapshot() == reference.residual.snapshot()
    evidence = reference.residual.contact_energy_evidence()
    assert len(evidence) == 2
    assert all(item["contact_potential_energy"] > 0 for item in evidence)
    assert all(item["contact_motion_work"] != 0 for item in evidence)
    assert resumed.history_records == reference.history_records
    last = reference.history_records[-1]
    assert last["material_internal_work"] > 0
    assert last["contact_motion_work"] == pytest.approx(
        sum(item["contact_motion_work"] for item in evidence)
    )
    assert "stored_energy" not in last
    if bond:
        assert last["interface_stored_energy"] > 0


@pytest.mark.parametrize("bond", [False, True])
def test_nested_contact_late_failure_is_atomic(monkeypatch, bond):
    step = make_step(contact=True, bond=bond)
    step.run(until_step=2)
    accepted = step.residual.snapshot()
    nodal = step.state.snapshot()
    work = step.residual.work_state
    original = work.commit

    def fail_after_commit():
        original()
        raise RuntimeError("outer contact failed after commit")

    monkeypatch.setattr(work, "commit", fail_after_commit)
    with pytest.raises(RuntimeError, match="outer contact"):
        step.run()
    assert step.residual.snapshot() == accepted
    for name, value in nodal["fields"].items():
        np.testing.assert_array_equal(value, step.state.snapshot()["fields"][name])


def test_bond_failure_clears_both_trials(monkeypatch):
    step = make_step(bond=True)
    residual = step.residual
    before = residual.snapshot()
    original = residual.cohesive.add_to_vector

    def broken(vector):
        response = original(vector)
        vector.array[0] = np.nan
        return response

    monkeypatch.setattr(residual.cohesive, "add_to_vector", broken)
    with pytest.raises(ValueError, match="Nonfinite"):
        step.run()
    assert residual.snapshot() == before
    assert residual.cohesive.assembler._trial is None


def test_accepted_force_does_not_update_material():
    step = make_step()
    step.run(until_step=2)
    record = step.residual.snapshot()
    vector = step.residual.assemble_accepted_vector()
    try:
        np.testing.assert_array_equal(vector.array, step.residual.accepted_force)
    finally:
        vector.destroy()
    assert step.residual.snapshot() == record
    step.state.u.value.x.array[0] += 1e-5
    with pytest.raises(RuntimeError, match="configuration"):
        step.residual.assemble_accepted_vector()


def test_contact_vertex_lookup_is_cell_local(monkeypatch):
    from agentfem.boundary_models import dolfinx_contact_trace as implementation

    original = implementation._matching_dof_block
    sizes = []

    def checked(point, coordinates, **options):
        sizes.append(len(coordinates))
        return original(point, coordinates, **options)

    monkeypatch.setattr(implementation, "_matching_dof_block", checked)
    make_step(contact=True, bond=True)
    assert sizes and max(sizes) == 8


def test_response_only_advancement_and_disk_restart(tmp_path):
    reference = make_step()
    reference.run()
    partial = make_step()
    partial.run(until_step=4)
    path = partial.save_checkpoint(tmp_path / "response")
    resumed = make_step()
    resumed.load_checkpoint(path)
    resumed.run()
    for name in ("u", "v", "a"):
        np.testing.assert_array_equal(
            getattr(resumed.state, name).value.x.array,
            getattr(reference.state, name).value.x.array,
        )
    assert resumed.residual.snapshot() == reference.residual.snapshot()
    assert reference.residual.internal_work > 0
    assert reference.residual.accepted_time == pytest.approx(0.001)


def test_material_failure_rolls_back_complete_step():
    step = make_step()
    step.residual.material.reduce = True
    before = step.state.snapshot()
    material_before = step.residual.snapshot()
    with pytest.raises(ValueError, match="smaller increment"):
        step.run()
    assert step.completed_steps == 0
    assert step.residual.snapshot() == material_before
    for k, v in before["fields"].items():
        np.testing.assert_array_equal(v, step.state.snapshot()["fields"][k])


def test_corrupt_restore_atomic_and_ceiling_guard():
    step = make_step()
    step.run(until_step=2)
    before = step.residual.snapshot()
    bad = deepcopy(before)
    bad["gradient"][0][0][0] = -1
    with pytest.raises(ValueError, match="gradient"):
        step.residual.restore(bad)
    assert step.residual.snapshot() == before
    with pytest.raises(ValueError, match="ceiling"):
        step.residual.validate_time_increment(0.1)


def test_late_commit_failure_restores_accepted_material_and_nodal_state(monkeypatch):
    step = make_step()
    step.run(until_step=2)
    nodal = step.state.snapshot()
    accepted = step.residual.snapshot()
    original = step.residual.internal.response.commit

    def fail_after_commit():
        original()
        raise RuntimeError("late commit failure")

    monkeypatch.setattr(step.residual.internal.response, "commit", fail_after_commit)
    with pytest.raises(RuntimeError, match="late commit"):
        step.run()
    assert step.residual.snapshot() == accepted
    for k, v in nodal["fields"].items():
        np.testing.assert_array_equal(v, step.state.snapshot()["fields"][k])


def test_rigid_translation_has_zero_work():
    step = make_step()
    step.state.v.value.x.array.reshape(-1, 3)[:] = [0.1, -0.2, 0.3]
    step.run()
    assert abs(step.residual.internal_work) < 1e-20
    np.testing.assert_allclose(
        step.state.u.value.x.array.reshape(-1, 3),
        np.tile([0.0001, -0.0002, 0.0003], (12, 1)),
        atol=1e-14,
    )
