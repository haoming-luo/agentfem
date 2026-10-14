# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from agentfem import boundary_models, interfaces
from test_nonmatching_global import _blocks
from test_quadrilateral_global import _trace
from test_response_material_step import problem


def make_step(**options):
    domain = _blocks(1, 2, cell_type="hexahedron")
    cells = domain.geometry.dofmaps[0]
    lower = np.unique(cells[domain.geometry.x[cells, 2].mean(axis=1) < 0])
    domain.geometry.x[lower, :2] = .25 + .5 * domain.geometry.x[lower, :2]
    model, u, policy = problem(domain)
    a, na = _trace(u.value.function_space, False)
    b, nb = _trace(u.value.function_space, True)
    a = interfaces.reference_trace(a.vertices, a.quadrilaterals, topology="quadrilateral",
                                    geometry="bilinear", tolerance=1e-10)
    b = interfaces.reference_trace(b.vertices, b.quadrilaterals, topology="quadrilateral",
                                    geometry="bilinear", tolerance=1e-10)
    model.add_boundary_model(boundary_models.deformable_contact_pair(a, b,
        secondary_dofs=na, primary_dofs=nb, candidate_nodes=np.ones(len(na), dtype=bool),
        penalty=100., search_radius=.5))
    step = model.step(target=u, element_policy=policy, dt=1e-4, steps=4,
                      time_increment_ceiling=1e-3, stress_free_initial_state=True, progress=False, **options)
    xyz = u.value.function_space.tabulate_dof_coordinates()
    for nodes in step.material_residual.internal.cell_nodes:
        step.state.v.value.x.array.reshape(-1, 3)[nodes, 2] = -.1 * np.sign(xyz[nodes, 2].mean())
    return step


def test_deformable_model_asset_work_and_restart(tmp_path):
    complete = make_step()
    complete.run()
    assert complete.residual.operator is not None
    assert complete.residual.accepted is not None
    partial = make_step()
    partial.run(until_step=2)
    path = partial.save_checkpoint(tmp_path / "deformable")
    resumed = make_step()
    resumed.load_checkpoint(path)
    resumed.run()
    np.testing.assert_array_equal(resumed.state.u.value.x.array, complete.state.u.value.x.array)
    assert resumed.residual.snapshot() == complete.residual.snapshot()
