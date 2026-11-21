# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from mpi4py import MPI
from dolfinx import mesh as dm
import numpy as np
import pytest
from agentfem import models, studies, fields, constitutive, operators, amplitudes
from agentfem.elements.tet10_mass import hrz_tet10_mass


def setup_model(degree=1, comm=MPI.COMM_SELF):
    mesh = dm.create_unit_cube(comm, 1, 1, 1)
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=mesh
    )
    u = m.field(fields.displacement(mesh, degree=degree))
    m.material(constitutive.isotropic_elastic(young=100, poisson=0.25, density=2))
    return m, u


def test_hrz_mass_and_rigid_translation():
    import basix

    el = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.tetrahedron,
        2,
        basix.LagrangeVariant.equispaced,
    )
    x = el.points @ np.array([[1, 0.2, 0], [0, 2, 0.1], [0, 0, 3.0]])
    mass = hrz_tet10_mass(x, 2)
    assert np.all(mass > 0)
    np.testing.assert_allclose(mass.sum(), 2, rtol=1e-13)
    np.testing.assert_allclose(mass @ x / mass.sum(), x[:4].mean(0), atol=1e-14)
    x[4] += 0.1
    with pytest.raises(NotImplementedError):
        hrz_tet10_mass(x, 2)


def test_global_hrz():
    m, u = setup_model(2, MPI.COMM_WORLD)
    mass = m.lumped_mass(u, method="tet10_hrz")
    n = u.function_space.dofmap.index_map.size_local * 3
    assert np.all(mass.mass > 0)
    total = MPI.COMM_WORLD.allreduce(mass.mass[:n].sum()) / 3
    assert total == pytest.approx(2, rel=1e-13)
    with pytest.raises(ValueError):
        m.lumped_mass(u)


def test_nodal_force_unique_owner_and_amplitude():
    m, u = setup_model(1, MPI.COMM_WORLD)
    force = m.nodal_force((1, 2, 3), at=(1, 1, 1), amplitude=amplitudes.constant(2))
    op = force.bind(u)
    v = op.assemble_vector()
    bs = u.function_space.dofmap.index_map_bs
    total = MPI.COMM_WORLD.allreduce(v.array.reshape(-1, bs).sum(axis=0))
    np.testing.assert_allclose(total, [2, 4, 6], atol=1e-14)
    v.destroy()
    with pytest.raises(ValueError):
        m.nodal_force((1, 0, 0), at=(2, 2, 2)).bind(u)


def test_public_explicit_nodal_workflow():
    m, u = setup_model(2)
    m.nodal_force((1, 0, 0), at=(1, 1, 1))
    mass = m.lumped_mass(u, method="tet10_hrz")
    step = m.step(target=u, mass=mass, dt=1e-4, steps=5, progress=False)
    r = step.solve_result()
    assert np.max(step.state.u.x.array) > 0
    assert np.all(np.isfinite(step.state.u.x.array))


def test_dynamic_history_and_restart(tmp_path):
    from agentfem import results

    def build():
        m, u = setup_model()
        m.nodal_force((1, 0, 0), at=(1, 1, 1))
        s = m.step(target=u, dt=1e-4, steps=8, progress=False)
        return s, results.dynamic_history("tip", at=(1, 1, 1))

    s, h = build()
    full = s.solve_result(history=h)
    a, h = build()
    a.run(history=h, until_step=4)
    a.save_checkpoint(tmp_path / "checkpoint")
    b, h = build()
    b.load_checkpoint(tmp_path / "checkpoint")
    r = b.solve_result(history=h)

    for key in ("u", "v", "a"):
        np.testing.assert_allclose(
            getattr(b.state, key).x.array, getattr(s.state, key).x.array, atol=1e-15
        )
    for key in ("external_work", "energy_balance_error"):
        np.testing.assert_allclose(
            [row[key] for row in b.history_records],
            [row[key] for row in s.history_records],
            atol=1e-15,
        )
    np.testing.assert_allclose(
        [x["tip.VMAG"] for x in b.history_records],
        [x["tip.VMAG"] for x in s.history_records],
        atol=1e-15,
    )
    for row in b.history_records:
        assert row["tip.VMAG"] == pytest.approx(
            np.linalg.norm([row[f"tip.V{i}"] for i in (1, 2, 3)])
        )
    results.write_dynamic_history(r, tmp_path / "history.csv")
    header = (tmp_path / "history.csv").read_text().splitlines()[0]
    assert "artificial_energy" in header and "external_force_1" in header


def test_auto_step_and_work_conservation():
    m, u = setup_model(2)
    m.nodal_force((1, 0, 0), at=(1, 1, 1))
    s = m.step(
        target=u,
        mass=m.lumped_mass(u, method="tet10_hrz"),
        dt="auto",
        steps=4,
        progress=False,
    )
    assert s.dt == s.stability.selected
    s.solve_result()
    assert s.history_records[-1]["external_work"] > 0
    assert s.history_records[-1]["external_force_1"] == pytest.approx(1)
    assert s.history_records[-1]["artificial_energy"] == 0
    # Constant nodal work equals F dot the displacement increment exactly.
    from agentfem import results

    displacement = results.probe(s.state.u, at=(1, 1, 1))
    assert s.history_records[-1]["external_work"] == pytest.approx(
        displacement[0], rel=1e-11
    )


def test_zero_thickness_tet10_seam():
    from agentfem.mesh.tet_seam import split_tetrahedral_seam

    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], float)
    xyz = corners.tolist()
    edgeids = {}
    cells = []
    for row in [(0, 1, 2, 3), (0, 2, 1, 4)]:
        nodes = list(row)
        for a, b in ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)):
            edge = tuple(sorted((row[a], row[b])))
            if edge not in edgeids:
                edgeids[edge] = len(xyz)
                xyz.append(((corners[edge[0]] + corners[edge[1]]) / 2).tolist())
            nodes.append(edgeids[edge])
        cells.append(nodes)
    seam = split_tetrahedral_seam(
        xyz, np.array(cells), positive_cells=[0], faces=[(0, 1, 2)]
    )
    assert len(seam.duplicate_of) == 6
    assert not set(seam.cells[0]).intersection(seam.cells[1])
    for new, old in seam.duplicate_of.items():
        np.testing.assert_array_equal(
            seam.coordinates[new - 1], seam.coordinates[old - 1]
        )
    with pytest.raises(ValueError):
        split_tetrahedral_seam(
            seam.coordinates, seam.cells, positive_cells=[0], faces=[(0, 1, 2)]
        )


def test_dynamic_comparison_raw_and_time_rules(tmp_path):
    from agentfem import results

    t = np.arange(100) * 0.01
    v = np.column_stack((np.sin(2 * np.pi * 5 * t), np.zeros((100, 2))))
    a = results.VelocityTrace(t, v, "s", "m/s", {})
    report = results.compare_dynamic_traces(a, a, arrival_threshold=0.1)
    assert report["metrics"]["normalized_vector_L2"] == 0
    assert report["candidate"]["spectrum"]["dominant_frequency"] == pytest.approx(5)
    assert np.all(np.diff(a.cumulative_max) >= 0)
    shifted = results.VelocityTrace(t + 0.001, v, "s", "m/s", {})
    with pytest.raises(ValueError):
        results.compare_dynamic_traces(a, shifted, arrival_threshold=0.1)
    results.write_dynamic_comparison(a, a, tmp_path, arrival_threshold=0.1)


def test_dynamic_inp_inventory(tmp_path):
    from agentfem.mesh.abaqus import read_dynamic_input

    path = tmp_path / "case.inp"
    path.write_text(
        "*NODE\n"
        + "".join(f"{i + 1}, {i}, 0, 0\n" for i in range(10))
        + "*ELEMENT, TYPE=C3D10M, ELSET=BODY\n1,1,2,3,4,5,6,7,8,9,10\n*NSET,NSET=SENSOR\n1\n*AMPLITUDE,NAME=PULSE\n0,0,1,1,2,0\n*CLOAD,AMPLITUDE=PULSE\nSENSOR,2,-1230\n*OUTPUT,HISTORY\n*NODE OUTPUT,NSET=SENSOR\nU,V,A\n"
    )
    deck = read_dynamic_input(path, unit_system="mm-tonne-s")
    assert deck.summary()["formulations"][0]["abaqus_formula_equivalent"] is False
    assert any(
        x.keyword == "*CLOAD" and x.data == ("SENSOR,2,-1230",) for x in deck.records
    )
    assert len(deck.nodes.labels) == 10
    path.write_text("*PART,NAME=BODY\n" + path.read_text())
    with pytest.raises(NotImplementedError):
        read_dynamic_input(path, unit_system="mm-tonne-s")


def test_label_identity_keeps_coincident_p2_nodes_independent():
    import basix
    import basix.ufl
    import ufl
    from agentfem import mesh as mesh_api
    from agentfem.mesh.abaqus import AbaqusNodeTable

    element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.tetrahedron,
        2,
        basix.LagrangeVariant.equispaced,
    )
    xyz = np.vstack((element.points, element.points + [0, 0, 0]))
    connectivity = np.arange(20, dtype=np.int64).reshape(2, 10)
    coordinate_element = basix.ufl.element("Lagrange", "tetrahedron", 2, shape=(3,))
    domain = dm.create_mesh(
        MPI.COMM_SELF, connectivity, ufl.Mesh(coordinate_element), xyz
    )
    model = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=domain
    )
    u = model.field(fields.displacement(domain, degree=2))
    model.material(constitutive.isotropic_elastic(young=100, poisson=0.25, density=2))
    source = AbaqusNodeTable(np.arange(101, 121), xyz)
    a = mesh_api.nodal_selection(labels=[101], source=source)
    b = mesh_api.nodal_selection(labels=[111], source=source)
    va, ia = a.bind(u)
    vb, ib = b.bind(u)
    assert ia[0] != ib[0]
    np.testing.assert_allclose(
        va.tabulate_dof_coordinates()[ia[0]], vb.tabulate_dof_coordinates()[ib[0]]
    )
    with pytest.raises(ValueError, match="global owner"):
        mesh_api.nodal_selection(at=(0, 0, 0)).bind(u)
    op = model.nodal_force((2, 0, 0), nodes=a).bind(u)
    v = op.assemble_vector()
    assert v.array[3 * ia[0]] == 2
    assert v.array[3 * ib[0]] == 0
    v.destroy()
    step = model.step(
        target=u,
        mass=model.lumped_mass(u, method="tet10_hrz"),
        dt=1e-4,
        steps=4,
        progress=False,
    )
    from agentfem import results

    histories = results.dynamic_history("unloaded", nodes=b)
    step.solve_result(history=histories)
    assert all(r["unloaded.UMAG"] == 0 for r in step.history_records)


def test_tabular_force_work_and_sparse_history(tmp_path):
    from agentfem import results

    def solve(every):
        m, u = setup_model()
        m.nodal_force(
            (3, 0, 0),
            at=(1, 1, 1),
            amplitude=amplitudes.tabular([0, 0.0004, 0.0008], [0, 1, 0]),
        )
        s = m.step(
            target=u,
            dt=1e-4,
            steps=8,
            history_every=every,
            save_every=4,
            progress=False,
        )
        h = results.dynamic_history(
            "tip", at=(1, 1, 1), length_unit="mm", time_unit="ms"
        )
        r = s.solve_result(history=h)
        return s, r

    full, result = solve(1)
    sparse, _ = solve(4)
    displacement = np.array([r["tip.U1"] for r in full.history_records])
    force = np.array(
        [
            3 * np.interp(r["time"], [0, 0.0004, 0.0008], [0, 1, 0])
            for r in full.history_records
        ]
    )
    independent = np.sum(0.5 * (force[:-1] + force[1:]) * np.diff(displacement))
    assert full.history_records[-1]["external_work"] == pytest.approx(
        independent, rel=1e-12
    )
    assert sparse.history_records[-1]["external_work"] == pytest.approx(
        independent, rel=1e-12
    )
    assert len(sparse.history_records) == 3
    assert result.histories["tip.U1"].abscissa_unit == "ms"
    assert (
        len(result.metadata["transient"]["sampling_evidence"]["tip.U1"]["weights"]) == 4
    )
    paths = results.write_dynamic_history(result, tmp_path / "sample.csv")
    import json

    assert (
        json.loads(paths[1].read_text())["sampling"]["tip.U1"]["method"]
        == "finite_element_interpolation"
    )


def test_source_labels_reject_fractional_values():
    from agentfem import mesh

    with pytest.raises(ValueError, match="integers"):
        mesh.nodal_selection(labels=[1.2])


def test_enriched_tet15_moments_and_public_dynamics():
    import basix
    from agentfem import elements, results
    from agentfem.elements.tet15_mass import scalar_element, nodal_weights

    element = scalar_element()
    points = element.points
    weights = nodal_weights()
    # Exact quadratic moments imply exact rigid-rotation inertia.
    q, w = basix.make_quadrature(basix.CellType.tetrahedron, 6)
    for i in range(3):
        for j in range(3):
            assert weights @ (points[:, i] * points[:, j]) == pytest.approx(
                w @ (q[:, i] * q[:, j]), abs=1e-14
            )
    np.testing.assert_allclose(
        element.tabulate(0, points)[0, :, :, 0], np.eye(15), atol=1e-13
    )
    assert np.all(weights > 0)
    domain = dm.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=domain
    )
    u = m.field(fields.displacement(domain, element=elements.tet15_mass_lumped()))
    m.material(constitutive.isotropic_elastic(young=100, poisson=0.25, density=2))
    mass = m.lumped_mass(u, method="tet15_nodal")
    n = u.function_space.dofmap.index_map.size_local * 3
    assert MPI.COMM_WORLD.allreduce(mass.mass[:n].sum()) / 3 == pytest.approx(
        2, rel=1e-12
    )
    m.nodal_force((1, 0, 0), at=(1, 1, 1))
    step = m.step(target=u, mass=mass, dt="auto", steps=5, progress=False)
    r = step.solve_result(history=results.dynamic_history("tip", at=(1, 1, 1)))
    assert np.isfinite(r.histories["tip.VMAG"].values).all()
    assert r.metadata["step"]["inertia_scheme"] == "tet15_nodal"


def test_partial_seam_requires_explicit_shared_crack_front():
    from agentfem.mesh.tet_seam import split_tetrahedral_seam

    xyz = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1], [0.5, 0.5, -1]]
    )
    cells = np.array([[0, 1, 2, 4], [1, 3, 2, 4], [0, 2, 1, 5], [1, 2, 3, 5]])
    with pytest.raises(ValueError, match="incomplete interface"):
        split_tetrahedral_seam(xyz, cells, positive_cells=[0, 1], faces=[(0, 1, 2)])
    seam = split_tetrahedral_seam(
        xyz, cells, positive_cells=[0, 1], faces=[(0, 1, 2)], front_nodes=[1, 2]
    )
    assert len(seam.duplicate_of) == 1
    assert seam.cells[0, 0] != seam.cells[2, 0]
    # The uncracked adjacent face remains connected through all three nodes.
    assert set(seam.cells[1]).intersection(seam.cells[3]) == {1, 2, 3}


def test_half_sine_support_and_peak():
    pulse = amplitudes.half_sine(25e-6, amplitude=1230)
    assert pulse(0) == pulse(25e-6) == pulse(50e-6) == pulse(-1) == 0
    assert pulse(12.5e-6) == pytest.approx(1230)
    with pytest.raises(ValueError):
        amplitudes.half_sine(0)


def test_tet15_output_preserves_native_dofs_and_cell_fields(tmp_path):
    from dolfinx import fem
    import h5py
    from agentfem import elements, results

    domain = dm.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    u = fields.displacement(domain, element=elements.tet15_mass_lumped()).value
    u.interpolate(lambda x: np.vstack((x[0] ** 2, np.zeros((2, x.shape[1])))))
    V0 = fem.functionspace(domain, ("DG", 0))
    s = fem.Function(V0, name="cell_marker")
    s.x.array[:] = np.arange(len(s.x.array)) + 1
    path = tmp_path / "enriched.xdmf"
    with results.UnifiedXDMFTimeSeries(path) as writer:
        writer.write_fields(0, u, s)
    with h5py.File(path.with_suffix(".h5")) as f:
        assert f["Mesh/SourceCellDofs"].shape == (6, 15)
        assert f["Mesh/Topology"].shape == (6 * 24, 4)
        assert f["Frames/0000/Point/U"].shape[0] == len(u.x.array) // 3
        parent = f["Mesh/VisualizationParentCell"][:]
        expected = s.x.array[np.asarray(V0.dofmap.list)[:, 0]][parent]
        np.testing.assert_array_equal(f["Frames/0000/Cell/cell_marker"][:], expected)
        np.testing.assert_allclose(
            f["Frames/0000/Point/U"][:], u.x.array.reshape(-1, 3)
        )


def test_history_nodeset_expands_named_stations():
    from agentfem import mesh, results

    selection = mesh.nodal_selection(at=[(0, 0, 0), (1, 1, 1)])
    requests = results.dynamic_history("corners", nodes=selection, variables=("V",))
    assert len(requests) == 10
    assert requests[0].name == "corners[P0].V1"
    assert requests[5].name == "corners[P1].V1"
    model, u = setup_model()
    model.nodal_force((1, 0, 0), at=(1, 1, 1))
    step = model.step(target=u, dt=1e-4, steps=2, progress=False)
    result = step.solve_result(history=requests)
    assert result.histories["corners[P1].VMAG"].values[-1] > 0


def test_partial_tet10_seam_keeps_mid_edge_front_connected():
    from agentfem.mesh.tet_seam import split_tetrahedral_seam

    xyz = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1], [0.5, 0.5, -1]]
    corners = [[0, 1, 2, 4], [1, 3, 2, 4], [0, 2, 1, 5], [1, 2, 3, 5]]
    edges = {}
    cells = []
    for cell in corners:
        row = list(cell)
        for a, b in ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)):
            key = tuple(sorted((cell[a], cell[b])))
            if key not in edges:
                edges[key] = len(xyz)
                xyz.append(((np.asarray(xyz[key[0]]) + xyz[key[1]]) / 2).tolist())
            row.append(edges[key])
        cells.append(row)
    # Both crack-front vertices and the midside node remain shared.
    front = [1, 2, edges[(1, 2)]]
    seam = split_tetrahedral_seam(
        xyz,
        np.array(cells),
        positive_cells=[0, 1],
        faces=[(0, 1, 2)],
        front_nodes=front,
    )
    assert len(seam.duplicate_of) == 3
    assert set(seam.cells[0]).intersection(seam.cells[2]) == set(front)
    expected = {1, 2, 3, edges[(1, 2)], edges[(1, 3)], edges[(2, 3)]}
    assert set(seam.cells[1]).intersection(seam.cells[3]) == expected
    for new, old in seam.duplicate_of.items():
        np.testing.assert_array_equal(
            seam.coordinates[new - 1], seam.coordinates[old - 1]
        )


def test_nodal_energy_route_rejects_unaccounted_support_work():
    from agentfem import constraints

    m, u = setup_model()
    m.constraint(
        constraints.DirichletConstraint.component(
            u.space, 0, lambda x: np.isclose(x[0], 0), value=0.01
        )
    )
    m.nodal_force((1, 0, 0), at=(1, 1, 1))
    with pytest.raises(NotImplementedError, match="homogeneous stationary supports"):
        m.step(target=u, dt=1e-4, steps=2, progress=False)


def test_enriched_free_body_linear_and_angular_impulse():
    from agentfem import elements

    domain = dm.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=domain
    )
    u = m.field(fields.displacement(domain, element=elements.tet15_mass_lumped()))
    m.material(constitutive.isotropic_elastic(young=100, poisson=0.25, density=2))
    f = np.array([1.0, 2.0, 3.0])
    point = np.array([1.0, 1.0, 1.0])
    pulse = amplitudes.half_sine(0.001)
    m.nodal_force(f, at=point, amplitude=pulse)
    mass = m.lumped_mass(u, method="tet15_nodal")
    s = m.step(target=u, mass=mass, dt=0.0001, steps=10, progress=False)
    s.solve_result()
    n = u.space.dofmap.index_map.size_local
    nodal_momentum = (mass.mass[: 3 * n] * s.state.v.x.array[: 3 * n]).reshape(n, 3)
    momentum = MPI.COMM_WORLD.allreduce(nodal_momentum.sum(axis=0))
    coords = u.space.tabulate_dof_coordinates()[:n]
    angular = MPI.COMM_WORLD.allreduce(np.cross(coords, nodal_momentum).sum(axis=0))
    scales = np.array([pulse(i * s.dt) for i in range(s.steps + 1)])
    impulse = s.dt * (scales[1:-1].sum() + 0.5 * (scales[0] + scales[-1])) * f
    np.testing.assert_allclose(momentum, impulse, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(
        angular, np.cross(point, impulse), rtol=1e-11, atol=1e-14
    )


def test_nodal_amplitude_change_rejected_on_restart(tmp_path):
    def build(scale):
        model, u = setup_model()
        model.nodal_force((1, 0, 0), at=(1, 1, 1), amplitude=amplitudes.constant(scale))
        return model.step(target=u, dt=1e-4, steps=4, progress=False)

    original = build(1)
    original.run(until_step=2)
    original.save_checkpoint(tmp_path / "checkpoint")
    changed = build(2)
    with pytest.raises(ValueError):
        changed.load_checkpoint(tmp_path / "checkpoint")
