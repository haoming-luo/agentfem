# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Public workflow regressions; no application or blade simulation."""

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dm
from agentfem import fields, models, studies, constitutive, elements, results


def build(comm=MPI.COMM_SELF, steps=5, mean=False):
    mesh = dm.create_unit_cube(comm, 1, 1, 1)
    model = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=mesh
    )
    u = model.field(fields.displacement(mesh, degree=2))
    model.material(constitutive.isotropic_elastic(young=100, poisson=0.25, density=2))
    model.nodal_force((1, 0.2, -0.1), at=(1, 1, 1))
    step = model.step(
        target=u,
        element_policy=elements.composite_tet10_policy(mean_dilatation=mean),
        dt=1e-3,
        steps=steps,
        progress=False,
        save_every=1,
    )
    return step


def test_public_step_output_history(tmp_path):
    import h5py
    import xml.etree.ElementTree as ET

    s = build(mean=True)
    h = results.dynamic_history("station", at=(0.8, 0.8, 0.8))
    result = s.solve_result(
        output=tmp_path / "fields.xdmf",
        field_variables=("U", "S", "E", "MISES", "SENER"),
        history=h,
    )
    assert s.completed_steps == 5
    assert result.metadata["field_output"]["warp_field"] == "U"
    assert (
        result.fields["Displacement"].processing["interpolation"]
        == "piecewise_linear_12_subtets"
    )
    assert np.isfinite(s.state.u.x.array).all()
    assert s.history_records[-1]["external_work"] > 0
    assert s.summary()["element_policy"]["abaqus_c3d10m_equivalent"] is False
    tree = ET.parse(tmp_path / "fields.xdmf")
    assert len(tree.findall('.//Grid[@GridType="Uniform"]')) == 6
    with h5py.File(tmp_path / "fields.h5") as f:
        assert f["Mesh/Topology"].shape == (72, 4)
        assert f["Mesh/SourceCellDofs"].shape == (6, 10)
        np.testing.assert_array_equal(np.sort(f["Mesh/SourceCellIds"][:]), np.arange(6))
        pts = f["Mesh/ReferenceGeometry"][:][f["Mesh/Topology"][:]]
        assert np.all(np.linalg.det((pts[:, 1:] - pts[:, :1]).transpose(0, 2, 1)) > 0)
        assert f["Frames/0005/Node/U"].shape == (33, 3)
        assert f["Frames/0005/Native/U"].shape == (27, 3)
        assert f["Frames/0005/Cell/S"].shape == (72, 9)
        assert {"U", "V", "A"} <= set(f["Frames/0005/Node"])
        cells = f["Mesh/SourceCellDofs"][:]
        u = f["Frames/0005/Native/U"][:]
        np.testing.assert_allclose(
            f["Frames/0005/Node/U"][27:], u[cells[:, 4:]].mean(1)
        )
    results.write_dynamic_history(result, tmp_path / "history.csv")
    assert (tmp_path / "history.csv").exists()


def test_restart(tmp_path):
    complete = build()
    complete.solve_result()
    first = build()
    first.run(until_step=2)
    first.save_checkpoint(tmp_path / "checkpoint")
    resumed = build()
    resumed.load_checkpoint(tmp_path / "checkpoint")
    resumed.solve_result()
    for name in ("u", "v", "a"):
        np.testing.assert_allclose(
            getattr(complete.state, name).x.array,
            getattr(resumed.state, name).x.array,
            atol=1e-15,
        )
    for name in ("external_work", "energy_balance_error"):
        assert complete.history_records[-1][name] == pytest.approx(
            resumed.history_records[-1][name], abs=1e-15
        )


def test_piecewise_probe_not_p2():
    s = build()
    V = s.state.u.function_space
    x = V.tabulate_dof_coordinates()
    # A quadratic nodal assignment must be interpolated piecewise linearly.
    s.state.u.x.array.reshape(-1, 3)[:] = x * x
    s.state.u.x.scatter_forward()
    point = np.array([0.15, 0.1, 0.05])
    measured = results.probe(s.state.u, at=point)
    assert np.linalg.norm(measured - point**2) > 0.01
    from agentfem.results.dynamic_history import _bind_spatial_point

    ds, weights, meta = _bind_spatial_point(V, point)
    np.testing.assert_allclose(measured, weights @ (x[ds] * x[ds]), atol=1e-14)
    assert meta["method"] == "composite_piecewise_linear"


def test_mpi_assembly_and_output(tmp_path):
    import h5py

    comm = MPI.COMM_WORLD
    s = build(comm=comm, mean=True)
    # All ranks use rank zero's output directory.
    path = comm.bcast(str(tmp_path / "distributed.xdmf"), root=0)
    total = comm.allreduce(s.residual.internal.mass_diagonal.sum()) / 3
    assert total == pytest.approx(2, rel=1e-13)
    s.solve_result(
        output=path,
        field_variables=("S", "E"),
        history=results.dynamic_history("q", at=(0.8, 0.8, 0.8)),
    )
    # Same policy for the independent serial reference.
    serial = build(mean=True)
    serial.solve_result(history=results.dynamic_history("q", at=(0.8, 0.8, 0.8)))
    for name in ("q.U1", "q.V2", "q.A3", "external_work", "hourglass_energy"):
        np.testing.assert_allclose(
            s.history_records[-1][name], serial.history_records[-1][name], atol=1e-13
        )
    if comm.rank == 0:
        with h5py.File(path.replace(".xdmf", ".h5")) as f:
            assert f["Mesh/SourceCellDofs"].shape == (6, 10)
            assert f["Frames/0005/Native/U"].shape == (27, 3)


@pytest.mark.parametrize("kind", ["body", "traction", "pressure"])
def test_consistent_load_resultant_and_virtual_work(kind):
    from agentfem import loads, mesh, amplitudes

    comm = MPI.COMM_WORLD
    msh = dm.create_unit_cube(comm, 1, 1, 1)
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=msh
    )
    u = m.field(fields.displacement(msh, degree=2))
    m.material(constitutive.isotropic_elastic(young=100, poisson=0.25, density=2))
    if kind == "body":
        asset = loads.body_force((2, 0, 0), domain=msh)
        expected = 2.0
    else:
        right = mesh.boundary_region(msh, lambda x: np.isclose(x[0], 1))
        asset = (
            loads.traction((2, 0, 0), on=right)
            if kind == "traction"
            else loads.pressure(2, on=right)
        )
        expected = 2.0 if kind == "traction" else -2.0
    m.load(loads.with_amplitude(asset, amplitudes.constant(3), domain=msh))
    s = m.step(
        target=u,
        element_policy=elements.composite_tet10_policy(),
        dt=0.001,
        steps=2,
        progress=False,
    )
    f = s.residual.external.assemble_vector()
    try:
        total = comm.allreduce(f.array.reshape(-1, 3).sum(0))
        np.testing.assert_allclose(total, [3 * expected, 0, 0], atol=1e-13)
        x = u.function_space.tabulate_dof_coordinates()[: len(f.array) // 3]
        work = comm.allreduce(float(np.sum(f.array.reshape(-1, 3) * x)))
        assert work == pytest.approx(3 * expected * (0.5 if kind == "body" else 1.0))
    finally:
        f.destroy()
    s.solve_result()
    assert s.history_records[-1]["external_work"] > 0


def test_final_result_export(tmp_path):
    s = build(mean=True)
    r = s.solve_result(field_variables=("S", "E"))
    artifact = results.write_result_fields(r, tmp_path / "final.xdmf")
    assert artifact.xdmf.exists()
    assert artifact.layout == "exact_piecewise_linear_subtets"


def test_declared_field_policy_and_operator_guard():
    msh = dm.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=msh
    )
    u = m.field(
        fields.displacement(msh, element_policy=elements.composite_tet10_policy())
    )
    m.material(constitutive.isotropic_elastic(young=100, poisson=0.25, density=2))
    assert u.summary()["element_policy"]["formulation"] == "composite_tet10"
    for method in (m.stiffness, m.mass, m.lumped_mass):
        with pytest.raises(ValueError, match="ordinary P2"):
            method(u)
    s = m.step(target=u, dt="auto", steps=2, progress=False)
    s.solve_result()
    assert s.completed_steps == 2
    with pytest.raises(ValueError, match="policies must agree"):
        m.step(
            target=u,
            element_policy=elements.composite_tet10_policy(mean_dilatation=True),
            dt="auto",
            steps=1,
            progress=False,
        )


@pytest.mark.parametrize("prescribed", [False, True])
def test_rigid_acceleration_energy(prescribed):
    from agentfem import loads, amplitudes

    msh = dm.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=msh
    )
    u = m.field(
        fields.displacement(
            msh, element_policy=elements.composite_tet10_policy(mean_dilatation=True)
        )
    )
    m.material(constitutive.isotropic_elastic(young=100, poisson=0.25, density=2))
    if prescribed:
        m.fix(
            u,
            on=lambda x: np.ones(x.shape[1], dtype=bool),
            components=0,
            value=amplitudes.Amplitude("motion", lambda t: 0.5 * t * t),
        )
    else:
        m.load(loads.body_force((2, 0, 0), domain=msh))
    s = m.step(target=u, dt=0.001, steps=20, history_every=7, progress=False)
    s.solve_result()
    row = s.history_records[-1]
    np.testing.assert_allclose(s.state.u.x.array[::3], 0.5 * 0.02**2, atol=1e-14)
    assert row["kinetic_energy"] == pytest.approx(0.02**2, rel=1e-9)
    assert row["external_work"] == pytest.approx(0.02**2, rel=1e-9)
    assert row["hourglass_energy"] < 1e-20


def test_material_partition_and_stress_recovery():
    from agentfem import mesh
    from agentfem.results._composite_tet import CompositeTetCellFields

    msh = dm.create_unit_cube(MPI.COMM_WORLD, 2, 1, 1)
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=msh
    )
    u = m.field(
        fields.displacement(msh, element_policy=elements.composite_tet10_policy())
    )
    regions = mesh.partition_cells(
        msh, left=lambda x: x[0] < 0.5, right=lambda x: x[0] >= 0.5
    )
    left, right = regions.left, regions.right
    m.material(
        constitutive.isotropic_elastic(young=100, poisson=0, density=2), region=left
    )
    m.material(
        constitutive.isotropic_elastic(young=200, poisson=0, density=4), region=right
    )
    s = m.step(target=u, dt=0.001, steps=1, progress=False)
    V = s.state.u.function_space
    x = V.tabulate_dof_coordinates()
    s.state.u.x.array.reshape(-1, 3)[:] = np.column_stack(
        (0.02 * x[:, 1], np.zeros((len(x), 2)))
    )
    live = CompositeTetCellFields(s.residual.internal, ("S", "E", "MISES", "SENER"))
    out = {f.name: f for f in live.update()}
    count = msh.topology.index_map(3).size_local
    for c in range(count):
        stress = out["S"].x.array.reshape(-1, 3, 3)[
            out["S"].function_space.dofmap.cell_dofs(c)[0]
        ]
        C = s.residual.internal.materials[c]
        expected = 0.02 * C[3, 3]
        assert stress[0, 1] == pytest.approx(expected, abs=1e-13)
        assert stress[1, 0] == pytest.approx(expected, abs=1e-13)
    mass = msh.comm.allreduce(s.residual.internal.mass_diagonal.sum()) / 3
    assert mass == pytest.approx(3.0)


def test_global_affine_patch_and_anisotropic_shear_order():
    from agentfem.constitutive import elasticity
    from agentfem.results._composite_tet import CompositeTetCellFields

    comm = MPI.COMM_WORLD
    msh = dm.create_unit_cube(comm, 2, 2, 2)
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=msh
    )
    u = m.field(
        fields.displacement(msh, element_policy=elements.composite_tet10_policy())
    )
    # Different shear moduli detect xy/yz/xz permutations that isotropy hides.
    C = np.diag([120.0, 130.0, 140.0, 7.0, 11.0, 17.0])
    m.material(elasticity.anisotropic_elastic_3d(stiffness_voigt=C, density=2))
    s = m.step(target=u, dt=0.0001, steps=1, progress=False)
    V = s.state.u.function_space
    x = V.tabulate_dof_coordinates()
    G = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]])
    s.state.u.x.array.reshape(-1, 3)[:] = x @ G.T + [0.1, 0.2, 0.3]
    f = s.residual.internal.assemble_vector()
    try:
        n = len(f.array) // 3
        force = f.array.reshape(-1, 3)
        inside = np.all((x[:n] > 1e-10) & (x[:n] < 1 - 1e-10), axis=1)
        np.testing.assert_allclose(force[inside], 0, atol=3e-13)
        np.testing.assert_allclose(comm.allreduce(force.sum(0)), 0, atol=3e-13)
        np.testing.assert_allclose(
            comm.allreduce(np.cross(x[:n], force).sum(0)), 0, atol=3e-13
        )
    finally:
        f.destroy()
    out = CompositeTetCellFields(s.residual.internal, ("S",)).update()[0]
    strain = np.array(
        [
            G[0, 0],
            G[1, 1],
            G[2, 2],
            G[1, 2] + G[2, 1],
            G[0, 2] + G[2, 0],
            G[0, 1] + G[1, 0],
        ]
    )
    stress = C @ strain
    expected = stress[np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])]
    count = msh.topology.index_map(3).size_local
    for c in range(count):
        dof = out.function_space.dofmap.cell_dofs(c)[0]
        np.testing.assert_allclose(
            out.x.array.reshape(-1, 3, 3)[dof], expected, atol=2e-13
        )


def test_option_contract_and_unsupported_normal():
    from agentfem import mesh, loads

    msh = dm.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=msh
    )
    u = m.field(
        fields.displacement(msh, element_policy=elements.composite_tet10_policy())
    )
    m.material(constitutive.isotropic_elastic(young=100, poisson=0.25, density=2))
    with pytest.raises(TypeError, match="mass"):
        m.step(target=u, dt=0.001, steps=1, mass=np.ones(81), progress=False)
    face = mesh.boundary_region(msh, lambda x: np.isclose(x[0], 1))
    import ufl

    m.load(loads.pressure(2, on=face, normal=ufl.as_vector((0, 1, 0))))
    with pytest.raises(NotImplementedError, match="custom normals"):
        m.step(target=u, dt=0.001, steps=1, progress=False)


def test_standard_project_publication(tmp_path, capsys):
    from pathlib import Path
    import shutil
    import json
    from agentfem import cli

    source = Path(__file__).resolve().parents[1] / "examples/composite_tet_dynamics"
    for name in ("case.py", "agentfem.toml"):
        shutil.copy2(source / name, tmp_path / name)
    assert (
        cli.main(
            ["run", "--project", str(tmp_path), "--name", "composite-test", "--json"]
        )
        == 0
    )
    capsys.readouterr()
    manifests = list((tmp_path / "outputs").rglob("result.json"))
    assert len(manifests) == 1
    record = json.loads(manifests[0].read_text())
    assert record["metadata"]["field_output"]["warp_field"] == "U"
    assert (
        record["metadata"]["step"]["element_policy"]["formulation"] == "composite_tet10"
    )
    assert (manifests[0].parent / "fields.xdmf").exists()
    assert (manifests[0].parent / "fields.h5").exists()
    assert (manifests[0].parent / "history.csv").exists()


def test_unified_reader_uses_true_subtet_interpolation(tmp_path):
    pytest.importorskip("pyvista")
    s = build()
    s.solve_result(output=tmp_path / "reader.xdmf")
    grids = results.read_unified_xdmf_series(tmp_path / "reader.xdmf")
    assert len(grids) == 6
    assert grids[-1].n_cells == 72
    assert grids[-1].n_points == 33
    assert set(("U", "V", "A")) <= set(grids[-1].point_data)
