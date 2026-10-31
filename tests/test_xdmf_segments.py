# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Continuation viewing must not mutate sources or invent saved states."""

import hashlib
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

from agentfem import results


def segment(folder, times, *, offset=0., incomplete=False):
    folder.mkdir(parents=True)
    heavy = folder / "data.h5"
    root = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(root, "Domain")
    if incomplete:
        ET.SubElement(domain, "Information", Name="status", Value="incomplete")
    collection = ET.SubElement(domain, "Grid", GridType="Collection", CollectionType="Temporal")
    with h5py.File(heavy, "w") as store:
        store["points"] = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        store["cells"] = np.array([[0, 1, 2, 3]], dtype=np.int64)
        for i, time in enumerate(times):
            grid = ET.SubElement(collection, "Grid", GridType="Uniform")
            ET.SubElement(grid, "Time", Value=format(time, ".17g"))
            topology = ET.SubElement(grid, "Topology", TopologyType="Tetrahedron", NumberOfElements="1")
            ET.SubElement(topology, "DataItem", Format="HDF", Dimensions="1 4",
                          NumberType="Int", Precision="8").text = "data.h5:/cells"
            geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
            ET.SubElement(geometry, "DataItem", Format="HDF", Dimensions="4 3",
                          NumberType="Float", Precision="8").text = "data.h5:/points"
            attribute = ET.SubElement(grid, "Attribute", Name="u", Center="Node", AttributeType="Vector")
            store[f"u{i}"] = np.full((4, 3), time + offset)
            ET.SubElement(attribute, "DataItem", Format="HDF", Dimensions="4 3",
                          NumberType="Float", Precision="8").text = f"data.h5:/u{i}"
    path = folder / "results.xdmf"
    ET.ElementTree(root).write(path)
    return path


def times(path):
    return [float(t.get("Value")) for t in ET.parse(path).findall(".//Time")]


def test_join_shared_boundary_rebases_without_copy_or_mutation(tmp_path):
    a = segment(tmp_path / "a", [0., 1.])
    b = segment(tmp_path / "b", [1., 2.])
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    output = results.join_xdmf_segments([a, b], tmp_path / "view" / "results.xdmf",
                                       expected_times=[0., 1., 2.])
    assert times(output) == [0., 1., 2.]
    assert len(list(output.parent.iterdir())) == 1
    for p, digest in originals.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
    meshio = pytest.importorskip("meshio")
    with meshio.xdmf.TimeSeriesReader(output) as reader:
        points, cells = reader.read_points_cells()
        assert points.shape == (4, 3) and cells[0].type == "tetra"
        for i in range(3):
            time, point_data, _ = reader.read_data(i)
            np.testing.assert_array_equal(point_data["u"], np.full((4, 3), time))


@pytest.mark.parametrize("failure", ["conflict", "overlap", "mesh", "schema", "missing_hdf", "dimensions"])
def test_reject_bad_segments_without_publishing(tmp_path, failure):
    a = segment(tmp_path / "a", [0., 1.])
    b = segment(tmp_path / "b", [.5, 2.] if failure == "overlap" else [1., 2.],
                offset=1. if failure == "conflict" else 0.)
    if failure == "mesh":
        with h5py.File(b.parent / "data.h5", "r+") as store:
            store["cells"][0] = [0, 2, 1, 3]
    if failure in {"schema", "dimensions"}:
        tree = ET.parse(b)
        if failure == "schema":
            tree.find(".//Attribute").set("Name", "temperature")
        else:
            tree.find(".//Attribute/DataItem").set("Dimensions", "5 3")
        tree.write(b)
    if failure == "missing_hdf":
        (b.parent / "data.h5").unlink()
    output = tmp_path / "view.xdmf"
    with pytest.raises((ValueError, OSError)):
        results.join_xdmf_segments([a, b], output)
    assert not output.exists()


def test_sparse_stations_do_not_claim_completeness(tmp_path):
    a = segment(tmp_path / "a", [0., 1.])
    b = segment(tmp_path / "b", [3., 4.], incomplete=True)
    output = results.join_xdmf_segments([a, b], tmp_path / "view.xdmf")
    assert times(output) == [0., 1., 3., 4.]
    info = {i.get("Name"): i.get("Value") for i in ET.parse(output).findall("Domain/Information")}
    assert info["temporal_coverage"] == "not_certified"
    assert info["source_status"] == "incomplete"
    with pytest.raises(ValueError, match="missing"):
        results.join_xdmf_segments([a, b], tmp_path / "invalid.xdmf", expected_times=[2.])


def test_preserve_existing_and_reject_repeated_input(tmp_path):
    a = segment(tmp_path / "a", [0., 1.])
    before = a.read_bytes()
    with pytest.raises(FileExistsError):
        results.join_xdmf_segments([a], a)
    with pytest.raises(ValueError, match="distinct"):
        results.join_xdmf_segments([a, a], tmp_path / "bad.xdmf")
    assert a.read_bytes() == before


def test_times_are_binary64_stations_not_pretty_printed_labels(tmp_path):
    a = segment(tmp_path / "a", [.1, np.nextafter(.1, np.inf)])
    output = results.join_xdmf_segments([a], tmp_path / "view.xdmf",
                                       expected_times=[.1, np.nextafter(.1, np.inf)])
    np.testing.assert_array_equal(times(output), [.1, np.nextafter(.1, np.inf)])


def test_real_live_writer_continuation(tmp_path):
    from agentfem.io import ResultWriter
    from test_xdmf_layout import fields

    domain, u, damage = fields()
    paths = [tmp_path / "first.xdmf", tmp_path / "resumed.xdmf"]
    for path, stations in zip(paths, ([0., .5], [.5, 1.])):
        with ResultWriter(path, domain, (u, damage), storage="frame_files") as writer:
            for station in stations:
                u.interpolate(lambda x, station=station: station * x)
                writer.write(station)
    output = results.join_xdmf_segments(paths, tmp_path / "combined.xdmf")
    assert times(output) == [0., .5, 1.]


def test_explicit_identical_overlap_checks_recomputed_interval(tmp_path):
    a = segment(tmp_path / "a", [0., 1., 2., 3.])
    b = segment(tmp_path / "b", [1., 2., 3., 4.])
    c = segment(tmp_path / "c", [2., 3., 4., 5.])
    originals = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="overlap"):
        results.join_xdmf_segments([a, b], tmp_path / "default.xdmf")
    view = results.join_xdmf_segments([a, b, c], tmp_path / "joined.xdmf",
                                     overlap="identical", expected_times=range(6))
    assert times(view) == list(range(6))
    for path, payload in originals.items():
        assert path.read_bytes() == payload
    info = {i.get("Name"): i.get("Value") for i in ET.parse(view).findall("Domain/Information")}
    assert info["overlap_policy"] == "identical"
    assert info["agentfem_view"] == "visualization_only"


@pytest.mark.parametrize("problem", ["late_difference", "unmatched_time", "mesh", "schema"])
def test_identical_overlap_never_selects_a_conflicting_branch(tmp_path, problem):
    a = segment(tmp_path / "a", [0., 1., 2., 3.])
    b = segment(tmp_path / "b", [1., 2.5, 3., 4.] if problem == "unmatched_time" else [1., 2., 3., 4.])
    if problem in {"late_difference", "mesh"}:
        with h5py.File(b.parent / "data.h5", "r+") as store:
            if problem == "late_difference":
                store["u2"][3, 2] += 1e-12
            else:
                store["cells"][0] = [0, 2, 1, 3]
    if problem == "schema":
        tree = ET.parse(b)
        tree.findall(".//Attribute")[2].set("Name", "other")
        tree.write(b)
    view = tmp_path / "failed.xdmf"
    with pytest.raises(ValueError):
        results.join_xdmf_segments([a, b], view, overlap="identical")
    assert not view.exists()


def test_overlap_policy_must_be_explicit(tmp_path):
    a = segment(tmp_path / "a", [0., 1.])
    with pytest.raises(ValueError, match="overlap"):
        results.join_xdmf_segments([a], tmp_path / "failed.xdmf", overlap="latest_wins")
