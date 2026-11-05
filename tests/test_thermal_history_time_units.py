import numpy as np
import pytest
from mpi4py import MPI

from agentfem import fields, histories, mesh, results
from agentfem.mechanics._prescribed_temperature import freeze_uniform_temperature
from test_small_strain_physical_time import make_step
from test_small_strain_temperature import ThermalMaterial
from test_history_material_temperature import build


def path(unit):
    return histories.temperature(300., coordinate_unit=unit).record(0.).record(2., 350.)


@pytest.mark.parametrize("unit", ["", "  ", 1, True])
def test_invalid_coordinate_unit_rejected(unit):
    with pytest.raises(ValueError, match="coordinate_unit"):
        path(unit)


@pytest.mark.parametrize("nodal", [False, True])
def test_coordinate_unit_survives_scalar_and_portable_archive(tmp_path, nodal):
    domain = mesh.rectangle((0., 0.), (1., 1.), (1, 1), comm=MPI.COMM_SELF)
    source = fields.temperature(domain, value=300.) if nodal else 300.
    history = histories.temperature(source, coordinate_unit="h")
    history.record(0.).record(2.)
    saved = history.save(tmp_path / "thermal")
    restored = histories.FieldHistory.load(saved, source=source)
    assert restored.coordinate_unit == "h"
    assert restored.portable_identity() == history.portable_identity()
    np.testing.assert_array_equal(restored.sample(1.), history.sample(1.))
    # Cached numerical hashes must not mask changed unit metadata.
    history.coordinate_unit = "s"
    assert history.portable_identity()["coordinate_unit"] == "s"
    history.coordinate_unit = None
    assert "coordinate_unit" not in history.portable_identity()


def test_legacy_identity_and_matching_path_remain_unchanged():
    old = path(None)
    assert "coordinate_unit" not in old.scientific_identity()
    expected = freeze_uniform_temperature(old, 2.)
    actual = freeze_uniform_temperature(path("h"), 2., time_unit="h")
    assert actual == expected
    old.coordinate_unit = "h"
    assert old.scientific_identity()["coordinate_unit"] == "h"
    old.coordinate_unit = None
    assert "coordinate_unit" not in old.scientific_identity()


@pytest.mark.parametrize("target", [None, "s"])
def test_declared_source_unit_needs_matching_target(target):
    with pytest.raises(ValueError, match="coordinate_unit.*time_unit"):
        freeze_uniform_temperature(path("h"), 2., time_unit=target)


def test_small_strain_hour_history_records_hour_clock():
    material = ThermalMaterial()
    step = make_step(material, temperature=path("h"), time_unit="h",
                     material_points=(results.material_point("a", cell=0),))
    step.solve()
    record = next(h for h in step.material_point_histories if h.name == "a.TEMP")
    assert record.abscissa_unit == "h"
    assert record.latest == 350.
    assert record.abscissa[-1] == 2.


def test_small_strain_mismatch_fails_before_material_update():
    material = ThermalMaterial()
    with pytest.raises(ValueError, match="coordinate_unit"):
        make_step(material, temperature=path("h"), time_unit="s")
    assert material.thermal_calls == []


def test_finite_strain_consumer_rejects_hours_and_accepts_seconds():
    with pytest.raises(ValueError, match="coordinate_unit"):
        build(temperature=path("h"))
    step, material = build(temperature=path("s"))
    step.solve()
    assert material.requests
