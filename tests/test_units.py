from agentfem import models, studies, units


def test_unit_system_records_consistent_base_and_derived_contracts():
    selected = units.n_mm_mpa(temperature="degC")

    assert selected.length == "mm"
    assert selected.mass == "tonne"
    assert selected.stress == "tonne/(mm*s^2)"
    assert selected.summary()["automatic_conversion"] is False


def test_model_manifest_exposes_units_without_changing_numerical_values():
    model = models.create(
        study=studies.static_solid(dimension=3),
        units=units.si(),
    )

    assert model.summary()["unit_system"]["name"] == "SI"
    assert model.manifest()["model"]["unit_system"]["base"]["length"] == "m"
