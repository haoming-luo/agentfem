from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentfem import constitutive, materials, models, studies


def test_named_material_resolves_mechanical_behavior_at_model_boundary():
    behavior = constitutive.isotropic_elastic(
        name="elastic law",
        young=210.0e9,
        poisson=0.3,
        density=7850.0,
    )
    steel = materials.define(
        "laboratory steel",
        behavior,
        source="project calibration",
    )
    model = models.create(study=studies.static_solid(dimension=3))

    resolved = model.material(steel)

    assert resolved is behavior
    assert model._material_record(steel).item is behavior
    assert model.materials[0].definition is steel
    assert model.materials[0].summary()["material_definition"]["name"] == "laboratory steel"


def test_material_definition_keeps_thermal_and_mechanical_roles_independent():
    mechanical = constitutive.isotropic_elastic(
        young=70.0e9,
        poisson=0.33,
        density=2700.0,
    )
    thermal = SimpleNamespace(
        conductivity=205.0,
        volumetric_heat_capacity=2.43e6,
    )
    aluminum = materials.define(
        "aluminum",
        mechanical=mechanical,
        thermal=thermal,
        source="project data",
    )

    solid_model = models.create(study=studies.static_solid(dimension=3))
    heat_model = models.create(study=studies.transient_heat_transfer(dimension=3))

    assert solid_model.material(aluminum) is mechanical
    assert heat_model.material(aluminum) is thermal
    assert aluminum.roles == ("mechanical", "thermal")


def test_dynamic_study_rejects_material_without_density_before_solve():
    behavior = SimpleNamespace(young=1.0e6, poisson=0.49)
    material = materials.define("rubber", behavior)
    model = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit")
    )

    with pytest.raises(ValueError, match="density"):
        model.material(material)


def test_packaged_cards_are_explicitly_reference_only():
    steel = materials.load_definition("steel_generic")

    assert steel.reference_only is True
    assert steel.metadata["library_id"] == "steel_generic"
    assert steel.behavior("mechanical").young == pytest.approx(200.0e9)


def test_load_uses_one_api_for_packaged_cards_and_project_python(tmp_path):
    asset = tmp_path / "active.py"
    asset.write_text(
        "\n".join(
            (
                "from agentfem import constitutive, materials",
                "material = materials.define(",
                "    'project alloy',",
                "    constitutive.isotropic_elastic(",
                "        young=123.0, poisson=0.25, density=7.0",
                "    ),",
                ")",
            )
        ),
        encoding="utf-8",
    )

    project_material = materials.load(asset)
    packaged_material = materials.load("steel_generic")

    assert project_material.name == "project alloy"
    assert project_material.behavior("mechanical").young == pytest.approx(123.0)
    assert project_material.metadata["source_sha256"]
    assert project_material.metadata["symbol"] == "material"
    assert packaged_material.reference_only is True


def test_load_python_requires_one_explicit_public_material_symbol(tmp_path):
    asset = tmp_path / "ambiguous.py"
    asset.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(materials.MaterialAssetError, match="must publish"):
        materials.load(asset)
