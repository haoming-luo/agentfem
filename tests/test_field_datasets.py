from __future__ import annotations

import numpy as np
import pytest

from agentfem import datasets, learning


def _field_dataset(case_count: int = 10):
    shape = (1, 8, 6)
    conductivity = learning.FieldEncoding(
        name="conductivity",
        role="input",
        unit="W/(m K)",
        representation="structured_grid",
        shape=shape,
        mesh_policy="mesh_independent_coordinates",
    )
    temperature = learning.FieldEncoding(
        name="temperature",
        role="output",
        unit="K",
        representation="structured_grid",
        shape=shape,
        mesh_policy="mesh_independent_coordinates",
    )
    x = np.linspace(0.0, 1.0, shape[-2])
    y = np.linspace(0.0, 0.5, shape[-1])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    conductivities = []
    temperatures = []
    coordinates = []
    for index in range(case_count):
        amplitude = 1.0 + 0.1 * index
        conductivities.append((amplitude + 0.1 * xx)[None, ...])
        temperatures.append((300.0 + amplitude * xx + yy)[None, ...])
        coordinates.append(np.stack((xx, yy), axis=0))
    return datasets.ScientificFieldDataset(
        case_ids=tuple(f"case-{index:03d}" for index in range(case_count)),
        encodings=(conductivity, temperature),
        fields={
            "conductivity": np.asarray(conductivities),
            "temperature": np.asarray(temperatures),
        },
        coordinates={"observation_grid": np.asarray(coordinates)},
        parameters={"amplitude": np.linspace(1.0, 1.9, case_count)},
        case_metadata=tuple(
            {"run_id": f"run-{index:03d}"} for index in range(case_count)
        ),
        name="heat_operator",
        metadata={"source": "manufactured_heat_family"},
    )


def test_field_dataset_contract_split_and_round_trip(tmp_path):
    dataset = _field_dataset()
    split = dataset.split(validation_fraction=0.2, test_fraction=0.2, seed=12)
    manifest = dataset.write(tmp_path / "operator_fields")
    restored = datasets.ScientificFieldDataset.read(manifest)

    assert dataset.input_names == ("conductivity",)
    assert dataset.output_names == ("temperature",)
    assert len(split.train.case_ids) == 6
    assert len(split.validation.case_ids) == 2
    assert split.test is not None and len(split.test.case_ids) == 2
    assert set(split.train.case_ids).isdisjoint(split.validation.case_ids)
    assert restored.fingerprint == dataset.fingerprint
    assert restored.case_ids == dataset.case_ids
    assert restored.encoding("temperature")["unit"] == "K"
    np.testing.assert_allclose(
        restored.fields["temperature"], dataset.fields["temperature"]
    )
    np.testing.assert_allclose(
        restored.coordinates["observation_grid"],
        dataset.coordinates["observation_grid"],
    )


def test_field_dataset_rejects_schema_and_array_mismatches(tmp_path):
    dataset = _field_dataset(case_count=4)
    with pytest.raises(ValueError, match="match encoding names"):
        datasets.ScientificFieldDataset(
            case_ids=dataset.case_ids,
            encodings=dataset.encodings,
            fields={"conductivity": dataset.fields["conductivity"]},
        )
    with pytest.raises(ValueError, match="leading case dimension"):
        datasets.ScientificFieldDataset(
            case_ids=dataset.case_ids,
            encodings=dataset.encodings,
            fields={
                "conductivity": dataset.fields["conductivity"][:-1],
                "temperature": dataset.fields["temperature"],
            },
        )

    manifest = dataset.write(tmp_path / "fingerprinted")
    arrays_path = manifest.parent / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {name: np.asarray(saved[name]) for name in saved.files}
    arrays["fields_0"] = arrays["fields_0"] + 1.0
    np.savez_compressed(arrays_path, **arrays)
    with pytest.raises(ValueError, match="fingerprint"):
        datasets.ScientificFieldDataset.read(manifest)


def test_field_dataset_masks_match_field_geometry():
    dataset = _field_dataset(case_count=3)
    with pytest.raises(ValueError, match="does not match field"):
        datasets.ScientificFieldDataset(
            case_ids=dataset.case_ids,
            encodings=dataset.encodings,
            fields=dataset.fields,
            masks={"temperature": np.ones((3, 2), dtype=bool)},
        )
