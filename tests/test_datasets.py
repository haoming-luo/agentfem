from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import campaigns, datasets, fields, mesh, surrogates


def _dataset():
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("x", 1.0e-2, 1.0e2, scale="log", unit="Pa"),
        campaigns.ChoiceParameter("mode", ("a", "b")),
    )
    quantities = (
        datasets.Quantity("qoi", unit="m"),
        datasets.Quantity(
            "field",
            shape=(3,),
            unit="K",
            kind="sampled_field",
            field_encoding={"coordinates": [0.0, 0.5, 1.0]},
        ),
    )
    samples = (
        datasets.Sample(
            case_id="a",
            inputs={"x": 1.0e-2, "mode": "a"},
            outputs={"qoi": 1.0, "field": [1.0, 2.0, 3.0]},
            provenance={"run": "run-a"},
        ),
        datasets.Sample(
            case_id="b",
            inputs={"x": 1.0e2, "mode": "b"},
            outputs={"qoi": 2.0, "field": [3.0, 2.0, 1.0]},
            provenance={"run": "run-b"},
        ),
    )
    return datasets.ScientificDataset(
        parameter_space=space,
        quantities=quantities,
        samples=samples,
        name="scientific",
    )


def test_dataset_matrix_contract_and_round_trip(tmp_path):
    dataset = _dataset()
    manifest = dataset.write(tmp_path / "dataset")
    restored = datasets.ScientificDataset.read(manifest)

    assert dataset.x_matrix().shape == (2, 3)
    assert dataset.y_matrix().shape == (2, 4)
    np.testing.assert_allclose(dataset.x_matrix()[0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(dataset.x_matrix()[1], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(restored.x_matrix(), dataset.x_matrix())
    np.testing.assert_allclose(restored.y_matrix(), dataset.y_matrix())
    assert restored.quantities[1].unit == "K"
    assert restored.samples[0].provenance["run"] == "run-a"


def test_choice_features_are_one_hot_and_round_trip_without_ordinal_distance():
    space = _dataset().parameter_space

    encoded_a = space.encode({"x": 1.0, "mode": "a"})
    encoded_b = space.encode({"x": 1.0, "mode": "b"})

    np.testing.assert_allclose(encoded_a, [0.5, 1.0, 0.0])
    np.testing.assert_allclose(encoded_b, [0.5, 0.0, 1.0])
    assert space.decode(encoded_a) == {"x": 1.0, "mode": "a"}
    assert space.decode(encoded_b) == {"x": 1.0, "mode": "b"}

    with pytest.raises(ValueError, match="one-hot"):
        space.decode([0.5, 0.5, 0.5])


def test_field_quantity_requires_an_explicit_encoding():
    with pytest.raises(ValueError, match="field_encoding"):
        datasets.Quantity(
            "temperature",
            shape=(4,),
            unit="K",
            kind="sampled_field",
        )


def test_dataset_rejects_wrong_output_shape():
    dataset = _dataset()
    with pytest.raises(ValueError, match="requires shape"):
        datasets.ScientificDataset(
            parameter_space=dataset.parameter_space,
            quantities=dataset.quantities,
            samples=(
                datasets.Sample(
                    case_id="bad",
                    inputs={"x": 1.0, "mode": "a"},
                    outputs={"qoi": 1.0, "field": [1.0, 2.0]},
                ),
            ),
        )


def test_dataset_split_is_reproducible_and_nonempty():
    base = _dataset()
    samples = tuple(
        datasets.Sample(
            case_id=str(index),
            inputs={"x": 10.0 ** (-2.0 + 4.0 * index / 9.0), "mode": "a"},
            outputs={"qoi": float(index), "field": [index, index + 1, index + 2]},
        )
        for index in range(10)
    )
    dataset = datasets.ScientificDataset(
        parameter_space=base.parameter_space,
        quantities=base.quantities,
        samples=samples,
    )

    first = dataset.split(validation_fraction=0.2, seed=10)
    second = dataset.split(validation_fraction=0.2, seed=10)

    assert [item.case_id for item in first.train.samples] == [
        item.case_id for item in second.train.samples
    ]
    assert len(first.train.samples) == 8
    assert len(first.validation.samples) == 2


def test_fem_field_sample_preserves_coordinates_values_and_encoding(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.5),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    temperature = fields.temperature(domain)
    temperature.value.interpolate(lambda x: 300.0 + 10.0 * x[0])
    encoding = surrogates.FieldEncoding(
        name="temperature",
        role="output",
        unit="K",
        representation="mesh_dofs",
    )

    sample = datasets.fem_field_sample(temperature, encoding)
    path = sample.write(tmp_path / "temperature_field.npz")

    assert sample.coordinates.shape[0] == sample.values.shape[0]
    assert sample.encoding["unit"] == "K"
    assert sample.metadata["source"] == "dolfinx_owned_coefficients"
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["values"], sample.values)
        assert "temperature" in str(saved["encoding_json"])


def test_structured_observation_grid_exports_fno_ready_field_and_mask(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.5),
        (4, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    temperature = fields.temperature(domain)
    temperature.value.interpolate(lambda x: 300.0 + 10.0 * x[0] + 4.0 * x[1])
    grid = surrogates.regular_grid(
        bounds=((0.0, 1.2), (0.0, 0.5)),
        shape=(4, 3),
    )

    sample = datasets.fem_observation_sample(
        temperature,
        grid,
        unit="K",
        outside="mask",
    )
    path = sample.write(tmp_path / "temperature_grid.npz")

    assert sample.values.shape == (4, 3)
    assert sample.mask.shape == (4, 3)
    assert np.all(sample.mask[:3])
    assert not np.any(sample.mask[3])
    np.testing.assert_allclose(sample.values[0, 0], 300.0)
    np.testing.assert_allclose(sample.values[2, 2], 310.0)
    assert sample.encoding["representation"] == "structured_grid"
    assert sample.encoding["mesh_policy"] == "mesh_independent_coordinates"
    assert sample.encoding["metadata"]["layout"] == (
        "grid_axes_then_value_components"
    )
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["mask"], sample.mask)


def test_observation_coordinate_map_and_rectilinear_contract_round_trip(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.5),
        (4, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    temperature = fields.temperature(domain)
    temperature.value.interpolate(lambda x: 100.0 + 10.0 * x[0] + x[1])
    grid = surrogates.regular_grid(
        bounds=((0.0, 2.0), (0.0, 1.0)),
        shape=(5, 3),
        coordinate_system="publication_panel",
        coordinate_unit="mm",
    )
    registration = surrogates.AffineCoordinateMap(
        matrix=np.diag((0.5, 0.5)),
        offset=np.zeros(2),
        source_coordinate_system="publication_panel",
        target_coordinate_system="reference_mesh",
        source_unit="mm",
        target_unit="mm",
    )
    sample = datasets.fem_observation_sample(
        temperature,
        grid,
        unit="K",
        coordinate_map=registration,
        configuration="current",
    )
    path = sample.write(tmp_path / "registered_field")
    assert path.suffix == ".npz"
    restored = datasets.FEMFieldSample.read(path)
    observation = datasets.RectilinearObservation.from_field_sample(restored)
    observation_path = observation.write(tmp_path / "publication_field")
    restored_observation = datasets.RectilinearObservation.read(observation_path)

    np.testing.assert_allclose(restored.sampling_coordinates[:, 0], 0.5 * restored.coordinates[:, 0])
    np.testing.assert_allclose(sample.values[-1, -1], 110.5)
    assert observation.values.shape == (3, 5)
    assert observation.configuration == "current"
    assert observation.coordinate_system == "publication_panel"
    assert observation.coordinate_unit == "mm"
    np.testing.assert_allclose(restored_observation.values, observation.values)
    assert restored_observation.summary()["valid_samples"] == 15


def test_observation_grid_rejects_ambiguous_or_invalid_axes():
    with pytest.raises(ValueError, match="strictly increasing"):
        surrogates.ObservationGrid.from_axes(x=(0.0, 0.5, 0.5))
    with pytest.raises(ValueError, match="same dimension"):
        surrogates.regular_grid(
            bounds=((0.0, 1.0),),
            shape=(3, 4),
        )


def test_scientific_dataset_has_optional_pytorch_bridge():
    torch = pytest.importorskip("torch")
    bundle = _dataset().to_torch()
    features, targets = next(iter(bundle.loader(batch_size=2, shuffle=False)))

    assert isinstance(features, torch.Tensor)
    assert features.shape == (2, 3)
    assert targets.shape == (2, 4)
    assert bundle.output_names == ("qoi", "field")
