# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Detached accepted integration-point fields and physical gauge measurements."""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from .quantities import sample_points
from .statistics import weighted_field_statistics
from .. import fields as field_api


def _immutable(array):
    array = np.asarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class AcceptedMaterialFields:
    """Owned integration-point snapshot; arrays cannot mutate the live Step.

    Coordinates and weights use the reference configuration. A selected ROI is
    a set of quadrature samples, not exact geometric clipping of cut cells.
    Stress and deformation-gradient channels retain their original 3D meaning.
    """
    def __init__(self, *, time, coordinate, coordinates, weights, cell_keys, fields, metadata, comm):
        self.time, self.coordinate = float(time), float(coordinate)
        self.coordinates, self.weights = _immutable(coordinates), _immutable(weights)
        self.cell_keys = _immutable(cell_keys)
        self.fields = MappingProxyType({key: _immutable(value) for key, value in fields.items()})
        self.metadata = MappingProxyType(dict(metadata))
        self.comm = comm

    def __getitem__(self, name):
        return self.fields[name]

    def statistics(self, name, *, within=None, component=None, quantiles=(.05, .5, .95), thresholds=()):
        """Reference-weighted scalar statistics, optionally on a fixed ROI mask."""
        values = self[name]
        if component is not None:
            index = (component,) if isinstance(component, int) else tuple(component)
            values = values[(slice(None), *index)]
        if values.ndim != 1:
            raise ValueError("AFM-OBSERVATION-COMPONENT: select one scalar component explicitly.")
        error = None
        try:
            selected = np.ones(len(values), dtype=bool) if within is None else np.asarray(within(self.coordinates))
            if selected.shape != (len(values),) or selected.dtype != np.dtype(bool):
                raise ValueError("ROI must return one boolean per reference quadrature point.")
        except (ValueError, TypeError) as exc:
            error = str(exc)
        errors = self.comm.allgather(error)
        if any(errors):
            raise ValueError("AFM-OBSERVATION-ROI: " + "; ".join(e for e in errors if e))
        return weighted_field_statistics(values[selected], self.weights[selected],
            location="accepted_reference_integration_points", representation="raw_constitutive",
            quantiles=quantiles, thresholds=thresholds, comm=self.comm)

    def summary(self):
        return {"kind": "accepted_material_fields", "time": self.time, "coordinate": self.coordinate,
            "point_count": len(self.weights), "fields": list(self.fields),
            "coordinate_configuration": "reference", "weight": self.metadata.get("reference_measure", "reference_volume"),
            "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class GaugeMeasurement:
    """Two-point length strain and separately named axial projection."""
    reference_length: float
    current_length: float
    engineering_strain: float
    axial_projected_strain: float
    reference_points: np.ndarray
    current_points: np.ndarray

    def summary(self):
        return {"kind": "two_point_gauge", "reference_length": self.reference_length,
            "current_length": self.current_length, "engineering_strain": self.engineering_strain,
            "axial_projected_strain": self.axial_projected_strain,
            "reference_points": self.reference_points.tolist(), "current_points": self.current_points.tolist(),
            "strain_measure": "end_to_end_length_change; not local Hencky strain"}


def gauge(displacement, *, start, end, padding=1e-10):
    """Measure two reference material points using FE displacement interpolation.

    Coordinate and displacement length units must agree. The objective length
    strain is distinguished from projected displacement along the original
    gauge direction, which changes under rigid rotation.
    """
    function = field_api.unwrap(displacement)
    dimension = function.function_space.mesh.geometry.dim
    if function.ufl_shape != (dimension,):
        raise ValueError("AFM-GAUGE-FIELD: a Cartesian displacement vector is required.")
    sampled = sample_points(function, (start, end), padding=padding, return_info=True)
    reference = sampled.coordinates[:, :dimension]
    direction = reference[1]-reference[0]
    length = float(np.linalg.norm(direction))
    if length <= 0 or not np.isfinite(length):
        raise ValueError("AFM-GAUGE-LENGTH: distinct finite reference points are required.")
    if not sampled.finite.all():
        raise ValueError("AFM-GAUGE-FIELD: finite displacements are required.")
    current = reference + sampled.values
    current_length = float(np.linalg.norm(current[1]-current[0]))
    return GaugeMeasurement(length, current_length, current_length/length-1,
        float((sampled.values[1]-sampled.values[0]) @ (direction/length) / length),
        _immutable(reference), _immutable(current))
