# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Trusted, instance-local Q1 geometry arithmetic extensions.

Core retains candidate selection, ambiguity decisions, force construction and
state transactions. Callbacks must preserve the reference geometric problem;
their identity is recorded, not treated as a correctness certificate.
"""
from dataclasses import dataclass, field
import json
import numpy as np


@dataclass(frozen=True)
class Q1ContactExecution:
    identity: dict
    closest_point: object = None
    classify_inside: object = None
    distance_bounds: object = None
    separated_paths: object = None
    _identity_json: str = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.identity, dict) or not self.identity:
            raise ValueError("Q1 execution requires a nonempty JSON identity")
        encoded = json.dumps(self.identity, sort_keys=True, allow_nan=False)
        object.__setattr__(self, "_identity_json", encoded)
        for name in ("closest_point", "classify_inside", "distance_bounds", "separated_paths"):
            if getattr(self, name) is not None and not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable or None")

    def summary(self):
        return {"kind": "trusted_q1_geometry_execution", "identity": json.loads(self._identity_json),
                "closest_point": self.closest_point is not None,
                "classify_inside": self.classify_inside is not None,
                "distance_bounds": self.distance_bounds is not None,
                "separated_paths": self.separated_paths is not None}

    def closest(self, vertices, point, reference):
        if self.closest_point is None:
            return reference(vertices, point)
        result = self.closest_point(vertices, point, reference)
        if not isinstance(result, (tuple, list)) or len(result) != 5:
            raise ValueError("Q1 kernel must return coordinates, weights, point, normal, distance")
        arrays = []
        for value, shape in zip(result, ((2,), (4,), (3,), (3,), ()), strict=True):
            raw = np.asarray(value)
            if np.iscomplexobj(raw) or raw.shape != shape or not np.isfinite(raw).all():
                raise ValueError("Q1 kernel returned invalid geometric arrays")
            arrays.append(np.asarray(raw, dtype=float))
        uv, weights, projected, normal, distance = arrays
        if np.any(uv < 0) or np.any(uv > 1) or distance < 0:
            raise ValueError("Q1 kernel returned invalid coordinates or distance")
        expected = np.array([(1-uv[0])*(1-uv[1]), uv[0]*(1-uv[1]),
                             uv[0]*uv[1], (1-uv[0])*uv[1]])
        scale = max(1., float(np.linalg.norm(vertices)), float(np.linalg.norm(point)))
        tolerance = 512*np.finfo(float).eps*scale
        if (not np.allclose(weights, expected, rtol=0, atol=512*np.finfo(float).eps)
                or np.linalg.norm(projected-weights@vertices) > tolerance
                or abs(float(distance)-np.linalg.norm(projected-point)) > tolerance
                or abs(np.linalg.norm(normal)-1) > 1e-10):
            raise ValueError("Q1 kernel returned inconsistent geometry")
        return uv.copy(), weights.copy(), projected.copy(), normal.copy(), float(distance)

    def inside(self, surface, points, reference):
        if self.classify_inside is None:
            return reference(points)
        result = np.asarray(self.classify_inside(surface, points, reference))
        if result.dtype.kind != "b" or result.shape != (len(points),):
            raise ValueError("Q1 classification kernel must return one boolean per point")
        return result.copy()

    def bounds(self, surface, points, reference, *, refinement_distance):
        if self.distance_bounds is None:
            return reference(points, refinement_distance=refinement_distance)
        values = np.asarray(self.distance_bounds(surface, points, reference,
                                                  refinement_distance=refinement_distance))
        if (values.shape != (len(points),) or np.iscomplexobj(values)
                or not np.isfinite(values).all() or np.any(values < 0)):
            raise ValueError("Q1 lower bounds must be finite nonnegative query-sized arrays")
        return values.copy()

    def separated(self, old, new, cells, start, end, reference):
        if self.separated_paths is None:
            return reference(old, new, cells, start, end)
        values = np.asarray(self.separated_paths(old, new, cells, start, end, reference))
        if values.dtype.kind != 'b' or values.shape != (len(start),):
            raise ValueError("Q1 swept separation requires one boolean certificate per path")
        return values.copy()
