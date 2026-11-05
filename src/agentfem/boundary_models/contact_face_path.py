# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted-station evidence for certified physical contact-face integrals.

This records certified faces only at accepted configurations. It is neither
a between-increment collision proof nor a pointwise pressure field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


_PAIR_KEYS = {
    "constraint", "accepted_station_count", "last_factor",
    "maximum_active_reference_area", "maximum_active_face_count",
    "maximum_mixed_feature_face_count", "all_observed_faces_certified",
}


def validate_contact_face_path(
    path: object, *, accepted_factor: float, accepted_count: int,
    complete: bool,
) -> dict[str, dict[str, object]]:
    """Reject malformed or falsely complete whole-face accepted evidence."""

    if not isinstance(path, dict):
        raise ValueError("Contact face path must be a pair-keyed mapping.")
    validated = {}
    for name, record in path.items():
        if (
            not isinstance(name, str) or not name
            or not isinstance(record, dict) or set(record) != _PAIR_KEYS
            or record["constraint"] != name
        ):
            raise ValueError("Contact face path pair identity or schema differs.")
        for key in (
            "accepted_station_count", "maximum_active_face_count",
            "maximum_mixed_feature_face_count",
        ):
            value = record[key]
            if type(value) is not int or value < (1 if key == "accepted_station_count" else 0):
                raise ValueError("Contact face path contains an invalid count.")
        area = record["maximum_active_reference_area"]
        factor = record["last_factor"]
        if (
            type(area) not in (int, float) or not np.isfinite(area) or area < 0.0
            or type(factor) not in (int, float) or not np.isfinite(factor)
            or abs(float(factor) - accepted_factor) > 1e-12
            or type(record["all_observed_faces_certified"]) is not bool
            or (complete and record["accepted_station_count"] != accepted_count + 1)
            or record["maximum_mixed_feature_face_count"]
            > record["maximum_active_face_count"]
        ):
            raise ValueError("Contact face path and accepted station disagree.")
        validated[name] = dict(record)
    return validated


def accepted_face_record(provider, *, factor: float, previous=None) -> dict[str, object]:
    """Capture one complete physical-face proof without inventing samples."""

    name = str(provider.name)
    evidence = provider.current_face_diagnostics()
    area = evidence.get("active_reference_areas")
    features = evidence.get("feature_kinds")
    certified = evidence.get("all_faces_certified_at_current_trial")
    if (
        evidence.get("location") != "boundary_faces"
        or not isinstance(area, list) or not isinstance(features, list)
        or len(area) != len(features)
        or any(type(value) not in (int, float) or not np.isfinite(value)
               or value < 0.0 for value in area)
        or any(item is not None and (
            not isinstance(item, list)
            or any(not isinstance(kind, str) for kind in item)
        ) for item in features)
        or type(certified) is not bool or not certified
    ):
        raise ValueError("Accepted contact face proof is incomplete or invalid.")
    active_count = sum(value > 0.0 for value in area)
    mixed_count = sum(item is not None and len(item) > 1 for item in features)
    if mixed_count > active_count:
        raise ValueError("Accepted contact face feature counts exceed active faces.")
    record = {
        "constraint": name,
        "accepted_station_count": (
            1 if previous is None else int(previous["accepted_station_count"]) + 1
        ),
        "last_factor": float(factor),
        "maximum_active_reference_area": max(
            float(sum(area)), 0.0 if previous is None else float(
                previous["maximum_active_reference_area"]
            ),
        ),
        "maximum_active_face_count": max(
            active_count, 0 if previous is None else int(
                previous["maximum_active_face_count"]
            ),
        ),
        "maximum_mixed_feature_face_count": max(
            mixed_count, 0 if previous is None else int(
                previous["maximum_mixed_feature_face_count"]
            ),
        ),
        "all_observed_faces_certified": (
            certified if previous is None else
            bool(previous["all_observed_faces_certified"]) and certified
        ),
    }
    return record


@dataclass(eq=False)
class AcceptedContactFacePath:
    """Observe whole-face proofs at each accepted affine/MPC station."""

    requires_field_snapshot = False

    providers: tuple[object, ...]
    complete: bool = field(default=True, init=False)
    accepted_station_count: int = field(default=0, init=False)
    pairs: dict[str, dict[str, object]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.providers = tuple(self.providers)
        names = tuple(str(item.name) for item in self.providers)
        if (
            not self.providers or any(not name for name in names)
            or len(set(names)) != len(names)
            or any(not callable(getattr(item, "current_face_diagnostics", None))
                   for item in self.providers)
        ):
            raise ValueError("Accepted contact face path requires unique face pairs.")

    def _capture(self, snapshot, *, reset: bool) -> None:
        factor = float(snapshot.load_factor)
        if not np.isfinite(factor) or (
            not reset and self.pairs
            and factor <= next(iter(self.pairs.values()))["last_factor"] + 1e-12
        ):
            raise ValueError("Accepted contact face stations must increase strictly.")
        next_pairs = {
            str(item.name): accepted_face_record(
                item, factor=factor,
                previous=None if reset else self.pairs.get(str(item.name)),
            )
            for item in self.providers
        }
        self.pairs = next_pairs
        self.accepted_station_count = int(snapshot.index) + 1

    def reset(self, snapshot) -> None:
        self.complete = True
        self._capture(snapshot, reset=True)

    def prepare_resume(self, snapshot) -> None:
        if (
            self.accepted_station_count != int(snapshot.index) + 1
            or set(self.pairs) != {str(item.name) for item in self.providers}
            or any(abs(item["last_factor"] - float(snapshot.load_factor)) > 1e-12
                   for item in self.pairs.values())
        ):
            raise ValueError("Accepted contact face path differs from resume station.")

    def accept(self, snapshot) -> None:
        if self.accepted_station_count != int(snapshot.index):
            raise ValueError("Accepted contact face path skipped a station.")
        self._capture(snapshot, reset=False)

    def snapshot_runtime_state(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "accepted_station_count": self.accepted_station_count,
            "pairs": {name: dict(item) for name, item in self.pairs.items()},
        }

    def restore_runtime_state(self, state) -> None:
        self.complete = bool(state["complete"])
        self.accepted_station_count = int(state["accepted_station_count"])
        self.pairs = {name: dict(item) for name, item in state["pairs"].items()}

    def evidence(self) -> dict[str, object]:
        return {
            "schema": "agentfem.contact-face-path.v1",
            "complete": self.complete,
            "accepted_station_count": self.accepted_station_count,
            "pairs": tuple(dict(self.pairs[name]) for name in sorted(self.pairs)),
        }

    def summary(self) -> dict[str, object]:
        return {
            "kind": "accepted_certified_contact_face_path",
            "pair_names": tuple(sorted(str(item.name) for item in self.providers)),
            "scope": "certified_accepted_faces_not_between_station_path",
            "complete": self.complete,
            "accepted_station_count": self.accepted_station_count,
        }

    def checkpoint_state(self) -> dict[str, object]:
        evidence = self.evidence()
        return evidence | {"pairs": list(evidence["pairs"])}

    def restore_checkpoint_state(self, record, *, current_snapshot) -> None:
        if (
            not isinstance(record, dict)
            or set(record) != {"schema", "complete", "accepted_station_count", "pairs"}
            or record["schema"] != "agentfem.contact-face-path.v1"
            or type(record["complete"]) is not bool
            or type(record["accepted_station_count"]) is not int
            or record["accepted_station_count"] != int(current_snapshot.index) + 1
            or not isinstance(record["pairs"], list)
        ):
            raise ValueError("Accepted contact face checkpoint schema differs.")
        raw = {}
        for item in record["pairs"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("constraint"), str)
                or item["constraint"] in raw
            ):
                raise ValueError("Accepted contact face checkpoint pair differs.")
            raw[item["constraint"]] = item
        selected = validate_contact_face_path(
            raw,
            accepted_factor=float(current_snapshot.load_factor),
            accepted_count=int(current_snapshot.index),
            complete=record["complete"],
        )
        if set(selected) != {str(item.name) for item in self.providers}:
            raise ValueError("Accepted contact face checkpoint pair coverage differs.")
        self.complete = record["complete"]
        self.accepted_station_count = record["accepted_station_count"]
        self.pairs = selected

    def restore_missing_checkpoint_state(self, *, current_snapshot) -> None:
        self.complete = False
        self._capture(current_snapshot, reset=True)
