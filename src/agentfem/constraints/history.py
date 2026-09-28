# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted-station histories for provider-owned constraint duals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConstraintDualHistory:
    """Restartable force--coordinate evidence from accepted nonlinear states.

    Providers own each dual's physical definition.  This container validates,
    transports, and integrates accepted samples, so failed Newton attempts
    never enter the scientific work ledger.
    """

    records: list[dict[str, object]] = field(default_factory=list)

    _SCHEMA = "agentfem.constraint-dual-history.v1"

    def clear(self) -> None:
        self.records.clear()

    def append(self, load_factor: float, evidence) -> None:
        record = {
            "load_factor": float(load_factor),
            "duals": [
                {
                    "constraint_name": item.constraint_name,
                    "role": item.role,
                    "force": np.asarray(item.force, dtype=float).reshape(-1).tolist(),
                    "coordinate": (
                        None
                        if item.coordinate is None
                        else np.asarray(item.coordinate, dtype=float)
                        .reshape(-1)
                        .tolist()
                    ),
                    "resultant": (
                        None
                        if item.resultant is None
                        else np.asarray(item.resultant, dtype=float)
                        .reshape(-1)
                        .tolist()
                    ),
                    "source": str(item.source),
                }
                for item in evidence
            ],
        }
        self.records[:] = self._validated((*self.records, record))

    @classmethod
    def _validated(cls, records, *, accepted_factor: float | None = None):
        selected: list[dict[str, object]] = []
        previous = -np.inf
        layout = None
        for station_index, raw in enumerate(records):
            if not isinstance(raw, dict):
                raise TypeError("Constraint dual history stations must be mappings.")
            factor = float(raw["load_factor"])
            if not np.isfinite(factor) or factor <= previous:
                raise ValueError(
                    f"Invalid constraint dual history station {station_index}."
                )
            station = []
            names = set()
            for dual_index, item in enumerate(raw.get("duals", ())):
                if not isinstance(item, dict):
                    raise TypeError("Constraint dual history entries must be mappings.")
                name = str(item.get("constraint_name", "")).strip()
                role = str(item.get("role", "")).strip()
                source = str(item.get("source", "")).strip()
                force = np.asarray(item.get("force", ()), dtype=float).reshape(-1)
                raw_coordinate = item.get("coordinate")
                coordinate = (
                    None
                    if raw_coordinate is None
                    else np.asarray(raw_coordinate, dtype=float).reshape(-1)
                )
                raw_resultant = item.get("resultant")
                resultant = (
                    None
                    if raw_resultant is None
                    else np.asarray(raw_resultant, dtype=float).reshape(-1)
                )
                invalid = (
                    not name
                    or name in names
                    or not role
                    or not source
                    or force.size == 0
                    or not np.all(np.isfinite(force))
                    or (
                        coordinate is not None
                        and (
                            coordinate.shape != force.shape
                            or not np.all(np.isfinite(coordinate))
                        )
                    )
                    or (
                        resultant is not None
                        and (resultant.size == 0 or not np.all(np.isfinite(resultant)))
                    )
                )
                if invalid:
                    raise ValueError(
                        "Invalid constraint dual at station "
                        f"{station_index}, entry {dual_index}."
                    )
                names.add(name)
                station.append(
                    {
                        "constraint_name": name,
                        "role": role,
                        "force": force.tolist(),
                        "coordinate": (
                            None if coordinate is None else coordinate.tolist()
                        ),
                        "resultant": (
                            None if resultant is None else resultant.tolist()
                        ),
                        "source": source,
                    }
                )
            station.sort(key=lambda item: item["constraint_name"])
            current_layout = tuple(
                (
                    item["constraint_name"],
                    item["role"],
                    len(item["force"]),
                    item["coordinate"] is not None,
                    None if item["resultant"] is None else len(item["resultant"]),
                    item["source"],
                )
                for item in station
            )
            if layout is None:
                layout = current_layout
            elif current_layout != layout:
                raise ValueError(
                    "Constraint dual providers changed their accepted-path layout."
                )
            selected.append({"load_factor": factor, "duals": station})
            previous = factor
        if (
            selected
            and accepted_factor is not None
            and abs(selected[-1]["load_factor"] - float(accepted_factor)) > 1.0e-12
        ):
            raise ValueError(
                "Constraint dual history does not end at the accepted coordinate."
            )
        return selected

    def snapshot_runtime_state(self) -> list[dict[str, object]]:
        return self._validated(self.records)

    def restore_runtime_state(self, records) -> None:
        self.records[:] = self._validated(records)

    def checkpoint_state(self) -> dict[str, object]:
        return {"schema": self._SCHEMA, "records": self._validated(self.records)}

    def restore_checkpoint_state(self, payload, *, accepted_factor: float) -> None:
        if payload is None:
            self.clear()
            return
        if not isinstance(payload, dict) or payload.get("schema") != self._SCHEMA:
            raise ValueError("Unsupported constraint dual history schema.")
        self.records[:] = self._validated(
            payload.get("records", ()), accepted_factor=accepted_factor
        )

    @property
    def factors(self) -> np.ndarray:
        return np.asarray([item["load_factor"] for item in self.records], dtype=float)

    @property
    def constraint_names(self) -> tuple[str, ...]:
        if not self.records:
            return ()
        return tuple(item["constraint_name"] for item in self.records[0]["duals"])

    def complete(self, *, accepted_factor: float) -> bool:
        return bool(
            len(self.records) >= 2
            and abs(float(self.records[0]["load_factor"])) <= 1.0e-12
            and abs(float(self.records[-1]["load_factor"]) - accepted_factor)
            <= 1.0e-12
        )

    def channel(self, name: str) -> dict[str, object]:
        selected = str(name)
        records = []
        for station in self.records:
            matches = [
                item
                for item in station["duals"]
                if item["constraint_name"] == selected
            ]
            if len(matches) != 1:
                raise KeyError(f"Unknown constraint dual history {selected!r}.")
            records.append(matches[0])
        return {
            "force": np.asarray([item["force"] for item in records], dtype=float),
            "coordinate": (
                None
                if any(item["coordinate"] is None for item in records)
                else np.asarray(
                    [item["coordinate"] for item in records], dtype=float
                )
            ),
            "role": records[0]["role"],
            "source": records[0]["source"],
        }

    def work(self, name: str) -> float:
        channel = self.channel(name)
        coordinates = channel["coordinate"]
        if coordinates is None:
            raise RuntimeError(
                f"Constraint dual history {name!r} has no work coordinate."
            )
        forces = channel["force"]
        return float(
            np.sum(0.5 * (forces[:-1] + forces[1:]) * np.diff(coordinates, axis=0))
        )


__all__ = ["ConstraintDualHistory"]
