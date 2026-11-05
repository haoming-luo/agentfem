# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted fixed-normal contact work, independent of output-frame thinning.

The observer stores scalar work at every accepted station and only the last
gap/pressure vectors. It neither commits a contact State nor integrates Newton
or multiplier iterations. The absolute bound concerns endpoint quadrature,
not the unseen continuous path or a physical energy-conservation certificate.
"""

from copy import deepcopy

import numpy as np

from ._augmented_plane_trial_provider import DolfinxAugmentedPlaneTrialProvider
from .contact_work import _fixed_normal_work_increment


class AcceptedNormalContactWork:
    schema = "agentfem.accepted-fixed-normal-work.v1"
    requires_field_snapshot = False

    def __init__(self, provider):
        if not isinstance(provider, DolfinxAugmentedPlaneTrialProvider):
            raise TypeError("Normal-work recording requires one constant-normal augmented plane.")
        self.provider = provider
        self.complete = True
        self.records = []
        self._last_sample = None

    def _sample(self):
        sample = self.provider.accepted_point_samples()
        return {
            name: np.array(sample[name], copy=True)
            for name in ("point_ids", "reference_weights", "signed_gaps", "normal_pressures")
        }

    def _capture(self, snapshot, *, reset=False):
        factor = float(snapshot.load_factor)
        if not np.isfinite(factor) or not 0. <= factor <= 1.:
            raise ValueError("Normal-work station must be a finite normalized factor.")
        sample = self._sample()
        record = dict(factor=factor, work=0., absolute_bound=0., open_gap_work=0.)
        if not reset:
            if self._last_sample is None or not self.records or factor <= self.records[-1]["factor"]:
                raise ValueError("Normal-work stations must increase from an accepted starting sample.")
            for name in ("point_ids", "reference_weights"):
                if not np.array_equal(sample[name], self._last_sample[name]):
                    raise ValueError("Normal-work trace identity changed.")
            work = _fixed_normal_work_increment(
                weights=sample["reference_weights"],
                old_gap=self._last_sample["signed_gaps"], new_gap=sample["signed_gaps"],
                old_pressure=self._last_sample["normal_pressures"], new_pressure=sample["normal_pressures"],
            )
            previous = self.records[-1]
            record.update(
                work=previous["work"] + work.relative_work,
                absolute_bound=previous["absolute_bound"] + work.absolute_work_bound,
                open_gap_work=previous["open_gap_work"] + work.open_gap_work,
            )
        if reset:
            self.records = [record]
        else:
            self.records.append(record)
        self._last_sample = sample

    def reset(self, snapshot):
        if snapshot.index != 0 or float(snapshot.load_factor) != 0.:
            raise ValueError("A complete normal-work path starts at station zero.")
        self.complete = True
        self._capture(snapshot, reset=True)

    def accept(self, snapshot):
        if self.complete and len(self.records) != int(snapshot.index):
            raise ValueError("Normal-work path skipped an accepted station.")
        self._capture(snapshot)

    def prepare_resume(self, snapshot):
        if not self.records or abs(self.records[-1]["factor"] - snapshot.load_factor) > 1e-12:
            raise ValueError("Normal-work resume coordinate differs.")
        if self.complete and len(self.records) != int(snapshot.index) + 1:
            raise ValueError("Normal-work resume station count differs.")
        sample = self._sample()
        if self._last_sample is None or any(
            not np.allclose(sample[key], self._last_sample[key], rtol=1e-12, atol=1e-12)
            for key in sample
        ):
            raise ValueError("Normal-work resume sample differs from the accepted contact State.")

    def snapshot_runtime_state(self):
        return deepcopy((self.complete, self.records, self._last_sample))

    def restore_runtime_state(self, state):
        self.complete, self.records, self._last_sample = deepcopy(state)

    def summary(self):
        return {
            "kind": "accepted_fixed_normal_work",
            "pair_name": self.provider.name,
            "complete": self.complete,
            "sample_count": len(self.records),
            "integration": "accepted_endpoint_trapezoidal",
            "automatic_energy_verification": False,
        }

    def checkpoint_state(self):
        return {
            "schema": self.schema, "pair_name": self.provider.name,
            "complete": self.complete, "records": deepcopy(self.records),
            "last_sample": None if self._last_sample is None else {
                name: values.tolist() for name, values in self._last_sample.items()
            },
        }

    def restore_checkpoint_state(self, record, *, current_snapshot):
        if (
            not isinstance(record, dict)
            or set(record) != {"schema", "pair_name", "complete", "records", "last_sample"}
            or record["schema"] != self.schema or record["pair_name"] != self.provider.name
            or type(record["complete"]) is not bool or not isinstance(record["records"], list)
            or not record["records"] or not isinstance(record["last_sample"], dict)
        ):
            raise ValueError("Normal-work checkpoint schema differs.")
        rows = record["records"]
        previous = None
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"factor", "work", "absolute_bound", "open_gap_work"}
                or any(type(value) not in (int, float) for value in row.values())
                or not np.isfinite(list(row.values())).all()
                or not 0. <= row["factor"] <= 1. or row["absolute_bound"] < 0.
            ):
                raise ValueError("Normal-work checkpoint has invalid scalar evidence.")
            if previous is not None:
                bound = row["absolute_bound"] - previous["absolute_bound"]
                if (row["factor"] <= previous["factor"] or bound < 0.
                        or abs(row["work"] - previous["work"]) > bound + 1e-12
                        or abs(row["open_gap_work"] - previous["open_gap_work"]) > bound + 1e-12):
                    raise ValueError("Normal-work checkpoint violates its discrete work bound.")
            previous = row
        if record["complete"] and (
            rows[0]["factor"] != 0. or any(rows[0][name] != 0.
                                         for name in ("work", "absolute_bound", "open_gap_work"))
        ):
            raise ValueError("Complete normal-work checkpoint lacks its zero origin.")
        sample = record["last_sample"]
        if set(sample) != {"point_ids", "reference_weights", "signed_gaps", "normal_pressures"}:
            raise ValueError("Normal-work checkpoint sample schema differs.")
        expected = self._sample()
        try:
            arrays = {name: np.asarray(values) for name, values in sample.items()}
            numeric = all(
                values.dtype.kind in "iuf" and np.isfinite(values).all()
                for values in arrays.values()
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Normal-work checkpoint sample must contain finite numeric arrays.") from exc
        if not numeric:
            raise ValueError("Normal-work checkpoint sample must contain finite numeric arrays.")
        if any(
            arrays[key].shape != expected[key].shape or not np.isfinite(arrays[key]).all()
            or (not np.array_equal(arrays[key], expected[key]) if key == "point_ids"
                else not np.allclose(arrays[key], expected[key], rtol=1e-12, atol=1e-12))
            for key in expected
        ):
            raise ValueError("Normal-work checkpoint sample differs from restored contact State.")
        saved = self.snapshot_runtime_state()
        try:
            self.complete, self.records, self._last_sample = (
                record["complete"], deepcopy(rows), arrays,
            )
            self.prepare_resume(current_snapshot)
        except BaseException:
            self.restore_runtime_state(saved)
            raise

    def restore_missing_checkpoint_state(self, *, current_snapshot):
        self.complete = False
        self._capture(current_snapshot, reset=True)
