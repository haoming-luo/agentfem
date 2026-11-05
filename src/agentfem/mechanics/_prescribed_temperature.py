# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen uniform thermal input; no live-field mutation or heat solve."""

from dataclasses import dataclass

import numpy as np

from ..histories import FieldHistory
from ..constitutive.user_material import _real_material_scalar


@dataclass(frozen=True)
class UniformTemperatureInput:
    times: tuple[float, ...]
    values: tuple[float, ...]

    def at(self, time):
        return float(np.interp(time, self.times, self.values))

    def summary(self):
        return {"kind": "prescribed_uniform_temperature", "unit": "K",
                "coordinate": "physical_time", "interpolation": "linear",
                "times": list(self.times), "values": list(self.values)}


def freeze_uniform_temperature(source, duration):
    if source is None:
        return None
    if isinstance(source, (bool, np.bool_)):
        raise ValueError("Temperature must be a real scalar, not a boolean.")
    if isinstance(source, FieldHistory):
        if (source.unit != "K" or source.coordinate_name != "time"
                or source.interpolation != "linear"):
            raise ValueError("Temperature history requires kelvin, physical time and linear interpolation.")
        if hasattr(source.value, "function_space"):
            raise NotImplementedError("This history-material Step accepts uniform scalar temperatures only.")
        times = np.asarray(source.times, dtype=float)
        if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or times[0] > 0 or times[-1] < duration):
            raise ValueError("Temperature history must cover the full physical duration without extrapolation.")
        selected_times = (0., *(float(t) for t in times if 0 < t < duration), float(duration))
        values = []
        for time in selected_times:
            sample = source.sample(time)
            if sample.shape != (1,) or np.iscomplexobj(sample):
                raise ValueError("Temperature history must contain real scalar samples.")
            values.append(float(sample[0]))
    else:
        value = _real_material_scalar(source, label="Temperature in kelvin")
        selected_times, values = (0., float(duration)), (value, value)
    if not np.isfinite(values).all() or np.any(np.asarray(values) <= 0):
        raise ValueError("Temperature must be finite and positive in kelvin.")
    return UniformTemperatureInput(tuple(selected_times), tuple(values))
