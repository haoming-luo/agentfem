# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Time-resolved assessment; no constitutive state or design-code data."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite

import numpy as np


_SOURCE_INTERVAL_RTOL = 1.e-12


@dataclass(frozen=True)
class _DamageInterval:
    start: float
    end: float
    stress: tuple[float, float]
    temperature: tuple[float, float]
    damage_rate: tuple[float, float]
    repetitions: float
    label: str
    damage: float

    def as_dict(self):
        return dict(start=self.start, end=self.end, stress=list(self.stress),
                    temperature=list(self.temperature), damage_rate=list(self.damage_rate),
                    repetitions=self.repetitions, label=self.label, damage=self.damage)


@dataclass(frozen=True)
class CreepDamageHistoryAssessment:
    """Trapezoidal time fractions on sampled stress/temperature histories.

    Construct with ``assessments.creep_damage_from_result``. This is an
    empirical postprocessor, not constitutive damage, a certified integration
    error bound, or a design-code approval. Intervals retain all evaluated
    inputs/rates and repetition-weighted contributions without storing a callable.
    """

    intervals: tuple[_DamageInterval, ...]
    damage: float
    source: str
    rupture_source: str
    stress_unit: str
    time_unit: str
    stress_measure: str
    maximum_stress_sample_interval: float
    maximum_temperature_sample_interval: float
    rupture_relation: dict | None = None
    stress_component: tuple[int, int] | None = None
    material_point_source: dict | None = None

    def as_dict(self):
        record = dict(
            kind="creep_time_fraction_history_assessment", maturity="postprocessor",
            method="trapezoidal_damage_rate", input_interpolation="piecewise_linear",
            quadrature_error_certified=False, source=self.source,
            rupture_source=self.rupture_source, stress_unit=self.stress_unit,
            time_unit=self.time_unit, temperature_unit="K", stress_measure=self.stress_measure,
            damage_rate_unit=f"1/{self.time_unit}", damage=float(self.damage),
            maximum_stress_sample_interval=self.maximum_stress_sample_interval,
            maximum_temperature_sample_interval=self.maximum_temperature_sample_interval,
            source_interval_relative_tolerance=_SOURCE_INTERVAL_RTOL,
            intervals=[interval.as_dict() for interval in self.intervals],
        )
        if self.rupture_relation is not None:
            from copy import deepcopy
            record["rupture_relation"] = deepcopy(self.rupture_relation)
        if self.stress_component is not None:
            record["stress_component"] = list(self.stress_component)
        if self.material_point_source is not None:
            from copy import deepcopy
            record["material_point_source"] = deepcopy(self.material_point_source)
        return record


def _point_source(result, stress_name, temperature_name):
    """Check recorded identities, without inferring location from name syntax."""
    from copy import deepcopy

    identity = result.metadata.get("material_point_history")
    if identity is None:
        return None
    if not isinstance(identity, dict) or identity.get("schema") != "agentfem.material-point-history.v1":
        raise ValueError("Unrecognized material-point history identity.")
    associations = {}
    for point in identity["points"]:
        for channel in identity["channels"]:
            name = f"{point['name']}.{channel}"
            if name in associations:
                raise ValueError("Duplicate material-point history identity.")
            associations[name] = (point, channel)
    selected = [associations.get(name) for name in (stress_name, temperature_name)]
    if not any(item is not None for item in selected):
        return None  # Ordinary caller-supplied scalar histories retain their contract.
    if any(item is None for item in selected):
        raise ValueError("Stress and temperature must identify the same recorded material point.")
    # Two user-facing names may intentionally select the same integration point.
    location_keys = ("cell", "point", "reference_position")
    if any(selected[0][0][key] != selected[1][0][key] for key in location_keys):
        raise ValueError("Stress and temperature must identify the same recorded material point.")
    stress, temperature = result.histories[stress_name], result.histories[temperature_name]
    if (not np.array_equal(stress.abscissa, temperature.abscissa)
            or stress.abscissa_unit != identity["time_unit"]
            or temperature.abscissa_unit != identity["time_unit"]):
        raise ValueError("Recorded material-point channels must share their accepted physical clock.")
    for history, (_, channel) in zip((stress, temperature), selected, strict=True):
        description = identity["channels"][channel]
        if history.value_shape != tuple(description["shape"]) or history.unit != description["unit"]:
            raise ValueError("Material-point channel differs from its recorded shape or unit.")
    return deepcopy(dict(point=selected[0][0], mesh=identity["mesh"],
                         histories={"stress": stress_name, "temperature": temperature_name},
                         quadrature_rule=identity["quadrature_rule"],
                         scientific_identity=identity["scientific_identity"],
                         channels={item[1]: identity["channels"][item[1]] for item in selected},
                         time_unit=identity["time_unit"]))


def _stress_component(history, component):
    from dataclasses import replace

    if component is None:
        if len(history.value_shape) == 2:
            raise ValueError("Tensor stress history requires explicit stress_component=(i, j).")
        return history, None
    if (not isinstance(component, tuple) or len(component) != 2
            or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in component)):
        raise ValueError("stress_component must be a pair of zero-based tensor indices.")
    shape = history.value_shape
    if len(shape) != 2 or shape[0] != shape[1] or any(i < 0 or i >= n for i, n in zip(component, shape, strict=True)):
        raise ValueError("stress_component lies outside the recorded square stress tensor.")
    indices = tuple(map(int, component))
    return replace(history, values=history.values[(slice(None), *indices)]), indices


def _scalar_history(history, *, unit, time_unit):
    if history.abscissa_name != "time" or history.abscissa_unit != time_unit:
        raise ValueError(f"History {history.name!r} must use physical time in {time_unit!r}.")
    if history.unit != unit:
        raise ValueError(f"History {history.name!r} must declare unit {unit!r}; no implicit conversion.")
    if np.iscomplexobj(history.abscissa) or np.iscomplexobj(history.values):
        raise ValueError("Creep assessment histories must be real.")
    times, values = np.asarray(history.abscissa, dtype=float), np.asarray(history.values, dtype=float)
    if (times.ndim != 1 or values.shape != times.shape or len(times) < 2
            or not np.isfinite(times).all() or not np.isfinite(values).all()
            or np.any(np.diff(times) <= 0.)):
        raise ValueError("Creep assessment requires finite scalar histories with increasing physical time.")
    return times, values


def creep_damage_from_result(
    result, *, stress_history: str, temperature_history: str, dwells,
    rupture_time, rupture_source: str, stress_unit: str, time_unit: str,
    stress_measure: str, maximum_source_interval: float | None = None,
    stress_component: tuple[int, int] | None = None,
) -> CreepDamageHistoryAssessment:
    """Integrate ``dt / rupture_time(stress, temperature)`` over explicit dwells.

    Callback inputs retain the signed scalar stress, declared stress unit and
    kelvin temperature; its output must be positive in ``time_unit``. Positive
    infinity explicitly means zero damage rate. No absolute-value transform,
    fitted rupture relation, unit conversion or extrapolation is implicit.
    Histories may have different time grids; their union and clipped dwell
    endpoints define the trapezoidal quadrature. Source sampling, not just the
    union grid, is checked by optional ``maximum_source_interval``, with a
    relative 1e-12 roundoff allowance and no absolute/time-origin allowance.

    Dwells must be ordered, disjoint and covered by both histories. A repeated
    dwell represents an explicitly assumed repeated response; it does not
    simulate cyclic evolution. Source identity must identify the selected
    rupture relation and calibration domain. Its callable owns domain rejection.
    Inputs must describe the same material location/path; separate spatial
    maxima are not automatically a valid or conservative local-life history.
    ``stress_component=(i, j)`` selects a zero-based tensor component without
    absolute values or a multiaxial criterion. Recorded material-point channels
    must identify the same point and accepted clock; that identity is retained.
    """
    from .assessments import DwellInterval
    from ._rupture_curve import TabulatedRuptureCurve

    for name, value in (("rupture_source", rupture_source), ("stress_unit", stress_unit),
                        ("time_unit", time_unit), ("stress_measure", stress_measure)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be explicitly declared.")
    if not callable(rupture_time):
        raise TypeError("rupture_time must be callable(stress, temperature).")
    relation = None
    if isinstance(rupture_time, TabulatedRuptureCurve):
        rupture_time.validate_usage(stress_unit=stress_unit, time_unit=time_unit,
                                    stress_measure=stress_measure)
        relation = rupture_time.as_dict()
    if maximum_source_interval is not None:
        maximum_source_interval = float(maximum_source_interval)
        if not isfinite(maximum_source_interval) or maximum_source_interval <= 0.:
            raise ValueError("maximum_source_interval must be finite and positive.")
    stress = result.histories[stress_history]
    temperature = result.histories[temperature_history]
    point_source = _point_source(result, stress_history, temperature_history)
    stress, component = _stress_component(stress, stress_component)
    sx, sy = _scalar_history(stress, unit=stress_unit, time_unit=time_unit)
    tx, ty = _scalar_history(temperature, unit="K", time_unit=time_unit)
    if np.any(ty <= 0.):
        raise ValueError("Absolute temperature must be positive in kelvin.")
    selected_dwells = tuple(dwells)
    if not selected_dwells or not all(isinstance(d, DwellInterval) for d in selected_dwells):
        raise ValueError("At least one DwellInterval is required.")
    # Validate the full selection before invoking a user-owned rupture relation.
    gaps = [0., 0.]
    source_intervals = (np.diff(sx), np.diff(tx))
    dwell_slices = []
    previous_end = -np.inf
    for dwell in selected_dwells:
        if dwell.start < previous_end:
            raise ValueError("Dwells must be ordered and nonoverlapping; use repetitions explicitly.")
        previous_end = dwell.end
        slices = []
        for index, times in enumerate((sx, tx)):
            if dwell.start < times[0] or dwell.end > times[-1]:
                raise ValueError("Dwell lies outside history coverage; extrapolation is disabled.")
            # Ordered time permits bounded views, not a full-history scan per
            # dwell. The enclosing source intervals still count when clipped.
            first = int(np.searchsorted(times, dwell.start, side="right"))
            stop = int(np.searchsorted(times, dwell.end, side="left"))
            slices.append(slice(first, stop))
            gap = float(np.max(source_intervals[index][first - 1:stop]))
            gaps[index] = max(gaps[index], gap)
            if (maximum_source_interval is not None
                    and gap - maximum_source_interval > _SOURCE_INTERVAL_RTOL * maximum_source_interval):
                raise ValueError("Source history exceeds maximum_source_interval; interpolation cannot add evidence.")
        dwell_slices.append(slices)
    intervals = []
    for dwell, (stress_slice, temperature_slice) in zip(selected_dwells, dwell_slices, strict=True):
        grid = np.unique(np.concatenate(([dwell.start, dwell.end],
            sx[stress_slice], tx[temperature_slice])))
        stresses, temperatures = np.interp(grid, sx, sy), np.interp(grid, tx, ty)
        rates = []
        for time, sigma, kelvin in zip(grid, stresses, temperatures, strict=True):
            try:
                rupture = rupture_time(float(sigma), float(kelvin))
                if (isinstance(rupture, (bool, np.bool_))
                        or np.ndim(rupture) != 0 or np.iscomplexobj(rupture)):
                    raise ValueError("Rupture time must be a real scalar.")
                rupture = float(rupture)
                if np.isnan(rupture) or rupture <= 0.:
                    raise ValueError("Rupture time must be positive (or positive infinity).")
                rate = 1. / rupture
                if not isfinite(rate) or (rate == 0. and isfinite(rupture)):
                    raise ValueError("Damage rate is outside the representable range.")
            except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
                raise ValueError(f"Rupture relation failed in dwell {dwell.label!r} at time {time:g}: {exc}") from exc
            rates.append(rate)
        for i in range(len(grid) - 1):
            rate = .5 * rates[i] + .5 * rates[i + 1]
            duration = float(grid[i + 1] - grid[i])
            damage = duration * rate * dwell.repetitions
            if (not isfinite(damage)
                    or (damage == 0. and rate > 0. and dwell.repetitions > 0.)):
                raise ValueError("Integrated creep damage is outside the representable range.")
            intervals.append(_DamageInterval(
                float(grid[i]), float(grid[i + 1]), tuple(map(float, stresses[i:i + 2])),
                tuple(map(float, temperatures[i:i + 2])), (rates[i], rates[i + 1]),
                float(dwell.repetitions), str(dwell.label), damage))
    try:
        damage = fsum(interval.damage for interval in intervals)
    except OverflowError as exc:
        raise ValueError("Total creep damage exceeds the representable range.") from exc
    return CreepDamageHistoryAssessment(
        tuple(intervals), damage, f"{result.name}:{stress_history},{temperature_history}",
        rupture_source, stress_unit, time_unit, stress_measure, *gaps, relation,
        component, point_source)
