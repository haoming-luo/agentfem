"""Research-only plastic consistency + creep + recoverable strains."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-inviscid-fast-branch.v1"
STATE_LAYOUT = spectrum.STATE_LAYOUT


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material

    def __post_init__(self):
        s = self.spectrum
        if (s.directional_recovery or any(v != 1 for v in
                (*s.strength_ratios, *s.hardening_ratios, *s.static_ratios, *s.recoverable_powers))
                or any(v != 0 for v in (*s.strength_rates, *s.hardening_rates,
                                       *s.static_powers, *s.static_history_rates))
                or any(b.saturation_ratio != 1 or b.softening_rate != 0 for b in s.backbone.branches)
                or s.backbone.branches[1].strength <= 0):
            raise ValueError("Plastic consistency requires the constant isolated spectrum")


def mapped(material):
    a, b = material.backbone.branches
    return Material(replace(material, backbone=replace(material.backbone,
        branches=(a, replace(b, strength=b.strength+120.)))))


def stored_energy(state, material):
    return spectrum.stored_energy(state, material.spectrum)


def yield_value(state, material):
    return abs(state[0]-state[4]-state[5])-material.spectrum.backbone.branches[1].strength


def components(state, strain_rate, material, *, stress_control=False, sign=1.):
    s = material.spectrum
    a, b = s.backbone.branches
    effective_a = state[0]-state[1]-state[2]
    flow_a = np.sign(effective_a)*a.reference_flow*(max(abs(effective_a)-a.strength, 0.)/120.)**a.exponent
    recovery = (state[0]/np.asarray(s.moduli)-state[10:])/s.times
    dynamic = np.array(b.hardening)/b.saturation*(abs(state[4:6])/b.saturation)**b.ratcheting
    numerator = sign*float(np.dot(b.static_recovery, state[4:6]))
    denominator = float(sum(np.array(b.hardening)-sign*dynamic*state[4:6]))
    if not stress_control:
        numerator += sign*s.backbone.modulus*(strain_rate-flow_a-sum(recovery))
        denominator += s.backbone.modulus
    return flow_a, recovery, numerator, denominator


def rates(state, strain_rate, material, *, stress_control=False, active=0.):
    flow_a, recovery, numerator, denominator = components(
        state, strain_rate, material, stress_control=stress_control, sign=active or 1.)
    multiplier = 0.
    if active and numerator > 0:
        if denominator <= 0:
            raise RuntimeError("Plastic consistency denominator is nonpositive")
        multiplier = numerator/denominator
    flows = [flow_a, active*multiplier]
    result = np.zeros(12)
    dissipation = 0.
    for j, (offset, b) in enumerate(zip((1, 4), material.spectrum.backbone.branches, strict=True)):
        speed = abs(flows[j])
        dissipation += (state[0]-sum(state[offset:offset+2]))*flows[j]
        for i, (h, a, r, chi) in enumerate(zip(b.hardening, b.saturation, b.static_recovery, b.ratcheting, strict=True)):
            x = state[offset+i]
            decay = h/a*(abs(x)/a)**chi*speed+r
            result[offset+i] = h*flows[j]-decay*x
            dissipation += decay*x*x/h
        result[offset+2] = speed
    result[10:] = recovery
    total = sum(flows)+sum(recovery)
    result[7] = total if stress_control else strain_rate
    result[0] = 0. if stress_control else material.spectrum.backbone.modulus*(strain_rate-total)
    result[8] = state[0]*result[7]
    result[9] = dissipation+sum(np.array(material.spectrum.moduli)*material.spectrum.times*recovery**2)
    return result


def integrate(old, duration, strain_rate, material, *, stress_control=False, method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float).copy()
    if (state.shape != (12,) or not np.isfinite(state).all() or np.any(state[[3, 6, 9]] < 0)
            or type(stress_control) is not bool or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid plastic consistency state or controls")
    bounds = np.array([v for b in material.spectrum.backbone.branches for v in b.saturation])
    if yield_value(state, material) > 1e-7 or np.any(abs(state[[1, 2, 4, 5]]) > bounds+1e-7):
        raise ValueError("Initial state outside yield or memory domain")
    sign = float(np.sign(state[0]-state[4]-state[5]))
    drive = components(state, strain_rate, material, stress_control=stress_control, sign=sign)[2]
    active = sign if abs(yield_value(state, material)) < 1e-8 and drive > 0 else 0.
    parts, transitions, start = [], [], 0.
    k = material.spectrum.backbone.branches[1].strength
    atol = [1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13]
    while start < duration:
        events = []
        if active:
            def release(t, y, sign=active):
                return components(y, strain_rate, material, stress_control=stress_control, sign=sign)[2]
            release.terminal, release.direction = True, -1
            events.append(release)
        else:
            for sign in (-1., 1.):
                def hit(t, y, sign=sign, initial=start):
                    gap = k-sign*(y[0]-y[4]-y[5])
                    return max(gap, 1e-12) if t == initial else gap
                hit.terminal, hit.direction = True, -1
                events.append(hit)
        part = solve_ivp(lambda t, y, active=active: rates(y, strain_rate, material,
            stress_control=stress_control, active=active), (start, duration), state,
            method=method, rtol=rtol, atol=atol, dense_output=True, events=events)
        if not part.success or not np.isfinite(part.y).all():
            raise RuntimeError("Plastic consistency integration failed; input retained")
        parts.append(part)
        state = part.y[:, -1].copy()
        end = float(part.t[-1])
        if part.status == 0:
            break
        event = next(i for i, times in enumerate(part.t_events) if len(times))
        active = 0. if active else (-1., 1.)[event]
        transitions.append((end, active))
        if end <= start or len(transitions) > 200:
            raise RuntimeError("Plastic consistency event stagnation; input retained")
        start = end
    t = np.concatenate([p.t if i == 0 else p.t[1:] for i, p in enumerate(parts)])
    y = np.concatenate([p.y if i == 0 else p.y[:, 1:] for i, p in enumerate(parts)], axis=1)
    if (np.any(abs(y[0]-y[4]-y[5])-k > 1e-6)
            or np.any(abs(y[[1, 2, 4, 5]]) > bounds[:, None]+1e-6)):
        raise RuntimeError("Plastic consistency or memory invariant violated")
    accumulated = y[[3, 6, 9]]
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None]+rtol*abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    balance = state[8]-old[8]-stored_energy(state, material)+stored_energy(old, material)-state[9]+old[9]
    if abs(balance) > 1e-7+1e-7*abs(state[8]-old[8]):
        raise RuntimeError("Integrated mechanical balance failed")

    def dense(times):
        query = np.asarray(times, dtype=float)
        if not np.isfinite(query).all() or np.any(query < 0) or np.any(query > duration):
            raise ValueError("Dense query outside integrated interval")
        positions = np.searchsorted([p.t[-1] for p in parts], query.ravel(), side="left")
        result = np.empty((12, query.size))
        for i, part in enumerate(parts):
            selected = positions == i
            if np.any(selected):
                result[:, selected] = part.sol(query.ravel()[selected])
        return result[:, 0] if query.ndim == 0 else result.reshape((12, *query.shape))

    return SimpleNamespace(success=True, t=t, y=y, sol=dense, transitions=transitions,
                           nfev=sum(p.nfev for p in parts))
